from __future__ import annotations

import inspect
import importlib.util
import json
import math
import os
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from qdrant_client import models

from evals.retrieval_ablation import ABLATION_CONFIGS, retrieve_ablation_points
from evals.retrieval_eval_runner import run_retrieval_eval
from evals.retrieval_metrics import ndcg_at_k
from mcp_server.tools import sec_retrieval


def _load_ablation_runner():
    path = Path("scripts/evals/retrieval/run_retrieval_ablation_v1.py")
    spec = importlib.util.spec_from_file_location("retrieval_ablation_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ABLATION_RUNNER = _load_ablation_runner()


def _point(point_id: str, doc_id: str, score: float = 1.0) -> models.ScoredPoint:
    return models.ScoredPoint(
        id=point_id,
        version=0,
        score=score,
        payload={"doc_id": doc_id, "doc_type": "text_chunk", "content": doc_id},
        vector=None,
        shard_key=None,
        order_value=None,
    )


@pytest.mark.parametrize(
    ("mode", "dense_calls", "bm25_calls"),
    [
        ("bm25_only", 0, 1),
        ("dense_only", 1, 0),
        ("hybrid", 1, 1),
    ],
)
def test_non_reranked_modes_activate_only_declared_components(
    mode: str,
    dense_calls: int,
    bm25_calls: int,
) -> None:
    dense = [_point("00000000-0000-0000-0000-000000000001", "A::text::1")]
    bm25 = [_point("00000000-0000-0000-0000-000000000002", "A::text::2")]
    client = Mock()

    with (
        patch("evals.retrieval_ablation.embed_query_qwen3", return_value=[0.1]) as embed,
        patch("evals.retrieval_ablation.dense_search_points", return_value=dense) as dense_search,
        patch("evals.retrieval_ablation.bm25_search_points", return_value=bm25) as bm25_search,
        patch("evals.retrieval_ablation.retrieve_scored_points") as production,
    ):
        _query, _candidates, ranked, timing = retrieve_ablation_points(
            retrieval_mode=mode,
            query="test query",
            ticker="AAPL",
            fiscal_year=2024,
            form_type="10-K",
            doc_types=["text_chunk"],
            client=client,
            collection_name="test_collection",
            embed_api_url="http://verified.example/api/embed",
            embed_model="verified-model",
        )

    assert embed.call_count == dense_calls
    assert dense_search.call_count == dense_calls
    assert bm25_search.call_count == bm25_calls
    production.assert_not_called()
    assert timing["components"] == ABLATION_CONFIGS[mode]
    assert len(ranked) == dense_calls + bm25_calls
    if dense_calls:
        assert dense_search.call_args.kwargs["client"] is client
        assert embed.call_args.kwargs["api_url"] == "http://verified.example/api/embed"
        assert embed.call_args.kwargs["model"] == "verified-model"
    if bm25_calls:
        assert bm25_search.call_args.kwargs["client"] is client


def test_full_mode_uses_unchanged_production_backend() -> None:
    point = _point("00000000-0000-0000-0000-000000000001", "A::text::1")
    client = Mock()
    production_timing = {
        "hybrid_retrieval_ms": 10,
        "cross_query_fusion_ms": 1,
        "dedupe_ms": 2,
        "enrichment_ms": 3,
        "rerank_ms": 4,
    }
    with patch(
        "evals.retrieval_ablation.retrieve_scored_points",
        return_value=("query", [point], [point], production_timing),
    ) as production:
        query, fused, ranked, timing = retrieve_ablation_points(
            retrieval_mode="hybrid_reranker",
            query="query",
            ticker="AAPL",
            fiscal_year=2024,
            form_type="10-K",
            doc_types=["text_chunk"],
            client=client,
            qdrant_host="verified-qdrant",
            qdrant_port=7333,
            collection_name="verified-collection",
            embed_api_url="http://verified.example/api/embed",
            embed_model="verified-model",
        )

    production.assert_called_once()
    assert query == "query"
    assert fused == ranked == [point]
    assert timing["candidate_retrieval_ms"] == 16
    assert timing["total_retrieval_ms"] == 20
    assert timing["components"] == ABLATION_CONFIGS["hybrid_reranker"]
    assert production.call_args.kwargs["client"] is client
    assert production.call_args.kwargs["collection_name"] == "verified-collection"
    assert production.call_args.kwargs["embed_api_url"] == "http://verified.example/api/embed"
    assert production.call_args.kwargs["embed_model"] == "verified-model"
    assert timing["provenance"] == {
        "qdrant": {
            "host": "verified-qdrant",
            "port": 7333,
            "collection": "verified-collection",
        },
        "embedding": {
            "api_url": "http://verified.example/api/embed",
            "model": "verified-model",
        },
    }


def test_ndcg_uses_all_gold_labels_for_ideal_ranking() -> None:
    actual = ndcg_at_k([False, True], 5, num_relevant=3)
    expected = (1.0 / math.log2(3)) / (
        1.0 + (1.0 / math.log2(3)) + (1.0 / math.log2(4))
    )
    assert actual == pytest.approx(expected)


def test_text_scoring_deduplicates_normalized_split_ids(tmp_path: Path) -> None:
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query": "query",
                "relevant_doc_ids": [1],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    points = [
        _point("00000000-0000-0000-0000-000000000001", "AAPL_10-K_2024::text::0::split::0"),
        _point("00000000-0000-0000-0000-000000000002", "AAPL_10-K_2024::text::0::split::1"),
        _point("00000000-0000-0000-0000-000000000003", "AAPL_10-K_2024::text::1::split::0"),
    ]

    with patch(
        "evals.retrieval_eval_runner.retrieve_ablation_points",
        return_value=("query", points, points, {"components": ABLATION_CONFIGS["hybrid"]}),
    ):
        _summary, rows, errors = run_retrieval_eval(
            eval_path=str(eval_path),
            out_dir=str(tmp_path / "out"),
            eval_mode="text",
            retrieval_mode="hybrid",
            enable_ragas=False,
        )

    assert errors == []
    assert rows[0].retrieved_doc_ids == [
        "AAPL_10-K_2024::text::0",
        "AAPL_10-K_2024::text::1",
    ]
    assert rows[0].metrics["mrr@3"] == 0.5


def test_default_eval_mode_keeps_full_production_stack() -> None:
    parameter = inspect.signature(run_retrieval_eval).parameters["retrieval_mode"]
    assert parameter.default == "hybrid_reranker"


def test_frozen_mode_definitions_match_evaluator() -> None:
    config_path = Path("data/evals/retrieval/retrieval_ablation_v1.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["modes"] == ABLATION_CONFIGS
    assert config["common"]["ragas_enabled"] is False
    assert config["common"]["min_total_score"] == 0.0


@pytest.mark.parametrize("value", [-1.0, 0.01, 1.0])
def test_retrieval_ablation_rejects_nonzero_minimum_score(tmp_path: Path, value: float) -> None:
    with pytest.raises(ValueError, match="min_total_score must be 0"):
        run_retrieval_eval(
            eval_path=str(tmp_path / "missing.jsonl"),
            out_dir=str(tmp_path / "out"),
            min_total_score=value,
            enable_ragas=False,
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("top_k", "k_values"),
    [
        (11, (1, 3, 5, 10)),
        (10, (1, 3, 5, 11)),
    ],
)
def test_retrieval_ablation_rejects_depth_above_comparable_limit(
    tmp_path: Path,
    top_k: int,
    k_values: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="depth must not exceed 10"):
        run_retrieval_eval(
            eval_path=str(tmp_path / "missing.jsonl"),
            out_dir=str(tmp_path / "out"),
            top_k=top_k,
            k_values=k_values,
            enable_ragas=False,
        )
    assert not (tmp_path / "out").exists()


def test_standalone_evaluator_uses_environment_embedding_defaults(tmp_path: Path) -> None:
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps({"query_id": "q1", "query": "query", "relevant_doc_ids": [1]}) + "\n",
        encoding="utf-8",
    )
    point = _point("00000000-0000-0000-0000-000000000001", "AAPL_10-K_2024::text::1")
    with (
        patch.dict(
            os.environ,
            {
                "QWEN3_EMBED_API_URL": "http://environment.example/api/embed",
                "QWEN3_EMBED_MODEL": "environment-model",
            },
            clear=False,
        ),
        patch(
            "evals.retrieval_eval_runner.retrieve_ablation_points",
            return_value=("query", [point], [point], {"components": ABLATION_CONFIGS["dense_only"]}),
        ) as retrieval,
    ):
        run_retrieval_eval(
            eval_path=str(eval_path),
            out_dir=str(tmp_path / "out"),
            eval_mode="text",
            retrieval_mode="dense_only",
            enable_ragas=False,
        )

    assert retrieval.call_args.kwargs["embed_api_url"] == "http://environment.example/api/embed"
    assert retrieval.call_args.kwargs["embed_model"] == "environment-model"


def _init_git_repo(path: Path) -> str:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    (path / "tracked.txt").write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=path, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_clean_checkout_is_accepted(tmp_path: Path) -> None:
    head = _init_git_repo(tmp_path)
    ABLATION_RUNNER._assert_clean_evaluated_checkout(head, repo_root=tmp_path)


def test_staged_tracked_change_is_rejected(tmp_path: Path) -> None:
    head = _init_git_repo(tmp_path)
    (tmp_path / "tracked.txt").write_text("staged\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    with pytest.raises(ValueError, match="staged tracked changes"):
        ABLATION_RUNNER._assert_clean_evaluated_checkout(head, repo_root=tmp_path)


def test_unstaged_tracked_change_is_rejected(tmp_path: Path) -> None:
    head = _init_git_repo(tmp_path)
    (tmp_path / "tracked.txt").write_text("unstaged\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unstaged tracked changes"):
        ABLATION_RUNNER._assert_clean_evaluated_checkout(head, repo_root=tmp_path)


def test_head_mismatch_is_rejected(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    with pytest.raises(ValueError, match="HEAD does not match"):
        ABLATION_RUNNER._assert_clean_evaluated_checkout("0" * 40, repo_root=tmp_path)


def test_runtime_affecting_untracked_file_is_rejected(tmp_path: Path) -> None:
    head = _init_git_repo(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "override.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside artifacts"):
        ABLATION_RUNNER._assert_clean_evaluated_checkout(head, repo_root=tmp_path)


def test_untracked_artifact_is_allowed(tmp_path: Path) -> None:
    head = _init_git_repo(tmp_path)
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "old.json").write_text("{}\n", encoding="utf-8")
    ABLATION_RUNNER._assert_clean_evaluated_checkout(head, repo_root=tmp_path)


@pytest.mark.parametrize(
    ("name", "value", "match"),
    [
        ("QDRANT_HOST", "other-qdrant", "QDRANT_HOST"),
        ("QDRANT_PORT", "7333", "QDRANT_PORT"),
        ("QDRANT_COLLECTION_NAME", "other-collection", "QDRANT_COLLECTION_NAME"),
        ("QWEN3_EMBED_API_URL", "http://other/api/embed", "QWEN3_EMBED_API_URL"),
        ("QWEN3_EMBED_MODEL", "other-model", "QWEN3_EMBED_MODEL"),
    ],
)
def test_conflicting_runtime_environment_is_rejected(
    name: str,
    value: str,
    match: str,
) -> None:
    matching_environment = {
        "QDRANT_HOST": "verified-qdrant",
        "QDRANT_PORT": "6333",
        "QDRANT_COLLECTION_NAME": "verified-collection",
        "QWEN3_EMBED_API_URL": "http://verified/api/embed",
        "QWEN3_EMBED_MODEL": "verified-model",
        name: value,
    }
    with patch.dict(os.environ, matching_environment, clear=False):
        with pytest.raises(ValueError, match=match):
            ABLATION_RUNNER._validate_runtime_environment(
                qdrant_host="verified-qdrant",
                qdrant_port=6333,
                collection="verified-collection",
                embedding_api_url="http://verified/api/embed",
                embedding_model="verified-model",
            )


def test_production_retrieval_defaults_are_preserved() -> None:
    client = Mock()
    with (
        patch.object(sec_retrieval, "_get_client", return_value=client),
        patch.object(
            sec_retrieval,
            "_run_dense_bm25_retrieval",
            return_value=("query", [], [], {}),
        ) as backend,
    ):
        sec_retrieval.retrieve_scored_points(
            queries=["query"],
            ticker="AAPL",
            fiscal_year=2024,
        )

    assert backend.call_args.kwargs["client"] is client
    assert backend.call_args.kwargs["collection_name"] == sec_retrieval.COLLECTION_NAME
    assert backend.call_args.kwargs["embed_api_url"] == sec_retrieval.QWEN3_EMBED_API_URL
    assert backend.call_args.kwargs["embed_model"] == sec_retrieval.QWEN3_EMBED_MODEL
