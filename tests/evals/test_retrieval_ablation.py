from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from qdrant_client import models

from evals.retrieval_ablation import ABLATION_CONFIGS, retrieve_ablation_points
from evals.retrieval_eval_runner import run_retrieval_eval
from evals.retrieval_metrics import ndcg_at_k


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
            client=Mock(),
            collection_name="test_collection",
        )

    assert embed.call_count == dense_calls
    assert dense_search.call_count == dense_calls
    assert bm25_search.call_count == bm25_calls
    production.assert_not_called()
    assert timing["components"] == ABLATION_CONFIGS[mode]
    assert len(ranked) == dense_calls + bm25_calls


def test_full_mode_uses_unchanged_production_backend() -> None:
    point = _point("00000000-0000-0000-0000-000000000001", "A::text::1")
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
        )

    production.assert_called_once()
    assert query == "query"
    assert fused == ranked == [point]
    assert timing["candidate_retrieval_ms"] == 16
    assert timing["total_retrieval_ms"] == 20
    assert timing["components"] == ABLATION_CONFIGS["hybrid_reranker"]


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
