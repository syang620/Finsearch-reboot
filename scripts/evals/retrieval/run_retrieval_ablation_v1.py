#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import requests
from qdrant_client import QdrantClient, models

from evals.retrieval_ablation import ABLATION_CONFIGS, RETRIEVAL_MODES
from evals.retrieval_eval_contracts import load_retrieval_eval_examples
from evals.retrieval_eval_runner import _normalize_text_doc_id, run_retrieval_eval
from mcp_server.tools.sec_retrieval import (
    COLLECTION_NAME,
    QWEN3_EMBED_API_URL,
    QWEN3_EMBED_MODEL,
    RERANK_CANDIDATE_LIMIT,
    RERANK_MODEL,
    RETRIEVAL_TOP_K,
    _current_qwen3_rerank_api_key,
    _current_qwen3_rerank_api_url,
)

CONFIG_PATH = Path("data/evals/retrieval/retrieval_ablation_v1.json")
RAW_FILENAMES = ("summary.json", "per_query.jsonl", "errors.jsonl")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _latency_stats(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, float]:
    values = [
        float(((row.get("trace") or {}).get("timing_ms") or {}).get(key) or 0.0)
        for row in rows
    ]
    return {
        "mean_ms": statistics.mean(values) if values else 0.0,
        "p50_ms": statistics.median(values) if values else 0.0,
        "p95_ms": _percentile(values, 0.95),
    }


def _delta(source: Dict[str, float], target: Dict[str, float]) -> Dict[str, Dict[str, float | None]]:
    out: Dict[str, Dict[str, float | None]] = {}
    for metric in sorted(set(source) | set(target)):
        before = float(source.get(metric, 0.0))
        after = float(target.get(metric, 0.0))
        absolute = after - before
        out[metric] = {
            "source": before,
            "target": after,
            "absolute": absolute,
            "relative_percent": (absolute / before) * 100.0 if before else None,
        }
    return out


def _collection_records(client: QdrantClient, collection: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    offset: Any = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=128,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for point in points:
            records.append(point.model_dump(mode="json"))
        if offset is None:
            break
    return sorted(records, key=lambda point: str(point.get("id")))


def _collection_fingerprint(records: Sequence[Dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for point in records:
        digest.update(
            json.dumps(point, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _verify_collection_fingerprint(actual: str, expected: str) -> None:
    if actual != expected:
        raise ValueError(f"Collection fingerprint mismatch: {actual} != {expected}")


def _available_labels(records: Sequence[Dict[str, Any]]) -> tuple[set[int], set[str]]:
    table_indices: set[int] = set()
    text_doc_ids: set[str] = set()
    for record in records:
        payload = record.get("payload") or {}
        if payload.get("ticker") != "AAPL" or int(payload.get("fiscal_year") or 0) != 2024:
            continue
        if str(payload.get("form_type") or "").upper() != "10-K":
            continue
        doc_type = str(payload.get("doc_type") or "")
        doc_id = str(payload.get("doc_id") or "")
        if doc_type == "table":
            try:
                table_indices.add(int(payload.get("table_index")))
            except (TypeError, ValueError):
                pass
        elif doc_type == "text_chunk":
            normalized = _normalize_text_doc_id(doc_id)
            if normalized:
                text_doc_ids.add(normalized)
    return table_indices, text_doc_ids


def _verify_gold_labels(config: Dict[str, Any], records: Sequence[Dict[str, Any]]) -> None:
    table_indices, text_doc_ids = _available_labels(records)
    table_examples = (
        load_retrieval_eval_examples(config["datasets"]["table"]["path"])
        if config["datasets"]["table"].get("enabled")
        else []
    )
    text_examples = (
        load_retrieval_eval_examples(config["datasets"]["text"]["path"])
        if config["datasets"]["text"].get("enabled")
        else []
    )
    missing_tables = sorted(
        {
            index
            for example in table_examples
            for index in example.relevant_table_indices()
            if index not in table_indices
        }
    )
    missing_text = sorted(
        {
            doc_id
            for example in text_examples
            for doc_id in example.relevant_text_doc_ids(
                ticker=example.infer_ticker("AAPL"),
                fiscal_year=example.infer_fiscal_year(2024),
                form_type=example.infer_form_type("10-K"),
            )
            if doc_id not in text_doc_ids
        }
    )
    if missing_tables or missing_text:
        raise ValueError(
            f"Gold labels absent from frozen collection: tables={missing_tables}, text={missing_text}"
        )


def _server_version(host: str, port: int) -> str:
    response = requests.get(f"http://{host}:{port}", timeout=10)
    response.raise_for_status()
    return str(response.json().get("version") or "")


def _ollama_digest(base_url: str, model: str) -> str:
    response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=10)
    response.raise_for_status()
    for item in response.json().get("models") or []:
        if str(item.get("name")) == model:
            return str(item.get("digest") or "")
    raise ValueError(f"Ollama model not installed: {model}")


def _git_head(repo_root: Path = Path(".")) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    ).stdout.strip()


def _git_has_changes(args: Sequence[str], repo_root: Path) -> bool:
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.returncode == 1


def _assert_clean_evaluated_checkout(
    evaluated_sha: str,
    *,
    repo_root: Path = Path("."),
) -> None:
    head = _git_head(repo_root)
    if head != evaluated_sha:
        raise ValueError(f"HEAD does not match --evaluated-sha: {head} != {evaluated_sha}")
    if _git_has_changes(["diff", "--quiet", "--ignore-submodules=none", "--"], repo_root):
        raise ValueError("Canonical benchmark requires no unstaged tracked changes")
    if _git_has_changes(
        ["diff", "--cached", "--quiet", "--ignore-submodules=none", "--"],
        repo_root,
    ):
        raise ValueError("Canonical benchmark requires no staged tracked changes")

    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    ).stdout.splitlines()
    unsafe_untracked = sorted(
        path for path in untracked if not Path(path).parts or Path(path).parts[0] != "artifacts"
    )
    if unsafe_untracked:
        raise ValueError(
            "Canonical benchmark rejects untracked files outside artifacts/: "
            + ", ".join(unsafe_untracked)
        )


def _validate_runtime_environment(
    *,
    qdrant_host: str,
    qdrant_port: int,
    collection: str,
    embedding_api_url: str,
    embedding_model: str,
) -> None:
    expected = {
        "QDRANT_HOST": str(qdrant_host),
        "QDRANT_PORT": str(int(qdrant_port)),
        "QDRANT_COLLECTION_NAME": str(collection),
        "QWEN3_EMBED_API_URL": embedding_api_url.rstrip("/"),
        "QWEN3_EMBED_MODEL": str(embedding_model),
    }
    for name, expected_value in expected.items():
        raw_value = os.getenv(name)
        if raw_value is None:
            continue
        actual_value = raw_value.strip().rstrip("/") if name.endswith("API_URL") else raw_value.strip()
        if actual_value != expected_value:
            raise ValueError(
                f"Runtime configuration mismatch for {name}: "
                f"environment={actual_value!r}, verified={expected_value!r}"
            )


def _validate_fixed_config(config: Dict[str, Any]) -> None:
    if config.get("modes") != ABLATION_CONFIGS:
        raise ValueError("Frozen mode definitions do not match evaluator definitions")
    common = config["common"]
    expected = {
        "top_k": 10,
        "min_total_score": 0.0,
        "k_values": [1, 3, 5, 10],
        "candidate_top_k": RETRIEVAL_TOP_K,
        "branch_limit": max(50, RETRIEVAL_TOP_K * 10),
        "dedupe_limit": RERANK_CANDIDATE_LIMIT,
        "dense_vector": "dense",
        "bm25_vector": "bm25",
        "rrf_k": 60,
        "dense_weight": 1.0,
        "bm25_weight": 1.0,
        "embedding_model": "qwen3-embedding:8b",
        "reranker_model": "Qwen/Qwen3-Reranker-8B",
        "ragas_enabled": False,
    }
    for key, value in expected.items():
        if common.get(key) != value:
            raise ValueError(f"Frozen common config mismatch for {key}: {common.get(key)!r} != {value!r}")


def _validate_outputs(
    all_rows: Dict[str, Dict[str, List[Dict[str, Any]]]],
    output_root: Path,
    dataset_names: Sequence[str],
    expected_provenance: Dict[str, Any],
) -> None:
    for dataset in dataset_names:
        expected_ids: List[str] | None = None
        for mode in RETRIEVAL_MODES:
            rows = all_rows[mode][dataset]
            ids = [str(row["id"]) for row in rows]
            if expected_ids is None:
                expected_ids = ids
            elif ids != expected_ids:
                raise ValueError(f"Case IDs differ for {dataset}/{mode}")
            expected_components = ABLATION_CONFIGS[mode]
            for row in rows:
                if not row.get("retrieval_ok"):
                    raise ValueError(f"Retrieval failed for {dataset}/{mode}/{row.get('id')}")
                components = (row.get("trace") or {}).get("components") or {}
                if components != expected_components:
                    raise ValueError(f"Component trace mismatch for {dataset}/{mode}/{row.get('id')}")
                provenance = (row.get("trace") or {}).get("provenance") or {}
                expected_row_provenance = dict(expected_provenance)
                expected_row_provenance["embedding"] = (
                    expected_provenance["embedding"]
                    if expected_components["dense_enabled"]
                    else None
                )
                if provenance != expected_row_provenance:
                    raise ValueError(
                        f"Runtime provenance mismatch for {dataset}/{mode}/{row.get('id')}"
                    )
                timing = (row.get("trace") or {}).get("timing_ms") or {}
                if mode != "hybrid_reranker" and float(timing.get("rerank_ms") or 0) != 0.0:
                    raise ValueError(f"Unexpected reranker timing for {dataset}/{mode}/{row.get('id')}")
                if mode == "hybrid_reranker":
                    rerank = timing.get("rerank") or {}
                    if rerank.get("applied_backend") != "qwen3_api" or rerank.get("fallback_used"):
                        raise ValueError(f"Reranker backend mismatch for {dataset}/{mode}/{row.get('id')}")
            errors = _jsonl(output_root / mode / dataset / "errors.jsonl")
            if errors:
                raise ValueError(f"Errors present for {dataset}/{mode}: {len(errors)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen retrieval ablation v1 benchmark.")
    parser.add_argument("--evaluated-sha", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--expected-points", type=int, required=True)
    parser.add_argument("--expected-qdrant-version", required=True)
    parser.add_argument("--expected-embedding-digest", required=True)
    parser.add_argument("--qdrant-host", default=os.getenv("QDRANT_HOST", "localhost"))
    parser.add_argument("--qdrant-port", type=int, default=int(os.getenv("QDRANT_PORT", "6333")))
    parser.add_argument("--ollama-base-url", default="http://localhost:11434")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    _assert_clean_evaluated_checkout(args.evaluated_sha, repo_root=repo_root)
    output_root = Path(args.out_root)
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    _validate_fixed_config(config)
    embedding_model = config["common"]["embedding_model"]
    embedding_base_url = args.ollama_base_url.rstrip("/")
    embedding_api_url = f"{embedding_base_url}/api/embed"
    _validate_runtime_environment(
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        collection=args.collection,
        embedding_api_url=embedding_api_url,
        embedding_model=embedding_model,
    )
    if COLLECTION_NAME != args.collection:
        raise ValueError(f"Production collection mismatch: {COLLECTION_NAME} != {args.collection}")
    if QWEN3_EMBED_API_URL.rstrip("/") != embedding_api_url:
        raise ValueError(
            f"Production embedding API URL mismatch: {QWEN3_EMBED_API_URL} != {embedding_api_url}"
        )
    if QWEN3_EMBED_MODEL != config["common"]["embedding_model"]:
        raise ValueError(f"Production embedding model mismatch: {QWEN3_EMBED_MODEL}")
    if RERANK_MODEL != config["common"]["reranker_model"]:
        raise ValueError(f"Production reranker model mismatch: {RERANK_MODEL}")
    if not _current_qwen3_rerank_api_key():
        raise ValueError("Qwen3 reranker credential is unavailable")
    if not _current_qwen3_rerank_api_url():
        raise ValueError("Qwen3 reranker API URL is unavailable")
    active_datasets = {
        name: dataset for name, dataset in config["datasets"].items() if dataset.get("enabled")
    }
    if not active_datasets:
        raise ValueError("Frozen benchmark has no enabled datasets")
    dataset_hashes_before: Dict[str, str] = {}
    for name, dataset in config["datasets"].items():
        path = Path(dataset["path"])
        digest = _sha256(path)
        dataset_hashes_before[name] = digest
        if digest != dataset["sha256"]:
            raise ValueError(f"Dataset hash mismatch for {name}: {digest}")
        if len(load_retrieval_eval_examples(path)) != int(dataset["case_count"]):
            raise ValueError(f"Dataset case count mismatch for {name}")
    source_html = Path(config["corpus_provenance"]["source_html"])
    if _sha256(source_html) != config["corpus_provenance"]["source_html_sha256"]:
        raise ValueError("Tracked source HTML hash mismatch")

    client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
    qdrant_version = _server_version(args.qdrant_host, args.qdrant_port)
    if qdrant_version != args.expected_qdrant_version:
        raise ValueError(f"Qdrant version mismatch: {qdrant_version} != {args.expected_qdrant_version}")
    collection_info = client.get_collection(args.collection)
    if int(collection_info.points_count or 0) != args.expected_points:
        raise ValueError(
            f"Collection point count mismatch: {collection_info.points_count} != {args.expected_points}"
        )
    records_before = _collection_records(client, args.collection)
    if len(records_before) != args.expected_points:
        raise ValueError(f"Scrolled point count mismatch: {len(records_before)} != {args.expected_points}")
    corpus_fingerprint_before = _collection_fingerprint(records_before)
    expected_corpus_fingerprint = config["corpus_provenance"][
        "collection_fingerprint_sha256"
    ]
    _verify_collection_fingerprint(corpus_fingerprint_before, expected_corpus_fingerprint)
    _verify_gold_labels(config, records_before)

    embedding_digest = _ollama_digest(embedding_base_url, embedding_model)
    if embedding_digest != args.expected_embedding_digest:
        raise ValueError(f"Embedding digest mismatch: {embedding_digest}")
    requests.post(
        embedding_api_url,
        json={"model": embedding_model, "input": ["retrieval ablation warmup probe"]},
        timeout=120,
    ).raise_for_status()

    output_root.mkdir(parents=True)
    all_rows: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    summaries: Dict[str, Dict[str, Dict[str, Any]]] = {}
    raw_hashes: Dict[str, str] = {}
    cache_roots: List[tempfile.TemporaryDirectory[str]] = []
    for mode in RETRIEVAL_MODES:
        cache_root = tempfile.TemporaryDirectory(prefix=f"retrieval-ablation-{mode}-")
        cache_roots.append(cache_root)
        os.environ["SEC_QUERY_EMBED_CACHE_DIR"] = cache_root.name
        all_rows[mode] = {}
        summaries[mode] = {}
        for dataset_name, dataset in active_datasets.items():
            destination = output_root / mode / dataset_name
            summary, rows, errors = run_retrieval_eval(
                eval_path=dataset["path"],
                out_dir=str(destination),
                eval_mode=dataset_name,
                retrieval_mode=mode,
                top_k=int(config["common"]["top_k"]),
                k_values=config["common"]["k_values"],
                default_ticker=config["common"]["ticker"],
                default_fiscal_year=int(config["common"]["fiscal_year"]),
                default_form_type=config["common"]["form_type"],
                default_doc_types=dataset["doc_types"],
                min_total_score=float(config["common"]["min_total_score"]),
                enable_ragas=False,
                text_embed_api_url=embedding_api_url,
                text_embed_model=embedding_model,
                retrieval_client=client,
                qdrant_host=args.qdrant_host,
                qdrant_port=args.qdrant_port,
                qdrant_collection_name=args.collection,
                fail_fast=True,
            )
            if errors:
                raise ValueError(f"Evaluation errors for {mode}/{dataset_name}: {errors}")
            all_rows[mode][dataset_name] = [row.model_dump(mode="json") for row in rows]
            summaries[mode][dataset_name] = summary.model_dump(mode="json")
    for cache_root in cache_roots:
        cache_root.cleanup()

    expected_provenance = {
        "qdrant": {
            "host": args.qdrant_host,
            "port": args.qdrant_port,
            "collection": args.collection,
        },
        "embedding": {
            "api_url": embedding_api_url,
            "model": embedding_model,
        },
    }
    _validate_outputs(
        all_rows,
        output_root,
        list(active_datasets),
        expected_provenance,
    )
    dataset_hashes_after = {
        name: _sha256(Path(dataset["path"])) for name, dataset in config["datasets"].items()
    }
    if dataset_hashes_after != dataset_hashes_before:
        raise ValueError("Dataset hashes changed during benchmark")
    records_after = _collection_records(client, args.collection)
    corpus_fingerprint_after = _collection_fingerprint(records_after)
    if corpus_fingerprint_after != corpus_fingerprint_before:
        raise ValueError("Collection fingerprint changed during benchmark")

    for mode in RETRIEVAL_MODES:
        for dataset in active_datasets:
            for filename in RAW_FILENAMES:
                path = output_root / mode / dataset / filename
                raw_hashes[str(path.relative_to(output_root))] = _sha256(path)

    absolute_metrics = {
        mode: {
            dataset: summaries[mode][dataset]["deterministic"]
            for dataset in active_datasets
        }
        for mode in RETRIEVAL_MODES
    }
    comparisons = {
        "dense_to_hybrid": ("dense_only", "hybrid"),
        "hybrid_to_hybrid_reranker": ("hybrid", "hybrid_reranker"),
        "dense_to_hybrid_reranker": ("dense_only", "hybrid_reranker"),
    }
    deltas = {
        comparison: {
            dataset: _delta(
                absolute_metrics[source][dataset],
                absolute_metrics[target][dataset],
            )
            for dataset in active_datasets
        }
        for comparison, (source, target) in comparisons.items()
    }
    latency = {
        mode: {
            dataset: {
                stage: _latency_stats(all_rows[mode][dataset], stage)
                for stage in ("candidate_retrieval_ms", "rerank_ms", "total_retrieval_ms")
            }
            for dataset in active_datasets
        }
        for mode in RETRIEVAL_MODES
    }
    comparison = {
        "benchmark_id": config["benchmark_id"],
        "evaluated_sha": args.evaluated_sha,
        "dataset_hashes_before": dataset_hashes_before,
        "dataset_hashes_after": dataset_hashes_after,
        "dataset_case_counts": {
            name: int(dataset["case_count"]) for name, dataset in active_datasets.items()
        },
        "excluded_datasets": {
            name: {
                "path": dataset["path"],
                "sha256": dataset["sha256"],
                "case_count": int(dataset["case_count"]),
                "reason": dataset["exclusion_reason"],
            }
            for name, dataset in config["datasets"].items()
            if not dataset.get("enabled")
        },
        "corpus": {
            "endpoint": {
                "host": args.qdrant_host,
                "port": args.qdrant_port,
            },
            "collection": args.collection,
            "points": args.expected_points,
            "qdrant_version": qdrant_version,
            "fingerprint_before": corpus_fingerprint_before,
            "fingerprint_after": corpus_fingerprint_after,
            "expected_fingerprint": expected_corpus_fingerprint,
            "provenance": config["corpus_provenance"],
        },
        "models": {
            "embedding_model": embedding_model,
            "embedding_digest": embedding_digest,
            "embedding_base_url": embedding_base_url,
            "embedding_api_url": embedding_api_url,
            "reranker_model": config["common"]["reranker_model"],
            "reranker_backend": "qwen3_api",
            "reranker_api_url": _current_qwen3_rerank_api_url(),
            "reranker_service_digest": None,
        },
        "common_config": config["common"],
        "supersedes": config["supersedes"],
        "configurations": config["modes"],
        "absolute_metrics": absolute_metrics,
        "deltas": deltas,
        "latency": latency,
        "errors": {mode: {dataset: 0 for dataset in active_datasets} for mode in RETRIEVAL_MODES},
        "raw_artifact_sha256": raw_hashes,
    }
    comparison_path = output_root / "comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest_entries = {**raw_hashes, "comparison.json": _sha256(comparison_path)}
    (output_root / "manifest.sha256").write_text(
        "".join(f"{digest}  {path}\n" for path, digest in sorted(manifest_entries.items())),
        encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
