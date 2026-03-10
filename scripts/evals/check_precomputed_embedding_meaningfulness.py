#!/usr/bin/env python3
"""Check whether precomputed text embeddings are semantically meaningful.

This script is intentionally narrow:
- one chunking strategy at a time
- Qwen3 dense-only retrieval probe
- reuse existing benchmark/eval artifacts where possible
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient

from retrieval.evaluator import dense_search_sec_docs, embed_query_qwen3


DEFAULT_STRATEGY = "flat_512_no_overlap"
DEFAULT_BENCHMARK_SPEC = "data/evals/retrieval/benchmark/chunking_strategy_eval_set_8_tickers_40.jsonl"
DEFAULT_ROW_MAP = "data/embedding_batches/row_map.parquet"
DEFAULT_UNIQUE_INDEX = "data/embedding_batches/unique_index.parquet"
DEFAULT_EMBEDDINGS = "data/embedding_batches/embeddings_unique.f16.npy"
DEFAULT_CHUNKING_INPUT = "data/embedding_batches/chunking_strategy_embed_inputs.jsonl"
DEFAULT_COLLECTION_PREFIX = "stage1_qwen3_timed_20260305"
DEFAULT_EXISTING_PER_QUERY_ROOT = "artifacts/evals/chunking_bench_stage1_qwen3_timed"
DEFAULT_OUTPUT_ROOT = "artifacts/evals/precomputed_embedding_check"
DEFAULT_FORM_TYPE = "10-K"
EASY_NOTE_HEAVY_TOKEN = "note_heavy"

_CHUNK_INDEX_RE = re.compile(r"::text::(?P<chunk>\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether precomputed embeddings are meaningful for retrieval."
    )
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--benchmark-spec-jsonl", default=DEFAULT_BENCHMARK_SPEC)
    parser.add_argument("--eval-jsonl", default=None)
    parser.add_argument("--row-map", default=DEFAULT_ROW_MAP)
    parser.add_argument("--unique-index", default=DEFAULT_UNIQUE_INDEX)
    parser.add_argument("--embeddings", default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--chunking-input", default=DEFAULT_CHUNKING_INPUT)
    parser.add_argument("--existing-per-query-jsonl", default=None)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--collection-prefix", default=DEFAULT_COLLECTION_PREFIX)
    parser.add_argument("--qdrant-host", default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--embed-api-url", default="http://localhost:11434/api/embed")
    parser.add_argument("--embed-model", default="qwen3-embedding:8b")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--probe-size", type=int, default=8)
    parser.add_argument("--pair-sample-size", type=int, default=200)
    parser.add_argument("--vector-sample-size", type=int, default=1000)
    parser.add_argument("--neighbor-sample-size", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _strip_split_suffix(doc_id: str) -> str:
    return str(doc_id or "").split("::split::", 1)[0]


def _canonicalize_doc_id(doc_id: str) -> str:
    text = str(doc_id or "").strip()
    if not text:
        return ""

    text = _strip_split_suffix(text)
    if "::text::" not in text:
        return text

    left, right = text.split("::text::", 1)
    prefix = left.split("::", 1)[0]
    return f"{prefix}::text::{right}"


def _parse_chunk_index(doc_id: str) -> int | None:
    match = _CHUNK_INDEX_RE.search(str(doc_id or ""))
    if not match:
        return None
    try:
        return int(match.group("chunk"))
    except ValueError:
        return None


def _safe_float_list(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _hit_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    return float(any(doc_id in relevant for doc_id in retrieved[:k]))


def _load_runner_eval_rows(path: Path) -> Dict[str, dict]:
    rows = _load_jsonl(path)
    return {str(row["query_id"]): row for row in rows}


def _load_existing_per_query(path: Path | None) -> Dict[str, dict]:
    if path is None or not path.exists():
        return {}
    rows = _load_jsonl(path)
    return {str(row["id"]): row for row in rows}


def _load_strategy_doc_metadata(path: Path, strategy: str) -> Dict[str, dict]:
    metadata_by_id: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            metadata = row.get("metadata", {})
            if str(metadata.get("chunking_strategy") or "") != strategy:
                continue
            metadata_by_id[str(row["id"])] = row
    return metadata_by_id


def compute_alignment_check(
    eval_rows: Dict[str, dict],
    per_query_rows: Dict[str, dict],
) -> Dict[str, Any]:
    if not per_query_rows:
        return {
            "status": "skipped",
            "reason": "existing per-query retrieval artifact not found",
        }

    exact_hits = 0
    canonical_hits = 0
    changed_to_hit = 0
    examples: List[Dict[str, Any]] = []
    evaluated = 0

    for query_id, per_query in per_query_rows.items():
        eval_row = eval_rows.get(query_id)
        if eval_row is None:
            continue

        gold_raw = [str(x).strip() for x in eval_row.get("relevant_doc_ids", []) if str(x).strip()]
        ret_raw = [str(x).strip() for x in per_query.get("retrieved_doc_ids", []) if str(x).strip()]
        if not gold_raw:
            continue

        evaluated += 1
        gold_exact = set(gold_raw)
        gold_canonical = {_canonicalize_doc_id(doc_id) for doc_id in gold_raw if _canonicalize_doc_id(doc_id)}
        ret_exact = ret_raw[:10]
        ret_canonical = [_canonicalize_doc_id(doc_id) for doc_id in ret_raw[:10] if _canonicalize_doc_id(doc_id)]

        exact_hit = any(doc_id in gold_exact for doc_id in ret_exact)
        canonical_hit = any(doc_id in gold_canonical for doc_id in ret_canonical)

        exact_hits += int(exact_hit)
        canonical_hits += int(canonical_hit)

        if not exact_hit and canonical_hit:
            changed_to_hit += 1
            if len(examples) < 10:
                overlap = [doc_id for doc_id in ret_canonical if doc_id in gold_canonical]
                examples.append(
                    {
                        "query_id": query_id,
                        "query": per_query.get("query"),
                        "gold_exact_sample": gold_raw[:6],
                        "gold_canonical_sample": sorted(gold_canonical)[:6],
                        "retrieved_exact_top10": ret_exact,
                        "retrieved_canonical_top10": ret_canonical,
                        "canonical_overlap": overlap,
                    }
                )

    return {
        "status": "ok",
        "num_queries_evaluated": evaluated,
        "exact_hit_at_10_rate": float(exact_hits) / float(evaluated) if evaluated else 0.0,
        "canonicalized_hit_at_10_rate": float(canonical_hits) / float(evaluated) if evaluated else 0.0,
        "changed_to_hit_count": changed_to_hit,
        "examples": examples,
    }


def compute_vector_integrity(
    *,
    row_map: pd.DataFrame,
    unique_index: pd.DataFrame,
    embeddings: np.ndarray,
    eval_rows: Dict[str, dict],
    doc_metadata_by_id: Dict[str, dict],
    vector_sample_size: int,
    pair_sample_size: int,
    neighbor_sample_size: int,
    random_seed: int,
) -> Dict[str, Any]:
    rng = random.Random(random_seed)
    row_map_ids = {str(value) for value in row_map["id"].tolist()}
    row_map_base_ids = {_strip_split_suffix(doc_id) for doc_id in row_map_ids}

    gold_total = 0
    gold_exact_mapped = 0
    gold_base_mapped = 0
    gold_unmapped: List[str] = []
    for row in eval_rows.values():
        for doc_id in row.get("relevant_doc_ids", []):
            text = str(doc_id).strip()
            if not text:
                continue
            gold_total += 1
            if text in row_map_ids:
                gold_exact_mapped += 1
            elif _strip_split_suffix(text) in row_map_base_ids:
                gold_base_mapped += 1
            else:
                if len(gold_unmapped) < 20:
                    gold_unmapped.append(text)

    unique_rows = sorted({int(value) for value in row_map["unique_row"].tolist()})
    sample_unique_rows = unique_rows[:]
    if len(sample_unique_rows) > vector_sample_size:
        sample_unique_rows = rng.sample(sample_unique_rows, vector_sample_size)
        sample_unique_rows.sort()

    sampled = np.asarray(embeddings[sample_unique_rows], dtype=np.float32)
    finite_mask = np.isfinite(sampled)
    finite_row_mask = np.all(finite_mask, axis=1) if sampled.size else np.asarray([], dtype=bool)
    norms = np.linalg.norm(sampled, axis=1) if sampled.size else np.asarray([], dtype=np.float32)
    zero_mask = norms <= 1e-8 if sampled.size else np.asarray([], dtype=bool)

    pair_cosines: List[float] = []
    if len(sample_unique_rows) >= 2:
        pair_count = min(pair_sample_size, len(sample_unique_rows) // 2)
        for _ in range(pair_count):
            left, right = rng.sample(sample_unique_rows, 2)
            left_vec = np.asarray(embeddings[left], dtype=np.float32)
            right_vec = np.asarray(embeddings[right], dtype=np.float32)
            denom = float(np.linalg.norm(left_vec) * np.linalg.norm(right_vec))
            if denom <= 1e-8 or not math.isfinite(denom):
                continue
            pair_cosines.append(float(np.dot(left_vec, right_vec) / denom))

    doc_rows = row_map.copy()
    doc_rows = doc_rows[doc_rows["id"].isin(doc_metadata_by_id.keys())].copy()
    doc_rows = doc_rows.sort_values("id").reset_index(drop=True)

    neighbor_samples: List[Dict[str, Any]] = []
    neighbor_summary = {
        "sampled_docs": 0,
        "top1_same_filing_rate": 0.0,
        "top5_same_section_rate": 0.0,
        "top5_adjacent_chunk_rate": 0.0,
    }
    if not doc_rows.empty:
        sample_size = min(neighbor_sample_size, len(doc_rows))
        sample_indices = rng.sample(list(range(len(doc_rows))), sample_size)
        doc_unique_rows = doc_rows["unique_row"].astype(int).to_numpy()
        doc_vectors = np.asarray(embeddings[doc_unique_rows], dtype=np.float32)
        doc_norms = np.linalg.norm(doc_vectors, axis=1)
        same_filing_top1 = 0
        same_section_top5 = 0
        adjacent_chunk_top5 = 0

        for idx in sample_indices:
            query_row = doc_rows.iloc[idx]
            query_meta = doc_metadata_by_id[str(query_row["id"])]
            query_payload = query_meta.get("metadata", {})
            query_vector = doc_vectors[idx]
            query_norm = float(doc_norms[idx])
            if query_norm <= 1e-8 or not math.isfinite(query_norm):
                continue

            denom = doc_norms * query_norm
            scores = np.divide(
                np.dot(doc_vectors, query_vector),
                denom,
                out=np.full_like(doc_norms, -1.0, dtype=np.float32),
                where=denom > 1e-8,
            )
            scores[idx] = -1.0
            neighbor_idx = np.argsort(scores)[-5:][::-1].tolist()
            neighbors: List[Dict[str, Any]] = []
            query_chunk = _parse_chunk_index(str(query_row["id"]))
            query_section = str(query_payload.get("section_path") or "")
            query_ticker = str(query_row["ticker"]).upper()
            query_year = int(query_row["fiscal_year"])

            same_filing_top1_flag = False
            same_section_top5_flag = False
            adjacent_chunk_top5_flag = False
            for rank, n_idx in enumerate(neighbor_idx, start=1):
                neighbor_row = doc_rows.iloc[n_idx]
                neighbor_meta = doc_metadata_by_id[str(neighbor_row["id"])]
                neighbor_payload = neighbor_meta.get("metadata", {})
                neighbor_chunk = _parse_chunk_index(str(neighbor_row["id"]))
                same_filing = (
                    str(neighbor_row["ticker"]).upper() == query_ticker
                    and int(neighbor_row["fiscal_year"]) == query_year
                )
                same_section = str(neighbor_payload.get("section_path") or "") == query_section and bool(query_section)
                adjacent_chunk = (
                    same_filing
                    and query_chunk is not None
                    and neighbor_chunk is not None
                    and abs(query_chunk - neighbor_chunk) <= 1
                )
                if rank == 1 and same_filing:
                    same_filing_top1_flag = True
                if same_section:
                    same_section_top5_flag = True
                if adjacent_chunk:
                    adjacent_chunk_top5_flag = True

                neighbors.append(
                    {
                        "rank": rank,
                        "doc_id": str(neighbor_row["id"]),
                        "score": float(scores[n_idx]),
                        "same_filing": same_filing,
                        "same_section": same_section,
                        "adjacent_chunk": adjacent_chunk,
                        "section_path": str(neighbor_payload.get("section_path") or ""),
                    }
                )

            same_filing_top1 += int(same_filing_top1_flag)
            same_section_top5 += int(same_section_top5_flag)
            adjacent_chunk_top5 += int(adjacent_chunk_top5_flag)
            neighbor_samples.append(
                {
                    "doc_id": str(query_row["id"]),
                    "section_path": query_section,
                    "neighbors": neighbors,
                }
            )

        sample_count = len(neighbor_samples)
        if sample_count:
            neighbor_summary = {
                "sampled_docs": sample_count,
                "top1_same_filing_rate": float(same_filing_top1) / float(sample_count),
                "top5_same_section_rate": float(same_section_top5) / float(sample_count),
                "top5_adjacent_chunk_rate": float(adjacent_chunk_top5) / float(sample_count),
            }

    return {
        "status": "ok",
        "rows_for_strategy": int(len(row_map)),
        "doc_rows_with_metadata": int(len(doc_rows)),
        "embedding_rows": int(len(embeddings)),
        "unique_index_rows": int(len(unique_index)),
        "embedding_count_matches_unique_index": bool(len(unique_index) == len(embeddings)),
        "gold_doc_id_mapping": {
            "total": gold_total,
            "exact_mapped": gold_exact_mapped,
            "base_mapped": gold_base_mapped,
            "unmapped_count": max(gold_total - gold_exact_mapped - gold_base_mapped, 0),
            "unmapped_examples": gold_unmapped,
        },
        "vector_sample": {
            "sample_size": int(len(sample_unique_rows)),
            "rows_all_finite": int(np.sum(finite_row_mask)) if len(finite_row_mask) else 0,
            "rows_with_non_finite": int(len(finite_row_mask) - int(np.sum(finite_row_mask))) if len(finite_row_mask) else 0,
            "zero_vector_count": int(np.sum(zero_mask)) if len(zero_mask) else 0,
            "zero_vector_rate": float(np.mean(zero_mask)) if len(zero_mask) else 0.0,
            "norm_stats": _safe_float_list(norms.tolist()),
        },
        "pairwise_cosine_stats": _safe_float_list(pair_cosines),
        "neighbor_sanity": {
            "summary": neighbor_summary,
            "samples": neighbor_samples[:10],
        },
    }


def _select_probe_queries(
    *,
    benchmark_rows: Sequence[dict],
    eval_rows: Dict[str, dict],
    probe_size: int,
) -> List[dict]:
    selected: List[dict] = []
    for row in benchmark_rows:
        qid = str(row.get("qid") or "")
        year_scope = row.get("year_scope") or []
        if len(year_scope) != 1:
            continue
        if str(row.get("difficulty") or "").lower() != "medium":
            continue
        if str(row.get("query_type") or "").lower() == "comparative":
            continue
        retrieval_profile = str(row.get("retrieval_profile") or "").lower()
        if EASY_NOTE_HEAVY_TOKEN in retrieval_profile:
            continue
        if qid not in eval_rows:
            continue
        selected.append(row)
        if len(selected) >= probe_size:
            break
    return selected


def run_semantic_probe(
    *,
    benchmark_rows: Sequence[dict],
    eval_rows: Dict[str, dict],
    collection: str,
    qdrant_host: str,
    qdrant_port: int,
    embed_api_url: str,
    embed_model: str,
    top_k: int,
    probe_size: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    selected = _select_probe_queries(
        benchmark_rows=benchmark_rows,
        eval_rows=eval_rows,
        probe_size=probe_size,
    )
    if not selected:
        return [], {
            "status": "skipped",
            "reason": "no eligible probe queries found",
            "num_queries": 0,
        }

    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    per_query_rows: List[Dict[str, Any]] = []

    for row in selected:
        qid = str(row["qid"])
        eval_row = eval_rows[qid]
        ticker = str(eval_row.get("ticker") or "").upper()
        fiscal_year = int(eval_row["fiscal_year"])
        query = str(eval_row["query"])
        gold_canonical = sorted(
            {
                _canonicalize_doc_id(doc_id)
                for doc_id in eval_row.get("relevant_doc_ids", [])
                if _canonicalize_doc_id(doc_id)
            }
        )

        results = dense_search_sec_docs(
            query,
            client=client,
            embed_fn=lambda value: embed_query_qwen3(
                value,
                api_url=embed_api_url,
                model=embed_model,
            ),
            collection_name=collection,
            using_dense="dense",
            top_k=top_k,
            doc_types=["text_chunk"],
            ticker=ticker,
            fiscal_year=fiscal_year,
            form_type=DEFAULT_FORM_TYPE,
        )

        retrieved: List[str] = []
        retrieved_raw: List[str] = []
        for point in results:
            payload = getattr(point, "payload", None) or {}
            raw_doc_id = str(payload.get("doc_id") or "")
            if not raw_doc_id:
                continue
            retrieved_raw.append(raw_doc_id)
            canonical = _canonicalize_doc_id(raw_doc_id)
            if canonical:
                retrieved.append(canonical)

        gold_set = set(gold_canonical)
        per_query_rows.append(
            {
                "query_id": qid,
                "query": query,
                "ticker": ticker,
                "fiscal_year": fiscal_year,
                "gold_canonical_doc_ids": gold_canonical,
                "retrieved_raw_doc_ids": retrieved_raw,
                "retrieved_canonical_doc_ids": retrieved,
                "hit_at_1": _hit_at_k(retrieved, gold_set, 1),
                "hit_at_3": _hit_at_k(retrieved, gold_set, 3),
                "hit_at_10": _hit_at_k(retrieved, gold_set, 10),
            }
        )

    hit1 = float(np.mean([row["hit_at_1"] for row in per_query_rows])) if per_query_rows else 0.0
    hit3 = float(np.mean([row["hit_at_3"] for row in per_query_rows])) if per_query_rows else 0.0
    hit10 = float(np.mean([row["hit_at_10"] for row in per_query_rows])) if per_query_rows else 0.0
    queries_with_hit10 = sum(int(row["hit_at_10"] > 0.0) for row in per_query_rows)
    passed = bool(
        per_query_rows
        and hit10 >= 0.75
        and hit3 >= 0.50
        and queries_with_hit10 >= 5
    )

    return per_query_rows, {
        "status": "ok",
        "num_queries": len(per_query_rows),
        "hit_at_1": hit1,
        "hit_at_3": hit3,
        "hit_at_10": hit10,
        "queries_with_hit_at_10": queries_with_hit10,
        "passed": passed,
        "thresholds": {
            "min_hit_at_10": 0.75,
            "min_hit_at_3": 0.50,
            "min_queries_with_hit_at_10": 5,
        },
    }


def build_final_assessment(
    *,
    alignment: Dict[str, Any],
    integrity: Dict[str, Any],
    semantic_summary: Dict[str, Any],
) -> Dict[str, Any]:
    mapping_ok = (
        integrity.get("embedding_count_matches_unique_index", False)
        and int(((integrity.get("gold_doc_id_mapping") or {}).get("unmapped_count") or 0)) == 0
    )
    vector_sample = integrity.get("vector_sample") or {}
    vectors_ok = (
        int(vector_sample.get("rows_with_non_finite", 0)) == 0
        and float(vector_sample.get("zero_vector_rate", 1.0)) <= 0.001
    )
    integrity_ok = bool(mapping_ok and vectors_ok)

    alignment_confounder = (
        alignment.get("status") == "ok"
        and float(alignment.get("canonicalized_hit_at_10_rate", 0.0))
        > float(alignment.get("exact_hit_at_10_rate", 0.0))
    )
    semantic_ok = semantic_summary.get("status") == "ok" and bool(semantic_summary.get("passed"))

    if not integrity_ok:
        classification = "mapping_broken"
        rationale = "Mapping or vector-integrity checks failed before semantic judgment."
    elif semantic_ok:
        classification = "meaningful"
        rationale = "Integrity checks passed and the small semantic probe met the acceptance thresholds."
    else:
        classification = "likely_not_meaningful"
        rationale = "Integrity checks passed, but the semantic probe did not meet the acceptance thresholds."

    return {
        "classification": classification,
        "rationale": rationale,
        "signals": {
            "alignment_confounder_detected": alignment_confounder,
            "integrity_ok": integrity_ok,
            "semantic_probe_passed": semantic_ok,
        },
        "evidence": {
            "alignment": {
                "status": alignment.get("status"),
                "exact_hit_at_10_rate": alignment.get("exact_hit_at_10_rate", 0.0),
                "canonicalized_hit_at_10_rate": alignment.get("canonicalized_hit_at_10_rate", 0.0),
                "changed_to_hit_count": alignment.get("changed_to_hit_count", 0),
            },
            "integrity": {
                "embedding_count_matches_unique_index": integrity.get("embedding_count_matches_unique_index"),
                "gold_doc_id_unmapped_count": ((integrity.get("gold_doc_id_mapping") or {}).get("unmapped_count")),
                "zero_vector_rate": ((integrity.get("vector_sample") or {}).get("zero_vector_rate")),
            },
            "semantic_probe": semantic_summary,
        },
    }


def main() -> int:
    args = parse_args()

    strategy = str(args.strategy)
    benchmark_spec_path = Path(args.benchmark_spec_jsonl)
    eval_jsonl_path = Path(
        args.eval_jsonl
        or f"data/evals/retrieval/benchmark/chunking_strategy_eval_set_8_tickers_40_{strategy}.runner.jsonl"
    )
    existing_per_query_path = Path(
        args.existing_per_query_jsonl
        or f"{DEFAULT_EXISTING_PER_QUERY_ROOT}/{strategy}/eval/per_query.jsonl"
    )
    collection = str(args.collection or f"{args.collection_prefix}_{strategy}")
    output_dir = Path(args.output_dir or f"{DEFAULT_OUTPUT_ROOT}/{strategy}")
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_rows = _load_jsonl(benchmark_spec_path)
    eval_rows = _load_runner_eval_rows(eval_jsonl_path)
    existing_per_query = _load_existing_per_query(existing_per_query_path)
    doc_metadata_by_id = _load_strategy_doc_metadata(Path(args.chunking_input), strategy)

    row_map = pd.read_parquet(args.row_map)
    row_map = row_map[row_map["chunking_strategy"] == strategy].copy()
    row_map = row_map.reset_index(drop=True)
    unique_index = pd.read_parquet(args.unique_index)
    embeddings = np.load(args.embeddings, mmap_mode="r")

    alignment = compute_alignment_check(eval_rows=eval_rows, per_query_rows=existing_per_query)
    integrity = compute_vector_integrity(
        row_map=row_map,
        unique_index=unique_index,
        embeddings=embeddings,
        eval_rows=eval_rows,
        doc_metadata_by_id=doc_metadata_by_id,
        vector_sample_size=args.vector_sample_size,
        pair_sample_size=args.pair_sample_size,
        neighbor_sample_size=args.neighbor_sample_size,
        random_seed=args.random_seed,
    )

    semantic_probe_rows: List[Dict[str, Any]] = []
    try:
        semantic_probe_rows, semantic_summary = run_semantic_probe(
            benchmark_rows=benchmark_rows,
            eval_rows=eval_rows,
            collection=collection,
            qdrant_host=args.qdrant_host,
            qdrant_port=int(args.qdrant_port),
            embed_api_url=args.embed_api_url,
            embed_model=args.embed_model,
            top_k=int(args.top_k),
            probe_size=int(args.probe_size),
        )
    except Exception as exc:
        semantic_summary = {
            "status": "error",
            "reason": str(exc),
            "collection": collection,
            "num_queries": 0,
            "passed": False,
        }

    final_assessment = build_final_assessment(
        alignment=alignment,
        integrity=integrity,
        semantic_summary=semantic_summary,
    )
    final_assessment["config"] = {
        "strategy": strategy,
        "eval_jsonl": str(eval_jsonl_path),
        "benchmark_spec_jsonl": str(benchmark_spec_path),
        "existing_per_query_jsonl": str(existing_per_query_path),
        "collection": collection,
        "embed_model": args.embed_model,
        "top_k": int(args.top_k),
        "probe_size": int(args.probe_size),
    }

    _write_json(output_dir / "alignment_check.json", alignment)
    _write_json(output_dir / "vector_integrity.json", integrity)
    _write_jsonl(output_dir / "semantic_probe_per_query.jsonl", semantic_probe_rows)
    _write_json(output_dir / "semantic_probe_summary.json", semantic_summary)
    _write_json(output_dir / "final_assessment.json", final_assessment)

    print(json.dumps(final_assessment, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
