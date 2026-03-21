from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from evals.ragas_retrieval_metrics import RagasRetrievalConfig, evaluate_ragas_retrieval
from evals.retrieval_eval_contracts import (
    RetrievalEvalRow,
    RetrievalEvalSummary,
    load_retrieval_eval_examples,
)
from evals.retrieval_metrics import hit_at_k, mrr_at_k, ndcg_at_k, recall_at_k
from mcp_server.tools.sec_retrieval import retrieve_scored_points, sec_retrieve_tables
from qdrant_client import QdrantClient, models
from retrieval.evaluator import dense_search_sec_docs, embed_query_qwen3

_TABLE_INDEX_RE = re.compile(r"::table::(\d+)")
_TEXT_DOC_ID_RE = re.compile(r"(?P<base>.+?::text::.+?)(?::+split::\d+)?$")


def _parse_table_index_from_doc_id(doc_id: str) -> Optional[int]:
    if not doc_id:
        return None
    m = _TABLE_INDEX_RE.search(doc_id)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _extract_payload(top_table: Dict[str, Any]) -> Dict[str, Any]:
    if any(key in top_table for key in ("doc_id", "content", "match_text", "section_path", "table_index")):
        return top_table

    table_obj = top_table.get("table")
    if isinstance(table_obj, dict):
        payload = table_obj.get("payload")
        if isinstance(payload, dict):
            return payload
    # sec_retrieve_tables currently returns `table` as a qdrant ScoredPoint object.
    payload_obj = getattr(table_obj, "payload", None)
    if isinstance(payload_obj, dict):
        return payload_obj
    if hasattr(table_obj, "model_dump"):
        try:
            dumped = table_obj.model_dump()
            payload = dumped.get("payload")
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    if hasattr(table_obj, "dict"):
        try:
            dumped = table_obj.dict()
            payload = dumped.get("payload")
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    return {}


def _extract_context(payload: Dict[str, Any]) -> str:
    for key in ("rerank_table_summary", "content", "rerank_original_content"):
        v = payload.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def _recall_at_k_doc_ids(retrieved_doc_ids: Sequence[str], relevant_doc_ids: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant = {str(x).strip() for x in relevant_doc_ids if str(x).strip()}
    if not relevant:
        return 0.0
    hits = {str(x).strip() for x in retrieved_doc_ids[:k] if str(x).strip() in relevant}
    return float(len(hits)) / float(len(relevant))


def _build_text_dense_client() -> QdrantClient:
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    return QdrantClient(host=host, port=port)


def _extract_payload_from_scored_point(point: models.ScoredPoint) -> Dict[str, Any]:
    payload = getattr(point, "payload", None)
    if isinstance(payload, dict):
        return payload
    return {}


def _normalize_text_doc_id(doc_id: str) -> str:
    doc_id = str(doc_id or "").strip()
    if not doc_id:
        return ""

    # Strategy-specific variants are expected in benchmark experiments:
    #   <prefix>::<strategy>::text::<rest>
    # Normalize to the canonical:
    #   <prefix>::text::<rest>
    if "::text::" in doc_id:
        left, right = doc_id.split("::text::", 1)
        if "::" in left:
            doc_id = f"{left.split('::', 1)[0]}::text::{right}"

    m = _TEXT_DOC_ID_RE.match(doc_id)
    if m:
        return m.group("base")

    split_token = "::split::"
    if split_token in doc_id:
        return doc_id.split(split_token, 1)[0]

    return doc_id


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_retrieval_eval(
    *,
    eval_path: str,
    out_dir: str,
    eval_mode: str = "auto",
    top_k: int = 10,
    k_values: Sequence[int] = (1, 3, 5, 10),
    default_ticker: str = "AAPL",
    default_fiscal_year: int = 2024,
    default_form_type: str = "10-K",
    default_doc_types: Optional[List[str]] = None,
    min_total_score: int = 0,
    enable_ragas: bool = True,
    text_dense_only: bool = False,
    text_embed_api_url: str = "http://localhost:11434/api/embed",
    text_embed_model: str = "qwen3-embedding:8b",
    ragas_config: Optional[RagasRetrievalConfig] = None,
    allow_empty_labels: bool = False,
    fail_fast: bool = False,
) -> Tuple[RetrievalEvalSummary, List[RetrievalEvalRow], List[Dict[str, Any]]]:
    examples = load_retrieval_eval_examples(eval_path)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    collection = os.getenv("QDRANT_COLLECTION_NAME", "sec_docs_dense_bm25")

    eval_mode = str(eval_mode or "auto").strip().lower()
    if eval_mode not in {"auto", "table", "text"}:
        raise ValueError(f"Unsupported eval_mode='{eval_mode}'. Expected one of: auto, table, text.")

    ks = sorted({int(k) for k in k_values if int(k) > 0})
    if not ks:
        ks = [1, 3, 5, 10]

    max_k = max(top_k, max(ks))
    table_doc_types = default_doc_types or ["table"]
    text_doc_types = default_doc_types or ["text_chunk"]

    rows: List[RetrievalEvalRow] = []
    errors: List[Dict[str, Any]] = []
    ragas_samples: List[Dict[str, Any]] = []

    total_start = time.perf_counter()
    text_client: Optional[QdrantClient] = None

    for ex in examples:
        ex_id = ex.example_id

        # Prefer explicit overrides; then fallback to row-level fields.
        ticker = ex.infer_ticker(default_ticker=default_ticker)
        fiscal_year = ex.infer_fiscal_year(default_fiscal_year=default_fiscal_year)
        form_type = ex.infer_form_type(default_form_type=default_form_type)

        if eval_mode == "table":
            mode = "table"
        elif eval_mode == "text":
            mode = "text"
        else:
            mode = "table" if ex.has_table_labels() else ("text" if ex.has_text_labels() else "unknown")

        per_row_doc_types = ex.doc_types or (table_doc_types if mode == "table" else text_doc_types)
        relevant_indices = ex.relevant_table_indices() if mode == "table" else []
        relevant_text_doc_ids = (
            ex.relevant_text_doc_ids(
                ticker=ticker,
                fiscal_year=fiscal_year,
                form_type=form_type,
            )
            if mode == "text"
            else []
        )

        row = RetrievalEvalRow(
            id=ex_id,
            mode=mode,
            query=ex.query,
            ticker=ticker,
            fiscal_year=fiscal_year,
            form_type=form_type,
            relevant_table_indices=relevant_indices,
            relevant_text_doc_ids=relevant_text_doc_ids,
        )

        if mode == "unknown":
            row.retrieval_ok = False
            row.retrieval_error = "NO_GOLD_LABELS"
            rows.append(row)
            errors.append(
                {
                    "id": ex_id,
                    "stage": "dataset",
                    "error": "No gold labels found (expected relevant_tables or relevant_doc_ids).",
                }
            )
            if fail_fast:
                break
            continue

        retrieved_indices: List[Optional[int]] = []
        retrieved_doc_ids: List[str] = []
        contexts: List[str] = []

        if mode == "table":
            if not relevant_indices:
                row.retrieval_ok = False
                row.retrieval_error = "NO_GOLD_TABLE_LABELS"
                rows.append(row)
                errors.append(
                    {
                        "id": ex_id,
                        "stage": "dataset",
                        "error": "No relevant_tables labels found; skipping deterministic scoring.",
                    }
                )
                if fail_fast:
                    break
                continue

            t0 = time.perf_counter()
            retrieval = sec_retrieve_tables(
                queries=[ex.query],
                ticker=ticker,
                fiscal_year=fiscal_year,
                form_type=form_type,
                doc_types=per_row_doc_types,
                top_k=max_k,
                min_total_score=min_total_score,
            )
            elapsed_ms = int((time.perf_counter() - t0) * 1000)

            row.retrieval_ok = bool(retrieval.ok)
            row.retrieval_error = retrieval.error

            top_tables = getattr(retrieval, "top_tables", None) or getattr(retrieval, "results", None) or []
            for item in top_tables[:max_k]:
                payload = _extract_payload(item)
                doc_id = str(payload.get("doc_id") or "")
                table_index = payload.get("table_index")
                if table_index is None:
                    table_index = _parse_table_index_from_doc_id(doc_id)
                else:
                    try:
                        table_index = int(table_index)
                    except Exception:
                        table_index = _parse_table_index_from_doc_id(doc_id)

                retrieved_indices.append(table_index)
                retrieved_doc_ids.append(doc_id)

                context = _extract_context(payload)
                if context:
                    contexts.append(context)

            row.retrieved_table_indices = retrieved_indices
            row.retrieved_doc_ids = retrieved_doc_ids

            relevant_set = set(relevant_indices)
            relevant_flags = [idx is not None and idx in relevant_set for idx in retrieved_indices]

            metrics: Dict[str, float] = {}
            for k in ks:
                metrics[f"hit@{k}"] = hit_at_k(relevant_flags, k)
                metrics[f"recall@{k}"] = recall_at_k(retrieved_indices, relevant_set, k)
                metrics[f"mrr@{k}"] = mrr_at_k(relevant_flags, k)
                metrics[f"ndcg@{k}"] = ndcg_at_k(relevant_flags, k)
            row.metrics = metrics

            row.trace = {
                "timing_ms": {
                    "retrieve": elapsed_ms,
                    "tool_total": ((retrieval.trace or {}).get("timing_ms") or {}).get("total"),
                },
                "counts": ((retrieval.trace or {}).get("counts") or {}),
            }
        else:
            if not relevant_text_doc_ids:
                if not allow_empty_labels:
                    row.retrieval_ok = False
                    row.retrieval_error = "NO_GOLD_TEXT_LABELS"
                    rows.append(row)
                    errors.append(
                        {
                            "id": ex_id,
                            "stage": "dataset",
                            "error": "No relevant_doc_ids labels found; skipping deterministic scoring.",
                        }
                    )
                    if fail_fast:
                        break
                    continue
                row.retrieval_error = "NO_GOLD_TEXT_LABELS"

            t0 = time.perf_counter()
            rerank_query = ""
            fused: List[models.ScoredPoint] = []
            reranked: List[models.ScoredPoint] = []
            retrieval_error: Optional[str] = None
            try:
                if text_dense_only:
                    if text_client is None:
                        text_client = _build_text_dense_client()
                    reranked = dense_search_sec_docs(
                        ex.query,
                        client=text_client,
                        embed_fn=lambda query: embed_query_qwen3(
                            query,
                            api_url=text_embed_api_url,
                            model=text_embed_model,
                        ),
                        collection_name=collection,
                        using_dense="dense",
                        top_k=max_k,
                        doc_types=per_row_doc_types,
                        ticker=ticker,
                        fiscal_year=fiscal_year,
                        form_type=form_type,
                    )
                    rerank_query = str(ex.query)
                    fused = []
                else:
                    rerank_query, fused, reranked, _retrieval_timing = retrieve_scored_points(
                        queries=[ex.query],
                        ticker=ticker,
                        fiscal_year=fiscal_year,
                        form_type=form_type,
                        doc_types=per_row_doc_types,
                    )
            except Exception as exc:
                retrieval_error = str(exc)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)

            row.retrieval_ok = retrieval_error is None
            row.retrieval_error = retrieval_error

            for point in reranked[:max_k]:
                payload = _extract_payload_from_scored_point(point)
                doc_id = _normalize_text_doc_id(str(payload.get("doc_id") or ""))
                retrieved_doc_ids.append(doc_id)
                context = _extract_context(payload)
                if context:
                    contexts.append(context)

            row.retrieved_doc_ids = retrieved_doc_ids
            row.retrieved_table_indices = []

            metrics = {}
            if relevant_text_doc_ids:
                relevant_set = {str(x).strip() for x in relevant_text_doc_ids if str(x).strip()}
                relevant_flags = [str(doc_id).strip() in relevant_set for doc_id in retrieved_doc_ids]
                for k in ks:
                    metrics[f"hit@{k}"] = hit_at_k(relevant_flags, k)
                    metrics[f"recall@{k}"] = _recall_at_k_doc_ids(retrieved_doc_ids, relevant_text_doc_ids, k)
                    metrics[f"mrr@{k}"] = mrr_at_k(relevant_flags, k)
                    metrics[f"ndcg@{k}"] = ndcg_at_k(relevant_flags, k)
            row.metrics = metrics

            row.trace = {
                "timing_ms": {
                    "retrieve": elapsed_ms,
                },
                "counts": {
                    "fused_candidates": len(fused) if not text_dense_only else 0,
                    "reranked": len(reranked),
                    "scored": len(reranked[:max_k]),
                },
                "rerank_query": rerank_query,
            }

        rows.append(row)

        if not row.retrieval_ok and row.retrieval_error:
            errors.append(
                {
                    "id": ex_id,
                    "stage": "retrieval",
                    "error": row.retrieval_error,
                }
            )
            if fail_fast:
                break

        ragas_samples.append(
            {
                "id": ex_id,
                "question": ex.query,
                "contexts": contexts,
                "ground_truth": ex.gold_answer or "",
            }
        )

    valid_rows = [r for r in rows if r.metrics]
    metric_keys = sorted({mk for r in valid_rows for mk in r.metrics.keys()})
    deterministic_summary = {
        key: _mean([float(r.metrics.get(key, 0.0)) for r in valid_rows]) for key in metric_keys
    }

    ragas_summary: Dict[str, float] = {}
    if enable_ragas and ragas_samples:
        ragas_cfg = ragas_config or RagasRetrievalConfig()
        per_sample_ragas, ragas_summary, ragas_errors = evaluate_ragas_retrieval(
            ragas_samples,
            config=ragas_cfg,
        )
        for row in rows:
            if row.id in per_sample_ragas:
                row.ragas = per_sample_ragas[row.id]
        errors.extend(ragas_errors)

    total_ms = int((time.perf_counter() - total_start) * 1000)

    summary = RetrievalEvalSummary(
        num_queries=len(rows),
        num_valid_queries=len(valid_rows),
        num_failures=len(errors),
        deterministic=deterministic_summary,
        ragas=ragas_summary,
        config={
            "eval_path": str(eval_path),
            "top_k": max_k,
            "k_values": ks,
            "eval_mode": eval_mode,
            "default_ticker": default_ticker,
            "default_fiscal_year": int(default_fiscal_year),
            "default_form_type": default_form_type,
            "default_doc_types_table": table_doc_types,
            "default_doc_types_text": text_doc_types,
            "min_total_score": int(min_total_score),
            "enable_ragas": bool(enable_ragas),
            "text_dense_only": bool(text_dense_only),
            "text_embed_api_url": text_embed_api_url,
            "text_embed_model": text_embed_model,
            "timing_ms": {"total": total_ms},
        },
    )

    _write_jsonl(
        out_path / "per_query.jsonl",
        [r.model_dump(mode="json") for r in rows],
    )
    (out_path / "summary.json").write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_jsonl(out_path / "errors.jsonl", errors)

    return summary, rows, errors
