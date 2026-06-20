#!/usr/bin/env python3
"""Benchmark text chunking strategies for retrieval-only experiments.

The script builds a small set of strategy-specific chunk variants, embeds them,
loads each variant into a dedicated Qdrant collection, and runs existing
`run_retrieval_eval` to capture deterministic retrieval metrics.

Current benchmark target:
- 8 SP100 tickers
- only text mode
- 3-stage split strategy as requested:
  1) Flat token-window size/overlap sweep
  2) Structure-aware (heading-aware) chunking
  3) Parent-child variants
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from qdrant_client import QdrantClient, models

from evals.retrieval_eval_runner import run_retrieval_eval
from ingestion.chunk_splitter import get_encoding, split_long_chunks
from ingestion.qdrant_ingester import count_points, docs_to_points, ensure_collection
from ingestion.sec_embedder import build_text_docs, embed_docs


try:
    from FlagEmbedding import BGEM3FlagModel
except Exception:
    BGEM3FlagModel = None


DEFAULT_TICKERS = [
    "VZ",
    "PYPL",
    "UBER",
    "MCD",
    "AMT",
    "HON",
    "MS",
    "C",
]

PRECOMPUTED_CHUNKING_INPUT = "data/embedding_batches/chunking_strategy_embed_inputs.jsonl"
PRECOMPUTED_ROW_MAP = "data/embedding_batches/row_map.parquet"
PRECOMPUTED_UNIQUE_INDEX = "data/embedding_batches/unique_index.parquet"
PRECOMPUTED_EMBEDDINGS = "data/embedding_batches/embeddings_unique.f16.npy"


@dataclass(frozen=True)
class StrategySpec:
    name: str
    stage: int
    max_tokens: int
    split_mode: str
    overlap_tokens: int = 0
    overlap_paragraphs: int = 0
    filter_levels: Tuple[str, ...] | None = None
    parent_expand: bool = False


def _load_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _safe_int(value: object, fallback: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return fallback


def _safe_str(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text


def _load_precomputed_rows(
    path: Path,
    tickers: Sequence[str],
    years: Sequence[int],
    strategies: Set[str],
) -> Dict[str, List[dict]]:
    ticker_set = {str(t).upper() for t in tickers}
    year_set = {_safe_int(y, None) for y in years}
    by_strategy: Dict[str, List[dict]] = {s: [] for s in strategies}

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            metadata = row.get("metadata", {}) if isinstance(row.get("metadata", {}), dict) else {}
            if _safe_str(metadata.get("ticker")).upper() not in ticker_set:
                continue
            fy = _safe_int(metadata.get("fiscal_year"), None)
            if fy is None or fy not in year_set:
                continue
            strategy = _safe_str(metadata.get("chunking_strategy"))
            if strategy in by_strategy:
                by_strategy[strategy].append(row)

    # Keep memory stable and deterministic.
    for rows in by_strategy.values():
        rows.sort(key=lambda r: _safe_str(r.get("id")))

    return by_strategy


def _load_precomputed_mapping(
    row_map_path: Path,
    embeddings_path: Path,
    unique_index_path: Path,
    allowed_strategies: Set[str],
) -> tuple[Dict[str, dict], "list"["list"[float]]]:
    import numpy as np
    import pandas as pd

    row_map = pd.read_parquet(row_map_path)
    required_columns = {
        "id",
        "unique_row",
        "ticker",
        "form_type",
        "fiscal_year",
        "chunking_strategy",
    }
    missing = required_columns - set(row_map.columns)
    if missing:
        raise RuntimeError(f"row_map missing required columns: {sorted(missing)}")

    filtered = row_map[row_map["chunking_strategy"].isin(allowed_strategies)].copy()
    if filtered.empty:
        raise RuntimeError("No matching rows found in row_map for selected strategies.")

    unique_index = pd.read_parquet(unique_index_path)
    embeddings = np.load(str(embeddings_path))
    if len(unique_index) != len(embeddings):
        raise RuntimeError(
            f"unique_index and embedding row counts mismatch: {len(unique_index)} vs {len(embeddings)}"
        )

    mapping = {
        row.id: {
            "unique_row": int(row.unique_row),
            "ticker": str(row.ticker).upper(),
            "fiscal_year": int(row.fiscal_year),
            "chunking_strategy": str(row.chunking_strategy),
        }
        for row in filtered.itertuples(index=False)
    }

    return mapping, embeddings


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _iter_prefixes(tickers: Sequence[str], years: Sequence[int]) -> List[Tuple[str, int, str]]:
    return [(t.upper(), int(year), f"{t.upper()}_10-K_{int(year)}") for t in tickers for year in years]


def _match_eval_record(
    rec: dict,
    tickers: Set[str],
    years: Set[int],
) -> bool:
    legacy_tickers = set(str(x).upper() for x in (rec.get("tickers") or []))
    direct_ticker = str(rec.get("ticker") or "").upper().strip()
    q_tickers = set(legacy_tickers)
    if direct_ticker:
        q_tickers.add(direct_ticker)

    if q_tickers and tickers.intersection(q_tickers):
        if years:
            q_years = {
                int(y)
                for y in rec.get("years", [])
                if isinstance(y, int) or (isinstance(y, str) and str(y).strip().isdigit())
            }
            if not q_years:
                q_years = {
                    int(y)
                    for y in rec.get("target_years", [])
                    if isinstance(y, int) or (isinstance(y, str) and str(y).strip().isdigit())
                }
            if q_years and years.isdisjoint(q_years):
                return False
        return True

    filing_keys = rec.get("filing_keys") or []
    for fk in filing_keys:
        if not isinstance(fk, str):
            continue
        parts = fk.split("_", 1)
        if not parts:
            continue
        if parts[0].upper() in tickers:
            return True

    return False


def _build_strategy_specs(include_ref_windows: bool) -> List[StrategySpec]:
    stage1 = [
        StrategySpec("flat_512_no_overlap", 1, max_tokens=512, split_mode="token", overlap_tokens=0),
        StrategySpec("flat_512_overlap128", 1, max_tokens=512, split_mode="token", overlap_tokens=128),
        StrategySpec("flat_800_no_overlap", 1, max_tokens=800, split_mode="token", overlap_tokens=0),
        StrategySpec("flat_800_overlap200", 1, max_tokens=800, split_mode="token", overlap_tokens=200),
    ]
    if include_ref_windows:
        stage1.append(StrategySpec("flat_800_overlap400", 1, max_tokens=800, split_mode="token", overlap_tokens=400))

    stage2 = [
        StrategySpec("structure_aware_500_para", 2, max_tokens=500, split_mode="paragraph", overlap_paragraphs=1),
    ]

    stage3 = [
        StrategySpec("parent_child_500_para_children", 3, max_tokens=500, split_mode="paragraph", overlap_paragraphs=1, filter_levels=("subsection",)),
        StrategySpec(
            "parent_child_500_para_with_parent",
            3,
            max_tokens=500,
            split_mode="paragraph",
            overlap_paragraphs=1,
            filter_levels=("subsection",),
            parent_expand=True,
        ),
    ]

    return stage1 + stage2 + stage3


def _find_source_chunk_path(chunk_root: Path, ticker: str, prefix: str) -> Optional[Path]:
    ticker_u = ticker.upper()
    base = chunk_root / ticker_u / "10-K"
    candidates = [
        base / f"{prefix}.text.jsonl",
        base / f"{prefix}.text.split.jsonl",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _resolve_parent_text_by_item_id(records: Sequence[dict]) -> Dict[str, str]:
    parents: Dict[str, str] = {}
    for rec in records:
        if str(rec.get("level") or "").lower() != "item":
            continue
        item_id = str(rec.get("item_id") or "").strip()
        if not item_id:
            continue
        text = (rec.get("text") or "").strip()
        if text:
            parents[item_id] = text
    return parents


def _apply_parent_expansion(records: Sequence[dict]) -> List[dict]:
    parents = _resolve_parent_text_by_item_id(records)
    out: List[dict] = []
    for rec in records:
        if str(rec.get("level") or "").lower() != "subsection":
            out.append(rec)
            continue

        out_rec = dict(rec)
        item_id = str(rec.get("item_id") or "").strip()
        parent_text = parents.get(item_id)
        if parent_text:
            text = str(out_rec.get("text") or "").strip()
            out_rec["text"] = f"{parent_text}\n\n{text}".strip()
            out_rec["parent_expanded"] = True
        out.append(out_rec)
    return out


def _filter_levels(records: Sequence[dict], levels: Tuple[str, ...] | None) -> List[dict]:
    if not levels:
        return [dict(r) for r in records]

    wanted = {str(x).lower() for x in levels}
    return [dict(r) for r in records if str(r.get("level") or "").lower() in wanted]


def build_strategy_chunks(
    *,
    source_path: Path,
    strategy: StrategySpec,
    encoding_model: str,
) -> List[dict]:
    raw_records = [dict(r) for r in _load_jsonl(source_path)]
    work_records = raw_records
    if strategy.parent_expand:
        work_records = _apply_parent_expansion(work_records)

    work_records = _filter_levels(work_records, strategy.filter_levels)

    if not work_records:
        return []

    encoding = get_encoding(encoding_model)
    if strategy.max_tokens <= 0:
        return work_records

    return split_long_chunks(
        encoding=encoding,
        chunks=work_records,
        max_tokens=strategy.max_tokens,
        overlap_paragraphs=strategy.overlap_paragraphs,
        split_mode=strategy.split_mode,
        overlap_tokens=strategy.overlap_tokens,
    )


def ensure_collection_for_strategy(
    *,
    client: QdrantClient,
    collection_name: str,
    dense_dim: int,
    bge_dim: int,
    dense_only: bool,
    recreate: bool,
) -> None:
    if dense_only:
        vectors_config = {
            "dense": models.VectorParams(size=dense_dim, distance=models.Distance.COSINE),
        }
        sparse_vectors_config = None
    else:
        vectors_config = {
            "dense": models.VectorParams(size=dense_dim, distance=models.Distance.COSINE),
            "bge_m3_dense": models.VectorParams(size=bge_dim, distance=models.Distance.COSINE),
        }
        sparse_vectors_config = {
            "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF),
            "bge_m3_sparse": models.SparseVectorParams(),
        }

    if client.collection_exists(collection_name=collection_name):
        if recreate:
            client.delete_collection(collection_name=collection_name)
        else:
            return
    ensure_collection(
        client=client,
        collection_name=collection_name,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config,
    )


def _build_eval_subset(
    eval_path: Path,
    out_path: Path,
    tickers: Sequence[str],
    years: Sequence[int],
) -> List[dict]:
    out: List[dict] = []
    selected_tickers = {t.upper() for t in tickers}
    selected_years = {int(y) for y in years}

    with eval_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if _match_eval_record(rec, selected_tickers, selected_years):
                out.append(rec)

    _write_jsonl(out_path, out)
    return out


def run_strategy(
    *,
    tickers: Sequence[str],
    years: Sequence[int],
    strategy: StrategySpec,
    chunk_root: Path,
    work_root: Path,
    embedding_api_url: str,
    embedding_model: str,
    embedding_batch_size: int,
    embedding_batch_timeout: float,
    embed_only: bool,
    qdrant_host: str,
    qdrant_port: int,
    qdrant_collection_prefix: str,
    qdrant_recreate_collection: bool,
    text_dense_only: bool,
    use_precomputed_embeddings: bool,
    precomputed_rows: Dict[str, List[dict]],
    precomputed_row_map: Optional[Dict[str, dict]],
    precomputed_embeddings: Optional[object],
    allow_empty_labels: bool,
    default_ticker: str,
    default_fiscal_year: int,
    eval_path: Path,
    eval_k_values: List[int],
    eval_top_k: int,
    enable_ragas: bool,
    encoding_model: str,
    bge_m3_model_name: str,
    bge_m3_local_files_only: bool,
    bge_m3_dim: int,
    bge_m3_fp16: bool,
) -> Dict[str, object]:
    strategy_root = work_root / strategy.name
    chunked_out = strategy_root / "chunked"
    embedding_out = strategy_root / "embedding"
    eval_out = strategy_root / "eval"

    strategy_start = time.perf_counter()
    strategy_results: Dict[str, object] = {
        "strategy": strategy.name,
        "stage": strategy.stage,
        "chunk_root": str(chunked_out),
        "embedding_root": str(embedding_out),
        "collection": "",
        "num_text_docs": 0,
        "eval": {},
        "errors": [],
        "counts": {},
        "timing_ms": {},
    }

    split_ms = 0
    embed_ms = 0
    ingest_ms = 0
    eval_ms = 0
    precomputed_ms = 0

    prepared_prefixes = _iter_prefixes(tickers, years)
    encoded_count = 0

    docs_for_collection: List[dict] = []
    if use_precomputed_embeddings:
        pre_start = time.perf_counter()
        if precomputed_row_map is None or precomputed_embeddings is None:
            raise RuntimeError("Precomputed embeddings requested but metadata/embeddings are missing.")

        rows = precomputed_rows.get(strategy.name, [])
        if not rows:
            strategy_results["errors"].append(f"no_precomputed_rows:{strategy.name}")
        for row in rows:
            row_id = _safe_str(row.get("id"))
            if not row_id:
                continue
            mapping = precomputed_row_map.get(row_id)
            if mapping is None:
                continue
            unique_row = mapping.get("unique_row")
            if unique_row is None:
                continue

            doc = {
                "id": row_id,
                "content": row.get("content", ""),
                "metadata": row.get("metadata", {}),
                "embedding": precomputed_embeddings[int(unique_row)].tolist(),
            }
            docs_for_collection.append(doc)
            encoded_count += 1

        if docs_for_collection:
            strategy_results["counts"]["precomputed_rows"] = len(docs_for_collection)
        else:
            strategy_results["errors"].append("no_precomputed_docs")
        precomputed_ms = int((time.perf_counter() - pre_start) * 1000)
    else:
        split_start = time.perf_counter()
        # 1) Build strategy-specific text splits.
        for ticker, year, prefix in prepared_prefixes:
            source_path = _find_source_chunk_path(chunk_root, ticker, prefix)
            if source_path is None:
                print(f"[warn] No source chunk file for {prefix} under {chunk_root}")
                continue

            chunks = build_strategy_chunks(
                source_path=source_path,
                strategy=strategy,
                encoding_model=encoding_model,
            )
            if not chunks:
                print(f"[warn] No chunks for {prefix} strategy={strategy.name}")
                continue

            out_path = chunked_out / ticker.upper() / "10-K" / f"{prefix}.text.split.jsonl"
            _write_jsonl(out_path, chunks)
            encoded_count += len(chunks)
        split_ms = int((time.perf_counter() - split_start) * 1000)

        strategy_results["counts"]["chunks_prepared"] = encoded_count

        embed_start = time.perf_counter()
        # 2) Embed strategy-specific text chunks.
        for ticker, year, prefix in prepared_prefixes:
            chunk_file = chunked_out / ticker.upper() / "10-K" / f"{prefix}.text.split.jsonl"
            if not chunk_file.exists():
                continue

            common_meta = {
                "ticker": ticker.upper(),
                "company_name": None,
                "form_type": "10-K",
                "fiscal_year": year,
                "prefix": prefix,
            }
            text_docs = build_text_docs(text_path=chunk_file, common_meta=common_meta)

            if not text_docs:
                print(f"[warn] No text docs from {chunk_file}")
                continue

            embedded_docs = embed_docs(
                text_docs,
                api_url=embedding_api_url,
                model=embedding_model,
                batch_size=embedding_batch_size,
                timeout=embedding_batch_timeout,
            )

            if not embedded_docs:
                continue

            out_file = embedding_out / ticker.upper() / "10-K" / f"{prefix}.text.embedded.jsonl"
            _write_jsonl(out_file, embedded_docs)

            if not use_precomputed_embeddings:
                docs_for_collection.extend(embedded_docs)
        embed_ms = int((time.perf_counter() - embed_start) * 1000)

    # 3) Ingest into Qdrant.
    collection_name = f"{qdrant_collection_prefix}_{strategy.name}"
    strategy_results["collection"] = collection_name
    if not embed_only:
        ingest_start = time.perf_counter()
        bge_model = None
        if not text_dense_only:
            if BGEM3FlagModel is None:
                raise RuntimeError(
                    "FlagEmbedding is required for benchmark ingest in this script. "
                    "Install it or use --dense-only/--embed-only."
                )
            bge_model = BGEM3FlagModel(
                bge_m3_model_name,
                use_fp16=bge_m3_fp16,
                local_files_only=bge_m3_local_files_only,
            )

        client = QdrantClient(host=qdrant_host, port=qdrant_port)

        if use_precomputed_embeddings:
            first_emb = docs_for_collection[0] if docs_for_collection else None
        else:
            first_emb = None
            for ticker, _, prefix in prepared_prefixes:
                emb_file = embedding_out / ticker.upper() / "10-K" / f"{prefix}.text.embedded.jsonl"
                if emb_file.exists():
                    with emb_file.open("r", encoding="utf-8") as handle:
                        for line in handle:
                            line = line.strip()
                            if not line:
                                continue
                            first_emb = json.loads(line)
                            break
                if first_emb is not None:
                    break

            if not docs_for_collection:
                # Fallback for legacy behavior: reload all written embedded files.
                for ticker, _, prefix in prepared_prefixes:
                    emb_file = embedding_out / ticker.upper() / "10-K" / f"{prefix}.text.embedded.jsonl"
                    if not emb_file.exists():
                        continue
                    with emb_file.open("r", encoding="utf-8") as handle:
                        for line in handle:
                            line = line.strip()
                            if not line:
                                continue
                            docs_for_collection.append(json.loads(line))

        if not docs_for_collection:
            strategy_results["errors"].append("no_embedded_docs")
        if first_emb is None:
            if docs_for_collection:
                first_emb = docs_for_collection[0]

        if first_emb is None:
            strategy_results["errors"].append("no_embedded_docs")
        else:
            dense_dim = len(first_emb.get("embedding", []))
            if not isinstance(dense_dim, int) or dense_dim <= 0:
                raise RuntimeError(f"Invalid dense embedding dimension in {first_emb}")

            ensure_collection_for_strategy(
                client=client,
                collection_name=collection_name,
                dense_dim=dense_dim,
                bge_dim=bge_m3_dim,
                dense_only=text_dense_only,
                recreate=qdrant_recreate_collection,
            )

            if not docs_for_collection:
                for ticker, year, prefix in prepared_prefixes:
                    emb_file = embedding_out / ticker.upper() / "10-K" / f"{prefix}.text.embedded.jsonl"
                    if not emb_file.exists():
                        continue
                    docs_for_collection.extend(_load_jsonl(emb_file))

            strategy_results["counts"]["emb_text_docs"] = len(docs_for_collection)
            if not docs_for_collection:
                strategy_results["errors"].append("no_docs_to_ingest")

            for start in range(0, len(docs_for_collection), 32):
                batch = docs_for_collection[start : start + 32]
                points = docs_to_points(
                    batch,
                    bge_m3_encode=not text_dense_only,
                    bge_m3_model=bge_model,
                    include_bm25=not text_dense_only,
                    bm25_vector_name="bm25",
                    bge_m3_dense_vector_name="bge_m3_dense",
                    bge_m3_sparse_vector_name="bge_m3_sparse",
                )
                client.upsert(collection_name=collection_name, points=points)

                strategy_results["counts"]["qdrant_count"] = count_points(client, collection_name)
                os.environ["QDRANT_COLLECTION_NAME"] = collection_name
        ingest_ms = int((time.perf_counter() - ingest_start) * 1000)

    # 4) Run deterministic eval.
    if embed_only:
        strategy_results["timing_ms"] = {
            "precomputed_ms": precomputed_ms,
            "split_ms": split_ms,
            "embed_ms": embed_ms,
            "ingest_ms": ingest_ms,
            "eval_ms": eval_ms,
            "strategy_total_ms": int((time.perf_counter() - strategy_start) * 1000),
        }
        return strategy_results

    # Create query subset file for the selected tickers.
    filtered_eval_path = eval_out / "eval_subset.jsonl"
    query_records = _build_eval_subset(
        eval_path=eval_path,
        out_path=eval_out / "eval_subset.jsonl",
        tickers=tickers,
        years=years,
    )

    if not query_records:
        raise RuntimeError("No evaluation rows matched the selected tickers.")
    # Ensure retrieval process reads the selected data from this benchmark run.
    os.environ["QDRANT_COLLECTION_NAME"] = collection_name
    os.environ["QDRANT_HOST"] = qdrant_host
    os.environ["QDRANT_PORT"] = str(qdrant_port)
    eval_start = time.perf_counter()
    summary, _rows, errors = run_retrieval_eval(
        eval_path=str(filtered_eval_path),
        out_dir=str(eval_out),
        eval_mode="text",
        top_k=eval_top_k,
        k_values=eval_k_values,
        default_ticker=default_ticker,
        default_fiscal_year=default_fiscal_year,
        default_form_type="10-K",
        default_doc_types=["text_chunk"],
        min_total_score=0,
        enable_ragas=enable_ragas,
        text_dense_only=bool(text_dense_only),
        text_embed_api_url=embedding_api_url,
        text_embed_model=embedding_model,
        allow_empty_labels=allow_empty_labels,
        fail_fast=False,
    )
    eval_ms = int((time.perf_counter() - eval_start) * 1000)

    strategy_results["eval"] = summary.model_dump(mode="json")
    strategy_results["errors"].extend(errors)
    strategy_results["counts"]["queries"] = len(query_records)
    strategy_results["counts"]["strategy_points"] = summary.num_queries

    strategy_results["timing_ms"] = {
        "precomputed_ms": precomputed_ms,
        "split_ms": split_ms,
        "embed_ms": embed_ms,
        "ingest_ms": ingest_ms,
        "eval_ms": eval_ms,
        "strategy_total_ms": int((time.perf_counter() - strategy_start) * 1000),
    }

    return strategy_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run chunking-strategy retrieval benchmark for selected 10-K filings.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS, help="Tickers to include.")
    parser.add_argument("--years", nargs="+", type=int, default=[2024, 2025], help="Fiscal years to include.")
    parser.add_argument("--eval-path", default="data/evals/retrieval/sp100_10k_eval_queries_full.jsonl", help="Evaluation dataset JSONL.")
    parser.add_argument("--chunk-root", default="data/chunked", help="Source chunk root.")
    parser.add_argument("--work-root", default="artifacts/evals/chunking_bench", help="Workspace for generated chunks/embeddings/eval artifacts.")
    parser.add_argument("--encoding-model", default="text-embedding-3-large", help="Tokenizer model for token splits.")
    parser.add_argument("--embedding-api-url", default="http://localhost:11434/api/embed", help="Embedding API URL.")
    parser.add_argument("--embedding-model", default="qwen3-embedding:8b", help="Embedding model for text chunks.")
    parser.add_argument("--embedding-batch", type=int, default=16, help="Embedding batch size.")
    parser.add_argument("--embedding-timeout", type=float, default=180.0, help="Embedding API timeout seconds.")
    parser.add_argument("--collection-prefix", default="bench_text_chunks", help="Qdrant collection name prefix.")
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant host.")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port.")
    parser.add_argument("--qdrant-recreate", action="store_true", default=False, help="Recreate strategy collection.")
    parser.add_argument("--default-ticker", default="AAPL", help="Fallback ticker when row misses metadata.")
    parser.add_argument("--default-fiscal-year", type=int, default=2024, help="Fallback fiscal year when row misses metadata.")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K for deterministic eval.")
    parser.add_argument("--k-values", default="1,3,5,10", help="Comma-separated k values.")
    parser.add_argument(
        "--use-precomputed-embeddings",
        action="store_true",
        default=False,
        help="Use precomputed strategy chunking + embedding artifacts instead of recomputing",
    )
    parser.add_argument("--precomputed-chunking-input", default=PRECOMPUTED_CHUNKING_INPUT)
    parser.add_argument("--precomputed-row-map", default=PRECOMPUTED_ROW_MAP)
    parser.add_argument("--precomputed-unique-index", default=PRECOMPUTED_UNIQUE_INDEX)
    parser.add_argument("--precomputed-embeddings", default=PRECOMPUTED_EMBEDDINGS)
    parser.add_argument("--enable-ragas", action="store_true", default=False, help="Enable ragas in eval (slower).")
    parser.add_argument("--include-800-400", action="store_true", default=False, help="Include 800/400 optional reference arm.")
    parser.add_argument("--embed-only", action="store_true", default=False, help="Prepare chunks+embeddings only; skip evaluation ingest.")
    parser.add_argument("--bge-m3-dim", type=int, default=1024, help="Expected BGE-M3 dense dimension.")
    parser.add_argument("--bge-m3-fp16", action="store_true", default=False, help="Use fp16 in BGE-M3.")
    parser.add_argument("--bge-m3-allow-download", action="store_true", default=False, help="Allow BGE-M3 model download from HuggingFace.")
    parser.add_argument("--dense-only", action="store_true", default=False, help="Use Qwen3 dense-only retrieval in text-mode eval (no BGE/BM25).")
    parser.add_argument(
        "--allow-empty-labels",
        action="store_true",
        default=False,
        help="Run eval even when relevant_doc_ids/text labels are missing in the dataset.",
    )
    parser.add_argument("--stages", default="1,2,3", help="Comma-separated stages to run (example: 1,2,3).")
    return parser.parse_args()


def _parse_k_values(raw: str) -> List[int]:
    out: List[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out or [1, 3, 5, 10]


def _parse_stages(raw: str) -> Set[int]:
    out: Set[int] = set()
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out or {1, 2, 3}


def main() -> int:
    args = parse_args()

    tickers = [str(t).upper() for t in args.tickers if str(t).strip()]
    if not tickers:
        raise ValueError("At least one --ticker is required.")
    years = [int(y) for y in args.years]
    if not years:
        raise ValueError("At least one --years value is required.")
    eval_path = Path(args.eval_path)
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_path}")

    selected_stages = _parse_stages(args.stages)
    strategies = [
        s for s in _build_strategy_specs(include_ref_windows=args.include_800_400) if s.stage in selected_stages
    ]
    if not strategies:
        raise ValueError(f"No strategies selected for stages={args.stages}")

    if args.use_precomputed_embeddings:
        precomputed_rows = _load_precomputed_rows(
            path=Path(args.precomputed_chunking_input),
            tickers=tickers,
            years=years,
            strategies={s.name for s in strategies},
        )
        precomputed_row_map, precomputed_embeddings = _load_precomputed_mapping(
            row_map_path=Path(args.precomputed_row_map),
            embeddings_path=Path(args.precomputed_embeddings),
            unique_index_path=Path(args.precomputed_unique_index),
            allowed_strategies={s.name for s in strategies},
        )
    else:
        precomputed_rows = {}
        precomputed_row_map = None
        precomputed_embeddings = None

    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    k_values = _parse_k_values(args.k_values)

    results: List[dict] = []
    progress_path = work_root / "strategy_progress.jsonl"
    for idx, strategy in enumerate(strategies, start=1):
        print(f"\n[{idx}/{len(strategies)}] Running strategy={strategy.name} stage={strategy.stage}")
        strategy_result = run_strategy(
            tickers=tickers,
            years=years,
            strategy=strategy,
            chunk_root=Path(args.chunk_root),
            work_root=work_root,
            embedding_api_url=args.embedding_api_url,
            embedding_model=args.embedding_model,
            embedding_batch_size=args.embedding_batch,
            embedding_batch_timeout=args.embedding_timeout,
            embed_only=args.embed_only,
            qdrant_host=args.qdrant_host,
            qdrant_port=args.qdrant_port,
            qdrant_collection_prefix=args.collection_prefix,
            qdrant_recreate_collection=args.qdrant_recreate,
            text_dense_only=bool(args.dense_only),
            use_precomputed_embeddings=bool(args.use_precomputed_embeddings),
            precomputed_rows=precomputed_rows,
            precomputed_row_map=precomputed_row_map,
            precomputed_embeddings=precomputed_embeddings,
            default_ticker=args.default_ticker,
            default_fiscal_year=args.default_fiscal_year,
            eval_path=eval_path,
            eval_k_values=k_values,
            eval_top_k=args.top_k,
            allow_empty_labels=bool(args.allow_empty_labels),
            enable_ragas=bool(args.enable_ragas),
            encoding_model=args.encoding_model,
            bge_m3_model_name="BAAI/bge-m3",
            bge_m3_local_files_only=not args.bge_m3_allow_download,
            bge_m3_dim=args.bge_m3_dim,
            bge_m3_fp16=args.bge_m3_fp16,
        )
        results.append(strategy_result)
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"strategy": strategy.name, "result": strategy_result}, ensure_ascii=False) + "\n")
        print(
            f"[complete] strategy={strategy.name} "
            f"errors={len(strategy_result.get('errors', []))} "
            f"queries={strategy_result.get('counts', {}).get('queries')} "
            f"retrieval_ok={strategy_result.get('counts', {}).get('retrieval_ok', 'n/a')}"
        )

    out_path = work_root / "benchmark_results.json"
    out_path.write_text(
        json.dumps(
            {
                "strategies": results,
                "input": {
                    "tickers": tickers,
                    "years": years,
                    "eval_path": str(eval_path),
                    "stages": sorted(_parse_stages(args.stages)),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved benchmark results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
