#!/usr/bin/env python3
"""Build an embedding manifest for all tested text chunking strategies.

Usage example:
    PYTHONPATH=src python scripts/evals/build_chunking_strategy_embedding_inputs.py \
        --tickers VZ PYPL UBER MCD AMT HON MS C \
        --years 2024 2025 \
        --out data/embedding_batches/chunking_strategy_embed_inputs.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ingestion.sec_embedder import build_text_content


def _load_benchmark_module():
    module_path = Path(__file__).with_name("benchmark_chunking_retrieval.py")
    spec = importlib.util.spec_from_file_location("benchmark_chunking_retrieval", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load benchmark module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


_bench = _load_benchmark_module()
StrategySpec = _bench.StrategySpec
build_strategy_chunks = _bench.build_strategy_chunks
_build_strategy_specs = _bench._build_strategy_specs


DEFAULT_TICKERS = ["VZ", "PYPL", "UBER", "MCD", "AMT", "HON", "MS", "C"]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build embedding JSONL manifest for strategy comparison runs.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help="Tickers to include (default: 8 strategy benchmarks set).",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2024, 2025],
        help="Filing years to include (default: 2024 2025).",
    )
    parser.add_argument(
        "--chunk-root",
        default="data/chunked",
        help="Root directory containing text chunks.",
    )
    parser.add_argument(
        "--out",
        default="data/embedding_batches/chunking_strategy_embed_inputs.jsonl",
        help="Output JSONL path for all selected strategy-doc rows.",
    )
    parser.add_argument(
        "--stages",
        default="1,2,3",
        help="Comma-separated chunking stages to include (default: 1,2,3).",
    )
    parser.add_argument(
        "--include-800-400",
        action="store_true",
        default=False,
        help="Include optional flat 800/400 reference strategy.",
    )
    parser.add_argument(
        "--encoding-model",
        default="text-embedding-3-large",
        help="Tokenizer model used for chunk length estimation.",
    )
    parser.add_argument(
        "--max-docs-per-file",
        type=int,
        default=0,
        help="Optional file sharding (0 means single combined file).",
    )
    parser.add_argument(
        "--manifest-out",
        default=None,
        help="Optional manifest JSON path (default: <out>.manifest.json).",
    )
    parser.add_argument(
        "--drop-empty",
        action="store_true",
        default=False,
        help="Deprecated for compatibility; currently no-op.",
    )
    return parser.parse_args(argv)


def _parse_stages(raw: str) -> List[int]:
    out: List[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out or [1, 2, 3]


def _iter_prefixes(tickers: Sequence[str], years: Sequence[int]) -> List[Tuple[str, int, str]]:
    return [(ticker.upper(), int(year), f"{ticker.upper()}_10-K_{int(year)}") for ticker in tickers for year in years]


def _find_source_chunk_path(chunk_root: Path, prefix: str) -> Path | None:
    base = chunk_root / prefix.split("_")[0] / "10-K"
    candidates = [
        base / f"{prefix}.text.split.jsonl",
        base / f"{prefix}.text.jsonl",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _safe_int(value: object, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return fallback


def _build_strategy_metadata(
    strategy: StrategySpec,
    prefix: str,
    chunk_path: Path,
    record: dict,
) -> Dict[str, object]:
    chunk_index = _safe_int(record.get("chunk_index"), _safe_int(record.get("source_chunk_index"), 0))
    split_count = _safe_int(record.get("split_count"), 1)
    split_index = _safe_int(record.get("split_index"), 0)

    if split_count > 1:
        source_chunk_id = f"{chunk_index}::split::{split_index}"
    else:
        source_chunk_id = str(chunk_index)

    doc_id = f"{prefix}::{strategy.name}::text::{source_chunk_id}"

    heading_path = record.get("heading_path")
    if isinstance(heading_path, list) and heading_path:
        section_path = " > ".join(str(p) for p in heading_path)
        section_title = record.get("section_title") or heading_path[-1]
    else:
        section_path = record.get("section_path")
        section_title = record.get("section_title")

    common_meta: Dict[str, object] = {
        "prefix": prefix,
        "ticker": prefix.split("_")[0],
        "form_type": "10-K",
        "fiscal_year": int(prefix.rsplit("_", 1)[1]),
        "doc_type": "text_chunk",
        "chunking_strategy": strategy.name,
        "chunking_stage": strategy.stage,
        "split_mode": strategy.split_mode,
        "max_tokens": strategy.max_tokens,
        "overlap_tokens": strategy.overlap_tokens,
        "overlap_paragraphs": strategy.overlap_paragraphs,
        "filter_levels": list(strategy.filter_levels or ()),
        "parent_expand": strategy.parent_expand,
        "chunk_index": chunk_index,
        "split_count": split_count,
        "split_index": split_index,
        "section_title": section_title,
        "section_path": section_path,
        "source": "text",
    }

    return {
        "id": doc_id,
        "content": build_text_content(record),
        "metadata": common_meta,
        "trace": {
            "doc_id": doc_id,
            "doc_type": "text_chunk",
            "source": {
                "prefix": prefix,
                "ticker": prefix.split("_")[0],
                "strategy": strategy.name,
                "strategy_stage": strategy.stage,
                "form_type": "10-K",
                "fiscal_year": int(prefix.rsplit("_", 1)[1]),
                "source_file": str(chunk_path),
                "group": "text",
            },
        },
    }


def _split_stages(
    include_ref_windows: bool,
    stages: Sequence[int],
) -> List[StrategySpec]:
    selected = [s for s in _build_strategy_specs(include_ref_windows=include_ref_windows) if s.stage in set(stages)]
    selected.sort(key=lambda s: (s.stage, s.name))
    return selected


def _chunk_records_to_docs(
    chunks: Sequence[dict],
    strategy: StrategySpec,
    prefix: str,
    chunk_path: Path,
) -> List[dict]:
    docs: List[dict] = []
    for rec in chunks:
        content = build_text_content(rec)
        if not content:
            continue
        doc = _build_strategy_metadata(strategy=strategy, prefix=prefix, chunk_path=chunk_path, record={**rec, "text": content})
        docs.append(doc)
    return docs

def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_batches(rows: Sequence[dict], out_root: Path, out_path: Path, max_docs_per_file: int) -> List[Path]:
    if max_docs_per_file <= 0:
        _write_jsonl(out_path, rows)
        return [out_path]

    out_root.mkdir(parents=True, exist_ok=True)
    total_batches = max(1, (len(rows) + max_docs_per_file - 1) // max_docs_per_file)
    out_files: List[Path] = []
    for idx in range(total_batches):
        start = idx * max_docs_per_file
        end = start + max_docs_per_file
        batch_rows = rows[start:end]
        batch_path = out_root / f"{out_path.stem}.batch_{idx + 1:03d}_of_{total_batches:03d}{out_path.suffix}"
        _write_jsonl(batch_path, batch_rows)
        out_files.append(batch_path)
    return out_files


def _write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    tickers = [t.upper() for t in args.tickers if str(t).strip()]
    years = [int(y) for y in args.years if int(y)]
    strategies = _split_stages(
        include_ref_windows=args.include_800_400,
        stages=_parse_stages(args.stages),
    )

    chunk_root = Path(args.chunk_root)
    if not chunk_root.exists():
        raise FileNotFoundError(f"chunk-root does not exist: {chunk_root}")

    all_rows: List[dict] = []
    strategy_counts: Counter[str] = Counter()

    for ticker, year, prefix in _iter_prefixes(tickers=tickers, years=years):
        source_path = _find_source_chunk_path(chunk_root=chunk_root, prefix=prefix)
        if source_path is None:
            print(f"[WARN] no source text chunk for {prefix}")
            continue

        for strategy in strategies:
            chunks = build_strategy_chunks(
                source_path=source_path,
                strategy=strategy,
                encoding_model=args.encoding_model,
            )
            docs = _chunk_records_to_docs(chunks=chunks, strategy=strategy, prefix=prefix, chunk_path=source_path)
            if not docs:
                print(f"[WARN] {prefix} strategy={strategy.name} produced no docs")
                continue

            strategy_counts[strategy.name] += len(docs)
            all_rows.extend(docs)

    if not all_rows:
        print("[ERROR] no docs generated")
        return 1

    out_path = Path(args.out)
    batch_paths = _write_batches(
        rows=all_rows,
        out_root=out_path.parent,
        out_path=out_path,
        max_docs_per_file=args.max_docs_per_file,
    )

    by_ticker = defaultdict(int)
    by_strategy = defaultdict(int)
    for row in all_rows:
        by_ticker[row.get("metadata", {}).get("ticker", "")] += 1
        by_strategy[row.get("metadata", {}).get("chunking_strategy", "")] += 1

    manifest = {
        "tickers": tickers,
        "years": years,
        "stages": _parse_stages(args.stages),
        "include_800_400": args.include_800_400,
        "encoding_model": args.encoding_model,
        "total_docs": len(all_rows),
        "by_strategy": dict(strategy_counts),
        "by_ticker": dict(by_ticker),
        "counts_by_doc_type": {"text_chunk": len(all_rows)},
        "outputs": {
            "combined": str(out_path),
            "batches": [str(p) for p in batch_paths],
        },
    }

    manifest_path = Path(args.manifest_out) if args.manifest_out else out_path.with_suffix(out_path.suffix + ".manifest.json")
    _write_manifest(manifest_path, manifest)

    print(f"[INFO] generated {len(all_rows)} embedding rows across {len(strategies)} strategies")
    print(f"[INFO] manifest: {manifest_path}")
    for p in batch_paths:
        print(f"[INFO] wrote {p}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
