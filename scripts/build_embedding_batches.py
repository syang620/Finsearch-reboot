#!/usr/bin/env python
"""Build traceable embedding batches from pre-generated text chunks and table summaries."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from ingestion.chunk_paths import parse_filing_prefix, resolve_chunk_file
from ingestion.sec_embedder import build_table_and_row_docs, build_text_docs, embed_batch_with_qwen3
from _common import load_tickers, parse_quarter_spec


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect all embeddable text/table content into traceable batches and "
            "optionally run embedding on each batch."
        ),
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Tickers to process explicitly (e.g., AAPL AMZN).",
    )
    parser.add_argument(
        "--from-file",
        help="Path to file containing one ticker per line.",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Discover tickers from nested chunk/summary folders and process all available filings.",
    )
    parser.add_argument(
        "--forms",
        nargs="+",
        default=["10-K", "10-Q"],
        help="Form types to process (default: 10-K 10-Q).",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        help="10-K fiscal years to process when forms include 10-K.",
    )
    parser.add_argument(
        "--quarters",
        nargs="+",
        help="10-Q quarter specs to process, e.g., 2025_Q1 2025_Q2.",
    )
    parser.add_argument(
        "--chunks-dir",
        default="data/chunked",
        help="Directory containing text chunk files.",
    )
    parser.add_argument(
        "--table-summaries-dir",
        default="data/table_summaries",
        help="Directory containing table summary files.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/embedding_batches",
        help="Directory to write combined and batched embedding payloads.",
    )
    parser.add_argument(
        "--out-prefix",
        default="all_embedding_inputs",
        help="Output filename prefix (without extension).",
    )
    parser.add_argument(
        "--max-docs-per-batch",
        type=int,
        default=1000,
        help="Maximum docs per batch JSONL file (default: 1000).",
    )
    parser.add_argument(
        "--file-types",
        nargs="+",
        choices=["text", "tables", "rows"],
        default=["text", "tables", "rows"],
        help="Which doc types to include.",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Call embedding API on collected batches.",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:11434/api/embed",
        help="Embedding API URL.",
    )
    parser.add_argument(
        "--model",
        default="qwen3-embedding:8b",
        help="Embedding model name.",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=16,
        help="Batch size for API embedding requests (default: 16).",
    )
    parser.add_argument(
        "--embed-timeout",
        type=float,
        default=180.0,
        help="Timeout seconds for embedding API request.",
    )
    return parser.parse_args(argv)


def _collect_explicit_prefixes(
    tickers: Sequence[str],
    forms: Sequence[str],
    years: Sequence[int] | None,
    quarter_specs: Sequence[Tuple[int, str]] | None,
) -> List[str]:
    forms_u = [str(form).upper() for form in forms]
    prefixes: List[str] = []
    for ticker in sorted(set(t.upper() for t in tickers)):
        for form in forms_u:
            if form == "10-K":
                if not years:
                    continue
                for year in years:
                    prefixes.append(f"{ticker}_10-K_{year}")
            elif form == "10-Q":
                if not quarter_specs:
                    continue
                for year, q_label in quarter_specs:
                    prefixes.append(f"{ticker}_10-Q_{year}{q_label}")
    return prefixes


def _discover_tickers(chunks_dir: Path, table_dir: Path) -> List[str]:
    tickers = set()
    for root in (chunks_dir, table_dir):
        if not root.exists():
            continue
        for item in root.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                tickers.add(item.name)
    return sorted(tickers)


def _candidate_prefix_from_filename(
    file_path: Path,
    ticker: str,
    form: str,
) -> str | None:
    stem = file_path.name
    if stem.startswith(f"{ticker}_"):
        base = stem.split(".", 1)[0]
        parsed = parse_filing_prefix(base)
        if parsed is not None:
            return base

    if stem.startswith(f"{form}_"):
        base = stem.split(".", 1)[0]
        return f"{ticker}_{base}"

    # Fallback for odd legacy formats: prefer filenames already shaped as
    # <ticker>_<form>_<rest>...
    for part in stem.split("."):
        if part.startswith(f"{ticker}_{form}_"):
            maybe = part
            parsed = parse_filing_prefix(maybe)
            return maybe if parsed is not None else None
    return None


def _discover_prefixes(
    chunks_dir: Path,
    table_dir: Path,
    tickers: Sequence[str],
    forms: Sequence[str],
) -> List[str]:
    prefixes = set()
    forms_u = {str(form).upper() for form in forms}

    for ticker in tickers:
        ticker = ticker.upper()
        for form in forms_u:
            # Nested layout: data/<scope>/<TICKER>/<FORM>/<...>
            form_chunk_dir = chunks_dir / ticker / form
            if form_chunk_dir.is_dir():
                for file_path in sorted(
                    list(form_chunk_dir.glob("*.text.split.jsonl"))
                    + list(form_chunk_dir.glob("*.text.jsonl"))
                ):
                    prefix = _candidate_prefix_from_filename(file_path, ticker, form)
                    if prefix:
                        prefixes.add(prefix)

            for form_dir in [table_dir / ticker / form, table_dir / ticker]:
                if form_dir.is_dir():
                    for file_path in sorted(form_dir.glob("*.tables.summaries.jsonl")):
                        prefix = _candidate_prefix_from_filename(file_path, ticker, form)
                        if prefix:
                            prefixes.add(prefix)

            # Legacy flat variants: data/<scope>/<TICKER>10-K_2024... or data/<scope>/<TICKER>/10-K_2024...
            form_dir_candidates = [table_dir / ticker, chunks_dir / ticker, table_dir, chunks_dir]
            for base_dir in form_dir_candidates:
                if not base_dir.exists():
                    continue
                for suffix in [f"{form}_*.text.split.jsonl", f"{form}_*.text.jsonl", f"{form}_*.tables.summaries.jsonl"]:
                    for file_path in sorted(base_dir.glob(suffix)):
                        prefix = _candidate_prefix_from_filename(file_path, ticker, form)
                        if prefix:
                            prefixes.add(prefix)

    return sorted(prefixes)


def _find_text_chunk_path(chunks_dir: Path, prefix: str) -> Path | None:
    split_name = f"{prefix}.text.split.jsonl"
    resolved = resolve_chunk_file(chunks_dir, prefix, split_name)
    if resolved is not None:
        return resolved

    plain_name = f"{prefix}.text.jsonl"
    return resolve_chunk_file(chunks_dir, prefix, plain_name)


def _find_table_summary_path(table_summaries_dir: Path, prefix: str) -> Path | None:
    parsed = parse_filing_prefix(prefix)
    if parsed is None:
        return None
    ticker, form, rest = parsed
    direct = resolve_chunk_file(table_summaries_dir, prefix, f"{prefix}.tables.summaries.jsonl")
    if direct is not None:
        return direct
    nested_alt = resolve_chunk_file(
        table_summaries_dir,
        prefix,
        f"{form}_{rest}.tables.summaries.jsonl",
    )
    if nested_alt is not None:
        return nested_alt
    with_ticker = f"{ticker}_{rest}"
    fallback = resolve_chunk_file(
        table_summaries_dir,
        prefix,
        f"{with_ticker}.tables.summaries.jsonl",
    )
    if fallback is not None:
        return fallback
    return None


def _parse_rest(form: str, rest: str) -> Tuple[int | None, str | None]:
    if not rest:
        return None, None
    if form == "10-K" and len(rest) >= 4 and rest[:4].isdigit():
        return int(rest[:4]), None
    if form == "10-Q":
        year_part = rest[:4]
        q_part = rest[4:] if len(rest) > 4 else ""
        if year_part.isdigit():
            return int(year_part), (q_part or None)
    return None, None


def _trace_for_record(
    doc_id: str | None,
    source_file: Path,
    prefix: str,
    source_ticker: str,
    source_form: str,
    fiscal_year: int | None,
    quarter_label: str | None,
    group: str,
) -> Dict[str, Any]:
    return {
        "doc_id": doc_id,
        "doc_type": group,
        "source": {
            "prefix": prefix,
            "ticker": source_ticker,
            "form_type": source_form,
            "fiscal_year": fiscal_year,
            "quarter": quarter_label,
            "source_file": str(source_file),
            "group": group,
        },
    }


def _attach_trace(
    records: List[Dict[str, Any]],
    source_file: Path,
    prefix: str,
    fiscal_year: int | None,
    quarter_label: str | None,
    form_type: str,
    ticker: str,
    group: str,
) -> None:
    for rec in records:
        trace = _trace_for_record(
            doc_id=rec.get("id"),
            source_file=source_file,
            prefix=prefix,
            source_ticker=ticker,
            source_form=form_type,
            fiscal_year=fiscal_year,
            quarter_label=quarter_label,
            group=group,
        )
        rec["trace"] = trace
        metadata = rec.setdefault("metadata", {})
        metadata["trace"] = {
            "source_file": str(source_file),
            "prefix": prefix,
            "fiscal_year": fiscal_year,
            "quarter": quarter_label,
            "doc_type": group,
        }


def _collect_docs_for_prefix(
    prefix: str,
    chunks_dir: Path,
    table_summaries_dir: Path,
    file_types: Sequence[str],
) -> List[Dict[str, Any]]:
    parsed = parse_filing_prefix(prefix)
    if parsed is None:
        print(f"[WARN] Unable to parse prefix '{prefix}', skipping.")
        return []

    ticker, form_type, rest = parsed
    fiscal_year, quarter_label = _parse_rest(form_type, rest)

    docs: List[Dict[str, Any]] = []
    common_meta = {
        "prefix": prefix,
        "ticker": ticker,
        "form_type": form_type,
        "fiscal_year": fiscal_year,
    }
    if quarter_label:
        common_meta["quarter"] = quarter_label

    if "text" in file_types:
        text_path = _find_text_chunk_path(chunks_dir, prefix)
        if text_path is None:
            print(f"[WARN] Missing text chunks for {prefix}")
        else:
            text_docs = build_text_docs(text_path=text_path, common_meta=common_meta)
            _attach_trace(
                records=text_docs,
                source_file=text_path,
                prefix=prefix,
                fiscal_year=fiscal_year,
                quarter_label=quarter_label,
                form_type=form_type,
                ticker=ticker,
                group="text",
            )
            docs.extend(text_docs)

    if ("tables" in file_types) or ("rows" in file_types):
        ts_path = _find_table_summary_path(table_summaries_dir, prefix)
        if ts_path is None:
            if "text" not in file_types:
                print(f"[WARN] Missing table summaries for {prefix}")
            # text-only path should still continue.
        else:
            docs_dict = build_table_and_row_docs(
                table_summaries_path=ts_path,
                common_meta=common_meta,
            )
            if "tables" in file_types and docs_dict.get("tables"):
                _attach_trace(
                    records=docs_dict["tables"],
                    source_file=ts_path,
                    prefix=prefix,
                    fiscal_year=fiscal_year,
                    quarter_label=quarter_label,
                    form_type=form_type,
                    ticker=ticker,
                    group="table",
                )
                docs.extend(docs_dict["tables"])

            if "rows" in file_types and docs_dict.get("rows"):
                _attach_trace(
                    records=docs_dict["rows"],
                    source_file=ts_path,
                    prefix=prefix,
                    fiscal_year=fiscal_year,
                    quarter_label=quarter_label,
                    form_type=form_type,
                    ticker=ticker,
                    group="table_row",
                )
                docs.extend(docs_dict["rows"])

    return docs


def _write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _iter_batches(items: Sequence[Dict[str, Any]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def _write_batches(
    records: Sequence[Dict[str, Any]],
    out_dir: Path,
    out_prefix: str,
    max_docs_per_batch: int,
) -> List[Path]:
    out_paths: List[Path] = []
    if max_docs_per_batch <= 0:
        out_path = out_dir / f"{out_prefix}.jsonl"
        _write_jsonl(out_path, records)
        return [out_path]

    total_batches = (len(records) + max_docs_per_batch - 1) // max_docs_per_batch
    for batch_idx, (_, batch_records) in enumerate(
        _iter_batches(records, max_docs_per_batch),
        start=1,
    ):
        out_path = (
            out_dir / f"{out_prefix}.batch_{batch_idx:03d}_of_{total_batches:03d}.jsonl"
        )
        _write_jsonl(out_path, batch_records)
        out_paths.append(out_path)
    return out_paths


def _embed_records_in_batches(
    records: Sequence[Dict[str, Any]],
    api_url: str,
    model: str,
    batch_size: int,
    timeout: float,
) -> List[Dict[str, Any]]:
    embedded: List[Dict[str, Any]] = []
    for batch_no, (batch_idx, batch) in enumerate(
        _iter_batches(records, batch_size),
        start=1,
    ):
        texts = [rec["content"] for rec in batch]
        embeddings = embed_batch_with_qwen3(
            texts=texts,
            api_url=api_url,
            model=model,
            timeout=timeout,
        )
        if len(embeddings) != len(batch):
            raise RuntimeError(
                f"Embedding mismatch for batch starting at {batch_idx}: "
                f"{len(embeddings)} embeddings for {len(batch)} docs."
            )

        for rec, emb in zip(batch, embeddings, strict=False):
            rec2 = {**rec}
            rec2["embedding"] = emb
            trace = rec2.setdefault("trace", {})
            trace["embedding_batch"] = batch_no
            embedded.append(rec2)
        print(f"[EMBED] Completed batch {batch_no}: {len(batch)} docs")

    return embedded


def _summarize_counts(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counter = Counter()
    for rec in records:
        meta = rec.get("metadata", {})
        doc_type = meta.get("doc_type") or rec.get("trace", {}).get("doc_type") or "unknown"
        counter[doc_type] += 1
    return {
        "total": len(records),
        "by_doc_type": dict(counter),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    tickers = load_tickers(
        tickers=args.tickers,
        from_file=args.from_file,
        required=not args.discover,
    )
    forms = [f.upper() for f in args.forms]

    quarters = [parse_quarter_spec(q) for q in args.quarters] if args.quarters else []

    chunks_dir = Path(args.chunks_dir)
    table_summaries_dir = Path(args.table_summaries_dir)
    out_dir = Path(args.out_dir)

    if not chunks_dir.exists() and not table_summaries_dir.exists():
        print(
            f"[ERROR] No source data found in {chunks_dir} or {table_summaries_dir}",
        )
        return 1

    if args.discover:
        discovered = _discover_tickers(chunks_dir, table_summaries_dir)
        if not tickers:
            tickers = discovered
        prefixes = _discover_prefixes(chunks_dir, table_summaries_dir, tickers, forms)
    else:
        prefixes = _collect_explicit_prefixes(
            tickers=tickers,
            forms=forms,
            years=args.years,
            quarter_specs=quarters,
        )

    if not prefixes:
        print("[WARN] No prefixes discovered/matched; nothing to process.")
        return 0

    all_records: List[Dict[str, Any]] = []
    for prefix in prefixes:
        records = _collect_docs_for_prefix(
            prefix=prefix,
            chunks_dir=chunks_dir,
            table_summaries_dir=table_summaries_dir,
            file_types=args.file_types,
        )
        if not records:
            print(f"[WARN] No records collected for {prefix}")
        all_records.extend(records)

    if not all_records:
        print("[ERROR] No embeddable records were collected.")
        return 1

    print(f"[INFO] Collected {len(all_records)} docs across {len(prefixes)} prefixes.")

    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / f"{args.out_prefix}.jsonl"
    _write_jsonl(all_path, all_records)

    batch_paths = _write_batches(
        records=all_records,
        out_dir=out_dir,
        out_prefix=args.out_prefix,
        max_docs_per_batch=args.max_docs_per_batch,
    )
    print(f"[INFO] Wrote combined payload: {all_path}")
    print(f"[INFO] Wrote {len(batch_paths)} batch payload file(s).")

    manifest = {
        "total_prefixes": len(prefixes),
        "total_docs": len(all_records),
        "counts": _summarize_counts(all_records),
        "outputs": {
            "combined": str(all_path),
            "batches": [str(p) for p in batch_paths],
        },
        "embed_requested": args.embed,
    }

    if args.embed:
        embedded_records = _embed_records_in_batches(
            records=all_records,
            api_url=args.api_url,
            model=args.model,
            batch_size=args.embed_batch_size,
            timeout=args.embed_timeout,
        )
        embedded_path = out_dir / f"{args.out_prefix}.embedded.jsonl"
        _write_jsonl(embedded_path, embedded_records)
        embedded_batch_paths = _write_batches(
            records=embedded_records,
            out_dir=out_dir,
            out_prefix=f"{args.out_prefix}.embedded",
            max_docs_per_batch=args.max_docs_per_batch,
        )
        manifest["embedded_total"] = len(embedded_records)
        manifest["outputs"]["embedded_combined"] = str(embedded_path)
        manifest["outputs"]["embedded_batches"] = [str(p) for p in embedded_batch_paths]
        print(f"[INFO] Wrote embedded combined payload: {embedded_path}")
        print(
            f"[INFO] Wrote {len(embedded_batch_paths)} embedded batch file(s)."
        )

    manifest_path = out_dir / f"{args.out_prefix}.manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
