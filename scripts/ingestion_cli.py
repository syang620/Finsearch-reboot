#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from _common import load_tickers, parse_quarter_specs

import orchestrate_chunk_html_filings
import orchestrate_embeddings
import orchestrate_html_downloads
import orchestrate_ingestion
import orchestrate_table_summaries


def _dispatch(name: str, argv: List[str]) -> int:
    if name == "download-html":
        return orchestrate_html_downloads.main(argv)
    if name == "chunk-html":
        return orchestrate_chunk_html_filings.main(argv)
    if name == "summarize-tables":
        return orchestrate_table_summaries.main(argv)
    if name == "embed":
        return orchestrate_embeddings.main(argv)
    if name == "ingest-qdrant":
        return orchestrate_ingestion.main(argv)
    raise ValueError(f"Unknown command: {name}")


def _run_step(name: str, argv: List[str]) -> None:
    print(f"\n=== {name} ===")
    print(f"[ingestion_cli] argv: {' '.join(argv)}")
    code = _dispatch(name, argv)
    if code != 0:
        raise SystemExit(code)


def _ticker_args(tickers: Sequence[str] | None, from_file: str | None) -> List[str]:
    out: List[str] = []
    if tickers:
        out.extend(["--tickers", *[str(t) for t in tickers]])
    if from_file:
        out.extend(["--from-file", str(from_file)])
    return out


def _iter_ingest_jobs(
    *,
    tickers: Sequence[str],
    forms: Sequence[str],
    years: Sequence[int] | None,
    quarters: Sequence[Tuple[int, str]] | None,
) -> List[Tuple[str, str, int, str | None]]:
    jobs: List[Tuple[str, str, int, str | None]] = []
    form_set = [str(f).upper() for f in forms]

    for ticker in tickers:
        if "10-K" in form_set:
            if not years:
                raise SystemExit("--years is required for 10-K ingestion jobs in run-all.")
            for year in years:
                jobs.append((ticker, "10-K", int(year), None))

        if "10-Q" in form_set:
            if not quarters:
                raise SystemExit("--quarters is required for 10-Q ingestion jobs in run-all.")
            for year, quarter in quarters:
                jobs.append((ticker, "10-Q", int(year), quarter))
    return jobs


def _run_all(args: argparse.Namespace) -> int:
    tickers = load_tickers(
        tickers=args.tickers,
        from_file=args.from_file,
        required=True,
    )
    quarter_specs = parse_quarter_specs(args.quarters)

    ticker_argv = _ticker_args(args.tickers, args.from_file)

    if not args.skip_download:
        argv = [
            *ticker_argv,
            "--output-dir",
            args.html_root,
            "--forms",
            *args.forms,
            "--per-form",
            str(args.per_form),
            "--sleep-seconds",
            str(args.sleep_seconds),
        ]
        _run_step("download-html", argv)

    if not args.skip_chunk:
        argv = [
            *ticker_argv,
            "--forms",
            *args.forms,
            "--html-root",
            args.html_root,
            "--out-dir",
            args.chunks_dir,
            "--python",
            args.python,
            "--split-max-tokens",
            str(args.split_max_tokens),
            "--split-overlap-paragraphs",
            str(args.split_overlap_paragraphs),
            "--split-encoding-model",
            args.split_encoding_model,
        ]
        if args.years:
            argv.extend(["--years", *[str(y) for y in args.years]])
        if args.quarters:
            argv.extend(["--quarters", *args.quarters])
        _run_step("chunk-html", argv)

    if not args.skip_summarize:
        argv = [
            *ticker_argv,
            "--chunks-dir",
            args.chunks_dir,
            "--out-dir",
            args.table_summaries_dir,
            "--api-url",
            args.summarize_api_url,
            "--model",
            args.summarize_model,
            "--temperature",
            str(args.summarize_temperature),
            "--python",
            args.python,
        ]
        if args.max_tables is not None:
            argv.extend(["--max-tables", str(args.max_tables)])
        _run_step("summarize-tables", argv)

    if not args.skip_embed:
        argv = [
            *ticker_argv,
            "--forms",
            *args.forms,
            "--chunks-dir",
            args.chunks_dir,
            "--table-summaries-dir",
            args.table_summaries_dir,
            "--out-root",
            args.embeddings_root,
            "--api-url",
            args.embed_api_url,
            "--model",
            args.embed_model,
            "--batch-size",
            str(args.embed_batch_size),
            "--file-types",
            *args.file_types,
        ]
        if args.years:
            argv.extend(["--years", *[str(y) for y in args.years]])
        if args.quarters:
            argv.extend(["--quarters", *args.quarters])
        _run_step("embed", argv)

    if not args.skip_ingest:
        jobs = _iter_ingest_jobs(
            tickers=tickers,
            forms=args.forms,
            years=args.years,
            quarters=quarter_specs,
        )
        first_job = True
        for ticker, form, year, quarter in jobs:
            argv = [
                "--ticker",
                ticker,
                "--form",
                form,
                "--year",
                str(year),
                "--embeddings-root",
                args.embeddings_root,
                "--file-types",
                *args.file_types,
                "--batch-size",
                str(args.ingest_batch_size),
                "--host",
                args.qdrant_host,
                "--port",
                str(args.qdrant_port),
                "--collection-name",
                args.collection_name,
            ]
            if quarter:
                argv.extend(["--quarter", quarter])
            if args.qdrant_https:
                argv.append("--https")
            if args.qdrant_api_key:
                argv.extend(["--api-key", args.qdrant_api_key])
            if args.bge_m3_allow_download:
                argv.append("--bge-m3-allow-download")
            if args.bge_m3_model_name:
                argv.extend(["--bge-m3-model-name", args.bge_m3_model_name])
            if args.bge_m3_cache_dir:
                argv.extend(["--bge-m3-cache-dir", args.bge_m3_cache_dir])
            if args.recreate_collection and first_job:
                argv.append("--recreate-collection")
            first_job = False
            _run_step("ingest-qdrant", argv)

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified ingestion CLI for FinSearch data pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for cmd in ("download-html", "chunk-html", "summarize-tables", "embed", "ingest-qdrant"):
        p = sub.add_parser(cmd, help=f"Alias to scripts/{cmd}.")
        p.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to the underlying script.")

    run_all = sub.add_parser(
        "run-all",
        help="Run download -> chunk -> summarize -> embed -> ingest in sequence.",
    )
    run_all.add_argument("--tickers", nargs="+")
    run_all.add_argument("--from-file")
    run_all.add_argument("--forms", nargs="+", default=["10-K", "10-Q"])
    run_all.add_argument("--years", nargs="+", type=int)
    run_all.add_argument("--quarters", nargs="+")

    run_all.add_argument("--html-root", default="data/html_filings")
    run_all.add_argument("--chunks-dir", default="data/chunked")
    run_all.add_argument("--table-summaries-dir", default="data/chunked/table_summaries")
    run_all.add_argument("--embeddings-root", default="data/embedding")
    run_all.add_argument("--python", default=sys.executable)

    run_all.add_argument("--per-form", type=int, default=2)
    run_all.add_argument("--sleep-seconds", type=float, default=0.2)
    run_all.add_argument("--split-max-tokens", type=int, default=1200)
    run_all.add_argument("--split-overlap-paragraphs", type=int, default=1)
    run_all.add_argument("--split-encoding-model", default="text-embedding-3-large")

    run_all.add_argument("--summarize-api-url", default="http://localhost:11434/api/generate")
    run_all.add_argument("--summarize-model", default="minimax-m2:cloud")
    run_all.add_argument("--summarize-temperature", type=float, default=0.0)
    run_all.add_argument("--max-tables", type=int)

    run_all.add_argument("--embed-api-url", default="http://localhost:11434/api/embed")
    run_all.add_argument("--embed-model", default="qwen3-embedding:8b")
    run_all.add_argument("--embed-batch-size", type=int, default=16)
    run_all.add_argument(
        "--file-types",
        nargs="+",
        choices=["text", "tables", "rows"],
        default=["text", "tables", "rows"],
    )

    run_all.add_argument("--qdrant-host", default="localhost")
    run_all.add_argument("--qdrant-port", type=int, default=6333)
    run_all.add_argument("--qdrant-https", action="store_true")
    run_all.add_argument("--qdrant-api-key")
    run_all.add_argument("--collection-name", default="sec_docs_hybrid")
    run_all.add_argument("--ingest-batch-size", type=int, default=128)
    run_all.add_argument("--recreate-collection", action="store_true")
    run_all.add_argument("--bge-m3-model-name", default="BAAI/bge-m3")
    run_all.add_argument("--bge-m3-cache-dir")
    run_all.add_argument("--bge-m3-allow-download", action="store_true")

    run_all.add_argument("--skip-download", action="store_true")
    run_all.add_argument("--skip-chunk", action="store_true")
    run_all.add_argument("--skip-summarize", action="store_true")
    run_all.add_argument("--skip-embed", action="store_true")
    run_all.add_argument("--skip-ingest", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run-all":
        return _run_all(args)

    forwarded = list(getattr(args, "args", []) or [])
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    return _dispatch(args.command, forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
