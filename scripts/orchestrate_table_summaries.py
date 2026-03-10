#!/usr/bin/env python
"""
Orchestrate table summarization using ingestion.tables_summarizer.

This script scans table chunk files in data/chunked (or a custom directory),
derives filing prefixes from *.tables.jsonl filenames, and invokes
`python -m ingestion.tables_summarizer` for each prefix.

Summaries are written to a separate directory (default: data/table_summaries)
with filenames that preserve the original prefix, e.g.:

  data/chunked/AAPL/10-K/10-K_2024.tables.jsonl
    -> data/table_summaries/AAPL/10-K/10-K_2024.tables.summaries.jsonl

Typical usage:

  PYTHONPATH=src python scripts/orchestrate_table_summaries.py \\
      --tickers AAPL MSFT AMZN \\
      --api-url https://dashscope.aliyuncs.com/compatible-mode/v1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Set

from _common import load_tickers_set_optional


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locate *.tables.jsonl files under a chunks directory and "
            "summarize them via ingestion.tables_summarizer."
        ),
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Optional list of tickers to include (e.g., AAPL MSFT). If omitted, process all prefixes.",
    )
    parser.add_argument(
        "--from-file",
        help="Optional path to a text file with one ticker per line.",
    )
    parser.add_argument(
        "--chunks-dir",
        default="data/chunked",
        help="Directory containing chunk JSONL files (default: data/chunked).",
    )
    parser.add_argument(
        "--out-dir",
        default="data/table_summaries",
        help="Directory to write per-prefix table summary JSONL files.",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help=(
            "HTTP endpoint for non-Qwen models, e.g. http://localhost:11434/api/generate. "
            "Qwen chat models use DASHSCOPE_BASE_URL / DASHSCOPE_API_KEY."
        ),
    )
    parser.add_argument(
        "--model",
        default="minimax-m2:cloud",
        help="LLM model name (default: minimax-m2:cloud).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Decoding temperature (default: 0.0).",
    )
    parser.add_argument(
        "--max-tables",
        type=int,
        help="Optional cap on tables per prefix to summarize.",
    )
    parser.add_argument(
        "--export-prompts-jsonl",
        help="Optional path to write prompts as id/text JSONL for remote summarization.",
    )
    parser.add_argument(
        "--import-results-jsonl",
        help="Optional path to import remote results JSONL and build summary JSONL.",
    )
    parser.add_argument(
        "--skip-invoke",
        action="store_true",
        help="Skip local LLM calls. Use with --export-prompts-jsonl or --import-results-jsonl.",
    )
    parser.add_argument(
        "--prompt-id-start",
        type=int,
        default=1,
        help="Starting prompt id for exported prompts.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use when invoking ingestion.tables_summarizer.",
    )
    return parser.parse_args(argv)


def _find_prefixes(chunks_dir: Path, tickers: Set[str] | None) -> List[str]:
    """
    Discover filing prefixes from *.tables.jsonl filenames in chunks_dir.

    For a file like:
        AAPL/10-K/10-K_2024.tables.jsonl
    we derive a prefix:
        AAPL_10-K_2024
    """
    prefixes: List[str] = []
    for path in sorted(chunks_dir.rglob("*.tables.jsonl")):
        # Ignore Jupyter checkpoints.
        if ".ipynb_checkpoints" in path.parts:
            continue
        stem = path.stem  # e.g., "AAPL_10-K_2024.tables"
        if not stem.endswith(".tables"):
            continue
        raw_prefix = stem[: -len(".tables")]

        if raw_prefix.startswith("10-K_") or raw_prefix.startswith("10-Q_"):
            # Nested layout: <ROOT>/<TICKER>/<FORM>/<FORM>_YYYY...
            parts = path.parts
            if len(parts) < 3:
                continue
            form = parts[-2]
            ticker = parts[-3]
            rest = raw_prefix[len(form) + 1 :]
            if form in {"10-K", "10-Q"} and ticker and rest:
                prefix = f"{ticker}_{form}_{rest}"
            else:
                prefix = raw_prefix
        else:
            # Legacy flat layout: <PREFIX>.tables.jsonl
            prefix = raw_prefix

        if tickers:
            # Expect prefix to start with "<TICKER>_"
            prefix_ticker = prefix.split("_", 1)[0].upper()
            if prefix_ticker not in tickers:
                continue

        prefixes.append(prefix)

    return prefixes


def _prefix_to_chunks_file(chunks_dir: Path, prefix: str) -> Path:
    """
    Resolve the table-chunk file path for a filing prefix discovered by _find_prefixes.
    """
    parts = prefix.split("_", 2)
    if len(parts) >= 3 and parts[1] in {"10-K", "10-Q"}:
        # Nested layout: <ROOT>/<TICKER>/<FORM>/<FORM>_YYYY...
        return chunks_dir / parts[0] / parts[1] / f"{parts[1]}_{parts[2]}.tables.jsonl"

    return chunks_dir / f"{prefix}.tables.jsonl"


def _count_table_chunks(chunks_file: Path) -> int:
    """
    Count table chunk entries for an export/import prompt id plan.
    """
    count = 0
    with chunks_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _run_tables_summarizer_for_prefix(
    python_exe: str,
    prefix: str,
    chunks_dir: Path,
    out_dir: Path,
    api_url: str | None,
    model: str,
    temperature: float,
    max_tables: int | None,
    skip_invoke: bool = False,
    export_prompts_jsonl: str | None = None,
    import_results_jsonl: str | None = None,
    prompt_id_start: int = 1,
) -> None:
    parts = prefix.split("_", 2)
    if len(parts) >= 3 and parts[1] in {"10-K", "10-Q"}:
        # Backward compatible nested output layout:
        # <out-dir>/<TICKER>/<FORM>/<FORM>_<REST>.tables.summaries.jsonl
        out_dir = out_dir / parts[0] / parts[1]
        out_path = out_dir / f"{parts[1]}_{parts[2]}.tables.summaries.jsonl"
    else:
        out_dir = out_dir
        out_path = out_dir / f"{prefix}.tables.summaries.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        out_path.unlink()

    cmd = [
        python_exe,
        "-m",
        "ingestion.tables_summarizer",
        "--prefixes",
        prefix,
        "--chunks-dir",
        str(chunks_dir),
        "--model",
        model,
        "--temperature",
        str(temperature),
        "--output-jsonl",
        str(out_path),
    ]
    if api_url is not None and not skip_invoke:
        cmd.extend(["--api-url", api_url])
    if skip_invoke:
        cmd.append("--skip-invoke")
    if export_prompts_jsonl is not None:
        cmd.extend(["--export-prompts-jsonl", export_prompts_jsonl])
    if import_results_jsonl is not None:
        cmd.extend(["--import-results-jsonl", import_results_jsonl])
    if prompt_id_start is not None:
        cmd.extend(["--prompt-id-start", str(prompt_id_start)])

    if max_tables is not None:
        cmd.extend(["--max-tables", str(max_tables)])

    print(f"Running tables_summarizer for {prefix}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.skip_invoke and args.import_results_jsonl is None and args.export_prompts_jsonl is None:
        raise ValueError(
            "--skip-invoke requires --export-prompts-jsonl or --import-results-jsonl"
        )
    if (
        not args.skip_invoke
        and args.api_url is None
        and args.import_results_jsonl is None
    ):
        raise ValueError("--api-url is required unless --skip-invoke is set.")
    chunks_dir = Path(args.chunks_dir)
    out_dir = Path(args.out_dir)

    tickers = load_tickers_set_optional(
        tickers=args.tickers,
        from_file=args.from_file,
    )
    prefixes = _find_prefixes(chunks_dir, tickers)

    if not prefixes:
        print(f"No *.tables.jsonl files found in {chunks_dir} matching the given filters.")
        return 0

    next_prompt_id = args.prompt_id_start
    for prefix in prefixes:
        chunks_file = _prefix_to_chunks_file(chunks_dir, prefix)
        if not chunks_file.exists():
            print(f"[WARN] Expected chunks file missing for prefix={prefix}: {chunks_file}")
            continue

        prompt_budget = _count_table_chunks(chunks_file)
        if args.max_tables is not None:
            prompt_budget = min(prompt_budget, args.max_tables)

        _run_tables_summarizer_for_prefix(
            python_exe=args.python,
            prefix=prefix,
            chunks_dir=chunks_dir,
            out_dir=out_dir,
            api_url=args.api_url,
            model=args.model,
            temperature=args.temperature,
            max_tables=args.max_tables,
            skip_invoke=args.skip_invoke,
            export_prompts_jsonl=args.export_prompts_jsonl,
            import_results_jsonl=args.import_results_jsonl,
            prompt_id_start=next_prompt_id,
        )
        next_prompt_id += prompt_budget

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
