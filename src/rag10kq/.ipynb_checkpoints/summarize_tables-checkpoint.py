#!/usr/bin/env python
"""
Summarize SEC table chunks with an LLM for later embedding.

This script:
  1) Loads table chunks using the helper in notebooks/chunks_loader.py.
  2) Builds a strict JSON summarization prompt (from table_summarization_eval).
  3) Calls an LLM (default model: minimax-m2:cloud) via an HTTP API
     compatible with the Ollama /table_summarization_eval pattern.
  4) Writes one JSONL file with annotations per filing prefix.

Example:
    PYTHONPATH=src python -m rag10kq.summarize_tables \\
        --prefixes AAPL_10-K_2024 AAPL_10-Q_2025Q1 \\
        --chunks-dir data/chunked \\
        --api-url http://localhost:11434/api/generate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import requests


DEFAULT_MODEL = "minimax-m2:cloud"


# ---------------------------------------------------------------------
# Import chunks_loader from notebooks
# ---------------------------------------------------------------------

def _import_chunks_loader():
    """
    Import chunks_loader.py from the notebooks directory.

    This keeps the loading logic in one place without duplicating code.
    """
    root = Path(__file__).resolve().parents[2]
    notebooks_dir = root / "notebooks"
    if not notebooks_dir.exists():
        raise RuntimeError(f"Cannot find notebooks directory at {notebooks_dir}")

    if str(notebooks_dir) not in sys.path:
        sys.path.append(str(notebooks_dir))

    try:
        from chunks_loader import load_filing_chunks  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Failed to import load_filing_chunks from notebooks/chunks_loader.py. "
            "Make sure that file exists and is importable."
        ) from exc

    return load_filing_chunks


load_filing_chunks = _import_chunks_loader()


def load_table_chunks_for_prefix(
    prefix: str,
    chunks_dir: Path,
) -> List[Dict[str, Any]]:
    """
    Load table chunks for a single filing prefix, using notebooks/chunks_loader.
    """
    _text_chunks, table_chunks = load_filing_chunks(
        prefix=prefix,
        out_dir=str(chunks_dir),
        use_split_text=True,
        rebuild_table_df=True,
    )
    return table_chunks


# ---------------------------------------------------------------------
#  Helpers: table → Markdown (from table_summarization_eval)
# ---------------------------------------------------------------------

def df_to_markdown_for_llm(df: pd.DataFrame) -> str:
    """Convert DataFrame to markdown (no index)."""
    return df.to_markdown(index=False)


# ---------------------------------------------------------------------
#  System Prompt for summarization (from table_summarization_eval)
# ---------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an experienced, meticulous financial reporting analyst.
You are given tables extracted from SEC 10-K and 10-Q filings.
Your job is to:
1. Describe in summary and in plain English to explain what the entire table is about.
2. Describe in summary what each non-header row represents.

STRICT RULES:
- Use ONLY information that appears in the table or in the provided metadata.
- Do NOT invent or infer any new facts, trends, or causes.
- Do NOT compute growth, percentages, or differences (no “increased”, “decreased”, “higher”, etc.).
- DO NOT include numeric values that appear as table entries.
- You may include year-like values (e.g., 2023, 2024, 2025) as long as they refer to dates.
- Copy dates and units exactly as they appear.
- Output MUST be a single valid JSON object, nothing else.

JSON FORMAT:
{
  "table_summary": "one or two sentences about the whole table",
  "row_summaries": [
    {
      "row_index": <0-based index>,
      "row_label": "<row label or null>",
      "description": "one sentence describing this row"
    }
  ]
}
"""


# ---------------------------------------------------------------------
#  Prompt builder (from table_summarization_eval, lightly adapted)
# ---------------------------------------------------------------------

def build_user_prompt(table_chunk: Dict[str, Any]) -> str:
    df: pd.DataFrame = table_chunk["table_df"]
    meta: Dict[str, Any] = table_chunk.get("meta", {})
    table_md = df_to_markdown_for_llm(df)
    heading_path = meta.get("heading_path", table_chunk.get("heading_path", []))

    return f"""Filing metadata:
- Company: {meta.get("company_name")}
- Ticker: {meta.get("ticker")}
- CIK: {meta.get("cik")}
- Form type: {meta.get("form_type")}
- Fiscal year: {meta.get("fiscal_year")}
- Filing date: {meta.get("filing_date")}
- Item: {meta.get("item_id")} – {meta.get("item_title")}
- Section path: { ' > '.join(heading_path) }

Table (first row is header):
{table_md}

Remember:
- Use only info in the table + metadata.
- Do not compute new metrics or trends.
- Do not repeat numeric values from the table.
- Years like 2023/2024/2025 are allowed.
- Respond with JSON only.
"""


# ---------------------------------------------------------------------
#  LLM summarization call (Ollama-style HTTP API)
# ---------------------------------------------------------------------

def summarize_table_with_llm(
    table_chunk: Dict[str, Any],
    model: str,
    api_url: str,
    temperature: float = 0.0,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """
    Call an LLM to summarize a table with strict JSON output.

    The API is expected to be compatible with Ollama's /generate endpoint:
      POST api_url with JSON:
        {
          "model": model,
          "prompt": full_prompt,
          "stream": false,
          "format": "json",
          "options": {"temperature": temperature, "num_predict": 4096}
        }

    The response JSON should contain a 'response' field that is a JSON string
    matching the schema described in SYSTEM_PROMPT.
    """
    user_prompt = build_user_prompt(table_chunk)
    full_prompt = SYSTEM_PROMPT.strip() + "\n\n---\n\n" + user_prompt.strip()

    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature, "num_predict": 4096},
    }

    resp = requests.post(api_url, json=payload, timeout=timeout)
    resp.raise_for_status()

    obj = resp.json()
    text = (obj.get("response") or "").strip()

    # Parse the JSON string returned by the model
    return json.loads(text)


# ---------------------------------------------------------------------
#  CLI + orchestration
# ---------------------------------------------------------------------

def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize SEC table chunks for later embedding.",
    )
    parser.add_argument(
        "--prefixes",
        nargs="+",
        required=True,
        help="Filing prefixes, e.g. AAPL_10-K_2024 AAPL_10-Q_2025Q1.",
    )
    parser.add_argument(
        "--chunks-dir",
        default="data/chunked",
        help="Directory containing chunk JSONL files (default: data/chunked).",
    )
    parser.add_argument(
        "--output-jsonl",
        help=(
            "Path to save table summaries as JSONL. "
            "If omitted, defaults to <chunks-dir>/table_summaries.jsonl."
        ),
    )
    parser.add_argument(
        "--api-url",
        required=True,
        help="HTTP endpoint for the LLM (e.g., http://localhost:11434/api/generate).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LLM model name (default: {DEFAULT_MODEL}).",
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
        help="Optional cap on number of tables per prefix to summarize.",
    )
    return parser.parse_args(argv)


def summarize_prefix(
    prefix: str,
    chunks_dir: Path,
    api_url: str,
    model: str,
    temperature: float,
    max_tables: int | None,
) -> List[Dict[str, Any]]:
    """
    Summarize all (or up to max_tables) tables for a single filing prefix.
    """
    table_chunks = load_table_chunks_for_prefix(prefix, chunks_dir)
    records: List[Dict[str, Any]] = []

    for idx, table_chunk in enumerate(table_chunks):
        if max_tables is not None and idx >= max_tables:
            break

        start = time.time()
        error: str | None = None
        annotation: Dict[str, Any] | None = None

        try:
            annotation = summarize_table_with_llm(
                table_chunk=table_chunk,
                model=model,
                api_url=api_url,
                temperature=temperature,
            )
        except Exception as exc:
            error = str(exc)

        elapsed = time.time() - start

        rec: Dict[str, Any] = {
            "prefix": prefix,
            "table_index": idx,
            "section_title": table_chunk.get("section_title"),
            "model": model,
            "elapsed_sec": elapsed,
            "error": error,
            "annotation": annotation,
        }
        records.append(rec)

    return records


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    chunks_dir = Path(args.chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        Path(args.output_jsonl)
        if args.output_jsonl
        else chunks_dir / "table_summaries.jsonl"
    )

    all_records: List[Dict[str, Any]] = []

    with output_path.open("w", encoding="utf-8") as f:
        for prefix in args.prefixes:
            print(f"\n=== Summarizing tables for prefix: {prefix} ===")
            records = summarize_prefix(
                prefix=prefix,
                chunks_dir=chunks_dir,
                api_url=args.api_url,
                model=args.model,
                temperature=args.temperature,
                max_tables=args.max_tables,
            )
            for rec in records:
                all_records.append(rec)
                f.write(json.dumps(rec, default=str) + "\n")

    print(f"\nWrote {len(all_records)} table summaries to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

