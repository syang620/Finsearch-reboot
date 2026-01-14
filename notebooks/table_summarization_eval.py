"""
table_summarization_eval.py

Utility for evaluating multiple LLMs on SEC table summarization.
- Checks JSON validity
- Checks row index correctness & coverage
- Enforces "no numbers except years" rule
- Produces a pandas DataFrame of metrics

You must provide `table_chunks` (list of dicts) and a list of model names.
"""

import json
import re
import time
from typing import Dict, Any, List, Optional

import pandas as pd
import requests


# =====================================================================
#  Helpers: table → Markdown
# =====================================================================

def df_to_markdown_for_llm(df: pd.DataFrame) -> str:
    """Convert DataFrame to markdown (no index)."""
    return df.to_markdown(index=False)


# =====================================================================
#  System Prompt for summarization
# =====================================================================

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


# =====================================================================
#  Prompt builder
# =====================================================================

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


# =====================================================================
#  Ollama annotation wrapper
# =====================================================================

def annotate_table_with_ollama(
    table_chunk: Dict[str, Any],
    model: str,
    api_url: str,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Call Ollama to summarize a table with strict JSON output."""
    user_prompt = build_user_prompt(table_chunk)
    full_prompt = SYSTEM_PROMPT.strip() + "\n\n---\n\n" + user_prompt.strip()
    # print(full_prompt)
    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature, "num_predict": 4096},
    }
    
    resp = requests.post(api_url, json=payload)
    resp.raise_for_status()

    obj = resp.json()
    text = (obj.get("response") or "").strip()
    print(obj)

    return json.loads(text)


# =====================================================================
#  Numeric detection (years OK, other numbers forbidden)
# =====================================================================

_NUMBER_TOKEN_RE = re.compile(r"\d[\d,\.]*")
_YEAR_MIN = 1900
_YEAR_MAX = 2100

def has_non_year_number(text: str) -> bool:
    """
    Returns True if text contains a number that is NOT a plausible year (1900–2100).
    Table values like "5.2" or "10,000" should be flagged; "2024" is allowed.
    """
    for match in _NUMBER_TOKEN_RE.finditer(text):
        raw = match.group(0)
        cleaned = raw.replace(",", "").strip(".")
        if not re.search(r"\d", cleaned):
            continue

        if cleaned.isdigit():
            if len(cleaned) == 4:
                yr = int(cleaned)
                if _YEAR_MIN <= yr <= _YEAR_MAX:
                    continue  # allowed year
            return True  # not a year
        else:
            return True  # decimals → not a year

    return False


# =====================================================================
#  Annotation evaluation
# =====================================================================

def _safe_get_row_summaries(ann: Dict[str, Any]):
    rs = ann.get("row_summaries")
    return rs if isinstance(rs, list) else None


def evaluate_single_annotation(
    table_chunk: Dict[str, Any],
    annotation: Dict[str, Any],
) -> Dict[str, Any]:

    df = table_chunk["table_df"]
    n_rows = len(df)

    results = {
        "json_valid": True,
        "n_table_rows": n_rows,
        "n_row_summaries": 0,
        "row_index_valid_fraction": 0.0,
        "row_index_coverage_fraction": 0.0,
        "rows_with_non_year_number_fraction": 0.0,
        "has_any_non_year_numbers": False,
        "table_summary_has_non_year_numbers": False,
    }

    # Check for missing row_summaries
    rs = _safe_get_row_summaries(annotation)
    if rs is None:
        results["json_valid"] = False
        return results

    results["n_row_summaries"] = len(rs)
    if len(rs) == 0 or n_rows == 0:
        return results

    # ----- Row index metrics -----
    row_indices = [r.get("row_index") for r in rs if isinstance(r.get("row_index"), int)]
    valid = [i for i in row_indices if 0 <= i < n_rows]

    if row_indices:
        results["row_index_valid_fraction"] = len(valid) / len(row_indices)
    results["row_index_coverage_fraction"] = len(set(valid)) / n_rows

    # ----- Check for table-number leakage -----
    descriptions = [(r.get("description") or "") for r in rs if isinstance(r, dict)]
    non_year_count = sum(1 for d in descriptions if has_non_year_number(d))

    results["rows_with_non_year_number_fraction"] = non_year_count / len(descriptions)
    results["has_any_non_year_numbers"] = non_year_count > 0

    table_summary = annotation.get("table_summary") or ""
    results["table_summary_has_non_year_numbers"] = has_non_year_number(table_summary)

    return results


# =====================================================================
#  Core driver: evaluate many models on many tables
# =====================================================================

def evaluate_models_on_tables(
    models: List[str],
    table_chunks: List[Dict[str, Any]],
    api_url: str,
    temperature: float = 0.0,
    save_jsonl_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Core evaluation function.

    Parameters
    ----------
    models : list of model names (str)
    table_chunks : list of dicts with keys including 'table_df'
    api_url : Ollama endpoint URL
    temperature : decoding temperature
    save_jsonl_path : optional log file path

    Returns
    -------
    pandas.DataFrame with one row per (model, table_index)
    """
    records = []
    jf = open(save_jsonl_path, "w", encoding="utf-8") if save_jsonl_path else None

    try:
        for model in models:
            for idx, table_chunk in enumerate(table_chunks):
                start = time.time()
                error = None
                annotation = None
                print(model, table_chunk["section_title"])
                try:
                    annotation = annotate_table_with_ollama(
                        table_chunk,
                        model=model,
                        api_url=api_url,
                        temperature=temperature,
                    )
                    metrics = evaluate_single_annotation(table_chunk, annotation)
                    
                except Exception as e:
                    error = str(e)
                    metrics = {
                        "json_valid": False,
                        "n_table_rows": len(table_chunk["table_df"]),
                        "n_row_summaries": 0,
                        "row_index_valid_fraction": 0.0,
                        "row_index_coverage_fraction": 0.0,
                        "rows_with_non_year_number_fraction": 0.0,
                        "has_any_non_year_numbers": False,
                        "table_summary_has_non_year_numbers": False,
                    }

                elapsed = time.time() - start

                rec = {
                    "model": model,
                    "table_index": idx,
                    "table_title": table_chunk.get("section_title"),
                    "elapsed_sec": elapsed,
                    "error": error,
                    **metrics,
                }

                if jf:
                    jf.write(json.dumps({**rec, "annotation": annotation}, default=str) + "\n")

                records.append(rec)

        return pd.DataFrame.from_records(records)

    finally:
        if jf:
            jf.close()


# =====================================================================
#  END OF FILE
# =====================================================================
