#!/usr/bin/env python
"""Normalize table summary import results from remote LLM summarization runs.

This utility handles common issues seen in batched summarization output:
- Duplicate prompt IDs.
- Missing/invalid payload fields.
- Malformed payload content.
- Extraction of retriable failures with provenance (prefix + table_index).

Typical flow:
    1) Run remote jobs using exported prompts.
    2) Save combined output JSONL.
    3) Run this script to dedupe/normalize and emit retry manifests.

Outputs:
- cleaned JSONL compatible with `ingestion.tables_summarizer --import-results-jsonl`
- failure manifest
- retry manifest grouped by filing prefix
- optional prompt manifest containing only retriable IDs (if prompt export file is provided)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


PAYLOAD_KEYS = ("annotation", "response", "result", "text", "output", "response_text")


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize and deduplicate table summarization import JSONL.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the raw import/result JSONL file from remote summarization.",
    )
    parser.add_argument(
        "--out-clean",
        default=None,
        help="Output path for cleaned JSONL (default: <input>.clean.jsonl).",
    )
    parser.add_argument(
        "--out-failures",
        default=None,
        help="Output path for failures JSONL (default: <input>.failures.jsonl).",
    )
    parser.add_argument(
        "--out-retry",
        default=None,
        help="Output path for retry cases JSONL (default: <input>.retries.jsonl).",
    )
    parser.add_argument(
        "--out-retry-by-prefix",
        default=None,
        help="Output path for grouped retry manifest by prefix (default: <input>.retries_by_prefix.jsonl).",
    )
    parser.add_argument(
        "--prompt-export",
        default=None,
        help=(
            "Optional exported prompt file (id/text JSONL) used for summarization."
            " When provided, script emits prompts for retriable IDs."
        ),
    )
    parser.add_argument(
        "--out-prompt-retry",
        default=None,
        help=(
            "Optional output for prompt-export rows that correspond to retriable IDs"
            " (default: <input>.retry_prompts.jsonl)."
        ),
    )
    parser.add_argument(
        "--mark-context-overflow-as-retriable",
        action="store_true",
        help=(
            "Keep token-length overflow/invalid-context rows in retriable list (default: true)."
        ),
    )
    parser.set_defaults(mark_context_overflow_as_retriable=True)
    return parser.parse_args(argv)


def _extract_payload(record: Dict[str, Any]) -> Tuple[Any, str | None]:
    for key in PAYLOAD_KEYS:
        if key in record:
            return record[key], key
    return None, None


def _validate_output_payload(payload: Any) -> Tuple[bool, str | None]:
    if payload is None:
        return False, "missing_payload"
    if isinstance(payload, dict):
        return True, None
    if isinstance(payload, str):
        if not payload.strip():
            return False, "empty_payload_string"
        try:
            json.loads(payload)
            return True, None
        except Exception as exc:
            return False, f"invalid_json_payload: {exc}"

    return False, f"unsupported_payload_type:{type(payload).__name__}"


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        return v in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _is_retriable(error: str | None, ok: bool, payload_valid: bool) -> bool:
    if ok:
        return False
    if not payload_valid:
        return True
    if error is None:
        return False

    lower_error = error.lower()
    if "context length" in lower_error or "input tokens" in lower_error:
        return True
    if "validationerror" in lower_error or "vllmvalidation" in lower_error:
        return True
    return False


def _group_retries_by_prefix(retry_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in sorted(retry_records, key=lambda x: (x.get("prefix", ""), x.get("table_index", 0), x.get("id", 0))):
        prefix = rec.get("prefix") or ""
        grouped[prefix].append(
            {
                "id": rec.get("id"),
                "table_index": rec.get("table_index"),
                "legacy_id": rec.get("legacy_id"),
                "error": rec.get("error"),
                "reason": rec.get("retry_reason"),
            }
        )

    return [
        {
            "prefix": prefix,
            "retry_count": len(entries),
            "entries": entries,
        }
        for prefix, entries in grouped.items()
    ]


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_prompt_map(path: Path) -> Dict[int, Dict[str, Any]]:
    prompts = {}
    for line_no, line in enumerate(path.open("r", encoding="utf-8"), start=1):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if "id" not in rec:
            raise ValueError(f"Missing id at prompt line {line_no}: {path}")
        prompt_id = int(rec["id"])
        prompts[prompt_id] = rec
    return prompts


def normalize_input(path: Path, mark_context_overflow_as_retriable: bool = True) -> Dict[str, Any]:
    seen: Dict[int, Dict[str, Any]] = {}
    raw_line_count = 0

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            raw_line_count += 1

            rec = json.loads(line)
            raw_id = rec.get("id")
            if raw_id is None:
                print(f"[WARN] Line {line_no}: missing id, skipping")
                continue
            try:
                prompt_id = int(raw_id)
            except Exception:
                print(f"[WARN] Line {line_no}: invalid id {raw_id}, skipping")
                continue

            ok = _normalize_bool(rec.get("ok", False))
            payload, payload_key = _extract_payload(rec)
            payload_valid, payload_error = _validate_output_payload(payload)

            normalized = {
                "id": prompt_id,
                "legacy_id": rec.get("legacy_id"),
                "ticker": rec.get("ticker"),
                "form_type": rec.get("form_type"),
                "fiscal_year": rec.get("fiscal_year"),
                "prefix": rec.get("prefix"),
                "table_index": rec.get("table_index"),
                "error": rec.get("error"),
                "ok": ok,
                "payload_key": payload_key,
                "output": payload if payload_key is not None else None,
                "payload_valid": payload_valid,
                "payload_error": payload_error,
            }

            if not payload_valid and normalized["ok"]:
                print(f"[WARN] Line {line_no}: id={prompt_id} marked ok=true but payload invalid ({payload_error})")

            prev = seen.get(prompt_id)
            if prev is None:
                seen[prompt_id] = normalized
                continue

            # Prefer successful attempts over failures.
            if not prev.get("ok", False) and ok:
                seen[prompt_id] = normalized
            else:
                # Last attempt wins for equal/unknown status.
                seen[prompt_id] = normalized

    cleaned = [seen[i] for i in sorted(seen.keys())]

    failed: List[Dict[str, Any]] = [
        r
        for r in cleaned
        if not r.get("ok", False)
    ]

    for r in failed:
        error = r.get("error")
        if isinstance(error, str):
            if not mark_context_overflow_as_retriable and (
                "context length" in error.lower()
                or "input tokens" in error.lower()
            ):
                r["retry_reason"] = "disabled_by_policy"
            elif "context length" in error.lower() or "input tokens" in error.lower():
                r["retry_reason"] = "context_length_overflow"
            elif "vllmvalidation" in error.lower():
                r["retry_reason"] = "validation_error"
            else:
                r["retry_reason"] = "model_error"
        elif r.get("payload_valid", True) is False:
            r["retry_reason"] = "invalid_payload"
        elif r.get("payload_error"):
            r["retry_reason"] = r["payload_error"]
        else:
            r["retry_reason"] = "missing_payload"

        if r.get("retry_reason") == "missing_payload" and r.get("error"):
            r["retry_reason"] = r.get("error")

    for r in failed:
        r["retriable"] = _is_retriable(
            error=r.get("error") if isinstance(r.get("error"), str) else None,
            ok=r.get("ok", False),
            payload_valid=r.get("payload_valid", False),
        )

    retryable = [r for r in failed if r.get("retriable")]
    grouped = _group_retries_by_prefix(retryable)

    return {
        "cleaned": cleaned,
        "failed": failed,
        "retryable": retryable,
        "grouped": grouped,
        "duplicates": max(raw_line_count - len(seen), 0),
        "total_lines": raw_line_count,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_clean = Path(args.out_clean) if args.out_clean else input_path.with_suffix(".clean.jsonl")
    out_fail = Path(args.out_failures) if args.out_failures else input_path.with_suffix(".failures.jsonl")
    out_retry = Path(args.out_retry) if args.out_retry else input_path.with_suffix(".retries.jsonl")
    out_retry_by_prefix = (
        Path(args.out_retry_by_prefix)
        if args.out_retry_by_prefix
        else input_path.with_suffix(".retries_by_prefix.jsonl")
    )

    result = normalize_input(input_path, args.mark_context_overflow_as_retriable)

    _write_jsonl(out_clean, result["cleaned"])
    _write_jsonl(out_fail, result["failed"])
    _write_jsonl(out_retry, result["retryable"])
    _write_jsonl(out_retry_by_prefix, result["grouped"])

    print(f"[INFO] input_lines={result['total_lines']}")
    print(f"[INFO] duplicate_ids={result['duplicates']}")
    print(f"[INFO] unique_ids={len(result['cleaned'])}")
    print(f"[INFO] cleaned={out_clean}")
    print(f"[INFO] failures={out_fail}")
    print(f"[INFO] retriable={out_retry}")
    print(f"[INFO] retry_by_prefix={out_retry_by_prefix}")

    if args.prompt_export:
        if not args.out_prompt_retry:
            out_prompt_retry = input_path.with_suffix(".retry_prompts.jsonl")
        else:
            out_prompt_retry = Path(args.out_prompt_retry)

        prompt_records = _load_prompt_map(Path(args.prompt_export))
        retry_ids = {r["id"] for r in result["retryable"]}

        rows: List[Dict[str, Any]] = []
        for rid in sorted(retry_ids):
            rec = prompt_records.get(rid)
            if rec is not None:
                rows.append(rec)

        _write_jsonl(out_prompt_retry, rows)
        print(f"[INFO] retry_prompts={out_prompt_retry}")
        print(f"[INFO] retry_prompt_count={len(rows)}")

        missing = sorted(retry_ids - set(prompt_records.keys()))
        if missing:
            print(f"[WARN] Missing prompt IDs in prompt-export map: count={len(missing)} first={missing[:10]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
