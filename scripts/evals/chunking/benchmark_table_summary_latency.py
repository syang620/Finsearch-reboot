#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from gradio_client import Client
except Exception:  # pragma: no cover - handled at runtime
    Client = None  # type: ignore[assignment]

from llm_client import dashscope_chat_completion, is_qwen_chat_model
from ingestion.tables_summarizer import (
    SYSTEM_PROMPT,
    build_user_prompt,
    load_table_chunks_for_prefix,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark table-chunk summarization latency between local Ollama and "
            "a Gradio-hosted endpoint while keeping prompt/model/hyperparameters fixed."
        ),
    )
    p.add_argument("--chunks-dir", default="data/chunked")
    p.add_argument("--prefix", default="AAPL_10-K_2024")
    p.add_argument(
        "--table-index",
        type=int,
        default=None,
        help=(
            "Explicit table index. If omitted, script tries to find the first table "
            "whose section title contains --target-section."
        ),
    )
    p.add_argument("--target-section", default="CONSOLIDATED BALANCE SHEETS")
    p.add_argument("--fallback-table-index", type=int, default=0)

    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--timeout-s", type=int, default=180)

    p.add_argument("--local-api-url", default="http://localhost:11434/api/generate")
    p.add_argument("--colab-url", required=True)
    p.add_argument("--colab-api-name", default="/generate")

    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--warmup-runs", type=int, default=1)

    p.add_argument("--out-dir", default="artifacts/benchmarks/table_summary_latency")
    return p


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    arr = sorted(values)
    rank = int(math.ceil((p / 100.0) * len(arr))) - 1
    rank = max(0, min(rank, len(arr) - 1))
    return arr[rank]


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.median(values))


def _as_float_or_none(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    return float(v)


def _resolve_table_index(
    table_chunks: List[Dict[str, Any]],
    explicit_index: Optional[int],
    target_section: str,
    fallback_index: int,
) -> int:
    if not table_chunks:
        raise ValueError("No table chunks found for selected prefix.")

    if explicit_index is not None:
        if explicit_index < 0 or explicit_index >= len(table_chunks):
            raise ValueError(
                f"--table-index {explicit_index} out of range [0, {len(table_chunks) - 1}]",
            )
        return explicit_index

    target = target_section.strip().upper()
    for idx, chunk in enumerate(table_chunks):
        section_title = str(chunk.get("section_title") or "").upper()
        if target and target in section_title:
            return idx

    if fallback_index < 0 or fallback_index >= len(table_chunks):
        raise ValueError(
            f"--fallback-table-index {fallback_index} out of range [0, {len(table_chunks) - 1}]",
        )
    return fallback_index


def _build_full_prompt(table_chunk: Dict[str, Any]) -> str:
    user_prompt = build_user_prompt(table_chunk)
    return SYSTEM_PROMPT.strip() + "\n\n---\n\n" + user_prompt.strip()


def _call_local_ollama(
    *,
    api_url: str,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    timeout_s: int,
) -> Tuple[str, float]:
    if is_qwen_chat_model(model):
        t0 = time.perf_counter()
        obj = dashscope_chat_completion(
            [{"role": "user", "content": prompt}],
            model=model,
            options={
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_new_tokens,
                "response_format": {"type": "json_object"},
            },
            timeout=timeout_s,
        )
        choices = obj.get("choices") or []
        message = choices[0].get("message") if choices else {}
        text = (message or {}).get("content") or ""
        if not isinstance(text, str):
            text = json.dumps(text, ensure_ascii=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return text.strip(), elapsed_ms

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_new_tokens,
        },
    }
    t0 = time.perf_counter()
    resp = requests.post(api_url, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    obj = resp.json()
    text = (obj.get("response") or "").strip()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return text, elapsed_ms


def _call_gradio_colab(
    *,
    client: Any,
    api_name: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> Tuple[str, float]:
    t0 = time.perf_counter()
    out = client.predict(
        prompt,
        temperature,
        top_p,
        max_new_tokens,
        api_name=api_name,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if isinstance(out, str):
        text = out.strip()
    else:
        text = json.dumps(out, ensure_ascii=False)
    return text, elapsed_ms


def _validate_json_schema(raw_text: str) -> Tuple[bool, bool, Optional[str], Optional[int]]:
    try:
        obj = json.loads(raw_text)
    except Exception as exc:
        return False, False, f"json_parse_error: {exc}", None

    if not isinstance(obj, dict):
        return True, False, "parsed_json_not_object", None

    table_summary = obj.get("table_summary")
    row_summaries = obj.get("row_summaries")

    schema_ok = isinstance(table_summary, str) and isinstance(row_summaries, list)
    row_count = len(row_summaries) if isinstance(row_summaries, list) else None
    if schema_ok:
        return True, True, None, row_count
    return True, False, "schema_missing_required_keys", row_count


def _run_one(
    *,
    endpoint: str,
    prompt: str,
    model: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    timeout_s: int,
    local_api_url: str,
    colab_client: Any,
    colab_api_name: str,
) -> Dict[str, Any]:
    started_at = _now_iso()
    raw_text = ""
    latency_ms: Optional[float] = None
    request_ok = False
    parse_ok = False
    schema_ok = False
    row_summaries_count: Optional[int] = None
    err: Optional[str] = None

    try:
        if endpoint == "local":
            raw_text, latency_ms = _call_local_ollama(
                api_url=local_api_url,
                model=model,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                timeout_s=timeout_s,
            )
        elif endpoint == "colab":
            raw_text, latency_ms = _call_gradio_colab(
                client=colab_client,
                api_name=colab_api_name,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
        else:
            raise ValueError(f"Unsupported endpoint: {endpoint}")
        request_ok = True
    except Exception as exc:
        err = f"request_error: {exc}"

    if request_ok:
        parse_ok, schema_ok, validation_err, row_summaries_count = _validate_json_schema(raw_text)
        if validation_err is not None:
            err = validation_err

    return {
        "timestamp_utc": started_at,
        "endpoint": endpoint,
        "latency_ms": _as_float_or_none(latency_ms),
        "request_ok": bool(request_ok),
        "parse_ok": bool(parse_ok),
        "schema_ok": bool(schema_ok),
        "success": bool(schema_ok),
        "response_chars": len(raw_text),
        "row_summaries_count": row_summaries_count,
        "error": err,
    }


def _aggregate(records: List[Dict[str, Any]], endpoint: str) -> Dict[str, Any]:
    rows = [r for r in records if r.get("endpoint") == endpoint]
    latencies = [float(r["latency_ms"]) for r in rows if r.get("latency_ms") is not None]
    successes = [r for r in rows if bool(r.get("success"))]
    parse_ok = [r for r in rows if bool(r.get("parse_ok"))]
    request_ok = [r for r in rows if bool(r.get("request_ok"))]

    n_total = len(rows)
    n_request_ok = len(request_ok)
    n_parse_ok = len(parse_ok)
    n_success = len(successes)

    return {
        "n_total": n_total,
        "n_request_ok": n_request_ok,
        "n_parse_ok": n_parse_ok,
        "n_success": n_success,
        "request_ok_rate": (n_request_ok / n_total) if n_total else None,
        "parse_ok_rate": (n_parse_ok / n_total) if n_total else None,
        "success_rate": (n_success / n_total) if n_total else None,
        "latency_ms_min": _as_float_or_none(min(latencies)) if latencies else None,
        "latency_ms_median": _as_float_or_none(_median(latencies)),
        "latency_ms_p95": _as_float_or_none(_percentile(latencies, 95.0)),
        "latency_ms_max": _as_float_or_none(max(latencies)) if latencies else None,
    }


def main() -> None:
    args = build_parser().parse_args()

    if Client is None:
        raise RuntimeError(
            "gradio_client is not installed. Install it via: python -m pip install gradio_client",
        )

    chunks_dir = Path(args.chunks_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_run_path = out_dir / "per_run.jsonl"
    summary_path = out_dir / "summary.json"

    table_chunks = load_table_chunks_for_prefix(args.prefix, chunks_dir)
    table_index = _resolve_table_index(
        table_chunks=table_chunks,
        explicit_index=args.table_index,
        target_section=args.target_section,
        fallback_index=args.fallback_table_index,
    )
    table_chunk = table_chunks[table_index]
    prompt = _build_full_prompt(table_chunk)

    colab_client = Client(args.colab_url)

    # Warmups are excluded from benchmark output.
    for _ in range(int(args.warmup_runs)):
        _run_one(
            endpoint="local",
            prompt=prompt,
            model=args.model,
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            max_new_tokens=int(args.max_new_tokens),
            timeout_s=int(args.timeout_s),
            local_api_url=args.local_api_url,
            colab_client=colab_client,
            colab_api_name=args.colab_api_name,
        )
        _run_one(
            endpoint="colab",
            prompt=prompt,
            model=args.model,
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            max_new_tokens=int(args.max_new_tokens),
            timeout_s=int(args.timeout_s),
            local_api_url=args.local_api_url,
            colab_client=colab_client,
            colab_api_name=args.colab_api_name,
        )

    records: List[Dict[str, Any]] = []
    with per_run_path.open("w", encoding="utf-8") as f:
        for run_idx in range(int(args.runs)):
            order = ["local", "colab"] if (run_idx % 2 == 0) else ["colab", "local"]
            for order_idx, endpoint in enumerate(order):
                rec = _run_one(
                    endpoint=endpoint,
                    prompt=prompt,
                    model=args.model,
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    max_new_tokens=int(args.max_new_tokens),
                    timeout_s=int(args.timeout_s),
                    local_api_url=args.local_api_url,
                    colab_client=colab_client,
                    colab_api_name=args.colab_api_name,
                )
                rec["run_index"] = run_idx
                rec["order_index"] = order_idx
                rec["order"] = order
                records.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(
                    f"[run={run_idx} endpoint={endpoint}] latency_ms={rec.get('latency_ms')} "
                    f"success={rec.get('success')} error={rec.get('error')}",
                )

    local_stats = _aggregate(records, "local")
    colab_stats = _aggregate(records, "colab")

    local_median = local_stats.get("latency_ms_median")
    colab_median = colab_stats.get("latency_ms_median")
    speed_ratio = None
    if isinstance(local_median, float) and isinstance(colab_median, float) and colab_median > 0:
        speed_ratio = local_median / colab_median

    summary = {
        "timestamp_utc": _now_iso(),
        "config": {
            "chunks_dir": str(chunks_dir),
            "prefix": args.prefix,
            "table_index": table_index,
            "section_title": table_chunk.get("section_title"),
            "model": args.model,
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "max_new_tokens": int(args.max_new_tokens),
            "timeout_s": int(args.timeout_s),
            "local_api_url": args.local_api_url,
            "colab_url": args.colab_url,
            "colab_api_name": args.colab_api_name,
            "runs": int(args.runs),
            "warmup_runs": int(args.warmup_runs),
        },
        "results": {
            "local": local_stats,
            "colab": colab_stats,
            "local_over_colab_median_ratio": speed_ratio,
        },
        "artifacts": {
            "per_run_jsonl": str(per_run_path.resolve()),
            "summary_json": str(summary_path.resolve()),
        },
    }

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"output_dir={out_dir.resolve()}")


if __name__ == "__main__":
    main()
