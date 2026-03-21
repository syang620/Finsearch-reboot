from __future__ import annotations

import asyncio
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from evals.llm_judge_contracts import (
    JudgeMode,
    JudgeVerdict,
    LLMJudgeDimensionScores,
    LLMJudgeEvalRow,
    LLMJudgeEvalSummary,
    LLMJudgeEvidenceChunkIds,
    LLMJudgeModelOutput,
)
from llm_client import build_chat_model


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def _load_records(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")

    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if p.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        for line_no, raw in enumerate(text.splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {p}:{line_no}")
            rows.append(obj)
        return rows

    parsed = json.loads(text)
    if isinstance(parsed, list):
        return [dict(item) for item in parsed]
    if isinstance(parsed, dict) and isinstance(parsed.get("rows"), list):
        return [dict(item) for item in parsed["rows"]]
    raise ValueError(f"Unsupported JSON structure in {p}")


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _extract_chunk_id(doc_id: str, fallback_table_id: Any = None) -> Tuple[Optional[str], Optional[int]]:
    raw_doc_id = str(doc_id or "").strip()
    match = re.search(r"::(text|table)::(\d+)$", raw_doc_id)
    if match:
        return match.group(1), int(match.group(2))

    if fallback_table_id is not None:
        try:
            return "table", int(fallback_table_id)
        except Exception:
            return None, None
    return None, None


def _build_joined_rows(
    gold_rows: Sequence[Dict[str, Any]],
    rag_rows: Sequence[Dict[str, Any]],
) -> Tuple[List[Tuple[int, Dict[str, Any], Dict[str, Any]]], List[Dict[str, Any]]]:
    errors: List[Dict[str, Any]] = []
    rag_by_index: Dict[int, Dict[str, Any]] = {}
    rag_by_prompt: Dict[str, Dict[str, Any]] = {}

    for row in rag_rows:
        prompt = _normalize_text(row.get("prompt"))
        if prompt:
            rag_by_prompt[prompt] = row
        try:
            idx = int(row.get("prompt_index"))
        except Exception:
            idx = None
        if idx is not None:
            rag_by_index[idx] = row

    joined: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []
    for pos, gold in enumerate(gold_rows, start=1):
        prompt = _normalize_text(gold.get("prompt"))
        rag = rag_by_index.get(pos)
        if rag is None and prompt:
            rag = rag_by_prompt.get(prompt)
        if rag is None:
            errors.append(
                {
                    "stage": "join",
                    "prompt_index": pos,
                    "prompt": prompt,
                    "error": "No matching RAG row found.",
                }
            )
            continue
        joined.append((pos, dict(gold), dict(rag)))

    return joined, errors


def _build_evidence_items(
    rag_row: Dict[str, Any],
    *,
    max_evidence_chunks: int,
    max_chars_per_chunk: int,
) -> Tuple[List[Dict[str, Any]], LLMJudgeEvidenceChunkIds, List[str]]:
    items: List[Dict[str, Any]] = []
    evidence_ids = LLMJudgeEvidenceChunkIds()
    retrieved_doc_ids: List[str] = []

    for chunk in rag_row.get("retrieved_chunks") or []:
        doc_id = _normalize_text(chunk.get("doc_id"))
        if doc_id:
            retrieved_doc_ids.append(doc_id)

    for item in (rag_row.get("context_items") or [])[: max(0, int(max_evidence_chunks))]:
        kind = _normalize_text(item.get("kind")).lower() or "text"
        source = item.get("source") or {}
        payload = item.get("payload") or {}
        doc_id = _normalize_text(source.get("doc_id"))
        section_path = _normalize_text(source.get("section_path"))
        chunk_kind, chunk_id = _extract_chunk_id(doc_id, source.get("table_id"))

        content = ""
        if kind == "table":
            content = _normalize_text(payload.get("table_markdown"))
        else:
            content = _normalize_text(payload.get("content"))
        if max_chars_per_chunk > 0 and len(content) > max_chars_per_chunk:
            content = content[:max_chars_per_chunk].rstrip() + "\n...[truncated]"

        if chunk_kind == "table" and chunk_id is not None:
            evidence_ids.tables.append(chunk_id)
        elif chunk_kind == "text" and chunk_id is not None:
            evidence_ids.text.append(chunk_id)

        items.append(
            {
                "doc_id": doc_id,
                "kind": kind,
                "chunk_id": chunk_id,
                "section_path": section_path,
                "content": content,
            }
        )

    return items, evidence_ids, retrieved_doc_ids


def _format_evidence_block(evidence_items: Sequence[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for item in evidence_items:
        label = "TABLE" if item.get("kind") == "table" else "TEXT"
        chunk_id = item.get("chunk_id")
        section_path = _normalize_text(item.get("section_path"))
        content = _normalize_text(item.get("content"))
        parts.append(
            "\n".join(
                [
                    f"[{label}_CHUNK_ID={chunk_id}]",
                    f"section_path: {section_path}",
                    content,
                ]
            )
        )
    return "\n\n".join(part for part in parts if part.strip())


def _answer_only_system_prompt() -> str:
    return (
        "You are a strict evaluator for SEC-filing question answering.\n\n"
        "Your task is to compare a candidate answer against a gold answer.\n"
        "Treat the gold answer as the reference for expected content.\n"
        "The candidate may use different wording or order and still be fully correct.\n"
        "Do not reward style, verbosity, or eloquence.\n"
        "Penalize incorrect numbers, wrong directionality, wrong entities, contradictions, or missing key facts.\n"
        "Minor rounding differences are acceptable if they do not change the meaning.\n"
        "If the prompt requires separating fact from inference, check that explicitly.\n"
        "Return JSON only."
    )


def _evidence_based_system_prompt() -> str:
    return (
        "You are a strict evaluator for SEC-filing question answering.\n\n"
        "Your task is to evaluate a candidate answer against a gold answer, using the provided filing evidence as the final source of truth.\n\n"
        "Important rules:\n"
        "1. Judge factual correctness based only on the provided evidence chunks.\n"
        "2. The gold answer is a reference for expected content, not the ultimate authority.\n"
        "3. The candidate answer may use different wording or order and still be fully correct.\n"
        "4. Do not reward style, verbosity, or eloquence.\n"
        "5. Penalize unsupported claims, incorrect numbers, wrong directionality, wrong entities, or omitted key facts.\n"
        "6. Minor rounding differences are acceptable if they do not change the meaning.\n"
        "7. If the prompt requires separating stated facts from inference, check that explicitly.\n"
        "8. Extra information is acceptable only if it is supported by the evidence and does not distract from or contradict the required answer.\n"
        "Return JSON only."
    )


def _score_rules(judge_mode: JudgeMode) -> str:
    if judge_mode == "answer_only":
        return (
            "Scoring requirements you MUST follow exactly:\n"
            "- `dimension_scores.correctness` must be an integer in {0,1,2,3,4}.\n"
            "- `dimension_scores.completeness` must be an integer in {0,1,2,3}.\n"
            "- `dimension_scores.grounding` must be null.\n"
            "- `dimension_scores.inference_handling` must be an integer in {0,1}.\n"
            "- `score` must be an integer equal to correctness + completeness + inference_handling.\n"
            "- In answer_only mode, `score` must be between 0 and 8.\n"
            "- Do NOT use percentages, 100-point scales, 10-point scales, decimals, or normalized scores like 0.85 or 0.5.\n"
            "- Do NOT output floats for any score field.\n"
            "- If a value is uncertain, round to the nearest allowed integer in the rubric range.\n"
        )
    return (
        "Scoring requirements you MUST follow exactly:\n"
        "- `dimension_scores.correctness` must be an integer in {0,1,2,3,4}.\n"
        "- `dimension_scores.completeness` must be an integer in {0,1,2,3}.\n"
        "- `dimension_scores.grounding` must be an integer in {0,1,2}.\n"
        "- `dimension_scores.inference_handling` must be an integer in {0,1}.\n"
        "- `score` must be an integer equal to correctness + completeness + grounding + inference_handling.\n"
        "- In evidence_based mode, `score` must be between 0 and 10.\n"
        "- Do NOT use percentages, 100-point scales, 10-point scales, decimals, or normalized scores like 0.85 or 0.5.\n"
        "- Do NOT output floats for any score field.\n"
        "- If a value is uncertain, round to the nearest allowed integer in the rubric range.\n"
    )


def _json_schema_instructions(judge_mode: JudgeMode) -> str:
    grounding_value = "null" if judge_mode == "answer_only" else "0"
    return (
        "{\n"
        '  "verdict": "correct" | "partially_correct" | "incorrect",\n'
        '  "score": 0,\n'
        '  "dimension_scores": {\n'
        '    "correctness": 0,\n'
        '    "completeness": 0,\n'
        f'    "grounding": {grounding_value},\n'
        '    "inference_handling": 0\n'
        "  },\n"
        '  "matched_key_points": ["..."],\n'
        '  "missed_key_points": ["..."],\n'
        '  "unsupported_or_wrong_claims": ["..."],\n'
        '  "used_evidence_chunk_ids": {\n'
        '    "text": [],\n'
        '    "tables": []\n'
        "  },\n"
        '  "explanation": "brief explanation"\n'
        "}"
    )


def _build_user_prompt(
    *,
    judge_mode: JudgeMode,
    prompt: str,
    gold_answer: str,
    candidate_answer: str,
    evidence_block: str,
) -> str:
    extra = ""
    if judge_mode == "answer_only":
        extra = (
            "Use only the prompt, gold answer, and candidate answer.\n"
            "Set dimension_scores.grounding to null.\n"
            "Set used_evidence_chunk_ids.text and used_evidence_chunk_ids.tables to empty lists.\n"
            "Score only the dimensions that can be judged without evidence.\n"
        )
    else:
        extra = (
            "Use the evidence chunks as the final source of truth.\n"
            "When citing used evidence chunk ids, only use ids that appear in the provided evidence.\n"
        )

    sections = [
        extra.strip(),
        _score_rules(judge_mode).strip(),
        "",
        "Prompt:",
        prompt,
        "",
        "Gold answer:",
        gold_answer,
        "",
        "Candidate answer:",
        candidate_answer or "[empty]",
        "",
    ]
    if judge_mode == "evidence_based":
        sections.extend(
            [
                "Evidence chunks:",
                evidence_block or "[no evidence provided]",
                "",
            ]
        )
    sections.extend(
        [
            "Return JSON only with this schema:",
            _json_schema_instructions(judge_mode),
        ]
    )
    return "\n".join(sections)


def _extract_json_payload(raw_text: str) -> Dict[str, Any]:
    text = str(raw_text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[start : end + 1])


def _score_max_for_mode(judge_mode: JudgeMode) -> int:
    return 8 if judge_mode == "answer_only" else 10


def _derive_verdict(score: int, score_max: int, dimension_scores: LLMJudgeDimensionScores) -> JudgeVerdict:
    ratio = (float(score) / float(score_max)) if score_max > 0 else 0.0
    if ratio >= 0.85 and int(dimension_scores.correctness) >= 3:
        return "correct"
    if ratio >= 0.45 and int(dimension_scores.correctness) >= 1:
        return "partially_correct"
    return "incorrect"


def _normalize_model_output(judge_mode: JudgeMode, model_output: LLMJudgeModelOutput) -> Tuple[int, int, JudgeVerdict]:
    dims = model_output.dimension_scores
    if judge_mode == "answer_only":
        dims.grounding = None
    grounding_value = int(dims.grounding or 0)
    score = int(dims.correctness) + int(dims.completeness) + grounding_value + int(dims.inference_handling)
    score_max = _score_max_for_mode(judge_mode)
    verdict = _derive_verdict(score, score_max, dims)
    return score, score_max, verdict


async def _invoke_judge_json(
    *,
    judge_model: Any,
    judge_mode: JudgeMode,
    prompt: str,
    gold_answer: str,
    candidate_answer: str,
    evidence_block: str,
) -> Tuple[LLMJudgeModelOutput, str]:
    messages = [
        SystemMessage(
            content=_answer_only_system_prompt() if judge_mode == "answer_only" else _evidence_based_system_prompt()
        ),
        HumanMessage(
            content=_build_user_prompt(
                judge_mode=judge_mode,
                prompt=prompt,
                gold_answer=gold_answer,
                candidate_answer=candidate_answer,
                evidence_block=evidence_block,
            )
        ),
    ]

    raw_response = await judge_model.ainvoke(messages)
    raw_text = getattr(raw_response, "content", raw_response)
    try:
        parsed = _extract_json_payload(str(raw_text or ""))
        return LLMJudgeModelOutput.model_validate(parsed), str(raw_text or "")
    except Exception:
        repair_messages = [
            SystemMessage(
                content=(
                    "Convert the following invalid model output into valid JSON only. "
                    "Do not add commentary. Preserve the substantive judgment. "
                    "You MUST use the exact rubric ranges and integer-only scoring rules."
                )
            ),
            HumanMessage(
                content=(
                    "Re-emit the judgment using these exact scoring rules:\n"
                    f"{_score_rules(judge_mode)}\n\n"
                    "Schema:\n"
                    f"{_json_schema_instructions(judge_mode)}\n\n"
                    "Invalid output to repair:\n"
                    f"{raw_text}"
                )
            ),
        ]
        repaired_response = await judge_model.ainvoke(repair_messages)
        repaired_text = getattr(repaired_response, "content", repaired_response)
        parsed = _extract_json_payload(str(repaired_text or ""))
        return LLMJudgeModelOutput.model_validate(parsed), str(repaired_text or "")


async def _run_judging_async(
    *,
    joined_rows: Sequence[Tuple[int, Dict[str, Any], Dict[str, Any]]],
    judge_model_name: str,
    judge_mode: JudgeMode,
    max_evidence_chunks: int,
    max_chars_per_chunk: int,
    fail_fast: bool,
    base_url: Optional[str],
    timeout: float,
) -> Tuple[List[LLMJudgeEvalRow], List[Dict[str, Any]]]:
    llm = build_chat_model(model=judge_model_name, temperature=0, base_url=base_url, timeout=timeout)
    rows: List[LLMJudgeEvalRow] = []
    errors: List[Dict[str, Any]] = []

    for prompt_index, gold_row, rag_row in joined_rows:
        prompt = _normalize_text(gold_row.get("prompt") or rag_row.get("prompt"))
        gold_answer = _normalize_text(gold_row.get("gold_answer") or rag_row.get("gold_answer"))
        candidate_answer = _normalize_text(rag_row.get("final_answer"))

        evidence_items, evidence_ids, retrieved_doc_ids = _build_evidence_items(
            rag_row,
            max_evidence_chunks=max_evidence_chunks,
            max_chars_per_chunk=max_chars_per_chunk,
        )
        evidence_block = _format_evidence_block(evidence_items) if judge_mode == "evidence_based" else ""

        started = time.perf_counter()
        try:
            model_output, raw_output = await _invoke_judge_json(
                judge_model=llm,
                judge_mode=judge_mode,
                prompt=prompt,
                gold_answer=gold_answer,
                candidate_answer=candidate_answer,
                evidence_block=evidence_block,
            )
            score, score_max, verdict = _normalize_model_output(judge_mode, model_output)
            row = LLMJudgeEvalRow(
                id=f"judge_{prompt_index}",
                prompt_index=prompt_index,
                prompt=prompt,
                judge_mode=judge_mode,
                judge_model=judge_model_name,
                retrieval_ok=rag_row.get("retrieval_ok"),
                analyst_ok=rag_row.get("analyst_ok"),
                gold_answer=gold_answer,
                candidate_answer=candidate_answer,
                evidence_provided=bool(judge_mode == "evidence_based" and evidence_items),
                score=score,
                score_max=score_max,
                verdict=verdict,
                dimension_scores=model_output.dimension_scores,
                matched_key_points=list(model_output.matched_key_points or []),
                missed_key_points=list(model_output.missed_key_points or []),
                unsupported_or_wrong_claims=list(model_output.unsupported_or_wrong_claims or []),
                used_evidence_chunk_ids=model_output.used_evidence_chunk_ids,
                evidence_chunk_ids=evidence_ids,
                retrieved_chunk_doc_ids=retrieved_doc_ids,
                trace={
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "raw_output": raw_output,
                },
            )
            rows.append(row)
        except Exception as exc:
            error = {
                "stage": "judge",
                "prompt_index": prompt_index,
                "prompt": prompt,
                "error": str(exc),
            }
            errors.append(error)
            rows.append(
                LLMJudgeEvalRow(
                    id=f"judge_{prompt_index}",
                    prompt_index=prompt_index,
                    prompt=prompt,
                    judge_mode=judge_mode,
                    judge_model=judge_model_name,
                    retrieval_ok=rag_row.get("retrieval_ok"),
                    analyst_ok=rag_row.get("analyst_ok"),
                    gold_answer=gold_answer,
                    candidate_answer=candidate_answer,
                    evidence_provided=bool(judge_mode == "evidence_based" and evidence_items),
                    score=0,
                    score_max=_score_max_for_mode(judge_mode),
                    verdict="incorrect",
                    dimension_scores=LLMJudgeDimensionScores(
                        correctness=0,
                        completeness=0,
                        grounding=None if judge_mode == "answer_only" else 0,
                        inference_handling=0,
                    ),
                    evidence_chunk_ids=evidence_ids,
                    retrieved_chunk_doc_ids=retrieved_doc_ids,
                    error=str(exc),
                    trace={"latency_ms": int((time.perf_counter() - started) * 1000)},
                )
            )
            if fail_fast:
                break

    return rows, errors


def _build_summary(
    *,
    rows: Sequence[LLMJudgeEvalRow],
    errors: Sequence[Dict[str, Any]],
    judge_mode: JudgeMode,
    judge_model: str,
    config: Dict[str, Any],
) -> LLMJudgeEvalSummary:
    valid_rows = [row for row in rows if not row.error]
    verdict_counts: Dict[str, int] = {"correct": 0, "partially_correct": 0, "incorrect": 0}
    for row in valid_rows:
        verdict_counts[row.verdict] = verdict_counts.get(row.verdict, 0) + 1

    score_values = [float(row.score) for row in valid_rows]
    score_ratios = [
        (float(row.score) / float(row.score_max))
        for row in valid_rows
        if row.score_max and not math.isnan(float(row.score_max))
    ]
    correctness_values = [float(row.dimension_scores.correctness) for row in valid_rows]
    completeness_values = [float(row.dimension_scores.completeness) for row in valid_rows]
    inference_values = [float(row.dimension_scores.inference_handling) for row in valid_rows]
    grounding_values = [
        float(row.dimension_scores.grounding)
        for row in valid_rows
        if row.dimension_scores.grounding is not None
    ]

    dimension_means = {
        "correctness": _mean(correctness_values),
        "completeness": _mean(completeness_values),
        "inference_handling": _mean(inference_values),
    }
    if grounding_values:
        dimension_means["grounding"] = _mean(grounding_values)

    return LLMJudgeEvalSummary(
        num_queries=len(rows),
        num_judged=len(valid_rows),
        num_failures=len(errors),
        judge_mode=judge_mode,
        judge_model=judge_model,
        verdict_counts=verdict_counts,
        mean_score=_mean(score_values),
        mean_score_ratio=_mean(score_ratios),
        dimension_means=dimension_means,
        config=config,
    )


def run_llm_judge_eval(
    *,
    gold_path: str | Path,
    rag_path: str | Path,
    out_dir: str | Path,
    judge_model: str,
    judge_mode: JudgeMode = "answer_only",
    max_evidence_chunks: int = 6,
    max_chars_per_chunk: int = 3000,
    limit: Optional[int] = None,
    fail_fast: bool = False,
    base_url: Optional[str] = None,
    timeout: float = 120.0,
) -> Tuple[LLMJudgeEvalSummary, List[LLMJudgeEvalRow], List[Dict[str, Any]]]:
    gold_rows = _load_records(gold_path)
    rag_rows = _load_records(rag_path)
    joined_rows, join_errors = _build_joined_rows(gold_rows, rag_rows)
    if limit is not None and limit >= 0:
        joined_rows = joined_rows[: int(limit)]

    rows, judge_errors = asyncio.run(
        _run_judging_async(
            joined_rows=joined_rows,
            judge_model_name=judge_model,
            judge_mode=judge_mode,
            max_evidence_chunks=max_evidence_chunks,
            max_chars_per_chunk=max_chars_per_chunk,
            fail_fast=fail_fast,
            base_url=base_url,
            timeout=timeout,
        )
    )
    errors = list(join_errors) + list(judge_errors)

    summary = _build_summary(
        rows=rows,
        errors=errors,
        judge_mode=judge_mode,
        judge_model=judge_model,
        config={
            "gold_path": str(Path(gold_path).resolve()),
            "rag_path": str(Path(rag_path).resolve()),
            "max_evidence_chunks": int(max_evidence_chunks),
            "max_chars_per_chunk": int(max_chars_per_chunk),
            "limit": limit,
        },
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_path / "per_query.jsonl", [row.model_dump(mode="json") for row in rows])
    (out_path / "summary.json").write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_jsonl(out_path / "errors.jsonl", errors)
    return summary, rows, errors
