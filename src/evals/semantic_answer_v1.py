"""Independent evaluation contracts, not production grounding or entailment rules."""
from __future__ import annotations

from collections import Counter
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re

STRATA = {"structured_numeric", "narrative", "hybrid", "comparison", "calculator",
          "multiple_claims", "difficult_attribution", "plausible_wrong_evidence", "insufficient_data"}
SUPPORT = {"fully_supported", "partially_supported", "unsupported"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip(): raise ValueError("Blank JSONL record")
        row = json.loads(line)
        if not isinstance(row, dict): raise ValueError("Expected JSON object")
        rows.append(row)
    return rows


def rate(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None}


def table_cell(content, spec):
    matches = []
    for line in content.splitlines():
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if cells and cells[0] == spec["row_label"]:
            value = cells[spec["value_column"]].replace("$", "").replace(",", "").replace("%", "")
            matches.append(Decimal(value) * Decimal(10) ** spec["scale"])
    if len(matches) != 1: raise ValueError("Ambiguous or absent source table cell")
    return matches[0]


def validate_cases(cases, facts, docs):
    for rows, key in ((cases, "id"), (facts, "fact_id"), (docs, "id")):
        ids = [r[key] for r in rows]
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("Duplicate/empty identity")
    catalog = {f["fact_id"]: f for f in facts}
    corpus = {d["id"]: d for d in docs}
    for case in cases:
        if case["stratum"] not in STRATA or not case["user_query"].strip(): raise ValueError("Invalid case")
        if case["expected_answerability"] not in {"answerable", "insufficient_data"}: raise ValueError("Invalid answerability")
        claims = case["required_claims"]
        if any(not c.get("claim_id") for c in claims) or len({c["claim_id"] for c in claims}) != len(claims): raise ValueError("Duplicate/empty gold claim")
        if case["expected_answerability"] == "answerable" and not claims: raise ValueError("Answerable case has no gold claims")
        if case["expected_answerability"] == "insufficient_data" and (claims or not case.get("answerability_reason") or not case.get("answerability_sources")):
            raise ValueError("Unjustified insufficient-data case")
        for claim in claims:
            if not claim["requirement"] or not claim["sources"]: raise ValueError("Missing gold requirement/source")
            number = claim.get("numeric")
            if number:
                if any(type(number[k]) not in (int, float) or not math.isfinite(number[k]) for k in ("value", "absolute_tolerance")) or number["absolute_tolerance"] < 0:
                    raise ValueError("Invalid numeric gold")
                if claim["claim_type"] == "structured_numeric":
                    if any(s["value"] != number["value"] or s["metric_id"] != number["metric_id"] or s["ticker"] != number["ticker"] or s["fact_fiscal_year"] != number["fiscal_year"] for s in claim["sources"]):
                        raise ValueError("Numeric label/source disagreement")
                elif claim["claim_type"] == "calculation":
                    previous, current = claim["sources"]
                    expected = (Decimal(current["value"]) - Decimal(previous["value"])) / Decimal(previous["value"]) * 100
                    if abs(float(expected) - number["value"]) > 1e-12: raise ValueError("Calculation gold mismatch")
                elif claim["claim_type"] == "kb_numeric":
                    for source in claim["sources"]:
                        if table_cell(corpus[source["evidence_id"]]["content"], number["source_cell"]) != Decimal(str(number["value"])):
                            raise ValueError("Table numeric label/source disagreement")
        sources = [s for c in claims for s in c["sources"]] + case.get("known_wrong_evidence", []) + case.get("answerability_sources", [])
        for source in sources:
            if source["kind"] == "inline_xbrl":
                expected = catalog.get(source["fact_id"])
                if expected is None or any(source.get(k) != v for k, v in expected.items()): raise ValueError("Unknown/modified inline source")
            elif source["kind"] == "kb":
                doc = corpus.get(source["evidence_id"])
                if doc is None: raise ValueError("Missing gold context ID")
                if source["content_sha256"] != sha_text(doc["content"]): raise ValueError("Gold content hash mismatch")
                if any(source[k] != doc["metadata"][k] for k in ("ticker", "fiscal_year", "form_type", "source_html", "source_sha256", "section_path")):
                    raise ValueError("Gold source metadata mismatch")
                for span in source["spans"]:
                    if not 0 <= span["start"] < span["end"] <= len(doc["content"]) or doc["content"][span["start"]:span["end"]] != span["quote"] or span["anchor"] not in span["quote"]:
                        raise ValueError("Invalid gold source span")
            else: raise ValueError("Unknown source kind")
        if case["stratum"] == "plausible_wrong_evidence" and not case.get("known_wrong_evidence"):
            raise ValueError("Missing explicit wrong-evidence example")
    return {"cases": len(cases), "required_claims": sum(len(c["required_claims"]) for c in cases),
            "answerable": sum(c["expected_answerability"] == "answerable" for c in cases),
            "insufficient_data": sum(c["expected_answerability"] == "insufficient_data" for c in cases),
            "strata": dict(sorted(Counter(c["stratum"] for c in cases).items())),
            "audit_cases": sum(c["audit_selected"] for c in cases)}


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def load_dataset(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    for name, digest in manifest["files_sha256"].items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or sha(root / path) != digest: raise ValueError("Frozen dataset hash mismatch")
    source_manifest = json.loads((root / "source_manifest.json").read_text())
    corpus_path = Path(source_manifest["corpus_path"]) / "corpus.jsonl"
    if sha(corpus_path) != source_manifest["corpus_sha256"]: raise ValueError("Corpus changed")
    for source in source_manifest["sources"]:
        path = Path(source["source_html"])
        if path.is_absolute() or ".." in path.parts or sha(path) != source["source_sha256"]: raise ValueError("Filing changed")
        if sha(root / source["table_sidecar"]) != source["table_sha256"]: raise ValueError("Hydration sidecar changed")
    cases, facts, docs = read_jsonl(root / "queries.jsonl"), read_jsonl(root / "numeric_source_facts.jsonl"), read_jsonl(corpus_path)
    report = validate_cases(cases, facts, docs)
    if report != manifest["counts"]: raise ValueError("Manifest counts mismatch")
    return cases, report


def visible_contexts(output):
    analyst = output.get("analyst") or {}
    packet = (output.get("evaluation_trace") or {}).get("analyst_packet") or {}
    trace = analyst.get("trace") or {}
    allowed = trace.get("analyst_visible_context_ids")
    items = packet.get("context_items") or []
    if allowed is None:
        limit = trace.get("context_item_limit")
        if type(limit) is not int: return {}
        items = items[:max(0, limit)]
    else:
        items = [c for c in items if c.get("context_id") in allowed]
    ids = [c["context_id"] for c in items]
    if len(ids) != len(set(ids)): raise ValueError("Duplicate analyst-visible context IDs")
    return {c["context_id"]: c for c in items}


def evidence_text(context):
    if context.get("kind") == "structured_fact":
        fact = context.get("structured_fact") or {}
        fields = ("metric_id", "metric_label", "status", "value", "unit", "ticker", "fiscal_year", "form_type", "accession_number", "report_date", "filed_date", "source_url", "components", "missing_component_groups")
        shown = {k: fact.get(k) for k in fields}
        if fact.get("start_date") is not None: shown["start_date"] = fact["start_date"]
        return json.dumps(shown, sort_keys=True)
    payload = context.get("payload") or {}
    content = str(payload.get("table_markdown") or payload.get("content") or payload.get("text") or "")
    # Independent transcription of the existing 12,000-character display
    # boundary, not an import of runtime grounding/validation or entailment.
    if len(content) <= 12000: return content
    lines, used = [], 0
    for line in content.splitlines():
        extra = len(line) + bool(lines)
        if used + extra > 12000:
            if not lines: return line[:12000].rstrip() + "\n... [truncated] ..."
            break
        lines.append(line); used += extra
    return "\n".join(lines + ["... [truncated] ..."])


def numeric_mentions(text, display_unit):
    """Normalize explicit/implied query units; not a semantic role parser."""
    text = text.replace("−", "-")
    pattern = r"(?<![\w.])([+-]?\d[\d,]*(?:\.\d+)?)\s*(trillion|billion|million|thousand|%|percent|bn|mn)?"
    values = []
    for match in re.finditer(pattern, text, re.I):
        raw, unit = match.groups()
        value = Decimal(raw.replace(",", ""))
        unit = (unit or "").lower()
        if unit in {"%", "percent"} and display_unit != "percent": continue
        if display_unit == "percent" and unit not in {"%", "percent", ""}: continue
        if not unit and value == int(value) and 1900 <= value <= 2100: continue
        scale = {"trillion": 12, "billion": 9, "bn": 9, "million": 6, "mn": 6, "thousand": 3}.get(unit)
        if scale is not None:
            value *= Decimal(10) ** scale
            if display_unit == "thousand_square_feet": value /= 1000
        elif not unit and display_unit == "USD_millions": value *= 1000000
        values.append(float(value))
    return values


def source_bound(context, gold):
    if gold["claim_type"] == "kb_numeric":
        doc_id = (context.get("source") or {}).get("doc_id") or (context.get("payload") or {}).get("doc_id")
        if context.get("kind") not in {"text", "table"} or doc_id not in {s["evidence_id"] for s in gold["sources"]}: return False
        try: return table_cell(evidence_text(context), gold["numeric"]["source_cell"]) == Decimal(str(gold["numeric"]["value"]))
        except (ValueError, IndexError, ArithmeticError): return False
    fact = context.get("structured_fact") or {}
    number = gold["numeric"]
    periods = gold["sources"]
    return (context.get("kind") == "structured_fact" and fact.get("status") == "ok"
        and all(fact.get(k) == number[k] for k in ("metric_id", "ticker", "fiscal_year", "unit"))
        and any(all(fact.get(k) == s.get(k) for k in ("form_type", "report_date", "start_date")) for s in periods)
        and type(fact.get("value")) in (int, float) and math.isfinite(fact["value"])
        and abs(fact["value"] - number["value"]) <= number["absolute_tolerance"])


def numeric_requirement(gold, claims, contexts, analyst):
    number = gold["numeric"]
    kind = gold["claim_type"]
    for claim in claims:
        if claim.get("claim_type") != kind: continue
        if kind == "structured_numeric" and claim.get("metric_id") != number["metric_id"]: continue
        refs = [contexts[i] for i in claim.get("context_ids", []) if i in contexts]
        values = numeric_mentions(claim.get("text", ""), number["requested_display_unit"])
        if not any(abs(v - number["value"]) <= number["absolute_tolerance"] for v in values): continue
        if kind == "calculation":
            computation = analyst.get("computation") or {}
            result = computation.get("result")
            if type(result) not in (int, float) or not math.isfinite(result): continue
            # A fraction-to-percent conversion is unit normalization, not re-computation.
            if not any(abs(v - number["value"]) <= number["absolute_tolerance"] for v in (result, result * 100)): continue
            if not (analyst.get("trace") or {}).get("used_financial_evaluator"): continue
            inputs = [{"claim_type": "structured_numeric", "sources": [s], "numeric": {
                "metric_id": s["metric_id"], "ticker": s["ticker"], "fiscal_year": s["fact_fiscal_year"],
                "value": s["value"], "unit": s["unit"], "absolute_tolerance": 0}} for s in gold["sources"]]
            if all(any(source_bound(c, operand) for c in refs) for operand in inputs): return True
        elif any(source_bound(c, gold) for c in refs): return True
    return False


def deterministic_case(case, output):
    analyst = output.get("analyst") or {}
    claims = analyst.get("claims") or []
    contexts = visible_contexts(output)
    statuses = []
    total_refs = valid_refs = covered = structured_n = structured_ok = kb_n = kb_ok = 0
    for claim in claims:
        refs = list(dict.fromkeys(claim.get("context_ids") or []))
        valid = [r for r in refs if r in contexts]
        total_refs += len(refs); valid_refs += len(valid); covered += bool(refs)
        kind = claim.get("claim_type")
        type_ok = True
        if kind == "structured_numeric":
            structured_n += 1
            type_ok = bool(claim.get("metric_id")) and any(contexts[r].get("kind") == "structured_fact" and (contexts[r].get("structured_fact") or {}).get("metric_id") == claim["metric_id"] for r in valid)
            structured_ok += type_ok
        elif kind in {"narrative", "attribution", "kb_numeric"}:
            kb_n += 1
            type_ok = any(contexts[r].get("kind") in {"text", "table"} and evidence_text(contexts[r]).strip() for r in valid)
            kb_ok += type_ok
        flags = []
        if not refs: flags.append("missing_citation")
        if len(valid) != len(refs): flags.append("invalid_context_id")
        if not type_ok: flags.append("evidence_type_mismatch")
        statuses.append({"claim_id": claim["claim_id"], "flags": flags})
    checks = [{"claim_id": g["claim_id"], "matched": numeric_requirement(g, claims, contexts, analyst)} for g in case["required_claims"] if g.get("numeric")]
    ok = bool(analyst.get("ok")) and analyst.get("status") in {"ok", "insufficient_data"}
    expected_insufficient = case["expected_answerability"] == "insufficient_data"
    insufficient_correct = ok and analyst.get("status") == "insufficient_data" and not claims
    return {"case_id": case["id"], "stratum": case["stratum"], "answer_emitted": ok,
        "status": analyst.get("status"), "orchestrator_status": output.get("status"),
        "claims": len(claims), "claim_checks": statuses, "numeric_checks": checks,
        "claim_citation_coverage": rate(covered, len(claims)), "valid_context_id_rate": rate(valid_refs, total_refs),
        "structured_evidence_compatibility": rate(structured_ok, structured_n),
        "kb_evidence_compatibility": rate(kb_ok, kb_n),
        "numeric_consistency": rate(sum(c["matched"] for c in checks), len(checks)),
        "unsupported_claim_flags": sum(bool(c["flags"]) for c in statuses),
        "expected_insufficient": expected_insufficient,
        "insufficient_data_correct": insufficient_correct if expected_insufficient else None,
        "answerability_correct": insufficient_correct if expected_insufficient else ok and analyst.get("status") == "ok" and bool(claims),
        "calculator_required": bool(case.get("calculator_required")),
        "calculator_used": bool((analyst.get("trace") or {}).get("used_financial_evaluator")),
        "latency_ms": (output.get("orchestrator_trace") or {}).get("total_ms")}


def summarize_deterministic(rows):
    result = {"cases": len(rows), "answers_emitted": sum(r["answer_emitted"] for r in rows),
              "observed_claims": sum(r["claims"] for r in rows),
              "unsupported_claim_flags": sum(r["unsupported_claim_flags"] for r in rows)}
    for key in ("claim_citation_coverage", "valid_context_id_rate", "structured_evidence_compatibility", "kb_evidence_compatibility", "numeric_consistency"):
        result[key] = rate(sum(r[key]["numerator"] for r in rows), sum(r[key]["denominator"] for r in rows))
    result["answerability_correctness"] = rate(sum(r["answerability_correct"] for r in rows), len(rows))
    insufficient = [r for r in rows if r["expected_insufficient"]]
    result["insufficient_data_correctness"] = rate(sum(r["insufficient_data_correct"] for r in insufficient), len(insufficient))
    latencies = sorted(r["latency_ms"] for r in rows if type(r.get("latency_ms")) in (int, float) and math.isfinite(r["latency_ms"]) and r["latency_ms"] >= 0)
    result["latency_ms"] = {"n": len(latencies), "p50": latencies[math.ceil(.5 * len(latencies)) - 1] if latencies else None,
                            "p95": latencies[math.ceil(.95 * len(latencies)) - 1] if latencies else None}
    return result


def judge_packet(case, output):
    """Complete visible evidence, never inferred from a citation existing."""
    analyst = output.get("analyst") or {}
    return {"question": case["user_query"], "expected_answerability": case["expected_answerability"],
        "required_claims": [{k: g[k] for k in ("claim_id", "claim_type", "requirement", "numeric") if k in g} for g in case["required_claims"]], "answerability_reason": case.get("answerability_reason"),
        "answer": {k: analyst.get(k) for k in ("ok", "status", "answer", "claims", "compare_rows", "computation")},
        "visible_contexts": [{"context_id": key, "kind": value.get("kind"),
            "source": value.get("source"), "evidence": evidence_text(value)} for key, value in visible_contexts(output).items()]}


def validate_judgment(case, output, judgment):
    """Schema/quote checks do not certify the judge's semantic conclusion."""
    analyst = output.get("analyst") or {}
    claims = analyst.get("claims") or []
    expected = {c["claim_id"]: c for c in claims}
    if len(expected) != len(claims): raise ValueError("Duplicate observed claim IDs")
    contexts = visible_contexts(output)
    judgments = judgment["claims"]
    if len(judgments) != len(expected) or {r["claim_id"] for r in judgments} != set(expected): raise ValueError("Judge claim coverage mismatch")
    for row in judgments:
        if row["support"] not in SUPPORT or not isinstance(row["reason"], str) or not row["reason"].strip(): raise ValueError("Invalid support judgment")
        quotes = row["evidence_quotes"]
        if not isinstance(quotes, list): raise ValueError("Quotes must be a list")
        if row["support"] != "unsupported" and not quotes: raise ValueError("Support without evidence quotation")
        for quote in quotes:
            cid = quote["context_id"]
            if cid not in expected[row["claim_id"]].get("context_ids", []) or cid not in contexts or not quote["quote"] or quote["quote"] not in evidence_text(contexts[cid]):
                raise ValueError("Judge cited non-cited or invented evidence")
    required = judgment["requirements"]
    if len(required) != len(case["required_claims"]) or {r["claim_id"] for r in required} != {c["claim_id"] for c in case["required_claims"]}: raise ValueError("Judge requirement coverage mismatch")
    for row in required:
        if row["fulfillment"] not in {"complete", "partial", "missing"} or not row["reason"]: raise ValueError("Invalid completeness judgment")
    for key in ("answer_relevant", "answerability_correct", "unbound_factual_prose"):
        if type(judgment[key]) is not bool: raise ValueError("Judge boolean required")
    if not isinstance(judgment["answer_reason"], str) or not judgment["answer_reason"].strip(): raise ValueError("Judge answer reason required")
    return judgment


def summarize_semantic(cases, outputs, judgments):
    counts = Counter()
    fully_grounded = complete = answerability = evaluated = 0
    required_total = required_complete = 0
    observed_total = sum(len((outputs[c["id"]].get("analyst") or {}).get("claims") or []) for c in cases)
    for case in cases:
        verdict = judgments.get(case["id"])
        if verdict is None: continue
        evaluated += 1
        counts.update(r["support"] for r in verdict["claims"])
        analyst = outputs[case["id"]].get("analyst") or {}
        successful = bool(analyst.get("ok")) and analyst.get("status") in {"ok", "insufficient_data"}
        supported = all(r["support"] == "fully_supported" for r in verdict["claims"])
        nonvacuous = bool(verdict["claims"]) or (case["expected_answerability"] == "insufficient_data" and analyst.get("status") == "insufficient_data")
        fully_grounded += successful and nonvacuous and supported and not verdict["unbound_factual_prose"] and verdict["answerability_correct"]
        required_total += len(verdict["requirements"])
        required_complete += sum(r["fulfillment"] == "complete" for r in verdict["requirements"])
        complete += successful and verdict["answer_relevant"] and verdict["answerability_correct"] and all(r["fulfillment"] == "complete" for r in verdict["requirements"])
        answerability += verdict["answerability_correct"]
    return {"cases": len(cases), "judge_valid_cases": evaluated, "judge_missing_or_error_cases": len(cases) - evaluated,
        "claim_evaluation_coverage": rate(sum(counts.values()), observed_total),
        **{key + "_claim_rate": rate(counts[key], sum(counts.values())) for key in sorted(SUPPORT)},
        "fully_grounded_answer_rate_judged_only": rate(fully_grounded, evaluated),
        "fully_grounded_answers_over_all_cases_lower_bound": rate(fully_grounded, len(cases)),
        "complete_relevant_answers_judged_only": rate(complete, evaluated),
        "gold_requirement_completeness_judged_only": rate(required_complete, required_total),
        "semantic_answerability_correctness_judged_only": rate(answerability, evaluated)}
