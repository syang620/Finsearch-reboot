from copy import deepcopy
import json
from pathlib import Path

import pytest

from evals.semantic_answer_v1 import (deterministic_case, judge_packet, load_dataset,
    evidence_text, numeric_mentions, numeric_requirement, read_jsonl, source_bound, summarize_deterministic,
    summarize_semantic, validate_cases, validate_judgment, visible_contexts)

ROOT = Path("data/evals/semantic_answer/v1")

@pytest.fixture(scope="module")
def cases(): return load_dataset(ROOT)[0]

def output_for(case):
    gold = case["required_claims"][0]
    s = gold["sources"][0]
    fact = {k: s[k] for k in ("metric_id", "ticker", "unit", "value", "start_date", "report_date", "form_type")}
    fact.update(fiscal_year=s["fact_fiscal_year"], status="ok")
    return {"status": "completed", "analyst": {"ok": True, "status": "ok", "answer": gold["requirement"],
        "claims": [{"claim_id": "c1", "claim_type": "structured_numeric", "metric_id": s["metric_id"],
                    "text": f"Revenue was {s['value']/1000000} million USD.", "context_ids": ["ctx1"]}],
        "trace": {"analyst_visible_context_ids": ["ctx1"]}},
        "evaluation_trace": {"analyst_packet": {"context_items": [{"context_id": "ctx1", "kind": "structured_fact", "structured_fact": fact}]}}}

def test_frozen_loading_deterministic(cases):
    assert cases == load_dataset(ROOT)[0]
    assert len(cases) == 60
    assert sum(c["audit_selected"] for c in cases) == 18
    assert sum(len(c["required_claims"]) for c in cases) == 86

@pytest.mark.parametrize("mutation", ["duplicate_case", "duplicate_claim", "missing_source", "wrong_value", "wrong_span", "wrong_id"])
def test_invalid_gold_rejected(cases, mutation):
    rows = deepcopy(cases)
    if mutation == "duplicate_case": rows.append(rows[0])
    elif mutation == "duplicate_claim": rows[0]["required_claims"] *= 2
    elif mutation == "missing_source": rows[0]["required_claims"][0]["sources"] = []
    elif mutation == "wrong_value": rows[0]["required_claims"][0]["numeric"]["value"] = 1
    else:
        c = next(c for c in rows if c["stratum"] == "narrative")
        source = c["required_claims"][0]["sources"][0]
        if mutation == "wrong_span": source["spans"][0]["quote"] = "invented quote"
        else: source["evidence_id"] = "invented ID"
    with pytest.raises(ValueError):
        validate_cases(rows, read_jsonl(ROOT/"numeric_source_facts.jsonl"), read_jsonl("data/evals/retrieval/benchmark_v2/corpus.jsonl"))

def test_hash_rejects_changed_dataset(tmp_path):
    (tmp_path/"queries.jsonl").write_text("changed")
    (tmp_path/"manifest.json").write_text(json.dumps({"files_sha256": {"queries.jsonl": "bad"}}))
    with pytest.raises(ValueError, match="hash mismatch"): load_dataset(tmp_path)

def test_numeric_and_period_binding(cases):
    case = cases[0]
    output = output_for(case)
    assert deterministic_case(case, output)["numeric_consistency"]["rate"] == 1
    context = output["evaluation_trace"]["analyst_packet"]["context_items"][0]
    for key, wrong in (("report_date", "2024-12-31"), ("start_date", "2024-07-01"), ("unit", "shares"), ("ticker", "MSFT"), ("status", "error"), ("value", 1)):
        changed = deepcopy(context); changed["structured_fact"][key] = wrong
        assert not source_bound(changed, case["required_claims"][0])

def test_valid_citation_does_not_rescue_wrong_number(cases):
    output = output_for(cases[0]); output["analyst"]["claims"][0]["text"] = "Revenue was 999 million USD."
    row = deterministic_case(cases[0], output)
    assert row["valid_context_id_rate"]["rate"] == 1
    assert row["numeric_consistency"]["rate"] == 0

@pytest.mark.parametrize("text,unit,expected", [("$391.035 billion", "USD_millions", 391035000000), ("391,035 million", "USD_millions", 391035000000), ("391,035", "USD_millions", 391035000000), ("growth 2.02% in FY2024", "percent", 2.02), ("20.434 million square feet", "thousand_square_feet", 20434)])
def test_units(text, unit, expected): assert expected in numeric_mentions(text, unit)

def test_wrong_unit_not_a_dollar_value(): assert numeric_mentions("73.9%", "USD_millions") == []

def test_duplicate_context_rejected(cases):
    output = output_for(cases[0]); output["evaluation_trace"]["analyst_packet"]["context_items"] *= 2
    with pytest.raises(ValueError, match="Duplicate"): visible_contexts(output)

def test_hidden_context_is_invalid(cases):
    output = output_for(cases[0]); output["analyst"]["trace"]["analyst_visible_context_ids"] = []
    row = deterministic_case(cases[0], output)
    assert row["valid_context_id_rate"]["rate"] == 0
    assert row["numeric_consistency"]["rate"] == 0
    assert row["unsupported_claim_flags"] == 1

def test_numeric_evidence_cannot_support_attribution(cases):
    output = output_for(cases[0]); output["analyst"]["claims"][0]["claim_type"] = "attribution"
    row = deterministic_case(cases[0], output)
    assert row["kb_evidence_compatibility"]["rate"] == 0
    assert row["unsupported_claim_flags"] == 1

def test_failure_not_insufficient(cases):
    case = next(c for c in cases if c["expected_answerability"] == "insufficient_data")
    assert not deterministic_case(case, {"status": "failed"})["insufficient_data_correct"]
    assert deterministic_case(case, {"analyst": {"ok": True, "status": "insufficient_data", "claims": []}})["insufficient_data_correct"]

def test_failures_remain_in_denominator(cases):
    row = deterministic_case(cases[0], {})
    result = summarize_deterministic([row])
    assert result["numeric_consistency"]["denominator"] == 1
    assert result["numeric_consistency"]["rate"] == 0
    assert result["claim_citation_coverage"]["rate"] is None
    assert result["latency_ms"]["p50"] is None

def judgment_for(case, output):
    quote = judge_packet(case, output)["visible_contexts"][0]["evidence"]
    return {"claims": [{"claim_id": "c1", "support": "fully_supported", "reason": "Exact annual value", "evidence_quotes": [{"context_id": "ctx1", "quote": quote}]}],
        "requirements": [{"claim_id": case["required_claims"][0]["claim_id"], "fulfillment": "complete", "reason": "Matches gold"}],
        "answer_relevant": True, "answerability_correct": True, "unbound_factual_prose": False, "answer_reason": "Supported answer"}

@pytest.mark.parametrize("mutation", ["none", "missing_claim", "invented_quote", "uncited_context", "missing_gold", "bad_label", "bad_boolean"])
def test_judge_contract(cases, mutation):
    case = cases[0]; output = output_for(case); j = judgment_for(case, output)
    if mutation == "none": assert validate_judgment(case, output, j) == j; return
    if mutation == "missing_claim": j["claims"] = []
    elif mutation == "invented_quote": j["claims"][0]["evidence_quotes"][0]["quote"] = "invented"
    elif mutation == "uncited_context": j["claims"][0]["evidence_quotes"][0]["context_id"] = "ctx2"
    elif mutation == "missing_gold": j["requirements"] = []
    elif mutation == "bad_label": j["claims"][0]["support"] = "looks plausible"
    else: j["answer_relevant"] = "true"
    with pytest.raises(ValueError): validate_judgment(case, output, j)

def test_semantic_unknown_not_pass(cases):
    case = cases[0]; output = output_for(case)
    result = summarize_semantic([case], {case["id"]: output}, {})
    assert result["fully_grounded_answer_rate_judged_only"]["rate"] is None
    assert result["fully_grounded_answers_over_all_cases_lower_bound"]["rate"] == 0
    assert result["claim_evaluation_coverage"]["rate"] == 0

def test_semantic_partial_and_unbound_prose(cases):
    case = cases[0]; output = output_for(case); j = judgment_for(case, output)
    j["claims"][0]["support"] = "partially_supported"
    result = summarize_semantic([case], {case["id"]: output}, {case["id"]: j})
    assert result["partially_supported_claim_rate"]["rate"] == 1
    assert result["fully_grounded_answer_rate_judged_only"]["rate"] == 0
    j["claims"][0]["support"] = "fully_supported"; j["unbound_factual_prose"] = True
    assert summarize_semantic([case], {case["id"]: output}, {case["id"]: j})["fully_grounded_answer_rate_judged_only"]["rate"] == 0

def test_calculation_needs_both_bound_operands_and_tool(cases):
    case = next(c for c in cases if c["stratum"] == "calculator")
    gold = case["required_claims"][-1]
    contexts = {}
    for i, g in enumerate(case["required_claims"][:2]):
        one = output_for({"required_claims": [g]})
        contexts[str(i)] = one["evaluation_trace"]["analyst_packet"]["context_items"][0]
    claims = [{"claim_type": "calculation", "text": gold["requirement"], "context_ids": list(contexts)}]
    analyst = {"computation": {"result": gold["numeric"]["value"]}, "trace": {"used_financial_evaluator": True}}
    assert numeric_requirement(gold, claims, contexts, analyst)
    assert not numeric_requirement(gold, claims, {"0": contexts["0"]}, analyst)
    analyst["trace"]["used_financial_evaluator"] = False
    assert not numeric_requirement(gold, claims, contexts, analyst)

def test_hidden_evidence_tail_not_judged():
    context = {"kind": "text", "payload": {"content": "a" * 12000 + "\nSECRET SUPPORT"}}
    assert "SECRET SUPPORT" not in evidence_text(context)
    assert "truncated" in evidence_text(context)

def test_artifact_redacts_local_environment_only(monkeypatch):
    from scripts.evals.agents.run_semantic_v1 import sanitize
    monkeypatch.setenv("SEC_USER_AGENT", "private-contact-for-test")
    assert sanitize({"p": str(Path.cwd()) + "/data", "contact": "private-contact-for-test", "value": 391035}) == {"p": "<WORKTREE>/data", "contact": "<REDACTED>", "value": 391035}
