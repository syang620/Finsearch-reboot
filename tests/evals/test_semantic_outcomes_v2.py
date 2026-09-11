from copy import deepcopy
import pytest

from evals.semantic_outcomes_v2 import accepted_answer, execution, summarize_numeric, summarize_outcomes


def answer():
    return {"ok": True, "status": "completed", "failure_stage": "none", "analyst": {
        "ok": True, "status": "ok", "answer": "Apple FY2024 revenue was $391.035 billion.",
        "claims": [{"claim_id": "c1", "text": "Apple FY2024 revenue was $391.035 billion.", "context_ids": ["x"]}]}}


@pytest.mark.parametrize("status", ["error", "tool_error", "grounding_error", "failed", "rejected", ""])
def test_rejected_answer_never_enters_semantic_channel(status):
    out = answer(); out['analyst']['status'] = status
    assert accepted_answer(out) is None


@pytest.mark.parametrize("field,value", [("ok", False), ("status", "failed"), ("failure_stage", "retrieval")])
def test_outer_failure_excludes_retained_success(field, value):
    out = answer(); out[field] = value
    assert accepted_answer(out) is None


def test_outer_error_diagnostic_cannot_retain_a_semantic_success():
    out=answer(); out['error']='Provider disconnected after producing a candidate.'
    assert accepted_answer(out) is None


def test_degraded_success_remains_eligible():
    out = answer(); out['status'] = 'degraded'
    out['open_issues'] = [{'code': 'KB_HYDRATION_ERROR', 'message': 'One recovered lane loss.'}]
    assert execution(out)['primary'] == 'substantive_answer'


def test_timeout_diagnostic_not_answer_text():
    out = answer(); out['analyst']['answer'] += ' ANALYST_MODEL_TIMEOUT'
    assert execution(out)['eligible']
    out['analyst']['error'] = 'ANALYST_MODEL_TIMEOUT after 120.0s'
    assert execution(out)['primary'] == 'analyst_timeout'
    assert accepted_answer(out) is None


def test_abstention_not_automatically_correct():
    out = answer(); out['analyst']['status'] = 'insufficient_data'; out['analyst']['claims'] = []
    summary = summarize_outcomes([{'execution': execution(out)}])
    assert summary['abstention_candidate_rate']['numerator'] == 1
    assert not any('correctness' in k for k in summary)


def test_failures_do_not_become_wrong_claims_or_disappear():
    rows = [{'execution': {'eligible': e}, 'numeric_checks': [{'truth': t, 'credit':t=='correct'}]} for e, t in
            [(True, 'correct'), (True, 'incorrect'), (True, 'unknown'), (True, 'missing'), (False, 'unassessed')]]
    s = summarize_numeric(rows)
    assert s['verified_numeric_credit_over_all_gold'] == {'numerator': 1, 'denominator': 5, 'rate': .2}
    assert s['verified_numeric_credit_given_eligible_answer']['denominator'] == 4
    assert s['numeric_correctness_resolved_only']['denominator'] == 2
    assert s['unassessed_due_to_execution'] == 1
    assert s['eligible_requirement_outcomes']['incorrect'] == 1


def test_truth_without_evidence_or_calculator_is_not_verified_credit():
    s=summarize_numeric([{'execution':{'eligible':True},'numeric_checks':[{'truth':'correct','credit':False}]}])
    assert s['verified_numeric_credit_over_all_gold']['numerator']==0
    assert s['verified_numeric_credit_given_eligible_answer']['numerator']==0
    assert s['parsed_numeric_truth_given_eligible_answer']['numerator']==1
    assert s['numeric_correctness_resolved_only']['rate']==1


def test_empty_denominators_are_unknown_not_one():
    assert summarize_numeric([])['numeric_correctness_resolved_only']['rate'] is None


def test_duplicate_claim_ids_rejected_without_mutation():
    out = answer(); out['analyst']['claims'] *= 2; before = deepcopy(out)
    with pytest.raises(ValueError, match='Duplicate'): accepted_answer(out)
    assert out == before


@pytest.mark.parametrize('stage,expected', [('planner', 'planner_failure'), ('retrieval', 'retrieval_failure'),
                                         ('structured_fact', 'tool_failure'), ('interrupted', 'clarification_stop')])
def test_execution_categories(stage, expected):
    out = answer(); out['failure_stage'] = stage; out['ok'] = False
    assert execution(out)['primary'] == expected
