import json

import pytest

from agents.planner.interactive_target_resolution import (
    InteractivePlannerAgent,
    _apply_anchored_filing_comparison,
    _intent_hint_from_query,
    build_target_resolution_payload,
)


_ANCHORED_QUERY = (
    "Using Apple's FY2024 filing, what was the percentage increase in "
    "Services net sales from 2023 to 2024?"
)


class _HostileAnchoredComparisonLLM:
    async def ainvoke(self, _prompt):
        return json.dumps(
            {
                "retrieval_needed": False,
                "route": "hybrid",
                "structured_fact_requests": [
                    {
                        "subquestion": "What were Services net sales in 2023?",
                        "metric_hint": "Services net sales",
                        "entity_hint": "Apple",
                        "fiscal_year": 2023,
                    },
                    {
                        "subquestion": "What were Services net sales in 2024?",
                        "metric_hint": "Services net sales",
                        "entity_hint": "Apple",
                        "fiscal_year": 2024,
                    },
                ],
                "task_class": "multi_target_compare",
                "targets": [
                    {
                        "target_id": 7,
                        "target_key": "AAPL_FY2023",
                        "company_name": "Apple",
                        "ticker": "AAPL",
                        "fiscal_year": 2023,
                        "form_type": "10-K",
                    },
                    {
                        "target_id": 8,
                        "target_key": "AAPL_FY2024",
                        "company_name": "Apple",
                        "ticker": "AAPL",
                        "fiscal_year": 2024,
                        "form_type": "10-K",
                    },
                ],
                "retrieval_plan": None,
                "needs_clarification": False,
                "clarification_reason": None,
                "clarification_questions": [],
                "open_issues": [
                    {
                        "code": "FISCAL_YEAR_AMBIGUOUS",
                        "message": "Multiple years were present.",
                        "severity": "warning",
                    }
                ],
            }
        )


@pytest.mark.parametrize(
    "query",
    [
        "What drove Apple Services growth in FY2024?",
        "Why did Apple revenue change in FY2024?",
        "How did management explain the revenue change in FY2024?",
    ],
)
def test_narrative_growth_and_change_remain_filing_fact(query):
    intent, task_type, retrieval_needed, cues = _intent_hint_from_query(
        query,
        "revenue",
    )

    assert intent.value == "filing_fact"
    assert task_type == "extract"
    assert retrieval_needed is True
    assert cues == ["narrative_explanation"]


@pytest.mark.parametrize(
    "query",
    [
        "Calculate Apple's Services growth in FY2024.",
        "What was Apple's Services growth rate in FY2024?",
        "What was the percentage increase in Services net sales?",
        "What was the percent change in Services net sales?",
        "How much did Services net sales increase?",
        "How much did Services net sales decrease?",
        "What was the difference between 2023 and 2024 Services net sales?",
        "What was the YoY change in Services net sales?",
        "What was the QoQ change in Services net sales?",
        "What was the Services CAGR?",
        "What was the Services margin?",
        "What was the price-to-earnings ratio?",
    ],
)
def test_explicit_quantitative_derivations_require_calculation(query):
    intent, task_type, retrieval_needed, cues = _intent_hint_from_query(
        query,
        "revenue",
    )

    assert intent.value == "filing_calc"
    assert task_type == "compute"
    assert retrieval_needed is True
    assert cues


def test_anchored_filing_payload_separates_target_from_comparison_periods():
    planner = InteractivePlannerAgent(
        llm=_HostileAnchoredComparisonLLM(),
        log_timing=False,
    )

    pre_llm = build_target_resolution_payload(
        planner=planner,
        user_query=_ANCHORED_QUERY,
    )

    payload = pre_llm["payload"]
    assert payload["anchored_filing_year"] == 2024
    assert payload["comparison_fiscal_years"] == [2023, 2024]
    assert payload["anchored_filing_comparison"] is True
    assert payload["deterministic_intent_hint"] == "filing_calc"
    assert payload["deterministic_targets"] == [
        {
            "target_id": 1,
            "target_key": "AAPL_FY2024",
            "company_name": "Apple",
            "ticker": "AAPL",
            "fiscal_year": 2024,
            "form_type": "10-K",
        }
    ]


def test_anchored_filing_comparison_normalizes_hostile_llm_output():
    planner = InteractivePlannerAgent(
        llm=_HostileAnchoredComparisonLLM(),
        log_timing=False,
    )

    output = planner.start(_ANCHORED_QUERY)["planner_output"]

    assert output["intent"] == "filing_calc"
    assert output["route"] == "kb"
    assert output["structured_fact_requests"] == []
    assert output["metadata"]["ticker"] == "AAPL"
    assert output["metadata"]["fiscal_year"] == 2024
    assert output["metadata"]["form_type"] == "10-K"
    assert output["targets"] == [
        {
            "target_id": 1,
            "target_key": "AAPL_FY2024",
            "company_name": "Apple",
            "ticker": "AAPL",
            "fiscal_year": 2024,
            "form_type": "10-K",
        }
    ]
    assert output["analysis_task"]["task_type"] == "compute"
    assert output["analysis_task"]["requires_calculation"] is True
    assert output["retrieval_plan"] == {
        "fanout_mode": "single_target",
        "jobs": [
            {
                "applies_to_target_ids": [1],
                "goal": "extract Services net sales for fiscal years 2023 and 2024",
                "job_type": "metric_extract",
            }
        ],
    }
    assert not any(
        issue["code"] in {"MULTI_YEAR_QUERY", "FISCAL_YEAR_AMBIGUOUS"}
        for issue in output["open_issues"]
    )


def test_non_anchored_multi_period_resolution_is_not_forced_to_kb():
    resolution = {
        "retrieval_needed": True,
        "route": "hybrid",
        "structured_fact_requests": [{"subquestion": "Get both values."}],
        "task_class": "multi_target_compare",
        "targets": [],
        "retrieval_plan": None,
        "needs_clarification": False,
        "clarification_reason": None,
        "clarification_questions": [],
        "open_issues": [],
    }

    normalized = _apply_anchored_filing_comparison(
        resolution,
        planner_state={"anchored_filing_comparison": False},
        metric_guess="revenue",
    )

    assert normalized is resolution
    assert normalized["route"] == "hybrid"


def test_anchored_normalization_preserves_explicit_target_form():
    resolution = {
        "retrieval_needed": True,
        "route": "hybrid",
        "structured_fact_requests": [],
        "task_class": "multi_target_compare",
        "targets": [
            {
                "target_id": 1,
                "target_key": "AAPL_FY2024",
                "company_name": "Apple",
                "ticker": "AAPL",
                "fiscal_year": 2024,
                "form_type": "10-Q",
            }
        ],
        "retrieval_plan": None,
        "needs_clarification": False,
        "clarification_reason": None,
        "clarification_questions": [],
        "open_issues": [],
    }
    planner_state = {
        "anchored_filing_comparison": True,
        "comparison_fiscal_years": [2023, 2024],
        "unresolved_blockers": [],
        "deterministic_targets": [
            {
                "target_id": 1,
                "target_key": "AAPL_FY2024",
                "company_name": "Apple",
                "ticker": "AAPL",
                "fiscal_year": 2024,
                "form_type": "10-K",
            }
        ],
    }

    normalized = _apply_anchored_filing_comparison(
        resolution,
        planner_state=planner_state,
        metric_guess="Services net sales",
    )

    assert normalized["targets"][0]["form_type"] == "10-Q"
