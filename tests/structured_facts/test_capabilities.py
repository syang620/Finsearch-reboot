from mcp_server.tools.sec_metric_registry import METRIC_REGISTRY
from structured_facts.capabilities import (
    DEFAULT_STRUCTURED_FACT_CAPABILITY_POLICY,
    StructuredFactQuestionClass,
)


def _classify(metric_hint: str, subquestion: str = ""):
    return DEFAULT_STRUCTURED_FACT_CAPABILITY_POLICY.classify_request(
        metric_hint=metric_hint,
        subquestion=subquestion,
    )


def test_capability_policy_explicitly_covers_metric_registry() -> None:
    assert {
        capability.metric_id
        for capability in DEFAULT_STRUCTURED_FACT_CAPABILITY_POLICY.capabilities
    } == set(METRIC_REGISTRY)


def test_exact_supported_phrases_precede_generic_ambiguity() -> None:
    cash = _classify("cash and cash equivalents")
    gross_profit = _classify("gross profit")

    assert cash.permitted
    assert cash.matched_metric_ids == ("cash_and_cash_equivalents",)
    assert gross_profit.permitted
    assert gross_profit.matched_metric_ids == ("gross_profit",)

    cash_question = _classify("", "What were Apple's cash and cash equivalents?")
    cash_amount_question = _classify(
        "cash and cash equivalents",
        "How much cash and cash equivalents did Apple report at FY2025 year end?",
    )
    profit_question = _classify("", "What was Apple's gross profit?")
    assert cash_question.permitted
    assert cash_question.matched_metric_ids == ("cash_and_cash_equivalents",)
    assert cash_amount_question.permitted
    assert cash_amount_question.matched_metric_ids == (
        "cash_and_cash_equivalents",
    )
    assert profit_question.permitted
    assert profit_question.matched_metric_ids == ("gross_profit",)


def test_generic_cash_profit_and_profitability_are_ambiguous() -> None:
    for metric_hint in ("cash", "profit", "profitability"):
        decision = _classify(metric_hint)

        assert not decision.permitted
        assert decision.question_class == StructuredFactQuestionClass.AMBIGUOUS
        assert decision.matched_metric_ids

    without_hint = _classify("", "What was cash in FY2025?")
    assert without_hint.matched_metric_ids == (
        "cash_and_cash_equivalents",
        "operating_cash_flow",
    )


def test_unsupported_semantics_precede_supported_metric_phrase() -> None:
    decision = _classify(
        "gross profit",
        "Calculate gross profit margin for FY2025.",
    )

    assert not decision.permitted
    assert decision.question_class == StructuredFactQuestionClass.UNSUPPORTED_RATIO


def test_supported_aliases_and_registry_derived_metrics_are_permitted() -> None:
    expected = {
        "net sales": "revenue",
        "EBIT": "operating_income",
        "CFO": "operating_cash_flow",
        "total debt": "total_debt",
        "capex": "capex",
    }
    for phrase, metric_id in expected.items():
        decision = _classify(phrase)

        assert decision.permitted
        assert decision.matched_metric_ids == (metric_id,)


def test_alias_does_not_match_inside_unrelated_metric_name() -> None:
    decision = _classify(
        "sales and marketing expense",
        "What were sales and marketing expenses?",
    )

    assert not decision.permitted
    assert decision.question_class == StructuredFactQuestionClass.UNKNOWN
    assert _classify("sales").matched_metric_ids == ("revenue",)


def test_supported_hint_does_not_override_unknown_subquestion() -> None:
    decision = _classify("revenue", "What were Apple's bookings?")

    assert not decision.permitted
    assert decision.question_class == StructuredFactQuestionClass.UNKNOWN
    assert decision.matched_metric_ids == ()


def test_supported_phrase_does_not_match_inside_modified_metric_name() -> None:
    decision = _classify("deferred revenue", "What was deferred revenue?")

    assert not decision.permitted
    assert decision.question_class == StructuredFactQuestionClass.UNKNOWN
    assert decision.matched_metric_ids == ()

    without_hint = _classify("", "What was deferred revenue?")
    assert not without_hint.permitted
    assert without_hint.question_class == StructuredFactQuestionClass.UNKNOWN

    contradictory_hint = _classify("revenue", "What was deferred revenue?")
    assert not contradictory_hint.permitted
    assert contradictory_hint.question_class == StructuredFactQuestionClass.UNKNOWN

    subscription = _classify("revenue", "What was subscription revenue?")
    assert not subscription.permitted
    assert subscription.question_class == StructuredFactQuestionClass.UNKNOWN


def test_unknown_metric_is_not_permitted() -> None:
    decision = _classify("bookings")

    assert not decision.permitted
    assert decision.question_class == StructuredFactQuestionClass.UNKNOWN


def test_original_unsupported_semantics_block_supported_looking_decomposition() -> None:
    decisions = DEFAULT_STRUCTURED_FACT_CAPABILITY_POLICY.classify_requests(
        [
            {"metric_hint": "total debt", "subquestion": "What was total debt?"},
            {
                "metric_hint": "stockholders equity",
                "subquestion": "What was stockholders equity?",
            },
        ],
        original_user_query="What was the debt-to-equity ratio?",
    )

    assert {decision.question_class for decision in decisions} == {
        StructuredFactQuestionClass.UNSUPPORTED_RATIO
    }
    assert not any(decision.permitted for decision in decisions)


def test_comparison_operand_conjunction_does_not_look_independent() -> None:
    decisions = DEFAULT_STRUCTURED_FACT_CAPABILITY_POLICY.classify_requests(
        [
            {"metric_hint": "total assets", "subquestion": "What were total assets?"},
            {
                "metric_hint": "total liabilities",
                "subquestion": "What were total liabilities?",
            },
        ],
        original_user_query=(
            "Calculate the difference between total assets and total liabilities."
        ),
    )

    assert {decision.question_class for decision in decisions} == {
        StructuredFactQuestionClass.UNSUPPORTED_COMPARISON
    }
    assert not any(decision.permitted for decision in decisions)


def test_mixed_supported_and_rejected_requests_remain_independent() -> None:
    decisions = DEFAULT_STRUCTURED_FACT_CAPABILITY_POLICY.classify_requests(
        [
            {"metric_hint": "revenue", "subquestion": "What was revenue?"},
            {"metric_hint": "ROE", "subquestion": "What was return on equity?"},
        ],
        original_user_query="What was revenue and return on equity?",
    )

    assert decisions[0].permitted
    assert decisions[1].question_class == StructuredFactQuestionClass.UNSUPPORTED_RATIO


def test_narrative_change_question_blocks_numeric_component_decomposition() -> None:
    decisions = DEFAULT_STRUCTURED_FACT_CAPABILITY_POLICY.classify_requests(
        [{"metric_hint": "revenue", "subquestion": "What was prior-year revenue?"}],
        original_user_query="Why did revenue increase?",
    )

    assert decisions[0].question_class == StructuredFactQuestionClass.UNSUPPORTED_DERIVED_METRIC
    assert not decisions[0].permitted


def test_original_unsupported_semantics_override_partially_rejected_decomposition() -> None:
    decisions = DEFAULT_STRUCTURED_FACT_CAPABILITY_POLICY.classify_requests(
        [
            {"metric_hint": "revenue", "subquestion": "What was revenue?"},
            {"metric_hint": "bookings", "subquestion": "What were bookings?"},
        ],
        original_user_query="Why did revenue increase?",
    )

    assert {decision.question_class for decision in decisions} == {
        StructuredFactQuestionClass.UNSUPPORTED_DERIVED_METRIC
    }
    assert not any(decision.permitted for decision in decisions)


def test_noun_phrase_conjunction_preserves_independent_mixed_requests() -> None:
    decisions = DEFAULT_STRUCTURED_FACT_CAPABILITY_POLICY.classify_requests(
        [
            {"metric_hint": "revenue", "subquestion": "What was revenue?"},
            {
                "metric_hint": "explanation",
                "subquestion": "Explain why revenue increased.",
            },
        ],
        original_user_query=(
            "Give me Apple's revenue and an explanation of why it increased."
        ),
    )

    assert decisions[0].permitted
    assert (
        decisions[1].question_class
        == StructuredFactQuestionClass.UNSUPPORTED_DERIVED_METRIC
    )


def test_alternate_clause_separators_preserve_independent_mixed_requests() -> None:
    for separator in ("as well as", "plus", ";"):
        decisions = DEFAULT_STRUCTURED_FACT_CAPABILITY_POLICY.classify_requests(
            [
                {"metric_hint": "revenue", "subquestion": "What was revenue?"},
                {
                    "metric_hint": "explanation",
                    "subquestion": "Explain why revenue increased.",
                },
            ],
            original_user_query=(
                f"Give me Apple's revenue {separator} an explanation of why it increased."
            ),
        )

        assert decisions[0].permitted
        assert (
            decisions[1].question_class
            == StructuredFactQuestionClass.UNSUPPORTED_DERIVED_METRIC
        )
