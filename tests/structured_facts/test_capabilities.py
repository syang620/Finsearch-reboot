from mcp_server.tools.sec_metric_registry import METRIC_REGISTRY
from structured_facts.capabilities import (
    DEFAULT_STRUCTURED_FACT_CAPABILITY_POLICY,
    StructuredFactQuestionClass,
)


def _classify(
    metric_hint: str,
    subquestion: str = "",
    *,
    fiscal_period=None,
    entity_hints=(),
):
    return DEFAULT_STRUCTURED_FACT_CAPABILITY_POLICY.classify_request(
        metric_hint=metric_hint,
        subquestion=subquestion,
        fiscal_period=fiscal_period,
        entity_hints=entity_hints,
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
        entity_hints=("Apple",),
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

    for subquestion in (
        "What was revenue from subscriptions?",
        "What was revenue by segment?",
        "What was revenue in 2024 by segment?",
        "What was revenue in fiscal 2024 from cloud?",
    ):
        qualified = _classify("revenue", subquestion)
        assert not qualified.permitted
        assert qualified.question_class == StructuredFactQuestionClass.UNKNOWN

    non_gaap = _classify("operating income", "What was non-GAAP operating income?")
    assert not non_gaap.permitted
    assert non_gaap.question_class == StructuredFactQuestionClass.UNKNOWN

    copular = _classify("revenue", "How much revenue was from subscriptions?")
    assert not copular.permitted
    assert copular.question_class == StructuredFactQuestionClass.UNKNOWN

    for subquestion in (
        "What was Microsoft's cloud revenue?",
        "What was advertising revenue?",
    ):
        component = _classify(
            "revenue",
            subquestion,
            entity_hints=("Microsoft",),
        )
        assert not component.permitted
        assert component.question_class == StructuredFactQuestionClass.UNKNOWN


def test_lowercase_entity_prefix_does_not_change_supported_classification() -> None:
    decision = _classify(
        "revenue",
        "what was apple revenue in 2024?",
        entity_hints=("apple",),
    )

    assert decision.permitted
    assert decision.matched_metric_ids == ("revenue",)

    issuer_name = _classify(
        "revenue",
        "What was International Paper's revenue in 2025?",
        entity_hints=("International Paper",),
    )
    assert issuer_name.permitted
    assert issuer_name.matched_metric_ids == ("revenue",)

    leading_article_issuer = _classify(
        "revenue",
        "What was The Trade Desk revenue in FY 2024?",
        entity_hints=("The Trade Desk",),
    )
    assert leading_article_issuer.permitted


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

    without_ratio_word = DEFAULT_STRUCTURED_FACT_CAPABILITY_POLICY.classify_requests(
        [
            {"metric_hint": "total debt", "subquestion": "What was total debt?"},
            {
                "metric_hint": "stockholders equity",
                "subquestion": "What was stockholders equity?",
            },
        ],
        original_user_query="What was Apple's debt-to-equity?",
    )
    assert {decision.question_class for decision in without_ratio_word} == {
        StructuredFactQuestionClass.UNSUPPORTED_RATIO
    }

    debt_to_assets = DEFAULT_STRUCTURED_FACT_CAPABILITY_POLICY.classify_requests(
        [
            {"metric_hint": "total debt", "subquestion": "What was total debt?"},
            {"metric_hint": "total assets", "subquestion": "What were total assets?"},
        ],
        original_user_query="What was Apple's debt-to-assets?",
    )
    assert {decision.question_class for decision in debt_to_assets} == {
        StructuredFactQuestionClass.UNSUPPORTED_RATIO
    }

    asset_turnover = _classify("total assets", "Calculate asset turnover.")
    assert asset_turnover.question_class == StructuredFactQuestionClass.UNSUPPORTED_RATIO

    percent_of = DEFAULT_STRUCTURED_FACT_CAPABILITY_POLICY.classify_requests(
        [
            {
                "metric_hint": "operating income",
                "subquestion": "What was operating income?",
            },
            {"metric_hint": "revenue", "subquestion": "What was revenue?"},
        ],
        original_user_query="What was operating income as a percent of revenue?",
    )
    assert {decision.question_class for decision in percent_of} == {
        StructuredFactQuestionClass.UNSUPPORTED_RATIO
    }


def test_trend_semantics_block_yearly_fact_decomposition() -> None:
    decisions = DEFAULT_STRUCTURED_FACT_CAPABILITY_POLICY.classify_requests(
        [
            {"metric_hint": "revenue", "subquestion": "What was revenue in 2022?"},
            {"metric_hint": "revenue", "subquestion": "What was revenue in 2023?"},
        ],
        original_user_query="Show the revenue trend over 2022 and 2023.",
    )
    assert {decision.question_class for decision in decisions} == {
        StructuredFactQuestionClass.UNSUPPORTED_DERIVED_METRIC
    }


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

    compared = DEFAULT_STRUCTURED_FACT_CAPABILITY_POLICY.classify_requests(
        [
            {"metric_hint": "total assets", "subquestion": "What were total assets?"},
            {
                "metric_hint": "total liabilities",
                "subquestion": "What were total liabilities?",
            },
        ],
        original_user_query="Total assets compared to total liabilities.",
    )
    assert {decision.question_class for decision in compared} == {
        StructuredFactQuestionClass.UNSUPPORTED_COMPARISON
    }


def test_spaced_annual_fiscal_period_suffix_is_supported() -> None:
    decision = _classify("revenue", "What was revenue for FY 2024?")
    assert decision.permitted
    assert decision.matched_metric_ids == ("revenue",)

    year_ended = _classify("revenue", "What was revenue for the year ended 2024?")
    assert year_ended.permitted


def test_quarterly_periods_are_rejected_from_annual_structured_execution() -> None:
    for question in (
        "What was revenue for Q1 2024?",
        "What was revenue for Q 1 2024?",
        "What was quarterly revenue?",
        "What was revenue for the first quarter?",
    ):
        decision = _classify("revenue", question)
        assert not decision.permitted
        assert (
            decision.question_class
            == StructuredFactQuestionClass.UNSUPPORTED_DERIVED_METRIC
        )

    guarded = DEFAULT_STRUCTURED_FACT_CAPABILITY_POLICY.classify_requests(
        [{"metric_hint": "revenue", "subquestion": "What was revenue?"}],
        original_user_query="What was revenue for Q1 2024?",
    )
    assert not guarded[0].permitted

    metadata_guarded = DEFAULT_STRUCTURED_FACT_CAPABILITY_POLICY.classify_requests(
        [
            {
                "metric_hint": "revenue",
                "subquestion": "What was revenue?",
                "fiscal_period": "Q1",
            }
        ],
        original_user_query="Revenue for the three months ended March 31, 2024.",
    )
    assert not metadata_guarded[0].permitted


def test_arithmetic_request_blocks_supported_looking_decomposition() -> None:
    decisions = DEFAULT_STRUCTURED_FACT_CAPABILITY_POLICY.classify_requests(
        [
            {"metric_hint": "total assets", "subquestion": "What were total assets?"},
            {
                "metric_hint": "total liabilities",
                "subquestion": "What were total liabilities?",
            },
        ],
        original_user_query="Calculate total assets minus total liabilities.",
    )

    assert {decision.question_class for decision in decisions} == {
        StructuredFactQuestionClass.UNSUPPORTED_DERIVED_METRIC
    }
    assert not any(decision.permitted for decision in decisions)

    for question in ("What was average revenue?", "What was the sum of revenue?"):
        decision = _classify("revenue", question)
        assert not decision.permitted
        assert (
            decision.question_class
            == StructuredFactQuestionClass.UNSUPPORTED_DERIVED_METRIC
        )

    sum_decisions = DEFAULT_STRUCTURED_FACT_CAPABILITY_POLICY.classify_requests(
        [
            {"metric_hint": "total assets", "subquestion": "What were total assets?"},
            {
                "metric_hint": "total liabilities",
                "subquestion": "What were total liabilities?",
            },
        ],
        original_user_query="Calculate the sum of total assets and total liabilities.",
    )
    assert {decision.question_class for decision in sum_decisions} == {
        StructuredFactQuestionClass.UNSUPPORTED_DERIVED_METRIC
    }
    assert not any(decision.permitted for decision in sum_decisions)

    plus_decisions = DEFAULT_STRUCTURED_FACT_CAPABILITY_POLICY.classify_requests(
        [
            {"metric_hint": "total assets", "subquestion": "What were total assets?"},
            {
                "metric_hint": "total liabilities",
                "subquestion": "What were total liabilities?",
            },
        ],
        original_user_query="Calculate total assets plus total liabilities.",
    )
    assert {decision.question_class for decision in plus_decisions} == {
        StructuredFactQuestionClass.UNSUPPORTED_DERIVED_METRIC
    }
    assert not any(decision.permitted for decision in plus_decisions)


def test_completed_expression_preserves_later_independent_request() -> None:
    decisions = DEFAULT_STRUCTURED_FACT_CAPABILITY_POLICY.classify_requests(
        [
            {"metric_hint": "total assets", "subquestion": "What were total assets?"},
            {
                "metric_hint": "total liabilities",
                "subquestion": "What were total liabilities?",
            },
            {"metric_hint": "revenue", "subquestion": "What was revenue?"},
        ],
        original_user_query=(
            "Calculate the difference between total assets and total liabilities; "
            "also give revenue."
        ),
    )

    assert (
        decisions[0].question_class
        == StructuredFactQuestionClass.UNSUPPORTED_COMPARISON
    )
    assert (
        decisions[1].question_class
        == StructuredFactQuestionClass.UNSUPPORTED_COMPARISON
    )
    assert decisions[2].permitted

    repeated = DEFAULT_STRUCTURED_FACT_CAPABILITY_POLICY.classify_requests(
        [
            {"metric_hint": "revenue", "subquestion": "What was revenue in 2024?"},
            {"metric_hint": "revenue", "subquestion": "What was revenue in 2023?"},
            {"metric_hint": "total assets", "subquestion": "What were total assets?"},
        ],
        original_user_query=(
            "Calculate the difference between revenue in 2024 and revenue in 2023; "
            "also give total assets."
        ),
    )
    assert all(not decision.permitted for decision in repeated[:2])
    assert repeated[2].permitted


def test_symbolic_arithmetic_blocks_supported_looking_decomposition() -> None:
    requests = [
        {"metric_hint": "total assets", "subquestion": "What were total assets?"},
        {
            "metric_hint": "total liabilities",
            "subquestion": "What were total liabilities?",
        },
    ]
    for operator in ("+", "/", "*", "-"):
        decisions = DEFAULT_STRUCTURED_FACT_CAPABILITY_POLICY.classify_requests(
            requests,
            original_user_query=(
                f"Calculate total assets {operator} total liabilities."
            ),
        )
        assert {decision.question_class for decision in decisions} == {
            StructuredFactQuestionClass.UNSUPPORTED_DERIVED_METRIC
        }
        assert not any(decision.permitted for decision in decisions)


def test_explicit_clauses_map_repeated_metrics_in_order() -> None:
    decisions = DEFAULT_STRUCTURED_FACT_CAPABILITY_POLICY.classify_requests(
        [
            {"metric_hint": "revenue", "subquestion": "Why did revenue increase?"},
            {"metric_hint": "revenue", "subquestion": "What was revenue?"},
        ],
        original_user_query="Why did revenue increase; also give revenue.",
    )

    assert (
        decisions[0].question_class
        == StructuredFactQuestionClass.UNSUPPORTED_DERIVED_METRIC
    )
    assert decisions[1].permitted


def test_conjoined_clauses_reject_only_unsupported_request() -> None:
    decisions = DEFAULT_STRUCTURED_FACT_CAPABILITY_POLICY.classify_requests(
        [
            {"metric_hint": "revenue", "subquestion": "What was revenue?"},
            {"metric_hint": "total assets", "subquestion": "What were total assets?"},
        ],
        original_user_query="Why did revenue increase and give me total assets?",
    )

    assert (
        decisions[0].question_class
        == StructuredFactQuestionClass.UNSUPPORTED_DERIVED_METRIC
    )
    assert decisions[1].permitted


def test_three_conjoined_clauses_preserve_supported_middle_request() -> None:
    decisions = DEFAULT_STRUCTURED_FACT_CAPABILITY_POLICY.classify_requests(
        [
            {"metric_hint": "revenue", "subquestion": "What was revenue?"},
            {"metric_hint": "total assets", "subquestion": "What were total assets?"},
            {
                "metric_hint": "explanation",
                "subquestion": "Explain why net income increased.",
            },
        ],
        original_user_query=(
            "Give revenue and total assets and explain why net income increased."
        ),
    )

    assert decisions[0].permitted
    assert decisions[1].permitted
    assert (
        decisions[2].question_class
        == StructuredFactQuestionClass.UNSUPPORTED_DERIVED_METRIC
    )


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

    for verb in ("grow", "grew", "decline", "fell", "rose"):
        variant = DEFAULT_STRUCTURED_FACT_CAPABILITY_POLICY.classify_requests(
            [{"metric_hint": "revenue", "subquestion": "What was revenue?"}],
            original_user_query=f"Why did revenue {verb}?",
        )
        assert (
            variant[0].question_class
            == StructuredFactQuestionClass.UNSUPPORTED_DERIVED_METRIC
        )


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
