import unittest

from mcp_server.tools.sec_metric import (
    _DERIVED_COMPUTE_STRATEGIES,
    MetricComponentResult,
    compute_capex_value,
    compute_total_debt_carrying_amount,
)
from mcp_server.tools.sec_metric_registry import get_metric_definition


class SecMetricComputationTests(unittest.TestCase):
    def test_registry_lookup_by_string_key_returns_metric_plan(self) -> None:
        revenue = get_metric_definition("revenue")
        total_debt = get_metric_definition("total_debt")
        unknown = get_metric_definition("some_unknown_metric")

        self.assertIsNotNone(revenue)
        self.assertEqual(revenue.metric_id, "revenue")
        self.assertEqual(revenue.kind, "atomic")
        self.assertIsNotNone(total_debt)
        self.assertEqual(total_debt.metric_id, "total_debt")
        self.assertEqual(total_debt.kind, "derived")
        self.assertEqual(total_debt.compute_strategy, "total_debt_carrying_amount")
        self.assertIsNotNone(get_metric_definition("cash_and_cash_equivalents"))
        self.assertIsNotNone(get_metric_definition("operating_cash_flow"))
        capex = get_metric_definition("capex")
        self.assertIsNotNone(capex)
        self.assertEqual(capex.compute_strategy, "capex_value")
        self.assertIsNone(unknown)
        self.assertEqual(
            sorted(_DERIVED_COMPUTE_STRATEGIES.keys()),
            ["capex_value", "total_debt_carrying_amount"],
        )

    def test_total_debt_carrying_amount_computation_handles_complete_and_partial_sets(self) -> None:
        current = MetricComponentResult(
            group_id="current_debt",
            group_label="Current debt carrying amount",
            taxonomy="us-gaap",
            concept_name="LongTermDebtCurrent",
            unit="USD",
            value=12_500_000_000.0,
        )
        noncurrent = MetricComponentResult(
            group_id="noncurrent_debt",
            group_label="Noncurrent debt carrying amount",
            taxonomy="us-gaap",
            concept_name="LongTermDebtNoncurrent",
            unit="USD",
            value=85_000_000_000.0,
        )

        complete = compute_total_debt_carrying_amount(
            components=[current, noncurrent],
            required_group_ids=["current_debt", "noncurrent_debt"],
        )
        self.assertEqual(
            complete,
            {
                "status": "ok",
                "value": 97_500_000_000.0,
                "missing_component_groups": [],
            },
        )

        partial = compute_total_debt_carrying_amount(
            components=[noncurrent],
            required_group_ids=["current_debt", "noncurrent_debt"],
        )
        self.assertEqual(
            partial,
            {
                "status": "partial",
                "value": None,
                "missing_component_groups": ["current_debt"],
            },
        )

        insufficient = compute_total_debt_carrying_amount(
            components=[],
            required_group_ids=["current_debt", "noncurrent_debt"],
        )
        self.assertEqual(
            insufficient,
            {
                "status": "not_found",
                "value": None,
                "missing_component_groups": ["current_debt", "noncurrent_debt"],
            },
        )

    def test_capex_computation_handles_deterministic_and_ambiguous_inputs(self) -> None:
        primary = MetricComponentResult(
            group_id="primary_cash_capex",
            group_label="Primary cash capital expenditures",
            taxonomy="us-gaap",
            concept_name="PaymentsToAcquirePropertyPlantAndEquipment",
            unit="USD",
            value=15_000_000_000.0,
        )
        productive = MetricComponentResult(
            group_id="productive_assets_additional",
            group_label="Additional productive assets cash outflows",
            taxonomy="us-gaap",
            concept_name="PaymentsToAcquireProductiveAssets",
            unit="USD",
            value=2_000_000_000.0,
        )
        fallback = MetricComponentResult(
            group_id="fallback_capex_total",
            group_label="Fallback total capital expenditures",
            taxonomy="us-gaap",
            concept_name="CapitalExpendituresIncurred",
            unit="USD",
            value=19_500_000_000.0,
        )

        deterministic = compute_capex_value(
            primary_cash_capex=primary,
            productive_assets_additional=productive,
            fallback_capex_total=None,
        )
        self.assertEqual(
            deterministic,
            {
                "status": "ok",
                "value": 17_000_000_000.0,
                "components": [primary, productive],
                "missing_component_groups": [],
                "error": None,
            },
        )

        ambiguous = compute_capex_value(
            primary_cash_capex=primary,
            productive_assets_additional=None,
            fallback_capex_total=fallback,
        )
        self.assertEqual(ambiguous["status"], "ambiguous")
        self.assertIsNone(ambiguous["value"])
        self.assertEqual(ambiguous["components"], [primary, fallback])


if __name__ == "__main__":
    unittest.main()
