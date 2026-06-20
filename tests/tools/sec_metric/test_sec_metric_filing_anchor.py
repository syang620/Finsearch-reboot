import asyncio
import os
from pathlib import Path
import unittest
from unittest import mock

from mcp_server.tools.sec_metric import get_metric


FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "sec_metric"
AMBIGUOUS_CAPEX_FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "sec_metric_capex_ambiguous"
PARTIAL_FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "sec_metric_partial"


class SecMetricFilingAnchorTests(unittest.TestCase):
    def _run_fixture_metric(self, metric_id: str, *, fixture_root: Path = FIXTURE_ROOT):
        env = {
            **os.environ,
            "SEC_USER_AGENT": "FinSearch Tests (tests@example.org)",
            "SEC_METRIC_FIXTURE_ROOT": str(fixture_root),
        }
        with mock.patch.dict(os.environ, env, clear=True):
            return asyncio.run(
                get_metric(
                    ticker="AAPL",
                    fiscal_year=2025,
                    metric_id=metric_id,
                )
            )

    def _assert_atomic_metric_result(
        self,
        *,
        result,
        metric_id: str,
        expected_value: float,
        expected_concept_name: str,
        expected_fp: str | None,
    ) -> None:
        self.assertTrue(result.ok)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.metric_id, metric_id)
        self.assertIsNotNone(result.value)
        self.assertEqual(result.value, expected_value)
        self.assertEqual(result.unit, "USD")
        self.assertEqual(result.ticker, "AAPL")
        self.assertEqual(result.cik, "0000320193")
        self.assertEqual(result.fiscal_year, 2025)
        self.assertEqual(result.form_type, "10-K")
        self.assertEqual(result.accession_number, "0000320193-25-000073")
        self.assertEqual(result.report_date, "2025-09-27")
        self.assertEqual(result.filed_date, "2025-10-31")
        self.assertEqual(
            result.source_url,
            "https://www.sec.gov/Archives/edgar/data/320193/000032019325000073/aapl-20250927.htm",
        )
        self.assertIsNotNone(result.primary_fact)
        self.assertEqual(result.primary_fact.concept_name, expected_concept_name)
        self.assertEqual(result.primary_fact.taxonomy, "us-gaap")
        self.assertEqual(result.primary_fact.value, expected_value)
        self.assertEqual(result.primary_fact.unit, "USD")
        self.assertEqual(result.primary_fact.accession_number, "0000320193-25-000073")
        self.assertEqual(result.primary_fact.report_date, "2025-09-27")
        self.assertEqual(result.primary_fact.filed_date, "2025-10-31")
        self.assertEqual(result.primary_fact.form_type, "10-K")
        self.assertEqual(result.primary_fact.fp, expected_fp)
        self.assertTrue(result.primary_fact.matched_by_accession)
        self.assertTrue(result.primary_fact.matched_by_report_date)
        self.assertEqual(result.components, [])
        self.assertEqual(result.missing_component_groups, [])
        self.assertIsNone(result.error)
        self.assertIn("anchor", result.trace)
        self.assertEqual(result.trace["anchor"]["accession_number"], "0000320193-25-000073")
        self.assertEqual(result.trace["anchor"]["report_date"], "2025-09-27")

    def _assert_failure_contract(
        self,
        *,
        result,
        metric_id: str,
        expected_status: str,
        expect_anchor_metadata: bool,
    ) -> None:
        self.assertFalse(result.ok)
        self.assertEqual(result.status, expected_status)
        self.assertEqual(result.metric_id, metric_id)
        self.assertIsNone(result.value)
        self.assertIsNone(result.unit)
        self.assertEqual(result.ticker, "AAPL")
        self.assertEqual(result.fiscal_year, 2025)
        self.assertIsNone(result.primary_fact)
        self.assertIsNotNone(result.error)
        if expect_anchor_metadata:
            self.assertEqual(result.cik, "0000320193")
            self.assertEqual(result.form_type, "10-K")
            self.assertEqual(result.accession_number, "0000320193-25-000073")
            self.assertEqual(result.report_date, "2025-09-27")
            self.assertEqual(result.filed_date, "2025-10-31")
            self.assertEqual(
                result.source_url,
                "https://www.sec.gov/Archives/edgar/data/320193/000032019325000073/aapl-20250927.htm",
            )
            self.assertIn("anchor", result.trace)
        else:
            self.assertIsNone(result.cik)
            self.assertIsNone(result.form_type)
            self.assertIsNone(result.accession_number)
            self.assertIsNone(result.report_date)
            self.assertIsNone(result.filed_date)
            self.assertIsNone(result.source_url)

    def test_get_metric_uses_submissions_anchor_before_later_amendment(self) -> None:
        result = self._run_fixture_metric("total_debt")

        self.assertTrue(result.ok)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.metric_id, "total_debt")
        self.assertEqual(result.accession_number, "0000320193-25-000073")
        self.assertEqual(result.report_date, "2025-09-27")
        self.assertEqual(result.value, 97_500_000_000.0)
        self.assertEqual(
            [component.accession_number for component in result.components],
            ["0000320193-25-000073", "0000320193-25-000073"],
        )
        self.assertIsNone(result.primary_fact)
        self.assertEqual(result.missing_component_groups, [])
        self.assertIsNone(result.error)

    def test_get_metric_selects_revenue_from_registry_priority_order(self) -> None:
        result = self._run_fixture_metric("revenue")
        self._assert_atomic_metric_result(
            result=result,
            metric_id="revenue",
            expected_value=410000000000.0,
            expected_concept_name="RevenueFromContractWithCustomerExcludingAssessedTax",
            expected_fp="FY",
        )

    def test_get_metric_returns_structured_unsupported_metric_for_unknown_id(self) -> None:
        result = self._run_fixture_metric("some_unknown_metric")
        self._assert_failure_contract(
            result=result,
            metric_id="some_unknown_metric",
            expected_status="unsupported_metric",
            expect_anchor_metadata=False,
        )
        self.assertIn("not registered", str(result.error))

    def test_get_metric_returns_not_found_for_supported_metric_without_anchored_fact(self) -> None:
        result = self._run_fixture_metric("gross_profit", fixture_root=AMBIGUOUS_CAPEX_FIXTURE_ROOT)
        self._assert_failure_contract(
            result=result,
            metric_id="gross_profit",
            expected_status="not_found",
            expect_anchor_metadata=True,
        )
        self.assertEqual(result.components, [])
        self.assertEqual(result.missing_component_groups, [])
        self.assertIn("No anchored fact found", str(result.error))

    def test_get_metric_returns_cash_and_cash_equivalents_success(self) -> None:
        result = self._run_fixture_metric("cash_and_cash_equivalents")
        self._assert_atomic_metric_result(
            result=result,
            metric_id="cash_and_cash_equivalents",
            expected_value=31500000000.0,
            expected_concept_name="CashAndCashEquivalentsAtCarryingValue",
            expected_fp="FY",
        )

    def test_get_metric_selects_operating_cash_flow_from_registry_priority_order(self) -> None:
        result = self._run_fixture_metric("operating_cash_flow")
        self._assert_atomic_metric_result(
            result=result,
            metric_id="operating_cash_flow",
            expected_value=118000000000.0,
            expected_concept_name="NetCashProvidedByUsedInOperatingActivities",
            expected_fp="FY",
        )

    def test_atomic_duration_metrics_resolve_with_annual_filing_anchor(self) -> None:
        cases = [
            ("gross_profit", 182000000000.0, "GrossProfit"),
            ("operating_income", 128000000000.0, "OperatingIncomeLoss"),
            ("net_income", 102000000000.0, "NetIncomeLoss"),
        ]
        for metric_id, expected_value, expected_concept_name in cases:
            with self.subTest(metric_id=metric_id):
                result = self._run_fixture_metric(metric_id)
                self._assert_atomic_metric_result(
                    result=result,
                    metric_id=metric_id,
                    expected_value=expected_value,
                    expected_concept_name=expected_concept_name,
                    expected_fp="FY",
                )

    def test_atomic_instant_metrics_resolve_with_period_end_anchor(self) -> None:
        cases = [
            ("total_assets", 372000000000.0, "Assets"),
            ("total_liabilities", 287000000000.0, "Liabilities"),
            ("stockholders_equity", 85000000000.0, "StockholdersEquity"),
        ]
        for metric_id, expected_value, expected_concept_name in cases:
            with self.subTest(metric_id=metric_id):
                result = self._run_fixture_metric(metric_id)
                self._assert_atomic_metric_result(
                    result=result,
                    metric_id=metric_id,
                    expected_value=expected_value,
                    expected_concept_name=expected_concept_name,
                    expected_fp="FY",
                )

    def test_get_metric_returns_deterministic_capex_success(self) -> None:
        result = self._run_fixture_metric("capex")

        self.assertTrue(result.ok)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.metric_id, "capex")
        self.assertEqual(result.value, 17000000000.0)
        self.assertEqual([component.group_id for component in result.components], [
            "primary_cash_capex",
            "productive_assets_additional",
        ])

    def test_get_metric_returns_ambiguous_for_overlapping_capex_facts(self) -> None:
        result = self._run_fixture_metric("capex", fixture_root=AMBIGUOUS_CAPEX_FIXTURE_ROOT)
        self._assert_failure_contract(
            result=result,
            metric_id="capex",
            expected_status="ambiguous",
            expect_anchor_metadata=True,
        )
        self.assertIn("overlaps", str(result.error))
        self.assertGreaterEqual(len(result.components), 1)
        self.assertEqual(result.missing_component_groups, [])

    def test_get_metric_returns_partial_when_required_total_debt_component_is_missing(self) -> None:
        result = self._run_fixture_metric("total_debt", fixture_root=PARTIAL_FIXTURE_ROOT)
        self._assert_failure_contract(
            result=result,
            metric_id="total_debt",
            expected_status="partial",
            expect_anchor_metadata=True,
        )
        self.assertEqual([component.group_id for component in result.components], ["current_debt"])
        self.assertEqual(result.missing_component_groups, ["noncurrent_debt"])
        self.assertIn("Missing carrying-amount components", str(result.error))
        self.assertIn("missing_groups_from_selection", result.trace)


if __name__ == "__main__":
    unittest.main()
