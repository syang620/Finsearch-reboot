import asyncio
import json
import os
from pathlib import Path
import sys
import unittest

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp import types
from mcp_server.tools import sec_metric as sec_metric_module


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "sec_metric"
AMBIGUOUS_CAPEX_FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "sec_metric_capex_ambiguous"
PARTIAL_FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "sec_metric_partial"


class SecMetricToolIntegrationTests(unittest.TestCase):
    def test_sec_get_metric_mcp_tool_returns_structured_statuses(self) -> None:
        async def _run(metric_id: str, fixture_root: Path) -> dict:
            env = {
                **os.environ,
                "SEC_USER_AGENT": "FinSearch Tests (tests@example.org)",
                "SEC_METRIC_FIXTURE_ROOT": str(fixture_root),
            }
            params = StdioServerParameters(
                command=sys.executable,
                args=[str(Path(sec_metric_module.__file__).resolve())],
                env=env,
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(
                        "sec_get_metric",
                        arguments={"ticker": "AAPL", "fiscal_year": 2025, "metric_id": metric_id},
                    )

            artifact = getattr(result, "structured_content", None)
            if artifact is None:
                artifact = getattr(result, "structuredContent", None)
            if not isinstance(artifact, dict):
                for block in getattr(result, "content", []) or []:
                    if isinstance(block, types.TextContent):
                        artifact = json.loads(block.text)
                        break
            if isinstance(artifact, dict) and isinstance(artifact.get("result"), dict):
                artifact = artifact["result"]
            return artifact

        debt_artifact = asyncio.run(_run("total_debt", FIXTURE_ROOT))
        revenue_artifact = asyncio.run(_run("revenue", FIXTURE_ROOT))
        capex_artifact = asyncio.run(_run("capex", FIXTURE_ROOT))
        unknown_artifact = asyncio.run(_run("some_unknown_metric", FIXTURE_ROOT))
        not_found_artifact = asyncio.run(_run("gross_profit", AMBIGUOUS_CAPEX_FIXTURE_ROOT))
        partial_artifact = asyncio.run(_run("total_debt", PARTIAL_FIXTURE_ROOT))
        ambiguous_artifact = asyncio.run(_run("capex", AMBIGUOUS_CAPEX_FIXTURE_ROOT))

        self.assertTrue(debt_artifact["ok"])
        self.assertEqual(debt_artifact["status"], "ok")
        self.assertEqual(debt_artifact["metric_id"], "total_debt")
        self.assertEqual(debt_artifact["value"], 97500000000.0)
        self.assertEqual(debt_artifact["accession_number"], "0000320193-25-000073")

        self.assertTrue(revenue_artifact["ok"])
        self.assertEqual(revenue_artifact["status"], "ok")
        self.assertEqual(revenue_artifact["metric_id"], "revenue")
        self.assertEqual(revenue_artifact["value"], 410000000000.0)
        self.assertEqual(
            revenue_artifact["primary_fact"]["concept_name"],
            "RevenueFromContractWithCustomerExcludingAssessedTax",
        )

        self.assertTrue(capex_artifact["ok"])
        self.assertEqual(capex_artifact["status"], "ok")
        self.assertEqual(capex_artifact["metric_id"], "capex")
        self.assertEqual(capex_artifact["value"], 17000000000.0)

        self.assertFalse(unknown_artifact["ok"])
        self.assertEqual(unknown_artifact["status"], "unsupported_metric")
        self.assertEqual(unknown_artifact["metric_id"], "some_unknown_metric")
        self.assertIsNone(unknown_artifact["value"])

        self.assertFalse(not_found_artifact["ok"])
        self.assertEqual(not_found_artifact["status"], "not_found")
        self.assertEqual(not_found_artifact["metric_id"], "gross_profit")
        self.assertIsNone(not_found_artifact["value"])
        self.assertEqual(not_found_artifact["accession_number"], "0000320193-25-000073")

        self.assertFalse(partial_artifact["ok"])
        self.assertEqual(partial_artifact["status"], "partial")
        self.assertEqual(partial_artifact["metric_id"], "total_debt")
        self.assertIsNone(partial_artifact["value"])
        self.assertEqual(partial_artifact["missing_component_groups"], ["noncurrent_debt"])

        self.assertFalse(ambiguous_artifact["ok"])
        self.assertEqual(ambiguous_artifact["status"], "ambiguous")
        self.assertEqual(ambiguous_artifact["metric_id"], "capex")
        self.assertIsNone(ambiguous_artifact["value"])


if __name__ == "__main__":
    unittest.main()
