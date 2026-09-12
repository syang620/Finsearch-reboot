from __future__ import annotations

import asyncio
import json
import unittest
from unittest import mock

from mcp import types

from agents.orchestrator import agent_orchestrator as orchestrator
from agents.retrieval.mcp_client import SecRetrievalMCPClient


def _output(*, status: str = "completed") -> dict:
    return {
        "run_id": "run-1234",
        "route": "hybrid",
        "status": status,
        "failure_stage": "none",
        "planner": {"status": "completed"},
        "lanes": {
            "kb": {"status": "ok"},
            "structured_fact": {"status": "ok"},
        },
        "degradation": {"active": False},
        "analyst": {"status": "ok"},
        "open_issues": [],
        "orchestrator_trace": {
            "total_ms": 125,
            "planner_timing_ms": {"planner_start_ms": 10},
            "retrieval_timing_ms": {"retrieve_ms": 80},
            "structured_fact_timing_ms": {"structured_facts_ms": 20},
        },
    }


class OrchestratorStructuredLoggingTests(unittest.TestCase):
    def test_unstructured_mcp_error_carries_controlled_marker(self) -> None:
        class ErrorSession:
            async def call_tool(self, _name, arguments):
                del arguments
                return type(
                    "Result",
                    (),
                    {
                        "structured_content": None,
                        "structuredContent": None,
                        "is_error": True,
                        "isError": False,
                        "content": [
                            types.TextContent(type="text", text="sensitive server response")
                        ],
                    },
                )()

        client = SecRetrievalMCPClient()
        client._session = ErrorSession()
        result = asyncio.run(
            client.retrieve_tables(
                queries=["revenue"],
                ticker="AAPL",
                fiscal_year=2024,
                timeout_s=1,
            )
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["dependency_error_categories"], ["mcp"])

    def test_dependency_error_record_is_redacted(self) -> None:
        with mock.patch.object(orchestrator.logger, "warning") as log_warning:
            orchestrator._log_dependency_error(
                run_id="run-1234",
                stage="retrieval",
                dependency="mcp",
                error=RuntimeError("password=do-not-log"),
            )

        rendered = log_warning.call_args.args[0]
        payload = json.loads(rendered)
        self.assertEqual(
            payload,
            {
                "schema_version": 1,
                "event": "orchestrator_dependency_error",
                "run_id": "run-1234",
                "stage": "retrieval",
                "dependency": "mcp",
                "exception_type": "RuntimeError",
            },
        )
        self.assertNotIn("do-not-log", rendered)

    def test_user_controlled_identifiers_are_not_logged(self) -> None:
        output = _output(status="failed")
        output["run_id"] = "password=do-not-log"
        output["open_issues"] = [
            {
                "code": "API_KEY_SECRET123",
                "message": "provider failed",
                "severity": "error",
            }
        ]

        with mock.patch.object(orchestrator.logger, "warning") as log_warning:
            orchestrator._log_orchestration_outcome(output)

        rendered = log_warning.call_args.args[0]
        payload = json.loads(rendered)
        self.assertEqual(payload["run_id"], "invalid-run-id")
        self.assertEqual(payload["error_codes"], ["UNCLASSIFIED_ERROR"])
        self.assertNotIn("do-not-log", rendered)

    def test_completed_outcome_has_stable_json_fields(self) -> None:
        with (
            mock.patch.object(orchestrator.logger, "info") as log_info,
            mock.patch.object(orchestrator.logger, "warning") as log_warning,
        ):
            orchestrator._log_orchestration_outcome(_output())

        log_warning.assert_not_called()
        payload = json.loads(log_info.call_args.args[0])
        self.assertEqual(
            set(payload),
            {
                "schema_version",
                "event",
                "run_id",
                "route",
                "status",
                "planner_status",
                "lane_statuses",
                "analyst_status",
                "failure_stage",
                "degraded",
                "timings_ms",
                "error_codes",
                "error_categories",
            },
        )
        self.assertEqual(payload["event"], "orchestrator_run_outcome")
        self.assertEqual(payload["lane_statuses"], {"kb": "ok", "structured_fact": "ok"})
        self.assertEqual(payload["timings_ms"]["total"], 125)

    def test_mcp_failure_is_classified_without_logging_raw_error(self) -> None:
        output = _output(status="degraded")
        output["lanes"]["kb"]["status"] = "failed"
        output["degradation"]["active"] = True
        output["open_issues"] = [
            {
                "code": "RETRIEVAL_ERROR",
                "message": "connection failed with password=do-not-log",
                "severity": "error",
            }
        ]

        with mock.patch.object(orchestrator.logger, "warning") as log_warning:
            orchestrator._log_orchestration_outcome(
                output,
                dependency_error_categories=["mcp"],
            )

        rendered = log_warning.call_args.args[0]
        payload = json.loads(rendered)
        self.assertEqual(payload["error_categories"], ["mcp"])
        self.assertEqual(payload["error_codes"], ["RETRIEVAL_ERROR"])
        self.assertNotIn("do-not-log", rendered)
        self.assertNotIn("password", rendered)

    def test_provider_failure_is_classified_without_logging_model_error(self) -> None:
        output = _output(status="failed")
        output["failure_stage"] = "planner"
        output["planner"] = {
            "status": "error",
            "error": "LLM_CALL_FAILED: api key sk-secret-value",
        }
        output["analyst"] = None
        output["open_issues"] = [
            {
                "code": "ANALYST_RUNTIME_ERROR",
                "message": "provider failed",
                "severity": "error",
            }
        ]

        with mock.patch.object(orchestrator.logger, "warning") as log_warning:
            orchestrator._log_orchestration_outcome(
                output,
                dependency_error_categories=["provider"],
            )

        rendered = log_warning.call_args.args[0]
        payload = json.loads(rendered)
        self.assertEqual(payload["error_categories"], ["provider"])
        self.assertEqual(payload["analyst_status"], "not_run")
        self.assertNotIn("sk-secret-value", rendered)
        self.assertNotIn("api key", rendered)

    def test_analyst_error_codes_are_included(self) -> None:
        output = _output(status="failed")
        output["analyst"] = {
            "status": "error",
            "open_issues": [
                {
                    "code": "ANALYST_MODEL_TIMEOUT",
                    "message": "provider detail",
                    "severity": "error",
                }
            ],
        }

        event = orchestrator._orchestration_log_event(
            output,
            dependency_error_categories=["provider"],
        )

        self.assertEqual(event["error_codes"], ["ANALYST_MODEL_TIMEOUT"])
        self.assertEqual(event["error_categories"], ["provider"])

    def test_internal_planner_and_grounding_codes_are_preserved(self) -> None:
        output = _output(status="failed")
        output["open_issues"] = [
            {
                "code": "PLANNER_RUNTIME_ERROR",
                "message": "planner failed",
                "severity": "error",
            },
            {
                "code": "GROUNDING_ROW_TEXT_MISMATCH",
                "message": "grounding failed",
                "severity": "error",
            },
        ]

        event = orchestrator._orchestration_log_event(output)

        self.assertEqual(
            event["error_codes"],
            ["GROUNDING_ROW_TEXT_MISMATCH", "PLANNER_RUNTIME_ERROR"],
        )

    def test_non_dependency_issue_does_not_gain_mcp_category(self) -> None:
        output = _output(status="failed")
        output["open_issues"] = [
            {
                "code": "ANALYST_GROUNDING_INVALID",
                "message": "business model is ambiguous",
                "severity": "error",
            }
        ]

        event = orchestrator._orchestration_log_event(output)

        self.assertEqual(event["error_categories"], [])

    def test_logging_handler_failure_does_not_escape(self) -> None:
        with mock.patch.object(
            orchestrator.logger,
            "info",
            side_effect=RuntimeError("handler failed"),
        ):
            orchestrator._log_orchestration_outcome(_output())

        with mock.patch.object(
            orchestrator.logger,
            "warning",
            side_effect=RuntimeError("handler failed"),
        ):
            orchestrator._log_dependency_error(
                run_id="run-1234",
                stage="retrieval",
                dependency="mcp",
                error=RuntimeError("request failed"),
            )

    def test_interrupted_outcome_uses_info_with_unknown_missing_fields(self) -> None:
        output = _output(status="interrupted")
        output["planner"] = {"status": "needs_clarification"}
        output["analyst"] = None
        output["lanes"] = {}

        with mock.patch.object(orchestrator.logger, "info") as log_info:
            orchestrator._log_orchestration_outcome(output)

        payload = json.loads(log_info.call_args.args[0])
        self.assertEqual(payload["planner_status"], "needs_clarification")
        self.assertEqual(payload["analyst_status"], "not_run")
        self.assertEqual(
            payload["lane_statuses"],
            {"kb": "unknown", "structured_fact": "unknown"},
        )

    def test_failed_metric_result_emits_dependency_record(self) -> None:
        class FailedMetricClient:
            async def get_metric(self, **_kwargs):
                return {"ok": False, "status": "error", "error": "401 Unauthorized"}

        plan = {
            "route": "structured_fact",
            "targets": [{"ticker": "AAPL", "fiscal_year": 2024, "form_type": "10-K"}],
            "structured_fact_requests": [{"metric_hint": "revenue"}],
        }
        with mock.patch.object(orchestrator.logger, "warning") as log_warning:
            results = asyncio.run(
                orchestrator._execute_structured_fact_requests(
                    plan_obj=plan,
                    client=FailedMetricClient(),
                    run_id="run-1234",
                )
            )

        self.assertEqual(results[0]["tool_result"]["status"], "error")
        events = [json.loads(call.args[0]) for call in log_warning.call_args_list]
        self.assertIn(
            {
                "schema_version": 1,
                "event": "orchestrator_dependency_error",
                "run_id": "run-1234",
                "stage": "structured_fact",
                "dependency": "mcp",
                "exception_type": "reported_error",
            },
            events,
        )

    def test_retrieval_returned_mcp_error_is_propagated(self) -> None:
        retrieval = {
            "ok": False,
            "job_runs": [
                {
                    "final_retrieval": {
                        "ok": False,
                        "unstructured": ["sensitive server response"],
                        "dependency_error_categories": ["mcp"],
                    }
                }
            ],
            "partial_failures": [],
        }
        state = {
            "plan_id": "run-1234",
            "retrieval_state": {},
            "retrieval_timing_ms": {},
        }
        with (
            mock.patch.object(
                orchestrator,
                "_get_orchestrator_mcp_client",
                new=mock.AsyncMock(return_value=object()),
            ),
            mock.patch.object(
                orchestrator,
                "retrieval_agent",
                new=mock.AsyncMock(return_value={"retrieval": retrieval}),
            ),
            mock.patch.object(
                orchestrator,
                "_reset_orchestrator_mcp_client",
                new=mock.AsyncMock(),
            ),
            mock.patch.object(orchestrator.logger, "warning") as log_warning,
        ):
            result = asyncio.run(orchestrator._retrieval_node(state))

        self.assertEqual(result["dependency_error_categories"], ["mcp"])
        payload = json.loads(log_warning.call_args.args[0])
        self.assertEqual(payload["dependency"], "mcp")
        self.assertNotIn("sensitive server response", log_warning.call_args.args[0])

    def test_retrieval_model_error_is_propagated(self) -> None:
        retrieval = {
            "ok": False,
            "job_runs": [
                {
                    "model_turns": [
                        {
                            "error": (
                                "RETRIEVAL_LLM_CALL_FAILED: RuntimeError: "
                                "api key secret"
                            ),
                            "dependency_error_categories": ["provider"],
                        }
                    ],
                    "final_retrieval": {"ok": False},
                }
            ],
            "partial_failures": [],
        }
        state = {
            "plan_id": "run-1234",
            "retrieval_state": {},
            "retrieval_timing_ms": {},
        }
        with (
            mock.patch.object(
                orchestrator,
                "_get_orchestrator_mcp_client",
                new=mock.AsyncMock(return_value=object()),
            ),
            mock.patch.object(
                orchestrator,
                "retrieval_agent",
                new=mock.AsyncMock(return_value={"retrieval": retrieval}),
            ),
            mock.patch.object(orchestrator.logger, "warning") as log_warning,
        ):
            result = asyncio.run(orchestrator._retrieval_node(state))

        self.assertEqual(result["dependency_error_categories"], ["provider"])
        payload = json.loads(log_warning.call_args.args[0])
        self.assertEqual(payload["dependency"], "provider")
        self.assertNotIn("api key secret", log_warning.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
