from __future__ import annotations

import json
import unittest
from unittest import mock

from agents.orchestrator import agent_orchestrator as orchestrator


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
            {"code": "api-key-do-not-log", "message": "provider failed"}
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
                "code": "RETRIEVAL_MCP_TRANSPORT_ERROR",
                "message": "connection failed with password=do-not-log",
            }
        ]

        with mock.patch.object(orchestrator.logger, "warning") as log_warning:
            orchestrator._log_orchestration_outcome(output)

        rendered = log_warning.call_args.args[0]
        payload = json.loads(rendered)
        self.assertEqual(payload["error_categories"], ["mcp"])
        self.assertEqual(payload["error_codes"], ["RETRIEVAL_MCP_TRANSPORT_ERROR"])
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
        output["open_issues"] = [{"code": "PLANNER_LLM_CALL_FAILED", "message": "provider failed"}]

        with mock.patch.object(orchestrator.logger, "warning") as log_warning:
            orchestrator._log_orchestration_outcome(output)

        rendered = log_warning.call_args.args[0]
        payload = json.loads(rendered)
        self.assertEqual(payload["error_categories"], ["provider"])
        self.assertEqual(payload["analyst_status"], "not_run")
        self.assertNotIn("sk-secret-value", rendered)
        self.assertNotIn("api key", rendered)

    def test_non_dependency_issue_does_not_gain_mcp_category(self) -> None:
        output = _output(status="failed")
        output["open_issues"] = [
            {"code": "ANALYST_GROUNDING_INVALID", "message": "pipeline validation failed"}
        ]

        event = orchestrator._orchestration_log_event(output)

        self.assertEqual(event["error_categories"], [])

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


if __name__ == "__main__":
    unittest.main()
