from __future__ import annotations

import copy
import unittest

from pydantic import ValidationError

from agents.contracts import PlannerOutput, PlannerRuntimeOutput


def _target() -> dict:
    return {
        "target_id": 1,
        "target_key": "AAPL_FY2024",
        "company_name": "Apple",
        "ticker": "AAPL",
        "fiscal_year": 2024,
        "form_type": "10-K",
    }


def _retrieval_plan() -> dict:
    return {
        "fanout_mode": "single_target",
        "jobs": [
            {
                "applies_to_target_ids": [1],
                "goal": "extract annual revenue",
                "job_type": "metric_extract",
            }
        ],
    }


def _structured_request() -> dict:
    return {
        "subquestion": "What was Apple revenue in FY2024?",
        "metric_hint": "revenue",
        "entity_hint": "Apple",
        "fiscal_year": 2024,
        "fiscal_period": "FY",
    }


def _runtime_payload() -> dict:
    return {
        "status": "completed",
        "retrieval_needed": True,
        "intent": "filing_fact",
        "route": "kb",
        "structured_fact_requests": [],
        "metadata": {
            "ticker": "AAPL",
            "company_name": "Apple",
            "fiscal_year": 2024,
            "form_type": "10-K",
            "doc_types": None,
            "fiscal_quarter": None,
        },
        "analysis_task": {
            "task_type": "extract",
            "metric": "revenue",
            "definition_notes": [],
            "requires_calculation": False,
            "expected_artifacts": ["table", "row", "text"],
            "output_format": "short_answer",
        },
        "task_class": "single_target_fact",
        "targets": [_target()],
        "retrieval_plan": _retrieval_plan(),
        "open_issues": [],
        "original_user_query": "What was Apple revenue in FY2024?",
        "effective_user_query": "What was Apple revenue in FY2024?",
        "clarification_history": [],
        "clarification_request": None,
    }


class PlannerRuntimeContractTests(unittest.TestCase):
    def assert_invalid(self, payload: dict) -> None:
        with self.assertRaises(ValidationError):
            PlannerRuntimeOutput.model_validate(payload)

    def test_planner_output_is_runtime_contract_alias(self) -> None:
        self.assertIs(PlannerOutput, PlannerRuntimeOutput)

    def test_valid_completed_route_truth_table(self) -> None:
        kb_with_retrieval = PlannerRuntimeOutput.model_validate(_runtime_payload())
        self.assertTrue(kb_with_retrieval.retrieval_needed)

        kb_without_retrieval = _runtime_payload()
        kb_without_retrieval.update(
            {
                "retrieval_needed": False,
                "retrieval_plan": None,
                "intent": "definition",
            }
        )
        self.assertFalse(
            PlannerRuntimeOutput.model_validate(kb_without_retrieval).retrieval_needed
        )

        structured = _runtime_payload()
        structured.update(
            {
                "retrieval_needed": False,
                "route": "structured_fact",
                "structured_fact_requests": [_structured_request()],
                "retrieval_plan": None,
            }
        )
        self.assertEqual(
            PlannerRuntimeOutput.model_validate(structured).route,
            "structured_fact",
        )

        hybrid = _runtime_payload()
        hybrid.update(
            {
                "route": "hybrid",
                "structured_fact_requests": [_structured_request()],
            }
        )
        self.assertEqual(PlannerRuntimeOutput.model_validate(hybrid).route, "hybrid")

    def test_kb_filing_intent_requires_retrieval(self) -> None:
        for intent in ("filing_fact", "filing_calc"):
            payload = _runtime_payload()
            payload.update(
                {
                    "intent": intent,
                    "retrieval_needed": False,
                    "retrieval_plan": None,
                }
            )
            with self.subTest(intent=intent):
                self.assert_invalid(payload)

    def test_kb_definition_may_skip_retrieval(self) -> None:
        payload = _runtime_payload()
        payload.update(
            {
                "intent": "definition",
                "retrieval_needed": False,
                "retrieval_plan": None,
            }
        )

        plan = PlannerRuntimeOutput.model_validate(payload)

        self.assertEqual(plan.intent.value, "definition")
        self.assertFalse(plan.retrieval_needed)

    def test_structured_fact_filing_intent_skips_kb_retrieval(self) -> None:
        payload = _runtime_payload()
        payload.update(
            {
                "route": "structured_fact",
                "retrieval_needed": False,
                "retrieval_plan": None,
                "structured_fact_requests": [_structured_request()],
            }
        )

        plan = PlannerRuntimeOutput.model_validate(payload)

        self.assertEqual(plan.intent.value, "filing_fact")
        self.assertFalse(plan.retrieval_needed)

    def test_valid_non_completed_states(self) -> None:
        clarification = _runtime_payload()
        clarification.update(
            {
                "status": "needs_clarification",
                "retrieval_needed": False,
                "retrieval_plan": None,
                "structured_fact_requests": [],
                "clarification_request": {
                    "reason": None,
                    "questions": ["Which fiscal year should be used?"],
                },
            }
        )
        self.assertEqual(
            PlannerRuntimeOutput.model_validate(clarification).status,
            "needs_clarification",
        )

        error = _runtime_payload()
        error.update(
            {
                "status": "error",
                "retrieval_needed": False,
                "retrieval_plan": None,
                "structured_fact_requests": [],
            }
        )
        self.assertEqual(PlannerRuntimeOutput.model_validate(error).status, "error")

    def test_nullable_fields_are_still_required(self) -> None:
        for field_name in ("retrieval_plan", "clarification_request"):
            payload = _runtime_payload()
            del payload[field_name]
            with self.subTest(field_name=field_name):
                self.assert_invalid(payload)

    def test_extra_fields_are_forbidden(self) -> None:
        payload = _runtime_payload()
        payload["unexpected"] = True
        self.assert_invalid(payload)

        payload = _runtime_payload()
        payload["targets"][0]["unexpected"] = True
        self.assert_invalid(payload)

        payload = _runtime_payload()
        payload["retrieval_plan"]["jobs"][0]["unexpected"] = True
        self.assert_invalid(payload)

        payload = _runtime_payload()
        payload.update(
            {
                "retrieval_needed": False,
                "route": "structured_fact",
                "retrieval_plan": None,
                "structured_fact_requests": [
                    {**_structured_request(), "metric_id": "revenue"}
                ],
            }
        )
        self.assert_invalid(payload)

    def test_target_fields_are_required_but_values_may_be_null(self) -> None:
        payload = _runtime_payload()
        payload["targets"][0]["form_type"] = None
        self.assertIsNone(
            PlannerRuntimeOutput.model_validate(payload).targets[0].form_type
        )

        payload = _runtime_payload()
        del payload["targets"][0]["form_type"]
        self.assert_invalid(payload)

    def test_target_ids_and_keys_must_be_valid_and_unique(self) -> None:
        payload = _runtime_payload()
        payload["targets"][0]["target_id"] = 0
        self.assert_invalid(payload)

        duplicate_id = copy.deepcopy(_target())
        duplicate_id["target_key"] = "MSFT_FY2024"
        payload = _runtime_payload()
        payload["targets"].append(duplicate_id)
        self.assert_invalid(payload)

        duplicate_key = copy.deepcopy(_target())
        duplicate_key["target_id"] = 2
        payload = _runtime_payload()
        payload["targets"].append(duplicate_key)
        self.assert_invalid(payload)

    def test_retrieval_plan_references_declared_targets(self) -> None:
        payload = _runtime_payload()
        payload["retrieval_plan"]["jobs"][0]["applies_to_target_ids"] = [2]
        self.assert_invalid(payload)

    def test_invalid_completed_route_combinations(self) -> None:
        cases = []

        kb_with_structured = _runtime_payload()
        kb_with_structured["structured_fact_requests"] = [_structured_request()]
        cases.append(kb_with_structured)

        kb_missing_plan = _runtime_payload()
        kb_missing_plan["retrieval_plan"] = None
        cases.append(kb_missing_plan)

        kb_unneeded_plan = _runtime_payload()
        kb_unneeded_plan["retrieval_needed"] = False
        cases.append(kb_unneeded_plan)

        structured_with_retrieval = _runtime_payload()
        structured_with_retrieval.update(
            {"route": "structured_fact", "structured_fact_requests": [_structured_request()]}
        )
        cases.append(structured_with_retrieval)

        structured_without_requests = _runtime_payload()
        structured_without_requests.update(
            {
                "route": "structured_fact",
                "retrieval_needed": False,
                "retrieval_plan": None,
            }
        )
        cases.append(structured_without_requests)

        hybrid_without_retrieval = _runtime_payload()
        hybrid_without_retrieval.update(
            {
                "route": "hybrid",
                "retrieval_needed": False,
                "retrieval_plan": None,
                "structured_fact_requests": [_structured_request()],
            }
        )
        cases.append(hybrid_without_retrieval)

        hybrid_without_requests = _runtime_payload()
        hybrid_without_requests["route"] = "hybrid"
        cases.append(hybrid_without_requests)

        for index, payload in enumerate(cases):
            with self.subTest(case=index):
                self.assert_invalid(payload)

    def test_non_completed_states_cannot_contain_executable_work(self) -> None:
        for status in ("needs_clarification", "error"):
            payload = _runtime_payload()
            payload["status"] = status
            if status == "needs_clarification":
                payload["clarification_request"] = {
                    "reason": "Need more information",
                    "questions": ["Which company?"],
                }
            with self.subTest(status=status):
                self.assert_invalid(payload)

            payload = _runtime_payload()
            payload.update(
                {
                    "status": status,
                    "retrieval_needed": False,
                    "retrieval_plan": None,
                    "structured_fact_requests": [_structured_request()],
                }
            )
            if status == "needs_clarification":
                payload["clarification_request"] = {
                    "reason": "Need more information",
                    "questions": ["Which company?"],
                }
            with self.subTest(status=status, structured=True):
                self.assert_invalid(payload)

    def test_clarification_request_matches_status(self) -> None:
        missing = _runtime_payload()
        missing.update(
            {
                "status": "needs_clarification",
                "retrieval_needed": False,
                "retrieval_plan": None,
            }
        )
        self.assert_invalid(missing)

        empty = copy.deepcopy(missing)
        empty["clarification_request"] = {"reason": None, "questions": []}
        self.assert_invalid(empty)

        completed = _runtime_payload()
        completed["clarification_request"] = {
            "reason": "Unexpected",
            "questions": ["Why?"],
        }
        self.assert_invalid(completed)


if __name__ == "__main__":
    unittest.main()
