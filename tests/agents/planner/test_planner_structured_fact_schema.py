import unittest

from agents.contracts import PlannerOutput
from agents.planner.interactive_target_resolution import (
    DEFAULT_TARGET_RESOLUTION_PROMPT_TEMPLATE,
    InteractivePlannerAgent,
    _build_fallback_target_resolution,
    _build_planner_output,
    _normalize_resolution_output,
    render_target_resolution_prompt,
)


def _base_planner_output_payload() -> dict:
    return {
        "status": "completed",
        "retrieval_needed": True,
        "intent": "filing_fact",
        "route": "kb",
        "structured_fact_requests": [],
        "metadata": {"ticker": "AAPL", "fiscal_year": 2025, "form_type": "10-K"},
        "analysis_task": {
            "task_type": "extract",
            "metric": "revenue",
            "requires_calculation": False,
            "expected_artifacts": ["table", "row", "text"],
            "output_format": "short_answer",
        },
        "task_class": "single_target_fact",
        "targets": [
            {
                "target_id": 1,
                "target_key": "AAPL_FY2025",
                "company_name": "Apple",
                "ticker": "AAPL",
                "fiscal_year": 2025,
                "form_type": "10-K",
            }
        ],
        "retrieval_plan": {
            "fanout_mode": "single_target",
            "jobs": [
                {
                    "applies_to_target_ids": [1],
                    "goal": "extract annual revenue",
                    "job_type": "metric_extract",
                }
            ],
        },
        "open_issues": [],
        "original_user_query": "What was Apple revenue in FY2025?",
        "effective_user_query": "What was Apple revenue in FY2025?",
        "clarification_history": [],
        "clarification_request": None,
    }


def _base_target_run() -> dict:
    return {
        "user_query": "What was Apple revenue in FY2025?",
        "planner_state": {
            "original_user_query": "What was Apple revenue in FY2025?",
            "effective_user_query": "What was Apple revenue in FY2025?",
            "clarification_history": [],
        },
        "deterministic_hints": {"ticker": "AAPL", "company_name": "Apple", "fiscal_year": 2025, "form_type": "10-K"},
        "deterministic_open_issues": [],
        "metric_guess": "revenue",
        "deterministic_intent_hint": "filing_fact",
        "deterministic_task_type_hint": "extract",
        "deterministic_retrieval_needed_hint": True,
        "final_resolution": None,
        "llm_error": None,
        "validation_error": None,
        "needs_clarification": False,
    }


class PlannerStructuredFactSchemaTests(unittest.TestCase):
    def test_planner_output_accepts_complete_kb_runtime_shape(self) -> None:
        planner_output = PlannerOutput.model_validate(_base_planner_output_payload())

        self.assertEqual(planner_output.route, "kb")
        self.assertEqual(planner_output.structured_fact_requests, [])

        dumped = planner_output.model_dump(mode="json")
        self.assertEqual(dumped["route"], "kb")
        self.assertEqual(dumped["structured_fact_requests"], [])

    def test_planner_output_accepts_structured_fact_route_schema(self) -> None:
        planner_output = PlannerOutput.model_validate(
            {
                **_base_planner_output_payload(),
                "retrieval_needed": False,
                "route": "structured_fact",
                "structured_fact_requests": [
                    {
                        "subquestion": "What was Apple total debt at FY2025 year end?",
                        "metric_hint": "total debt",
                        "entity_hint": "Apple",
                        "fiscal_year": 2025,
                        "fiscal_period": "FY",
                    }
                ],
                "retrieval_plan": None,
            }
        )

        self.assertEqual(planner_output.route, "structured_fact")
        self.assertEqual(len(planner_output.structured_fact_requests), 1)
        self.assertEqual(
            planner_output.structured_fact_requests[0].subquestion,
            "What was Apple total debt at FY2025 year end?",
        )

    def test_planner_output_accepts_hybrid_route_with_kb_and_structured_fact_requests_schema(self) -> None:
        planner_output = PlannerOutput.model_validate(
            {
                **_base_planner_output_payload(),
                "route": "hybrid",
                "structured_fact_requests": [
                    {
                        "subquestion": "What was Apple revenue in FY2025?",
                        "metric_hint": "revenue",
                        "entity_hint": "Apple",
                        "fiscal_year": 2025,
                    }
                ],
            }
        )

        dumped = planner_output.model_dump(mode="json")
        self.assertEqual(dumped["route"], "hybrid")
        self.assertEqual(dumped["structured_fact_requests"][0]["metric_hint"], "revenue")

    def test_normalize_resolution_output_preserves_structured_fact_route_and_clarification_shape(self) -> None:
        normalized = _normalize_resolution_output(
            {
                "retrieval_needed": True,
                "route": "structured_fact",
                "structured_fact_requests": [
                    {
                        "subquestion": "What was Apple total debt at FY2025 year end?",
                        "metric_hint": "total debt",
                        "entity_hint": "Apple",
                        "fiscal_year": 2025,
                        "fiscal_period": "FY",
                    }
                ],
                "task_class": "single_target_fact",
                "targets": [
                    {
                        "target_id": 1,
                        "target_key": "AAPL_FY2025",
                        "company_name": "Apple",
                        "ticker": "AAPL",
                        "fiscal_year": 2025,
                        "form_type": "10-K",
                    }
                ],
                "retrieval_plan": None,
                "needs_clarification": True,
                "clarification_reason": "Ticker confirmation required",
                "clarification_questions": ["Please confirm the ticker."],
                "open_issues": [],
            }
        )

        self.assertEqual(normalized["route"], "structured_fact")
        self.assertEqual(len(normalized["structured_fact_requests"]), 1)
        self.assertIsNone(normalized["retrieval_plan"])
        self.assertEqual(normalized["targets"], [])
        self.assertTrue(normalized["needs_clarification"])
        self.assertEqual(normalized["clarification_questions"], ["Please confirm the ticker."])

    def test_packaged_turn_and_full_plan_preserve_hybrid_route(self) -> None:
        agent = InteractivePlannerAgent(llm=object(), auto_run_full_planner=False, log_timing=False)
        target_run = {
            **_base_target_run(),
            "final_resolution": {
                "retrieval_needed": True,
                "route": "hybrid",
                "structured_fact_requests": [
                    {
                        "subquestion": "What was Apple revenue in FY2025?",
                        "metric_hint": "revenue",
                        "entity_hint": "Apple",
                        "fiscal_year": 2025,
                        "fiscal_period": "FY",
                    }
                ],
                "task_class": "single_target_fact",
                "targets": [
                    {
                        "target_id": 1,
                        "target_key": "AAPL_FY2025",
                        "company_name": "Apple",
                        "ticker": "AAPL",
                        "fiscal_year": 2025,
                        "form_type": "10-K",
                    }
                ],
                "retrieval_plan": {
                    "fanout_mode": "single_target",
                    "jobs": [
                        {
                            "applies_to_target_ids": [1],
                            "goal": "extract annual revenue",
                            "job_type": "metric_extract",
                        }
                    ],
                },
                "needs_clarification": False,
                "clarification_reason": None,
                "clarification_questions": [],
                "open_issues": [],
            },
        }

        turn = agent._package_turn(target_run)

        self.assertEqual(turn["planner_output"]["route"], "hybrid")
        self.assertEqual(len(turn["planner_output"]["structured_fact_requests"]), 1)
        self.assertEqual(turn["full_plan"]["route"], "hybrid")
        self.assertEqual(len(turn["full_plan"]["structured_fact_requests"]), 1)

    def test_build_planner_output_preserves_structured_fact_route_when_retrieval_plan_is_none(self) -> None:
        planner_output = _build_planner_output(
            status="needs_clarification",
            target_run=_base_target_run(),
            target_resolution={
                "retrieval_needed": True,
                "route": "structured_fact",
                "structured_fact_requests": [
                    {
                        "subquestion": "What was Apple cash at FY2025 year end?",
                        "entity_hint": "Apple",
                        "fiscal_year": 2025,
                    }
                ],
                "task_class": "single_target_fact",
                "targets": [],
                "retrieval_plan": None,
                "needs_clarification": True,
                "clarification_reason": "Need company confirmation",
                "clarification_questions": ["Which company should be analyzed?"],
                "open_issues": [],
            },
            clarification_request={"reason": "Need company confirmation", "questions": ["Which company should be analyzed?"]},
        )

        self.assertEqual(planner_output["route"], "structured_fact")
        self.assertEqual(planner_output["structured_fact_requests"], [])
        self.assertIsNone(planner_output["retrieval_plan"])

    def test_build_planner_output_canonicalizes_completed_route_execution(self) -> None:
        target = {
            "target_id": 1,
            "target_key": "AAPL_FY2025",
            "company_name": "Apple",
            "ticker": "AAPL",
            "fiscal_year": 2025,
            "form_type": "10-K",
        }
        request = {
            "subquestion": "What was Apple revenue in FY2025?",
            "metric_hint": "revenue",
            "entity_hint": "Apple",
            "fiscal_year": 2025,
            "fiscal_period": "FY",
        }
        retrieval_plan = {
            "fanout_mode": "single_target",
            "jobs": [
                {
                    "applies_to_target_ids": [1],
                    "goal": "extract annual revenue",
                    "job_type": "metric_extract",
                }
            ],
        }

        structured = _build_planner_output(
            status="completed",
            target_run=_base_target_run(),
            target_resolution={
                "retrieval_needed": True,
                "route": "structured_fact",
                "structured_fact_requests": [request],
                "task_class": "single_target_fact",
                "targets": [target],
                "retrieval_plan": retrieval_plan,
                "open_issues": [],
            },
            clarification_request=None,
        )
        self.assertFalse(structured["retrieval_needed"])
        self.assertIsNone(structured["retrieval_plan"])
        self.assertEqual(len(structured["structured_fact_requests"]), 1)

        hybrid = _build_planner_output(
            status="completed",
            target_run=_base_target_run(),
            target_resolution={
                "retrieval_needed": False,
                "route": "hybrid",
                "structured_fact_requests": [request],
                "task_class": "single_target_fact",
                "targets": [target],
                "retrieval_plan": None,
                "open_issues": [],
            },
            clarification_request=None,
        )
        self.assertTrue(hybrid["retrieval_needed"])
        self.assertIsNotNone(hybrid["retrieval_plan"])
        self.assertEqual(len(hybrid["structured_fact_requests"]), 1)

        kb = _build_planner_output(
            status="completed",
            target_run=_base_target_run(),
            target_resolution={
                "retrieval_needed": True,
                "route": "kb",
                "structured_fact_requests": [request],
                "task_class": "single_target_fact",
                "targets": [target],
                "retrieval_plan": retrieval_plan,
                "open_issues": [],
            },
            clarification_request=None,
        )
        self.assertEqual(kb["structured_fact_requests"], [])

    def test_fallback_target_resolution_stays_kb_only(self) -> None:
        fallback = _build_fallback_target_resolution(
            target_run={
                "planner_state": {
                    "unresolved_blockers": ["ticker"],
                    "deterministic_targets": [],
                }
            }
        )

        self.assertEqual(fallback["route"], "kb")
        self.assertEqual(fallback["structured_fact_requests"], [])
        self.assertTrue(fallback["needs_clarification"])
        self.assertIsNone(fallback["retrieval_plan"])

    def test_normalize_resolution_output_preserves_hybrid_route(self) -> None:
        normalized = _normalize_resolution_output(
            {
                "retrieval_needed": True,
                "route": "hybrid",
                "structured_fact_requests": [
                    {
                        "subquestion": "What was Apple revenue in FY2025?",
                        "metric_hint": "revenue",
                    }
                ],
                "task_class": "single_target_fact",
                "targets": [
                    {
                        "target_id": 1,
                        "target_key": "AAPL_FY2025",
                        "company_name": "Apple",
                        "ticker": "AAPL",
                        "fiscal_year": 2025,
                        "form_type": "10-K",
                    }
                ],
                "retrieval_plan": {
                    "fanout_mode": "single_target",
                    "jobs": [
                        {
                            "applies_to_target_ids": [1],
                            "goal": "extract annual revenue",
                            "job_type": "metric_extract",
                        }
                    ],
                },
                "needs_clarification": False,
                "clarification_reason": None,
                "clarification_questions": [],
                "open_issues": [],
            }
        )

        self.assertEqual(normalized["route"], "hybrid")
        self.assertEqual(len(normalized["structured_fact_requests"]), 1)
        self.assertEqual(normalized["retrieval_plan"]["jobs"][0]["goal"], "extract annual revenue")

    def test_prompt_template_keeps_unsupported_finance_questions_on_kb(self) -> None:
        rendered = render_target_resolution_prompt(
            DEFAULT_TARGET_RESOLUTION_PROMPT_TEMPLATE,
            user_query="What was Apple's revenue?",
            payload={},
        )

        self.assertNotIn("{{STRUCTURED_FACT_CAPABILITY_POLICY}}", rendered)
        self.assertIn("Ratios, margins, yields, per-share metrics", rendered)
        self.assertIn("Generic cash, profit, and profitability", rendered)
        self.assertIn("Compare Apple and Microsoft revenue in FY2024", rendered)
        self.assertIn("Keep them on `kb`.", rendered)

    def test_prompt_template_requires_human_readable_metric_hints(self) -> None:
        self.assertIn("Keep `metric_hint` human-readable", DEFAULT_TARGET_RESOLUTION_PROMPT_TEMPLATE)
        self.assertIn('"cash_and_cash_equivalents"', DEFAULT_TARGET_RESOLUTION_PROMPT_TEMPLATE)
        self.assertIn('"total_debt"', DEFAULT_TARGET_RESOLUTION_PROMPT_TEMPLATE)
        self.assertIn('"stockholders_equity"', DEFAULT_TARGET_RESOLUTION_PROMPT_TEMPLATE)

    def test_normalize_resolution_output_humanizes_metric_hint(self) -> None:
        normalized = _normalize_resolution_output(
            {
                "retrieval_needed": True,
                "route": "structured_fact",
                "structured_fact_requests": [
                    {
                        "subquestion": "What were cash and cash equivalents?",
                        "metric_hint": "cash_and_cash_equivalents",
                        "entity_hint": "Apple",
                        "fiscal_year": 2025,
                    }
                ],
                "task_class": "single_target_fact",
                "targets": [
                    {
                        "target_id": 1,
                        "target_key": "AAPL_FY2025",
                        "company_name": "Apple",
                        "ticker": "AAPL",
                        "fiscal_year": 2025,
                        "form_type": "10-K",
                    }
                ],
                "retrieval_plan": None,
                "needs_clarification": False,
                "clarification_reason": None,
                "clarification_questions": [],
                "open_issues": [],
            }
        )

        self.assertEqual(normalized["structured_fact_requests"][0]["metric_hint"], "cash and cash equivalents")

    def test_normalize_resolution_output_forces_kb_for_unsupported_margin_and_per_share_requests(self) -> None:
        for metric_hint, subquestion in (
            ("gross margin", "What was the gross margin?"),
            ("earnings per share", "What was the earnings per share?"),
        ):
            normalized = _normalize_resolution_output(
                {
                    "retrieval_needed": True,
                    "route": "structured_fact",
                    "structured_fact_requests": [
                        {
                            "subquestion": subquestion,
                            "metric_hint": metric_hint,
                            "entity_hint": "Apple",
                            "fiscal_year": 2025,
                        }
                    ],
                    "task_class": "single_target_fact",
                    "targets": [
                        {
                            "target_id": 1,
                            "target_key": "AAPL_FY2025",
                            "company_name": "Apple",
                            "ticker": "AAPL",
                            "fiscal_year": 2025,
                            "form_type": "10-K",
                        }
                    ],
                    "retrieval_plan": None,
                    "needs_clarification": False,
                    "clarification_reason": None,
                    "clarification_questions": [],
                    "open_issues": [],
                }
            )

            self.assertEqual(normalized["route"], "kb")
            self.assertEqual(normalized["structured_fact_requests"], [])

    def test_normalize_resolution_output_forces_kb_for_multi_company_structured_fact_compare(self) -> None:
        normalized = _normalize_resolution_output(
            {
                "retrieval_needed": True,
                "route": "structured_fact",
                "structured_fact_requests": [
                    {
                        "subquestion": "What is the revenue?",
                        "metric_hint": "revenue",
                        "entity_hint": "Apple",
                        "fiscal_year": 2024,
                    },
                    {
                        "subquestion": "What is the revenue?",
                        "metric_hint": "revenue",
                        "entity_hint": "Microsoft",
                        "fiscal_year": 2024,
                    },
                ],
                "task_class": "multi_target_compare",
                "targets": [],
                "retrieval_plan": None,
                "needs_clarification": True,
                "clarification_reason": "Ambiguous tickers for Apple and Microsoft.",
                "clarification_questions": ["Please provide the ticker symbols for Apple and Microsoft."],
                "open_issues": [
                    {
                        "code": "MULTI_COMPANY_QUERY",
                        "message": "Multiple company entities detected; current crawl mode expects one primary company.",
                        "severity": "warning",
                    }
                ],
            }
        )

        self.assertEqual(normalized["route"], "kb")
        self.assertEqual(normalized["structured_fact_requests"], [])

    def test_normalize_resolution_output_keeps_supported_subset_in_hybrid_fallback(self) -> None:
        normalized = _normalize_resolution_output(
            {
                "retrieval_needed": False,
                "route": "structured_fact",
                "structured_fact_requests": [
                    {
                        "subquestion": "What was Apple's revenue in FY2025?",
                        "metric_hint": "revenue",
                        "entity_hint": "Apple",
                        "fiscal_year": 2025,
                    },
                    {
                        "subquestion": "What was Apple's return on equity in FY2025?",
                        "metric_hint": "ROE",
                        "entity_hint": "Apple",
                        "fiscal_year": 2025,
                    },
                ],
                "task_class": "single_target_fact",
                "targets": [
                    {
                        "target_id": 1,
                        "target_key": "AAPL_FY2025",
                        "company_name": "Apple",
                        "ticker": "AAPL",
                        "fiscal_year": 2025,
                        "form_type": "10-K",
                    }
                ],
                "retrieval_plan": None,
                "needs_clarification": False,
                "clarification_reason": None,
                "clarification_questions": [],
                "open_issues": [],
            }
        )

        self.assertEqual(normalized["route"], "hybrid")
        self.assertTrue(normalized["retrieval_needed"])
        self.assertEqual(
            [request["metric_hint"] for request in normalized["structured_fact_requests"]],
            ["revenue"],
        )
        fallback_job = normalized["retrieval_plan"]["jobs"][0]
        self.assertIn("ROE", fallback_job["goal"])
        self.assertNotIn("Apple", fallback_job["goal"])
        self.assertNotIn("2025", fallback_job["goal"])
        issue = normalized["open_issues"][0]
        self.assertEqual(issue["code"], "STRUCTURED_FACT_CAPABILITY_REJECTED")
        self.assertEqual(issue["metadata"]["question_class"], "unsupported_ratio")
        self.assertEqual(issue["metadata"]["metric_hint"], "ROE")
        self.assertEqual(issue["metadata"]["effective_route"], "hybrid")
        self.assertEqual(issue["metadata"]["outcome"], "partial_kb_fallback")

    def test_normalize_resolution_output_clarifies_material_metric_ambiguity(self) -> None:
        normalized = _normalize_resolution_output(
            {
                "retrieval_needed": False,
                "route": "structured_fact",
                "structured_fact_requests": [
                    {
                        "subquestion": "What was Apple's cash in FY2025?",
                        "metric_hint": "cash",
                        "entity_hint": "Apple",
                        "fiscal_year": 2025,
                    }
                ],
                "task_class": "single_target_fact",
                "targets": [
                    {
                        "target_id": 1,
                        "target_key": "AAPL_FY2025",
                        "company_name": "Apple",
                        "ticker": "AAPL",
                        "fiscal_year": 2025,
                        "form_type": "10-K",
                    }
                ],
                "retrieval_plan": None,
                "needs_clarification": False,
                "clarification_reason": None,
                "clarification_questions": [],
                "open_issues": [],
            }
        )

        self.assertEqual(normalized["route"], "kb")
        self.assertTrue(normalized["needs_clarification"])
        self.assertFalse(normalized["retrieval_needed"])
        self.assertEqual(normalized["structured_fact_requests"], [])
        self.assertEqual(normalized["targets"], [])
        self.assertIsNone(normalized["retrieval_plan"])
        self.assertIn("cash and cash equivalents", normalized["clarification_questions"][0])
        self.assertEqual(
            normalized["open_issues"][0]["metadata"]["candidate_metric_ids"],
            ["cash_and_cash_equivalents", "operating_cash_flow"],
        )

    def test_normalize_resolution_output_rejects_ratio_component_decomposition(self) -> None:
        normalized = _normalize_resolution_output(
            {
                "retrieval_needed": True,
                "route": "hybrid",
                "structured_fact_requests": [
                    {
                        "subquestion": "What was Apple's total debt in FY2024?",
                        "metric_hint": "total debt",
                    },
                    {
                        "subquestion": "What was Apple's stockholders equity in FY2024?",
                        "metric_hint": "stockholders equity",
                    },
                ],
                "task_class": "single_target_fact",
                "targets": [
                    {
                        "target_id": 1,
                        "target_key": "AAPL_FY2024",
                        "company_name": "Apple",
                        "ticker": "AAPL",
                        "fiscal_year": 2024,
                        "form_type": "10-K",
                    }
                ],
                "retrieval_plan": {
                    "fanout_mode": "single_target",
                    "jobs": [
                        {
                            "applies_to_target_ids": [1],
                            "goal": "extract debt and equity evidence",
                            "job_type": "metric_extract",
                        }
                    ],
                },
                "needs_clarification": False,
                "clarification_reason": None,
                "clarification_questions": [],
                "open_issues": [],
            },
            original_user_query="What was Apple's debt-to-equity ratio in FY2024?",
        )

        self.assertEqual(normalized["route"], "kb")
        self.assertEqual(normalized["structured_fact_requests"], [])
        self.assertEqual(
            {issue["metadata"]["question_class"] for issue in normalized["open_issues"]},
            {"unsupported_ratio"},
        )


if __name__ == "__main__":
    unittest.main()
