import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage
import unittest

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from agents.contracts import FormType
from agents.planner.interactive_target_resolution import (
    _build_metadata,
    _build_planner_output,
    _pick_form_type,
)
from agents.retrieval.query_planner_v2 import RetrievalWorkflowAgent
from agents.analyst.agent import build_packet_from_retrieval_output


class FakeNoToolLLM:
    def bind_tools(self, _tools, tool_choice: str = "any"):
        class Bound:
            async def ainvoke(self, _messages):
                return AIMessage(content="No tool calls emitted.", tool_calls=[])

        return Bound()


class FakeAcceptReviewer:
    def with_structured_output(self, _model):
        return self

    async def ainvoke(self, _prompt):
        return {
            "action": "accept",
            "reason": "accept",
            "rewrite_notes": "",
            "revised_doc_types": None,
        }


class CaptureClient:
    def __init__(self) -> None:
        self.form_type: Optional[str] = None

    async def retrieve_tables(
        self,
        *,
        queries,
        ticker,
        fiscal_year,
        form_type,
        doc_types,
        top_k,
        min_total_score,
        timeout_s,
    ):
        self.form_type = form_type
        return {
            "ok": True,
            "error": None,
            "queries_used": list(queries),
            "top_tables": [],
            "metadata_used": {
                "ticker": ticker,
                "fiscal_year": fiscal_year,
                "form_type": form_type,
                "doc_types": doc_types,
            },
            "max_total_score": None,
        }


class FormTypeNoDefaultTests(unittest.TestCase):
    def test_fiscal_year_cues_infer_ten_k(self) -> None:
        queries = [
            "What was Apple revenue in FY2024?",
            "What was Apple revenue in FY '24?",
            "What was Apple revenue for fiscal year 2024?",
            "What was Apple revenue for fiscal-year 2024?",
        ]

        for query in queries:
            with self.subTest(query=query):
                self.assertEqual(_pick_form_type(query), FormType.TEN_K)

    def test_quarterly_cues_take_precedence_over_fiscal_year(self) -> None:
        queries = [
            "What was Apple revenue in its FY2024 10-Q?",
            "What was Apple revenue in FY2024 Q1?",
            "What was Apple revenue in the second quarter of fiscal year 2024?",
        ]

        for query in queries:
            with self.subTest(query=query):
                self.assertEqual(_pick_form_type(query), FormType.TEN_Q)

    def test_unresolved_form_type_cues_remain_null(self) -> None:
        queries = [
            "What was Apple revenue in 2024?",
            "What did Apple disclose in its filing?",
            "What was Apple revenue in FY2024 Q4?",
        ]

        for query in queries:
            with self.subTest(query=query):
                self.assertIsNone(_pick_form_type(query))

    def test_build_planner_output_propagates_deterministic_form_type(self) -> None:
        query = "What was Apple revenue in FY2024?"
        target_run = {
            "planner_state": {
                "original_user_query": query,
                "effective_user_query": query,
                "clarification_history": [],
            },
            "deterministic_hints": {
                "ticker": "AAPL",
                "fiscal_year": 2024,
                "form_type": "10-K",
            },
            "deterministic_open_issues": [],
            "metric_guess": "revenue",
            "deterministic_intent_hint": "filing_fact",
            "deterministic_task_type_hint": "extract",
        }
        target_resolution = {
            "retrieval_needed": True,
            "route": "kb",
            "structured_fact_requests": [],
            "task_class": "single_target_fact",
            "targets": [
                {
                    "target_id": 1,
                    "target_key": "AAPL_FY2024",
                    "company_name": "Apple",
                    "ticker": "AAPL",
                    "fiscal_year": 2024,
                    "form_type": None,
                }
            ],
            "retrieval_plan": {
                "fanout_mode": "single_target",
                "jobs": [
                    {
                        "applies_to_target_ids": [1],
                        "goal": "revenue",
                        "job_type": "metric_extract",
                    }
                ],
            },
            "open_issues": [],
        }

        for llm_form_type in (None, "10-Q"):
            with self.subTest(llm_form_type=llm_form_type):
                target_resolution["targets"][0]["form_type"] = llm_form_type
                output = _build_planner_output(
                    status="completed",
                    target_run=target_run,
                    target_resolution=target_resolution,
                    clarification_request=None,
                )

                self.assertEqual(output["targets"][0]["form_type"], "10-K")
                self.assertEqual(output["metadata"]["form_type"], "10-K")

    def test_build_metadata_keeps_form_type_null_when_unresolved(self) -> None:
        metadata = _build_metadata(
            targets=[{"ticker": "AAPL", "fiscal_year": 2024}],
            deterministic_hints={"ticker": "AAPL", "fiscal_year": 2024},
        )

        self.assertIsNone(metadata.get("form_type"))

    def test_retrieval_workflow_target_normalization_keeps_form_type_null(self) -> None:
        wf = RetrievalWorkflowAgent(
            retrieval_llm=FakeNoToolLLM(),
            reviewer_llm=FakeAcceptReviewer(),
            max_attempts=1,
        )
        state = {
            "targets": [{"ticker": "AAPL", "fiscal_year": 2024}],
            "original_user_query": "cash flow trend",
            "retrieval_plan": {"jobs": [{"job_type": "fact_lookup", "goal": "cash flow"}]},
        }

        runs = asyncio.run(wf.run(state=state, client=CaptureClient()))
        self.assertEqual(len(runs), 1)
        self.assertIsNone(runs[0]["target"].get("form_type"))
        self.assertIsNone(runs[0]["attempts"][0]["retrieval"]["metadata_used"].get("form_type"))

    def test_retrieve_with_client_forwards_none_form_type(self) -> None:
        wf = RetrievalWorkflowAgent(
            retrieval_llm=FakeNoToolLLM(),
            reviewer_llm=FakeAcceptReviewer(),
        )
        client = CaptureClient()
        result = asyncio.run(
            wf._retrieve_with_client(
                client=client,
                request={"queries": ["cash flow"]},
                target={"ticker": "AAPL", "fiscal_year": 2024, "form_type": None},
            )
        )

        self.assertIsNone(client.form_type)
        self.assertIsNone(result["metadata_used"].get("form_type"))

    def test_build_packet_does_not_default_form_type_to_ten_k(self) -> None:
        packet = build_packet_from_retrieval_output(
            user_query="cash flow",
            retrieval_output={
                "ok": True,
                "queries_used": ["cash flow"],
                "top_tables": [],
                "metadata_used": {
                    "ticker": "AAPL",
                    "fiscal_year": 2024,
                },
            },
            metric="cash flow",
        )

        self.assertIsNone(packet.metadata.form_type)


if __name__ == "__main__":
    unittest.main()
