import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage
import unittest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from agents.planner.interactive_target_resolution import _build_metadata
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
