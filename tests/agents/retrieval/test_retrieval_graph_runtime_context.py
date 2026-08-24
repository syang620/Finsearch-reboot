from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from agents.retrieval.query_planner_v2 import RetrievalWorkflowAgent


class _ToolCallingRetrieverLLM:
    def bind_tools(self, _tools: Any, tool_choice: str = "any") -> Any:
        del tool_choice

        class _BoundRetriever:
            async def ainvoke(self, _messages: Any) -> AIMessage:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "sec_retrieve_tables",
                            "args": {
                                "queries": ["total debt"],
                                "doc_types": ["table"],
                                "reason": "locate the reported value",
                            },
                            "id": "retrieval-call",
                        }
                    ],
                )

        return _BoundRetriever()


class _AcceptingReviewerLLM:
    def with_structured_output(self, _model: Any) -> "_AcceptingReviewerLLM":
        return self

    async def ainvoke(self, _prompt: Any) -> dict[str, Any]:
        return {
            "action": "accept",
            "reason": "sufficient evidence",
            "rewrite_notes": "",
            "revised_doc_types": None,
        }


class _Overlap:
    def __init__(self) -> None:
        self.started = 0
        self.all_started = asyncio.Event()

    async def wait_for_both(self) -> None:
        self.started += 1
        if self.started == 2:
            self.all_started.set()
        await self.all_started.wait()


class _NonSerializableClient:
    def __init__(self, marker: str, overlap: _Overlap | None = None) -> None:
        self.marker = marker
        self.overlap = overlap
        self.calls = 0

    async def retrieve_tables(self, **request: Any) -> dict[str, Any]:
        self.calls += 1
        if self.overlap is not None:
            await self.overlap.wait_for_both()
        return {
            "ok": True,
            "error": None,
            "queries_used": list(request["queries"]),
            "top_tables": [
                {
                    "doc_id": self.marker,
                    "doc_type": "table",
                    "summary": f"evidence from {self.marker}",
                    "total_score": 1.0,
                }
            ],
            "metadata_used": {
                "ticker": request["ticker"],
                "fiscal_year": request["fiscal_year"],
                "form_type": request["form_type"],
                "doc_types": request["doc_types"],
            },
            "max_total_score": 1.0,
        }


class _OuterState(TypedDict, total=False):
    request: dict[str, Any]
    runs: list[dict[str, Any]]


def _request() -> dict[str, Any]:
    return {
        "targets": [{"ticker": "AAPL", "fiscal_year": 2024, "form_type": "10-K"}],
        "original_user_query": "What is total debt?",
        "retrieval_plan": {
            "jobs": [{"job_type": "metric_extract", "goal": "total debt"}],
        },
    }


def _contains_identity(value: Any, target: Any) -> bool:
    if value is target:
        return True
    if isinstance(value, dict):
        return any(_contains_identity(item, target) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_identity(item, target) for item in value)
    return False


class RetrievalGraphRuntimeContextTests(unittest.IsolatedAsyncioTestCase):
    def _workflow(self) -> RetrievalWorkflowAgent:
        return RetrievalWorkflowAgent(
            retrieval_llm=_ToolCallingRetrieverLLM(),
            reviewer_llm=_AcceptingReviewerLLM(),
        )

    async def test_live_client_is_not_serialized_by_inherited_sqlite_checkpointer(self) -> None:
        workflow = self._workflow()
        client = _NonSerializableClient("sqlite-client")

        async def run_retrieval(state: _OuterState) -> dict[str, Any]:
            return {"runs": await workflow.run(state=state["request"], client=client)}

        builder = StateGraph(_OuterState)
        builder.add_node("run_retrieval", run_retrieval)
        builder.add_edge(START, "run_retrieval")
        builder.add_edge("run_retrieval", END)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "retrieval.sqlite"
            async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
                await saver.setup()
                graph = builder.compile(checkpointer=saver)
                result = await graph.ainvoke(
                    {"request": _request()},
                    config={"configurable": {"thread_id": "retrieval-runtime-context"}},
                )

        run = result["runs"][0]
        self.assertEqual(client.calls, 1)
        self.assertIsNone(run["final_retrieval"]["error"])
        self.assertEqual(run["final_retrieval"]["top_tables"][0]["doc_id"], "sqlite-client")
        self.assertFalse(_contains_identity(result, client))

    async def test_concurrent_invocations_keep_runtime_clients_isolated(self) -> None:
        workflow = self._workflow()
        overlap = _Overlap()
        first_client = _NonSerializableClient("first-client", overlap)
        second_client = _NonSerializableClient("second-client", overlap)

        first_runs, second_runs = await asyncio.gather(
            workflow.run(state=_request(), client=first_client),
            workflow.run(state=_request(), client=second_client),
        )

        self.assertEqual(first_client.calls, 1)
        self.assertEqual(second_client.calls, 1)
        self.assertEqual(
            first_runs[0]["final_retrieval"]["top_tables"][0]["doc_id"],
            "first-client",
        )
        self.assertEqual(
            second_runs[0]["final_retrieval"]["top_tables"][0]["doc_id"],
            "second-client",
        )


if __name__ == "__main__":
    unittest.main()
