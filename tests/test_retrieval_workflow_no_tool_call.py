import asyncio
import sys
from pathlib import Path

from langchain_core.messages import AIMessage

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from agents.retrieval.query_planner_v2 import RetrievalWorkflowAgent


class FakeRetrieverLLM:
    def bind_tools(self, _tools, tool_choice="any"):
        class BoundRetriever:
            async def ainvoke(self, _messages):
                return AIMessage(content="No tool calls emitted.", tool_calls=[])

        return BoundRetriever()


class FakeReviewerLLM:
    def __init__(self) -> None:
        self.calls = 0

    def with_structured_output(self, _model):
        return self

    async def ainvoke(self, _prompt):
        self.calls += 1
        return {
            "action": "retry" if self.calls == 1 else "accept",
            "reason": "retry review on first pass to verify attempt counting" if self.calls == 1 else "accept",
            "rewrite_notes": "",
            "revised_doc_types": None,
        }


class FakeClient:
    async def retrieve_tables(
        self,
        queries,
        ticker,
        fiscal_year,
        form_type,
        doc_types,
        top_k,
        min_total_score,
        timeout_s,
    ):
        return {
            "ok": False,
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


def test_retrieval_no_tool_call_increments_attempts_and_stops_on_budget():
    wf = RetrievalWorkflowAgent(
        retrieval_llm=FakeRetrieverLLM(),
        reviewer_llm=FakeReviewerLLM(),
        max_attempts=2,
    )

    state = {
        "targets": [{"ticker": "AAPL", "fiscal_year": 2024}],
        "original_user_query": "What is total debt?",
        "retrieval_plan": {"jobs": [{"job_type": "fact_lookup", "goal": "total debt"}]},
    }

    runs = asyncio.run(wf.run(state=state, client=FakeClient()))
    run = runs[0]

    assert len(run["attempts"]) == 2
    assert run["attempts"][0]["attempt_index"] == 1
    assert run["attempts"][1]["attempt_index"] == 2
    assert run["attempts"][1]["retrieval"]["error"] == "RETRIEVAL_TOOL_CALL_MISSING: model response had no tool_calls"
    assert run["review_feedback"]["action"] == "accept"
