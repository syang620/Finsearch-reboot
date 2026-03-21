import sys
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents.retrieval.query_planner_v2 import retrieval_agent as retrieval_agent_entry


class FakeLegacyAgent:
    async def run(self, state, client):
        return {
            "runs": [
                {
                    "retrieval": {
                        "ok": True,
                        "error": None,
                        "queries_used": ["test query"],
                        "top_tables": [{"id": "legacy-table"}],
                        "metadata_used": {
                            "ticker": state["targets"][0]["ticker"],
                            "fiscal_year": state["targets"][0]["fiscal_year"],
                        },
                        "max_total_score": 0.5,
                    }
                }
            ]
        }


def _base_state():
    return {
        "targets": [{"ticker": "AAPL", "fiscal_year": 2024, "form_type": "10-K"}],
        "retrieval_plan": {"jobs": [{"job_type": "fact_lookup", "goal": "total debt"}]},
        "original_user_query": "total debt",
    }


class RetrievalAgentDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_dispatch_uses_v2(self) -> None:
        expected_output = {
            "ok": True,
            "queries_used": [],
            "metadata_used": {"retrieval_agent_flow": "query_planner_v2"},
        }

        async def fake_v2(*, state, client, retrieval_llm, reviewer_llm):
            return {"ok": True, "state": state, "retrieval": expected_output}

        with (
            patch("llm_client.build_chat_model", return_value=SimpleModel("mock")),
            patch("agents.retrieval.query_planner_v2.retrieval_agent_v2", new=AsyncMock(side_effect=fake_v2)) as mock_v2,
        ):
            result = await retrieval_agent_entry(_base_state(), client=object())

        self.assertEqual(mock_v2.await_count, 1)
        self.assertEqual(result.get("retrieval", {}).get("metadata_used", {}).get("retrieval_agent_flow"), "query_planner_v2")

    async def test_explicit_legacy_agent_bypasses_v2(self) -> None:
        with patch("agents.retrieval.query_planner_v2.retrieval_agent_v2") as mock_v2:
            result = await retrieval_agent_entry(
                _base_state(),
                client=object(),
                agent=FakeLegacyAgent(),
            )

        mock_v2.assert_not_called()
        self.assertTrue(result["retrieval"]["ok"])


class SimpleModel:
    def __init__(self, model_name: str):
        self.model_name = model_name


if __name__ == "__main__":
    unittest.main()
