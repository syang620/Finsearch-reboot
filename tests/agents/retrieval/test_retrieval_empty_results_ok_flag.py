import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from agents.orchestrator.agent_orchestrator import _compact_retrieval_result_for_user
from agents.retrieval.query_planner_v2 import _build_retrieval_output


class RetrievalOkFlagTests(unittest.TestCase):
    def test_v2_empty_tables_no_errors_is_not_ok(self) -> None:
        output = _build_retrieval_output(
            state={
                "targets": [{"ticker": "AAPL", "fiscal_year": 2024}],
                "original_user_query": "cash flow trend",
            },
            runs_payload=[
                {
                    "final_retrieval": {
                        "top_tables": [],
                        "queries_used": ["cash flow trend"],
                        "error": None,
                    }
                }
            ],
            model_name="test-model",
        )
        self.assertFalse(output.get("ok"))

    def test_v2_with_tables_is_ok(self) -> None:
        output = _build_retrieval_output(
            state={
                "targets": [{"ticker": "AAPL", "fiscal_year": 2024}],
                "original_user_query": "cash flow trend",
            },
            runs_payload=[
                {
                    "final_retrieval": {
                        "top_tables": [{"id": "t1"}],
                        "queries_used": ["cash flow trend"],
                        "error": None,
                    }
                }
            ],
            model_name="test-model",
        )
        self.assertTrue(output.get("ok"))

    def test_compacted_retrieval_uses_ok_flag_only(self) -> None:
        compact = _compact_retrieval_result_for_user(
            retrieval_output={
                "ok": False,
                "top_tables": [{}],
                "targets": [{"ticker": "AAPL", "fiscal_year": 2024}],
            }
        )
        self.assertFalse(compact.get("ok"))
        self.assertEqual(compact.get("retrieved_candidate_count"), 1)


if __name__ == "__main__":
    unittest.main()
