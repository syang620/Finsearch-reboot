from __future__ import annotations

import asyncio
import unittest
from unittest import mock

from langgraph.checkpoint.memory import InMemorySaver

from agents.analyst import AnalystRunResult
from agents.contracts import (
    AnalysisTask,
    ContextItem,
    ContextItemKind,
    ContextQuality,
    FilingMetadata,
    FormType,
    PlannerIntent,
    SourceRef,
)
from agents.orchestrator.agent_orchestrator import (
    _get_pooled_analyst,
    _get_orchestrator_graph,
    _graph_config,
    _planner_error_node,
    _route_after_planner_turn,
    _invoke_orchestrator,
    aclose_orchestrator_runtime,
)
from tests.snapshot_utils import assert_graph_snapshot_jsonable


class _ConcurrentPlanner:
    model = "planner-model"
    enable_query_expansion = True
    auto_run_full_planner = False
    default_doc_types = []
    company_ticker_map = None
    full_planner_include_trace = False

    async def aplan_turn(self, **_kwargs):
        return {
            "planner_output": {
                "status": "completed",
                "retrieval_needed": False,
                "intent": "filing_fact",
                "metadata": {"ticker": "AAPL", "fiscal_year": 2024, "form_type": "10-K"},
                "analysis_task": {
                    "task_type": "extract",
                    "metric": "revenue",
                    "requires_calculation": False,
                    "expected_artifacts": ["text"],
                    "output_format": "short_answer",
                },
                "open_issues": [],
            }
        }


class _RetrievalPlanner:
    model = "planner-model"
    enable_query_expansion = True
    auto_run_full_planner = False
    default_doc_types = []
    company_ticker_map = None
    full_planner_include_trace = False

    async def aplan_turn(self, **_kwargs):
        return {
            "planner_output": {
                "status": "completed",
                "retrieval_needed": True,
                "intent": "filing_fact",
                "metadata": {"ticker": "AAPL", "fiscal_year": 2024, "form_type": "10-K"},
                "targets": [
                    {"target_id": 1, "ticker": "AAPL", "fiscal_year": 2024, "form_type": "10-K"}
                ],
                "retrieval_plan": {
                    "fanout_mode": "single_target",
                    "jobs": [
                        {
                            "job_type": "metric_extract",
                            "goal": "Find revenue",
                            "applies_to_target_ids": [1],
                        }
                    ],
                },
                "analysis_task": {
                    "task_type": "extract",
                    "metric": "revenue",
                    "requires_calculation": False,
                    "expected_artifacts": ["text"],
                    "output_format": "short_answer",
                },
                "open_issues": [],
            }
        }


class _ConcurrentRuntime:
    def __init__(self) -> None:
        self.closed = 0

    async def call_tool(self, _name, _args):
        return {
            "content": '{"result": 42.0}',
            "artifact": {"result": 42.0},
            "status": "success",
        }

    async def aclose(self) -> None:
        self.closed += 1


class _ConcurrentFinalAnswerModel:
    async def ainvoke(self, _messages):
        from langchain_core.messages import AIMessage

        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "FinalAnswer",
                    "id": "call-final",
                    "args": {
                        "status": "ok",
                        "answer": "Revenue is reported in the filing context.",
                        "used_context_ids": [],
                        "missing_values": [],
                        "confidence": 0.9,
                        "calculation": None,
                        "compare_rows": [],
                    },
                }
            ],
        )


class _ConcurrentModelFactory:
    def bind_tools(self, _tools):
        return _ConcurrentFinalAnswerModel()


class OrchestratorPlannerErrorTests(unittest.TestCase):
    def test_route_after_planner_turn_routes_error_and_unknown_status_to_planner_error(self) -> None:
        completed_no_error = {
            "status": "completed",
            "retrieval_needed": False,
        }
        needs_clarification = {"status": "needs_clarification"}
        planner_error = {"status": "error"}
        unknown_status = {"status": "weird_status"}

        self.assertEqual(
            _route_after_planner_turn(
                {"plan_obj": completed_no_error, "user_query": "q", "plan_id": "p"}
            ),
            "build_packet_without_retrieval",
        )
        self.assertEqual(
            _route_after_planner_turn(
                {"plan_obj": needs_clarification, "user_query": "q", "plan_id": "p"}
            ),
            "planner_interrupt",
        )
        self.assertEqual(
            _route_after_planner_turn(
                {"plan_obj": planner_error, "user_query": "q", "plan_id": "p"}
            ),
            "planner_error",
        )
        self.assertEqual(
            _route_after_planner_turn(
                {"plan_obj": unknown_status, "user_query": "q", "plan_id": "p"}
            ),
            "planner_error",
        )

    def test_planner_error_node_builds_failed_analyst_result(self) -> None:
        state = {
            "user_query": "What is revenue growth?",
            "plan_id": "run-id",
            "plan_obj": {
                "status": "error",
                "intent": "filing_fact",
                "metadata": {},
                "analysis_task": {
                    "task_type": "extract",
                    "metric": "revenue",
                    "expected_artifacts": ["table"],
                    "output_format": "step_by_step",
                },
                "open_issues": [],
            },
            "planner_turn": {
                "planner_output": {"status": "error"},
                "llm_error": "LLM parse failure",
                "validation_error": None,
            },
        }
        result = _planner_error_node(state)
        analyst_result = result["analyst_result"]
        packet = result["packet"]

        self.assertFalse(analyst_result.ok)
        self.assertEqual(analyst_result.status, "error")
        self.assertIsNotNone(analyst_result.error)
        self.assertTrue(analyst_result.error.startswith("LLM error"))
        self.assertEqual(packet.context_items, [])
        self.assertTrue(any(issue.code == "PLANNER_RUNTIME_ERROR" for issue in analyst_result.open_issues))


class OrchestratorRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self) -> None:
        await aclose_orchestrator_runtime()

    async def test_invoke_orchestrator_uses_async_state_snapshot(self) -> None:
        class _FakeGraph:
            def __init__(self) -> None:
                self.used_get_state = False
                self.used_aget_state = False

            async def ainvoke(self, _payload, config=None):
                self.last_config = config

            def get_state(self, _config):
                self.used_get_state = True
                raise AssertionError("sync get_state should not be used")

            async def aget_state(self, _config):
                self.used_aget_state = True
                return {"ok": True}

        fake_graph = _FakeGraph()

        with (
            mock.patch("agents.orchestrator.agent_orchestrator._get_orchestrator_checkpointer", new=mock.AsyncMock(return_value=object())),
            mock.patch("agents.orchestrator.agent_orchestrator._resolve_runtime_planner", new=mock.AsyncMock(return_value=None)),
            mock.patch("agents.orchestrator.agent_orchestrator._get_orchestrator_graph", return_value=fake_graph),
            mock.patch("agents.orchestrator.agent_orchestrator._graph_config", return_value={"thread_id": "run-1"}),
            mock.patch("agents.orchestrator.agent_orchestrator._format_run_output", return_value={"status": "completed"}),
            mock.patch("agents.orchestrator.agent_orchestrator._delete_thread_checkpoints", new=mock.AsyncMock()),
        ):
            output = await _invoke_orchestrator({"user_query": "q"}, run_id="run-1", planner=None)

        self.assertEqual(output["status"], "completed")
        self.assertFalse(fake_graph.used_get_state)
        self.assertTrue(fake_graph.used_aget_state)

    async def test_aclose_orchestrator_runtime_closes_cached_resources(self) -> None:
        class _FakeAnalyst:
            def __init__(self) -> None:
                self.closed = 0

            async def aclose(self) -> None:
                self.closed += 1

        class _FakeClient:
            def __init__(self) -> None:
                self.closed = 0

            async def __aexit__(self, *_args) -> None:
                self.closed += 1

        class _FakeSaver:
            def __init__(self) -> None:
                self.closed = 0

            async def __aexit__(self, *_args) -> None:
                self.closed += 1

        analyst = _FakeAnalyst()
        client = _FakeClient()
        saver = _FakeSaver()

        import agents.orchestrator.agent_orchestrator as orchestrator

        orchestrator._ANALYST_CACHE["model"] = analyst
        orchestrator._ORCHESTRATOR_MCP_CLIENT = client
        orchestrator._ORCHESTRATOR_CHECKPOINTER = saver

        await aclose_orchestrator_runtime()

        self.assertEqual(analyst.closed, 1)
        self.assertEqual(client.closed, 1)
        self.assertEqual(saver.closed, 1)
        self.assertEqual(len(orchestrator._ANALYST_CACHE), 0)

    async def test_get_pooled_analyst_reuses_single_instance_under_concurrency(self) -> None:
        created = []

        class _FakeAnalyst:
            def __init__(self, *, model, max_context_items) -> None:
                self.model = model
                self.max_context_items = max_context_items
                self.builds = 0
                created.append(self)

            @property
            def is_ready(self) -> bool:
                return self.builds > 0

            async def abuild(self):
                await asyncio.sleep(0.01)
                self.builds += 1
                return self

            async def aclose(self) -> None:
                return None

        import agents.orchestrator.agent_orchestrator as orchestrator

        await aclose_orchestrator_runtime()
        with mock.patch("agents.orchestrator.agent_orchestrator.AnalystAgent", _FakeAnalyst):
            analyst1, analyst2 = await asyncio.gather(
                _get_pooled_analyst("shared-model"),
                _get_pooled_analyst("shared-model"),
            )

        self.assertIs(analyst1, analyst2)
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].builds, 1)
        self.assertIs(orchestrator._ANALYST_CACHE["shared-model"], analyst1)

    async def test_invoke_orchestrator_concurrently_reuses_pooled_analyst(self) -> None:
        import agents.orchestrator.agent_orchestrator as orchestrator

        runtime = _ConcurrentRuntime()
        create_calls = []

        async def _fake_create(**_kwargs):
            create_calls.append(_kwargs)
            return runtime

        planner = _ConcurrentPlanner()
        await aclose_orchestrator_runtime()
        saver = InMemorySaver()
        arun_calls = []

        async def _fake_arun(self, packet, debug=False):
            arun_calls.append((id(self), packet.plan_id, debug))
            return AnalystRunResult(
                ok=True,
                status="ok",
                answer="Revenue is reported in the filing context.",
                intent=PlannerIntent.FILING_FACT,
                metric=packet.analysis_task.metric,
            )

        with (
            mock.patch("agents.analyst.agent._FinancialToolRuntime.create", new=_fake_create),
            mock.patch("agents.analyst.agent.AnalystAgent.arun", new=_fake_arun),
            mock.patch("agents.orchestrator.agent_orchestrator._get_orchestrator_checkpointer", new=mock.AsyncMock(return_value=saver)),
            mock.patch("agents.orchestrator.agent_orchestrator._orchestrator_checkpoint_ttl_seconds", return_value=0),
        ):
            output1, output2 = await asyncio.gather(
                _invoke_orchestrator(
                    {
                        "user_query": "What was revenue?",
                        "plan_id": "run-a",
                        "analyst_model": "shared-model",
                        "tables_dir": "data/chunked",
                        "debug": False,
                    },
                    run_id="run-a",
                    planner=planner,
                ),
                _invoke_orchestrator(
                    {
                        "user_query": "What was revenue?",
                        "plan_id": "run-b",
                        "analyst_model": "shared-model",
                        "tables_dir": "data/chunked",
                        "debug": False,
                    },
                    run_id="run-b",
                    planner=planner,
                ),
            )

        self.assertEqual(output1["status"], "completed")
        self.assertEqual(output2["status"], "completed")
        self.assertTrue(output1["analyst"]["ok"])
        self.assertTrue(output2["analyst"]["ok"])
        self.assertEqual(len(create_calls), 1)
        self.assertEqual(len({call[0] for call in arun_calls}), 1)
        self.assertEqual(len(arun_calls), 2)
        self.assertEqual(len(orchestrator._ANALYST_CACHE), 1)
        self.assertIn("shared-model", orchestrator._ANALYST_CACHE)

    async def test_orchestrator_graph_state_round_trips_without_retrieval(self) -> None:
        import agents.orchestrator.agent_orchestrator as orchestrator

        await aclose_orchestrator_runtime()
        saver = InMemorySaver()
        orchestrator._ORCHESTRATOR_CHECKPOINTER = saver
        orchestrator._get_orchestrator_graph.cache_clear()

        class _FakeAnalyst:
            async def arun(self, packet, debug=False):
                return AnalystRunResult(
                    ok=True,
                    status="ok",
                    answer="Revenue is reported in the filing context.",
                    intent=PlannerIntent.FILING_FACT,
                    metric=packet.analysis_task.metric,
                )

        planner = _ConcurrentPlanner()
        graph = _get_orchestrator_graph(id(saver))
        config = _graph_config(run_id="snapshot-no-retrieval", planner=planner)

        with mock.patch("agents.orchestrator.agent_orchestrator._get_pooled_analyst", new=mock.AsyncMock(return_value=_FakeAnalyst())):
            await graph.ainvoke(
                {
                    "user_query": "What was revenue?",
                    "plan_id": "snapshot-no-retrieval",
                    "analyst_model": "shared-model",
                    "tables_dir": "data/chunked",
                    "debug": False,
                },
                config=config,
            )

        snapshot = await graph.aget_state(config)
        values = dict(snapshot.values or {})
        assert_graph_snapshot_jsonable(snapshot)

        self.assertIn("packet", values)
        self.assertEqual(values["packet"].analysis_task.metric, "revenue")
        self.assertEqual(values["packet"].context_items, [])
        self.assertIn("analyst_result", values)
        self.assertTrue(values["analyst_result"].ok)
        self.assertFalse(values.get("retrieval_output"))

    async def test_orchestrator_graph_state_round_trips_with_retrieval(self) -> None:
        import agents.orchestrator.agent_orchestrator as orchestrator

        await aclose_orchestrator_runtime()
        saver = InMemorySaver()
        orchestrator._ORCHESTRATOR_CHECKPOINTER = saver
        orchestrator._get_orchestrator_graph.cache_clear()

        class _FakeAnalyst:
            async def arun(self, packet, debug=False):
                return AnalystRunResult(
                    ok=True,
                    status="ok",
                    answer="Revenue was $100.",
                    intent=PlannerIntent.FILING_FACT,
                    metric=packet.analysis_task.metric,
                )

        async def _fake_retrieval_agent(state, client=None):
            return {
                "retrieval": {
                    "ok": True,
                    "top_tables": [],
                    "partial_failures": [],
                    "metadata_used": {"ticker": "AAPL", "fiscal_year": 2024, "form_type": "10-K"},
                    "max_total_score": 12,
                }
            }

        def _fake_packet_builder(**_kwargs):
            return orchestrator._build_packet_without_retrieval(
                user_query="What was revenue?",
                plan_obj={
                    "intent": "filing_fact",
                    "metadata": {"ticker": "AAPL", "fiscal_year": 2024, "form_type": "10-K"},
                    "analysis_task": {
                        "task_type": "extract",
                        "metric": "revenue",
                        "requires_calculation": False,
                        "expected_artifacts": ["text"],
                        "output_format": "short_answer",
                    },
                },
                plan_id="snapshot-with-retrieval",
            ).model_copy(
                update={
                    "context_quality": ContextQuality.MEDIUM,
                    "context_items": [
                        ContextItem(
                            context_id="ctx_1",
                            target_id="1",
                            kind=ContextItemKind.TEXT,
                            source=SourceRef(ticker="AAPL", fiscal_year=2024, form_type=FormType.TEN_K),
                            payload={"content": "Revenue was $100."},
                        )
                    ],
                }
            )

        planner = _RetrievalPlanner()
        graph = _get_orchestrator_graph(id(saver))
        config = _graph_config(run_id="snapshot-with-retrieval", planner=planner)

        with (
            mock.patch("agents.orchestrator.agent_orchestrator._get_pooled_analyst", new=mock.AsyncMock(return_value=_FakeAnalyst())),
            mock.patch("agents.orchestrator.agent_orchestrator._get_orchestrator_mcp_client", new=mock.AsyncMock(return_value=object())),
            mock.patch("agents.orchestrator.agent_orchestrator.retrieval_agent", new=_fake_retrieval_agent),
            mock.patch("agents.orchestrator.agent_orchestrator.build_packet_from_retrieval_output", new=_fake_packet_builder),
        ):
            await graph.ainvoke(
                {
                    "user_query": "What was revenue?",
                    "plan_id": "snapshot-with-retrieval",
                    "analyst_model": "shared-model",
                    "tables_dir": "data/chunked",
                    "debug": False,
                },
                config=config,
            )

        snapshot = await graph.aget_state(config)
        values = dict(snapshot.values or {})
        assert_graph_snapshot_jsonable(snapshot)

        self.assertIn("retrieval_output", values)
        self.assertTrue(values["retrieval_output"]["ok"])
        self.assertIn("packet", values)
        self.assertEqual(len(values["packet"].context_items), 1)
        self.assertEqual(values["packet"].context_items[0].payload["content"], "Revenue was $100.")
        self.assertIn("analyst_result", values)
        self.assertTrue(values["analyst_result"].ok)


if __name__ == "__main__":
    unittest.main()
