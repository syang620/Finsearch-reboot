from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError

from agents.analyst import AnalystRunResult
from agents.contracts import (
    AnalysisTask,
    ContextItem,
    ContextItemKind,
    ContextQuality,
    FilingMetadata,
    FormType,
    PlannerIntent,
    PlannerRuntimeOutput,
    SourceRef,
)
from agents.orchestrator.agent_orchestrator import (
    _build_runtime_contract_error_plan,
    _format_run_output,
    _format_runtime_contract_validation_error,
    _get_pooled_analyst,
    _get_orchestrator_checkpointer,
    _get_orchestrator_graph,
    _graph_config,
    _planner_error_node,
    _resolve_metric_id_for_structured_fact_request,
    _route_after_planner_turn,
    _structured_fact_capability_decisions,
    _structured_facts_node,
    _invoke_orchestrator,
    aclose_orchestrator_runtime,
)
from tests.snapshot_utils import assert_graph_snapshot_jsonable


def test_orchestrator_capability_guard_uses_metric_clarification_answer() -> None:
    decisions = _structured_fact_capability_decisions(
        plan_obj={
            "original_user_query": "What was Apple's cash?",
            "effective_user_query": (
                "What was Apple's cash?\n\nAnswer: cash and cash equivalents"
            ),
            "clarification_history": [
                {
                    "question": "Which precise financial metric did you mean?",
                    "answer": "cash and cash equivalents",
                }
            ],
            "targets": [
                {"company_name": "Apple", "ticker": "AAPL"},
            ],
        },
        requests=[
            {
                "metric_hint": "cash and cash equivalents",
                "subquestion": "What were Apple's cash and cash equivalents?",
                "entity_hint": "Apple",
            }
        ],
    )

    assert decisions[0].permitted


def test_orchestrator_capability_guard_rejects_nonannual_target() -> None:
    decisions = _structured_fact_capability_decisions(
        plan_obj={
            "original_user_query": "What revenue did Apple report in its 2024 10-Q?",
            "targets": [{"ticker": "AAPL", "fiscal_year": 2024, "form_type": "10-Q"}],
        },
        requests=[{"metric_hint": "revenue", "subquestion": "What was revenue?"}],
    )

    assert not decisions[0].permitted
    assert decisions[0].question_class.value == "unsupported_derived_metric"


def _runtime_output(
    *,
    query: str,
    fiscal_year: int,
    intent: str = "filing_fact",
    route: str = "kb",
    retrieval_needed: bool = False,
    structured_fact_requests: list[dict] | None = None,
) -> dict:
    targets = []
    retrieval_plan = None
    if retrieval_needed:
        targets = [
            {
                "target_id": 1,
                "target_key": f"AAPL_FY{fiscal_year}",
                "company_name": "Apple",
                "ticker": "AAPL",
                "fiscal_year": fiscal_year,
                "form_type": "10-K",
            }
        ]
        retrieval_plan = {
            "fanout_mode": "single_target",
            "jobs": [
                {
                    "job_type": "metric_extract",
                    "goal": "Find revenue",
                    "applies_to_target_ids": [1],
                }
            ],
        }
    return {
        "status": "completed",
        "retrieval_needed": retrieval_needed,
        "intent": intent,
        "route": route,
        "structured_fact_requests": list(structured_fact_requests or []),
        "metadata": {
            "ticker": "AAPL",
            "fiscal_year": fiscal_year,
            "form_type": "10-K",
        },
        "analysis_task": {
            "task_type": "extract",
            "metric": "revenue",
            "requires_calculation": False,
            "expected_artifacts": ["text"],
            "output_format": "short_answer",
        },
        "task_class": "single_target_fact",
        "targets": targets,
        "retrieval_plan": retrieval_plan,
        "open_issues": [],
        "original_user_query": query,
        "effective_user_query": query,
        "clarification_history": [],
        "clarification_request": None,
    }


class _ConcurrentPlanner:
    model = "planner-model"
    enable_query_expansion = True
    auto_run_full_planner = False
    default_doc_types = []
    company_ticker_map = None
    full_planner_include_trace = False

    async def aplan_turn(self, **_kwargs):
        return {
            "planner_output": _runtime_output(
                query="What is revenue?",
                fiscal_year=2024,
                intent="definition",
            )
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
            "planner_output": _runtime_output(
                query="What was revenue?",
                fiscal_year=2024,
                retrieval_needed=True,
            )
        }


class _KBRoutePlanner(_RetrievalPlanner):
    async def aplan_turn(self, **_kwargs):
        planner_turn = await super().aplan_turn(**_kwargs)
        planner_turn["planner_output"]["route"] = "kb"
        planner_turn["planner_output"]["structured_fact_requests"] = []
        return planner_turn


class _StructuredFactPlanner:
    model = "planner-model"
    enable_query_expansion = True
    auto_run_full_planner = False
    default_doc_types = []
    company_ticker_map = None
    full_planner_include_trace = False

    def __init__(
        self,
        *,
        metric_hint: str = "revenue",
        subquestion: str = "What was Apple revenue in FY2025?",
        entity_hint: str = "Apple",
    ) -> None:
        self.metric_hint = metric_hint
        self.subquestion = subquestion
        self.entity_hint = entity_hint

    async def aplan_turn(self, **_kwargs):
        return {
            "planner_output": _runtime_output(
                query=self.subquestion,
                fiscal_year=2025,
                route="structured_fact",
                structured_fact_requests=[
                    {
                        "subquestion": self.subquestion,
                        "metric_hint": self.metric_hint,
                        "entity_hint": self.entity_hint,
                        "fiscal_year": 2025,
                        "fiscal_period": "FY",
                    }
                ],
            )
        }


class _HybridPlanner(_RetrievalPlanner):
    async def aplan_turn(self, **_kwargs):
        planner_turn = await super().aplan_turn(**_kwargs)
        planner_turn["planner_output"]["route"] = "hybrid"
        planner_turn["planner_output"]["structured_fact_requests"] = [
            {
                "subquestion": "What was Apple revenue in FY2024?",
                "metric_hint": "revenue",
                "entity_hint": "Apple",
                "fiscal_year": 2024,
                "fiscal_period": "FY",
            }
        ]
        return planner_turn


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


class _FakeStructuredFactClient:
    def __init__(self, tool_result=None) -> None:
        self.tool_result = tool_result or {
            "ok": True,
            "status": "ok",
            "metric_id": "revenue",
            "value": 410000000000.0,
            "unit": "USD",
            "ticker": "AAPL",
            "cik": "0000320193",
            "fiscal_year": 2025,
            "form_type": "10-K",
            "accession_number": "0000320193-25-000073",
            "report_date": "2025-09-27",
            "filed_date": "2025-10-31",
            "source_url": "https://www.sec.gov/example",
        }
        self.calls = []

    async def get_metric(self, *, ticker: str, fiscal_year: int, metric_id: str):
        self.calls.append(
            {
                "ticker": ticker,
                "fiscal_year": fiscal_year,
                "metric_id": metric_id,
            }
        )
        return dict(self.tool_result)


class OrchestratorPlannerErrorTests(unittest.TestCase):
    def test_insufficient_data_analyst_result_completes_runtime(self) -> None:
        snapshot = mock.Mock()
        snapshot.values = {
            "plan_obj": {
                "status": "completed",
                "route": "kb",
                "retrieval_needed": True,
            },
            "retrieval_state": {"targets": [{"ticker": "AAPL", "fiscal_year": 2024}]},
            "retrieval_output": {"ok": True},
            "analyst_result": AnalystRunResult(
                ok=True,
                status="insufficient_data",
                answer="The filing does not contain the required market price.",
                intent=PlannerIntent.FILING_CALC,
                metric="price-to-earnings ratio",
                missing_values=["market share price"],
                error=None,
            ),
            "total_ms": 1,
        }
        snapshot.interrupts = ()

        output = _format_run_output(run_id="run-id", state_snapshot=snapshot)

        self.assertTrue(output["ok"])
        self.assertEqual(output["status"], "completed")
        self.assertEqual(output["failure_stage"], "none")
        self.assertEqual(output["analyst"]["status"], "insufficient_data")

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

    def test_runtime_contract_error_plan_is_valid_and_not_duplicated(self) -> None:
        plan_obj = _build_runtime_contract_error_plan(
            user_query="What was revenue?",
            clarification_history=[],
            validation_error="PLANNER_RUNTIME_CONTRACT_INVALID: missing route",
        )
        PlannerRuntimeOutput.model_validate(plan_obj)

        result = _planner_error_node(
            {
                "user_query": "What was revenue?",
                "plan_id": "run-id",
                "plan_obj": plan_obj,
                "planner_turn": {
                    "planner_output": plan_obj,
                    "validation_error": "PLANNER_RUNTIME_CONTRACT_INVALID: missing route",
                    "runtime_contract_error": True,
                },
            }
        )

        issue_codes = [
            issue.code for issue in result["analyst_result"].open_issues
        ]
        self.assertEqual(issue_codes.count("PLANNER_RUNTIME_CONTRACT_INVALID"), 1)

    def test_runtime_contract_validation_error_omits_input_values(self) -> None:
        payload = _runtime_output(
            query="What was revenue?",
            fiscal_year=2024,
            retrieval_needed=True,
        )
        payload["route"] = "sensitive-route-value"
        payload["targets"][0]["target_id"] = 0

        with self.assertRaises(ValidationError) as caught:
            PlannerRuntimeOutput.model_validate(payload)

        message = _format_runtime_contract_validation_error(caught.exception)

        self.assertIn("planner_output.route: Input should be", message)
        self.assertIn(
            "planner_output.targets.0.target_id: Input should be greater than or equal to 1",
            message,
        )
        self.assertNotIn("sensitive-route-value", message)
        self.assertNotIn("input_value", message)

    def test_runtime_contract_validation_error_is_capped(self) -> None:
        payload = _runtime_output(
            query="What was revenue?",
            fiscal_year=2024,
            retrieval_needed=True,
        )
        for field_name in (
            "status",
            "retrieval_needed",
            "intent",
            "route",
            "structured_fact_requests",
            "metadata",
        ):
            del payload[field_name]

        with self.assertRaises(ValidationError) as caught:
            PlannerRuntimeOutput.model_validate(payload)

        message = _format_runtime_contract_validation_error(caught.exception)

        self.assertEqual(
            message.splitlines()[-1],
            "... 1 additional validation error omitted",
        )
        self.assertEqual(len(message.splitlines()), 6)

    def test_resolver_keeps_specific_total_debt_alias(self) -> None:
        metric_id, status, _reason = _resolve_metric_id_for_structured_fact_request(
            metric_hint="total debt",
            subquestion="What was total debt?",
        )

        self.assertEqual(metric_id, "total_debt")
        self.assertEqual(status, "resolved")

    def test_resolver_no_longer_maps_profit_to_net_income(self) -> None:
        metric_id, status, _reason = _resolve_metric_id_for_structured_fact_request(
            metric_hint="profit",
            subquestion="What was profit?",
        )

        self.assertIsNone(metric_id)
        self.assertEqual(status, "unresolved")

    def test_resolver_no_longer_maps_assets_to_total_assets(self) -> None:
        metric_id, status, _reason = _resolve_metric_id_for_structured_fact_request(
            metric_hint="assets",
            subquestion="What were assets?",
        )

        self.assertIsNone(metric_id)
        self.assertEqual(status, "unresolved")

    def test_resolver_no_longer_maps_equity_to_stockholders_equity(self) -> None:
        metric_id, status, _reason = _resolve_metric_id_for_structured_fact_request(
            metric_hint="equity",
            subquestion="What was equity?",
        )

        self.assertIsNone(metric_id)
        self.assertEqual(status, "unresolved")

    def test_resolver_keeps_cash_ambiguous(self) -> None:
        metric_id, status, _reason = _resolve_metric_id_for_structured_fact_request(
            metric_hint="cash",
            subquestion="What was cash?",
        )

        self.assertIsNone(metric_id)
        self.assertEqual(status, "ambiguous")


class OrchestratorRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self) -> None:
        await aclose_orchestrator_runtime()

    async def _track_blocked_prune(self, orchestrator):
        prune_cancelled = asyncio.Event()
        allow_prune_to_finish = asyncio.Event()

        async def _pending_prune() -> None:
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                prune_cancelled.set()
                await allow_prune_to_finish.wait()
                raise

        prune_task = asyncio.create_task(_pending_prune())
        orchestrator._BACKGROUND_TASKS.add(prune_task)
        prune_task.add_done_callback(orchestrator._observe_background_task)
        await asyncio.sleep(0)
        return prune_task, prune_cancelled, allow_prune_to_finish

    async def _assert_checkpointer_initialization_cleanup(
        self,
        failure: BaseException,
        *,
        cleanup_failure: BaseException | None = None,
    ) -> None:
        import agents.orchestrator.agent_orchestrator as orchestrator

        class _FailingConnection:
            async def execute(self, _sql: str) -> None:
                raise failure

        class _FakeSaver:
            def __init__(self) -> None:
                self.conn = _FailingConnection()

        class _FakeOwner:
            def __init__(self) -> None:
                self.exited = 0
                self.exit_args = None

            async def __aenter__(self):
                return _FakeSaver()

            async def __aexit__(self, *args):
                self.exited += 1
                self.exit_args = args
                if cleanup_failure is not None:
                    raise cleanup_failure

        await aclose_orchestrator_runtime()
        owner = _FakeOwner()
        with (
            mock.patch(
                "agents.orchestrator.agent_orchestrator._orchestrator_checkpoint_path",
                return_value=Path("unused-checkpointer.sqlite"),
            ),
            mock.patch.object(
                orchestrator.AsyncSqliteSaver,
                "from_conn_string",
                return_value=owner,
            ),
            mock.patch.object(orchestrator.logger, "exception") as log_exception,
        ):
            caught = None
            try:
                await _get_orchestrator_checkpointer()
            except BaseException as exc:
                caught = exc

        self.assertIs(caught, failure)
        self.assertEqual(owner.exited, 1)
        self.assertIs(owner.exit_args[1], failure)
        self.assertIsNone(orchestrator._ORCHESTRATOR_CHECKPOINTER)
        self.assertIsNone(orchestrator._ORCHESTRATOR_CHECKPOINTER_OWNER)
        self.assertEqual(log_exception.call_count, int(cleanup_failure is not None))

    async def test_real_sqlite_checkpointer_lifetime_and_reinitialization(self) -> None:
        import agents.orchestrator.agent_orchestrator as orchestrator

        async def _select_one(saver) -> tuple[int]:
            async with saver.conn.execute("SELECT 1") as cursor:
                return await cursor.fetchone()

        await aclose_orchestrator_runtime()
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "orchestrator.sqlite"
            with mock.patch(
                "agents.orchestrator.agent_orchestrator._orchestrator_checkpoint_path",
                return_value=checkpoint_path,
            ):
                first = await _get_orchestrator_checkpointer()
                self.assertEqual(await _select_one(first), (1,))

                reused = await _get_orchestrator_checkpointer()
                self.assertIs(reused, first)
                self.assertEqual(await _select_one(reused), (1,))

                await aclose_orchestrator_runtime()
                self.assertIsNone(orchestrator._ORCHESTRATOR_CHECKPOINTER)
                self.assertIsNone(orchestrator._ORCHESTRATOR_CHECKPOINTER_OWNER)
                with self.assertRaises(Exception):
                    await first.conn.execute("SELECT 1")

                second = await _get_orchestrator_checkpointer()
                self.assertIsNot(second, first)
                self.assertEqual(await _select_one(second), (1,))
                await aclose_orchestrator_runtime()

    async def test_shutdown_drains_prune_tasks_before_checkpointer_teardown(self) -> None:
        import agents.orchestrator.agent_orchestrator as orchestrator

        async def _select_one(saver) -> tuple[int]:
            async with saver.conn.execute("SELECT 1") as cursor:
                return await cursor.fetchone()

        prune_cancelled = asyncio.Event()
        allow_prune_to_finish = asyncio.Event()

        async def _pending_prune() -> None:
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                prune_cancelled.set()
                await allow_prune_to_finish.wait()
                raise

        await aclose_orchestrator_runtime()
        original_factory = orchestrator.AsyncSqliteSaver.from_conn_string
        factory_paths: list[str] = []

        def _tracked_factory(path: str):
            factory_paths.append(path)
            return original_factory(path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "orchestrator.sqlite"
            with (
                mock.patch(
                    "agents.orchestrator.agent_orchestrator._orchestrator_checkpoint_path",
                    return_value=checkpoint_path,
                ),
                mock.patch.object(
                    orchestrator.AsyncSqliteSaver,
                    "from_conn_string",
                    side_effect=_tracked_factory,
                ),
            ):
                first = await _get_orchestrator_checkpointer()
                self.assertEqual(await _select_one(first), (1,))
                self.assertEqual(len(factory_paths), 1)

                prune_task = asyncio.create_task(_pending_prune())
                orchestrator._BACKGROUND_TASKS.add(prune_task)
                prune_task.add_done_callback(orchestrator._observe_background_task)
                await asyncio.sleep(0)

                close_task = asyncio.create_task(aclose_orchestrator_runtime())
                await asyncio.wait_for(prune_cancelled.wait(), timeout=1)

                with self.assertRaisesRegex(RuntimeError, "lifecycle is closing"):
                    await asyncio.wait_for(
                        _get_orchestrator_checkpointer(),
                        timeout=1,
                    )
                self.assertEqual(len(factory_paths), 1)

                allow_prune_to_finish.set()
                await asyncio.wait_for(close_task, timeout=1)

                self.assertTrue(prune_task.cancelled())
                self.assertEqual(orchestrator._BACKGROUND_TASKS, set())
                self.assertIsNone(orchestrator._ORCHESTRATOR_CHECKPOINTER)
                self.assertIsNone(orchestrator._ORCHESTRATOR_CHECKPOINTER_OWNER)
                self.assertFalse(orchestrator._ORCHESTRATOR_CHECKPOINTER_CLOSING)

                second = await _get_orchestrator_checkpointer()
                self.assertIsNot(second, first)
                self.assertEqual(len(factory_paths), 2)
                self.assertEqual(await _select_one(second), (1,))
                await aclose_orchestrator_runtime()

    async def test_cancelled_close_caller_does_not_strand_teardown(self) -> None:
        import agents.orchestrator.agent_orchestrator as orchestrator

        class _FakeOwner:
            def __init__(self) -> None:
                self.closed = 0

            async def __aexit__(self, *_args) -> None:
                self.closed += 1

        async def _select_one(saver) -> tuple[int]:
            async with saver.conn.execute("SELECT 1") as cursor:
                return await cursor.fetchone()

        await aclose_orchestrator_runtime()
        owner = _FakeOwner()
        orchestrator._ORCHESTRATOR_CHECKPOINTER = object()
        orchestrator._ORCHESTRATOR_CHECKPOINTER_OWNER = owner
        prune_task, prune_cancelled, allow_prune_to_finish = (
            await self._track_blocked_prune(orchestrator)
        )

        close_caller = asyncio.create_task(aclose_orchestrator_runtime())
        await asyncio.wait_for(prune_cancelled.wait(), timeout=1)
        shared_teardown = orchestrator._ORCHESTRATOR_CHECKPOINTER_CLOSE_TASK
        self.assertIsNotNone(shared_teardown)
        self.assertFalse(shared_teardown.done())

        close_caller.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await close_caller

        self.assertFalse(shared_teardown.cancelled())
        self.assertFalse(shared_teardown.done())
        self.assertTrue(orchestrator._ORCHESTRATOR_CHECKPOINTER_CLOSING)
        self.assertEqual(owner.closed, 0)

        second_caller = asyncio.create_task(aclose_orchestrator_runtime())
        await asyncio.sleep(0)
        self.assertFalse(second_caller.done())
        self.assertIs(
            orchestrator._ORCHESTRATOR_CHECKPOINTER_CLOSE_TASK,
            shared_teardown,
        )

        allow_prune_to_finish.set()
        await asyncio.wait_for(second_caller, timeout=1)

        self.assertTrue(shared_teardown.done())
        self.assertFalse(shared_teardown.cancelled())
        self.assertTrue(prune_task.cancelled())
        self.assertEqual(orchestrator._BACKGROUND_TASKS, set())
        self.assertIsNone(orchestrator._ORCHESTRATOR_CHECKPOINTER)
        self.assertIsNone(orchestrator._ORCHESTRATOR_CHECKPOINTER_OWNER)
        self.assertFalse(orchestrator._ORCHESTRATOR_CHECKPOINTER_CLOSING)
        self.assertEqual(owner.closed, 1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "orchestrator.sqlite"
            with mock.patch(
                "agents.orchestrator.agent_orchestrator._orchestrator_checkpoint_path",
                return_value=checkpoint_path,
            ):
                saver = await _get_orchestrator_checkpointer()
                self.assertEqual(await _select_one(saver), (1,))
                await aclose_orchestrator_runtime()

    async def test_concurrent_close_callers_await_shared_teardown(self) -> None:
        import agents.orchestrator.agent_orchestrator as orchestrator

        class _FakeOwner:
            def __init__(self) -> None:
                self.closed = 0

            async def __aexit__(self, *_args) -> None:
                self.closed += 1

        await aclose_orchestrator_runtime()
        owner = _FakeOwner()
        orchestrator._ORCHESTRATOR_CHECKPOINTER = object()
        orchestrator._ORCHESTRATOR_CHECKPOINTER_OWNER = owner
        prune_task, prune_cancelled, allow_prune_to_finish = (
            await self._track_blocked_prune(orchestrator)
        )

        caller_a = asyncio.create_task(aclose_orchestrator_runtime())
        await asyncio.wait_for(prune_cancelled.wait(), timeout=1)
        shared_teardown = orchestrator._ORCHESTRATOR_CHECKPOINTER_CLOSE_TASK
        self.assertIsNotNone(shared_teardown)

        caller_b = asyncio.create_task(aclose_orchestrator_runtime())
        await asyncio.sleep(0)

        self.assertFalse(caller_a.done())
        self.assertFalse(caller_b.done())
        self.assertFalse(shared_teardown.done())
        self.assertIs(
            orchestrator._ORCHESTRATOR_CHECKPOINTER_CLOSE_TASK,
            shared_teardown,
        )
        self.assertEqual(owner.closed, 0)

        allow_prune_to_finish.set()
        await asyncio.wait_for(asyncio.gather(caller_a, caller_b), timeout=1)

        self.assertTrue(shared_teardown.done())
        self.assertTrue(prune_task.cancelled())
        self.assertEqual(orchestrator._BACKGROUND_TASKS, set())
        self.assertIsNone(orchestrator._ORCHESTRATOR_CHECKPOINTER)
        self.assertIsNone(orchestrator._ORCHESTRATOR_CHECKPOINTER_OWNER)
        self.assertFalse(orchestrator._ORCHESTRATOR_CHECKPOINTER_CLOSING)
        self.assertEqual(owner.closed, 1)

    async def test_checkpointer_initialization_failure_preserves_original_error(self) -> None:
        await self._assert_checkpointer_initialization_cleanup(
            RuntimeError("setup failed"),
            cleanup_failure=ValueError("cleanup failed"),
        )

    async def test_checkpointer_initialization_cancellation_cleans_up(self) -> None:
        await self._assert_checkpointer_initialization_cleanup(
            asyncio.CancelledError("initialization cancelled")
        )

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

    async def test_malformed_planner_payload_fails_at_planner_boundary(self) -> None:
        class _MalformedPlanner:
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
                        "unexpected": object(),
                    }
                }

        await aclose_orchestrator_runtime()
        saver = InMemorySaver()
        analyst = mock.AsyncMock()
        with (
            mock.patch(
                "agents.orchestrator.agent_orchestrator._get_orchestrator_checkpointer",
                new=mock.AsyncMock(return_value=saver),
            ),
            mock.patch(
                "agents.orchestrator.agent_orchestrator._orchestrator_checkpoint_ttl_seconds",
                return_value=0,
            ),
            mock.patch(
                "agents.orchestrator.agent_orchestrator._get_pooled_analyst",
                new=analyst,
            ),
        ):
            output = await _invoke_orchestrator(
                {
                    "user_query": "What was revenue?",
                    "plan_id": "invalid-plan",
                    "analyst_model": "shared-model",
                    "tables_dir": "data/chunked",
                    "debug": False,
                },
                run_id="invalid-plan",
                planner=_MalformedPlanner(),
            )

        self.assertEqual(output["status"], "failed")
        self.assertEqual(output["failure_stage"], "planner")
        self.assertEqual(output["planner"]["status"], "error")
        self.assertIn(
            "invalid_planner_output",
            output["planner_turn"],
        )
        self.assertEqual(
            set(output["planner_turn"]["invalid_planner_output"]),
            {"repr"},
        )
        self.assertFalse(output["retrieval"]["ok"])
        self.assertEqual(output["retrieval"]["attempts"], [])
        self.assertEqual(output["structured_fact_results"], [])
        self.assertFalse(analyst.await_count)
        self.assertEqual(
            [issue["code"] for issue in output["open_issues"]].count(
                "PLANNER_RUNTIME_CONTRACT_INVALID"
            ),
            1,
        )

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

        class _FakeOwner:
            def __init__(self) -> None:
                self.closed = 0

            async def __aexit__(self, *_args) -> None:
                self.closed += 1

        analyst = _FakeAnalyst()
        client = _FakeClient()
        saver = object()
        owner = _FakeOwner()

        import agents.orchestrator.agent_orchestrator as orchestrator

        orchestrator._ANALYST_CACHE["model"] = analyst
        orchestrator._ORCHESTRATOR_MCP_CLIENT = client
        orchestrator._ORCHESTRATOR_CHECKPOINTER = saver
        orchestrator._ORCHESTRATOR_CHECKPOINTER_OWNER = owner

        await asyncio.gather(
            aclose_orchestrator_runtime(),
            aclose_orchestrator_runtime(),
        )
        await aclose_orchestrator_runtime()

        self.assertEqual(analyst.closed, 1)
        self.assertEqual(client.closed, 1)
        self.assertEqual(owner.closed, 1)
        self.assertEqual(len(orchestrator._ANALYST_CACHE), 0)
        self.assertIsNone(orchestrator._ORCHESTRATOR_CHECKPOINTER)
        self.assertIsNone(orchestrator._ORCHESTRATOR_CHECKPOINTER_OWNER)

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

    async def test_invoke_orchestrator_kb_route_skips_structured_fact_execution(self) -> None:
        client = _FakeStructuredFactClient()

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
            from agents.orchestrator import agent_orchestrator as orchestrator

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
                plan_id="kb-run",
            )

        class _FakeAnalyst:
            async def arun(self, packet, debug=False):
                return AnalystRunResult(
                    ok=True,
                    status="ok",
                    answer="Revenue is in KB context.",
                    intent=PlannerIntent.FILING_FACT,
                    metric=packet.analysis_task.metric,
                )

        await aclose_orchestrator_runtime()
        saver = InMemorySaver()
        with (
            mock.patch("agents.orchestrator.agent_orchestrator._get_orchestrator_checkpointer", new=mock.AsyncMock(return_value=saver)),
            mock.patch("agents.orchestrator.agent_orchestrator._orchestrator_checkpoint_ttl_seconds", return_value=0),
            mock.patch("agents.orchestrator.agent_orchestrator._get_orchestrator_mcp_client", new=mock.AsyncMock(return_value=client)),
            mock.patch("agents.orchestrator.agent_orchestrator.retrieval_agent", new=_fake_retrieval_agent),
            mock.patch("agents.orchestrator.agent_orchestrator.build_packet_from_retrieval_output", new=_fake_packet_builder),
            mock.patch("agents.orchestrator.agent_orchestrator._get_pooled_analyst", new=mock.AsyncMock(return_value=_FakeAnalyst())),
        ):
            output = await _invoke_orchestrator(
                {
                    "user_query": "What was revenue?",
                    "plan_id": "kb-run",
                    "analyst_model": "shared-model",
                    "tables_dir": "data/chunked",
                    "debug": False,
                },
                run_id="kb-run",
                planner=_KBRoutePlanner(),
            )

        self.assertEqual(output["route"], "kb")
        self.assertEqual(output["structured_fact_results"], [])
        self.assertEqual(client.calls, [])
        self.assertTrue(output["retrieval"]["ok"])

    async def test_orchestrator_graph_state_round_trips_with_structured_fact_route(self) -> None:
        import agents.orchestrator.agent_orchestrator as orchestrator

        await aclose_orchestrator_runtime()
        saver = InMemorySaver()
        orchestrator._ORCHESTRATOR_CHECKPOINTER = saver
        orchestrator._get_orchestrator_graph.cache_clear()
        client = _FakeStructuredFactClient()

        class _FakeAnalyst:
            async def arun(self, packet, debug=False):
                return AnalystRunResult(
                    ok=True,
                    status="ok",
                    answer="Revenue was $410B.",
                    intent=PlannerIntent.FILING_FACT,
                    metric=packet.analysis_task.metric,
                )

        graph = _get_orchestrator_graph(id(saver))
        config = _graph_config(run_id="snapshot-structured-fact", planner=_StructuredFactPlanner())

        with (
            mock.patch("agents.orchestrator.agent_orchestrator._get_pooled_analyst", new=mock.AsyncMock(return_value=_FakeAnalyst())),
            mock.patch("agents.orchestrator.agent_orchestrator._get_orchestrator_mcp_client", new=mock.AsyncMock(return_value=client)),
        ):
            await graph.ainvoke(
                {
                    "user_query": "What was Apple revenue in FY2025?",
                    "plan_id": "snapshot-structured-fact",
                    "analyst_model": "shared-model",
                    "tables_dir": "data/chunked",
                    "debug": False,
                },
                config=config,
            )

        snapshot = await graph.aget_state(config)
        values = dict(snapshot.values or {})
        assert_graph_snapshot_jsonable(snapshot)

        self.assertEqual(values["structured_fact_results"][0]["resolved_metric_id"], "revenue")
        self.assertFalse(values.get("retrieval_output"))
        self.assertEqual(len(values["packet"].context_items), 1)
        self.assertIn("Structured fact: revenue = 410000000000.0 USD", values["packet"].context_items[0].payload["content"])
        self.assertEqual(client.calls[0]["metric_id"], "revenue")

    async def test_invoke_orchestrator_hybrid_route_returns_retrieval_and_structured_fact_results(self) -> None:
        client = _FakeStructuredFactClient(
            tool_result={
                "ok": True,
                "status": "ok",
                "metric_id": "revenue",
                "value": 391000000000.0,
                "unit": "USD",
                "ticker": "AAPL",
                "cik": "0000320193",
                "fiscal_year": 2024,
                "form_type": "10-K",
                "accession_number": "0000320193-24-000123",
                "report_date": "2024-09-28",
                "filed_date": "2024-11-01",
                "source_url": "https://www.sec.gov/example",
            }
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
            from agents.orchestrator import agent_orchestrator as orchestrator

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
                plan_id="hybrid-run",
            ).model_copy(
                update={
                    "context_quality": ContextQuality.MEDIUM,
                    "context_items": [
                        ContextItem(
                            context_id="ctx_1",
                            target_id="1",
                            kind=ContextItemKind.TEXT,
                            source=SourceRef(ticker="AAPL", fiscal_year=2024, form_type=FormType.TEN_K),
                            payload={"content": "Revenue was supported by KB context."},
                        )
                    ],
                }
            )

        class _FakeAnalyst:
            async def arun(self, packet, debug=False):
                return AnalystRunResult(
                    ok=True,
                    status="ok",
                    answer="Revenue was supported by KB and structured facts.",
                    intent=PlannerIntent.FILING_FACT,
                    metric=packet.analysis_task.metric,
                )

        await aclose_orchestrator_runtime()
        saver = InMemorySaver()
        with (
            mock.patch("agents.orchestrator.agent_orchestrator._get_orchestrator_checkpointer", new=mock.AsyncMock(return_value=saver)),
            mock.patch("agents.orchestrator.agent_orchestrator._orchestrator_checkpoint_ttl_seconds", return_value=0),
            mock.patch("agents.orchestrator.agent_orchestrator._get_orchestrator_mcp_client", new=mock.AsyncMock(return_value=client)),
            mock.patch("agents.orchestrator.agent_orchestrator.retrieval_agent", new=_fake_retrieval_agent),
            mock.patch("agents.orchestrator.agent_orchestrator.build_packet_from_retrieval_output", new=_fake_packet_builder),
            mock.patch("agents.orchestrator.agent_orchestrator._get_pooled_analyst", new=mock.AsyncMock(return_value=_FakeAnalyst())),
        ):
            output = await _invoke_orchestrator(
                {
                    "user_query": "What was revenue?",
                    "plan_id": "hybrid-run",
                    "analyst_model": "shared-model",
                    "tables_dir": "data/chunked",
                    "debug": False,
                },
                run_id="hybrid-run",
                planner=_HybridPlanner(),
            )

        self.assertEqual(output["route"], "hybrid")
        self.assertTrue(output["retrieval"]["ok"])
        self.assertEqual(len(output["structured_fact_results"]), 1)
        self.assertEqual(output["structured_fact_results"][0]["resolved_metric_id"], "revenue")
        self.assertEqual(client.calls[0]["metric_id"], "revenue")

    async def test_invoke_orchestrator_hybrid_route_preserves_structured_fact_when_kb_retrieval_fails(self) -> None:
        client = _FakeStructuredFactClient(
            tool_result={
                "ok": True,
                "status": "ok",
                "metric_id": "revenue",
                "value": 391000000000.0,
                "unit": "USD",
                "ticker": "AAPL",
                "cik": "0000320193",
                "fiscal_year": 2024,
                "form_type": "10-K",
                "accession_number": "0000320193-24-000123",
                "report_date": "2024-09-28",
                "filed_date": "2024-11-01",
                "source_url": "https://www.sec.gov/example",
            }
        )

        async def _fake_retrieval_agent(state, client=None):
            return {
                "retrieval": {
                    "ok": False,
                    "top_tables": [],
                    "partial_failures": [],
                    "metadata_used": {"ticker": "AAPL", "fiscal_year": 2024, "form_type": "10-K"},
                    "error": "KB retrieval failed for this hybrid run.",
                    "max_total_score": None,
                }
            }

        class _FakeAnalyst:
            async def arun(self, packet, debug=False):
                self.packet = packet
                return AnalystRunResult(
                    ok=True,
                    status="ok",
                    answer="Structured facts still answered the question.",
                    intent=PlannerIntent.FILING_FACT,
                    metric=packet.analysis_task.metric,
                )

        fake_analyst = _FakeAnalyst()

        await aclose_orchestrator_runtime()
        saver = InMemorySaver()
        with (
            mock.patch("agents.orchestrator.agent_orchestrator._get_orchestrator_checkpointer", new=mock.AsyncMock(return_value=saver)),
            mock.patch("agents.orchestrator.agent_orchestrator._orchestrator_checkpoint_ttl_seconds", return_value=0),
            mock.patch("agents.orchestrator.agent_orchestrator._get_orchestrator_mcp_client", new=mock.AsyncMock(return_value=client)),
            mock.patch("agents.orchestrator.agent_orchestrator.retrieval_agent", new=_fake_retrieval_agent),
            mock.patch("agents.orchestrator.agent_orchestrator._get_pooled_analyst", new=mock.AsyncMock(return_value=fake_analyst)),
        ):
            output = await _invoke_orchestrator(
                {
                    "user_query": "What was revenue?",
                    "plan_id": "hybrid-kb-fail-run",
                    "analyst_model": "shared-model",
                    "tables_dir": "data/chunked",
                    "debug": False,
                },
                run_id="hybrid-kb-fail-run",
                planner=_HybridPlanner(),
            )

        self.assertEqual(output["route"], "hybrid")
        self.assertEqual(output["status"], "completed")
        self.assertFalse(output["retrieval"]["ok"])
        self.assertEqual(len(output["structured_fact_results"]), 1)
        self.assertTrue(output["structured_fact_results"][0]["tool_result"]["ok"])
        self.assertEqual(output["structured_fact_results"][0]["tool_result"]["status"], "ok")
        self.assertIn(
            "Structured fact: revenue = 391000000000.0 USD",
            fake_analyst.packet.context_items[-1].payload["content"],
        )

    async def test_invoke_orchestrator_hybrid_route_preserves_kb_when_structured_fact_degrades(self) -> None:
        client = _FakeStructuredFactClient(
            tool_result={
                "ok": False,
                "status": "partial",
                "metric_id": "revenue",
                "value": None,
                "error": "Structured fact returned partial evidence only.",
            }
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
            from agents.orchestrator import agent_orchestrator as orchestrator

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
                plan_id="hybrid-structured-fact-partial-run",
            ).model_copy(
                update={
                    "context_quality": ContextQuality.MEDIUM,
                    "context_items": [
                        ContextItem(
                            context_id="ctx_1",
                            target_id="1",
                            kind=ContextItemKind.TEXT,
                            source=SourceRef(ticker="AAPL", fiscal_year=2024, form_type=FormType.TEN_K),
                            payload={"content": "Revenue was supported by KB context."},
                        )
                    ],
                }
            )

        class _FakeAnalyst:
            async def arun(self, packet, debug=False):
                self.packet = packet
                return AnalystRunResult(
                    ok=True,
                    status="ok",
                    answer="KB still carried the run.",
                    intent=PlannerIntent.FILING_FACT,
                    metric=packet.analysis_task.metric,
                )

        fake_analyst = _FakeAnalyst()

        await aclose_orchestrator_runtime()
        saver = InMemorySaver()
        with (
            mock.patch("agents.orchestrator.agent_orchestrator._get_orchestrator_checkpointer", new=mock.AsyncMock(return_value=saver)),
            mock.patch("agents.orchestrator.agent_orchestrator._orchestrator_checkpoint_ttl_seconds", return_value=0),
            mock.patch("agents.orchestrator.agent_orchestrator._get_orchestrator_mcp_client", new=mock.AsyncMock(return_value=client)),
            mock.patch("agents.orchestrator.agent_orchestrator.retrieval_agent", new=_fake_retrieval_agent),
            mock.patch("agents.orchestrator.agent_orchestrator.build_packet_from_retrieval_output", new=_fake_packet_builder),
            mock.patch("agents.orchestrator.agent_orchestrator._get_pooled_analyst", new=mock.AsyncMock(return_value=fake_analyst)),
        ):
            output = await _invoke_orchestrator(
                {
                    "user_query": "What was revenue?",
                    "plan_id": "hybrid-structured-fact-partial-run",
                    "analyst_model": "shared-model",
                    "tables_dir": "data/chunked",
                    "debug": False,
                },
                run_id="hybrid-structured-fact-partial-run",
                planner=_HybridPlanner(),
            )

        self.assertEqual(output["route"], "hybrid")
        self.assertEqual(output["status"], "completed")
        self.assertTrue(output["retrieval"]["ok"])
        self.assertEqual(len(output["structured_fact_results"]), 1)
        self.assertEqual(output["structured_fact_results"][0]["tool_result"]["status"], "partial")
        self.assertIn("Revenue was supported by KB context.", fake_analyst.packet.context_items[0].payload["content"])
        self.assertIn(
            "Structured fact: revenue returned status partial.",
            fake_analyst.packet.context_items[-1].payload["content"],
        )

    async def test_structured_facts_node_returns_unresolved_without_tool_call(self) -> None:
        client = _FakeStructuredFactClient()
        with mock.patch("agents.orchestrator.agent_orchestrator._get_orchestrator_mcp_client", new=mock.AsyncMock(return_value=client)):
            result = await _structured_facts_node(
                {
                    "plan_obj": {
                        "metadata": {"ticker": "AAPL", "fiscal_year": 2025, "form_type": "10-K"},
                        "structured_fact_requests": [
                            {
                                "subquestion": "What was Apple mystery metric in FY2025?",
                                "metric_hint": "mystery metric",
                                "entity_hint": "Apple",
                                "fiscal_year": 2025,
                            }
                        ],
                    }
                }
            )

        self.assertEqual(result["structured_fact_results"][0]["resolver_status"], "unresolved")
        self.assertIsNone(result["structured_fact_results"][0]["resolved_metric_id"])
        self.assertIsNone(result["structured_fact_results"][0]["tool_result"])
        self.assertEqual(client.calls, [])

    async def test_structured_facts_node_returns_ambiguous_without_tool_call(self) -> None:
        client = _FakeStructuredFactClient()
        with mock.patch("agents.orchestrator.agent_orchestrator._get_orchestrator_mcp_client", new=mock.AsyncMock(return_value=client)):
            result = await _structured_facts_node(
                {
                    "plan_obj": {
                        "metadata": {"ticker": "AAPL", "fiscal_year": 2025, "form_type": "10-K"},
                        "structured_fact_requests": [
                            {
                                "subquestion": "What was Apple cash in FY2025?",
                                "metric_hint": "cash",
                                "entity_hint": "Apple",
                                "fiscal_year": 2025,
                            }
                        ],
                    }
                }
            )

        self.assertEqual(result["structured_fact_results"][0]["resolver_status"], "ambiguous")
        self.assertIsNone(result["structured_fact_results"][0]["resolved_metric_id"])
        self.assertIsNone(result["structured_fact_results"][0]["tool_result"])
        self.assertEqual(client.calls, [])

    async def test_structured_facts_node_rejects_hostile_ratio_before_client_initialization(self) -> None:
        get_client = mock.AsyncMock()
        with mock.patch(
            "agents.orchestrator.agent_orchestrator._get_orchestrator_mcp_client",
            new=get_client,
        ):
            result = await _structured_facts_node(
                {
                    "plan_obj": {
                        "route": "structured_fact",
                        "metadata": {
                            "ticker": "AAPL",
                            "fiscal_year": 2025,
                            "form_type": "10-K",
                        },
                        "structured_fact_requests": [
                            {
                                "subquestion": "Calculate Apple's return on equity in FY2025.",
                                "metric_hint": "revenue",
                                "entity_hint": "Apple",
                                "fiscal_year": 2025,
                            }
                        ],
                    }
                }
            )

        get_client.assert_not_awaited()
        rejected = result["structured_fact_results"][0]
        self.assertEqual(rejected["resolver_status"], "unresolved")
        self.assertIsNone(rejected["resolved_metric_id"])
        self.assertIsNone(rejected["tool_result"])
        issue = result["open_issues"][0]
        self.assertEqual(issue["code"], "STRUCTURED_FACT_CAPABILITY_REJECTED")
        self.assertEqual(issue["metadata"]["question_class"], "unsupported_ratio")
        self.assertEqual(issue["metadata"]["metric_hint"], "revenue")
        self.assertIn("return on equity", issue["metadata"]["subquestion"])
        self.assertEqual(issue["metadata"]["candidate_metric_ids"], [])

    async def test_structured_facts_node_rejects_alias_inside_unrelated_metric_name(self) -> None:
        get_client = mock.AsyncMock()
        with mock.patch(
            "agents.orchestrator.agent_orchestrator._get_orchestrator_mcp_client",
            new=get_client,
        ):
            result = await _structured_facts_node(
                {
                    "plan_obj": {
                        "route": "structured_fact",
                        "metadata": {
                            "ticker": "MSFT",
                            "fiscal_year": 2025,
                            "form_type": "10-K",
                        },
                        "structured_fact_requests": [
                            {
                                "subquestion": (
                                    "What were Microsoft's sales and marketing expenses "
                                    "in FY2025?"
                                ),
                                "metric_hint": "sales and marketing expense",
                                "entity_hint": "Microsoft",
                                "fiscal_year": 2025,
                            }
                        ],
                    }
                }
            )

        get_client.assert_not_awaited()
        rejected = result["structured_fact_results"][0]
        self.assertEqual(rejected["resolver_status"], "unresolved")
        self.assertIsNone(rejected["resolved_metric_id"])
        self.assertIsNone(rejected["tool_result"])
        issue = result["open_issues"][0]
        self.assertEqual(issue["code"], "STRUCTURED_FACT_CAPABILITY_REJECTED")
        self.assertEqual(issue["metadata"]["question_class"], "unknown")

    async def test_structured_facts_node_rejects_supported_hint_for_unknown_subquestion(self) -> None:
        get_client = mock.AsyncMock()
        with mock.patch(
            "agents.orchestrator.agent_orchestrator._get_orchestrator_mcp_client",
            new=get_client,
        ):
            result = await _structured_facts_node(
                {
                    "plan_obj": {
                        "route": "structured_fact",
                        "metadata": {
                            "ticker": "AAPL",
                            "fiscal_year": 2025,
                            "form_type": "10-K",
                        },
                        "structured_fact_requests": [
                            {
                                "subquestion": "What were Apple's bookings in FY2025?",
                                "metric_hint": "revenue",
                                "entity_hint": "Apple",
                                "fiscal_year": 2025,
                            }
                        ],
                    }
                }
            )

        get_client.assert_not_awaited()
        rejected = result["structured_fact_results"][0]
        self.assertEqual(rejected["resolver_status"], "unresolved")
        self.assertIsNone(rejected["resolved_metric_id"])
        self.assertIsNone(rejected["tool_result"])
        issue = result["open_issues"][0]
        self.assertEqual(issue["code"], "STRUCTURED_FACT_CAPABILITY_REJECTED")
        self.assertEqual(issue["metadata"]["question_class"], "unknown")

    async def test_structured_facts_node_preserves_non_ok_tool_result(self) -> None:
        client = _FakeStructuredFactClient(
            tool_result={
                "ok": False,
                "status": "partial",
                "metric_id": "total_debt",
                "value": None,
                "error": "Missing noncurrent debt component.",
            }
        )
        with mock.patch("agents.orchestrator.agent_orchestrator._get_orchestrator_mcp_client", new=mock.AsyncMock(return_value=client)):
            result = await _structured_facts_node(
                {
                    "plan_obj": {
                        "metadata": {"ticker": "AAPL", "fiscal_year": 2025, "form_type": "10-K"},
                        "structured_fact_requests": [
                            {
                                "subquestion": "What was Apple total debt in FY2025?",
                                "metric_hint": "total debt",
                                "entity_hint": "Apple",
                                "fiscal_year": 2025,
                            }
                        ],
                    }
                }
            )

        self.assertEqual(result["structured_fact_results"][0]["resolver_status"], "resolved")
        self.assertEqual(result["structured_fact_results"][0]["tool_result"]["status"], "partial")
        self.assertEqual(client.calls[0]["metric_id"], "total_debt")


if __name__ == "__main__":
    unittest.main()
