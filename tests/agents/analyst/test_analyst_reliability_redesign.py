from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from agents.analyst.agent import (
    AnalystAgent,
    AnalystPacket,
    AnalystStructuredAnswer,
    AnalysisTask,
    ContextItem,
    ContextItemKind,
    ContextQuality,
    FilingMetadata,
    OpenIssue,
    PlannerIntent,
    Severity,
    SourceRef,
    _FinancialToolRuntime,
    _computations_match,
    _parse_agent_messages,
    _should_retry_response,
    _tool_error_is_retryable,
)
from agents.contracts import FormType
from tests.snapshot_utils import assert_graph_snapshot_jsonable


class _FakeBoundModel:
    def __init__(self, responses):
        self._responses = list(responses)
        self._index = 0
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        idx = min(self._index, len(self._responses) - 1)
        self._index += 1
        response = self._responses[idx]
        if isinstance(response, Exception):
            raise response
        return response


class _FakeModelFactory:
    def __init__(self, responses):
        self._responses = list(responses)

    def bind_tools(self, _tools):
        return _FakeBoundModel(self._responses)


class _FakeRuntime:
    def __init__(self, result=None):
        self.result = result or {
            "content": '{"result": 42.0, "expression": "a + b", "variables": {"a": "20", "b": "22"}}',
            "artifact": {"result": 42.0, "expression": "a + b", "variables": {"a": "20", "b": "22"}},
            "status": "success",
        }
        self.closed = 0

    async def call_tool(self, _name, _args):
        return self.result

    async def aclose(self):
        self.closed += 1
        return None


class _BlockingRuntime:
    def __init__(self):
        self.entered = 0
        self.max_in_flight = 0
        self.release = asyncio.Event()
        self.closed = 0

    async def call_tool(self, _name, _args):
        self.entered += 1
        self.max_in_flight = max(self.max_in_flight, self.entered)
        try:
            await self.release.wait()
            return {
                "content": '{"result": 42.0}',
                "artifact": {"result": 42.0},
                "status": "success",
            }
        finally:
            self.entered -= 1

    async def aclose(self):
        self.closed += 1
        return None


class _ConcurrentRuntime:
    def __init__(self):
        self.entered = 0
        self.max_in_flight = 0
        self.release = asyncio.Event()
        self.closed = 0

    async def call_tool(self, _name, args):
        self.entered += 1
        self.max_in_flight = max(self.max_in_flight, self.entered)
        try:
            await self.release.wait()
            expression = str((args or {}).get("expression") or "a+b")
            variables = dict((args or {}).get("variables") or {"a": "20", "b": "22"})
            return {
                "content": '{"result": 42.0}',
                "artifact": {"result": 42.0, "expression": expression, "variables": variables},
                "status": "success",
            }
        finally:
            self.entered -= 1

    async def aclose(self):
        self.closed += 1
        return None


class _ConcurrentComputeModel:
    async def ainvoke(self, messages):
        tool_messages = [
            msg
            for msg in messages
            if isinstance(msg, ToolMessage) and getattr(msg, "name", None) == "financial_evaluator"
        ]
        if tool_messages:
            artifact = getattr(tool_messages[-1], "artifact", None) or {"result": 42.0}
            result = float(artifact.get("result", 42.0))
            expression = artifact.get("expression") or "a+b"
            variables = artifact.get("variables") or {"a": "20", "b": "22"}
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "FinalAnswer",
                        "id": "call-final",
                        "args": {
                            "status": "ok",
                            "answer": f"The computed result is {result}.",
                            "used_context_ids": ["ctx_1"],
                            "missing_values": [],
                            "confidence": 0.9,
                            "calculation": {
                                "expression": expression,
                                "variables": variables,
                                "result": result,
                            },
                            "compare_rows": [],
                        },
                    }
                ],
            )
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "financial_evaluator",
                    "id": "tool-call",
                    "args": {"expression": "a+b", "variables": {"a": "20", "b": "22"}},
                }
            ],
        )


class _ConcurrentModelFactory:
    def bind_tools(self, _tools):
        return _ConcurrentComputeModel()


class _FakeInjectedTool:
    def __init__(self, result=None):
        self.result = result or {"result": 42.0, "expression": "a + b", "variables": {"a": "20", "b": "22"}}

    async def ainvoke(self, _args):
        return self.result


class _SequentialInjectedTool:
    def __init__(self, results):
        self._results = list(results)
        self._index = 0

    async def ainvoke(self, _args):
        result = self._results[self._index]
        self._index += 1
        return result


def _run_computation_history_case(*, tool_calls, tool_results, final_calculation):
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "financial_evaluator",
                    "id": f"tool-{index}",
                    "args": {
                        "expression": expression,
                        "variables": variables,
                    },
                }
            ],
        )
        for index, (expression, variables) in enumerate(tool_calls, start=1)
    ]
    responses.append(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "FinalAnswer",
                    "id": "call-final",
                    "args": {
                        "status": "ok",
                        "answer": "The computed result is 12.87%.",
                        "used_context_ids": ["ctx_1"],
                        "missing_values": [],
                        "confidence": 0.9,
                        "calculation": final_calculation,
                        "compare_rows": [],
                    },
                }
            ],
        )
    )

    agent = AnalystAgent(max_attempts=1, max_tool_rounds=max(4, len(tool_calls)))
    agent._bound_model_override = _FakeBoundModel(responses)
    agent._tool_map = {"financial_evaluator": _SequentialInjectedTool(tool_results)}
    agent._tools_available = True
    agent._graph = agent._build_workflow()
    return asyncio.run(agent.arun(_packet("compute")))


def _packet(task_type: str = "extract") -> AnalystPacket:
    return AnalystPacket(
        plan_id="plan-1",
        user_query="What is total debt?",
        intent=PlannerIntent.FILING_CALC,
        metadata=FilingMetadata(ticker="AAPL", fiscal_year=2024, form_type=FormType.TEN_K),
        analysis_task=AnalysisTask(task_type=task_type, metric="total debt"),
        targets=[{"target_id": "AAPL:2024:10-K", "ticker": "AAPL", "fiscal_year": 2024, "form_type": "10-K"}],
        context_quality=ContextQuality.MEDIUM,
        context_items=[
            ContextItem(
                context_id="ctx_1",
                target_id="AAPL:2024:10-K",
                kind=ContextItemKind.TEXT,
                source=SourceRef(ticker="AAPL", fiscal_year=2024, form_type=FormType.TEN_K),
                payload={"content": "Total debt was $42 million."},
            )
        ],
        open_issues=[],
    )


def test_parse_agent_messages_reads_final_answer_tool():
    msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "FinalAnswer",
                "id": "call-1",
                "args": {
                    "status": "ok",
                    "answer": "Total debt was $42 million.",
                    "used_context_ids": ["ctx_1"],
                    "missing_values": [],
                    "confidence": 0.8,
                    "calculation": None,
                    "compare_rows": [],
                },
            }
        ],
    )
    parsed = _parse_agent_messages([msg])
    assert parsed["final_output_valid"] is True
    assert parsed["final_output"].answer == "Total debt was $42 million."


def test_analyst_graph_state_round_trips_with_checkpointer():
    async def _run():
        saver = InMemorySaver()
        final_answer = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "FinalAnswer",
                    "id": "call-1",
                    "args": {
                        "status": "ok",
                        "answer": "Recovered.",
                        "used_context_ids": ["ctx_1"],
                        "missing_values": [],
                        "confidence": 0.8,
                        "calculation": None,
                        "compare_rows": [],
                    },
                }
            ],
        )

        agent = AnalystAgent(max_attempts=1)
        agent._bound_model_override = _FakeBoundModel([final_answer])
        graph = agent._build_workflow(checkpointer=saver)
        config = {"configurable": {"thread_id": "analyst-checkpoint"}}

        await graph.ainvoke(
            {
                "packet": _packet("extract"),
                "messages": [],
                "tools_available": False,
                "tool_setup_error": None,
                "max_attempts": 1,
                "max_tool_rounds": 4,
            },
            config=config,
        )
        snapshot = await graph.aget_state(config)
        assert_graph_snapshot_jsonable(snapshot)
        parsed = snapshot.values["parsed"]

        assert isinstance(parsed["final_output"], dict)
        assert parsed["final_output"]["answer"] == "Recovered."
        assert parsed["tool_calls"][0]["name"] == "FinalAnswer"

    asyncio.run(_run())


def test_should_retry_response_retries_invalid_non_compute():
    parsed = {"final_output_valid": False}
    assert _should_retry_response(_packet("extract"), parsed, 1, 2, tools_available=False) is True


def test_should_not_retry_when_tool_round_limit_exceeded():
    parsed = {"final_output_valid": False, "tool_round_limit_exceeded": True}
    assert _should_retry_response(_packet("compute"), parsed, 1, 3, tools_available=True) is False


def test_compare_with_calculation_signal_retries_without_tool_use():
    packet = _packet("compare")
    packet.analysis_task.requires_calculation = True
    parsed = {
        "final_output_valid": True,
        "used_financial_evaluator": False,
        "calculation_blocked": False,
        "numeric_result": None,
        "tool_error": None,
    }
    assert _should_retry_response(packet, parsed, 1, 2, tools_available=True) is True


def test_compute_tool_error_final_output_does_not_retry():
    parsed = {
        "final_output_valid": True,
        "final_output": {
            "status": "tool_error",
            "answer": "The calculator failed.",
            "used_context_ids": ["ctx_1"],
            "missing_values": [],
            "confidence": 0.2,
            "calculation": None,
            "compare_rows": [],
        },
        "used_financial_evaluator": True,
        "calculation_blocked": False,
        "numeric_result": None,
        "tool_error": "bad expression",
    }
    assert _should_retry_response(_packet("compute"), parsed, 1, 2, tools_available=True) is False


def test_retryable_tool_error_is_detected():
    parsed = {"tool_error": "SyntaxError: invalid syntax"}
    final_output = AnalystStructuredAnswer(
        status="tool_error",
        answer="The calculator failed.",
        used_context_ids=["ctx_1"],
        missing_values=[],
        confidence=0.2,
        calculation=None,
        compare_rows=[],
    )
    assert _tool_error_is_retryable(parsed, final_output) is True


def test_structured_retryable_tool_error_code_is_detected():
    parsed = {"tool_error": "division looked wrong", "tool_error_code": "invalid_syntax"}
    final_output = AnalystStructuredAnswer(
        status="tool_error",
        answer="The calculator failed.",
        used_context_ids=["ctx_1"],
        missing_values=[],
        confidence=0.2,
        calculation=None,
        compare_rows=[],
    )
    assert _tool_error_is_retryable(parsed, final_output) is True


def test_non_retryable_tool_error_is_not_detected():
    parsed = {"tool_error": "bad expression"}
    final_output = AnalystStructuredAnswer(
        status="tool_error",
        answer="The calculator failed.",
        used_context_ids=["ctx_1"],
        missing_values=[],
        confidence=0.2,
        calculation=None,
        compare_rows=[],
    )
    assert _tool_error_is_retryable(parsed, final_output) is False


def test_structured_non_retryable_tool_error_code_is_not_detected():
    parsed = {"tool_error": "division by zero", "tool_error_code": "division_by_zero"}
    final_output = AnalystStructuredAnswer(
        status="tool_error",
        answer="The calculator failed.",
        used_context_ids=["ctx_1"],
        missing_values=[],
        confidence=0.2,
        calculation=None,
        compare_rows=[],
    )
    assert _tool_error_is_retryable(parsed, final_output) is False


def test_compute_fails_closed_when_runtime_unavailable(monkeypatch):
    async def _raise_create(**_kwargs):
        raise RuntimeError("SSE unavailable")

    monkeypatch.setattr(_FinancialToolRuntime, "create", _raise_create)

    agent = AnalystAgent()
    result = asyncio.run(agent.arun(_packet("compute")))
    assert result.ok is False
    assert result.status == "tool_error"
    assert any(issue.code == "TOOL_UNAVAILABLE_FOR_COMPUTE" for issue in result.open_issues)


def test_tool_loop_is_bounded(monkeypatch):
    async def _fake_create(**_kwargs):
        return _FakeRuntime()

    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "financial_evaluator",
                    "id": "tool-1",
                    "args": {"expression": "a + b", "variables": {"a": "20", "b": "22"}},
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "financial_evaluator",
                    "id": "tool-2",
                    "args": {"expression": "a + b", "variables": {"a": "20", "b": "22"}},
                }
            ],
        ),
    ]

    monkeypatch.setattr(_FinancialToolRuntime, "create", _fake_create)
    monkeypatch.setattr(
        "agents.analyst.agent.build_chat_model",
        lambda **_kwargs: _FakeModelFactory(responses),
    )

    agent = AnalystAgent(max_attempts=2, max_tool_rounds=1, timeout_s=1.0)
    result = asyncio.run(agent.arun(_packet("compute")))
    assert result.ok is False
    assert any(issue.code == "ANALYST_TOOL_LOOP_LIMIT" for issue in result.open_issues)
    assert result.error == "ANALYST_OUTPUT_INVALID"


def test_context_item_factory_requires_context_id():
    with pytest.raises(ValueError):
        ContextItem.from_table_candidate(
            {"doc_id": "doc-1", "table_id": "1", "total_score": 1.0},
            context_id="",
        )


def test_compare_row_context_ids_are_promoted_to_citations():
    final_answer = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "FinalAnswer",
                "id": "call-1",
                "args": {
                    "status": "ok",
                    "answer": "Comparison complete.",
                    "used_context_ids": [],
                    "missing_values": [],
                    "confidence": 0.8,
                    "calculation": None,
                    "compare_rows": [
                        {
                            "target_id": "AAPL:2024:10-K",
                            "label": "Total debt",
                            "value": "$42 million",
                            "context_ids": ["ctx_1"],
                        }
                    ],
                },
            }
        ],
    )

    agent = AnalystAgent(max_attempts=1)
    agent._bound_model_override = _FakeBoundModel([final_answer])
    agent._graph = agent._build_workflow()

    result = asyncio.run(agent.arun(_packet("compare")))

    assert result.ok is True
    assert result.used_context_ids == ["ctx_1"]
    assert [citation.context_id for citation in result.citations] == ["ctx_1"]


def test_parse_agent_messages_resets_state_after_retry_human_message():
    invalid_final = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "FinalAnswer",
                "id": "call-1",
                "args": {
                    "status": "ok",
                    "answer": "stale answer",
                    "used_context_ids": ["ctx_1"],
                    "missing_values": [],
                    "confidence": 0.8,
                },
            }
        ],
    )
    parsed = _parse_agent_messages(
        [
            AIMessage(content="", tool_calls=[{"name": "financial_evaluator", "id": "t1", "args": {"expression": "a+b", "variables": {"a": "20", "b": "22"}}}]),
            ToolMessage(
                content='{"result":42,"expression":"a+b","variables":{"a":"20","b":"22"}}',
                name="financial_evaluator",
                tool_call_id="t1",
                artifact={"result": 42, "expression": "a+b", "variables": {"a": "20", "b": "22"}},
                status="success",
            ),
            invalid_final,
            HumanMessage(content="Retry with a valid answer."),
            AIMessage(content="I forgot to call FinalAnswer this turn."),
        ]
    )
    assert parsed["final_output_valid"] is False
    assert parsed["final_answer"] == "I forgot to call FinalAnswer this turn."
    assert parsed["used_financial_evaluator"] is True
    assert parsed["numeric_result"] == 42.0


def test_parse_agent_messages_clears_tool_error_after_successful_retry():
    parsed = _parse_agent_messages(
        [
            AIMessage(content="", tool_calls=[{"name": "financial_evaluator", "id": "t1", "args": {"expression": "a+b"}}]),
            ToolMessage(content='{"error":"bad expression"}', name="financial_evaluator", tool_call_id="t1", artifact={"error": "bad expression"}, status="error"),
            HumanMessage(content="Retry with corrected syntax."),
            AIMessage(content="", tool_calls=[{"name": "financial_evaluator", "id": "t2", "args": {"expression": "a+b", "variables": {"a": "20", "b": "22"}}}]),
            ToolMessage(
                content='{"result":42,"expression":"a+b","variables":{"a":"20","b":"22"}}',
                name="financial_evaluator",
                tool_call_id="t2",
                artifact={"result": 42, "expression": "a+b", "variables": {"a": "20", "b": "22"}},
                status="success",
            ),
        ]
    )
    assert parsed["tool_error"] is None
    assert parsed["numeric_result"] == 42.0


def test_parse_agent_messages_clears_stale_numeric_result_after_tool_error():
    parsed = _parse_agent_messages(
        [
            AIMessage(content="", tool_calls=[{"name": "financial_evaluator", "id": "t1", "args": {"expression": "a+b"}}]),
            ToolMessage(
                content='{"result":42,"expression":"a+b","variables":{"a":"20","b":"22"}}',
                name="financial_evaluator",
                tool_call_id="t1",
                artifact={"result": 42, "expression": "a+b", "variables": {"a": "20", "b": "22"}},
                status="success",
            ),
            AIMessage(content="", tool_calls=[{"name": "financial_evaluator", "id": "t2", "args": {"expression": "a+/b"}}]),
            ToolMessage(
                content='{"error":"bad expression"}',
                name="financial_evaluator",
                tool_call_id="t2",
                artifact={"error": "bad expression"},
                status="error",
            ),
        ]
    )
    assert parsed["tool_error"] == "bad expression"
    assert parsed["numeric_result"] is None


def test_retry_history_contains_synthetic_finalanswer_tool_message():
    invalid_final = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "FinalAnswer",
                "id": "call-1",
                "args": {
                    "status": "ok",
                    "used_context_ids": ["ctx_1"],
                    "missing_values": [],
                    "confidence": 0.8,
                    "calculation": None,
                    "compare_rows": [],
                },
            }
        ],
    )
    valid_final = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "FinalAnswer",
                "id": "call-2",
                "args": {
                    "status": "ok",
                    "answer": "Recovered on retry.",
                    "used_context_ids": ["ctx_1"],
                    "missing_values": [],
                    "confidence": 0.8,
                    "calculation": None,
                    "compare_rows": [],
                },
            }
        ],
    )

    agent = AnalystAgent(max_attempts=2)
    bound_model = _FakeBoundModel([invalid_final, valid_final])
    agent._bound_model_override = bound_model
    agent._graph = agent._build_workflow()

    result = asyncio.run(agent.arun(_packet("extract")))

    assert result.ok is True
    second_call_messages = bound_model.calls[1]
    assert any(isinstance(msg, ToolMessage) and getattr(msg, "name", None) == "FinalAnswer" for msg in second_call_messages)
    assert any(isinstance(msg, HumanMessage) for msg in second_call_messages)


def test_mixed_finalanswer_and_calculator_calls_emit_finalanswer_error_tool_message():
    mixed_response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "financial_evaluator",
                "id": "tool-1",
                "args": {"expression": "a+b", "variables": {"a": "20", "b": "22"}},
            },
            {
                "name": "FinalAnswer",
                "id": "call-1",
                "args": {
                    "status": "ok",
                    "answer": "Jumped ahead.",
                    "used_context_ids": ["ctx_1"],
                    "missing_values": [],
                    "confidence": 0.8,
                    "calculation": {"expression": "a+b", "variables": {"a": "20", "b": "22"}, "result": 42},
                    "compare_rows": [],
                },
            },
        ],
    )
    valid_final = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "FinalAnswer",
                "id": "call-2",
                "args": {
                    "status": "ok",
                    "answer": "Recovered.",
                    "used_context_ids": ["ctx_1"],
                    "missing_values": [],
                    "confidence": 0.8,
                    "calculation": {"expression": "a+b", "variables": {"a": "20", "b": "22"}, "result": 42},
                    "compare_rows": [],
                },
            }
        ],
    )

    agent = AnalystAgent(max_attempts=2)
    bound_model = _FakeBoundModel([mixed_response, valid_final])
    agent._bound_model_override = bound_model
    agent._tool_map = {
        "financial_evaluator": _FakeInjectedTool(
            result={
                "result": 42,
                "expression": "a+b",
                "variables": {"a": "20", "b": "22"},
            }
        )
    }
    agent._tools_available = True
    agent._graph = agent._build_workflow()

    result = asyncio.run(agent.arun(_packet("compute")))

    assert result.ok is True
    first_call_messages = bound_model.calls[1]
    tool_message_names = [getattr(msg, "name", None) for msg in first_call_messages if isinstance(msg, ToolMessage)]
    assert tool_message_names[-2:] == ["financial_evaluator", "FinalAnswer"]
    finalanswer_errors = [
        msg for msg in first_call_messages if isinstance(msg, ToolMessage) and getattr(msg, "name", None) == "FinalAnswer"
    ]
    assert finalanswer_errors
    assert getattr(finalanswer_errors[0], "status", None) == "error"
    assert "Wait for tool results first" in str(finalanswer_errors[0].content)


def test_tool_round_limit_invalidates_same_turn_finalanswer():
    first_tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "financial_evaluator",
                "id": "tool-1",
                "args": {"expression": "a+b", "variables": {"a": "20", "b": "22"}},
            }
        ],
    )
    blocked_mixed_response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "financial_evaluator",
                "id": "tool-2",
                "args": {"expression": "a+b", "variables": {"a": "20", "b": "22"}},
            },
            {
                "name": "FinalAnswer",
                "id": "call-2",
                "args": {
                    "status": "ok",
                    "answer": "Hallucinated completion.",
                    "used_context_ids": ["ctx_1"],
                    "missing_values": [],
                    "confidence": 0.8,
                    "calculation": {"expression": "a+b", "variables": {"a": "20", "b": "22"}, "result": 42},
                    "compare_rows": [],
                },
            },
        ],
    )

    agent = AnalystAgent(max_attempts=1, max_tool_rounds=1)
    bound_model = _FakeBoundModel([first_tool_call, blocked_mixed_response])
    agent._bound_model_override = bound_model
    agent._tool_map = {
        "financial_evaluator": _FakeInjectedTool(
            result={
                "result": 42,
                "expression": "a+b",
                "variables": {"a": "20", "b": "22"},
            }
        )
    }
    agent._tools_available = True
    agent._graph = agent._build_workflow()

    result = asyncio.run(agent.arun(_packet("compute")))

    assert result.ok is False
    assert result.error == "ANALYST_OUTPUT_INVALID"
    assert any(issue.code == "ANALYST_TOOL_LOOP_LIMIT" for issue in result.open_issues)


def test_structured_answer_schema_requires_explicit_lists():
    required = set(AnalystStructuredAnswer.model_json_schema().get("required") or [])
    assert {"status", "answer", "used_context_ids", "missing_values", "compare_rows"} <= required


def test_missing_computation_accepts_insufficient_data_as_terminal():
    final_answer = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "FinalAnswer",
                "id": "call-1",
                "args": {
                    "status": "insufficient_data",
                    "answer": "The filing does not provide the required inputs.",
                    "used_context_ids": ["ctx_1"],
                    "missing_values": ["weighted average shares"],
                    "confidence": 0.4,
                    "calculation": None,
                    "compare_rows": [],
                },
            }
        ],
    )

    packet = _packet("compute")
    agent = AnalystAgent(max_attempts=1)
    agent._bound_model_override = _FakeBoundModel([final_answer])
    agent._tool_map = {"financial_evaluator": object()}
    agent._tools_available = True
    agent._graph = agent._build_workflow()

    result = asyncio.run(agent.arun(packet))

    assert result.ok is True
    assert result.status == "insufficient_data"
    assert result.error is None
    assert result.missing_values == ["weighted average shares"]
    assert not any(issue.code == "COMPUTE_RESULT_MISSING" for issue in result.open_issues)


def test_missing_computation_with_ok_status_still_fails_closed():
    final_answer = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "FinalAnswer",
                "id": "call-1",
                "args": {
                    "status": "ok",
                    "answer": "The result is unavailable.",
                    "used_context_ids": ["ctx_1"],
                    "missing_values": [],
                    "confidence": 0.4,
                    "calculation": None,
                    "compare_rows": [],
                },
            }
        ],
    )

    packet = _packet("compute")
    agent = AnalystAgent(max_attempts=1)
    agent._bound_model_override = _FakeBoundModel([final_answer])
    agent._tool_map = {"financial_evaluator": object()}
    agent._tools_available = True
    agent._graph = agent._build_workflow()

    result = asyncio.run(agent.arun(packet))

    assert result.ok is False
    assert result.status == "error"
    assert result.error == "COMPUTE_RESULT_MISSING"
    assert any(issue.code == "COMPUTE_RESULT_MISSING" for issue in result.open_issues)


def test_tool_result_mismatch_fails_closed():
    first = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "financial_evaluator",
                "id": "tool-1",
                "args": {"expression": "a+b", "variables": {"a": "20", "b": "22"}},
            }
        ],
    )
    final = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "FinalAnswer",
                "id": "call-2",
                "args": {
                    "status": "ok",
                    "answer": "The computed result is 99.",
                    "used_context_ids": ["ctx_1"],
                    "missing_values": [],
                    "confidence": 0.9,
                    "calculation": {
                        "expression": "a+b",
                        "variables": {"a": "20", "b": "22"},
                        "result": 99.0,
                    },
                    "compare_rows": [],
                },
            }
        ],
    )

    agent = AnalystAgent(max_attempts=1)
    agent._bound_model_override = _FakeBoundModel([first, final])
    agent._tool_map = {
        "financial_evaluator": _FakeInjectedTool(
            result={
                "result": 42.0,
                "expression": "a+b",
                "variables": {"a": "20", "b": "22"},
            }
        )
    }
    agent._tools_available = True
    agent._graph = agent._build_workflow()

    result = asyncio.run(agent.arun(_packet("compute")))

    assert result.ok is False
    assert result.error == "CALCULATION_RESULT_MISMATCH"
    assert result.computation is not None
    assert result.computation.result == 42.0
    assert any(issue.code == "CALCULATION_RESULT_MISMATCH" for issue in result.open_issues)


def test_final_calculation_matches_earlier_successful_computation():
    growth_expression = "((services_2024 - services_2023) / services_2023) * 100"
    variables = {"services_2023": "85200", "services_2024": "96169"}
    growth_result = 12.874413145539906
    result = _run_computation_history_case(
        tool_calls=[
            (growth_expression, variables),
            ("services_2024", variables),
        ],
        tool_results=[
            {"result": growth_result, "expression": growth_expression, "variables": variables},
            {"result": 96169.0, "expression": "services_2024", "variables": variables},
        ],
        final_calculation={
            "expression": growth_expression,
            "variables": variables,
            "result": growth_result,
        },
    )

    assert result.ok is True
    assert result.status == "ok"
    assert result.error is None
    assert result.computation is not None
    assert result.computation.expression == growth_expression
    assert result.computation.result == pytest.approx(growth_result)
    assert not any(issue.code == "CALCULATION_RESULT_MISMATCH" for issue in result.open_issues)
    assert [call.get("name") for call in result.trace.tool_calls].count("financial_evaluator") == 2


def test_final_calculation_matches_later_successful_computation():
    growth_expression = "((services_2024 - services_2023) / services_2023) * 100"
    variables = {"services_2023": "85200", "services_2024": "96169"}
    growth_result = 12.874413145539906
    result = _run_computation_history_case(
        tool_calls=[
            ("services_2024", variables),
            (growth_expression, variables),
        ],
        tool_results=[
            {"result": 96169.0, "expression": "services_2024", "variables": variables},
            {"result": growth_result, "expression": growth_expression, "variables": variables},
        ],
        final_calculation={
            "expression": growth_expression,
            "variables": variables,
            "result": growth_result,
        },
    )

    assert result.ok is True
    assert result.computation is not None
    assert result.computation.expression == growth_expression
    assert result.computation.result == pytest.approx(growth_result)


def test_equal_results_use_unique_final_calculation_provenance():
    tool_variables = {"revenue": "100", "expense": "100.0"}
    result = _run_computation_history_case(
        tool_calls=[
            ("revenue", tool_variables),
            ("expense", tool_variables),
        ],
        tool_results=[
            {"result": 100.0, "expression": "revenue", "variables": tool_variables},
            {"result": 100.0, "expression": "expense", "variables": tool_variables},
        ],
        final_calculation={
            "expression": " expense ",
            "variables": {"expense": "100"},
            "result": 100.0,
        },
    )

    assert result.ok is True
    assert result.computation is not None
    assert result.computation.expression == "expense"
    assert result.computation.variables == tool_variables


def test_equal_results_without_matching_provenance_are_ambiguous():
    result = _run_computation_history_case(
        tool_calls=[
            ("revenue", {"revenue": "100"}),
            ("expense", {"expense": "100"}),
        ],
        tool_results=[
            {"result": 100.0, "expression": "revenue", "variables": {"revenue": "100"}},
            {"result": 100.0, "expression": "expense", "variables": {"expense": "100"}},
        ],
        final_calculation={
            "expression": "margin",
            "variables": {"margin": "100"},
            "result": 100.0,
        },
    )

    assert result.ok is False
    assert result.error == "CALCULATION_RESULT_AMBIGUOUS"
    assert result.computation is None
    assert any(issue.code == "CALCULATION_RESULT_AMBIGUOUS" for issue in result.open_issues)


def test_duplicate_matching_provenance_is_ambiguous():
    result = _run_computation_history_case(
        tool_calls=[
            ("revenue", {"revenue": "100"}),
            ("revenue", {"revenue": "100.0"}),
        ],
        tool_results=[
            {"result": 100.0, "expression": "revenue", "variables": {"revenue": "100"}},
            {"result": 100.0, "expression": "revenue", "variables": {"revenue": "100.0"}},
        ],
        final_calculation={
            "expression": "revenue",
            "variables": {"revenue": "100"},
            "result": 100.0,
        },
    )

    assert result.ok is False
    assert result.error == "CALCULATION_RESULT_AMBIGUOUS"
    assert result.computation is None


def test_expression_normalization_preserves_token_boundaries():
    result = _run_computation_history_case(
        tool_calls=[("100", {}), ("50+50", {})],
        tool_results=[
            {"result": 100.0, "expression": "100", "variables": {}},
            {"result": 100.0, "expression": "50+50", "variables": {}},
        ],
        final_calculation={
            "expression": "1 00",
            "variables": {},
            "result": 100.0,
        },
    )

    assert result.ok is False
    assert result.error == "CALCULATION_RESULT_AMBIGUOUS"
    assert result.computation is None


def test_final_calculation_must_match_successful_computation_history():
    variables = {"services_2023": "85200", "services_2024": "96169"}
    result = _run_computation_history_case(
        tool_calls=[
            ("services_2024 - services_2023", variables),
            ("services_2024", variables),
        ],
        tool_results=[
            {
                "result": 10969.0,
                "expression": "services_2024 - services_2023",
                "variables": variables,
            },
            {"result": 96169.0, "expression": "services_2024", "variables": variables},
        ],
        final_calculation={
            "expression": "unrelated",
            "variables": variables,
            "result": 42.0,
        },
    )

    assert result.ok is False
    assert result.error == "CALCULATION_RESULT_MISMATCH"
    assert any(issue.code == "CALCULATION_RESULT_MISMATCH" for issue in result.open_issues)


def test_omitted_final_calculation_uses_only_successful_computation():
    variables = {"a": "20", "b": "22"}
    result = _run_computation_history_case(
        tool_calls=[("a+b", variables)],
        tool_results=[{"result": 42.0, "expression": "a+b", "variables": variables}],
        final_calculation=None,
    )

    assert result.ok is True
    assert result.computation is not None
    assert result.computation.expression == "a+b"
    assert result.computation.result == 42.0


def test_omitted_final_calculation_rejects_ambiguous_computation_history():
    variables = {"a": "20", "b": "22"}
    result = _run_computation_history_case(
        tool_calls=[("a+b", variables), ("a", variables)],
        tool_results=[
            {"result": 42.0, "expression": "a+b", "variables": variables},
            {"result": 20.0, "expression": "a", "variables": variables},
        ],
        final_calculation=None,
    )

    assert result.ok is False
    assert result.error == "CALCULATION_RESULT_AMBIGUOUS"
    assert result.computation is None
    assert any(issue.code == "CALCULATION_RESULT_AMBIGUOUS" for issue in result.open_issues)


def test_later_tool_error_preserves_history_and_still_fails_closed():
    variables = {"a": "20", "b": "22"}
    parsed = _parse_agent_messages(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "financial_evaluator",
                        "id": "tool-1",
                        "args": {"expression": "a+b", "variables": variables},
                    }
                ],
            ),
            ToolMessage(
                content='{"result":42,"expression":"a+b","variables":{"a":"20","b":"22"}}',
                name="financial_evaluator",
                tool_call_id="tool-1",
                artifact={"result": 42.0, "expression": "a+b", "variables": variables},
                status="success",
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "financial_evaluator",
                        "id": "tool-2",
                        "args": {"expression": "a+/b", "variables": variables},
                    }
                ],
            ),
            ToolMessage(
                content='{"error":"bad expression","error_code":"invalid_expression"}',
                name="financial_evaluator",
                tool_call_id="tool-2",
                artifact={"error": "bad expression", "error_code": "invalid_expression"},
                status="error",
            ),
        ]
    )
    assert parsed["tool_error"] == "bad expression"
    assert parsed["numeric_result"] is None
    assert parsed["successful_computations"] == [
        {"expression": "a+b", "variables": variables, "result": 42.0}
    ]

    result = _run_computation_history_case(
        tool_calls=[("a+b", variables), ("a+/b", variables)],
        tool_results=[
            {"result": 42.0, "expression": "a+b", "variables": variables},
            {
                "error": "bad expression",
                "error_code": "invalid_expression",
                "expression": "a+/b",
                "variables": variables,
            },
        ],
        final_calculation={"expression": "a+b", "variables": variables, "result": 42.0},
    )

    assert result.ok is False
    assert any(issue.code == "FINANCIAL_EVALUATOR_ERROR" for issue in result.open_issues)


def test_tool_result_rounding_difference_is_accepted():
    assert _computations_match(
        structured=type("Calc", (), {"result": 15000000.33})(),
        tool_computation=type("Calc", (), {"result": 15000000.33333333})(),
    )


def test_non_compute_prefers_tool_computation_when_available():
    first = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "financial_evaluator",
                "id": "tool-1",
                "args": {"expression": "a+b", "variables": {"a": "20", "b": "22"}},
            }
        ],
    )
    final = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "FinalAnswer",
                "id": "call-2",
                "args": {
                    "status": "ok",
                    "answer": "Extracted result.",
                    "used_context_ids": ["ctx_1"],
                    "missing_values": [],
                    "confidence": 0.9,
                    "calculation": {
                        "expression": "different",
                        "variables": {"a": "20", "b": "22"},
                        "result": 42.0,
                    },
                    "compare_rows": [],
                },
            }
        ],
    )

    agent = AnalystAgent(max_attempts=1)
    agent._bound_model_override = _FakeBoundModel([first, final])
    agent._tool_map = {
        "financial_evaluator": _FakeInjectedTool(
            result={
                "result": 42.0,
                "expression": "a+b",
                "variables": {"a": "20", "b": "22"},
            }
        )
    }
    agent._tools_available = True
    agent._graph = agent._build_workflow()

    result = asyncio.run(agent.arun(_packet("extract")))

    assert result.ok is True
    assert result.computation is not None
    assert result.computation.expression == "a+b"
    assert result.computation.result == 42.0


def test_non_compute_does_not_fail_closed_on_final_calculation_mismatch():
    first = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "financial_evaluator",
                "id": "tool-1",
                "args": {"expression": "a+b", "variables": {"a": "20", "b": "22"}},
            }
        ],
    )
    final = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "FinalAnswer",
                "id": "call-2",
                "args": {
                    "status": "ok",
                    "answer": "Extracted answer using multiple derived values.",
                    "used_context_ids": ["ctx_1"],
                    "missing_values": [],
                    "confidence": 0.9,
                    "calculation": {
                        "expression": "different",
                        "variables": {"a": "20", "b": "22"},
                        "result": 99.0,
                    },
                    "compare_rows": [],
                },
            }
        ],
    )

    agent = AnalystAgent(max_attempts=1)
    agent._bound_model_override = _FakeBoundModel([first, final])
    agent._tool_map = {
        "financial_evaluator": _FakeInjectedTool(
            result={
                "result": 42.0,
                "expression": "a+b",
                "variables": {"a": "20", "b": "22"},
            }
        )
    }
    agent._tools_available = True
    agent._graph = agent._build_workflow()

    result = asyncio.run(agent.arun(_packet("extract")))

    assert result.ok is True
    assert result.error is None
    assert not any(issue.code == "CALCULATION_RESULT_MISMATCH" for issue in result.open_issues)


def test_graph_exception_returns_structured_runtime_error():
    class _BoomGraph:
        async def ainvoke(self, _payload):
            raise RuntimeError("graph exploded")

    agent = AnalystAgent(max_attempts=1)
    agent._graph = _BoomGraph()

    result = asyncio.run(agent.arun(_packet("extract")))

    assert result.ok is False
    assert result.error == "graph exploded"
    assert any(issue.code == "ANALYST_RUNTIME_ERROR" for issue in result.open_issues)


def test_graph_accepts_structured_tool_error_without_extra_retry(monkeypatch):
    class _ErrorTool:
        async def ainvoke(self, _args):
            return {"error": "division by zero", "error_code": "division_by_zero"}

    tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "financial_evaluator",
                "id": "tool-1",
                "args": {"expression": "bad +", "variables": {"bad": "1"}},
            }
        ],
    )
    final_error = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "FinalAnswer",
                "id": "call-2",
                "args": {
                    "status": "tool_error",
                    "answer": "The calculation tool failed.",
                    "used_context_ids": ["ctx_1"],
                    "missing_values": [],
                    "confidence": 0.2,
                    "calculation": None,
                    "compare_rows": [],
                },
            }
        ],
    )

    agent = AnalystAgent(max_attempts=2)
    bound_model = _FakeBoundModel([tool_call, final_error])
    agent._bound_model_override = bound_model
    agent._tool_map = {"financial_evaluator": _ErrorTool()}
    agent._graph = agent._build_workflow()

    result = asyncio.run(agent.arun(_packet("compute")))

    assert result.ok is False
    assert result.status == "tool_error"
    assert len(bound_model.calls) == 2
    tool_issue = next(issue for issue in result.open_issues if issue.code == "FINANCIAL_EVALUATOR_ERROR")
    assert tool_issue.metadata == {"tool_error_code": "division_by_zero"}
    assert result.trace.tool_error_code == "division_by_zero"


def test_invalid_tool_calls_feed_back_error_tool_message():
    invalid = AIMessage(
        content="",
        invalid_tool_calls=[
            {
                "name": "financial_evaluator",
                "id": "bad-1",
                "error": "Malformed arguments JSON",
            }
        ],
    )
    valid_final = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "FinalAnswer",
                "id": "call-2",
                "args": {
                    "status": "ok",
                    "answer": "Recovered.",
                    "used_context_ids": ["ctx_1"],
                    "missing_values": [],
                    "confidence": 0.8,
                    "calculation": None,
                    "compare_rows": [],
                },
            }
        ],
    )

    agent = AnalystAgent(max_attempts=2)
    bound_model = _FakeBoundModel([invalid, valid_final])
    agent._bound_model_override = bound_model
    agent._graph = agent._build_workflow()

    result = asyncio.run(agent.arun(_packet("extract")))

    assert result.ok is True
    second_call_messages = bound_model.calls[1]
    tool_errors = [
        msg
        for msg in second_call_messages
        if isinstance(msg, ToolMessage) and getattr(msg, "name", None) == "financial_evaluator"
    ]
    assert tool_errors
    assert "Malformed arguments JSON" in str(tool_errors[0].content)


def test_invalid_final_answer_feeds_back_error_tool_message():
    invalid_final = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "FinalAnswer",
                "id": "call-1",
                "args": {
                    "status": "ok",
                    "answer": "Recovered.",
                    "confidence": 0.8,
                    "calculation": None,
                },
            }
        ],
    )
    valid_final = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "FinalAnswer",
                "id": "call-2",
                "args": {
                    "status": "ok",
                    "answer": "Recovered.",
                    "used_context_ids": ["ctx_1"],
                    "missing_values": [],
                    "confidence": 0.8,
                    "calculation": None,
                    "compare_rows": [],
                },
            }
        ],
    )

    agent = AnalystAgent(max_attempts=2)
    bound_model = _FakeBoundModel([invalid_final, valid_final])
    agent._bound_model_override = bound_model
    agent._graph = agent._build_workflow()

    result = asyncio.run(agent.arun(_packet("extract")))

    assert result.ok is True
    second_call_messages = bound_model.calls[1]
    final_answer_errors = [
        msg
        for msg in second_call_messages
        if isinstance(msg, ToolMessage) and getattr(msg, "name", None) == "FinalAnswer"
    ]
    assert final_answer_errors
    assert getattr(final_answer_errors[0], "status", None) == "error"
    assert "FinalAnswer validation failed" in str(final_answer_errors[0].content)


def test_injected_tool_map_makes_tools_available_without_runtime(monkeypatch):
    bound_calls = []

    class _Factory:
        def bind_tools(self, tools):
            bound_calls.append([getattr(tool, "name", getattr(tool, "__name__", "")) for tool in tools])
            return _FakeBoundModel(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "FinalAnswer",
                                "id": "call-1",
                                "args": {
                                    "status": "ok",
                                    "answer": "Done.",
                                    "used_context_ids": ["ctx_1"],
                                    "missing_values": [],
                                    "confidence": 0.8,
                                    "calculation": None,
                                    "compare_rows": [],
                                },
                            }
                        ],
                    )
                ]
            )

    async def _should_not_create(**_kwargs):
        raise AssertionError("runtime should not be created when an injected tool is available")

    monkeypatch.setattr("agents.analyst.agent.build_chat_model", lambda **_kwargs: _Factory())
    monkeypatch.setattr(_FinancialToolRuntime, "create", _should_not_create)

    agent = AnalystAgent(max_attempts=1)
    agent._tool_map = {"financial_evaluator": _FakeInjectedTool()}
    agent._graph = agent._build_workflow()

    result = asyncio.run(agent.arun(_packet("compute")))

    assert result.ok is False or result.ok is True
    assert any("financial_evaluator" in tools for tools in bound_calls)


def test_persistent_runtime_is_reused_and_closed_once(monkeypatch):
    create_calls = []
    runtime = _FakeRuntime()

    async def _fake_create(**_kwargs):
        create_calls.append(_kwargs)
        return runtime

    final_answer = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "FinalAnswer",
                "id": "call-1",
                "args": {
                    "status": "ok",
                    "answer": "Done.",
                    "used_context_ids": ["ctx_1"],
                    "missing_values": [],
                    "confidence": 0.8,
                    "calculation": None,
                    "compare_rows": [],
                },
            }
        ],
    )

    monkeypatch.setattr(_FinancialToolRuntime, "create", _fake_create)
    monkeypatch.setattr(
        "agents.analyst.agent.build_chat_model",
        lambda **_kwargs: _FakeModelFactory([final_answer, final_answer]),
    )

    agent = AnalystAgent(max_attempts=1)
    asyncio.run(agent.arun(_packet("extract")))
    asyncio.run(agent.arun(_packet("extract")))
    asyncio.run(agent.aclose())

    assert len(create_calls) == 1
    assert runtime.closed == 1


def test_call_managed_tool_runtime_allows_concurrent_io():
    async def _run():
        runtime = _BlockingRuntime()
        agent = AnalystAgent()
        agent._tool_runtime = runtime
        agent._tool_runtime_lock = asyncio.Lock()

        task1 = asyncio.create_task(agent._call_managed_tool_runtime(runtime, "financial_evaluator", {"x": "1"}))
        task2 = asyncio.create_task(agent._call_managed_tool_runtime(runtime, "financial_evaluator", {"x": "2"}))

        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert runtime.max_in_flight == 2

        runtime.release.set()
        await asyncio.gather(task1, task2)

    asyncio.run(_run())


def test_concurrent_arun_reuses_one_runtime_and_allows_overlapping_tool_calls(monkeypatch):
    async def _run():
        create_calls = []
        runtime = _ConcurrentRuntime()

        async def _fake_create(**_kwargs):
            create_calls.append(_kwargs)
            return runtime

        monkeypatch.setattr(_FinancialToolRuntime, "create", _fake_create)
        monkeypatch.setattr(
            "agents.analyst.agent.build_chat_model",
            lambda **_kwargs: _ConcurrentModelFactory(),
        )

        agent = AnalystAgent(max_attempts=1, timeout_s=5.0)
        try:
            task1 = asyncio.create_task(agent.arun(_packet("compute")))
            task2 = asyncio.create_task(agent.arun(_packet("compute")))

            for _ in range(50):
                if runtime.max_in_flight >= 2:
                    break
                await asyncio.sleep(0.01)
            assert runtime.max_in_flight == 2

            runtime.release.set()
            result1, result2 = await asyncio.gather(task1, task2)
        finally:
            await agent.aclose()

        assert len(create_calls) == 1
        assert runtime.closed == 1
        assert result1.ok is True
        assert result2.ok is True
        assert result1.computation is not None and result1.computation.result == 42.0
        assert result2.computation is not None and result2.computation.result == 42.0

    asyncio.run(_run())
