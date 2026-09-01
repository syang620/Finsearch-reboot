import sys
from pathlib import Path
from unittest import mock

import unittest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages import HumanMessage

sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))

from agents.analyst.agent import (
    AnalystAgent,
    AnalystStructuredAnswer,
    _context_item_to_text,
    _extract_json_payload,
    _first_float,
    _first_float_from_object,
    _parse_agent_messages,
    _retry_reason_message,
    _should_retry_compute,
    _table_dict_to_markdown,
    _to_float,
    _validate_financial_evaluator_args,
    build_analyst_prompt,
    build_packet_from_retrieval_output,
)
from agents.contracts import (
    AnalystPacket,
    AnalysisTask,
    ContextItem,
    ContextItemKind,
    FilingMetadata,
    FormType,
    PlannerIntent,
    SourceRef,
    StructuredFactEvidence,
)


def _build_compute_packet(task_type: str = "compute") -> AnalystPacket:
    return AnalystPacket(
        plan_id="test-plan",
        user_query="What is metric?",
        intent=PlannerIntent.FILING_CALC,
        metadata=FilingMetadata(
            ticker="AAPL",
            fiscal_year=2024,
        ),
        analysis_task=AnalysisTask(task_type=task_type, metric="cash flow"),
    )


class AnalystParsingTests(unittest.TestCase):
    def test_first_float_parses_commas(self) -> None:
        value = _first_float("Computed value: $12,345.67")
        self.assertEqual(value, 12345.67)

    def test_first_float_uses_json_result_field(self) -> None:
        value = _first_float('{"version":2,"result":123.4}')
        self.assertEqual(value, 123.4)

    def test_first_float_ignores_version_leading_number(self) -> None:
        value = _first_float("version: 2, result: 123.4")
        self.assertEqual(value, 123.4)

    def test_first_float_does_not_use_trailing_confidence_number(self) -> None:
        value = _first_float("Result: 40000. Confidence: 0.95")
        self.assertEqual(value, 40000.0)

    def test_first_float_parses_accounting_parentheses(self) -> None:
        value = _first_float("(123.45)")
        self.assertEqual(value, -123.45)

    def test_first_float_preserves_parenthesized_commas(self) -> None:
        value = _first_float("result: (5,000)")
        self.assertEqual(value, -5000.0)

    def test_first_float_parses_percentages(self) -> None:
        value = _first_float("Result: 12.5%")
        self.assertEqual(value, 0.125)

    def test_first_float_parses_parenthesized_percentages(self) -> None:
        value = _first_float("(12.5%)")
        self.assertEqual(value, -0.125)

    def test_first_float_from_object_parses_parenthesized_result_string(self) -> None:
        value = _first_float_from_object({"result": "(5,000)"})
        self.assertEqual(value, -5000.0)

    def test_first_float_from_object_parses_percentage_value_string(self) -> None:
        value = _first_float_from_object({"value": "12.5%"})
        self.assertEqual(value, 0.125)

    def test_to_float_parses_percentages(self) -> None:
        self.assertEqual(_to_float("12%"), 0.12)

    def test_extract_json_payload_handles_following_code_block(self) -> None:
        payload = _extract_json_payload(
            '```json\n{"status":"OK","answer":"done","used_context_ids":[],"missing_values":[],"compare_rows":[]}\n```\n```python\nprint({"x": 1})\n```'
        )
        self.assertEqual(payload["status"], "OK")

    def test_parse_agent_messages_prefers_structured_final_answer(self) -> None:
        parsed = _parse_agent_messages(
            [
                AIMessage(content="Calling tool", tool_calls=[{"name": "financial_evaluator", "args": {"expression": "a/b"}, "id": "call-1"}]),
                ToolMessage(
                    content='{"result": 42.5, "expression": "a/b", "variables": {"a": "85", "b": "2"}}',
                    tool_call_id="call-1",
                    name="financial_evaluator",
                    artifact={"result": 42.5, "expression": "a/b", "variables": {"a": "85", "b": "2"}},
                ),
                AIMessage(
                    content='{"status":"ok","answer":"The computed result is 42.5.","used_context_ids":["ctx_1"],"missing_values":[],"confidence":0.9,"calculation":{"expression":"a/b","variables":{"a":"85","b":"2"},"result":42.5},"compare_rows":[]}'
                ),
            ]
        )
        self.assertEqual(parsed["numeric_result"], 42.5)
        self.assertTrue(parsed["final_output_valid"])
        self.assertEqual(parsed["final_output"].used_context_ids, ["ctx_1"])

    def test_parse_agent_messages_normalizes_status_case(self) -> None:
        parsed = _parse_agent_messages(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "FinalAnswer",
                            "id": "call-1",
                            "args": {
                                "status": "OK",
                                "answer": "The value is available.",
                                "used_context_ids": ["ctx_1"],
                                "missing_values": [],
                                "confidence": 0.9,
                                "calculation": None,
                                "compare_rows": [],
                            },
                        }
                    ],
                )
            ]
        )
        self.assertTrue(parsed["final_output_valid"])
        self.assertEqual(parsed["final_output"].status, "ok")

    def test_context_item_truncation_preserves_line_boundaries(self) -> None:
        table = "\n".join(f"| row{i} | value{i} |" for i in range(2000))
        item = ContextItem(
            context_id="ctx_1",
            source=SourceRef(ticker="AAPL", fiscal_year=2024),
            payload={"table_name": "test_table", "table_markdown": table},
        )
        rendered = _context_item_to_text(item, 1)
        self.assertIn("\n... [truncated] ...\n", rendered)
        lines = rendered.splitlines()
        marker_index = lines.index("... [truncated] ...")
        self.assertTrue(lines[marker_index - 1].endswith("|"))

    def test_typed_structured_fact_renders_native_value_and_provenance(self) -> None:
        item = ContextItem(
            context_id="ctx_1",
            kind=ContextItemKind.STRUCTURED_FACT,
            source=SourceRef(
                ticker="AAPL",
                fiscal_year=2024,
                form_type=FormType.TEN_K,
                accession_no="0000320193-24-000123",
                report_date="2024-09-28",
                filing_date="2024-11-01",
                source_url="https://www.sec.gov/example",
            ),
            structured_fact=StructuredFactEvidence(
                metric_id="revenue",
                metric_label="Revenue",
                value=391_000_000_000.0,
                unit="USD",
                ticker="AAPL",
                fiscal_year=2024,
                form_type=FormType.TEN_K,
                accession_number="0000320193-24-000123",
                report_date="2024-09-28",
                filed_date="2024-11-01",
                source_url="https://www.sec.gov/example",
            ),
        )

        rendered = _context_item_to_text(item, 1)

        self.assertIn("structured_fact:", rendered)
        self.assertIn('"metric_id":"revenue"', rendered)
        self.assertIn('"metric_label":"Revenue"', rendered)
        self.assertIn('"value":391000000000.0', rendered)
        self.assertIn('"accession_number":"0000320193-24-000123"', rendered)
        self.assertIn('"source_url":"https://www.sec.gov/example"', rendered)
        self.assertNotIn("content:\nStructured fact:", rendered)

    def test_build_analyst_prompt_surfaces_definition_notes(self) -> None:
        packet = _build_compute_packet("extract")
        packet.analysis_task.definition_notes = ["Use GAAP revenue, not segment revenue."]
        rendered = build_analyst_prompt(packet)
        self.assertIn("Metric definition notes", rendered)
        self.assertIn("Use GAAP revenue", rendered)

    def test_build_analyst_prompt_omits_tool_call_when_tools_unavailable(self) -> None:
        packet = _build_compute_packet("compute")
        rendered = build_analyst_prompt(packet, tools_available=False)
        self.assertNotIn("call financial_evaluator before finishing", rendered)
        self.assertIn('return status="tool_error"', rendered)

    def test_build_analyst_prompt_includes_single_expression_guidance(self) -> None:
        packet = _build_compute_packet("compute")
        rendered = build_analyst_prompt(packet, tools_available=True)
        self.assertIn("exactly one scalar arithmetic expression", rendered)
        self.assertIn("Do not use assignments", rendered)

    def test_validate_financial_evaluator_args_rejects_assignment_statements(self) -> None:
        error = _validate_financial_evaluator_args(
            {
                "expression": "growth_rate = (sales_2025 - sales_2024) / sales_2024 * 100",
                "variables": {"sales_2025": "10", "sales_2024": "8"},
            }
        )
        self.assertIsNotNone(error)
        self.assertEqual(error["error_code"], "invalid_expression")

    def test_validate_financial_evaluator_args_rejects_unknown_variables(self) -> None:
        error = _validate_financial_evaluator_args(
            {
                "expression": "us_sales_percentage + china_sales_percentage",
                "variables": {"us_sales_2025": "10", "china_sales_2025": "8"},
            }
        )
        self.assertIsNotNone(error)
        self.assertEqual(error["error_code"], "unknown_variable")

    def test_validate_financial_evaluator_args_rejects_unsupported_functions(self) -> None:
        error = _validate_financial_evaluator_args(
            {
                "expression": "max(japan_growth, china_growth)",
                "variables": {"japan_growth": "15", "china_growth": "-4"},
            }
        )
        self.assertIsNotNone(error)
        self.assertEqual(error["error_code"], "unsupported_function")


class AnalystRetryTests(unittest.TestCase):
    def test_retry_for_compute_if_tool_not_used_and_no_missing_data(self) -> None:
        packet = _build_compute_packet()
        parsed = {
            "used_financial_evaluator": False,
            "calculation_blocked": False,
        }
        self.assertTrue(
            _should_retry_compute(packet, parsed, attempt=1, max_attempts=2, tools_available=True)
        )

    def test_no_retry_if_calculation_blocked_by_missing_data(self) -> None:
        packet = _build_compute_packet()
        parsed = {
            "used_financial_evaluator": False,
            "calculation_blocked": True,
        }
        self.assertFalse(
            _should_retry_compute(packet, parsed, attempt=1, max_attempts=2, tools_available=True)
        )

    def test_no_retry_if_tools_are_unavailable(self) -> None:
        packet = _build_compute_packet()
        parsed = {
            "used_financial_evaluator": False,
            "calculation_blocked": False,
        }
        self.assertFalse(
            _should_retry_compute(packet, parsed, attempt=1, max_attempts=2, tools_available=False)
        )

    def test_retry_reason_includes_tool_error(self) -> None:
        packet = _build_compute_packet()
        parsed = {"tool_error": "bad expression"}
        self.assertIn("bad expression", _retry_reason_message(packet, parsed))

    def test_retry_reason_explains_single_expression_contract(self) -> None:
        packet = _build_compute_packet()
        parsed = {"tool_error": "unsupported characters", "tool_error_code": "invalid_expression"}
        message = _retry_reason_message(packet, parsed)
        self.assertIn("one scalar arithmetic expression", message)
        self.assertIn("Do not use assignments", message)

    def test_retry_reason_explains_unknown_variable_contract(self) -> None:
        packet = _build_compute_packet()
        parsed = {"tool_error": "not provided", "tool_error_code": "unknown_variable"}
        message = _retry_reason_message(packet, parsed)
        self.assertIn("not passed in `variables`", message)
        self.assertIn("Compute one scalar at a time", message)

    def test_retry_reason_explains_unsupported_function_contract(self) -> None:
        packet = _build_compute_packet()
        parsed = {"tool_error": "max is unsupported", "tool_error_code": "unsupported_function"}
        message = _retry_reason_message(packet, parsed)
        self.assertIn("does not support functions like max", message)
        self.assertIn("call FinalAnswer now", message)

    def test_retry_reason_pushes_final_answer_after_numeric_result(self) -> None:
        packet = _build_compute_packet()
        parsed = {
            "tool_error": None,
            "final_output_error": None,
            "used_financial_evaluator": True,
            "numeric_result": 42.0,
        }
        message = _retry_reason_message(packet, parsed)
        self.assertIn("already have a valid financial_evaluator result", message)
        self.assertIn("call FinalAnswer now", message)

    def test_retry_reason_requires_calculator_when_missing(self) -> None:
        packet = _build_compute_packet()
        parsed = {"tool_error": None, "final_output_error": None, "used_financial_evaluator": False, "numeric_result": None}
        self.assertIn("requires grounded calculation", _retry_reason_message(packet, parsed))


class AnalystWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_retry_and_tool_execution_produce_structured_result(self) -> None:
        packet = _build_compute_packet()
        packet.context_items = [
            ContextItem(
                context_id="ctx_1",
                source=SourceRef(ticker="AAPL", fiscal_year=2024),
                payload={"table_markdown": "| metric | value |\n| revenue | 100 |\n| shares | 2 |"},
            )
        ]

        class FakeBoundModel:
            def __init__(self) -> None:
                self.calls = []

            async def ainvoke(self, messages):
                self.calls.append(list(messages))
                if len(self.calls) == 1:
                    return AIMessage(content='{"status":"ok","answer":"I think it is 50.","used_context_ids":["ctx_1"],"missing_values":[],"confidence":0.3,"calculation":null,"compare_rows":[]}')
                if len(self.calls) == 2:
                    return AIMessage(
                        content="Calling tool",
                        tool_calls=[
                            {
                                "name": "financial_evaluator",
                                "args": {"expression": "100/2", "variables": {"revenue": "100", "shares": "2"}},
                                "id": "call-1",
                            }
                        ],
                    )
                return AIMessage(
                    content='{"status":"ok","answer":"The computed result is 50.","used_context_ids":["ctx_1"],"missing_values":[],"confidence":0.95,"calculation":{"expression":"100/2","variables":{"revenue":"100","shares":"2"},"result":50},"compare_rows":[]}'
                )

        class FakeTool:
            name = "financial_evaluator"

            async def ainvoke(self, args):
                return {"result": 50, "expression": args["expression"], "variables": args["variables"]}

        agent = AnalystAgent(max_attempts=2)
        agent._bound_model_override = FakeBoundModel()
        agent._tool_map = {"financial_evaluator": FakeTool()}
        agent._tools_available = True
        agent._graph = agent._build_workflow()

        result = await agent.arun(packet)

        self.assertTrue(result.ok)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.computation.result, 50.0)
        self.assertEqual(result.used_context_ids, ["ctx_1"])
        self.assertEqual(len(result.citations), 1)
        self.assertTrue(result.trace.used_financial_evaluator)
        self.assertEqual(len(agent._bound_model_override.calls), 3)
        retry_messages = [msg for msg in agent._bound_model_override.calls[1] if isinstance(msg, HumanMessage)]
        self.assertTrue(retry_messages)
        self.assertIn("requires grounded calculation", str(retry_messages[-1].content))

    async def test_compute_task_without_tools_fails_closed(self) -> None:
        packet = _build_compute_packet()
        agent = AnalystAgent(max_attempts=1)
        agent._bound_model_override = object()
        agent._graph = object()
        agent._tools_available = False
        agent._tool_setup_error = "tool init failed"

        result = await agent.arun(packet)

        self.assertFalse(result.ok)
        self.assertEqual(result.status, "tool_error")
        issue_codes = [issue.code for issue in result.open_issues]
        self.assertIn("TOOL_UNAVAILABLE_FOR_COMPUTE", issue_codes)

    async def test_invalid_financial_expression_is_blocked_before_tool_dispatch_and_recovers(self) -> None:
        packet = _build_compute_packet()
        packet.context_items = [
            ContextItem(
                context_id="ctx_1",
                source=SourceRef(ticker="AAPL", fiscal_year=2025),
                payload={"table_markdown": "| metric | value |\n| sales_2025 | 10 |\n| sales_2024 | 8 |"},
            )
        ]

        class FakeBoundModel:
            def __init__(self) -> None:
                self.calls = []

            async def ainvoke(self, messages):
                self.calls.append(list(messages))
                if len(self.calls) == 1:
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "financial_evaluator",
                                "id": "call-1",
                                "args": {
                                    "expression": "growth_rate = (sales_2025 - sales_2024) / sales_2024 * 100",
                                    "variables": {"sales_2025": "10", "sales_2024": "8"},
                                },
                            }
                        ],
                    )
                if len(self.calls) == 2:
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "financial_evaluator",
                                "id": "call-2",
                                "args": {
                                    "expression": "((sales_2025 - sales_2024) / sales_2024) * 100",
                                    "variables": {"sales_2025": "10", "sales_2024": "8"},
                                },
                            }
                        ],
                    )
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "FinalAnswer",
                            "id": "call-3",
                            "args": {
                                "status": "ok",
                                "answer": "Growth rate was 25%.",
                                "used_context_ids": ["ctx_1"],
                                "missing_values": [],
                                "confidence": 0.9,
                                "calculation": {
                                    "expression": "((sales_2025 - sales_2024) / sales_2024) * 100",
                                    "variables": {"sales_2025": "10", "sales_2024": "8"},
                                    "result": 25.0,
                                },
                                "compare_rows": [],
                            },
                        }
                    ],
                )

        class FakeTool:
            name = "financial_evaluator"

            def __init__(self) -> None:
                self.calls = []

            async def ainvoke(self, args):
                self.calls.append(dict(args))
                return {"result": 25.0, "expression": args["expression"], "variables": args["variables"]}

        fake_tool = FakeTool()
        agent = AnalystAgent(max_attempts=2)
        agent._bound_model_override = FakeBoundModel()
        agent._tool_map = {"financial_evaluator": fake_tool}
        agent._tools_available = True
        agent._graph = agent._build_workflow()

        result = await agent.arun(packet)

        self.assertTrue(result.ok)
        self.assertEqual(len(fake_tool.calls), 1)
        self.assertEqual(fake_tool.calls[0]["expression"], "((sales_2025 - sales_2024) / sales_2024) * 100")
        retry_messages = [msg for msg in agent._bound_model_override.calls[1] if isinstance(msg, HumanMessage)]
        self.assertTrue(retry_messages)
        self.assertIn("one scalar arithmetic expression", str(retry_messages[-1].content))


class AnalystPacketBuilderTests(unittest.TestCase):
    def test_skips_failed_table_hydration_without_consuming_budget(self) -> None:
        retrieval_output = {
            "ok": True,
            "top_tables": [
                {
                    "table_name": "bad",
                    "total_score": 50,
                    "table": {"payload": {"doc_type": "table", "doc_id": "bad::table::0", "prefix": "bad", "table_index": 0}},
                },
                {
                    "table_name": "good-text",
                    "total_score": 40,
                    "table": {"payload": {"doc_type": "text", "content": "useful text", "ticker": "AAPL", "fiscal_year": 2024}},
                },
            ],
            "metadata_used": {"ticker": "AAPL", "fiscal_year": 2024, "form_type": "10-K"},
        }

        with mock.patch("agents.analyst.agent.load_table_data", side_effect=[None]):
            packet = build_packet_from_retrieval_output(
                user_query="What happened?",
                retrieval_output=retrieval_output,
                analysis_task={"task_type": "extract", "metric": "revenue"},
                max_tables=1,
            )

        self.assertEqual(len(packet.context_items), 1)
        self.assertEqual(packet.context_items[0].context_id, "ctx_1")
        self.assertEqual(packet.context_items[0].payload.get("content"), "useful text")

    def test_table_payload_excludes_raw_table_dict(self) -> None:
        retrieval_output = {
            "ok": True,
            "top_tables": [
                {
                    "table_name": "good",
                    "total_score": 50,
                    "table": {
                        "payload": {
                            "doc_type": "table",
                            "doc_id": "good::table::0",
                            "prefix": "good",
                            "table_index": 0,
                            "ticker": "AAPL",
                            "fiscal_year": 2024,
                        }
                    },
                }
            ],
            "metadata_used": {"ticker": "AAPL", "fiscal_year": 2024, "form_type": "10-K"},
        }

        with mock.patch("agents.analyst.agent.load_table_data", return_value={"col": ["x"]}):
            packet = build_packet_from_retrieval_output(
                user_query="What happened?",
                retrieval_output=retrieval_output,
                analysis_task={"task_type": "extract", "metric": "revenue"},
                max_tables=1,
            )

        self.assertEqual(len(packet.context_items), 1)
        self.assertNotIn("table_dict", packet.context_items[0].payload)

    def test_table_markdown_render_failure_falls_back_to_csv(self) -> None:
        fake_df = mock.Mock()
        fake_df.to_markdown.side_effect = RuntimeError("no tabulate")
        fake_df.to_csv.return_value = "col1,col2\n1,2\n"
        fake_df.__len__ = mock.Mock(return_value=1)

        with mock.patch("agents.analyst.agent.pd") as fake_pd:
            fake_pd.DataFrame.return_value = fake_df
            rendered = _table_dict_to_markdown({"col1": [1], "col2": [2]})

        self.assertEqual(rendered, "col1,col2\n1,2\n")

    def test_table_dict_markdown_falls_back_to_json(self) -> None:
        with mock.patch("agents.analyst.agent.pd") as fake_pd:
            fake_pd.DataFrame.side_effect = Exception("bad table")
            rendered = build_packet_from_retrieval_output.__globals__["_table_dict_to_markdown"]({"a": object()})

        self.assertTrue(rendered.startswith("{"))

    def test_packet_builder_prefers_planner_target_id_for_context_items(self) -> None:
        retrieval_output = {
            "ok": True,
            "targets": [
                {
                    "target_id": 1,
                    "ticker": "AAPL",
                    "fiscal_year": 2024,
                    "form_type": "10-K",
                }
            ],
            "top_tables": [
                {
                    "table_name": "good-text",
                    "total_score": 40,
                    "table": {
                        "payload": {
                            "doc_type": "text",
                            "content": "useful text",
                            "ticker": "AAPL",
                            "fiscal_year": 2024,
                            "form_type": "10-K",
                        }
                    },
                },
            ],
            "metadata_used": {"ticker": "AAPL", "fiscal_year": 2024, "form_type": "10-K"},
        }

        packet = build_packet_from_retrieval_output(
            user_query="What happened?",
            retrieval_output=retrieval_output,
            analysis_task={"task_type": "extract", "metric": "revenue"},
            max_tables=1,
        )

        self.assertEqual(packet.targets[0]["target_id"], 1)
        self.assertEqual(packet.context_items[0].target_id, "1")


if __name__ == "__main__":
    unittest.main()
