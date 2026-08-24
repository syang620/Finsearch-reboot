from __future__ import annotations

from typing import Any
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from llm_client import OllamaChatModel


def _generate_tool_call(arguments: Any):
    raw_tool_calls = [
        {
            "function": {
                "name": "financial_evaluator",
                "arguments": arguments,
            },
            "id": "tool-call-1",
        }
    ]
    response = {
        "model": "qwen3:14b",
        "message": {"content": "", "tool_calls": raw_tool_calls},
        "done_reason": "stop",
    }
    with patch("llm_client._ollama_chat_completion", return_value=response):
        result = OllamaChatModel(model_name="qwen3:14b")._generate(
            [HumanMessage(content="Calculate the result")]
        )
    return result.generations[0].message, raw_tool_calls


def test_ollama_accepts_decoded_dict_tool_arguments() -> None:
    arguments = {
        "expression": "(current - prior) / prior * 100",
        "variables": {"current": "96.2", "prior": "85.2"},
    }

    message, raw_tool_calls = _generate_tool_call(arguments)

    assert message.tool_calls == [
        {
            "name": "financial_evaluator",
            "args": arguments,
            "id": "tool-call-1",
            "type": "tool_call",
        }
    ]
    assert message.invalid_tool_calls == []
    assert message.additional_kwargs["tool_calls"] == raw_tool_calls


def test_ollama_still_accepts_json_string_tool_arguments() -> None:
    message, _ = _generate_tool_call(
        '{"expression":"a+b","variables":{"a":"20","b":"22"}}'
    )

    assert message.tool_calls[0]["args"] == {
        "expression": "a+b",
        "variables": {"a": "20", "b": "22"},
    }
    assert message.invalid_tool_calls == []


def test_ollama_preserves_malformed_json_as_invalid_tool_call() -> None:
    message, _ = _generate_tool_call('{"expression":')

    assert message.tool_calls == []
    assert len(message.invalid_tool_calls) == 1
    assert message.invalid_tool_calls[0]["args"] == '{"expression":'
