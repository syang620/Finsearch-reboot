import asyncio

from agents.analyst.agent import _FinancialToolRuntime, _default_financial_tool_script
from mcp_server.tools import financial_evaluator as financial_evaluator_module


def test_main_passes_host_and_port_to_fastmcp(monkeypatch):
    captured = {"build": None, "run": None}

    class _FakeServer:
        def run(self, **kwargs):
            captured["run"] = kwargs

    def _fake_build_mcp_server(**kwargs):
        captured["build"] = kwargs
        return _FakeServer()

    monkeypatch.setattr(financial_evaluator_module, "build_mcp_server", _fake_build_mcp_server)

    financial_evaluator_module.main(
        ["--transport", "sse", "--host", "127.0.0.1", "--port", "54321", "--mount-path", "/sse"]
    )

    assert captured["build"] == {
        "host": "127.0.0.1",
        "port": 54321,
        "mount_path": "/sse",
    }
    assert captured["run"] == {
        "transport": "sse",
        "mount_path": "/sse",
    }


def test_runtime_supports_concurrent_calls_against_real_sse_server():
    async def _run():
        runtime = await _FinancialToolRuntime.create(
            tool_script=_default_financial_tool_script(),
            timeout_s=20.0,
        )
        try:
            results = await asyncio.gather(
                runtime.call_tool(
                    "financial_evaluator",
                    {"expression": "a + b", "variables": {"a": "20", "b": "22"}},
                ),
                runtime.call_tool(
                    "financial_evaluator",
                    {"expression": "revenue / shares", "variables": {"revenue": "100", "shares": "4"}},
                ),
            )
        finally:
            await runtime.aclose()

        first, second = results
        assert first["status"] == "success"
        assert second["status"] == "success"
        assert first["artifact"]["result"] == 42.0
        assert second["artifact"]["result"] == 25.0

    asyncio.run(_run())


def test_financial_evaluator_returns_structured_success_payload():
    result = financial_evaluator_module.financial_evaluator(
        {"revenue": "100", "shares": "4"},
        "revenue / shares",
    )

    assert result["result"] == 25.0
    assert result["expression"] == "revenue / shares"
    assert result["variables"] == {"revenue": "100", "shares": "4"}


def test_financial_evaluator_returns_structured_error_code():
    result = financial_evaluator_module.financial_evaluator(
        {"revenue": "100"},
        "revenue / unknown",
    )

    assert result["error_code"] == "unknown_variable"
    assert "unknown" in result["error"].lower()
