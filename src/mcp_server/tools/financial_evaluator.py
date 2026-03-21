from __future__ import annotations

import argparse
from typing import Any, Dict
import re
import simpleeval

from mcp.server.fastmcp import FastMCP


def _to_number(x: str) -> float:
    """
    Parse numbers like:
    "1,234.56", "$1,234", "(123.4)" -> -123.4, "12%" -> 0.12
    """
    s = str(x).strip()
    if s == "":
        raise ValueError("Empty numeric value")

    is_percent = s.endswith("%")
    s = s.replace("$", "").replace(",", "").strip()

    # (123.4) => -123.4
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1].strip()

    val = float(s)
    return val / 100.0 if is_percent else val


def _classify_error_code(exc: BaseException) -> str:
    message = str(exc).strip().lower()
    if "not defined" in message:
        return "unknown_variable"
    if "invalid syntax" in message or "syntaxerror" in message:
        return "invalid_syntax"
    if isinstance(exc, ValueError):
        if "unsupported characters" in message:
            return "unsupported_characters"
        if "empty numeric value" in message:
            return "invalid_number"
        return "invalid_input"
    if isinstance(exc, ZeroDivisionError):
        return "division_by_zero"
    if isinstance(exc, SyntaxError):
        return "invalid_syntax"
    if isinstance(exc, NameError):
        return "unknown_variable"
    if isinstance(exc, simpleeval.InvalidExpression):
        return "invalid_expression"
    if isinstance(exc, simpleeval.NameNotDefined):
        return "unknown_variable"
    if isinstance(exc, simpleeval.FunctionNotDefined):
        return "unsupported_function"
    return "evaluation_error"


def financial_evaluator(variables: Dict[str, str], expression: str) -> Dict[str, Any]:
    """
    Safely evaluates a math expression using provided variables.

    Args:
        variables: mapping of variable name -> numeric string
        expression: e.g. "revenue - cost" or "(revenue - cost) / revenue"
    Returns:
        Computed result as float
    """
    try:
        cleaned_vars = {k: _to_number(v) for k, v in variables.items()}

        # Optional extra hardening: only allow common math tokens + names
        # (simpleeval is already fairly safe, but this reduces surprise.)
        if not re.fullmatch(r"[0-9A-Za-z_\s\.\+\-\*\/\(\)]+", expression):
            raise ValueError("Expression contains unsupported characters")

        result = float(simpleeval.simple_eval(expression, names=cleaned_vars))
        return {
            "result": result,
            "expression": expression,
            "variables": {str(k): str(v) for k, v in variables.items()},
        }
    except Exception as exc:
        return {
            "error": str(exc).strip() or exc.__class__.__name__,
            "error_code": _classify_error_code(exc),
            "expression": expression,
            "variables": {str(k): str(v) for k, v in variables.items()},
        }


def register_tools(mcp: FastMCP) -> None:
    mcp.tool()(financial_evaluator)


def build_mcp_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    mount_path: str = "/",
) -> FastMCP:
    mcp = FastMCP("financial-math", host=host, port=port, mount_path=mount_path)
    register_tools(mcp)
    return mcp


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the financial evaluator MCP server.")
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse", "streamable-http"),
        default="stdio",
    )
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--mount-path", default=None)
    args = parser.parse_args(argv)

    # IMPORTANT: don't print() to stdout in stdio transport (breaks JSON-RPC)
    build_mcp_server(
        host=args.host or "127.0.0.1",
        port=int(args.port) if args.port is not None else 8000,
        mount_path=args.mount_path or "/",
    ).run(
        transport=args.transport,
        mount_path=args.mount_path,
    )


if __name__ == "__main__":
    main()
