from __future__ import annotations

from typing import Dict
import re
import simpleeval

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("financial-math")


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


@mcp.tool()
def financial_evaluator(variables: Dict[str, str], expression: str) -> float:
    """
    Safely evaluates a math expression using provided variables.

    Args:
        variables: mapping of variable name -> numeric string
        expression: e.g. "revenue - cost" or "(revenue - cost) / revenue"
    Returns:
        Computed result as float
    """
    cleaned_vars = {k: _to_number(v) for k, v in variables.items()}

    # Optional extra hardening: only allow common math tokens + names
    # (simpleeval is already fairly safe, but this reduces surprise.)
    if not re.fullmatch(r"[0-9A-Za-z_\s\.\+\-\*\/\(\)]+", expression):
        raise ValueError("Expression contains unsupported characters")

    return float(simpleeval.simple_eval(expression, names=cleaned_vars))


def main():
    # IMPORTANT: don't print() to stdout in stdio transport (breaks JSON-RPC)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
