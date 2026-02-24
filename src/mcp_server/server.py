from __future__ import annotations

from pathlib import Path
import sys

from mcp.server.fastmcp import FastMCP

# Allow running this file directly without installing the package.
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mcp_server.tools.financial_evaluator import register_tools as register_financial_math_tools  # noqa: E402
from mcp_server.tools.sec_retrieval import register_tools as register_sec_retrieval_tools  # noqa: E402


def build_mcp_server() -> FastMCP:
    mcp = FastMCP("finsearch-tools")
    register_financial_math_tools(mcp)
    register_sec_retrieval_tools(mcp)
    return mcp


def main() -> None:
    build_mcp_server().run(transport="stdio")


if __name__ == "__main__":
    main()
