"""Compatibility shim. Prefer `mcp_server.server`."""

from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mcp_server.server import *  # noqa: F401,F403,E402


if __name__ == "__main__":
    from mcp_server.server import main

    main()
