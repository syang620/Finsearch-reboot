#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    new_script = Path(__file__).resolve().parent / "evals" / "eval_retrieval_v0.py"
    display_path = str(new_script)
    try:
        display_path = str(new_script.relative_to(Path.cwd()))
    except ValueError:
        pass
    print(
        f"[deprecated] Use 'python {display_path}' instead.",
        file=sys.stderr,
    )
    runpy.run_path(str(new_script), run_name="__main__")


if __name__ == "__main__":
    main()
