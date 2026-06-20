#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "chunking" / "benchmark_chunking_retrieval.py"
    runpy.run_path(str(script), run_name="__main__")
