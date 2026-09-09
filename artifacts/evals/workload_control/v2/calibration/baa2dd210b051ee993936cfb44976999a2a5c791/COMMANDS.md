# Commands

Published paths are repository-relative and sanitized. Absolute user paths were
used only in local execution.

```bash
PYTHONPATH=src:. $FINSEARCH_PYTHON scripts/diagnostics/run_workload_control_v2_calibration.py \
  --scenario <SCENARIO> --raw-output <APPEND_ONLY_RAW> \
  --summary-output <LOCAL_SUMMARY> [--index-manifest <LOCAL_INDEX_MANIFEST> \
  --preflight-output <APPEND_ONLY_S3_PREFLIGHT>]

gzip -n <APPEND_ONLY_RAW>

PYTHONPATH=src:. $FINSEARCH_PYTHON \
  scripts/diagnostics/analyze_workload_control_v2_calibration.py \
  --preregistration docs/evals/workload_control_v2_preregistration.json \
  --scenario <ID> <RAW> ... \
  --s3-preflight <S3_PREFLIGHT> --frozen-provenance <FROZEN_STARTED_JSON> \
  --pr32-raw <PR32_RAW> --prior-disposition <DISPOSITION> ... \
  --output <APPEND_ONLY_COMPARISON>

PYTHONPATH=src:. $FINSEARCH_PYTHON -m pytest -q tests/diagnostics \
  --import-mode=importlib
PYTHONPATH=src:. $FINSEARCH_PYTHON -m pytest tests -q \
  --import-mode=importlib
git diff --check
```
