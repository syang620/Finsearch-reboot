# Commands

The final analysis reuses immutable `baa2dd2` raw captures. Published paths are
repository-relative and sanitized; absolute paths remained local.

```bash
PYTHONPATH=src:. $FINSEARCH_PYTHON \
  scripts/diagnostics/analyze_workload_control_v2_calibration.py \
  --preregistration docs/evals/workload_control_v2_preregistration.json \
  --scenario <ID> <BAA2DD2_RAW> ... \
  --s3-preflight <BAA2DD2_S3_PREFLIGHT> \
  --frozen-provenance <FROZEN_STARTED_JSON> \
  --pr32-raw <PR32_RAW> --prior-disposition <DISPOSITION> ... \
  --output <APPEND_ONLY_COMPARISON>

cmp <FIRST_COMPARISON> <IDENTICAL_INPUT_RERUN>
PYTHONPATH=src:. $FINSEARCH_PYTHON -m pytest -q tests/diagnostics \
  --import-mode=importlib
PYTHONPATH=src:. $FINSEARCH_PYTHON -m pytest tests -q \
  --import-mode=importlib
git diff --check
```
