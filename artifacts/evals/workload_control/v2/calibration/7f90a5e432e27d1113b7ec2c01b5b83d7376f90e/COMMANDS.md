# Commands

Published paths are sanitized. The local plan used absolute paths only in ignored
execution files.

```bash
PYTHONPATH=src:. $FINSEARCH_PYTHON scripts/diagnostics/run_workload_control_v2_calibration.py \
  --scenario S2_TERMINAL_ONLY_IDLE --raw-output <append-only-raw> \
  --summary-output <append-only-summary>

gzip -n -c <raw.jsonl> > <raw.jsonl.gz>

PYTHONPATH=src:. $FINSEARCH_PYTHON \
  scripts/diagnostics/analyze_workload_control_v2_calibration.py \
  --preregistration docs/evals/workload_control_v2_preregistration.json \
  --scenario <ID> <RAW> ... --pr32-raw <PR32_RAW> \
  --prior-disposition <29bf-disposition> \
  --prior-disposition <488e112-disposition> \
  --output <append-only-comparison>

PYTHONPATH=src:. $FINSEARCH_PYTHON -m pytest -q tests/diagnostics \
  --import-mode=importlib
PYTHONPATH=src:. $FINSEARCH_PYTHON -m pytest tests -q \
  --import-mode=importlib
git diff --check
```
