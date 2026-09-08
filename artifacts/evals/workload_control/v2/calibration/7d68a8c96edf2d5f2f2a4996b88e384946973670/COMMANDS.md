# Commands

Paths are sanitized for publication. The exact local execution plan used absolute
paths; no absolute user-home path is committed.

```bash
PYTHONPATH=src:. $FINSEARCH_PYTHON scripts/diagnostics/run_workload_control_v2_calibration.py \
  --scenario <preregistered-scenario> --raw-output <append-only-raw> \
  --summary-output <append-only-summary>

SEC_USER_AGENT=$EXISTING_SEC_USER_AGENT PYTHONPATH=src:. $FINSEARCH_PYTHON \
  scripts/diagnostics/run_workload_control_v2_calibration.py \
  --scenario S3_REQUIRED_SERVICE_ACTIVITY \
  --index-manifest $USER_HOME/Documents/GitHub/FinSearch-retrieval-benchmark/.cache/retrieval_benchmark_v2_index/completed.json \
  --preflight-output <append-only-preflight> \
  --raw-output <append-only-raw> --summary-output <append-only-summary>

gzip -n -c <raw.jsonl> > <raw.jsonl.gz>

PYTHONPATH=src:. $FINSEARCH_PYTHON \
  scripts/diagnostics/analyze_workload_control_v2_calibration.py \
  --preregistration docs/evals/workload_control_v2_preregistration.json \
  --scenario <ID> <RAW> ... \
  --pr32-raw artifacts/diagnostics/semantic-v2-workload-controls/a252feeb204ae417e2e024788cfbbff941b319ce/formal_observation.jsonl \
  --prior-disposition <29bf-disposition> \
  --prior-disposition <488e112-disposition> \
  --output <append-only-comparison>

PYTHONPATH=src:. $FINSEARCH_PYTHON -m pytest -q tests/diagnostics
git diff --check
```
