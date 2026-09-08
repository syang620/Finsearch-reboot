# Diagnostic command record

Paths below use `$USER_HOME` in place of the local home directory. The SEC contact
environment variable was supplied locally and is intentionally not reproduced.

```sh
pmset -g batt
pmset -g custom
PYTHONPATH=src:. $FINSEARCH_PYTHON -c 'from scripts.evals.retrieval.run_benchmark_v3 import controls; ...'

PYTHONPATH=src:. $FINSEARCH_PYTHON -u \
  scripts/diagnostics/observe_semantic_workload.py \
  --output .cache/semantic_workload_investigation_20260908/formal_observation.jsonl \
  --duration-seconds 1200 --interval-seconds 1

PYTHONPATH=src:. $FINSEARCH_PYTHON -u \
  scripts/diagnostics/simulate_semantic_preflight_readonly.py \
  --index-manifest $USER_HOME/Documents/GitHub/FinSearch-retrieval-benchmark/.cache/retrieval_benchmark_v2_index/completed.json \
  --output .cache/semantic_workload_investigation_20260908/preflight_simulation.json

lsof -nP -iTCP:6333 -sTCP:LISTEN
docker ps --format '{{.ID}} {{.Image}} {{.Names}} {{.Status}}'
tmutil status
mdutil -s /
log show --style compact --start '2026-09-08 10:27:30' \
  --end '2026-09-08 10:29:30' \
  --predicate '(process == "git") OR (process CONTAINS[c] "Safari") OR (process CONTAINS[c] "Chrome") OR (process CONTAINS[c] "Docker") OR (process CONTAINS[c] "mds") OR (process CONTAINS[c] "backup")'

PYTHONPATH=src:. $FINSEARCH_PYTHON -m pytest \
  tests/diagnostics/test_observe_semantic_workload.py -q
git diff --check
```

The historical unified-log query returned no matching retained event. Process
command/parent recovery for the old `git` and browser samples was therefore
unavailable. No hooks, Git configuration, service configuration or process state
was changed.
