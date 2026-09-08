# Semantic-v2 workload-control investigation

Status: diagnostic tooling prepared; formal 15–20 minute observation pending
required AC power. This work is isolated on `codex/semantic-workload-investigation`
from reviewed PR31 head `7667af52169fb12965adf5680bae58a74c7c977e`.
PR31 remains frozen, draft, release-blocked and unmerged. This investigation
does not authorize another baseline.

## Frozen boundary

`investigation_provenance.json` records the benchmark, scorer, launcher, control
and two invalid-diagnostic hashes. The same hashes must verify after observation.
No file under `src/`, `scripts/evals/`, `data/` or existing artifact paths may
change. New scripts live only under `scripts/diagnostics/`; their output is
diagnostic evidence, not a benchmark result.

The authoritative observation calls the exact imported frozen `controls()`
function once per one-second diagnostic sample. This does not change the
launcher's boundary sampling. Immediately afterward, a richer `ps` snapshot
records PID, PPID, executable, command line, CPU, memory, start time, current
working directory where accessible, and parent ancestry. Because the two reads
are sequential, a very short process can disappear between them; unmatched
historical or live control entries must be reported as unavailable rather than
guessed. User-home path prefixes are replaced by `$USER_HOME` in publishable data.

The formal window is 1,200 seconds: passive observation, an explicitly marked
read-only launch-sequence simulation, then continued observation. Awake protection
is active. The simulation verifies repository/freeze state, service/model identity,
both indexes, imports and planner construction, followed by the existing 30-second
settle. It performs no benchmark case, planner call, analyst call, retrieval query,
embedding, reranker call or other model inference. The reranker health probe is
intentionally omitted because it would be inference; this limitation will be in
the final report.

No observed process is killed. No threshold, exemption, cadence, production code,
model, timeout, retry, benchmark case, gold label or historical artifact changes.
The final recommendation must be exactly one of `ENVIRONMENT_CAN_BE_CLEANED`,
`CONTROL_CONTRACT_NEEDS_REVIEW` or `ROOT_CAUSE_UNRESOLVED`.
