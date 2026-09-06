# PR8 single controlled live run

Evaluated implementation: `99cf1b20e2796741df735365e78e241ea6667c2d`.
Finished: 2026-09-06 12:59:33 UTC. PR26 remains draft and blocked.

## Result and disposition

The unchanged 15-case gate ran once: 14 non-critical rows, one critical row,
deterministic score 0.981481, zero evaluator errors. The strict gate is **failed**
because its critical-failure allowance is zero. No terminal analyst timeout was
reported. The failed case is `AGENT_V1_ANALYST_001`, with
`ANALYST_GROUNDING_INVALID` / `GROUNDING_ROW_TEXT_MISMATCH` on all three grounding
decisions. Calculator use did not yield an accepted grounded final answer.

Raw candidates and decisions remain in
`artifacts/evals/agents/v1/baselines/99cf1b2/per_query.jsonl`.
The calculation claim contains a full formula sentence, while its bound row is
labelled "Services net sales increase percentage" with value "12.87%". The model
did not resolve the binding mismatch within the existing retry budget. This
observation does not justify weakening PR6 validation or extending PR8.

No rerun, post-gate runtime change, release exception or merge was performed.
Earlier PR7/PR24 exceptions do not apply. Stop pending owner direction.

## Execution controls

- Exact detached implementation SHA; clean tracked tree at launch, fresh
  checkpoint database, new SHA-keyed output directory.
- Same live evaluator driver, model digests/settings, dataset, corpus, index,
  120-second model timeout, retry budgets and gate thresholds.
- AC power, Low Power Mode off, no Chrome/Safari processes in all 188 samples;
  owned `caffeinate -dimsu` assertion monitored throughout and released afterward.
- Maximum sample gap 16.399 seconds; no control violation. Controller wall time
  3,070.766 seconds; evaluator time 3,063.972 seconds.
- 40 samples recorded transient heavy competing processes. None reached three
  consecutive heavy samples, which would have aborted without automatic rerun.
  Median CPU idle 83.175%, range 69.55–89.62%. This is sampled workload control,
  not exclusive hardware; all bursts remain in `system_samples.jsonl`.
- Same 582-point index before/after, SHA-256
  `6c72627b05dbec1e7f65d2132b8572b9fa12c47af711f2a12b8b776967994cee`.
- Python 3.11.14; environment-only pytest 9.0.2; no dependency changes.

`provenance.json`, `system_samples.jsonl` and `completion.json` are original
controller output. The live `manifest.json` records driver/source/raw-output
hashes, complete redacted evaluator arguments, review disposition and limitations.
Absolute local paths and the local SEC contact are excluded from publication.

## Review and commands

The renderer P1 was fixed before freeze at `99cf1b2`. Fresh review of that SHA
completed on 2026-09-05; its P2 missing-artifact reference was addressed by evidence
commit `2803adf`, remotely verified, and resolved before launch. No unresolved
review findings remained at launch.

Local-only execution commands are represented without machine-specific paths:

```text
PYTHONPATH=.:src <finsearch-arm-python> <LOCAL_PREFLIGHT_DRIVER>
git switch --detach 99cf1b20e2796741df735365e78e241ea6667c2d
PYTHONPATH=.:src <finsearch-arm-python> <LOCAL_PR8_CONTROL_DRIVER>
<finsearch-arm-python> <LOCAL_PR8_POSTFLIGHT_RECORDER>
git switch codex/pr8-filing-period-semantics
git diff --check
git diff 99cf1b2 -- src scripts tests data/evals/agents/v1
```

The control driver invokes the unchanged live driver once; the postflight recorder
only reads completed results/fingerprints and creates the manifest. No evaluator
was rerun during evidence publication. Publication changes are limited to this
control directory, `artifacts/evals/agents/v1/baselines/99cf1b2/`, and the current
status in `docs/EVALUATION_BASELINES.md` and `docs/IMPLEMENTATION_PLAN.md`.
