# Semantic-v2 workload-control investigation

## Decision

**`CONTROL_CONTRACT_NEEDS_REVIEW`**

The 20-minute diagnostic did not produce a reproducible clean external procedure.
Even with AC power, Low Power Mode off, awake protection, no Chrome/Safari process
and no benchmark/model/retrieval call, the exact frozen control reported a violation
in 235/1,200 one-second samples (19.6%). Required current tooling was a substantial
source: ChatGPT/Codex UI processes crossed the threshold in 142 process-samples,
and Docker Desktop's renderer did so in 85. The latter belongs to the process tree
hosting the required Qdrant container.

This report does not change or criticize the threshold by inference. It recommends
a separate control-design decision before any new run authority. PR31 remains
frozen, release-blocked and unmerged; neither diagnostic is a baseline.

## Method and validity

The observation ran from 15:52:57 to 16:12:57 UTC on 2026-09-08 at diagnostic
implementation `a252feeb204ae417e2e024788cfbbff941b319ce`, forked from reviewed
PR31 head `7667af52169fb12965adf5680bae58a74c7c977e`. It collected 1,200 samples
at one-second diagnostic intervals. Every sample called the imported frozen
`controls()` function; a second immediate process-table read added identity,
command, cwd and ancestry without replacing the frozen decision.

All samples recorded AC power and Low Power Mode zero. Browser count was zero.
The observer held awake protection. No process was killed. No benchmark case,
planner/analyst call, retrieval query, embedding, reranker request or other model
inference occurred. A 41-second marked simulation performed only read-only
repository/freeze, service identity, index identity, import and planner-construction
steps, followed by the existing 30-second settle.

| Phase | Samples | Violation samples |
| --- | ---: | ---: |
| Passive | 675 | 134 |
| Read-only preflight simulation | 41 | 11 |
| Post-simulation | 484 | 90 |
| **Total** | **1,200** | **235** |

Process occurrences overlap, so their counts do not sum to unique violation samples.
The rich detail read follows the authoritative control read; sub-second exits can
make their counts differ. The raw artifact preserves both.

## Violation shape

| Process | Frozen samples | Peak CPU | Longest consecutive | Shape |
| --- | ---: | ---: | ---: | --- |
| ChatGPT | 125 | 99.7% | 2 | recurring short bursts; median 6 s between burst starts |
| Docker Desktop Helper (Renderer) | 85 | 96.2% | 1 | recurring periodic single samples; median 10 s |
| Codex (Renderer) | 16 | 160.3% | 1 | recurring UI-renderer samples; median 60 s |
| Python diagnostic/setup processes | 7 | 449.4% | 4 | startup/preflight bursts |
| `duetexpertd` | 7 | 99.7% | 7 | one short 7-s burst |
| `coreaudiod` | 3 | 67.0% | 1 | three isolated samples |
| Phone / Siri inference | 2 each | 118.1% / 99.8% | 2 | one short launch burst each |
| Eight other OS/UI services | 1 each | 50.0–79.8% | 1 | isolated single samples |

No process was sustained at or above the threshold for ten consecutive samples.
The recurring ChatGPT/Codex and Docker events nevertheless made violations common
across the full window rather than confined to preflight.

## Root-cause confidence and required status

### ChatGPT and Codex UI — identity high, trigger medium

Executable paths, commands and ancestry identify the main ChatGPT app, two Codex
renderers and one Codex GPU service. The main process was launched by `launchd`;
renderers/services are children of ChatGPT. Codex-renderer events roughly align
with active task updates, but this observation cannot attribute each ChatGPT main
burst to a specific message or tool event.

Classification: **required tooling activity for this agent-driven run**. It might
be avoidable only with a separately validated terminal-only launch in which Codex
does not supervise the run. This investigation did not test or authorize that.

### Docker Desktop renderer — identity and origin high

The full command names Docker Desktop's renderer. Its ancestors are Docker Desktop
and `com.docker.backend`; cwd is Docker's container data directory. Read-only
checks confirmed `com.docker` owns port 6333 and a `qdrant/qdrant:v1.16.2` container
provides the required index service. The renderer crossed the threshold throughout
passive, simulation and post phases.

Classification: **required current service-hosting process tree; renderer-specific
avoidability unknown**. The control's `com.docker` exemption matches the backend
name but not the renderer path. This is an observed identity mismatch, not authority
to edit the exemption or host Qdrant differently.

### Historical `git` — cause unresolved, confidence low

The fresh invalid diagnostic retained only basename `git` and CPU 59.8%; it has no
PID, PPID, command, cwd or start time. The process had exited before this study.
The formal window observed zero threshold-crossing `git` samples, including the
4.45-second repository/freeze step. A historical unified-log query returned no
matching retained event. It therefore cannot be attributed to Codex polling,
GitHub tooling, hooks, an editor or a shell command. Classification: **unknown**.

### Historical browser count — cause unresolved, confidence low

The same diagnostic stored count one but no identity. The formal window observed
zero Chrome/Safari-rule samples. Consequently the historical process cannot be
distinguished as Safari, Chrome or a related helper. ChatGPT, Codex and Docker
renderers are explicitly *not* counted by the frozen browser substring rule; they
violated only as heavy non-model processes. Classification: **unknown**.

### Operating-system background services — identity high, trigger low

Full paths and `launchd` ancestry identify Phone, FaceTime message store, Siri
inference, media analysis, mobile assets, duet, audio, weather, ecosystem and
Safari Safe Browsing activity. Their external triggers are not recorded. Each was
short; none is benchmark execution. Classification: **unknown external background
activity**, not evidence for a benchmark code defect.

## Launch-sequence simulation

All read-only steps succeeded: frozen repository/launcher checks, Ollama/Qdrant/SEC
health, model digests, 948-point current index, 582-point historical index, planner
imports/construction and a 30-second settle. The reranker health probe was omitted
because this task forbids model inference.

The simulation Python process reached 449.4% in rich sampling and the frozen check
immediately after setup saw Python at 90.8%. After the unchanged 30-second settle,
Python was no longer a violation—but ChatGPT was at 64.1%. Thus the existing settle
works for setup work while unrelated required UI work can still invalidate the
same boundary. No `git` violation reproduced.

## System-event correlation

Docker/Qdrant linkage is directly verified. Time Machine was not running in the
post-observation check. Spotlight was enabled, but no Spotlight process crossed
the threshold. No backup, Chrome, Safari, or Git violation appeared in the formal
window. These negative observations do not reconstruct the two earlier snapshots.

## Recommended next action

Before another baseline is authorized, conduct a separate control-contract review
that explicitly decides how required service-hosting and run-supervision processes
are identified. That review may evaluate a terminal-only workflow, Docker renderer
identity, and whether single-sample system bursts should be treated differently;
this report does **not** approve any exemption, threshold, cadence or persistence
change.

There is no evidence-backed manual procedure that can currently guarantee a clean
agent-supervised run. Do not launch another baseline, merge PR31, complete the
30-case audit without a valid full capture, or declare optimization readiness.

## Claim limits

The evidence supports only workload-control diagnosis under this machine/session.
It does not establish that background load caused analyst timeouts, that the old
`git`/browser samples share these causes, or that observed process rates generalize.
It supports no semantic-quality, retrieval, latency, or before/after claim.
