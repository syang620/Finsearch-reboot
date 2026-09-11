# Fresh-v7 revision 2: superseded stabilization candidate

This design is retained as historical review evidence. It is superseded for future
operational use by the [local-once protocol](semantic_local_once_status.md) and must
not receive a new approval or execution authorization. The replacement adopts the
trusted-local boundary that this document already stated explicitly and removes the
disk-image, external-cache, ACL, and inode-binding machinery that could not provide
hostile same-user isolation.

The revision-2 implementation is `scripts/operations/run_semantic_v7.py`, backed by
the standard-library-only `semantic_v7_snapshot.py` preparation utility. This
revision does not delegate operation orchestration to the v4/v2 wrappers. The
frozen benchmark launcher and its workload adapter still provide the benchmark
behavior. No revision-2 approval or execution-authorization record exists.

The original fresh-v7 wrapper and approval are retained as historical bytes;
they are not the entrypoint for this revision and must not receive an execution
authorization. Revision 2 supersedes their operational use without rewriting
them. V6 remains retired/currently unexecutable, with its wrapper and approval
unchanged. All earlier consumption records remain authoritative.

## Contracts and trust boundary

The external attempt root is fixed:
`/Users/shicheny/.local/share/finsearch/semantic-baseline/fresh-v7-r2`.
Preparation exclusively creates it with mode 0700 and never overwrites an
existing preparation. A failed preparation is preserved for inspection; there
is no automatic cleanup/retry or conversion into execution authority.
The preparation receipt binds the attempt root and writable cache by device,
inode, owner, mode, and an exact macOS deny-delete ACL. The ACL prevents ordinary
same-user rename or removal through child launch while permitting cache-content
writes; the controller revalidates the recorded identities at each execution gate.

Preparation fetches a full commit and its ancestry into an independent Git
repository with an empty hook template and no shared object storage. Every
tracked file's bytes and executable mode must match its Git object. Tracked
symlinks and submodules are rejected. The only allowed untracked source-tree
entry is `.cache`, a sealed symlink to this attempt's external cache directory.
The independent repository's verified `.git/info/exclude` contains only
`/.cache`, so existing clean-checkout checks accept the symlink without changing
the tracked `.gitignore` or hiding unexpected files from snapshot verification.
The image includes Git history so existing provenance checks remain usable.

`hdiutil` creates a UDRO image, mounted read-only. Its backing file is mode 0400
with the macOS user-immutable flag. Both image digest and the complete mounted
tracked-file manifest are verified. Ordinary edits/checkouts in the original
repository cannot affect execution. The host administrator, malicious same-user
code that deliberately removes filesystem protections, and mutations of the
installed Python environment are outside this isolation boundary. The existing
interpreter hash check remains mandatory; this is not a hermetic container.

The new approval path is
`docs/evals/semantic_answer_v2_fresh_v7_r2_approval.json`. It is absent by default.
After clean candidate review, it must contain:

- `status: approved_candidate_not_execution`, `execution_contract_version: "2"`,
  `authorization_id: SEMANTIC-V2-FRESH-V7-R2`, and the exact external `artifact_root`;
- `effective_environment_contract_version: "2"` and the controller's SHA-256 as
  `preflight_implementation_sha256`;
- the pinned interpreter path and SHA-256;
- exactly the `files_sha256` mapping required by the controller's `BOUND_FILES`;
- the existing strict review fields: `review_status: completed_clean`,
  `pull_request`, `review_comment_id`, `review_url`, `review_body_sha256`, and
  full `reviewed_commit` for the candidate.

Source under `src`, `scripts`, and `data` cannot change between candidate review
and the approval commit. The old v7 integration approval is used solely for the
unchanged launcher's frozen provenance checks; the new controller separately
requires its own approval and execution authorization.

The external `execution_authorization.json` has a closed schema matching
`authorization()`: version, authorization ID, status
`authorized_for_one_invocation`, boolean explicit user authorization, integer
`max_invocations: 1`, approval SHA-256, final reviewed commit, image SHA-256,
artifact root, and strict clean-review fields. It must bind the final approval
commit's clean review. The record is opened once without symlink following;
ownership/type/mode are checked on the descriptor, and the same bounded bytes
are parsed and hashed. Accepted bytes authorize that invocation; replacing the
pathname is not an in-flight revocation mechanism. Interrupt the process to stop.

## Preparing and validating after review

Preparation must execute the standard-library-only utility from its Git blob,
not import code from a mutable worktree. The following recipe takes the reviewed
approval commit as its final argument (replace `FULL_REVIEWED_APPROVAL_COMMIT`):

```sh
/Users/shicheny/miniforge3/envs/finsearch-arm/bin/python -I -c '
import os, subprocess, sys
repo, head = sys.argv[1:]
env = {k:v for k,v in os.environ.items() if not k.startswith("GIT_")}
env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null")
source = subprocess.check_output(["/usr/bin/git", "--no-replace-objects", "-C", repo,
    "show", head + ":scripts/operations/semantic_v7_snapshot.py"], env=env)
sys.argv = ["sealed-preparation", "prepare", "--repository", repo, "--head", head]
exec(compile(source, "<reviewed-preparation-blob>", "exec"))
' /Users/shicheny/Documents/GitHub/FinSearch-semantic-benchmark-v2 FULL_REVIEWED_APPROVAL_COMMIT
```

Then, from the mounted `source` directory, use the pinned interpreter to run
`scripts/operations/run_semantic_v7.py preflight --env-file` with the dedicated
external env-file path. This mode does not require execution authorization and
cannot create a consumption marker, console log, outcome, or staging directory.
It performs imports, remote review checks, frozen-launcher verification, and
read-only Qdrant identity validation; it does not call models or retrieve cases.
Preparation and preflight are allowed to leave their distinct preparation files
and attempt lock. They never create one-use execution authority.

Use `PYTHONPATH=src:.` if that is the approved invocation environment. The
controller preserves inherited values and rejects paths that resolve outside
the mounted source. It does not silently fix an unsafe import environment.
The env file is frozen once; it is not passed to the benchmark. Credentials are
neither included in the image nor serialized into the execution contract.

Only after separate user authorization may the strict external authorization
record be created and `execute` invoked. No command in this document grants that
permission. The actual Popen argv includes a fixed standard-library signal-mask
bootstrap followed by the file-path launcher; the contract hashes the entire
argv. The bootstrap clears the inherited blocked signal mask before exec.

## Failure and preservation semantics

The lifecycle is lock, validation, final preflight, consumption, spawn, and
outcome. The lock is OS-held. Existing marker/outcome/log/staging or nonempty
attempt cache prevents execution. Marker creation is exclusive and fsynced;
even a partially written marker consumes the attempt. SIGINT/SIGTERM before
consumption skips launch. After consumption, caught exceptions/signals produce
an outcome; interruption is forwarded to the child's process group, with a
bounded grace period before forced termination.

SIGKILL, power loss, and failed storage can prevent an outcome. A marker without
an outcome is unresolved, never rerun authority. Signals arriving after the
documented outcome observation cutoff do not retroactively change that outcome.
Artifacts are append-only. The image and mounted source are retained for audit;
detach only after verifying no child processes remain, and never automatically
remove the image, caches, marker, or other evidence.

The unchanged launcher verifies the complete historical Qdrant identity before
cases and after finalization. Any verification failure withholds the official
summary. V7 evaluates the surviving historically fingerprinted collection; the
deleted original serialized build cache is not independently byte-reproducible.

## Historical rollout (superseded; do not execute)

1. Pass controller, canonical-index, existing semantic-control, compilation,
   diff, and historical-content checks, including the native image integration.
2. Review the entire preparation-to-outcome sequence at the candidate commit.
3. If clean, create the separate revision-2 approval and review that exact commit.
4. Prepare its image and run non-consuming preflight from the sealed source.
5. Present identities, results, and absent-artifact checks for separate one-use
   user authorization. Never reuse an earlier v7 or v6 authorization.
