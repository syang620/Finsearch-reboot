# Semantic-v2 local-once protocol: inactive reconstruction candidate

`scripts/operations/run_semantic_once.py` is the active reconstruction candidate for a
future semantic-v2 baseline attempt. It replaces the fresh-v7 revision-2 operational
design with a smaller protocol whose security boundary matches its intended use: a
trusted local operator running reviewed code from a private checkout.

The candidate grants no execution authority. Its approval file,
`docs/evals/semantic_answer_v2_local_once_v1_approval.json`, does not exist. The fixed
attempt root, `/Users/shicheny/.local/share/finsearch/semantic-baseline/local-once-v1`,
must also remain absent until an exact candidate review and a separate approval commit
have both completed cleanly. Reconstruction and review require no model calls,
retrieval cases, re-indexing, or changes to Qdrant.

## Trust boundary

The protocol detects accidental or ordinary operator changes before launch. It does
not claim isolation from malicious code running concurrently as the same macOS user,
from an administrator, or from mutation of the installed Python environment after its
final check. Such isolation requires a distinct OS account, VM, or equivalent security
principal and is outside this protocol.

Preparation creates the fixed attempt root and its `source` directory with mode 0700.
It initializes an independent Git repository, fetches one full 40-character commit,
checks it out detached, disables hooks, and rejects alternates, grafts, shallow history,
replacement refs, tracked symlinks, and submodules. Verification hashes every tracked
regular file, checks executable modes and the Git tree, and requires no modified or
untracked non-ignored content.

The frozen launcher writes its SHA-keyed cache beneath
`source/.cache/semantic_answer_v2/`. That directory is already ignored by the tracked
`.gitignore`; no external cache symlink, ACL, disk image, inode receipt, or native
filesystem bridge is used. Preparation is exclusive and never overwrites an existing
attempt root. A partial or failed preparation remains evidence and requires a new
reviewed namespace rather than deletion and reuse.

## Approval and execution contracts

After a clean exact-head candidate review, a separate commit may add the approval file.
Its closed schema binds:

- contract version `1`, authorization ID `SEMANTIC-V2-LOCAL-ONCE-V1`, the fixed root,
  and status `approved_candidate_not_execution`;
- the clean candidate review identity and body digest;
- the pinned interpreter path and SHA-256;
- exact SHA-256 values for the controller, environment helper, frozen launcher,
  historical integration and quality approvals, and canonical-index attestation.

The reviewed candidate must be an ancestor of the prepared approval commit, and the
approval JSON must be the only changed path between those commits. The environment
helper retains contract version 2 and freezes one effective environment without
serializing credential values. That effective environment must explicitly select
`QDRANT_HOST=127.0.0.1`, `QDRANT_PORT=6333`, and the collection named by the bound
canonical-index attestation. Both the preflight identity check and benchmark child use
that same target.

Execution additionally requires an external, owned, mode-0600
`execution_authorization.json`. Its closed schema binds the approval digest, prepared
commit, tree and tracked-file manifest, fixed artifact root, one invocation, explicit
user authorization, and a clean review of the final approval commit. The controller
opens, parses, and hashes the same descriptor bytes before checking remote review
provenance.

## Lifecycle and failure behavior

The supported commands are:

```text
run_semantic_once.py prepare --repository REPOSITORY --head FULL_COMMIT
run_semantic_once.py preflight [--env-file PATH]
run_semantic_once.py execute [--env-file PATH]
```

`preflight` runs imports, approval and launcher checks, remote review verification, and
read-only canonical Qdrant identity verification in the pinned interpreter. It does not
execute benchmark cases or consume the attempt.

`execute` requires the separate authorization, reruns preflight, revalidates the Git
checkout and absent artifacts, then exclusively and durably writes `consumed.json`
immediately before process creation. Any later error consumes the attempt. The child
runs in a new process group with the frozen environment and argv. Console output is
written to mode-0600 `console.log` with credential-like environment values redacted.
SIGINT and SIGTERM are forwarded, and a durable `outcome.json` records the terminal
state whenever storage remains available. Each observed signal is forwarded at most
once. The outcome records an explicit signal-observation cutoff; signals that arrive
while that outcome is being persisted are delivered using the caller's prior signal
disposition after persistence rather than being discarded.

Existing marker, outcome, console, staging, or SHA-keyed cache content forbids another
launch. A marker without an outcome is unresolved and never grants retry authority.
All historical fresh-v7 and earlier approvals, wrappers, results, and consumption
records remain authoritative audit evidence and are not repurposed.

## Rollout gates

1. Review the uncommitted reconstruction diff and pass the focused and historical
   controller tests without live evaluation calls.
2. Commit and push the inactive candidate for an exact-head review.
3. After a clean review, add its candidate approval in a separate commit and review
   that exact head.
4. Prepare the independent checkout and run non-consuming preflight.
5. Present the prepared identities and preflight evidence for separate, explicit
   one-use execution authorization.
6. Execute once, preserve all evidence, and never repair or recycle the namespace.

Review findings must be assessed against the trusted-local boundary above. A demand
for hostile same-user isolation changes the architecture and should trigger a separate
OS-identity or VM design instead of another pathname or ACL patch.
