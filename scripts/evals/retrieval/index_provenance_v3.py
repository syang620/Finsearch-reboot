"""Read-only binding to the archived v2 build and post-build index fingerprint.

No ranking results are consulted and no index is built, modified or queried.
"""
import json
from pathlib import Path

from evals.retrieval_benchmark_v2 import sha256


# Earliest archived v2 baseline: binds the completed build record to the exact
# served dense/sparse vectors, payloads, point IDs and collection configuration.
# The old builder did not capture a live vector fingerprint at build completion;
# do not misrepresent this historical post-build capture as an immediate one.
REFERENCE = Path("artifacts/evals/retrieval/benchmark_v2/baselines/2d50cfe0dc7b624676b472b7407aab7dc11f9648/manifest.json")
REFERENCE_SHA256 = "625047fc2cb5039ec0ee44af4979e7c2ee5bde32b6587a286d04658dce54219e"


def frozen_index_reference():
    if sha256(REFERENCE) != REFERENCE_SHA256:
        raise ValueError("Archived index provenance hash mismatch")
    archived = json.loads(REFERENCE.read_text())
    if not archived["index"]["completed"] or archived["index_before"] != archived["index_after"]:
        raise ValueError("Archived index reference is incomplete or changed")
    return {"manifest_path": REFERENCE.as_posix(), "manifest_sha256": REFERENCE_SHA256,
            "build": archived["index"], "snapshot": archived["index_before"],
            "capture_scope": "earliest archived post-build v2 snapshot; not a new v3 live fingerprint"}


def verify_frozen_index(index, snapshot, embedded_sha256):
    reference = frozen_index_reference()
    if index != reference["build"]:
        raise ValueError("Index build manifest differs from frozen historical build")
    if embedded_sha256 != reference["build"]["embedded_sha256"]:
        raise ValueError("Original build embedding cache hash mismatch")
    if snapshot != reference["snapshot"]:
        raise ValueError("Live index differs from frozen dense/sparse/payload/config fingerprint")
    return reference
