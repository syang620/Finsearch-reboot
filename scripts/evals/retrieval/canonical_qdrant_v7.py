"""Read-only canonical binding for the surviving semantic-v2 Qdrant index."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evals.retrieval_benchmark_v2 import sha256


CONTRACT = Path("docs/evals/semantic_answer_v2_fresh_v7_qdrant_candidate.json")
PAGE_SIZE = 128


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def fingerprint_records(records):
    """Hash complete records independently of Qdrant pagination/order."""
    ordered = sorted(records, key=lambda record: str(record["id"]))
    digest = hashlib.sha256()
    for record in ordered:
        digest.update((_json(record) + "\n").encode())
    return digest.hexdigest()


def canonical_snapshot(client, collection):
    """Read every point and exact config, rejecting incomplete pagination."""
    try:
        info = client.get_collection(collection)
        config = info.config.model_dump(mode="json")
        reported_count = client.count(collection_name=collection, exact=True).count
    except Exception as exc:
        raise RuntimeError("Canonical Qdrant identity query failed") from exc

    records = []
    seen_ids = set()
    seen_offsets = set()
    offset = None
    while True:
        try:
            page, next_offset = client.scroll(
                collection_name=collection,
                offset=offset,
                limit=PAGE_SIZE,
                with_payload=True,
                with_vectors=True,
            )
        except Exception as exc:
            raise RuntimeError("Canonical Qdrant pagination failed") from exc
        if not page and next_offset is not None:
            raise ValueError("Canonical Qdrant pagination returned an empty nonterminal page")
        for point in page:
            record = point.model_dump(mode="json")
            identity = str(record.get("id"))
            if not identity or identity in seen_ids:
                raise ValueError("Canonical Qdrant snapshot has a missing or duplicate point ID")
            seen_ids.add(identity)
            records.append(record)
        if next_offset is None:
            break
        offset_key = _json(next_offset)
        if offset_key in seen_offsets:
            raise ValueError("Canonical Qdrant pagination repeated an offset")
        seen_offsets.add(offset_key)
        offset = next_offset

    if len(records) != reported_count:
        raise ValueError("Canonical Qdrant pagination did not return the exact reported count")
    return {
        "collection": collection,
        "points": len(records),
        "payload_vectors_sha256": fingerprint_records(records),
        "config": config,
    }, records


def load_contract(path=CONTRACT):
    contract = json.loads(Path(path).read_text())
    if contract.get("status") != "inactive_v7_candidate" or contract.get("authority") != "none":
        raise ValueError("Fresh-v7 Qdrant candidate contract is not inactive")
    return contract


def verify_snapshot(client, contract=None):
    """Return an exact matching snapshot or fail closed without Qdrant writes."""
    contract = contract or load_contract()
    corpus_path = Path(contract["corpus_path"])
    if sha256(corpus_path) != contract["corpus_sha256"]:
        raise ValueError("Fresh-v7 corpus hash differs from the candidate contract")
    snapshot, records = canonical_snapshot(client, contract["collection"])
    dense = snapshot["config"].get("params", {}).get("vectors", {}).get(
        contract["dense_vector_name"], {}
    )
    checks = {
        "collection": snapshot["collection"] == contract["collection"],
        "points": snapshot["points"] == contract["points"],
        "payload_vectors_sha256": (
            snapshot["payload_vectors_sha256"] == contract["payload_vectors_sha256"]
        ),
        "collection_config": snapshot["config"] == contract["collection_config"],
        "dense_vector_dimensions": dense.get("size") == contract["dense_vector_dimensions"],
    }
    if not all(checks.values()):
        raise ValueError(f"Fresh-v7 canonical Qdrant identity mismatch: {checks}")
    return snapshot, records


class FrozenIndexGuard:
    """Require the same exact canonical identity before and after an attempt."""

    def __init__(self, client, contract=None):
        self.client = client
        self.contract = contract or load_contract()
        self.before = None

    def verify_before(self):
        self.before, records = verify_snapshot(self.client, self.contract)
        return self.before, records

    def verify_after(self):
        if self.before is None:
            raise RuntimeError("Fresh-v7 before snapshot was not verified")
        after, records = verify_snapshot(self.client, self.contract)
        if after != self.before:
            raise ValueError("Fresh-v7 Qdrant identity changed during execution")
        return after, records
