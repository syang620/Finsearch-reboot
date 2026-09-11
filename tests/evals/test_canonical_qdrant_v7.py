from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.evals.retrieval import canonical_qdrant_v7 as canonical


class Dumpable:
    def __init__(self, value):
        self.value = value

    def model_dump(self, mode):
        assert mode == "json"
        return deepcopy(self.value)


class FakeClient:
    def __init__(self, records, config=None, pages=None, reported_count=None):
        self.records = records
        self.config = config or {
            "params": {"vectors": {"dense": {"size": 3}}},
        }
        self.pages = pages
        self.reported_count = len(records) if reported_count is None else reported_count

    def get_collection(self, collection):
        assert collection == "expected_collection"
        return SimpleNamespace(config=Dumpable(self.config))

    def count(self, collection_name, exact):
        assert collection_name == "expected_collection"
        assert exact is True
        return SimpleNamespace(count=self.reported_count)

    def scroll(self, collection_name, offset, limit, with_payload, with_vectors):
        assert collection_name == "expected_collection"
        assert limit == canonical.PAGE_SIZE
        assert with_payload is True
        assert with_vectors is True
        if self.pages is not None:
            page, next_offset = self.pages[offset]
            return [Dumpable(record) for record in page], next_offset
        if offset is None:
            return [Dumpable(record) for record in self.records], None
        raise AssertionError(f"unexpected offset: {offset}")


def record(identity, value):
    return {
        "id": identity,
        "payload": {"doc_id": f"doc-{identity}", "value": value},
        "vector": {"dense": [value, value + 0.5, value + 1.0]},
    }


def contract_for(client, monkeypatch, tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("{}\n")
    snapshot, _ = canonical.canonical_snapshot(client, "expected_collection")
    monkeypatch.setattr(canonical, "sha256", lambda path: "corpus-digest")
    return {
        "status": "inactive_v7_candidate",
        "authority": "none",
        "collection": "expected_collection",
        "corpus_path": str(corpus),
        "corpus_sha256": "corpus-digest",
        "points": snapshot["points"],
        "payload_vectors_sha256": snapshot["payload_vectors_sha256"],
        "dense_vector_name": "dense",
        "dense_vector_dimensions": 3,
        "collection_config": snapshot["config"],
    }


def test_fingerprint_is_independent_of_pagination_and_record_order():
    records = [record("a", 1.0), record("b", 2.0), record("c", 3.0)]
    first = FakeClient(records, pages={None: (records[:2], "next"), "next": (records[2:], None)})
    second_order = [records[2], records[0], records[1]]
    second = FakeClient(
        second_order,
        pages={None: (second_order[:1], "next"), "next": (second_order[1:], None)},
    )

    first_snapshot, _ = canonical.canonical_snapshot(first, "expected_collection")
    second_snapshot, _ = canonical.canonical_snapshot(second, "expected_collection")

    assert first_snapshot == second_snapshot


@pytest.mark.parametrize("field", ("payload", "vector"))
def test_same_count_changed_payload_or_vector_fails(monkeypatch, tmp_path, field):
    records = [record("a", 1.0), record("b", 2.0)]
    original = FakeClient(deepcopy(records))
    contract = contract_for(original, monkeypatch, tmp_path)
    changed_records = deepcopy(records)
    if field == "payload":
        changed_records[0]["payload"]["value"] = 99.0
    else:
        changed_records[0]["vector"]["dense"][0] = 99.0

    with pytest.raises(ValueError, match="canonical Qdrant identity mismatch"):
        canonical.verify_snapshot(FakeClient(changed_records), contract)


def test_collection_configuration_drift_fails(monkeypatch, tmp_path):
    records = [record("a", 1.0)]
    original = FakeClient(records)
    contract = contract_for(original, monkeypatch, tmp_path)
    changed_config = deepcopy(original.config)
    changed_config["params"]["vectors"]["dense"]["size"] = 4

    with pytest.raises(ValueError, match="canonical Qdrant identity mismatch"):
        canonical.verify_snapshot(FakeClient(records, config=changed_config), contract)


def test_incomplete_pagination_fails_closed():
    records = [record("a", 1.0), record("b", 2.0)]
    client = FakeClient(records, pages={None: ([records[0]], None)}, reported_count=2)

    with pytest.raises(ValueError, match="exact reported count"):
        canonical.canonical_snapshot(client, "expected_collection")


def test_empty_nonterminal_page_fails_closed():
    client = FakeClient([], pages={None: ([], "next")}, reported_count=1)

    with pytest.raises(ValueError, match="empty nonterminal page"):
        canonical.canonical_snapshot(client, "expected_collection")


def test_before_after_mutation_is_detected(monkeypatch, tmp_path):
    records = [record("a", 1.0), record("b", 2.0)]
    client = FakeClient(deepcopy(records))
    contract = contract_for(client, monkeypatch, tmp_path)
    guard = canonical.FrozenIndexGuard(client, contract)

    before, _ = guard.verify_before()
    assert before["points"] == 2
    client.records[1]["vector"]["dense"][1] = 100.0

    with pytest.raises(ValueError, match="canonical Qdrant identity mismatch"):
        guard.verify_after()


def test_service_or_pagination_exception_fails_closed():
    class BrokenClient(FakeClient):
        def scroll(self, **kwargs):
            raise OSError("service unavailable")

    with pytest.raises(RuntimeError, match="pagination failed"):
        canonical.canonical_snapshot(BrokenClient([record("a", 1.0)]), "expected_collection")
