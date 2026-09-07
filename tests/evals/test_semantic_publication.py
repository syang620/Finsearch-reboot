import hashlib
import json

import pytest

from scripts.evals.publish_semantic_evidence import publish


def fixture(source):
    source.mkdir()
    (source / "judge").mkdir()
    parent = "/" + "Users" + "/build/.ollama/models/blobs/sha256-unchanged"
    (source / "started.json").write_text(json.dumps({"model": {"digest": "unchanged", "details": {"parent_model": parent}}}))
    (source / "raw_answers.jsonl").write_text('{"answer":"391035","value":391035}\n')
    (source / "source_audit.jsonl").write_text('{"claim":"fully_supported"}\n')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}
    (source / "answer_manifest.json").write_text(json.dumps({"files_sha256": hashes}))
    (source / "judge/manifest.json").write_text('{"files_sha256":{}}')


def test_publication_preserves_results_and_originals(tmp_path):
    source, destination = tmp_path / "original", tmp_path / "public"
    fixture(source)
    before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    lineage = publish(source, destination)
    assert before == {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    for name in ("raw_answers.jsonl", "source_audit.jsonl"):
        assert (destination / name).read_bytes() == before[name]
    model = json.loads((destination / "started.json").read_text())["model"]
    assert model["digest"] == "unchanged"
    assert model["details"]["parent_model"] == "<MODEL_BUILD_HOME>/.ollama/models/blobs/sha256-unchanged"
    manifest = json.loads((destination / "answer_manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == expected
    assert set(lineage["changes"]) == {"started.json", "answer_manifest.json"}
    with pytest.raises(ValueError, match="exists"):
        publish(source, destination)


def test_publication_rejects_unexpected_answer_paths(tmp_path):
    source, destination = tmp_path / "original", tmp_path / "public"
    fixture(source)
    (source / "raw_answers.jsonl").write_text(json.dumps({"answer": "/" + "Users/person/private"}) + "\n")
    with pytest.raises(ValueError, match="Unexpected local path"):
        publish(source, destination)
    assert not destination.exists()


def test_publication_rejects_corrupt_original_manifest(tmp_path):
    source, destination = tmp_path / "original", tmp_path / "public"
    fixture(source)
    (source / "raw_answers.jsonl").write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="hash mismatch"):
        publish(source, destination)
    assert not destination.exists()
