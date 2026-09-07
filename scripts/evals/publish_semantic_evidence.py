"""Publish a path-redacted copy; preserve local original bytes and hash lineage.

This utility does not run models, load gold, or calculate scores. Only an upstream
model's parent_model home prefix and dependent artifact checksums may change.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def redacted(value, pointer=""):
    changes = []
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            child = pointer + "/" + str(key).replace("~", "~0").replace("/", "~1")
            if key == "parent_model" and isinstance(item, str):
                updated = re.sub(r"^/Users/[^/]+/", "<MODEL_BUILD_HOME>/", item)
                if updated != item:
                    changes.append(child)
                    item = updated
            result[key], nested = redacted(item, child)
            changes.extend(nested)
        return result, changes
    if isinstance(value, list):
        result = []
        for index, item in enumerate(value):
            updated, nested = redacted(item, pointer + "/" + str(index))
            result.append(updated)
            changes.extend(nested)
        return result, changes
    return value, changes


def publish(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Original and publication must be separate non-nested directories")
    if destination.exists():
        raise ValueError("Publication already exists; refuse overwrite")
    files = sorted(p for p in source.rglob("*") if p.is_file())
    if not files or any(p.is_symlink() or p.suffix not in {".json", ".jsonl"} for p in files):
        raise ValueError("Expected original JSON evidence files only")
    original_hashes = {str(p.relative_to(source)): digest(p) for p in files}
    pending, changes = {}, {}
    for path in files:
        relative = str(path.relative_to(source))
        raw = path.read_bytes()
        if path.suffix == ".jsonl":
            # Answers/judgments are byte-preserved. Unexpected paths stop export.
            if re.search(rb"/Users/[^/\s\"\\]+/", raw):
                raise ValueError("Unexpected local path in JSONL evidence")
            pending[relative] = raw
        else:
            original = json.loads(raw)
            updated, pointers = redacted(original)
            candidate = (json.dumps(updated, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
            if re.search(rb"/Users/[^/\s\"\\]+/", candidate):
                raise ValueError("Unrecognized local path outside model parent metadata")
            pending[relative] = candidate if pointers else raw
            if pointers:
                changes[relative] = {"metadata_json_pointers": pointers}
    # Rebind only existing artifact checksums after the allowed metadata change.
    for name in ("answer_manifest.json", "judge/manifest.json"):
        manifest = json.loads(pending[name])
        rebound = []
        for child, old_digest in manifest["files_sha256"].items():
            relative = str(Path(name).parent / child)
            if original_hashes.get(relative) != old_digest:
                raise ValueError("Original evidence hash mismatch")
            new_digest = hashlib.sha256(pending[relative]).hexdigest()
            if new_digest != old_digest:
                manifest["files_sha256"][child] = new_digest
                rebound.append(child)
        if rebound:
            pending[name] = (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
            changes.setdefault(name, {})["rebound_artifact_hashes"] = rebound
    publication_hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in pending.items()}
    lineage = {
        "policy": "Publication-only redaction of upstream parent_model home prefixes; dependent file hashes rebound. Original bytes retained locally. No answers, judgments, numeric values, scores, model digests, or source hashes changed.",
        "original_files_sha256": original_hashes,
        "publication_files_sha256": publication_hashes,
        "changes": changes,
    }
    # Validate everything before creating the immutable destination.
    destination.mkdir(parents=True, exist_ok=False)
    for name, raw in pending.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(raw)
    with (destination / "publication_lineage.json").open("x") as handle:
        json.dump(lineage, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return lineage


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    result = publish(args.source, args.destination)
    print(json.dumps({"files": len(result["publication_files_sha256"]), "changed_metadata_files": list(result["changes"])}, indent=2))
