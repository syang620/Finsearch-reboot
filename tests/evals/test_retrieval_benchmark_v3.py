from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import pytest

from evals.retrieval_benchmark_v3 import (
    classify_results, correlation_metadata, grouped_summary, load_dataset, metrics,
    normalized_question, summarize, validate, validate_pairs, verify_history,
    verify_source_locations,
)
from scripts.evals.audit_benchmark_quality_v1 import retrieval_arithmetic_check
from scripts.evals.retrieval.adjudications_v3 import decisions
from scripts.evals.retrieval.build_benchmark_v3 import new_judgment
from scripts.evals.retrieval.inspect_sources_v3 import normalize

DATA = Path(os.getenv("V3_TEST_DATASET", "data/evals/retrieval/benchmark_v3"))


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DATA)


def test_deterministic_versioned_loading(dataset):
    assert dataset == load_dataset(DATA)
    cases, docs, manifest = dataset
    assert manifest["version"] == 3
    assert len(cases) == 126 and len(docs) == 948
    assert len({c["parent_case_id"] for c in cases}) == 126
    assert all(c["id"].startswith("KBV3_") for c in cases)


@pytest.mark.parametrize("year,table", [(2024, 8), (2024, 72), (2025, 8), (2025, 65)])
def test_four_missed_microsoft_tables_receive_full_relevance(dataset, year, table):
    cases, _, _ = dataset
    case = next(c for c in cases if c["id"] == f"KBV3_MSFT_{year}_01")
    identifier = f"MSFT_10-K_{year}::table::{table}"
    judgment = next(j for j in case["judgments"] if j["evidence_id"] == identifier)
    assert judgment["grade"] == 2
    assert judgment["reason_code"] == "B1_MISSED_EQUIVALENT_TOTALS"
    assert len(judgment["adjudicated_cells"]) == 2
    result = metrics([identifier], case)
    assert result["mrr@10"] == result["evidence_group_recall@10"] == 1
    assert result["recall@10"] == .25  # One of four equivalent total tables.


@pytest.mark.parametrize("year", [2023, 2024])
def test_policy_alternative_does_not_erase_other_required_facets(dataset, year):
    cases, _, _ = dataset
    multi = next(c for c in cases if c["id"] == f"KBV3_AMZN_{year}_20")
    alternative = next(j for j in multi["judgments"] if j["reason_code"] == "B1_EQUIVALENT_SUBSCRIPTION_POLICY")
    assert metrics([alternative["evidence_id"]], multi)["evidence_group_recall@10"] == .5
    benefits = next(j for j in multi["judgments"] if "prime" in j["groups"])
    assert metrics([alternative["evidence_id"], benefits["evidence_id"]], multi)["evidence_group_recall@10"] == 1
    general = next(c for c in cases if c["id"] == f"KBV3_AMZN_{year}_05")
    result = metrics([alternative["evidence_id"]], general)
    assert result["mrr@10"] == result["evidence_group_recall@10"] == 0
    assert result["ndcg@10"] > 0


def test_numeric_context_is_not_a_management_explanation(dataset):
    case = next(c for c in dataset[0] if c["id"] == "KBV3_MSFT_2024_11")
    result = metrics(["MSFT_10-K_2024::table::8"], case)
    assert result["recall@10"] == 0 and result["ndcg@10"] > 0


def test_no_implicit_case_or_whitespace_relevance_normalization(dataset):
    decision = next(d for d in decisions() if "quote" in d)
    document = next(d for d in dataset[1] if d["id"] == decision["evidence_id"])
    assert new_judgment(decision, document)["grade"] == 2
    for replacement in (decision["quote"].lower(), decision["quote"].replace(" ", "  ")):
        altered = {**decision, "quote": replacement}
        with pytest.raises(ValueError):
            new_judgment(altered, document)
    assert normalize("A\u00a0 B\nC") == "ABC"  # Source-location typography only.
    assert normalize("Operating Income") != normalize("Operating income")
    assert normalized_question("FY2024  policy") == "FYYEAR policy"


def test_group_macro_prevents_duplicate_family_weighting():
    cases = [{"id": i, "family_id": family} for i, family in [("a", "same"), ("b", "same"), ("c", "other")]]
    from evals.retrieval_benchmark_v2 import METRICS
    rows = [{"id": i, "metrics": {m: value for m in METRICS}} for i, value in [("a", 1), ("b", 1), ("c", 0)]]
    result = grouped_summary(rows, cases, "family_id")
    assert result["groups"] == 2
    assert result["macro_metrics"]["recall@10"] == .5  # Query mean would be 2/3.


def test_transitive_shared_evidence_and_year_family_components():
    cases = [
        {"id": "a", "status": "answerable", "family_id": "one", "judgments": [{"evidence_id": "x", "grade": 2}]},
        {"id": "b", "status": "answerable", "family_id": "two", "judgments": [{"evidence_id": "x", "grade": 2}]},
        {"id": "c", "status": "answerable", "family_id": "two", "judgments": [{"evidence_id": "y", "grade": 2}]},
    ]
    assert len(set(correlation_metadata(cases).values())) == 1
    assert correlation_metadata(cases) == correlation_metadata(list(reversed(cases)))


def test_unknown_is_not_negative_or_missing(dataset):
    cases, docs, _ = dataset
    case = next(c for c in cases if c["stratum"] == "hard_negative")
    judged = {j["evidence_id"] for j in case["judgments"]}
    negative = next(j["evidence_id"] for j in case["judgments"] if j["grade"] == 0)
    unknown = next(d["id"] for d in docs if d["id"] not in judged and all(d["metadata"][k] == case[k] for k in ("ticker", "fiscal_year", "form_type")))
    result = classify_results([negative, unknown, "absent", unknown], case, docs)
    assert result == {"explicit_irrelevant_returned_ids": 1, "unjudged_returned_ids": 2,
                      "missing_corpus_ids": 1, "duplicate_returned_ids": 1, "incompatible_filter_ids": 0}


def test_duplicate_rank_occupancy_and_graded_oracle():
    assert retrieval_arithmetic_check()["metric_values_checked"] == 6000
    case = {"status": "answerable", "required_groups": ["x", "y"], "judgments": [
        {"evidence_id": "a", "grade": 2, "groups": ["x"]},
        {"evidence_id": "b", "grade": 2, "groups": ["y"]},
        {"evidence_id": "p", "grade": 1, "groups": []}]}
    assert metrics(["a"] * 10 + ["b"], case)["recall@10"] == .5
    assert metrics(["p", "a", "b"], case)["mrr@10"] == .5
    assert metrics(["p"], case)["ndcg@5"] == pytest.approx(1 / (3 + 3/math.log2(3) + 1/math.log2(4)))


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unexpected"])
def test_full_pair_coverage_fail_closed(dataset, mutation):
    cases, _, _ = dataset
    modes = ["bm25_only", "dense_only", "hybrid", "hybrid_reranker"]
    rows = [{"id": c["id"], "mode": m} for c in cases if c["status"] == "answerable" for m in modes]
    assert validate_pairs(rows, cases, modes)["completed_pairs"] == 480
    if mutation == "missing": rows.pop()
    elif mutation == "duplicate": rows.append(rows[0])
    else: rows[0]["mode"] = "unknown"
    with pytest.raises(ValueError): validate_pairs(rows, cases, modes)


def test_partial_comparison_is_explicit_not_complete(dataset):
    result = summarize([], dataset[0], ["bm25_only"], complete=False)
    assert not result["complete"] and result["expected_pairs"] == 120
    assert result["modes"]["bm25_only"]["overall"]["metrics"]["recall@10"] is None


def test_history_and_exact_label_lineage(dataset):
    assert verify_history(DATA)["historical_files_verified"] > 100
    lineage = json.loads((DATA / "lineage.json").read_text())
    assert lineage["membership_additions"] == lineage["membership_removals"] == []
    assert lineage["grade_changes_on_previously_judged_ids"] == 0
    changes = [json.loads(l) for l in (DATA / "label_changes.jsonl").read_text().splitlines()]
    assert sum(c["change_kind"] == "label_added" for c in changes) == 12
    for change in changes:
        if change["before"] is not None:
            assert all(change["before"][k] == change["after"][k] for k in ("grade", "groups", "spans"))


@pytest.mark.parametrize("mutation", ["duplicate", "missing_id", "wrong_group", "wrong_source_quote", "wrong_section"])
def test_adversarial_dataset_and_source_validation(dataset, mutation):
    cases, docs, _ = deepcopy(dataset)
    if mutation == "duplicate": cases.append(cases[0])
    elif mutation == "missing_id": cases[0]["judgments"][0]["evidence_id"] = "absent"
    elif mutation == "wrong_group": cases[0]["judgments"][0]["groups"] = ["invented"]
    elif mutation == "wrong_source_quote": cases[0]["judgments"][0]["source_locations"][0]["quote_normalized"] = "invented"
    else: cases[0]["judgments"][0]["source_locations"][0]["item"] = "1C"
    with pytest.raises(ValueError):
        validate(cases, docs)
        verify_source_locations(cases, json.loads((DATA / "source_sections.json").read_text()))


def test_frozen_hash_and_lineage_tamper_rejected(tmp_path):
    (tmp_path / "dataset_manifest.json").write_text(json.dumps({"benchmark_id": "sec_retrieval_benchmark_v3", "version": 3, "files": {"queries.jsonl": "bad"}}))
    (tmp_path / "queries.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="hash mismatch"): load_dataset(tmp_path)


def test_original_source_hash_is_checked(dataset):
    sections = json.loads((DATA / "source_sections.json").read_text())
    sections[0]["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Source provenance hash"):
        verify_source_locations(dataset[0], sections)


@pytest.mark.parametrize("mutation", ["absent", "unapproved", "wrong_hash", "wrong_commit", "wrong_review_url", "changed_code"])
def test_no_comparison_before_matching_benchmark_review(dataset, tmp_path, monkeypatch, mutation):
    from scripts.evals.retrieval import run_benchmark_v3 as runner
    from evals.retrieval_benchmark_v2 import sha256
    approval = {"status": "approved_for_narrow_known_label_baseline", "reviewed_commit": "a" * 40,
                "dataset_manifest_sha256": sha256(DATA / "dataset_manifest.json"),
                "review_url": "https://github.com/syang620/Finsearch-reboot/pull/30#issuecomment-test"}
    monkeypatch.setattr(runner, "git", lambda *args: "")
    monkeypatch.setattr(runner, "committed_approval", lambda p: json.loads(p.read_text()))
    monkeypatch.setattr(runner, "verify_remote_review", lambda a: pytest.fail("Invalid local binding reached GitHub"))
    path = tmp_path / "approval.json"
    if mutation == "absent":
        with pytest.raises(FileNotFoundError): runner.verify_approval(path, DATA)
        return
    if mutation == "unapproved": approval["status"] = "pending"
    elif mutation == "wrong_hash": approval["dataset_manifest_sha256"] = "wrong"
    elif mutation == "wrong_commit": approval["reviewed_commit"] = "HEAD"
    elif mutation == "wrong_review_url": approval["review_url"] = "unverified"
    else: monkeypatch.setattr(runner, "git", lambda *args: "changed" if args[0] == "diff" else "")
    path.write_text(json.dumps(approval))
    with pytest.raises(ValueError): runner.verify_approval(path, DATA)


@pytest.fixture
def approval_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    subprocess.run(["git", "config", "user.name", "Benchmark Test"], check=True)
    subprocess.run(["git", "config", "user.email", "benchmark@example.invalid"], check=True)
    path = tmp_path / "approval.json"
    path.write_text(json.dumps({"status": "approved_for_narrow_known_label_baseline",
                                "dataset_manifest_sha256": "d" * 64, "reviewed_commit": "a" * 40}))
    return path


def test_approval_must_be_committed_and_byte_identical(approval_repo):
    from scripts.evals.retrieval.run_benchmark_v3 import committed_approval
    with pytest.raises(ValueError, match="not committed"):
        committed_approval(approval_repo)
    subprocess.run(["git", "add", "approval.json"], check=True)
    with pytest.raises(ValueError, match="not committed"):
        committed_approval(approval_repo)
    subprocess.run(["git", "commit", "-qm", "Freeze test approval"], check=True)
    assert committed_approval(approval_repo) == json.loads(approval_repo.read_text())
    approval_repo.write_text(approval_repo.read_text() + "\n")
    with pytest.raises(ValueError, match="uncommitted changes"):
        committed_approval(approval_repo)


def test_approval_outside_repo_rejected(approval_repo, monkeypatch):
    from scripts.evals.retrieval import run_benchmark_v3 as runner
    monkeypatch.setattr(runner, "git", lambda *a: str(approval_repo.parent / "nested"))
    with pytest.raises(ValueError, match="inside this repository"):
        runner.committed_approval(approval_repo)


def test_offline_approval_is_bound_to_evaluated_commit(approval_repo):
    from scripts.evals.retrieval.verify_benchmark_v3 import verify_committed_approval
    subprocess.run(["git", "add", "approval.json"], check=True)
    subprocess.run(["git", "commit", "-qm", "Freeze test approval"], check=True)
    approval = json.loads(approval_repo.read_text())
    manifest = {"implementation_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "annotation_approval": approval, "annotation_approval_path": "approval.json",
                "annotation_approval_sha256": hashlib.sha256(approval_repo.read_bytes()).hexdigest(),
                "dataset_manifest_sha256": "d" * 64, "dataset_freeze_sha": "a" * 40}
    verify_committed_approval(manifest)
    for field in ("annotation_approval_sha256", "dataset_freeze_sha", "dataset_manifest_sha256"):
        altered = {**manifest, field: "bad"}
        with pytest.raises(ValueError): verify_committed_approval(altered)
    with pytest.raises(ValueError):
        verify_committed_approval({**manifest, "annotation_approval": {**approval, "status": "pending"}})
    with pytest.raises(ValueError):
        verify_committed_approval({**manifest, "annotation_approval_path": "../approval.json"})


@pytest.fixture
def remote_review(monkeypatch):
    from scripts.evals.retrieval import run_benchmark_v3 as runner
    prefix = f"repos/{runner.REVIEW_REPOSITORY}"
    body = "Codex Review: Didn't find any major issues.\n\n**Reviewed commit:** `aaaaaaaaaa`"
    approval = {"pull_request": 30, "review_comment_id": 123, "reviewed_commit": "a" * 40,
                "review_url": f"https://github.com/{runner.REVIEW_REPOSITORY}/pull/30#issuecomment-123",
                "review_body_sha256": hashlib.sha256(body.encode()).hexdigest()}
    responses = {
        f"{prefix}/issues/comments/123": {"id": 123, "html_url": approval["review_url"],
            "issue_url": f"https://api.github.com/{prefix}/issues/30", "body": body,
            "user": {"login": runner.REVIEW_AUTHOR, "type": "Bot"}},
        f"{prefix}/pulls/30": {"state": "open", "base": {"repo": {"full_name": runner.REVIEW_REPOSITORY}}},
        f"{prefix}/pulls/30/reviews": [], f"{prefix}/pulls/30/comments": [],
    }
    def api(endpoint, paginate=False):
        assert paginate == (endpoint.endswith("/reviews") or endpoint.endswith("/comments"))
        return responses[endpoint]
    monkeypatch.setattr(runner, "github_json", api)
    return approval, responses, prefix


def test_verified_clean_review_accepts_old_sha_findings(remote_review):
    from scripts.evals.retrieval import run_benchmark_v3 as runner
    approval, responses, prefix = remote_review
    responses[f"{prefix}/pulls/30/comments"] = [{"user": {"login": runner.REVIEW_AUTHOR}, "original_commit_id": "b" * 40}]
    runner.verify_remote_review(approval)


@pytest.mark.parametrize("mutation", ["wrong_id", "invented_url", "wrong_pr", "wrong_author", "not_bot",
    "body_changed", "wrong_sha", "findings_body", "wrong_repo", "closed_pr", "changes_requested", "inline", "inline_fallback", "missing_id"])
def test_remote_approval_adversarial_bindings(remote_review, mutation):
    from scripts.evals.retrieval import run_benchmark_v3 as runner
    approval, responses, prefix = remote_review
    comment = responses[f"{prefix}/issues/comments/123"]
    if mutation == "wrong_id": comment["id"] = 999
    elif mutation == "invented_url": approval["review_url"] += "fabricated"
    elif mutation == "wrong_pr": comment["issue_url"] = comment["issue_url"].replace("/30", "/29")
    elif mutation == "wrong_author": comment["user"]["login"] = "untrusted"
    elif mutation == "not_bot": comment["user"]["type"] = "User"
    elif mutation == "body_changed": comment["body"] += " altered"
    elif mutation in ("wrong_sha", "findings_body"):
        comment["body"] = comment["body"].replace("aaaaaaaaaa", "bbbbbbbbbb") if mutation == "wrong_sha" else "Found issues. **Reviewed commit:** `aaaaaaaaaa`"
        approval["review_body_sha256"] = hashlib.sha256(comment["body"].encode()).hexdigest()
    elif mutation == "wrong_repo": responses[f"{prefix}/pulls/30"]["base"]["repo"]["full_name"] = "other/repo"
    elif mutation == "closed_pr": responses[f"{prefix}/pulls/30"]["state"] = "closed"
    elif mutation == "changes_requested": responses[f"{prefix}/pulls/30/reviews"] = [{"commit_id": "a" * 40, "state": "CHANGES_REQUESTED"}]
    elif mutation in ("inline", "inline_fallback"):
        key = "original_commit_id" if mutation == "inline" else "commit_id"
        responses[f"{prefix}/pulls/30/comments"] = [{"user": {"login": runner.REVIEW_AUTHOR}, key: "a" * 40}]
    else: approval["review_comment_id"] = True
    with pytest.raises(ValueError): runner.verify_remote_review(approval)


@pytest.mark.parametrize("failure", [FileNotFoundError(), subprocess.CalledProcessError(1, "gh"), "invalid-json"])
def test_github_unavailable_fails_closed(monkeypatch, failure):
    from scripts.evals.retrieval import run_benchmark_v3 as runner
    def output(*a, **kw):
        if isinstance(failure, Exception): raise failure
        return failure
    monkeypatch.setattr(runner.subprocess, "check_output", output)
    with pytest.raises(ValueError, match="comparison remains blocked"):
        runner.github_json("repos/example/missing")


def test_github_findings_all_pages_are_checked(monkeypatch):
    from scripts.evals.retrieval import run_benchmark_v3 as runner
    # Exercise the real pagination adapter: a finding on the second page is retained.
    finding = {"user": {"login": runner.REVIEW_AUTHOR}, "original_commit_id": "a" * 40}
    def output(command, **kw):
        assert command[-2:] == ["--paginate", "--slurp"]
        return json.dumps([[], [finding]])
    monkeypatch.setattr(runner.subprocess, "check_output", output)
    assert runner.github_json("repos/example/pulls/30/comments", paginate=True) == [finding]


def test_original_build_and_archived_vector_reference():
    from scripts.evals.retrieval.index_provenance_v3 import frozen_index_reference, verify_frozen_index
    reference = frozen_index_reference()
    assert reference["snapshot"]["points"] == 948
    assert verify_frozen_index(reference["build"], reference["snapshot"], reference["build"]["embedded_sha256"]) == reference


@pytest.mark.parametrize("mutation", ["manifest", "embedding_cache", "vector_digest", "config", "point_count"])
def test_unchanged_but_wrong_index_cannot_pass(mutation):
    from scripts.evals.retrieval.index_provenance_v3 import frozen_index_reference, verify_frozen_index
    reference = frozen_index_reference()
    index, snapshot = deepcopy(reference["build"]), deepcopy(reference["snapshot"])
    digest = index["embedded_sha256"]
    if mutation == "manifest": index["embedding_model"] = "other-model"
    elif mutation == "embedding_cache": digest = "0" * 64
    elif mutation == "vector_digest": snapshot["payload_vectors_sha256"] = "0" * 64
    elif mutation == "config": snapshot["config"]["params"]["sparse_vectors"]["bm25"]["modifier"] = None
    else: snapshot["points"] -= 1
    # Even if this changed snapshot equals itself before and after a run,
    # it must not pass the historical origin binding.
    with pytest.raises(ValueError): verify_frozen_index(index, snapshot, digest)


def test_archived_reference_tampering_rejected(monkeypatch):
    from scripts.evals.retrieval import index_provenance_v3 as provenance
    monkeypatch.setattr(provenance, "REFERENCE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="provenance hash mismatch"):
        provenance.frozen_index_reference()


@pytest.mark.parametrize("mutation", ["dense", "sparse_index", "sparse_weight", "payload", "point_id"])
def test_live_snapshot_digest_covers_dense_and_sparse_vectors(mutation):
    from types import SimpleNamespace
    from scripts.evals.retrieval.run_benchmark_v3 import snapshot
    record = {"id": "point", "payload": {"doc_id": "doc", "content": "source"},
              "vector": {"dense": [1.0, 0.0], "bm25": {"indices": [1, 2], "values": [.4, .6]}}}
    client = SimpleNamespace(
        scroll=lambda **kw: ([SimpleNamespace(model_dump=lambda **kw: deepcopy(record))], None),
        get_collection=lambda name: SimpleNamespace(config=SimpleNamespace(model_dump=lambda **kw: {})))
    before, _ = snapshot(client, "collection")
    if mutation == "dense": record["vector"]["dense"][0] = .5
    elif mutation == "sparse_index": record["vector"]["bm25"]["indices"][0] = 3
    elif mutation == "sparse_weight": record["vector"]["bm25"]["values"][0] = .5
    elif mutation == "payload": record["payload"]["content"] = "changed"
    else: record["id"] = "different-point"
    after, _ = snapshot(client, "collection")
    assert before["payload_vectors_sha256"] != after["payload_vectors_sha256"]
