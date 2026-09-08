from copy import deepcopy
import json
import math
import os
from pathlib import Path

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
