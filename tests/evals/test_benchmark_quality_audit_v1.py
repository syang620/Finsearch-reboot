"""Audit reproductions of historical limitations, NOT desired v2 scorer rules."""
import json
from pathlib import Path

from scripts.evals.audit_benchmark_quality_v1 import (
    RETRIEVAL, audit, read, reference_metrics, retrieval_arithmetic_check,
)
from evals.retrieval_benchmark_v2 import metrics


def test_frozen_audit_observations_reproduce():
    recorded = json.loads(Path("artifacts/evals/benchmark_quality_audit/v1/observations.json").read_text())
    assert audit() == recorded


def test_independent_retrieval_arithmetic_oracle():
    assert retrieval_arithmetic_check()["mismatches"] == 0


def test_alternative_ids_and_necessary_facets_are_different_denominators():
    judgments = [{"evidence_id": i, "grade": 2, "groups": [g]}
                 for i, g in [("a", "one"), ("a_copy", "one"), ("b", "two")]]
    case = {"status": "answerable", "judgments": judgments, "required_groups": ["one", "two"]}
    actual = metrics(["a", "b"], case)
    assert actual == reference_metrics(["a", "b"], judgments, case["required_groups"])
    assert actual["recall@10"] == 2 / 3
    assert actual["evidence_group_recall@10"] == 1


def test_documented_missing_microsoft_alternatives_remain_historical():
    cases = {c["id"]: c for c in read(RETRIEVAL / "queries.jsonl")}
    docs = {d["id"]: d for d in read(RETRIEVAL / "corpus.jsonl")}
    for year, ids, revenue, income in [
        (2024, [8, 72], "245122", "109433"),
        (2025, [8, 65], "281724", "128528"),
    ]:
        case = cases[f"KBV2_MSFT_{year}_01"]
        for number in ids:
            identifier = f"MSFT_10-K_{year}::table::{number}"
            content = docs[identifier]["content"]
            assert all(token in content for token in ["Revenue", "Operating Income", revenue, income])
            assert identifier not in {j["evidence_id"] for j in case["judgments"]}
            assert metrics([identifier], case)["mrr@10"] == 0


def test_historical_numeric_false_positive_characterizations():
    probes = audit()["semantic"]["numeric_counterexamples"]
    assert set(probes) == {"correct_control", "negation", "wrong_issuer", "wrong_currency",
                           "wrong_quantity_role", "rejected_answer"}
    # Preserves evidence of the old scorer defect. A new scorer must reject the
    # five non-control examples; it must not change this historical module.
    assert all(p["rate"] == 1 for p in probes.values())
