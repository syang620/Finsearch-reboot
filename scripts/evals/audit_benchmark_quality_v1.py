"""Read-only quality audit. Prints observations; never relabels or runs models.

Run from repository root with PYTHONPATH=src. Counterexamples characterize the
historical evaluator, not desirable behavior for a future evaluator version.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import re

from evals.retrieval_benchmark_v2 import load_dataset as load_retrieval, metrics
from evals.semantic_answer_v1 import load_dataset as load_semantic, deterministic_case

RETRIEVAL = Path("data/evals/retrieval/benchmark_v2")
SEMANTIC = Path("data/evals/semantic_answer/v1")
RBASE = Path("artifacts/evals/retrieval/benchmark_v2/baselines/54d31917c27263ea25ebf5d905caa0a4ee34f74f")
SBASE = Path("artifacts/evals/semantic_answer/v1/baselines/3793929f343fb9fc93c4d5a83cd0b2f16fe6e6cb")


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def counts(values):
    return dict(sorted(Counter(values).items()))


def composition(cases, query_key):
    return {
        "cases": len(cases),
        "issuers": counts(c["ticker"] for c in cases),
        "filings": counts(f'{c["ticker"]}_{c["fiscal_year"]}' for c in cases),
        "strata": counts(c["stratum"] for c in cases),
        "exact_duplicate_text_excess": len(cases) - len({c[query_key] for c in cases}),
        "year_normalized_unique_texts": len({re.sub(r"20\d{2}", "YEAR", c[query_key]) for c in cases}),
    }


def reference_metrics(predictions, judgments, groups):
    """Independent direct summation of the documented known-label convention."""
    grades = {j["evidence_id"]: j["grade"] for j in judgments}
    positives = [i for i, grade in grades.items() if grade == 2]
    first_rank = {}
    for rank, identifier in enumerate(predictions[:10], 1):
        first_rank.setdefault(identifier, rank)
    result = {}
    for k in (5, 10):
        result[f"recall@{k}"] = sum(first_rank.get(i, 11) <= k for i in positives) / len(positives)
        numerator = sum((2 ** grade - 1) / math.log2(first_rank[i] + 1)
                        for i, grade in grades.items() if first_rank.get(i, 11) <= k)
        denominator = sum((2 ** grade - 1) / math.log2(rank + 1)
                          for rank, grade in enumerate(sorted(grades.values(), reverse=True)[:k], 1))
        result[f"ndcg@{k}"] = numerator / denominator
    ranks = [first_rank[i] for i in positives if i in first_rank]
    result["mrr@10"] = 1 / min(ranks) if ranks else 0.0
    result["evidence_group_recall@10"] = sum(
        any(j["grade"] == 2 and group in j["groups"] and j["evidence_id"] in first_rank
            for j in judgments) for group in groups) / len(groups)
    return result


def retrieval_arithmetic_check():
    rng = random.Random(620)
    for _ in range(1000):
        judgments = [{"evidence_id": str(i), "grade": 2 if i == 0 else rng.randrange(3),
                      "groups": [str(i % 2)]} for i in range(8)]
        groups = sorted({g for j in judgments if j["grade"] == 2 for g in j["groups"]})
        predictions = [str(rng.randrange(12)) for _ in range(rng.randrange(16))]
        case = {"status": "answerable", "judgments": judgments, "required_groups": groups}
        actual = metrics(predictions, case)
        reference = reference_metrics(predictions, judgments, groups)
        assert all(math.isclose(actual[k], reference[k], abs_tol=1e-12) for k in reference)
    return {"randomized_rankings": 1000, "metric_values_checked": 6000, "mismatches": 0}


def semantic_counterexamples(case):
    gold = case["required_claims"][0]
    source = gold["sources"][0]
    fact = {k: source[k] for k in ("metric_id", "ticker", "unit", "value", "start_date", "report_date", "form_type")}
    fact.update(fiscal_year=source["fact_fiscal_year"], status="ok")
    value = source["value"] / 1000000
    output = {"status": "completed", "analyst": {
        "ok": True, "status": "ok", "trace": {"analyst_visible_context_ids": ["ctx1"]},
        "claims": [{"claim_id": "c1", "claim_type": "structured_numeric", "metric_id": "revenue",
                    "text": f"Apple revenue was {value} million USD.", "context_ids": ["ctx1"]}]},
        "evaluation_trace": {"analyst_packet": {"context_items": [
            {"context_id": "ctx1", "kind": "structured_fact", "structured_fact": fact}]}}}
    probes = {
        "correct_control": f"Apple revenue was {value} million USD.",
        "negation": f"Apple revenue was not {value} million USD; it was 1 million USD.",
        "wrong_issuer": f"Microsoft revenue was {value} million USD.",
        "wrong_currency": f"Apple revenue was {value} million EUR.",
        "wrong_quantity_role": f"Apple operating income was {value} million USD; revenue was 1 million USD.",
    }
    results = {}
    for name, text in probes.items():
        candidate = deepcopy(output)
        candidate["analyst"]["claims"][0]["text"] = text
        results[name] = deterministic_case(case, candidate)["numeric_consistency"]
    rejected = deepcopy(output)
    rejected["analyst"].update(ok=False, status="grounding_error")
    rejected["status"] = "failed"
    results["rejected_answer"] = deterministic_case(case, rejected)["numeric_consistency"]
    return results


def audit():
    retrieval, documents, validation = load_retrieval(RETRIEVAL)
    semantic, _ = load_semantic(SEMANTIC)
    scored = [c for c in retrieval if c["status"] == "answerable"]
    docs = {d["id"]: d for d in documents}
    positives = [j for c in scored for j in c["judgments"] if j["grade"] == 2]
    positive_sets = defaultdict(list)
    for c in scored:
        positive_sets[tuple(j["evidence_id"] for j in c["judgments"] if j["grade"] == 2)].append(c["id"])
    gold = [g for c in semantic for g in c["required_claims"]]
    kb_ids = {s["evidence_id"] for g in gold for s in g["sources"] if s["kind"] == "kb"}
    positive_ids = {j["evidence_id"] for j in positives}
    ranked = read(RBASE / "per_query.jsonl")
    unjudged = {}
    for mode in sorted({r["mode"] for r in ranked}):
        rows = [r for r in ranked if r["mode"] == mode]
        labels = {c["id"]: {j["evidence_id"] for j in c["judgments"]} for c in scored}
        unjudged[mode] = {"unjudged": sum(i not in labels[r["id"]] for r in rows for i in r["ranked_ids"]),
                          "returned": sum(len(r["ranked_ids"]) for r in rows)}
    files = [RETRIEVAL / "queries.jsonl", RETRIEVAL / "dataset_manifest.json", RETRIEVAL / "corpus.jsonl",
             SEMANTIC / "queries.jsonl", SEMANTIC / "manifest.json", SEMANTIC / "judge_rubric.txt",
             Path("src/evals/retrieval_benchmark_v2.py"), Path("src/evals/semantic_answer_v1.py"),
             RBASE / "per_query.jsonl", SBASE / "raw_answers.jsonl", SBASE / "judge/judgments.jsonl",
             SBASE / "source_audit.jsonl", Path(__file__).relative_to(Path.cwd()) if Path(__file__).is_absolute() else Path(__file__)]
    return {
        "audit_version": 1, "optimization_freeze_approved": False,
        "inputs_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "retrieval": {**composition(scored, "query"), "excluded": validation["excluded"],
            "corpus_documents": len(documents), "corpus_types": counts(d["metadata"]["doc_type"] for d in documents),
            "judgment_grades": counts(str(j["grade"]) for c in scored for j in c["judgments"]),
            "positive_document_types": counts(docs[j["evidence_id"]]["metadata"]["doc_type"] for j in positives),
            "positive_query_mix": counts("+".join(sorted({docs[j["evidence_id"]]["metadata"]["doc_type"] for j in c["judgments"] if j["grade"] == 2})) for c in scored),
            "positive_count_distribution": counts(str(sum(j["grade"] == 2 for j in c["judgments"])) for c in scored),
            "unique_positive_documents": len(positive_ids), "unique_positive_sets": len(positive_sets),
            "repeated_positive_sets": sorted(v for v in positive_sets.values() if len(v) > 1),
            "stored_section_labels_not_validated_sec_sections": counts(j["section_path"] for j in positives),
            "unjudged_top10": unjudged, "arithmetic": retrieval_arithmetic_check()},
        "semantic": {**composition(semantic, "user_query"), "gold_claims": len(gold),
            "claim_types": counts(g["claim_type"] for g in gold),
            "numeric_metrics": counts(g["numeric"]["metric_id"] for g in gold if g.get("numeric")),
            "explicit_wrong_evidence_cases": sum(bool(c.get("known_wrong_evidence")) for c in semantic),
            "unique_gold_kb_ids": len(kb_ids), "gold_kb_ids_also_retrieval_positive": len(kb_ids & positive_ids),
            "judge_statuses": counts(r["status"] for r in read(SBASE / "judge/judgments.jsonl")),
            "numeric_counterexamples": semantic_counterexamples(semantic[0])},
    }


if __name__ == "__main__":
    print(json.dumps(audit(), sort_keys=True, indent=2))
