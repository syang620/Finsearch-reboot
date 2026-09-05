"""Evaluate frozen PR8 outputs and raw-evidence invariants without runtime helpers."""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess

DATASET = Path("data/evals/agents/v1/filing_period_pr8.json")


class FixtureClient:
    def __init__(self, case): self.case, self.calls = case, []
    async def resolve_cik(self, ticker): self.calls.append(["resolve_cik", ticker]); return "0000320193"
    async def get_submissions(self, cik): self.calls.append(["get_submissions", cik]); return deepcopy(self.case["submissions"])
    async def get_companyfacts(self, cik): self.calls.append(["get_companyfacts", cik]); return deepcopy(self.case["companyfacts"])


def differences(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return [p for k in sorted(a.keys() | b.keys()) for p in (
            differences(a[k], b[k], f"{path}/{k}") if k in a and k in b else [f"{path}/{k}"])]
    return [] if a == b else [path]


def provenance_consistent(case, result):
    """Verify returned facts against raw records, never selection's own truth flags."""
    chosen = result["components"] + ([result["primary_fact"]] if result["primary_fact"] else [])
    signatures = set()
    instant = case["request"]["metric_id"] == "total_debt"
    for c in chosen:
        raw = case["companyfacts"]["facts"].get(c["taxonomy"], {}).get(c["concept_name"], {}).get("units", {}).get(c["unit"], [])
        if not any(f.get("accn") == c["accession_number"] and f.get("end") == c["report_date"]
                   and f.get("form") == c["form_type"] and f.get("start") == c["start_date"]
                   and f.get("val") == c["value"] and f.get("filed") == c["filed_date"] for f in raw): return False
        if not (c["accession_number"] == result["accession_number"] and c["report_date"] == result["report_date"]
                and c["form_type"] == result["form_type"] == "10-K"): return False
        if instant:
            if c["start_date"] is not None: return False
        else:
            try:
                if (date.fromisoformat(c["report_date"])-date.fromisoformat(c["start_date"])).days+1 not in (364, 365, 366, 371): return False
            except (TypeError, ValueError): return False
        signatures.add((c["accession_number"], c["report_date"], c["form_type"], c["start_date"]))
    if result["ok"]:
        return bool(chosen) and len(signatures) == 1 and result["value"] == sum(c["value"] for c in chosen)
    return result["value"] is None and result["unit"] is None


def shuffled(case, seed):
    c = deepcopy(case)
    rng = random.Random(seed)
    recent = c["submissions"]["filings"]["recent"]
    order = list(range(len(recent["form"])))
    rng.shuffle(order)
    for key, column in recent.items(): recent[key] = [column[i] for i in order]
    for taxonomy in c["companyfacts"]["facts"].values():
        for payload in taxonomy.values():
            for records in payload["units"].values(): rng.shuffle(records)
    return c


async def evaluate():
    # Import only the public entrypoint under test, not a runtime selection helper.
    from mcp_server.tools.sec_metric import get_metric
    rows = []
    for c in json.loads(DATASET.read_text())["cases"]:
        client = FixtureClient(c)
        actual = (await get_metric(**c["request"], client=client)).model_dump(mode="json")
        mismatches = differences(c["expected_pr8_result"], actual)
        delta = differences(c["old_result"], actual)
        unexpected = [p for p in delta if p not in c["allowed_fields"]]
        invariant = True
        for seed in range(5):
            variant = shuffled(c, seed)
            observed = (await get_metric(**c["request"], client=FixtureClient(variant))).model_dump(mode="json")
            invariant &= observed == actual
        expected_calls = [["resolve_cik", "AAPL"], ["get_submissions", "0000320193"]]
        if "anchor" in c["expected_pr8_result"]["trace"]: expected_calls.append(["get_companyfacts", "0000320193"])
        # Unchanged-case parity concerns every legacy field outside the explicitly
        # frozen additions/semantic changes, not an ignored whole response.
        unchanged = not c["semantic_change_expected"]
        parity = not unexpected and (not unchanged or all(actual[k] == c["old_result"][k]
                    for k in ("status", "value", "accession_number")))
        rows.append({"id": c["id"], "exact_expected": not mismatches, "mismatches": mismatches,
                     "unexpected_differences": unexpected, "order_invariant": invariant,
                     "provenance_consistent": provenance_consistent(c, actual),
                     "unchanged_case": unchanged, "unchanged_parity": parity,
                     "calls_correct": client.calls == expected_calls,
                     "semantic_change_expected": c["semantic_change_expected"], "actual": actual})
    return rows


def summarize(rows):
    n = len(rows)
    unchanged = [r for r in rows if r["unchanged_case"]]
    return {"cases": n, "approved_change_accuracy": sum(r["exact_expected"] for r in rows)/n,
            "unexpected_difference_rate": sum(bool(r["unexpected_differences"]) for r in rows)/n,
            "unchanged_cases": len(unchanged), "unchanged_case_parity": sum(r["unchanged_parity"] for r in unchanged)/len(unchanged),
            "intentional_semantic_changes": sum(r["semantic_change_expected"] for r in rows),
            "order_invariance_rate": sum(r["order_invariant"] for r in rows)/n,
            "provenance_consistency_rate": sum(r["provenance_consistent"] for r in rows)/n,
            "network_call_contract_rate": sum(r["calls_correct"] for r in rows)/n,
            "passed": all(r["exact_expected"] and not r["unexpected_differences"] and r["order_invariant"]
                          and r["provenance_consistent"] and r["calls_correct"] for r in rows)}


def main():
    import pytest
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    rows = asyncio.run(evaluate())
    summary = summarize(rows)
    files = [DATASET, Path(__file__).resolve().relative_to(Path.cwd()), Path("src/mcp_server/tools/sec_metric.py"),
             Path("src/agents/contracts.py"), Path("src/agents/orchestrator/agent_orchestrator.py")]
    manifest = {"evaluated_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "tracked_worktree_clean": not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip(),
                "created_at": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                "pytest": pytest.__version__, "snapshot_source": json.loads(DATASET.read_text())["source_commit"],
                "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    for name, data in [("summary", summary), ("manifest", manifest)]:
        (out/f"{name}.json").write_text(json.dumps(data, indent=2)+"\n")
    (out/"per_case.jsonl").write_text("".join(json.dumps(r, sort_keys=True)+"\n" for r in rows))
    print(json.dumps(summary))
    for r in rows:
        if not r["exact_expected"]: print(r["id"], r["mismatches"])
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__": main()
