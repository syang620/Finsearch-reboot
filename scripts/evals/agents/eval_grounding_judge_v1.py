"""Secondary, frozen-rubric semantic assessment of observed bound claims."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


def main(args):
    inputs = Path(args.per_case)
    rubric_path = Path("data/evals/agents/v1/grounding_judge_rubric_v1.md")
    rubric = rubric_path.read_text()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    with urlopen(args.ollama_url + "/api/tags", timeout=10) as response:
        tags = json.load(response)
    digest = next((model["digest"] for model in tags["models"] if model["name"] == args.model), None)
    manifest = {"model": args.model, "model_digest": digest, "temperature": 0,
                "rubric_sha256": hashlib.sha256(rubric_path.read_bytes()).hexdigest(),
                "input_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
                "protocol": "Observed finalized claims only; no inferred legacy bindings; fixture labels never shown to judge."}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = []
    total_claims = 0
    with (out / "per_claim.jsonl").open("x") as handle:
        for row in map(json.loads, inputs.read_text().splitlines()):
            analyst = row["analyst"]
            if not analyst.get("ok"):
                continue
            contexts = {item["context_id"]: item for item in row["analyst_packet"]["context_items"]}
            for claim in analyst.get("claims", []):
                total_claims += 1
                item = {"case_id": row["id"], "claim_id": claim["claim_id"], "claim": claim}
                evidence = [contexts[ref] for ref in claim["context_ids"] if ref in contexts]
                item["evidence"] = evidence
                if not evidence:
                    item["label"] = "unassessable"
                else:
                    payload = {
                        "model": args.model, "stream": False, "format": "json",
                        "options": {"temperature": 0, "num_predict": 1024},
                        "messages": [{"role": "system", "content": rubric + '\nReturn JSON with keys label and reason.'},
                                     {"role": "user", "content": json.dumps({"claim": claim, "evidence": evidence})}],
                    }
                    try:
                        request = Request(args.ollama_url + "/api/chat", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
                        with urlopen(request, timeout=180) as response:
                            raw = json.load(response)
                        item["raw_response"] = raw
                        judgment = json.loads(raw["message"]["content"])
                        if judgment.get("label") not in {"fully_supported", "partially_supported", "unsupported"}:
                            raise ValueError("Invalid judge label")
                        item["label"] = judgment["label"]
                        item["reason"] = str(judgment.get("reason", ""))
                    except Exception as exc:
                        item["label"], item["error"] = "error", type(exc).__name__
                results.append(item)
                handle.write(json.dumps(item, sort_keys=True) + "\n")
                handle.flush()
                print(row["id"], claim["claim_id"], item["label"], flush=True)
    assessed = [item for item in results if item["label"] in {"fully_supported", "partially_supported", "unsupported"}]
    supported = sum(item["label"] == "fully_supported" for item in assessed)
    summary = {"observed_claims": total_claims, "assessed_claims": len(assessed),
               "errors": sum(item["label"] == "error" for item in results),
               "claim_support_precision": supported / len(assessed) if assessed else None,
               "unsupported_claim_rate": sum(item["label"] == "unsupported" for item in assessed) / len(assessed) if assessed else None,
               "grounded_claim_rate": supported / total_claims if total_claims else None,
               "assessment_coverage": len(assessed) / total_claims if total_claims else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-case", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="qwen2.5:14b-instruct")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    main(parser.parse_args())
