#!/usr/bin/env python3
"""Freeze source-grounded judgments, never inferred from ranked results."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from bs4 import BeautifulSoup
from benchmark_v2_annotations import GROUPS, SPECS, OVERRIDES, QUERY_OVERRIDES

ROOT = Path("data/evals/retrieval/benchmark_v2")
SEED = Path("data/evals/retrieval/text/aapl_2024_10k_text_retrieval_eval_split_with_uids.jsonl")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_normalize(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def freeze():
    outputs=[ROOT/name for name in ["queries.jsonl", "seed_compatibility.json", "dataset_manifest.json"]]
    if any(p.exists() for p in outputs):
        raise FileExistsError("Dataset already frozen; choose a new benchmark version.")
    docs=[json.loads(line) for line in (ROOT/"corpus.jsonl").read_text().splitlines()]
    corpus_manifest=json.loads((ROOT/"corpus_manifest.json").read_text())
    source_text={s["source_html"]:source_normalize(BeautifulSoup(Path(s["source_html"]).read_text(), "lxml").get_text(" "))
                 for s in corpus_manifest["sources"]}
    seed=[json.loads(line) for line in SEED.read_text().splitlines()]
    cases=[]
    for src in corpus_manifest["sources"]:
        ticker,year=src["ticker"],src["fiscal_year"]
        filing_docs=[d for d in docs if d["metadata"]["ticker"]==ticker and d["metadata"]["fiscal_year"]==year]
        group_matches={}
        for name,base in GROUPS[ticker].items():
            kind,needles=OVERRIDES.get((ticker,year,name),base)
            needles=[s.format(year=year) for s in needles]
            matched=[d for d in filing_docs if d["metadata"]["doc_type"]==kind and all(n in d["content"] for n in needles)]
            # A tax-credit row is not an R&D expense row. Exact cell semantics,
            # not loose term overlap, determine the independently authored label.
            if name=="rd_table":
                matched=[d for d in matched if re.search(r"\|\s*Research and development\s*\|",d["content"])]
            if not matched:
                raise ValueError(f"Unresolved source group: {ticker}/{year}/{name}")
            group_matches[name]=(matched,needles)
        for number,(stratum,query,groups,partials,negatives) in enumerate(SPECS[ticker],1):
            query=QUERY_OVERRIDES.get((ticker,year,number),query.format(year=year))
            seed_id=None
            if (ticker,year,number)==("AAPL",2024,9):
                query=seed[0]["query"]
                seed_id=seed[0]["query_id"]
            judgments={}
            for grade,names in [(2,groups),(1,partials),(0,negatives)]:
                for name in names:
                    matched,needles=group_matches[name]
                    for d in matched:
                        if d["id"] in judgments and judgments[d["id"]]["grade"] != grade:
                            raise ValueError(f"Conflicting judgments: {ticker}/{year}/{number}/{d['id']}")
                        spans=[]
                        for needle in needles:
                            pos=d["content"].index(needle)
                            start=d["content"].rfind("\n\n",0,pos)+2
                            if start==1: start=0
                            end=d["content"].find("\n\n",pos+len(needle))
                            if end==-1:end=len(d["content"])
                            normalized=source_normalize(needle)
                            source_offset=source_text[src["source_html"]].find(normalized)
                            if source_offset<0:
                                raise ValueError(f"Anchor absent from HTML: {ticker}/{year}/{name}: {needle}")
                            spans.append({"start":start,"end":end,"quote":d["content"][start:end],
                                          "anchor":needle,"source_normalized_offset":source_offset})
                        entry=judgments.setdefault(d["id"],{"evidence_id":d["id"],"grade":grade,
                                     "groups":[],"spans":[],"source_html":src["source_html"],
                                     "source_sha256":src["source_sha256"],
                                     "section_path":d["metadata"]["section_path"],
                                     "content_sha256":d["content_sha256"]})
                        entry["groups"].append(name)
                        entry["spans"].extend(spans)
            cases.append({"id":f"KBV2_{ticker}_{year}_{number:02}", "query":query,
                          "ticker":ticker,"fiscal_year":year,"form_type":"10-K",
                          "stratum":stratum,"status":"answerable","required_groups":groups,
                          "judgments":sorted(judgments.values(),key=lambda j:j["evidence_id"]),
                          "seed_query_id":seed_id,
                          "annotation":"Source-first AI-authored judgment; literal supporting anchors expanded to all matching corpus occurrences, before retrieval; not independent human adjudication."})
        cases.append({"id":f"KBV2_{ticker}_{year}_EX", "query":f"What will {ticker}'s audited full-year revenue be in {year+5}, as reported in its {year} 10-K?",
                      "ticker":ticker,"fiscal_year":year,"form_type":"10-K",
                      "stratum":"unanswerable","status":"excluded","required_groups":[],"judgments":[],
                      "exclusion_reason":"Future audited result cannot be established by the source-year filing. Excluded from ranking denominators; not an abstention benchmark."})
    # Audit the historical seed without changing any semantic labels or IDs.
    audit=[]
    aapl=[d for d in docs if d["metadata"]["ticker"]=="AAPL" and d["metadata"]["fiscal_year"]==2024 and d["metadata"]["doc_type"]=="text_chunk"]
    for s in seed:
        mappings=[]
        for e in s.get("evidence",[]):
            anchor=e.get("anchor_text","")
            truncated="…" in anchor or "..." in anchor
            matches=[d["id"] for d in aapl if source_normalize(anchor) in source_normalize(d["content"])] if anchor and not truncated else []
            mappings.append({"original_doc_id":e.get("doc_id"),"original_chunk_uid":e.get("chunk_uid"),
                             "anchor_sha256":hashlib.sha256(anchor.encode()).hexdigest(),
                             "mapped_evidence_ids":matches,"status":"exact_anchor_preserved" if matches else "unresolved_truncated_anchor" if truncated else "unresolved_anchor"})
        audit.append({"source_query_id":s["query_id"],"original_relevant_doc_ids":s["relevant_doc_ids"],
                      "original_relevant_chunk_uids":s.get("relevant_chunk_uids",[]),
                      "mapping":mappings,"adopted_query_id":"KBV2_AAPL_2024_09" if s["query_id"]==seed[0]["query_id"] else None,
                      "disposition":"exact source-query adoption, separately inspected" if s["query_id"]==seed[0]["query_id"] else "compatibility audit only; no semantic relabeling or metric inclusion"})
    outputs[0].write_text("".join(json.dumps(c,ensure_ascii=False,sort_keys=True)+"\n" for c in cases))
    outputs[1].write_text(json.dumps({"source":SEED.as_posix(),"sha256":sha(SEED),"case_count":len(seed),"cases":audit},indent=2)+"\n")
    manifest={"benchmark_id":"sec_retrieval_benchmark_v2","version":2,"base_commit":"9ec25f568402d700c88a5fa35b7e5e750bc36d5f",
              "query_count":len(cases),"answerable_count":sum(c["status"]=="answerable" for c in cases),
              "strata":dict(sorted(Counter(c["stratum"] for c in cases).items())),
              "files":{p.name:sha(p) for p in [ROOT/"corpus.jsonl",ROOT/"corpus_manifest.json",*outputs[:2]]},
              "annotation_specs_sha256":sha("scripts/evals/retrieval/benchmark_v2_annotations.py"),
              "historical_seed":{"path":SEED.as_posix(),"sha256":sha(SEED),"unchanged":True},
              "freeze_rule":"Commit before any ranked retrieval measurement; immutable thereafter. No query selected from mode outcomes.",
              "limitations":["Three large technology/commerce companies; six annual filings; paired topic templates across years are correlated, not 120 independent intents.",
                             "AI source-inspected annotations, not double-blind human labels; known-gold recall, incomplete judgments outside explicitly labeled spans.",
                             "Existing parser heading mistakes retained; table corpus contains raw tables, no model summaries or table-row documents.",
                             "Metadata-filtered single-filing KB retrieval, not company/year resolution, answer generation or XBRL execution."]}
    outputs[2].write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"answerable":manifest["answerable_count"],"cases":len(cases),"dataset_sha256":sha(outputs[0])}))


if __name__=="__main__":
    freeze()
