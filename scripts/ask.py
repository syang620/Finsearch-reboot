#!/usr/bin/env python
from __future__ import annotations
import argparse, os, json
from rag10kq.utils import read_jsonl, Chunk, FilingMeta
from rag10kq.qclassify import classify
from rag10kq.retrieval import hybrid_search, business_rule_adjust, rerank
from rag10kq.answer import best_narrative, numeric_from_tables

def load_chunks(outdir: str):
    dicts = read_jsonl(os.path.join(outdir, "chunks.jsonl"))
    chunks_by_id = {}
    chunks = []
    for d in dicts:
        f = FilingMeta(**d["filing"])
        c = Chunk(
            chunk_id=d["chunk_id"], text=d["text"], section_path=d["section_path"],
            is_table=d["is_table"], statement_type=d.get("statement_type"),
            filing=f, extra=d.get("extra", {})
        )
        chunks_by_id[c.chunk_id] = c
        chunks.append(c)
    return chunks, chunks_by_id

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="artifacts", help="Artifacts dir from build_corpus.py")
    ap.add_argument("--question", "-q", required=True)
    ap.add_argument("--topk", type=int, default=8)
    args = ap.parse_args()

    chunks, chunks_by_id = load_chunks(args.outdir)
    # Hybrid retrieval
    cand = hybrid_search(args.question, chunks_by_id, os.path.join(args.outdir, "index"))
    # Classification + business rules
    cls = classify(args.question)
    cand = business_rule_adjust(cls["scope"], cls["preferred_form"], cand, chunks_by_id)
    # Rerank
    chosen = rerank(args.question, cand, chunks_by_id, top_n=args.topk)

    # Answering
    # Try numeric if metric recognized
    tables = read_jsonl(os.path.join(args.outdir, "tables.jsonl"))
    if cls["metric"]:
        ans = numeric_from_tables(cls["metric"], tables, prefer_form=cls["preferred_form"])
        if ans:
            print(json.dumps({"question": args.question, "classification": cls, "answer": ans}, ensure_ascii=False, indent=2))
            return

    # Else narrative extractive answer
    ans = best_narrative(chosen, chunks_by_id)
    print(json.dumps({"question": args.question, "classification": cls, "answer": ans}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
