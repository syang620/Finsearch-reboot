#!/usr/bin/env python
from __future__ import annotations
import argparse, os, json
from typing import List, Dict, Tuple
from rag10kq.utils import read_jsonl, Chunk, FilingMeta
from rag10kq.retrieval import hybrid_search, business_rule_adjust

def ndcg_at_k(rel: List[int], k: int = 10) -> float:
    # rel is binary relevance in ranked order
    import math
    dcg = sum((rel[i] / math.log2(i+2) for i in range(min(len(rel), k))))
    # ideal
    rel_sorted = sorted(rel, reverse=True)
    idcg = sum((rel_sorted[i] / math.log2(i+2) for i in range(min(len(rel_sorted), k))))
    return dcg / idcg if idcg > 0 else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="artifacts")
    ap.add_argument("--gold", required=True, help="Path to gold JSONL with fields: question, gold_chunk_ids[]")
    args = ap.parse_args()

    # load chunks
    dicts = read_jsonl(os.path.join(args.outdir, "chunks.jsonl"))
    chunks_by_id = {}
    for d in dicts:
        f = FilingMeta(**d["filing"])
        chunks_by_id[d["chunk_id"]] = Chunk(
            chunk_id=d["chunk_id"], text=d["text"], section_path=d["section_path"],
            is_table=d["is_table"], statement_type=d.get("statement_type"),
            filing=f, extra=d.get("extra", {})
        )

    # eval
    ndcgs = []
    rec5 = []
    with open(args.gold, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            q = ex["question"]
            gold_ids = set(ex["gold_chunk_ids"])
            cand = hybrid_search(q, chunks_by_id, os.path.join(args.outdir, "index"))
            rel = [1 if cid in gold_ids else 0 for cid, _ in cand][:10]
            ndcgs.append(ndcg_at_k(rel, k=10))
            rec5.append(1.0 if any(cid in gold_ids for cid, _ in cand[:5]) else 0.0)

    print(json.dumps({
        "n": len(ndcgs),
        "nDCG@10": sum(ndcgs)/len(ndcgs) if ndcgs else 0.0,
        "Recall@5": sum(rec5)/len(rec5) if rec5 else 0.0
    }, indent=2))

if __name__ == "__main__":
    main()
