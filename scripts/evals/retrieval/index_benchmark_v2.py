#!/usr/bin/env python3
"""Create an isolated, immutable-name benchmark index; never recreate indexes."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time

import requests
from qdrant_client import QdrantClient,models

from evals.retrieval_benchmark_v2 import load_dataset,sha256
from ingestion.sec_embedder import embed_batch_with_qwen3
from ingestion.qdrant_ingester import docs_to_points


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path,default=Path("data/evals/retrieval/benchmark_v2"))
    p.add_argument("--qdrant-url",default="http://127.0.0.1:6333")
    p.add_argument("--embedding-url",default="http://127.0.0.1:11434/api/embed")
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args()
    _,docs,validation=load_dataset(a.dataset)
    config=json.loads((a.dataset/"comparison_config.json").read_text())
    digest=sha256(a.dataset/"corpus.jsonl")
    collection="finsearch_benchmark_v2_"+digest[:16]
    client=QdrantClient(url=a.qdrant_url,timeout=120)
    if client.collection_exists(collection):
        raise FileExistsError("Benchmark collection already exists; never overwrite or recreate it.")
    model_url=a.embedding_url.rsplit("/api/",1)[0]
    response=requests.get(model_url+"/api/tags",timeout=10)
    response.raise_for_status()
    models_info=response.json()["models"]
    assert any(m["name"]==config["embedding_model"] and m["digest"]==config["embedding_digest"] for m in models_info),"Embedding digest mismatch"
    a.out_dir.mkdir(parents=True,exist_ok=False)
    avg_len=sum(len(d["content"].split()) for d in docs)/len(docs)
    started=time.perf_counter()
    manifest={"collection":collection,"corpus_sha256":digest,"qdrant_url":a.qdrant_url,
              "embedding_url":a.embedding_url,"embedding_model":config["embedding_model"],
              "embedding_digest":config["embedding_digest"],"bm25_model":"Qdrant/bm25",
              "bm25_avg_len":avg_len,"validation":validation,"started_at":datetime.now(timezone.utc).isoformat(),
              "completed":False,"batch_size":8,"production_collection_mutated":False}
    (a.out_dir/"started.json").write_text(json.dumps(manifest,indent=2)+"\n")
    with (a.out_dir/"embedded.jsonl").open("x") as cache:
        for start in range(0,len(docs),8):
            batch=docs[start:start+8]
            vectors=embed_batch_with_qwen3([d["content"] for d in batch],api_url=a.embedding_url,model=config["embedding_model"])
            if len(vectors)!=len(batch) or not all(len(v)==len(vectors[0]) for v in vectors):
                raise ValueError("Embedding response count/dimension mismatch")
            if start==0:
                client.create_collection(collection_name=collection,
                    vectors_config={"dense":models.VectorParams(size=len(vectors[0]),distance=models.Distance.COSINE)},
                    sparse_vectors_config={"bm25":models.SparseVectorParams(modifier=models.Modifier.IDF)})
            embedded=[{**d,"embedding":v} for d,v in zip(batch,vectors)]
            client.upsert(collection_name=collection,points=docs_to_points(embedded,bge_m3_encode=False,bm25_avg_len=avg_len),wait=True)
            for d in embedded:cache.write(json.dumps(d)+"\n")
            cache.flush()
            print(json.dumps({"indexed":min(start+8,len(docs)),"total":len(docs)}),flush=True)
    manifest.update(completed=True,wall_seconds=time.perf_counter()-started,
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    embedded_sha256=sha256(a.out_dir/"embedded.jsonl"),
                    indexed_points=client.count(collection_name=collection,exact=True).count,
                    qdrant_version=requests.get(a.qdrant_url,timeout=10).json()["version"])
    assert manifest["indexed_points"]==len(docs)
    (a.out_dir/"completed.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__=="__main__":main()
