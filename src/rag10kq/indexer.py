from __future__ import annotations
import os, json, pickle, numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from .utils import Chunk, word_tokens, ensure_dir

BGE_MODEL = "BAAI/bge-small-en-v1.5"  # strong & light. Use query instruction.
CE_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # re-ranker.
BGE_QINSTR = "Represent this sentence for searching relevant passages: "  # per model card.

def build_bm25(chunks: List[Chunk], outdir: str) -> None:
    docs = [word_tokens(c.text) for c in chunks]
    bm25 = BM25Okapi(docs)
    obj = {"bm25": bm25, "doc_ids": [c.chunk_id for c in chunks]}
    with open(os.path.join(outdir, "bm25.pkl"), "wb") as f:
        pickle.dump(obj, f)

def load_bm25(outdir: str):
    with open(os.path.join(outdir, "bm25.pkl"), "rb") as f:
        return pickle.load(f)

def build_dense_index(chunks: List[Chunk], outdir: str) -> None:
    model = SentenceTransformer(BGE_MODEL)
    texts = [c.text for c in chunks]
    embs = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embs = np.asarray(embs, dtype="float32")
    index = faiss.IndexFlatIP(embs.shape[1])  # cosine via normalized dot
    index.add(embs)
    faiss.write_index(index, os.path.join(outdir, "faiss.index"))
    with open(os.path.join(outdir, "dense_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"doc_ids": [c.chunk_id for c in chunks]}, f)

def load_dense_index(outdir: str):
    index = faiss.read_index(os.path.join(outdir, "faiss.index"))
    meta = json.load(open(os.path.join(outdir, "dense_meta.json"), encoding="utf-8"))
    return index, meta["doc_ids"]

def get_cross_encoder():
    return CrossEncoder(CE_MODEL)

def encode_query(text: str) -> str:
    # prepend instruction for BGE query encoding
    return BGE_QINSTR + text
