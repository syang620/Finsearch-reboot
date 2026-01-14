from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from .utils import word_tokens, minmax_scale
from .indexer import load_bm25, load_dense_index, get_cross_encoder, encode_query, BGE_MODEL
from .utils import Chunk

def hybrid_search(question: str, chunks_by_id: Dict[str, Chunk], outdir: str,
                  k_sparse: int = 40, k_dense: int = 40) -> List[Tuple[str, float]]:
    # BM25
    bm25_obj = load_bm25(outdir)
    bm25 = bm25_obj["bm25"]; doc_ids = bm25_obj["doc_ids"]
    tok_q = word_tokens(question)
    bm25_scores = bm25.get_scores(tok_q)
    # Dense
    index, dense_doc_ids = load_dense_index(outdir)
    assert doc_ids == dense_doc_ids, "doc id mismatch"
    encoder = SentenceTransformer(BGE_MODEL)
    qvec = encoder.encode([encode_query(question)], normalize_embeddings=True)
    D, I = index.search(np.asarray(qvec, dtype="float32"), k_dense)
    dense_scores = np.zeros_like(bm25_scores, dtype=float)
    for rank, doc_idx in enumerate(I[0]):
        if doc_idx < 0: continue
        dense_scores[doc_idx] = D[0, rank]

    # Score fusion
    s_sparse = minmax_scale(bm25_scores.tolist())
    s_dense  = minmax_scale(dense_scores.tolist())
    scores = [0.4 * s_sparse[i] + 0.6 * s_dense[i] for i in range(len(doc_ids))]

    # return top candidates
    pairs = sorted([(doc_ids[i], scores[i]) for i in range(len(doc_ids))],
                   key=lambda x: x[1], reverse=True)[:max(k_sparse, k_dense)]
    return pairs

def business_rule_adjust(question_scope: str, pref_form: str, candidates: List[Tuple[str,float]],
                         chunks_by_id: Dict[str, Chunk]) -> List[Tuple[str,float]]:
    out = []
    for cid, s in candidates:
        c = chunks_by_id[cid]
        delta = 0.0
        if question_scope == "annual_metric":
            delta += 0.10 if c.filing.form_type == "10-K" else -0.05
        elif question_scope == "quarter_metric":
            delta += 0.08 if c.filing.form_type == "10-Q" else -0.03
        if pref_form and c.filing.form_type == pref_form:
            delta += 0.02
        out.append((cid, s + delta))
    out.sort(key=lambda x: x[1], reverse=True)
    return out

def rerank(question: str, candidates: List[Tuple[str,float]],
           chunks_by_id: Dict[str, Chunk], top_n: int = 10) -> List[Tuple[str,float]]:
    ce = get_cross_encoder()
    pairs = [(question, chunks_by_id[cid].text) for cid, _ in candidates]
    scores = ce.predict(pairs).tolist()
    rescored = [(candidates[i][0], scores[i]) for i in range(len(candidates))]
    rescored.sort(key=lambda x: x[1], reverse=True)
    return rescored[:top_n]
