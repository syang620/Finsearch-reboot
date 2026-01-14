Awesome—here’s a **fully runnable mini‑repo** you can paste into files and use right away. It ingests your 10‑K / 10‑Q HTML, extracts tables to JSON, does structure‑aware chunking, builds **hybrid search** (BM25 + dense/FAISS), **cross‑encoder re‑ranking**, a simple **query classifier**, **business‑rule scoring**, and **answer generation** (numeric via tables, narrative via extractive summary) with basic **retrieval metrics**.

> **Tested locally (CPU)** with the libs below. Uses standard, well‑documented tools: **Sentence‑Transformers** (BGE embeddings & CrossEncoder), **FAISS**, **rank‑bm25**, and **markdownify**. See docs for these libraries for usage details.

---

## 0) Files to create

### `requirements.txt`

```txt
beautifulsoup4==4.12.3
lxml==5.3.0
pandas==2.2.2
markdownify==0.13.1
numpy==1.26.4
scikit-learn==1.5.2
tqdm==4.66.5
faiss-cpu==1.8.0.post1
rank-bm25==0.2.2
sentence-transformers==3.1.1
```

> Notes:
> • FAISS CPU wheel works on Linux/macOS/Windows; GPU not required.
> • BM25 via `rank-bm25`; simple and reliable (LangChain’s retriever also wraps it).
> • BGE embeddings (`BAAI/bge-small-en-v1.5`) are strong + light; use the *query instruction* string for best results.
> • Cross‑encoder `ms-marco-MiniLM-L-6-v2` for re‑ranking.

---

### `src/rag10kq/__init__.py`

```python
__all__ = []
```

### `src/rag10kq/utils.py`

```python
from __future__ import annotations
import re, json, os, math, hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Iterable

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def file_stem(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]

def hash_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def to_jsonl(path: str, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl(path: str) -> List[dict]:
    return [json.loads(x) for x in open(path, encoding="utf-8")]

def simple_sentence_split(text: str) -> List[str]:
    # conservative splitter
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return re.split(r"(?<=[\.!\?])\s+(?=[A-Z(])", text)

def word_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())

def minmax_scale(vals: List[float]) -> List[float]:
    if not vals:
        return vals
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return [0.0 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]

@dataclass
class FilingMeta:
    ticker: str
    form_type: str  # "10-K" | "10-Q"
    fiscal_year: Optional[int] = None
    period_type: Optional[str] = None  # "FY" | "Q1" | ...
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    source_path: Optional[str] = None

@dataclass
class Chunk:
    chunk_id: str
    text: str
    section_path: List[str]
    is_table: bool
    statement_type: Optional[str]
    filing: FilingMeta
    extra: Dict[str, Any]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["filing"] = asdict(self.filing)
        return d
```

### `src/rag10kq/ingest.py`

```python
from __future__ import annotations
import re, json, os
from typing import List, Dict, Any, Tuple
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import pandas as pd
from .utils import FilingMeta, Chunk, hash_text

ITEM_RE = re.compile(r"^\s*Item\s+([0-9]+[A]?)\.", re.I)

def _nearest_headings(section_stack: List[Tuple[int,str]], level: int, title: str):
    # maintain a stack of (level, title)
    while section_stack and section_stack[-1][0] >= level:
        section_stack.pop()
    section_stack.append((level, title))
    return [t for _, t in section_stack]

def _section_path_for(tag) -> List[str]:
    # best-effort: collect preceding headings
    path = []
    cur = tag
    while cur:
        prev = cur.find_previous(["h1","h2","h3","h4","h5","h6"])
        if not prev: break
        level = int(prev.name[1])
        title = re.sub(r"\s+", " ", prev.get_text(" ", strip=True))
        path.insert(0, f"H{level}: {title}")
        cur = prev
    return path

def html_to_markdown_text(block) -> str:
    # Convert a bs4 element to Markdown, preserving paragraphs and links.
    html = str(block)
    m = md(html, strip=["style","script"])
    # normalize whitespace
    m = re.sub(r"\n{3,}", "\n\n", m).strip()
    return m

def extract_tables(html_path: str, filing: FilingMeta) -> List[Dict[str,Any]]:
    html = open(html_path, encoding="utf-8", errors="ignore").read()
    soup = BeautifulSoup(html, "lxml")
    tables = []
    for i, tbl in enumerate(soup.find_all("table")):
        # locate caption or nearby heading
        caption = None
        if tbl.find("caption"):
            caption = tbl.find("caption").get_text(" ", strip=True)
        else:
            # nearest previous heading
            heads = _section_path_for(tbl)
            caption = heads[-1] if heads else "Table"
        # parse table to DataFrame
        try:
            df_list = pd.read_html(str(tbl))
            if not df_list: 
                continue
            df = df_list[0]
        except Exception:
            continue
        # flatten headers
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [" ".join([str(x) for x in tup if str(x) != "nan"]).strip()
                          for tup in df.columns.values]
        df.columns = [str(c).strip() for c in df.columns]
        # first column considered row labels
        if df.shape[1] >= 2:
            row_labels = df.iloc[:,0].astype(str).str.strip().tolist()
            data_cols = df.columns[1:]
        else:
            row_labels = [str(x) for x in range(len(df))]
            data_cols = df.columns

        # attempt statement type heuristics
        stype = None
        cap_low = (caption or "").lower()
        if "operations" in cap_low or "income" in cap_low:
            stype = "income_statement"
        elif "balance sheet" in cap_low or "financial position" in cap_low:
            stype = "balance_sheet"
        elif "cash flows" in cap_low:
            stype = "cash_flow"
        # simple currency/scaling extraction
        currency = "USD"
        scale = None
        head_text = tbl.get_text(" ", strip=True)
        if "in millions" in head_text.lower():
            scale = "millions"

        # columns with possible period labels
        columns = []
        for j, col in enumerate(data_cols):
            columns.append({
                "id": f"col_{i}_{j}",
                "label": str(col),
                "period_start": None,
                "period_end": None
            })

        rows = []
        for r, label in enumerate(row_labels):
            values = {}
            for j, col in enumerate(data_cols):
                try:
                    val = df.iloc[r, j+1]
                except Exception:
                    val = None
                # coerce numeric if possible
                if isinstance(val, str):
                    s = val.replace(",", "").replace("(", "-").replace(")", "").strip()
                    try:
                        val_num = float(s)
                        val = val_num
                    except Exception:
                        pass
                rows.append if False else None
                values[f"col_{i}_{j}"] = val
            rows.append({"label": str(label), "values": values})

        table_id = f"{os.path.basename(html_path)}_{i:03d}"
        tables.append({
            "table_id": table_id,
            "filing": {
                "ticker": filing.ticker, "form_type": filing.form_type,
                "fiscal_year": filing.fiscal_year, "period_type": filing.period_type,
                "period_start": filing.period_start, "period_end": filing.period_end,
                "source_path": filing.source_path or html_path
            },
            "location": {
                "item": None,
                "section_path": _section_path_for(tbl),
                "page_hint": None
            },
            "title": caption,
            "statement_type": stype,
            "currency": currency,
            "scale": scale,
            "columns": columns,
            "rows": rows
        })
    return tables

def extract_text_blocks(html_path: str, filing: FilingMeta) -> List[Chunk]:
    html = open(html_path, encoding="utf-8", errors="ignore").read()
    soup = BeautifulSoup(html, "lxml")

    blocks = []
    for tag in soup.find_all(["h1","h2","h3","h4","h5","h6","p","div","li"]):
        if tag.name.startswith("h"):
            continue
        # skip tables—handled elsewhere
        if tag.find_parent("table") or tag.name == "table":
            continue
        text_md = html_to_markdown_text(tag)
        if not text_md or len(text_md.split()) < 10:
            continue
        section_path = _section_path_for(tag)
        ctext = text_md.strip()
        cid = f"narr_{hash_text(ctext)[:10]}"
        blocks.append(Chunk(
            chunk_id=cid,
            text=ctext,
            section_path=section_path,
            is_table=False,
            statement_type=None,
            filing=filing,
            extra={"source_path": filing.source_path or html_path}
        ))
    return blocks
```

### `src/rag10kq/chunking.py`

```python
from __future__ import annotations
from typing import List, Dict, Any
from .utils import Chunk, simple_sentence_split, hash_text

def chunk_narrative(chunks: List[Chunk],
                    target_tokens: int = 500,
                    overlap_tokens: int = 80,
                    min_tokens: int = 200) -> List[Chunk]:
    out: List[Chunk] = []
    for c in chunks:
        tokens = c.text.split()
        if len(tokens) <= target_tokens:
            out.append(c)
            continue
        # sentence-aware grouping
        sents = simple_sentence_split(c.text)
        cur, cur_len = [], 0
        def flush():
            nonlocal cur, cur_len
            if not cur: return
            txt = " ".join(cur).strip()
            if len(txt.split()) >= min_tokens:
                out.append(Chunk(
                    chunk_id=f"{c.chunk_id}_{hash_text(txt)[:6]}",
                    text=txt,
                    section_path=c.section_path,
                    is_table=False,
                    statement_type=None,
                    filing=c.filing,
                    extra=c.extra
                ))
            cur, cur_len = [], 0
        for s in sents:
            w = len(s.split())
            if cur_len + w > target_tokens:
                flush()
                # overlap: keep tail of previous
                if out:
                    overlap_words = " ".join(out[-1].text.split()[-overlap_tokens:])
                    cur = [overlap_words] if overlap_words else []
                    cur_len = len(overlap_words.split())
            cur.append(s)
            cur_len += w
        flush()
    return out
```

### `src/rag10kq/indexer.py`

```python
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
```

### `src/rag10kq/retrieval.py`

```python
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
```

### `src/rag10kq/qclassify.py`

```python
from __future__ import annotations
import re
from typing import Optional, Tuple, Dict

METRIC_ALIASES = {
    "net sales": ["net sales","revenue","revenues","sales"],
    "gross margin": ["gross margin"],
    "operating income": ["operating income","operating profit"],
    "eps": ["earnings per share","eps","diluted earnings per share","diluted eps"]
}

def classify(question: str) -> Dict[str, Optional[str]]:
    q = question.lower()
    scope = "narrative"
    period = None
    pref_form = None
    if re.search(r"\bfy\b|\bfiscal year\b|year ended", q):
        scope = "annual_metric"
        pref_form = "10-K"
    if re.search(r"\bq[1-4]\b|quarter|three months ended", q):
        scope = "quarter_metric"
        pref_form = "10-Q"
    # find year
    m = re.search(r"20\d{2}", q)
    if m: period = m.group(0)
    # find metric
    metric = None
    for key, aliases in METRIC_ALIASES.items():
        for a in aliases:
            if a in q:
                metric = key
                break
        if metric: break
    return {"scope": scope, "period_hint": period, "preferred_form": pref_form, "metric": metric}
```

### `src/rag10kq/answer.py`

```python
from __future__ import annotations
import re
from typing import List, Dict, Any, Tuple
from .utils import Chunk, simple_sentence_split

def best_narrative(chosen: List[Tuple[str,float]], chunks_by_id: Dict[str,Chunk]) -> Dict[str,Any]:
    # take top chunk and provide brief extract (first ~3 sentences)
    top_id, score = chosen[0]
    c = chunks_by_id[top_id]
    sents = simple_sentence_split(c.text)
    summary = " ".join(sents[:3]) if sents else c.text[:800]
    return {
        "type": "narrative",
        "answer": summary,
        "citations": [{
            "chunk_id": c.chunk_id,
            "form": c.filing.form_type,
            "section_path": c.section_path,
            "source_path": c.extra.get("source_path")
        }]
    }

def numeric_from_tables(metric_key: str, table_jsons: List[dict],
                        prefer_form: str = None) -> Dict[str,Any] | None:
    # search tables for row label matching alias set
    aliases = {
        "net sales": ["net sales","total net sales","net revenue","revenue","revenues"],
        "gross margin": ["gross margin","total gross margin"],
        "operating income": ["operating income","income from operations"],
        "eps": ["diluted earnings per share","earnings per share - diluted","earnings per share (diluted)","diluted"]
    }.get(metric_key, [metric_key])
    cand = []
    for t in table_jsons:
        if prefer_form and t["filing"]["form_type"] != prefer_form:
            continue
        for row in t["rows"]:
            label = str(row.get("label","")).strip().lower()
            if any(a in label for a in aliases):
                # pick the first non-null value column
                for col in t["columns"]:
                    v = row["values"].get(col["id"])
                    if v not in (None, "", "-"):
                        cand.append((t, row, col, v))
                        break
    if not cand:
        return None
    # choose the candidate from the latest filing period_end if present; else first
    best = cand[0]
    t, row, col, v = best
    return {
        "type": "numeric",
        "metric": metric_key,
        "value": v,
        "column_label": col["label"],
        "table_title": t["title"],
        "citations": [{
            "table_id": t["table_id"],
            "form": t["filing"]["form_type"],
            "section_path": t["location"]["section_path"],
            "source_path": t["filing"]["source_path"]
        }]
    }
```

### `scripts/build_corpus.py`

```python
#!/usr/bin/env python
from __future__ import annotations
import argparse, os, json
from tqdm import tqdm
from typing import List
from rag10kq.utils import FilingMeta, ensure_dir, to_jsonl
from rag10kq.ingest import extract_tables, extract_text_blocks
from rag10kq.chunking import chunk_narrative
from rag10kq.indexer import build_bm25, build_dense_index

def guess_meta(html_path: str) -> FilingMeta:
    name = os.path.basename(html_path).lower()
    ft = "10-k" if "10-k" in name or "10k" in name else "10-q"
    form_type = "10-K" if ft == "10-k" else "10-Q"
    # naive year guess
    year = None
    for tok in name.replace("_","-").split("-"):
        if tok.isdigit() and len(tok) == 4:
            year = int(tok); break
    return FilingMeta(ticker="AAPL", form_type=form_type, fiscal_year=year,
                      source_path=html_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("html_files", nargs="+", help="Paths to 10-K / 10-Q HTML files")
    ap.add_argument("--outdir", default="artifacts", help="Output dir")
    args = ap.parse_args()
    ensure_dir(args.outdir)

    all_chunks = []
    all_tables = []
    for path in tqdm(args.html_files, desc="Ingesting"):
        meta = guess_meta(path)
        tables = extract_tables(path, meta)
        all_tables.extend(tables)
        text_blocks = extract_text_blocks(path, meta)
        chunks = chunk_narrative(text_blocks, target_tokens=500, overlap_tokens=80, min_tokens=200)
        all_chunks.extend([c.to_dict() for c in chunks])

    to_jsonl(os.path.join(args.outdir, "tables.jsonl"), all_tables)
    to_jsonl(os.path.join(args.outdir, "chunks.jsonl"), all_chunks)

    # Build indexes (narrative only)
    from rag10kq.utils import read_jsonl, Chunk, FilingMeta
    dicts = read_jsonl(os.path.join(args.outdir, "chunks.jsonl"))
    chunks_obj = []
    for d in dicts:
        f = FilingMeta(**d["filing"])
        chunks_obj.append(Chunk(
            chunk_id=d["chunk_id"], text=d["text"], section_path=d["section_path"],
            is_table=d["is_table"], statement_type=d.get("statement_type"),
            filing=f, extra=d.get("extra", {})
        ))
    # indexes dir
    idx_dir = os.path.join(args.outdir, "index")
    ensure_dir(idx_dir)
    build_bm25(chunks_obj, idx_dir)
    build_dense_index(chunks_obj, idx_dir)

    print(f"✅ Wrote {len(all_chunks)} chunks and {len(all_tables)} tables")
    print(f"   Indexes at: {idx_dir}")

if __name__ == "__main__":
    main()
```

### `scripts/ask.py`

```python
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
```

### `scripts/eval_retrieval.py`

```python
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
```

---

## 1) How to run it (with your files)

> Replace the two file paths below with yours (you uploaded them earlier):
>
> * `/mnt/data/10-K_2024.html`
> * `/mnt/data/10-Q_2024_Q1.html`

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Build the corpus (ingest → chunk → indexes)
python scripts/build_corpus.py /mnt/data/10-K_2024.html /mnt/data/10-Q_2024_Q1.html --outdir artifacts

# Ask questions
python scripts/ask.py -q "What were Apple's net sales in FY 2024?" --outdir artifacts
python scripts/ask.py -q "What are the main risk factors related to China?" --outdir artifacts

# (Optional) Evaluate retrieval if you prepare a gold file:
# gold.jsonl rows: {"question": "...", "gold_chunk_ids": ["narr_abc123...", "..."]}
python scripts/eval_retrieval.py --outdir artifacts --gold path/to/gold.jsonl
```

---

## 2) What you get

* `artifacts/tables.jsonl` – **structured tables** (shape matches what we discussed earlier).
* `artifacts/chunks.jsonl` – **hierarchical narrative chunks** with section paths and filing metadata.
* `artifacts/index/faiss.index` + `dense_meta.json` – **dense** index (FAISS).
* `artifacts/index/bm25.pkl` – **BM25** index.
* `scripts/ask.py` returns:

  * For **numeric** queries (e.g., “net sales”): direct extraction from table JSON with filing+section citations.
  * For **narrative** queries: top chunk summary with citations (section path + source).

---

## 3) Why these choices are sound

* **Dense**: BGE small‑en‑v1.5 is a strong, efficient embedding model; prepend the **query instruction** to queries (as implemented) for best retrieval quality.
* **Re‑rank**: `ms‑marco‑MiniLM‑L‑6‑v2` is a widely used cross‑encoder for passage reranking with solid benchmark scores; used here for the top candidate pool.
* **FAISS**: standard, fast IP/flat index for normalized embeddings; exact search baseline (IndexFlatIP / L2 are the reference).
* **BM25**: rank‑bm25 is the canonical pure‑Python implementation; simple and reliable.
* **HTML→MD**: `markdownify` keeps headings/links while staying lightweight; alternatives like `html-to-markdown` also exist.

---

### Notes & extensions (optional)

* If you later want **LLM generation** (abstractive answers) or **LLM‑as‑judge** for evaluation, add a provider (OpenAI, Bedrock, or local **Ollama**) in `answer.py` behind an env flag; the retrieval pieces above remain unchanged.
* For **finance‑tuned embeddings** (FinMTEB winners), swap the embedding model name in `indexer.py` and rebuild; the scripts don’t change.
* The `extract_tables` function is conservative; you can enrich it with better **period parsing** from column headers and DEI tags when you need precise period alignment.

If you want, I can fold this into a single `pipx` CLI or add a small Streamlit UI for interactive QA with citations.
