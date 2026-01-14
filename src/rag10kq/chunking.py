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
