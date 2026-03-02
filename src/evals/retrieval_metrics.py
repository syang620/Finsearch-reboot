from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Set


def hit_at_k(relevant_flags: Sequence[bool], k: int) -> float:
    if k <= 0:
        return 0.0
    return 1.0 if any(bool(x) for x in relevant_flags[:k]) else 0.0


def mrr_at_k(relevant_flags: Sequence[bool], k: int) -> float:
    if k <= 0:
        return 0.0
    for idx, flag in enumerate(relevant_flags[:k]):
        if flag:
            return 1.0 / float(idx + 1)
    return 0.0


def recall_at_k(
    retrieved_ids: Sequence[Optional[int]],
    relevant_ids: Iterable[int],
    k: int,
) -> float:
    if k <= 0:
        return 0.0

    relevant_set: Set[int] = {int(x) for x in relevant_ids}
    if not relevant_set:
        return 0.0

    hits: Set[int] = set()
    for rid in retrieved_ids[:k]:
        if rid is None:
            continue
        if rid in relevant_set:
            hits.add(rid)

    return float(len(hits)) / float(len(relevant_set))


def ndcg_at_k(relevant_flags: Sequence[bool], k: int) -> float:
    if k <= 0:
        return 0.0

    rel = [1.0 if bool(v) else 0.0 for v in relevant_flags[:k]]
    if not rel:
        return 0.0

    dcg = 0.0
    for i, r in enumerate(rel, start=1):
        if r <= 0.0:
            continue
        dcg += r / math.log2(i + 1)

    ideal = sorted(rel, reverse=True)
    idcg = 0.0
    for i, r in enumerate(ideal, start=1):
        if r <= 0.0:
            continue
        idcg += r / math.log2(i + 1)

    if idcg <= 0.0:
        return 0.0
    return dcg / idcg
