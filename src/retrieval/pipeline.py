from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from qdrant_client import QdrantClient, models

from retrieval.rerank_enricher import enrich_candidates_with_table_summaries
from retrieval.evaluator import (
    dedupe_scored_points,
    get_bge_m3_model,
    get_bge_reranker_large_model,
    multi_query_hybrid_search_bge_m3,
    normalize_doc_id_to_table,
    rerank_with_bge_reranker_large,
)


@dataclass(frozen=True)
class ExpansionConfig:
    model: str = "qwen3:4b-instruct"
    include_original: bool = True
    dedupe: bool = True
    max_expansions: Optional[int] = 3
    options: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class RetrievalConfig:
    bge_m3_model: str = "BAAI/bge-m3"
    use_fp16: bool = False
    collection_name: str = "sec_docs"
    top_k: int = 50
    using_bge_dense: str = "bge_m3_dense"
    using_bge_sparse: str = "bge_m3_sparse"
    prefetch_k: Optional[int] = None
    bge_sparse_top_k: int = 256
    doc_types: Optional[List[str]] = None
    max_workers: Optional[int] = None
    fuse: bool = True
    rrf_k: int = 60
    weights: Optional[Sequence[float]] = None


@dataclass(frozen=True)
class DedupeConfig:
    max_candidates: int = 20


@dataclass(frozen=True)
class RerankConfig:
    model_name: str = "BAAI/bge-reranker-v2-m3"
    use_fp16: bool = False
    top_k: int = 10
    max_passage_chars: int = 2000
    enrich_batch_size: int = 256
    table_content_field: str = "content"
    content_field: str = "content"
    joiner: str = "\n\n"


@dataclass(frozen=True)
class LexicalScoringConfig:
    tables_dir: str = "../data/chunked"


@dataclass(frozen=True)
class GenerationConfig:
    model: str = "deepseek-r1:14b"
    num_predct: int = 1024
    temperature: float = 0.1
    options: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class PipelineConfig:
    allowed_line_items: Sequence[str] | str | None = None
    expansion: ExpansionConfig = field(default_factory=ExpansionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    dedupe: DedupeConfig = field(default_factory=DedupeConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    lexical_scoring: LexicalScoringConfig = field(default_factory=LexicalScoringConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)


@dataclass(frozen=True)
class PipelineResult:
    answer: str
    selected_tables: List[Dict[str, Any]]
    queries: List[str]
    rerank_query: str
    fused_candidates: List[models.ScoredPoint]
    reranked_candidates: List[models.ScoredPoint]


def _point_doc_id(point: models.ScoredPoint) -> str:
    md = point.payload or {}
    return str(md.get("doc_id") or md.get("id") or "")


class FinanceRAGPipeline:
    def __init__(self, client: QdrantClient, config: PipelineConfig):
        self.client = client
        self.config = config
        self.bge_model = get_bge_m3_model(
            model_name=config.retrieval.bge_m3_model,
            use_fp16=config.retrieval.use_fp16,
        )
        self.reranker_model = get_bge_reranker_large_model(
            model_name=config.rerank.model_name,
            use_fp16=config.rerank.use_fp16,
        )

    def run_hybrid_search_pipeline(
        self,
        queries: Sequence[str],
        *,
        ticker: str,
        fiscal_year: int,
        form_type: Optional[str] = None,
        doc_types: Optional[List[str]] = None,
    ) -> Tuple[str, List[models.ScoredPoint], List[models.ScoredPoint]]:
        retrieval_doc_types = doc_types if doc_types is not None else self.config.retrieval.doc_types
        retrieval_result = multi_query_hybrid_search_bge_m3(
            queries,
            client=self.client,
            collection_name=self.config.retrieval.collection_name,
            top_k=self.config.retrieval.top_k,
            using_bge_dense=self.config.retrieval.using_bge_dense,
            using_bge_sparse=self.config.retrieval.using_bge_sparse,
            prefetch_k=self.config.retrieval.prefetch_k,
            bge_sparse_top_k=self.config.retrieval.bge_sparse_top_k,
            doc_types=retrieval_doc_types,
            ticker=ticker,
            fiscal_year=fiscal_year,
            form_type=form_type,
            bge_model=self.bge_model,
            max_workers=self.config.retrieval.max_workers,
            fuse=self.config.retrieval.fuse,
            rrf_k=self.config.retrieval.rrf_k,
            weights=self.config.retrieval.weights,
        )

        if self.config.retrieval.fuse:
            fused_candidates, _hits_by_query = retrieval_result  # type: ignore[assignment]
        else:
            hits_by_query = retrieval_result  # type: ignore[assignment]
            fused_candidates = [p for pts in hits_by_query.values() for p in pts]

        deduped = dedupe_scored_points(
            fused_candidates,
            key_fn=lambda p: normalize_doc_id_to_table(_point_doc_id(p)),
        )
        if self.config.dedupe.max_candidates is not None:
            deduped = deduped[: self.config.dedupe.max_candidates]

        enriched = enrich_candidates_with_table_summaries(
            deduped,
            client=self.client,
            collection_name=self.config.retrieval.collection_name,
            batch_size=self.config.rerank.enrich_batch_size,
            table_content_field=self.config.rerank.table_content_field,
            content_field=self.config.rerank.content_field,
            joiner=self.config.rerank.joiner,
        )

        if queries:
            if len(queries) > 1:
                rerank_query = f"{queries[0]} ({', '.join(queries[1:])})"
            else:
                rerank_query = str(queries[0])
        else:
            rerank_query = ""

        reranked = rerank_with_bge_reranker_large(
            rerank_query,
            enriched,
            top_k=self.config.rerank.top_k,
            max_passage_chars=self.config.rerank.max_passage_chars,
            model=self.reranker_model,
        )

        return rerank_query, fused_candidates, reranked
