# Retrieval Evaluation v0

This evaluator measures retrieval quality only (no planner/analyst scoring).
It supports both table-label and text-label datasets.

For controlled comparisons, `--retrieval-mode` selects one of:

- `bm25_only`: BM25 candidate retrieval; dense and reranking disabled.
- `dense_only`: dense candidate retrieval; BM25 and reranking disabled.
- `hybrid`: dense and BM25 retrieval with RRF; reranking disabled.
- `hybrid_reranker`: the current production dense + BM25 + RRF + Qwen3 reranker stack (default).

## What it computes

Deterministic metrics (primary):

- `hit@k`
- `recall@k`
- `mrr@k`
- `ndcg@k`

Optional RAGAS context metrics (secondary):

- context precision
- context recall

## Inputs

Default table eval set:

- `data/evals/retrieval/table/table_eval_v1.jsonl`

Text eval set:

- `data/evals/retrieval/text/aapl_2024_10k_text_retrieval_eval_split_with_uids.jsonl`

Expected row shape (simplified):

```json
{
  "query_id": 1,
  "query": "...",
  "gold_answer": "...",
  "relevant_tables": [{"table_index": 12}]
}
```

`table_eval_v1` does not include ticker/year per row, so CLI defaults are used unless row-level overrides are present.

## Run

```bash
PYTHONPATH=src python scripts/evals/retrieval/eval_retrieval_v0.py \
  --eval-mode table \
  --eval-path data/evals/retrieval/table/table_eval_v1.jsonl \
  --top-k 10 \
  --k-values 1,3,5,10 \
  --out-dir artifacts/evals/retrieval/v0
```

Text retrieval eval:

```bash
PYTHONPATH=src python scripts/evals/retrieval/eval_retrieval_v0.py \
  --eval-mode text \
  --eval-path data/evals/retrieval/text/aapl_2024_10k_text_retrieval_eval_split_with_uids.jsonl \
  --doc-types text_chunk \
  --top-k 10 \
  --k-values 1,3,5,10 \
  --out-dir artifacts/evals/retrieval/v0
```

The frozen text dataset is not compatible with the PR2 Qdrant collection used by
the 2026-08-31 ablation baseline. Its split IDs and `chunk_uid` labels come from a
different chunk lineage, while that collection does not preserve `chunk_uid`.
That baseline therefore excludes text metrics rather than remapping labels.

## Controlled ablation

The frozen definition is
`data/evals/retrieval/retrieval_ablation_v1.json`. Run all enabled datasets and
all four modes with:

```bash
QDRANT_COLLECTION_NAME=<FROZEN_COLLECTION> \
PYTHONPATH=src <PYTHON> scripts/evals/retrieval/run_retrieval_ablation_v1.py \
  --evaluated-sha <IMPLEMENTATION_SHA> \
  --out-root artifacts/evals/retrieval/ablations/<IMPLEMENTATION_SHA> \
  --collection <FROZEN_COLLECTION> \
  --expected-points <POINT_COUNT> \
  --expected-qdrant-version <QDRANT_VERSION> \
  --expected-embedding-digest <OLLAMA_MODEL_DIGEST>
```

The runner refuses existing output, verifies dataset and corpus identity before
and after the run, uses a fresh embedding cache per mode, disables RAGAS, checks
component traces, and writes `comparison.json` plus `manifest.sha256`.

Disable RAGAS if Ollama judge is unavailable:

```bash
PYTHONPATH=src python scripts/evals/retrieval/eval_retrieval_v0.py \
  --disable-ragas
```

## Outputs

- `artifacts/evals/retrieval/v0/per_query.jsonl`
- `artifacts/evals/retrieval/v0/summary.json`
- `artifacts/evals/retrieval/v0/errors.jsonl`

## Environment

Retrieval backend:

- `QDRANT_HOST` (default `localhost`)
- `QDRANT_PORT` (default `6333`)
- `QDRANT_COLLECTION_NAME` (default `sec_docs_dense_bm25`)
- `TABLES_DIR` (default from retrieval tool)

RAGAS/Ollama backend:

- `RAGAS_OLLAMA_BASE_URL` (default `http://localhost:11434`)
- `RAGAS_JUDGE_MODEL` (default `llama3.1:8b-instruct`)
- `RAGAS_EMBED_MODEL` (default `nomic-embed-text`)

## Notes

- Deterministic metrics are the release gate for v0.
- RAGAS metrics are supplemental and may vary by local judge model quality.
