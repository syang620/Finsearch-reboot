# Retrieval Evaluation v0

This evaluator measures retrieval quality only (no planner/analyst scoring).
It supports both table-label and text-label datasets.

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
PYTHONPATH=src python scripts/evals/eval_retrieval_v0.py \
  --eval-mode table \
  --eval-path data/evals/retrieval/table/table_eval_v1.jsonl \
  --top-k 10 \
  --k-values 1,3,5,10 \
  --out-dir artifacts/evals/retrieval/v0
```

Text retrieval eval:

```bash
PYTHONPATH=src python scripts/evals/eval_retrieval_v0.py \
  --eval-mode text \
  --eval-path data/evals/retrieval/text/aapl_2024_10k_text_retrieval_eval_split_with_uids.jsonl \
  --doc-types text_chunk \
  --top-k 10 \
  --k-values 1,3,5,10 \
  --out-dir artifacts/evals/retrieval/v0
```

Disable RAGAS if Ollama judge is unavailable:

```bash
PYTHONPATH=src python scripts/evals/eval_retrieval_v0.py \
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
