# Semantic benchmark v1 changeset

All changes relative to master `21a792c9653e662964fee8fb550eba80fa0ad5de` are new files. No existing production, historical dataset/artifact, dependency, or lockfile is modified. The baseline remains observational and service-affected; see the [report and executed commands](../../artifacts/evals/semantic_answer/v1/baselines/3793929f343fb9fc93c4d5a83cd0b2f16fe6e6cb/REPORT.md).

Diff scope: independent source-first gold builders and frozen data; an evaluation-only scorer, runner and verifier; metric/publication tests; blind source audits; immutable answer/judge evidence and publication hash lineage. The publication-only helper and documentation added after the implementation freeze do not alter its recorded source hashes or scores.

The full diff is available in the PR Files changed view, or locally:

```sh
git diff 21a792c9653e662964fee8fb550eba80fa0ad5de..HEAD
```

## Files added

```text
artifacts/evals/semantic_answer/v1/baselines/3793929f343fb9fc93c4d5a83cd0b2f16fe6e6cb/REPORT.md
artifacts/evals/semantic_answer/v1/baselines/3793929f343fb9fc93c4d5a83cd0b2f16fe6e6cb/answer_manifest.json
artifacts/evals/semantic_answer/v1/baselines/3793929f343fb9fc93c4d5a83cd0b2f16fe6e6cb/deterministic.jsonl
artifacts/evals/semantic_answer/v1/baselines/3793929f343fb9fc93c4d5a83cd0b2f16fe6e6cb/deterministic_summary.json
artifacts/evals/semantic_answer/v1/baselines/3793929f343fb9fc93c4d5a83cd0b2f16fe6e6cb/judge/judgments.jsonl
artifacts/evals/semantic_answer/v1/baselines/3793929f343fb9fc93c4d5a83cd0b2f16fe6e6cb/judge/manifest.json
artifacts/evals/semantic_answer/v1/baselines/3793929f343fb9fc93c4d5a83cd0b2f16fe6e6cb/judge/started.json
artifacts/evals/semantic_answer/v1/baselines/3793929f343fb9fc93c4d5a83cd0b2f16fe6e6cb/judge/summary.json
artifacts/evals/semantic_answer/v1/baselines/3793929f343fb9fc93c4d5a83cd0b2f16fe6e6cb/publication_lineage.json
artifacts/evals/semantic_answer/v1/baselines/3793929f343fb9fc93c4d5a83cd0b2f16fe6e6cb/publication_manifest.json
artifacts/evals/semantic_answer/v1/baselines/3793929f343fb9fc93c4d5a83cd0b2f16fe6e6cb/raw_answers.jsonl
artifacts/evals/semantic_answer/v1/baselines/3793929f343fb9fc93c4d5a83cd0b2f16fe6e6cb/runtime_observation.json
artifacts/evals/semantic_answer/v1/baselines/3793929f343fb9fc93c4d5a83cd0b2f16fe6e6cb/source_audit.jsonl
artifacts/evals/semantic_answer/v1/baselines/3793929f343fb9fc93c4d5a83cd0b2f16fe6e6cb/started.json
artifacts/evals/semantic_answer/v1/baselines/3793929f343fb9fc93c4d5a83cd0b2f16fe6e6cb/system_observation_late.json
artifacts/evals/semantic_answer/v1/baselines/3793929f343fb9fc93c4d5a83cd0b2f16fe6e6cb/tests.txt
artifacts/evals/semantic_answer/v1/baselines/3793929f343fb9fc93c4d5a83cd0b2f16fe6e6cb/verification-publication.json
artifacts/evals/semantic_answer/v1/baselines/3793929f343fb9fc93c4d5a83cd0b2f16fe6e6cb/verification.json
data/evals/semantic_answer/v1/ANNOTATION.md
data/evals/semantic_answer/v1/draft_counts.json
data/evals/semantic_answer/v1/evaluation_config.json
data/evals/semantic_answer/v1/judge_rubric.txt
data/evals/semantic_answer/v1/manifest.json
data/evals/semantic_answer/v1/numeric_source_facts.jsonl
data/evals/semantic_answer/v1/queries.jsonl
data/evals/semantic_answer/v1/source_manifest.json
data/evals/semantic_answer/v1/tables/AAPL_10-K_2024.tables.jsonl
data/evals/semantic_answer/v1/tables/AAPL_10-K_2025.tables.jsonl
data/evals/semantic_answer/v1/tables/AMZN_10-K_2023.tables.jsonl
data/evals/semantic_answer/v1/tables/AMZN_10-K_2024.tables.jsonl
data/evals/semantic_answer/v1/tables/MSFT_10-K_2024.tables.jsonl
data/evals/semantic_answer/v1/tables/MSFT_10-K_2025.tables.jsonl
docs/evals/semantic_answer_benchmark_v1.md
docs/evals/semantic_answer_changes_v1.md
scripts/evals/agents/build_semantic_dataset_v1.py
scripts/evals/agents/build_semantic_sources_v1.py
scripts/evals/agents/freeze_semantic_v1.py
scripts/evals/agents/run_semantic_v1.py
scripts/evals/agents/semantic_annotations_v1.py
scripts/evals/agents/verify_semantic_v1.py
scripts/evals/publish_semantic_evidence.py
src/evals/semantic_answer_v1.py
tests/evals/test_semantic_answer_v1.py
tests/evals/test_semantic_publication.py
```
