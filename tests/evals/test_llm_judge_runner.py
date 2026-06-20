from evals.llm_judge_contracts import LLMJudgeDimensionScores, LLMJudgeEvidenceChunkIds, LLMJudgeModelOutput
from evals.llm_judge_runner import (
    _build_evidence_items,
    _build_joined_rows,
    _derive_verdict,
    _normalize_model_output,
)


def test_build_joined_rows_matches_by_index_then_prompt() -> None:
    gold_rows = [
        {"prompt": "Prompt A", "gold_answer": "Gold A"},
        {"prompt": "Prompt B", "gold_answer": "Gold B"},
    ]
    rag_rows = [
        {"prompt_index": 1, "prompt": "Prompt A", "final_answer": "Answer A"},
        {"prompt": "Prompt B", "final_answer": "Answer B"},
    ]

    joined, errors = _build_joined_rows(gold_rows, rag_rows)

    assert errors == []
    assert len(joined) == 2
    assert joined[0][0] == 1
    assert joined[0][2]["final_answer"] == "Answer A"
    assert joined[1][0] == 2
    assert joined[1][2]["final_answer"] == "Answer B"


def test_build_evidence_items_extracts_text_and_table_chunk_ids() -> None:
    rag_row = {
        "retrieved_chunks": [
            {"doc_id": "AAPL_10-K_2025::table::2"},
            {"doc_id": "AAPL_10-K_2025::text::33"},
        ],
        "context_items": [
            {
                "kind": "table",
                "source": {
                    "doc_id": "AAPL_10-K_2025::table::2",
                    "section_path": "Section A",
                    "table_id": "2",
                },
                "payload": {"table_markdown": "table body"},
            },
            {
                "kind": "text",
                "source": {
                    "doc_id": "AAPL_10-K_2025::text::33",
                    "section_path": "Section B",
                },
                "payload": {"content": "text body"},
            },
        ],
    }

    items, evidence_ids, retrieved_doc_ids = _build_evidence_items(
        rag_row,
        max_evidence_chunks=5,
        max_chars_per_chunk=1000,
    )

    assert len(items) == 2
    assert evidence_ids == LLMJudgeEvidenceChunkIds(text=[33], tables=[2])
    assert retrieved_doc_ids == ["AAPL_10-K_2025::table::2", "AAPL_10-K_2025::text::33"]


def test_normalize_model_output_answer_only_nulls_grounding() -> None:
    model_output = LLMJudgeModelOutput(
        verdict="correct",
        score=99,
        dimension_scores=LLMJudgeDimensionScores(
            correctness=4,
            completeness=3,
            grounding=2,
            inference_handling=1,
        ),
    )

    score, score_max, verdict = _normalize_model_output("answer_only", model_output)

    assert score == 8
    assert score_max == 8
    assert verdict == "correct"
    assert model_output.dimension_scores.grounding is None


def test_derive_verdict_respects_score_ratio() -> None:
    dims = LLMJudgeDimensionScores(correctness=2, completeness=1, grounding=0, inference_handling=0)
    assert _derive_verdict(3, 10, dims) == "incorrect"
    assert _derive_verdict(5, 8, dims) == "partially_correct"
