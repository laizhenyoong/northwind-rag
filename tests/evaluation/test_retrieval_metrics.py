import math

import pytest

from rag.evaluation import GoldQuestion, RetrievedChunk, score_retrieval


def test_score_retrieval_uses_unique_source_documents() -> None:
    question = GoldQuestion(
        id="Q001",
        question="What is the domestic per diem rate?",
        expected_answer="RM 180 per day.",
        source_files=("data/policies/travel-expense-v2.1.md",),
        concept="version-conflict",
        difficulty="medium",
        answerable=True,
    )
    chunks = (
        RetrievedChunk(
            chunk_id="travel-v1:0001",
            source_path="data/policies/travel-expense-v1.4.md",
            rank=1,
            score=0.91,
        ),
        RetrievedChunk(
            chunk_id="travel-v2:0001",
            source_path="data/policies/travel-expense-v2.1.md",
            rank=2,
            score=0.87,
        ),
        RetrievedChunk(
            chunk_id="travel-v2:0002",
            source_path="data/policies/travel-expense-v2.1.md",
            rank=3,
            score=0.82,
        ),
    )

    metrics = score_retrieval(question, chunks, k=5)

    assert metrics is not None
    assert metrics.precision_at_k == pytest.approx(0.2)
    assert metrics.recall_at_k == pytest.approx(1.0)
    assert metrics.reciprocal_rank == pytest.approx(0.5)
    assert metrics.ndcg_at_k == pytest.approx(1 / math.log2(3))


def test_score_retrieval_skips_unanswerable_questions() -> None:
    question = GoldQuestion(
        id="Q079",
        question="How many units did the company sell?",
        expected_answer="Not stated in the corpus.",
        source_files=(),
        concept="unanswerable-refusal",
        difficulty="medium",
        answerable=False,
    )

    assert score_retrieval(question, (), k=5) is None
