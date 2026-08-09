import pytest

from rag.evaluation import GoldQuestion, RetrievedChunk, RunTrace, evaluate_answers, score_answer


def question(*, answerable: bool = True) -> GoldQuestion:
    return GoldQuestion(
        id="Q001" if answerable else "Q079",
        question="What is the policy?",
        expected_answer="A supported answer",
        source_files=("data/policies/current.md",) if answerable else (),
        concept="baseline",
        difficulty="easy",
        answerable=answerable,
    )


def trace(*, question_id: str = "Q001", answer: str | None, error: str | None = None) -> RunTrace:
    return RunTrace(
        question_id=question_id,
        question="What is the policy?",
        pipeline_config={},
        retrieved_chunks=(
            RetrievedChunk("current.md:0000", "data/policies/current.md", 1, 0.9),
            RetrievedChunk("old.md:0000", "data/policies/old.md", 2, 0.8),
        ),
        answer=answer,
        error=error,
    )


def test_score_answer_checks_citations_against_the_retrieval_trace() -> None:
    metrics = score_answer(question(), trace(answer="The policy is current. [S1]"))

    assert metrics.produced_answer is True
    assert metrics.citations_valid is True
    assert metrics.cites_expected_source is True


def test_score_answer_marks_unknown_or_missing_citations_as_invalid() -> None:
    unknown = score_answer(question(), trace(answer="The policy is current. [S3]"))
    missing = score_answer(question(), trace(answer="The policy is current."))

    assert unknown.citations_valid is False
    assert missing.citations_valid is False
    assert missing.cites_expected_source is False


def test_evaluate_answers_separates_answerable_citation_and_refusal_rates() -> None:
    answerable = question()
    unanswerable = question(answerable=False)
    evaluation = evaluate_answers(
        [answerable, unanswerable],
        [
            trace(answer="The policy is current. [S1]"),
            trace(
                question_id="Q079",
                answer="I don't know based on the provided context.",
            ),
        ],
    )

    assert evaluation.summary.answer_rate == 1.0
    assert evaluation.summary.citation_valid_rate == 1.0
    assert evaluation.summary.expected_source_citation_rate == 1.0
    assert evaluation.summary.refusal_rate == 1.0


def test_score_answer_rejects_a_question_and_trace_id_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match"):
        score_answer(question(), trace(question_id="Q999", answer="Answer [S1]"))
