from rag.constants import REFUSAL
from rag.evaluation import (
    GoldQuestion,
    RetrievedChunk,
    RunTrace,
    SemanticAnswerJudge,
    evaluate_semantic_answers,
)


class FakeChatModel:
    def __init__(self, response: str) -> None:
        self.response = response

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        return self.response


def question(*, question_id: str | None = None, answerable: bool = True) -> GoldQuestion:
    return GoldQuestion(
        id=question_id or ("Q001" if answerable else "Q079"),
        question="What is the current rate?",
        expected_answer="RM 180 under policy v2.1.",
        source_files=("data/policies/current.md",),
        concept="version-conflict",
        difficulty="medium",
        answerable=answerable,
    )


def trace(
    *,
    question_id: str = "Q001",
    answer: str = "RM 180. [S1]",
    error: str | None = None,
) -> RunTrace:
    return RunTrace(
        question_id=question_id,
        question="What is the current rate?",
        pipeline_config={},
        retrieved_chunks=(RetrievedChunk("current.md:0000", "data/policies/current.md", 1, 0.9),),
        context_sent_to_model="[S1] RM 180 under policy v2.1.",
        answer=answer,
        error=error,
    )


def test_semantic_judge_parses_a_structured_verdict() -> None:
    judge = SemanticAnswerJudge(
        FakeChatModel(
            '{"correct": true, "citation_supported": true, '
            '"used_correct_version": true, "proper_refusal": false, "reason": "Matches v2.1."}'
        )
    )

    judgement = judge.judge(question(), trace())

    assert judgement.correct is True
    assert judgement.citation_supported is True
    assert judgement.used_correct_version is True
    assert judgement.proper_refusal is False
    assert judgement.error is None


def test_semantic_evaluation_separates_answer_quality_and_refusal_quality() -> None:
    judge = SemanticAnswerJudge(
        FakeChatModel(
            '{"correct": true, "citation_supported": true, '
            '"used_correct_version": true, "proper_refusal": true, "reason": "Valid refusal."}'
        )
    )

    evaluation = evaluate_semantic_answers(
        [question(), question(answerable=False)],
        # Worded unlike REFUSAL on purpose, so the refusal still reaches the judge
        # rather than the deterministic bypass this test is not about.
        [trace(), trace(question_id="Q079", answer="That is not covered by the provided context.")],
        judge=judge,
    )

    assert evaluation.summary.correctness_rate == 1.0
    assert evaluation.summary.citation_supported_rate == 1.0
    assert evaluation.summary.correct_version_rate == 1.0
    assert evaluation.summary.proper_refusal_rate == 1.0


def test_semantic_judge_keeps_a_bad_model_response_as_an_inspectable_error() -> None:
    judgement = SemanticAnswerJudge(FakeChatModel("Not JSON")).judge(question(), trace())

    assert judgement.correct is False
    assert judgement.error == "Judge did not return a JSON object"
    assert judgement.failure_stage == "judge"


def test_unanswerable_exact_refusal_bypasses_an_inconsistent_llm_judge() -> None:
    judgement = SemanticAnswerJudge(FakeChatModel("Not JSON")).judge(
        question(answerable=False),
        trace(question_id="Q079", answer=REFUSAL),
    )

    assert judgement.correct is True
    assert judgement.proper_refusal is True
    assert judgement.error is None
    assert judgement.reason.startswith("Deterministic pass")


def test_semantic_summary_excludes_judge_failures_from_quality_denominators() -> None:
    failed = trace(question_id="Q002", error="RuntimeError: answer model unavailable")
    evaluation = evaluate_semantic_answers(
        [question(), question(question_id="Q002")],
        [trace(), failed],
        judge=SemanticAnswerJudge(
            FakeChatModel(
                '{"correct": true, "citation_supported": true, '
                '"used_correct_version": true, "proper_refusal": false, "reason": "Matches."}'
            )
        ),
    )

    assert evaluation.summary.error_count == 1
    assert evaluation.summary.pipeline_error_count == 1
    assert evaluation.summary.judge_error_count == 0
    assert evaluation.summary.judged_answerable_question_count == 1
    assert evaluation.summary.correctness_rate == 1.0


def test_judge_prompt_rejects_conflicting_historical_values() -> None:
    model = FakeChatModel(
        '{"correct": false, "citation_supported": true, '
        '"used_correct_version": false, "proper_refusal": false, '
        '"reason": "It also gives the superseded RM 150 rate."}'
    )

    judgement = SemanticAnswerJudge(model).judge(
        question(), trace(answer="The current rate is RM 180, but RM 150 also applies. [S1]")
    )

    assert judgement.correct is False
    assert judgement.used_correct_version is False
