import pytest

from rag.evaluation import GoldQuestion, RetrievedChunk, RunTrace, SemanticAnswerJudge
from rag.evaluation.rejudge import rejudge_traces


VERDICT = (
    '{"correct": true, "citation_supported": true, "used_correct_version": true, '
    '"proper_refusal": false, "reason": "Matches the gold answer."}'
)


class FakeChatModel:
    def __init__(self, response: str = VERDICT) -> None:
        self.response = response
        self.calls = 0

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        return self.response


def question(question_id: str) -> GoldQuestion:
    return GoldQuestion(
        id=question_id,
        question="What is the current rate?",
        expected_answer="RM 180 under policy v2.1.",
        source_files=("data/policies/current.md",),
        concept="version-conflict",
        difficulty="medium",
        answerable=True,
    )


def trace(question_id: str) -> RunTrace:
    return RunTrace(
        question_id=question_id,
        question="What is the current rate?",
        pipeline_config={},
        retrieved_chunks=(RetrievedChunk("current.md:0000", "data/policies/current.md", 1, 0.9),),
        context_sent_to_model="[S1] RM 180 under policy v2.1.",
        answer="RM 180. [S1]",
    )


def test_rejudging_scores_every_trace_in_the_file() -> None:
    chat_model = FakeChatModel()

    evaluation = rejudge_traces(
        [question("Q001"), question("Q002")],
        [trace("Q001"), trace("Q002")],
        judge=SemanticAnswerJudge(chat_model),
    )

    assert chat_model.calls == 2
    assert evaluation.summary.correctness_rate == 1.0


def test_a_limited_trace_file_narrows_the_questions_instead_of_failing() -> None:
    """A --limit run saves fewer traces than there are gold questions."""
    evaluation = rejudge_traces(
        [question("Q001"), question("Q002"), question("Q003")],
        [trace("Q002")],
        judge=SemanticAnswerJudge(FakeChatModel()),
    )

    assert evaluation.summary.question_count == 1
    assert evaluation.judgements[0].question_id == "Q002"


def test_traces_matching_no_gold_question_are_refused() -> None:
    with pytest.raises(ValueError, match="No gold question"):
        rejudge_traces(
            [question("Q001")],
            [trace("Q999")],
            judge=SemanticAnswerJudge(FakeChatModel()),
        )
