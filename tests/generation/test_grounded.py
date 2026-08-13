import pytest

from rag.generation.grounded import GroundedAnswerer, GroundingError, REFUSAL
from rag.retrieval.semantic import RetrievedPassage


class FakeChatModel:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.response


def passage(chunk_id: str = "policy.md:0000") -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=chunk_id,
        text="The domestic per diem is RM 180.",
        score=0.9,
        metadata={"source_path": "data/policy.md"},
    )


def test_grounded_answer_keeps_only_citations_to_retrieved_chunks() -> None:
    model = FakeChatModel("The domestic per diem is RM 180. [S1]")

    answer = GroundedAnswerer(model).answer("What is the rate?", [passage()])

    assert answer.text == "The domestic per diem is RM 180. [S1]"
    assert answer.cited_chunk_ids == ("policy.md:0000",)
    assert "[S1]" in answer.context
    assert "data/policy.md" in model.calls[0][1]


def test_grounded_answerer_prompt_instructs_version_aware_answers() -> None:
    model = FakeChatModel("The rate is RM 180. [S1]")

    GroundedAnswerer(model).answer("What is the current rate?", [passage()])

    system_prompt, _ = model.calls[0]
    assert "multiple document versions" in system_prompt
    assert "superseded values unless" in system_prompt


def test_grounded_answer_rejects_missing_or_unknown_citations() -> None:
    with pytest.raises(GroundingError, match="no context citations"):
        GroundedAnswerer(FakeChatModel("The rate is RM 180.")).answer("What is the rate?", [passage()])

    with pytest.raises(GroundingError, match="not provided: S2"):
        GroundedAnswerer(FakeChatModel("The rate is RM 180. [S2]")).answer(
            "What is the rate?", [passage()]
        )


def test_grounded_answer_refuses_without_context_without_calling_the_model() -> None:
    model = FakeChatModel("Should not be used")

    answer = GroundedAnswerer(model).answer("Unknown question", [])

    assert answer.text == REFUSAL
    assert answer.cited_chunk_ids == ()
    assert model.calls == []
