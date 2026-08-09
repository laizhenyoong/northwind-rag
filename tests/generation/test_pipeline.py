from rag.generation.grounded import GroundedAnswer
from rag.generation.pipeline import answer_question
from rag.retrieval.semantic import RetrievedPassage


class FakeRetriever:
    def retrieve(self, question: str, *, top_k: int) -> list[RetrievedPassage]:
        return [
            RetrievedPassage(
                chunk_id="policy.md:0000",
                text="The domestic per diem is RM 180.",
                score=0.9,
                metadata={"source_path": "data/policy.md"},
            )
        ]


class FakeAnswerer:
    def answer(self, question: str, passages: list[RetrievedPassage]) -> GroundedAnswer:
        return GroundedAnswer(
            text="The domestic per diem is RM 180. [S1]",
            context="[S1] policy context",
            cited_chunk_ids=("policy.md:0000",),
        )


def test_answer_pipeline_stores_the_retrieval_context_and_answer() -> None:
    run = answer_question(
        question_id="Q001",
        question="What is the domestic rate?",
        retriever=FakeRetriever(),
        answerer=FakeAnswerer(),
        top_k=5,
        pipeline_config={"generation_model": "gemma4"},
    )

    assert run.trace.question_id == "Q001"
    assert run.trace.answer == "The domestic per diem is RM 180. [S1]"
    assert run.trace.context_sent_to_model == "[S1] policy context"
    assert run.trace.retrieved_chunks[0].chunk_id == "policy.md:0000"
    assert run.trace.error is None
