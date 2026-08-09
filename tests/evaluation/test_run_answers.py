from rag.evaluation import GoldQuestion
from rag.evaluation.run_answers import run_answer_evaluation
from rag.generation.grounded import GroundedAnswer
from rag.retrieval import RetrievedPassage


class FakeRetriever:
    def retrieve(self, question: str, *, top_k: int) -> list[RetrievedPassage]:
        return [
            RetrievedPassage(
                chunk_id="policy.md:0000",
                text="The policy is current.",
                score=0.9,
                metadata={"source_path": "data/policy.md"},
            )
        ]


class FakeAnswerer:
    def answer(self, question: str, passages: list[RetrievedPassage]) -> GroundedAnswer:
        return GroundedAnswer("The policy is current. [S1]", "[S1] policy", ("policy.md:0000",))


def test_run_answer_evaluation_keeps_every_generated_trace() -> None:
    questions = [
        GoldQuestion(
            id="Q001",
            question="What is current?",
            expected_answer="The policy",
            source_files=("data/policy.md",),
            concept="baseline",
            difficulty="easy",
            answerable=True,
        )
    ]

    result = run_answer_evaluation(
        questions,
        retriever=FakeRetriever(),
        answerer=FakeAnswerer(),
        top_k=5,
        pipeline_config={"generation_model": "fake"},
    )

    assert result.traces[0].answer == "The policy is current. [S1]"
    assert result.evaluation.summary.expected_source_citation_rate == 1.0
