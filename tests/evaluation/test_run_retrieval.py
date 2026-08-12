from rag.evaluation import GoldQuestion
from pathlib import Path

from rag.evaluation.run_retrieval import default_output_path, evaluate_retrieval
from rag.retrieval import RetrievedPassage


class FakeRetriever:
    def retrieve(self, question: str, *, top_k: int) -> list[RetrievedPassage]:
        assert top_k == 5
        if question == "Answerable question":
            return [
                RetrievedPassage(
                    chunk_id="data/policies/current.md:0000",
                    text="Current policy",
                    score=0.9,
                    metadata={"source_path": "data/policies/current.md"},
                )
            ]
        return []


def test_evaluate_retrieval_writes_one_trace_per_question_and_averages_scores() -> None:
    questions = [
        GoldQuestion(
            id="Q001",
            question="Answerable question",
            expected_answer="Current policy",
            source_files=("data/policies/current.md",),
            concept="baseline",
            difficulty="easy",
            answerable=True,
        ),
        GoldQuestion(
            id="Q002",
            question="Unanswerable question",
            expected_answer="Not in the corpus",
            source_files=(),
            concept="refusal",
            difficulty="easy",
            answerable=False,
        ),
    ]

    evaluation = evaluate_retrieval(
        questions,
        retriever=FakeRetriever(),
        top_k=5,
        pipeline_config={"retrieval_strategy": "fake"},
    )

    assert [trace.question_id for trace in evaluation.traces] == ["Q001", "Q002"]
    assert evaluation.traces[0].pipeline_config == {
        "retrieval_strategy": "fake",
        "top_k": 5,
    }
    assert evaluation.traces[0].retrieved_chunks[0].source_path == "data/policies/current.md"
    assert evaluation.summary.question_count == 2
    assert evaluation.summary.answerable_question_count == 1
    assert evaluation.summary.error_count == 0
    assert evaluation.summary.mean_precision_at_k == 0.2
    assert evaluation.summary.mean_recall_at_k == 1.0
    assert evaluation.summary.mean_reciprocal_rank == 1.0
    assert evaluation.summary.mean_ndcg_at_k == 1.0


def test_default_output_path_keeps_retrieval_experiments_separate() -> None:
    assert default_output_path("semantic") == Path("results/retrieval-baseline.jsonl")
    assert default_output_path("keyword") == Path("results/retrieval-keyword.jsonl")
    assert default_output_path("hybrid") == Path("results/retrieval-hybrid.jsonl")
    assert default_output_path("hybrid-decomposed-multihop") == Path(
        "results/retrieval-hybrid-decomposed-multihop.jsonl"
    )
