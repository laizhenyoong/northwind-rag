from rag.evaluation import GoldQuestion, RetrievedChunk, RunTrace
from rag.evaluation.analyze_retrieval import analyze_retrieval, format_report


def test_analyze_retrieval_groups_hits_and_misses_by_concept() -> None:
    questions = [
        GoldQuestion(
            id="Q001",
            question="First?",
            expected_answer="A",
            source_files=("data/current.md",),
            concept="version-conflict",
            difficulty="easy",
            answerable=True,
        ),
        GoldQuestion(
            id="Q002",
            question="Second?",
            expected_answer="B",
            source_files=("data/other.md",),
            concept="version-conflict",
            difficulty="easy",
            answerable=True,
        ),
    ]
    traces = [
        RunTrace(
            question_id="Q001",
            question="First?",
            pipeline_config={},
            retrieved_chunks=(RetrievedChunk("a", "data/current.md", 1, 0.9),),
        ),
        RunTrace(
            question_id="Q002",
            question="Second?",
            pipeline_config={},
            retrieved_chunks=(RetrievedChunk("b", "data/wrong.md", 1, 0.8),),
        ),
    ]

    summaries, outcomes = analyze_retrieval(questions, traces, top_k=5)

    assert summaries[0].concept == "version-conflict"
    assert summaries[0].hit_at_1 == 1
    assert summaries[0].hit_at_k == 1
    assert summaries[0].missing_from_top_k == 1
    assert outcomes[1].first_relevant_rank is None
    assert "Hit@5" in format_report(summaries, top_k=5)
