from rag.query_transformation.decomposition import QueryDecompositionRetriever
from rag.retrieval import RetrievedPassage


class FakeDecomposer:
    def decompose(self, question: str) -> list[str]:
        return ["supplier", "lead time"]


class FakeRetriever:
    def __init__(self) -> None:
        self.calls = []

    def retrieve(self, question, *, top_k, metadata_filter=None):
        self.calls.append((question, top_k, metadata_filter))
        results = {
            "original": [passage("supplier-doc")],
            "supplier": [passage("supplier-doc"), passage("contract")],
            "lead time": [passage("lead-time-doc"), passage("supplier-doc")],
        }
        return results[question]


def passage(chunk_id: str) -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=chunk_id,
        text=chunk_id,
        score=0.1,
        metadata={"source_path": f"data/{chunk_id}.md"},
    )


def test_decomposition_fuses_original_and_subquery_rankings() -> None:
    candidate_retriever = FakeRetriever()
    retriever = QueryDecompositionRetriever(candidate_retriever, FakeDecomposer(), candidate_k=2)
    metadata_filter = {"is_current_version": {"$eq": True}}

    passages = retriever.retrieve("original", top_k=2, metadata_filter=metadata_filter)

    assert [passage.chunk_id for passage in passages] == ["supplier-doc", "lead-time-doc"]
    assert retriever.last_queries == ("original", "supplier", "lead time")
    assert candidate_retriever.calls == [
        ("original", 2, metadata_filter),
        ("supplier", 2, metadata_filter),
        ("lead time", 2, metadata_filter),
    ]


def test_decomposition_can_reserve_evidence_for_each_subquery() -> None:
    candidate_retriever = FakeRetriever()
    retriever = QueryDecompositionRetriever(
        candidate_retriever,
        FakeDecomposer(),
        candidate_k=2,
        coverage_per_query=1,
    )

    passages = retriever.retrieve("original", top_k=2)

    assert [passage.chunk_id for passage in passages] == ["supplier-doc", "lead-time-doc"]
