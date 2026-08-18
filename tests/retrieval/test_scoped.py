import pytest

from rag.retrieval.scoped import DocumentScopedRetriever
from rag.retrieval.semantic import RetrievedPassage


def passage(chunk_id: str, source_path: str, *, score: float = 1.0):
    return RetrievedPassage(
        chunk_id=chunk_id,
        text=f"text of {chunk_id}",
        score=score,
        metadata={"source_path": source_path},
    )


class RecordingRetriever:
    """Returns a different result per call, so the two passes stay separable."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def retrieve(self, question, *, top_k=5, metadata_filter=None):
        self.calls.append({"top_k": top_k, "filter": metadata_filter})
        return list(self.responses[min(len(self.calls) - 1, len(self.responses) - 1)])


def test_the_second_pass_is_restricted_to_the_leading_documents() -> None:
    inner = RecordingRetriever(
        [passage("a.md:0000", "a.md"), passage("b.md:0000", "b.md")],
        [passage("a.md:0007", "a.md")],
    )
    retriever = DocumentScopedRetriever(inner, document_k=2, candidate_k=20)

    retriever.retrieve("question", top_k=5)

    assert inner.calls[0]["filter"] is None
    assert inner.calls[1]["filter"] == {
        "$or": [
            {"source_path": {"$eq": "a.md"}},
            {"source_path": {"$eq": "b.md"}},
        ]
    }


def test_the_second_pass_can_surface_a_chunk_the_first_pass_ranked_too_low() -> None:
    """The whole point: a row that loses corpus-wide wins inside its own document."""
    inner = RecordingRetriever(
        [passage("a.md:0000", "a.md")],
        [passage("a.md:0000", "a.md"), passage("a.md:0007", "a.md")],
    )
    retriever = DocumentScopedRetriever(inner, document_k=1)

    results = retriever.retrieve("question")

    assert [r.chunk_id for r in results] == ["a.md:0000", "a.md:0007"]


def test_document_k_limits_how_many_documents_stay_in_scope() -> None:
    inner = RecordingRetriever(
        [passage("a.md:0", "a.md"), passage("b.md:0", "b.md"), passage("c.md:0", "c.md")],
        [passage("a.md:1", "a.md")],
    )
    retriever = DocumentScopedRetriever(inner, document_k=1)

    retriever.retrieve("question")

    assert inner.calls[1]["filter"] == {"$or": [{"source_path": {"$eq": "a.md"}}]}


def test_a_caller_filter_is_kept_alongside_the_document_scope() -> None:
    """Dropping it would let a version filter silently stop applying."""
    inner = RecordingRetriever([passage("a.md:0", "a.md")], [passage("a.md:1", "a.md")])
    retriever = DocumentScopedRetriever(inner, document_k=1)

    retriever.retrieve("question", metadata_filter={"status": {"$eq": "current"}})

    assert inner.calls[1]["filter"] == {
        "$and": [
            {"status": {"$eq": "current"}},
            {"$or": [{"source_path": {"$eq": "a.md"}}]},
        ]
    }


def test_an_empty_first_pass_returns_nothing_rather_than_searching_everything() -> None:
    inner = RecordingRetriever([], [passage("a.md:0", "a.md")])
    retriever = DocumentScopedRetriever(inner)

    assert retriever.retrieve("question") == []
    assert len(inner.calls) == 1


def test_an_empty_second_pass_falls_back_to_the_first(monkeypatch) -> None:
    inner = RecordingRetriever([passage("a.md:0", "a.md")], [])
    retriever = DocumentScopedRetriever(inner, document_k=1)

    assert [r.chunk_id for r in retriever.retrieve("question")] == ["a.md:0"]


def test_document_k_must_be_at_least_one() -> None:
    inner = RecordingRetriever([passage("a.md:0", "a.md")])

    with pytest.raises(ValueError, match="document_k"):
        DocumentScopedRetriever(inner, document_k=0).retrieve("question")
