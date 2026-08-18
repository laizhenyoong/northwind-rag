from pathlib import Path

import pytest

from rag.retrieval.parent import ParentDocumentRetriever
from rag.retrieval.semantic import RetrievedPassage


def passage(chunk_id: str, source_path: str, *, score: float = 1.0, index: int = 0):
    return RetrievedPassage(
        chunk_id=chunk_id,
        text=f"text of {chunk_id}",
        score=score,
        metadata={"source_path": source_path, "chunk_index": index},
    )


class FakeRetriever:
    def __init__(self, passages):
        self.passages = passages
        self.calls = []

    def retrieve(self, question, *, top_k=5, metadata_filter=None):
        self.calls.append((question, top_k, metadata_filter))
        return list(self.passages)


def test_a_ranked_chunk_is_replaced_by_the_document_it_came_from() -> None:
    inner = FakeRetriever([passage("doc.md:0002", "doc.md")])
    retriever = ParentDocumentRetriever(inner, {"doc.md": "the whole document"})

    results = retriever.retrieve("question")

    assert [r.text for r in results] == ["the whole document"]
    assert results[0].chunk_id == "doc.md"
    assert results[0].metadata["expanded_from"] == "doc.md:0002"


def test_two_chunks_from_one_document_send_that_document_once() -> None:
    """Without this the model reads the same file twice and pays for it twice."""
    inner = FakeRetriever(
        [passage("doc.md:0002", "doc.md"), passage("doc.md:0007", "doc.md", index=7)]
    )
    retriever = ParentDocumentRetriever(inner, {"doc.md": "the whole document"})

    assert len(retriever.retrieve("question")) == 1


def test_documents_keep_the_rank_order_their_chunks_earned() -> None:
    inner = FakeRetriever(
        [passage("a.md:0000", "a.md", score=0.9), passage("b.md:0000", "b.md", score=0.4)]
    )
    retriever = ParentDocumentRetriever(inner, {"a.md": "A", "b.md": "B"})

    assert [r.text for r in retriever.retrieve("question")] == ["A", "B"]


def test_an_oversized_document_falls_back_to_the_chunk() -> None:
    """A long document is several unrelated topics, not one coherent passage."""
    inner = FakeRetriever([passage("big.md:0003", "big.md")])
    retriever = ParentDocumentRetriever(
        inner, {"big.md": "x" * 5000}, max_document_characters=1000
    )

    results = retriever.retrieve("question")

    assert results[0].text == "text of big.md:0003"


def test_a_chunk_with_no_known_document_is_returned_unchanged() -> None:
    inner = FakeRetriever([passage("missing.md:0000", "missing.md")])
    retriever = ParentDocumentRetriever(inner, {})

    assert retriever.retrieve("question")[0].chunk_id == "missing.md:0000"


def test_the_parent_passage_drops_the_chunk_position_it_no_longer_has() -> None:
    """Neighbor expansion keys off chunk_index, and the parent already holds them."""
    inner = FakeRetriever([passage("doc.md:0002", "doc.md", index=2)])
    retriever = ParentDocumentRetriever(inner, {"doc.md": "whole"})

    assert "chunk_index" not in retriever.retrieve("question")[0].metadata


def test_from_corpus_loads_real_documents_by_their_recorded_path() -> None:
    inner = FakeRetriever([])
    retriever = ParentDocumentRetriever.from_corpus(inner, Path("data"))

    assert "data/directory/staff-directory.md" in retriever.documents
    assert "Marine product application" in retriever.documents[
        "data/directory/staff-directory.md"
    ]
