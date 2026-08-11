from rag.chunking.semantic import chunk_semantic_document
from rag.ingestion import Document


class FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = {
            "# Launch": [1.0, 0.0],
            "The product launch begins in June.": [1.0, 0.0],
            "The launch team will train distributors.": [0.9, 0.1],
            "Bronze material cost increased sharply.": [0.0, 1.0],
        }
        return [vectors[text] for text in texts]


def test_semantic_chunking_splits_when_neighbouring_meaning_changes() -> None:
    document = Document(
        content=(
            "# Launch\n\nThe product launch begins in June. "
            "The launch team will train distributors. "
            "Bronze material cost increased sharply."
        ),
        metadata={"document_key": "data/example.md"},
    )

    chunks = chunk_semantic_document(
        document,
        embedder=FakeEmbedder(),
        chunk_size=200,
        min_chunk_size=1,
        similarity_threshold=0.8,
    )

    assert [chunk.text for chunk in chunks] == [
        "# Launch\n\nThe product launch begins in June.\n\nThe launch team will train distributors.",
        "Bronze material cost increased sharply.",
    ]
    assert all(chunk.metadata["chunking_strategy"] == "semantic-v1" for chunk in chunks)


def test_semantic_chunking_rejects_invalid_configuration() -> None:
    document = Document(content="One sentence.", metadata={"document_key": "data/example.md"})

    try:
        chunk_semantic_document(document, embedder=FakeEmbedder(), min_chunk_size=501)
    except ValueError as error:
        assert str(error) == "min_chunk_size must be between 1 and chunk_size"
    else:
        raise AssertionError("Expected invalid min_chunk_size to fail")
