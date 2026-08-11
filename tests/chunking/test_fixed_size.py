from rag.chunking.fixed_size import chunk_document
from rag.ingestion import Document


def test_chunk_document_splits_text_and_copies_metadata() -> None:
    document = Document(
        content="abcdefghijkl",
        metadata={"document_key": "data/example.md", "status": "current"},
    )

    chunks = chunk_document(document, chunk_size=5)

    assert [chunk.id for chunk in chunks] == [
        "data/example.md:0000",
        "data/example.md:0001",
        "data/example.md:0002",
    ]
    assert [chunk.text for chunk in chunks] == ["abcde", "fghij", "kl"]
    assert chunks[1].metadata == {
        "document_key": "data/example.md",
        "status": "current",
        "chunk_index": 1,
        "character_start": 5,
        "character_end": 10,
    }
