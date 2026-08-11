from rag.chunking import chunk_document
from rag.ingestion import Document


def document(content: str) -> Document:
    return Document(content=content, metadata={"document_key": "data/example.md"})


def test_markdown_chunking_keeps_a_sentence_intact() -> None:
    chunks = chunk_document(
        document(
            "# Supplier change\n\n"
            "The incumbent supplier lead time has extended to 14 weeks. "
            "Tanaka will reduce it to 6 weeks."
        ),
        chunk_size=60,
    )

    assert [chunk.text for chunk in chunks] == [
        "# Supplier change",
        "The incumbent supplier lead time has extended to 14 weeks.",
        "Tanaka will reduce it to 6 weeks.",
    ]
    assert all(chunk.metadata["chunking_strategy"] == "markdown-aware-v1" for chunk in chunks)


def test_markdown_chunking_keeps_small_section_together() -> None:
    chunks = chunk_document(
        document("# Rate\n\nThe rate is RM 180. It applies to domestic travel."),
        chunk_size=100,
    )

    assert [chunk.text for chunk in chunks] == [
        "# Rate\n\nThe rate is RM 180. It applies to domestic travel."
    ]


def test_markdown_chunking_does_not_split_numbered_headings_as_sentences() -> None:
    chunks = chunk_document(
        document("## 2. Seal Supply\n\nThe supplier lead time is 14 weeks."),
        chunk_size=20,
    )

    assert chunks[0].text == "## 2. Seal Supply"
