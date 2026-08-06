"""The deliberately simple fixed-size chunking baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag.ingestion import Document


@dataclass(frozen=True, slots=True)
class Chunk:
    """A searchable slice of a source document."""

    id: str
    text: str
    metadata: dict[str, Any]


def chunk_document(document: Document, *, chunk_size: int = 500) -> list[Chunk]:
    """Split one document into fixed-length character slices without overlap."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")

    document_key = document.metadata["document_key"]
    chunks = []

    for chunk_index, start in enumerate(range(0, len(document.content), chunk_size)):
        end = start + chunk_size
        metadata = {
            **document.metadata,
            "chunk_index": chunk_index,
            "character_start": start,
            "character_end": min(end, len(document.content)),
        }
        chunks.append(
            Chunk(
                id=f"{document_key}:{chunk_index:04d}",
                text=document.content[start:end],
                metadata=metadata,
            )
        )

    return chunks


def chunk_corpus(documents: list[Document], *, chunk_size: int = 500) -> list[Chunk]:
    """Chunk every document in the corpus using the same baseline setting."""
    return [
        chunk
        for document in documents
        for chunk in chunk_document(document, chunk_size=chunk_size)
    ]
