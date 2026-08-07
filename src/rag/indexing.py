"""Build Pinecone records from the local Markdown corpus."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol

from rag.chunking import Chunk, chunk_corpus
from rag.embeddings import OllamaEmbedder
from rag.ingestion import load_corpus
from rag.vector_store import PineconeSettings, ensure_index


class Embedder(Protocol):
    """The small interface the indexing pipeline needs from an embedding model."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class VectorIndex(Protocol):
    """The small interface the indexing pipeline needs from a vector database."""

    def upsert(self, *, vectors: list[dict[str, Any]], namespace: str) -> Any: ...


@dataclass(frozen=True, slots=True)
class IndexingResult:
    """A short, non-secret record of one indexing run."""

    document_count: int
    chunk_count: int


def index_corpus(
    corpus_root: Path,
    *,
    embedder: Embedder,
    vector_index: VectorIndex,
    chunk_size: int = 500,
    batch_size: int = 32,
    namespace: str = "",
) -> IndexingResult:
    """Embed every chunk and upsert it under its stable chunk ID.

    Re-running with the same corpus and chunking settings overwrites matching
    IDs, so it is safe to retry an interrupted upload.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    documents = load_corpus(corpus_root)
    chunks = chunk_corpus(documents, chunk_size=chunk_size)

    for batch in _batches(chunks, batch_size):
        embeddings = embedder.embed([chunk.text for chunk in batch])
        if len(embeddings) != len(batch):
            raise RuntimeError("Embedding model returned the wrong number of vectors")

        vector_index.upsert(
            vectors=[
                _pinecone_record(chunk, embedding)
                for chunk, embedding in zip(batch, embeddings, strict=True)
            ],
            namespace=namespace,
        )

    return IndexingResult(document_count=len(documents), chunk_count=len(chunks))


def _pinecone_record(chunk: Chunk, embedding: list[float]) -> dict[str, Any]:
    """Turn our Chunk into Pinecone's ID, vector, metadata record shape."""
    return {
        "id": chunk.id,
        "values": embedding,
        # Retrieval needs the actual source passage, not only its vector.
        "metadata": {"text": chunk.text, **_metadata_for_pinecone(chunk.metadata)},
    }


def _metadata_for_pinecone(metadata: dict[str, Any]) -> dict[str, str | int | float | bool | list[str]]:
    """Convert YAML values, especially dates, to Pinecone metadata values."""
    normalized: dict[str, str | int | float | bool | list[str]] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, datetime | date):
            normalized[key] = value.isoformat()
        elif isinstance(value, str | int | float | bool):
            normalized[key] = value
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            normalized[key] = value
        else:
            normalized[key] = json.dumps(value, default=str, sort_keys=True)

    # Pinecone compares ranges as numbers, not dates. Keep the readable ISO date
    # above for inspection and add an ordinal for reliable "as of" filters.
    effective_date = metadata.get("effective_date")
    if isinstance(effective_date, datetime | date):
        normalized["effective_date_ordinal"] = effective_date.toordinal()
        expiry_date = metadata.get("expiry_date")
        normalized["expiry_date_ordinal"] = (
            expiry_date.toordinal()
            if isinstance(expiry_date, datetime | date)
            else date.max.toordinal()
        )

    # doc_id is the stable family identifier shared by document revisions.
    if isinstance(metadata.get("doc_id"), str):
        normalized["document_family"] = metadata["doc_id"]
    if isinstance(metadata.get("version"), str):
        normalized["is_current_version"] = metadata.get("status") == "current"
    return normalized


def _batches(items: Sequence[Chunk], size: int) -> Iterable[Sequence[Chunk]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main() -> None:
    """Run the baseline indexer from the project root."""
    parser = argparse.ArgumentParser(description="Embed the corpus and upload it to Pinecone")
    parser.add_argument("--corpus-root", type=Path, default=Path("data"))
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--namespace", default="")
    arguments = parser.parse_args()

    settings = PineconeSettings.from_environment()
    result = index_corpus(
        arguments.corpus_root,
        embedder=OllamaEmbedder(),
        vector_index=ensure_index(settings),
        chunk_size=arguments.chunk_size,
        batch_size=arguments.batch_size,
        namespace=arguments.namespace,
    )
    print(f"Indexed {result.chunk_count} chunks from {result.document_count} documents.")


if __name__ == "__main__":
    main()
