"""Embedding-based chunking that creates boundaries at topic changes."""

from __future__ import annotations

import math
import re
from typing import Any, Protocol

from rag.chunking.fixed_size import Chunk
from rag.ingestion import Document


_HEADING = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9*#])")


class Embedder(Protocol):
    """The small interface semantic chunking needs from an embedding model."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def chunk_semantic_document(
    document: Document,
    *,
    embedder: Embedder,
    chunk_size: int = 500,
    min_chunk_size: int = 200,
    similarity_threshold: float = 0.75,
) -> list[Chunk]:
    """Group nearby sentences until their meaning changes substantially.

    The threshold applies to cosine similarity between neighbouring sentence
    embeddings. A lower score means their meanings are less alike and starts a
    new chunk, provided the current chunk has reached ``min_chunk_size``.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if min_chunk_size < 1 or min_chunk_size > chunk_size:
        raise ValueError("min_chunk_size must be between 1 and chunk_size")
    if not -1 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be between -1 and 1")

    units = _semantic_units(document.content)
    if not units:
        return []
    embeddings = embedder.embed(units)
    if len(embeddings) != len(units):
        raise RuntimeError("Embedding model returned the wrong number of vectors")

    pieces = _pack_units(
        units,
        embeddings,
        chunk_size=chunk_size,
        min_chunk_size=min_chunk_size,
        similarity_threshold=similarity_threshold,
    )
    return _chunks_from_pieces(document.content, document.metadata, pieces)


def chunk_semantic_corpus(
    documents: list[Document], *, embedder: Embedder, chunk_size: int = 500,
    min_chunk_size: int = 200, similarity_threshold: float = 0.75,
) -> list[Chunk]:
    """Chunk every document with the same semantic configuration."""
    return [
        chunk
        for document in documents
        for chunk in chunk_semantic_document(
            document,
            embedder=embedder,
            chunk_size=chunk_size,
            min_chunk_size=min_chunk_size,
            similarity_threshold=similarity_threshold,
        )
    ]


def _semantic_units(text: str) -> list[str]:
    """Split into headings and sentences without cutting Markdown table rows."""
    units: list[str] = []
    matches = list(_HEADING.finditer(text))
    starts = [0, *(match.start() for match in matches), len(text)]
    for index in range(len(starts) - 1):
        section = text[starts[index] : starts[index + 1]].strip()
        if not section:
            continue
        lines = section.splitlines()
        if lines and _HEADING.fullmatch(lines[0]):
            units.append(lines.pop(0).strip())
        body = "\n".join(lines).strip()
        for paragraph in re.split(r"\n\s*\n", body):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if "|" in paragraph or paragraph.startswith(("- ", "* ", "1. ")):
                units.append(paragraph)
            else:
                units.extend(piece.strip() for piece in _SENTENCE.split(paragraph) if piece.strip())
    return units


def _pack_units(
    units: list[str],
    embeddings: list[list[float]],
    *,
    chunk_size: int,
    min_chunk_size: int,
    similarity_threshold: float,
) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    previous_embedding: list[float] | None = None

    for unit, embedding in zip(units, embeddings, strict=True):
        candidate = "\n\n".join([*current, unit])
        topic_changed = (
            current
            and not _is_heading(unit)
            and previous_embedding is not None
            and _cosine_similarity(previous_embedding, embedding) < similarity_threshold
        )
        if current and (len(candidate) > chunk_size or (topic_changed and len("\n\n".join(current)) >= min_chunk_size)):
            chunks.append("\n\n".join(current))
            current = []
        current.append(unit)
        previous_embedding = embedding

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _is_heading(text: str) -> bool:
    return bool(_HEADING.fullmatch(text))


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding vectors must have the same dimension")
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if denominator == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / denominator


def _chunks_from_pieces(
    content: str, metadata: dict[str, Any], pieces: list[str]
) -> list[Chunk]:
    document_key = metadata["document_key"]
    chunks: list[Chunk] = []
    search_start = 0
    for chunk_index, text in enumerate(pieces):
        start = content.find(text, search_start)
        if start < 0:
            start = search_start
        end = start + len(text)
        search_start = end
        chunks.append(
            Chunk(
                id=f"{document_key}:{chunk_index:04d}",
                text=text,
                metadata={
                    **metadata,
                    "chunk_index": chunk_index,
                    "character_start": start,
                    "character_end": end,
                    "chunking_strategy": "semantic-v1",
                },
            )
        )
    return chunks
