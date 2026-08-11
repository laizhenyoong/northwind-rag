"""Markdown-aware chunking that preserves sections and sentences."""

from __future__ import annotations

import re
from typing import Any

from rag.chunking.fixed_size import Chunk
from rag.ingestion import Document


_HEADING = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9*#])")


def chunk_markdown_document(document: Document, *, chunk_size: int = 500) -> list[Chunk]:
    """Split Markdown at section, paragraph, then sentence boundaries.

    ``chunk_size`` is a target rather than a hard character cut: a single
    sentence is never truncated just to meet it.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")

    pieces = _pack_blocks(_markdown_blocks(document.content), chunk_size)
    return _chunks_from_pieces(document.content, document.metadata, pieces)


def chunk_markdown_corpus(
    documents: list[Document], *, chunk_size: int = 500
) -> list[Chunk]:
    """Chunk every Markdown document using the document-aware strategy."""
    return [
        chunk
        for document in documents
        for chunk in chunk_markdown_document(document, chunk_size=chunk_size)
    ]


def _markdown_blocks(text: str) -> list[str]:
    """Return section-aware blocks, retaining each heading with its content."""
    matches = list(_HEADING.finditer(text))
    if not matches:
        return _paragraphs(text)

    blocks = _paragraphs(text[: matches[0].start()])
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section = text[match.start() : end].strip()
        if section:
            blocks.append(section)
    return blocks


def _paragraphs(text: str) -> list[str]:
    return [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]


def _pack_blocks(blocks: list[str], chunk_size: int) -> list[str]:
    """Greedily pack semantic blocks; divide only oversized blocks further."""
    chunks: list[str] = []
    current = ""
    for block in blocks:
        for piece in _split_oversized_block(block, chunk_size):
            candidate = f"{current}\n\n{piece}" if current else piece
            if current and len(candidate) > chunk_size:
                chunks.append(current)
                current = piece
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks


def _split_oversized_block(block: str, chunk_size: int) -> list[str]:
    """Split a large section by paragraphs and sentences, never raw characters."""
    if len(block) <= chunk_size:
        return [block]

    parts: list[str] = []
    paragraphs = _paragraphs(block)
    if paragraphs and _HEADING.fullmatch(paragraphs[0]):
        # A numbered heading (for example, ``## 2. Seal Supply``) contains a
        # full stop but is not a sentence.  Keep it whole and attach it to the
        # following paragraph during the normal packing step.
        parts.append(paragraphs.pop(0))
    for paragraph in paragraphs:
        sentences = _SENTENCE_BOUNDARY.split(paragraph)
        parts.extend(sentence.strip() for sentence in sentences if sentence.strip())
    return parts or [block]


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
                    "chunking_strategy": "markdown-aware-v1",
                },
            )
        )
    return chunks
