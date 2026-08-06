"""Divide source documents into retrievable text chunks."""

from rag.chunking.fixed_size import Chunk, chunk_corpus, chunk_document

__all__ = ["Chunk", "chunk_corpus", "chunk_document"]
