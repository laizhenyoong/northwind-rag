"""Divide source documents into retrievable text chunks."""

from rag.chunking.fixed_size import Chunk
from rag.chunking.markdown import chunk_markdown_corpus as chunk_corpus
from rag.chunking.markdown import chunk_markdown_document as chunk_document

__all__ = ["Chunk", "chunk_corpus", "chunk_document"]
