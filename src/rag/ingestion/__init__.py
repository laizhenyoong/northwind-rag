"""Convert source files into structured RAG documents."""

from rag.ingestion.markdown import Document, load_corpus, load_markdown_document

__all__ = ["Document", "load_corpus", "load_markdown_document"]
