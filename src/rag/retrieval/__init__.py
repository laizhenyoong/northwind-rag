"""Retrieve relevant source passages for a user question."""

from typing import Any

__all__ = ["HybridRetriever", "RetrievedPassage", "SemanticRetriever", "build_context"]


def __getattr__(name: str) -> Any:
    """Avoid importing the executable module before ``python -m`` runs it."""
    if name in __all__:
        from rag.retrieval.hybrid import HybridRetriever
        from rag.retrieval.semantic import RetrievedPassage, SemanticRetriever, build_context

        return {
            "HybridRetriever": HybridRetriever,
            "RetrievedPassage": RetrievedPassage,
            "SemanticRetriever": SemanticRetriever,
            "build_context": build_context,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
