"""Search once to choose documents, then search again inside only those."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from rag.retrieval.semantic import RetrievedPassage


class CandidateRetriever(Protocol):
    """The ranking step this retriever runs twice."""

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 5,
        metadata_filter: Mapping[str, object] | None = None,
    ) -> list[RetrievedPassage]: ...


@dataclass(frozen=True, slots=True)
class DocumentScopedRetriever:
    """Narrow to the best few documents, then rank within them.

    A chunk's rank depends on its competition, not only on itself. A row that
    loses against the whole corpus can win easily against its own document's
    other rows, so a second pass restricted to the chosen documents surfaces
    material the first pass ranked too low to return.
    """

    candidate_retriever: CandidateRetriever
    document_k: int = 3
    candidate_k: int = 20

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 5,
        metadata_filter: Mapping[str, object] | None = None,
    ) -> list[RetrievedPassage]:
        """Return the best chunks from the documents the first pass chose."""
        if self.document_k < 1:
            raise ValueError("document_k must be at least 1")

        first_pass = self.candidate_retriever.retrieve(
            question, top_k=self.candidate_k, metadata_filter=metadata_filter
        )
        source_paths = _leading_source_paths(first_pass, self.document_k)
        if not source_paths:
            return first_pass[:top_k]

        second_pass = self.candidate_retriever.retrieve(
            question,
            top_k=top_k,
            metadata_filter=_scoped_filter(source_paths, metadata_filter),
        )
        return second_pass or first_pass[:top_k]


def _leading_source_paths(
    passages: list[RetrievedPassage], document_k: int
) -> list[str]:
    """Take the documents behind the highest ranked chunks, in rank order."""
    source_paths: list[str] = []
    for passage in passages:
        source_path = passage.metadata.get("source_path")
        if isinstance(source_path, str) and source_path not in source_paths:
            source_paths.append(source_path)
            if len(source_paths) == document_k:
                break
    return source_paths


def _scoped_filter(
    source_paths: list[str], metadata_filter: Mapping[str, object] | None
) -> dict[str, object]:
    """Restrict to the chosen documents without discarding the caller's filter."""
    scope: dict[str, object] = {
        "$or": [{"source_path": {"$eq": source_path}} for source_path in source_paths]
    }
    if metadata_filter is None:
        return scope
    return {"$and": [dict(metadata_filter), scope]}
