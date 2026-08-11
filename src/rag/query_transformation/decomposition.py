"""Fuse retrieval results from an original question and focused sub-queries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from rag.retrieval.semantic import RetrievedPassage


class CandidateRetriever(Protocol):
    """Retrieve a wide candidate list for each transformed query."""

    def retrieve(
        self,
        question: str,
        *,
        top_k: int,
        metadata_filter: Mapping[str, object] | None = None,
    ) -> list[RetrievedPassage]: ...


class QueryDecomposer(Protocol):
    """Turn one complex question into focused retrieval sub-queries."""

    def decompose(self, question: str) -> list[str]: ...


@dataclass(slots=True)
class QueryDecompositionRetriever:
    """Retrieve for multiple focused queries and fuse their candidate rankings."""

    candidate_retriever: CandidateRetriever
    decomposer: QueryDecomposer
    candidate_k: int = 20
    rrf_k: int = 60
    coverage_per_query: int = 0
    last_queries: tuple[str, ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        if self.candidate_k < 1:
            raise ValueError("candidate_k must be at least 1")
        if self.rrf_k < 0:
            raise ValueError("rrf_k must be at least 0")
        if self.coverage_per_query < 0:
            raise ValueError("coverage_per_query must be at least 0")

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 5,
        metadata_filter: Mapping[str, object] | None = None,
    ) -> list[RetrievedPassage]:
        """Fuse original and decomposed-query rankings without raw-score mixing."""
        if not question.strip():
            raise ValueError("question must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if top_k > self.candidate_k:
            raise ValueError("top_k cannot be greater than candidate_k")

        self.last_queries = (question,)
        subqueries = self.decomposer.decompose(question)
        queries = tuple(dict.fromkeys((question, *subqueries)))
        self.last_queries = queries

        fused_scores: dict[str, float] = {}
        passages_by_id: dict[str, RetrievedPassage] = {}
        passages_by_query: list[list[RetrievedPassage]] = []
        for query in queries:
            passages = self.candidate_retriever.retrieve(
                query, top_k=self.candidate_k, metadata_filter=metadata_filter
            )
            passages_by_query.append(passages)
            for rank, passage in enumerate(passages, start=1):
                fused_scores[passage.chunk_id] = fused_scores.get(passage.chunk_id, 0) + 1 / (
                    self.rrf_k + rank
                )
                passages_by_id.setdefault(passage.chunk_id, passage)

        fused_passages = [
            RetrievedPassage(
                chunk_id=chunk_id,
                text=passages_by_id[chunk_id].text,
                score=score,
                metadata=passages_by_id[chunk_id].metadata,
            )
            for chunk_id, score in sorted(
                fused_scores.items(), key=lambda item: (-item[1], item[0])
            )[:top_k]
        ]
        return _with_evidence_coverage(
            fused_passages,
            passages_by_query[1:],
            top_k=top_k,
            coverage_per_query=self.coverage_per_query,
        )


def _with_evidence_coverage(
    fused_passages: list[RetrievedPassage],
    subquery_passages: list[list[RetrievedPassage]],
    *,
    top_k: int,
    coverage_per_query: int,
) -> list[RetrievedPassage]:
    """Reserve evidence for each sub-query, then fill remaining RRF positions."""
    if coverage_per_query == 0 or not subquery_passages:
        return fused_passages

    selected: list[RetrievedPassage] = []
    selected_ids: set[str] = set()
    for passages in subquery_passages:
        added = 0
        for passage in passages:
            if passage.chunk_id not in selected_ids:
                selected.append(passage)
                selected_ids.add(passage.chunk_id)
                added += 1
            if added == coverage_per_query or len(selected) == top_k:
                break
        if len(selected) == top_k:
            return selected

    for passage in fused_passages:
        if passage.chunk_id not in selected_ids:
            selected.append(passage)
            selected_ids.add(passage.chunk_id)
        if len(selected) == top_k:
            break
    return selected
