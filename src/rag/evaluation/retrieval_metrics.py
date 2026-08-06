"""Measure whether retrieval returned the expected source documents."""

from __future__ import annotations

import math
from dataclasses import dataclass

from rag.evaluation.questions import GoldQuestion
from rag.evaluation.traces import RetrievedChunk


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Document-level retrieval metrics for one answerable question."""

    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float


def score_retrieval(
    question: GoldQuestion,
    retrieved_chunks: tuple[RetrievedChunk, ...],
    *,
    k: int,
) -> RetrievalMetrics | None:
    """Score retrieved source paths against the question's known sources.

    Unanswerable questions have no relevant source document, so retrieval
    metrics do not apply to them. Their refusal quality is scored separately.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if not question.answerable:
        return None

    expected_sources = set(question.source_files)
    ranked_sources = _unique_source_paths(retrieved_chunks)[:k]
    relevant_ranks = [
        rank
        for rank, source_path in enumerate(ranked_sources, start=1)
        if source_path in expected_sources
    ]

    precision_at_k = len(relevant_ranks) / k
    recall_at_k = len(relevant_ranks) / len(expected_sources)
    reciprocal_rank = 1 / relevant_ranks[0] if relevant_ranks else 0.0

    dcg = sum(1 / math.log2(rank + 1) for rank in relevant_ranks)
    ideal_hits = min(k, len(expected_sources))
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return RetrievalMetrics(
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        reciprocal_rank=reciprocal_rank,
        ndcg_at_k=dcg / ideal_dcg,
    )


def _unique_source_paths(chunks: tuple[RetrievedChunk, ...]) -> list[str]:
    """Keep only the first matching chunk from each source document."""
    source_paths: list[str] = []
    seen_paths: set[str] = set()

    for chunk in sorted(chunks, key=lambda chunk: chunk.rank):
        if chunk.source_path not in seen_paths:
            source_paths.append(chunk.source_path)
            seen_paths.add(chunk.source_path)

    return source_paths
