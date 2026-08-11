"""Retrieve once, derive one evidence-based follow-up query, then retrieve again."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from rag.retrieval.semantic import RetrievedPassage


class CandidateRetriever(Protocol):
    def retrieve(
        self,
        question: str,
        *,
        top_k: int,
        metadata_filter: Mapping[str, object] | None = None,
    ) -> list[RetrievedPassage]: ...


class FollowupQueryGenerator(Protocol):
    def generate(self, question: str, passages: list[RetrievedPassage]) -> str | None: ...


@dataclass(slots=True)
class MultiHopRetriever:
    """Use discovered evidence to make one targeted second retrieval pass."""

    candidate_retriever: CandidateRetriever
    followup_generator: FollowupQueryGenerator
    candidate_k: int = 20
    rrf_k: int = 60
    evidence_k: int = 5
    last_queries: tuple[str, ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        if self.candidate_k < 1:
            raise ValueError("candidate_k must be at least 1")
        if self.rrf_k < 0:
            raise ValueError("rrf_k must be at least 0")
        if self.evidence_k < 1:
            raise ValueError("evidence_k must be at least 1")

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 5,
        metadata_filter: Mapping[str, object] | None = None,
    ) -> list[RetrievedPassage]:
        """Fuse first-pass and follow-up results with reciprocal-rank fusion."""
        if not question.strip():
            raise ValueError("question must not be empty")
        if top_k < 1 or top_k > self.candidate_k:
            raise ValueError("top_k must be between 1 and candidate_k")

        first_pass = self.candidate_retriever.retrieve(
            question, top_k=self.candidate_k, metadata_filter=metadata_filter
        )
        initial_queries = tuple(getattr(self.candidate_retriever, "last_queries", ()) or (question,))
        followup = self.followup_generator.generate(question, first_pass[: self.evidence_k])
        if not followup or followup.strip().casefold() in {query.casefold() for query in initial_queries}:
            self.last_queries = initial_queries
            return first_pass[:top_k]

        second_pass = self.candidate_retriever.retrieve(
            followup, top_k=self.candidate_k, metadata_filter=metadata_filter
        )
        self.last_queries = (*initial_queries, followup)
        return _fuse_ranked_passages((first_pass, second_pass), top_k=top_k, rrf_k=self.rrf_k)


def _fuse_ranked_passages(
    ranked_lists: tuple[list[RetrievedPassage], ...], *, top_k: int, rrf_k: int
) -> list[RetrievedPassage]:
    scores: dict[str, float] = {}
    passages_by_id: dict[str, RetrievedPassage] = {}
    for passages in ranked_lists:
        for rank, passage in enumerate(passages, start=1):
            scores[passage.chunk_id] = scores.get(passage.chunk_id, 0) + 1 / (rrf_k + rank)
            passages_by_id.setdefault(passage.chunk_id, passage)
    return [
        RetrievedPassage(
            chunk_id=chunk_id,
            text=passages_by_id[chunk_id].text,
            score=score,
            metadata=passages_by_id[chunk_id].metadata,
        )
        for chunk_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    ]
