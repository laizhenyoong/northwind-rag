"""Apply a precise reranker after a wider, cheaper candidate search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rag.retrieval.semantic import RetrievedPassage


class CandidateRetriever(Protocol):
    """Retrieve a wider candidate set for the second-stage reranker."""

    def retrieve(self, question: str, *, top_k: int) -> list[RetrievedPassage]: ...


class Reranker(Protocol):
    """Rescore question-passage pairs and retain the best passages."""

    def rerank(
        self, question: str, passages: list[RetrievedPassage], *, top_k: int
    ) -> list[RetrievedPassage]: ...


@dataclass(slots=True)
class RerankingRetriever:
    """Retrieve broadly, then reorder candidates with a cross-encoder."""

    candidate_retriever: CandidateRetriever
    reranker: Reranker
    candidate_k: int = 20

    def __post_init__(self) -> None:
        if self.candidate_k < 1:
            raise ValueError("candidate_k must be at least 1")

    def retrieve(self, question: str, *, top_k: int = 5) -> list[RetrievedPassage]:
        """Use the reranker only on the candidate set, never the whole corpus."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if top_k > self.candidate_k:
            raise ValueError("top_k cannot be greater than candidate_k")

        candidates = self.candidate_retriever.retrieve(question, top_k=self.candidate_k)
        return self.reranker.rerank(question, candidates, top_k=top_k)
