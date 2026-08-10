"""Fuse semantic and keyword rankings with reciprocal-rank fusion."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rag.embeddings import OllamaEmbedder
from rag.retrieval.keyword import KeywordRetriever
from rag.retrieval.semantic import RetrievedPassage, SemanticRetriever
from rag.retrieval.version_filters import (
    as_of_date_filter,
    current_version_filter,
    exact_version_filter,
    parse_as_of_date,
)
from rag.vector_store import PineconeSettings, ensure_index


class Retriever(Protocol):
    """The common retrieval operation needed for rank fusion."""

    def retrieve(
        self,
        question: str,
        *,
        top_k: int,
        metadata_filter: Mapping[str, object] | None = None,
    ) -> list[RetrievedPassage]: ...


@dataclass(slots=True)
class HybridRetriever:
    """Combine semantic and BM25 results without mixing their raw scores."""

    semantic_retriever: Retriever
    keyword_retriever: Retriever
    candidate_k: int = 20
    rrf_k: int = 60

    def __post_init__(self) -> None:
        if self.candidate_k < 1:
            raise ValueError("candidate_k must be at least 1")
        if self.rrf_k < 0:
            raise ValueError("rrf_k must be at least 0")

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 5,
        metadata_filter: Mapping[str, object] | None = None,
    ) -> list[RetrievedPassage]:
        """Return the best passages according to reciprocal-rank fusion.

        A passage receives ``1 / (rrf_k + rank)`` from each retrieval list in
        which it appears. A passage found by both systems therefore moves up.
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if top_k > self.candidate_k:
            raise ValueError("top_k cannot be greater than candidate_k")

        semantic_passages = self.semantic_retriever.retrieve(
            question, top_k=self.candidate_k, metadata_filter=metadata_filter
        )
        keyword_passages = self.keyword_retriever.retrieve(
            question, top_k=self.candidate_k, metadata_filter=metadata_filter
        )

        fused_scores: dict[str, float] = {}
        passages_by_id: dict[str, RetrievedPassage] = {}
        for ranked_passages in (semantic_passages, keyword_passages):
            for rank, passage in enumerate(ranked_passages, start=1):
                fused_scores[passage.chunk_id] = fused_scores.get(passage.chunk_id, 0) + 1 / (
                    self.rrf_k + rank
                )
                passages_by_id.setdefault(passage.chunk_id, passage)

        return [
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


def main() -> None:
    """Run one hybrid query from the project root."""
    parser = argparse.ArgumentParser(description="Fuse semantic and BM25 retrieval")
    parser.add_argument("question")
    parser.add_argument("--corpus-root", type=Path, default=Path("data"))
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--document-id")
    version_selection = parser.add_mutually_exclusive_group()
    version_selection.add_argument("--current", action="store_true")
    version_selection.add_argument("--version")
    version_selection.add_argument("--as-of")
    arguments = parser.parse_args()

    if (arguments.current or arguments.version or arguments.as_of) and not arguments.document_id:
        parser.error("--document-id is required with --current, --version, or --as-of")

    metadata_filter = None
    if arguments.current:
        metadata_filter = current_version_filter(arguments.document_id)
    elif arguments.version:
        metadata_filter = exact_version_filter(arguments.document_id, arguments.version)
    elif arguments.as_of:
        try:
            metadata_filter = as_of_date_filter(
                arguments.document_id, parse_as_of_date(arguments.as_of)
            )
        except ValueError as error:
            parser.error(str(error))

    retriever = HybridRetriever(
        semantic_retriever=SemanticRetriever(
            embedder=OllamaEmbedder(),
            vector_index=ensure_index(PineconeSettings.from_environment()),
        ),
        keyword_retriever=KeywordRetriever.from_corpus(
            arguments.corpus_root, chunk_size=arguments.chunk_size
        ),
        candidate_k=arguments.candidate_k,
    )
    for rank, passage in enumerate(
        retriever.retrieve(
            arguments.question, top_k=arguments.top_k, metadata_filter=metadata_filter
        ),
        start=1,
    ):
        print(
            f"{rank}. rrf_score={passage.score:.4f} "
            f"source={passage.metadata.get('source_path')} "
            f"chunk={passage.chunk_id}"
        )


if __name__ == "__main__":
    main()
