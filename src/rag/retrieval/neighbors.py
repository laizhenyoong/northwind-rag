"""Expand retrieved evidence with adjacent chunks from the same document."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from rag.chunking import Chunk, chunk_corpus
from rag.ingestion import load_corpus
from rag.retrieval.semantic import RetrievedPassage


class CandidateRetriever(Protocol):
    """The retrieval operation expanded after ranking is complete."""

    def retrieve(
        self,
        question: str,
        *,
        top_k: int,
        metadata_filter: Mapping[str, object] | None = None,
    ) -> list[RetrievedPassage]: ...


@dataclass(slots=True)
class NeighborExpandingRetriever:
    """Keep ranked chunks, then append nearby source chunks as answer context.

    This is deliberately a final-context operation. It does not change the
    candidate rankings used by hybrid retrieval, reranking, or multi-hop search.
    """

    candidate_retriever: CandidateRetriever
    chunks: tuple[Chunk, ...]
    neighbor_window: int = 1
    neighbor_source_k: int | None = None
    _chunks_by_location: dict[tuple[str, int], Chunk] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.neighbor_window < 1:
            raise ValueError("neighbor_window must be at least 1")
        if self.neighbor_source_k is not None and self.neighbor_source_k < 1:
            raise ValueError("neighbor_source_k must be at least 1")
        self._chunks_by_location = {
            location: chunk
            for chunk in self.chunks
            if (location := _location(chunk.metadata)) is not None
        }

    @classmethod
    def from_corpus(
        cls,
        candidate_retriever: CandidateRetriever,
        corpus_root: Path,
        *,
        chunk_size: int = 500,
        neighbor_window: int = 1,
        neighbor_source_k: int | None = None,
    ) -> NeighborExpandingRetriever:
        """Load the same chunk layout that was used to index the corpus."""
        return cls(
            candidate_retriever=candidate_retriever,
            chunks=tuple(chunk_corpus(load_corpus(corpus_root), chunk_size=chunk_size)),
            neighbor_window=neighbor_window,
            neighbor_source_k=neighbor_source_k,
        )

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 5,
        metadata_filter: Mapping[str, object] | None = None,
    ) -> list[RetrievedPassage]:
        """Return ranked evidence followed by unique adjacent source chunks."""
        passages = self.candidate_retriever.retrieve(
            question, top_k=top_k, metadata_filter=metadata_filter
        )
        expanded = list(passages)
        selected_ids = {passage.chunk_id for passage in passages}
        expansion_sources = passages[: self.neighbor_source_k]
        for passage in expansion_sources:
            location = _location(passage.metadata)
            if location is None:
                continue
            source_path, chunk_index = location
            for offset in range(-self.neighbor_window, self.neighbor_window + 1):
                if offset == 0:
                    continue
                neighbor = self._chunks_by_location.get((source_path, chunk_index + offset))
                if neighbor is not None and neighbor.id not in selected_ids:
                    expanded.append(
                        RetrievedPassage(
                            chunk_id=neighbor.id,
                            text=neighbor.text,
                            score=passage.score,
                            metadata={**neighbor.metadata, "expanded_from": passage.chunk_id},
                        )
                    )
                    selected_ids.add(neighbor.id)
        return expanded


def _location(metadata: Mapping[str, object]) -> tuple[str, int] | None:
    source_path = metadata.get("source_path")
    chunk_index = metadata.get("chunk_index")
    if not isinstance(source_path, str) or not isinstance(chunk_index, int):
        return None
    return source_path, chunk_index
