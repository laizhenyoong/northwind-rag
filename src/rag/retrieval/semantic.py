"""Baseline semantic retrieval using Ollama embeddings and Pinecone."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from rag.embeddings import OllamaEmbedder
from rag.evaluation.traces import RetrievedChunk
from rag.retrieval.version_filters import (
    as_of_date_filter,
    current_version_filter,
    exact_version_filter,
    parse_as_of_date,
)
from rag.vector_store import PineconeSettings, ensure_index


class Embedder(Protocol):
    """The one embedding operation needed at query time."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class VectorIndex(Protocol):
    """The Pinecone query operation needed at query time."""

    def query(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class RetrievedPassage:
    """A source passage returned from semantic vector search."""

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]

    def as_trace_chunk(self, *, rank: int) -> RetrievedChunk:
        """Keep the inspectable retrieval fields required by our evaluator."""
        source_path = self.metadata.get("source_path")
        if not isinstance(source_path, str):
            raise RuntimeError(f"Retrieved chunk {self.chunk_id} has no source_path")
        return RetrievedChunk(
            chunk_id=self.chunk_id,
            source_path=source_path,
            rank=rank,
            score=self.score,
        )


@dataclass(slots=True)
class SemanticRetriever:
    """Embed a question, then retrieve its nearest Pinecone passages."""

    embedder: Embedder
    vector_index: VectorIndex

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 5,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[RetrievedPassage]:
        """Return the top semantic matches, with an optional metadata constraint."""
        if not question.strip():
            raise ValueError("question must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        embeddings = self.embedder.embed([question])
        if len(embeddings) != 1:
            raise RuntimeError("Embedding model did not return exactly one query vector")

        query_arguments: dict[str, Any] = {
            "vector": embeddings[0],
            "top_k": top_k,
            "include_metadata": True,
        }
        if metadata_filter is not None:
            query_arguments["filter"] = dict(metadata_filter)

        response = self.vector_index.query(**query_arguments)
        return [_passage_from_match(match) for match in _response_matches(response)]


def build_context(passages: list[RetrievedPassage]) -> str:
    """Format retrieved passages for the grounded answer-generation step."""
    return "\n\n".join(
        f"[Source: {passage.metadata.get('source_path', 'unknown')} | "
        f"Chunk: {passage.chunk_id}]\n{passage.text}"
        for passage in passages
    )


def _response_matches(response: Any) -> list[Any]:
    if isinstance(response, Mapping):
        matches = response.get("matches", [])
    else:
        matches = getattr(response, "matches", [])
    if not isinstance(matches, list):
        raise RuntimeError("Pinecone returned an unexpected query response")
    return matches


def _passage_from_match(match: Any) -> RetrievedPassage:
    chunk_id = _match_field(match, "id")
    score = _match_field(match, "score")
    raw_metadata = _match_field(match, "metadata")
    if not isinstance(chunk_id, str) or not isinstance(score, int | float):
        raise RuntimeError("Pinecone returned a match without an ID or score")
    if not isinstance(raw_metadata, Mapping):
        raise RuntimeError(f"Retrieved chunk {chunk_id} has invalid metadata")

    metadata = dict(raw_metadata)
    text = metadata.pop("text", None)
    if not isinstance(text, str):
        raise RuntimeError(f"Retrieved chunk {chunk_id} has no stored text")

    return RetrievedPassage(
        chunk_id=chunk_id,
        text=text,
        score=float(score),
        metadata=metadata,
    )


def _match_field(match: Any, field: str) -> Any:
    if isinstance(match, Mapping):
        return match.get(field)
    return getattr(match, field, None)


def main() -> None:
    """Run one semantic retrieval query from the project root."""
    parser = argparse.ArgumentParser(description="Retrieve semantic matches from Pinecone")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
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

    retriever = SemanticRetriever(
        embedder=OllamaEmbedder(),
        vector_index=ensure_index(PineconeSettings.from_environment()),
    )
    for rank, passage in enumerate(
        retriever.retrieve(
            arguments.question,
            top_k=arguments.top_k,
            metadata_filter=metadata_filter,
        ),
        start=1,
    ):
        print(
            f"{rank}. score={passage.score:.4f} "
            f"source={passage.metadata.get('source_path')} "
            f"chunk={passage.chunk_id}"
        )


if __name__ == "__main__":
    main()
