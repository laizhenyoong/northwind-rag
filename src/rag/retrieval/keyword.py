"""Exact-token retrieval with the BM25 ranking algorithm."""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from rag.chunking import Chunk, chunk_corpus
from rag.ingestion import load_corpus
from rag.retrieval.semantic import RetrievedPassage


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Normalize words while preserving hyphenated identifiers as one token."""
    return [match.group().lower() for match in TOKEN_PATTERN.finditer(text)]


@dataclass(slots=True)
class KeywordRetriever:
    """Search chunks by exact token overlap, weighted with BM25."""

    chunks: tuple[Chunk, ...]
    k1: float = 1.5
    b: float = 0.75
    _document_tokens: tuple[list[str], ...] = field(init=False, repr=False)
    _document_frequencies: Counter[str] = field(init=False, repr=False)
    _average_document_length: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.k1 <= 0:
            raise ValueError("k1 must be greater than 0")
        if not 0 <= self.b <= 1:
            raise ValueError("b must be between 0 and 1")

        self._document_tokens = tuple(tokenize(chunk.text) for chunk in self.chunks)
        self._document_frequencies = Counter(
            token for tokens in self._document_tokens for token in set(tokens)
        )
        self._average_document_length = (
            sum(map(len, self._document_tokens)) / len(self._document_tokens)
            if self._document_tokens
            else 0.0
        )

    @classmethod
    def from_corpus(cls, corpus_root: Path, *, chunk_size: int = 500) -> KeywordRetriever:
        """Create an in-memory keyword index from the same chunks as Pinecone."""
        return cls(tuple(chunk_corpus(load_corpus(corpus_root), chunk_size=chunk_size)))

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 5,
        metadata_filter: Mapping[str, object] | None = None,
    ) -> list[RetrievedPassage]:
        """Return matching chunks, optionally constrained by the same metadata policy."""
        if not question.strip():
            raise ValueError("question must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_tokens = tokenize(question)
        scores = [self._score(query_tokens, tokens) for tokens in self._document_tokens]
        ranked = sorted(
            (
                (score, chunk)
                for score, chunk in zip(scores, self.chunks, strict=True)
                if score > 0 and _matches_metadata_filter(chunk.metadata, metadata_filter)
            ),
            key=lambda item: (-item[0], item[1].id),
        )
        return [
            RetrievedPassage(
                chunk_id=chunk.id,
                text=chunk.text,
                score=score,
                metadata=dict(chunk.metadata),
            )
            for score, chunk in ranked[:top_k]
        ]

    def _score(self, query_tokens: list[str], document_tokens: list[str]) -> float:
        if not document_tokens or not query_tokens:
            return 0.0

        term_frequencies = Counter(document_tokens)
        document_length = len(document_tokens)
        score = 0.0
        for token in set(query_tokens):
            frequency = term_frequencies[token]
            if not frequency:
                continue
            idf = math.log(
                1 + (len(self.chunks) - self._document_frequencies[token] + 0.5)
                / (self._document_frequencies[token] + 0.5)
            )
            length_normalizer = self.k1 * (
                1 - self.b + self.b * document_length / self._average_document_length
            )
            score += idf * frequency * (self.k1 + 1) / (frequency + length_normalizer)
        return score


def _matches_metadata_filter(
    metadata: Mapping[str, object], metadata_filter: Mapping[str, object] | None
) -> bool:
    """Evaluate the Pinecone filter operators used by our local BM25 index."""
    if metadata_filter is None:
        return True

    for field, condition in metadata_filter.items():
        if field == "$and":
            if not isinstance(condition, list) or not all(
                isinstance(item, Mapping) and _matches_metadata_filter(metadata, item)
                for item in condition
            ):
                return False
            continue
        if field == "$or":
            if not isinstance(condition, list) or not any(
                isinstance(item, Mapping) and _matches_metadata_filter(metadata, item)
                for item in condition
            ):
                return False
            continue
        if not isinstance(condition, Mapping):
            raise ValueError(f"Invalid metadata filter condition for {field}")

        actual = metadata.get(field)
        for operator, expected in condition.items():
            if operator == "$eq":
                matched = actual == expected
            elif operator == "$lte":
                matched = actual is not None and actual <= expected
            elif operator == "$gte":
                matched = actual is not None and actual >= expected
            else:
                raise ValueError(f"Unsupported local metadata filter operator: {operator}")
            if not matched:
                return False
    return True


def main() -> None:
    """Run one BM25 query over the local corpus."""
    parser = argparse.ArgumentParser(description="Retrieve exact-token BM25 matches")
    parser.add_argument("question")
    parser.add_argument("--corpus-root", type=Path, default=Path("data"))
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=5)
    arguments = parser.parse_args()

    retriever = KeywordRetriever.from_corpus(
        arguments.corpus_root, chunk_size=arguments.chunk_size
    )
    for rank, passage in enumerate(
        retriever.retrieve(arguments.question, top_k=arguments.top_k), start=1
    ):
        print(
            f"{rank}. score={passage.score:.4f} "
            f"source={passage.metadata.get('source_path')} "
            f"chunk={passage.chunk_id}"
        )


if __name__ == "__main__":
    main()
