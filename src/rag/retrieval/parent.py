"""Rank small chunks, then answer from the documents they came from."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rag.ingestion import load_corpus
from rag.retrieval.semantic import RetrievedPassage


class CandidateRetriever(Protocol):
    """The ranking step this retriever expands after."""

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 5,
        metadata_filter: Mapping[str, object] | None = None,
    ) -> list[RetrievedPassage]: ...


@dataclass(frozen=True, slots=True)
class ParentDocumentRetriever:
    """Replace every ranked chunk with the whole document it came from.

    Search is best on small chunks and answering is best on whole passages.
    This keeps the ranking a chunk earned and hands over the document around
    it, so facts split across chunks too far apart to be neighbors arrive
    together. A document past ``max_document_characters`` is left as its chunk,
    because a long document is several unrelated topics bound together rather
    than one coherent passage.
    """

    candidate_retriever: CandidateRetriever
    documents: Mapping[str, str]
    max_document_characters: int | None = None

    @classmethod
    def from_corpus(
        cls,
        candidate_retriever: CandidateRetriever,
        corpus_root: Path,
        *,
        max_document_characters: int | None = None,
    ) -> ParentDocumentRetriever:
        """Load every document once, keyed by the path chunks record."""
        documents = {}
        for document in load_corpus(corpus_root):
            source_path = document.metadata.get("source_path")
            if isinstance(source_path, str):
                documents[source_path] = document.content
        return cls(
            candidate_retriever=candidate_retriever,
            documents=documents,
            max_document_characters=max_document_characters,
        )

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 5,
        metadata_filter: Mapping[str, object] | None = None,
    ) -> list[RetrievedPassage]:
        """Return one passage per document behind the ranked chunks."""
        passages = self.candidate_retriever.retrieve(
            question, top_k=top_k, metadata_filter=metadata_filter
        )
        expanded: list[RetrievedPassage] = []
        seen_paths: set[str] = set()
        for passage in passages:
            source_path = passage.metadata.get("source_path")
            document = (
                self.documents.get(source_path) if isinstance(source_path, str) else None
            )
            if document is None or self._exceeds_limit(document):
                expanded.append(passage)
                continue
            if source_path in seen_paths:
                continue
            seen_paths.add(source_path)
            expanded.append(
                RetrievedPassage(
                    chunk_id=source_path,
                    text=document,
                    score=passage.score,
                    metadata=_parent_metadata(passage),
                )
            )
        return expanded

    def _exceeds_limit(self, document: str) -> bool:
        return (
            self.max_document_characters is not None
            and len(document) > self.max_document_characters
        )


def _parent_metadata(passage: RetrievedPassage) -> dict[str, object]:
    """Carry the chunk's metadata over, minus the position it no longer has.

    Dropping ``chunk_index`` keeps neighbor expansion from stacking adjacent
    chunks onto a passage that already contains them.
    """
    metadata = {
        key: value for key, value in passage.metadata.items() if key != "chunk_index"
    }
    metadata["expanded_from"] = passage.chunk_id
    return metadata
