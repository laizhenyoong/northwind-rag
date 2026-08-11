"""Incrementally synchronize the Markdown corpus with the Pinecone index."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from rag.chunking import Chunk, chunk_document
from rag.embeddings import OllamaEmbedder
from rag.indexing import _pinecone_record
from rag.ingestion import Document, load_corpus
from rag.ingestion.manifest import (
    IngestionManifest,
    ManifestEntry,
    load_manifest,
    save_manifest,
)
from rag.vector_store import PineconeSettings, ensure_index


MANIFEST_SCHEMA_VERSION = 2


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class VectorIndex(Protocol):
    def upsert(self, *, vectors: list[dict[str, Any]], namespace: str) -> Any: ...

    def fetch(self, *, ids: Sequence[str], namespace: str) -> Any: ...

    def delete(self, *, ids: Sequence[str], namespace: str) -> Any: ...


@dataclass(frozen=True, slots=True)
class SyncAction:
    """One source-level change detected during an incremental run."""

    source_path: str
    kind: str


@dataclass(frozen=True, slots=True)
class SyncPlan:
    """The decisions made before the index is changed."""

    actions: tuple[SyncAction, ...]
    entries: dict[str, ManifestEntry]


def build_sync_plan(
    documents: Sequence[Document],
    previous_manifest: IngestionManifest | None,
    *,
    index_config: dict[str, object],
    chunk_size: int,
) -> SyncPlan:
    """Classify every source as new, content-changed, metadata-changed, or unchanged."""
    previous_entries = previous_manifest.documents if previous_manifest else {}
    config_changed = (
        previous_manifest is not None and previous_manifest.index_config != index_config
    )
    actions: list[SyncAction] = []
    entries: dict[str, ManifestEntry] = {}

    for document in documents:
        source_path = _source_path(document)
        content_hash = _hash_text(document.content)
        metadata_hash = _hash_metadata(document.metadata)
        previous = previous_entries.get(source_path)

        if previous is None:
            kind = "new"
        elif config_changed or previous.content_sha256 != content_hash:
            kind = "content"
        elif previous.metadata_sha256 != metadata_hash:
            kind = "metadata"
        else:
            kind = "unchanged"

        chunk_ids = (
            previous.chunk_ids if kind == "unchanged" else tuple(
                chunk.id for chunk in chunk_document(document, chunk_size=chunk_size)
            )
        )
        entries[source_path] = ManifestEntry(
            source_path=source_path,
            document_id=_optional_string(document.metadata.get("doc_id")),
            version=_optional_string(document.metadata.get("version")),
            content_sha256=content_hash,
            metadata_sha256=metadata_hash,
            chunk_ids=chunk_ids,
            indexed_at=previous.indexed_at if kind == "unchanged" else _timestamp(),
        )
        actions.append(SyncAction(source_path, kind))

    for source_path in sorted(set(previous_entries) - set(entries)):
        actions.append(SyncAction(source_path, "deleted"))

    return SyncPlan(actions=tuple(sorted(actions, key=lambda action: action.source_path)), entries=entries)


def synchronize_corpus(
    corpus_root: Path,
    *,
    manifest_path: Path,
    embedder: Embedder,
    vector_index: VectorIndex,
    index_config: dict[str, object],
    chunk_size: int = 500,
    batch_size: int = 32,
    namespace: str = "",
    dry_run: bool = False,
) -> SyncPlan:
    """Apply the minimum safe Pinecone changes, then persist the new manifest."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    previous_manifest = load_manifest(manifest_path)
    documents = load_corpus(corpus_root)
    documents_by_path = {_source_path(document): document for document in documents}
    plan = build_sync_plan(
        documents,
        previous_manifest,
        index_config=index_config,
        chunk_size=chunk_size,
    )
    if dry_run:
        return plan

    previous_entries = previous_manifest.documents if previous_manifest else {}
    for action in plan.actions:
        document = documents_by_path.get(action.source_path)
        if action.kind in {"new", "content"}:
            assert document is not None
            chunks = chunk_document(document, chunk_size=chunk_size)
            _embed_and_upsert(chunks, embedder, vector_index, batch_size, namespace)
            old_chunk_ids = set(previous_entries.get(action.source_path, ManifestEntry(
                action.source_path, None, None, "", "", (), ""
            )).chunk_ids)
            stale_chunk_ids = sorted(old_chunk_ids - {chunk.id for chunk in chunks})
            _delete_ids(stale_chunk_ids, vector_index, namespace)
        elif action.kind == "metadata":
            assert document is not None
            chunks = chunk_document(document, chunk_size=chunk_size)
            _replace_metadata_without_embedding(chunks, vector_index, namespace)
        elif action.kind == "deleted":
            _delete_ids(previous_entries[action.source_path].chunk_ids, vector_index, namespace)

    save_manifest(
        manifest_path,
        IngestionManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            index_config=index_config,
            documents=plan.entries,
        ),
    )
    return plan


def _embed_and_upsert(
    chunks: Sequence[Chunk],
    embedder: Embedder,
    vector_index: VectorIndex,
    batch_size: int,
    namespace: str,
) -> None:
    for batch in _batches(chunks, batch_size):
        embeddings = embedder.embed([chunk.text for chunk in batch])
        if len(embeddings) != len(batch):
            raise RuntimeError("Embedding model returned the wrong number of vectors")
        vector_index.upsert(
            vectors=[
                _pinecone_record(chunk, embedding)
                for chunk, embedding in zip(batch, embeddings, strict=True)
            ],
            namespace=namespace,
        )


def _replace_metadata_without_embedding(
    chunks: Sequence[Chunk], vector_index: VectorIndex, namespace: str
) -> None:
    """Fetch stored vectors so a metadata-only change never calls the embedder."""
    for batch in _batches(chunks, 1_000):
        response = vector_index.fetch(ids=[chunk.id for chunk in batch], namespace=namespace)
        vectors = _response_vectors(response)
        records = []
        for chunk in batch:
            stored_vector = vectors.get(chunk.id)
            values = _vector_values(stored_vector)
            if values is None:
                raise RuntimeError(f"Cannot update metadata; vector is missing: {chunk.id}")
            records.append(_pinecone_record(chunk, values))
        vector_index.upsert(vectors=records, namespace=namespace)


def _delete_ids(ids: Sequence[str], vector_index: VectorIndex, namespace: str) -> None:
    for batch in _batches(ids, 1_000):
        vector_index.delete(ids=list(batch), namespace=namespace)


def _response_vectors(response: Any) -> Mapping[str, Any]:
    if isinstance(response, Mapping):
        vectors = response.get("vectors", {})
    else:
        vectors = getattr(response, "vectors", {})
    if not isinstance(vectors, Mapping):
        raise RuntimeError("Pinecone returned an unexpected fetch response")
    return vectors


def _vector_values(vector: Any) -> list[float] | None:
    if isinstance(vector, Mapping):
        values = vector.get("values")
    else:
        values = getattr(vector, "values", None)
    return list(values) if isinstance(values, Sequence) else None


def _source_path(document: Document) -> str:
    source_path = document.metadata.get("source_path")
    if not isinstance(source_path, str):
        raise ValueError("Every document must have source_path metadata")
    return source_path


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_metadata(metadata: dict[str, Any]) -> str:
    canonical = json.dumps(metadata, default=str, ensure_ascii=False, sort_keys=True)
    return _hash_text(canonical)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _batches(items: Sequence[Any], size: int) -> Sequence[Sequence[Any]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def main() -> None:
    """Run a safe incremental corpus sync from the project root."""
    parser = argparse.ArgumentParser(description="Synchronize corpus changes to Pinecone")
    parser.add_argument("--corpus-root", type=Path, default=Path("data"))
    parser.add_argument("--manifest", type=Path, default=Path(".rag/ingestion-manifest.json"))
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    embedder = OllamaEmbedder()
    # The planner returns before using the index, so a dry run needs neither an
    # API key nor a Pinecone network call.
    vector_index: Any = (
        object() if arguments.dry_run else ensure_index(PineconeSettings.from_environment())
    )
    plan = synchronize_corpus(
        arguments.corpus_root,
        manifest_path=arguments.manifest,
        embedder=embedder,
        vector_index=vector_index,
        index_config={
            "embedding_model": embedder.model,
            "chunk_size": arguments.chunk_size,
            "chunking_strategy": "markdown-aware-v1",
        },
        chunk_size=arguments.chunk_size,
        batch_size=arguments.batch_size,
        dry_run=arguments.dry_run,
    )
    for action in plan.actions:
        print(f"{action.kind.upper():9} {action.source_path}")
    if arguments.dry_run:
        print("Dry run: no Pinecone records or manifest were changed.")


if __name__ == "__main__":
    main()
