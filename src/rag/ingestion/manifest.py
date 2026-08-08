"""Persistent record of the source documents already indexed."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """The prior index state for one source document."""

    source_path: str
    document_id: str | None
    version: str | None
    content_sha256: str
    metadata_sha256: str
    chunk_ids: tuple[str, ...]
    indexed_at: str


@dataclass(frozen=True, slots=True)
class IngestionManifest:
    """The generated state needed to make incremental indexing safe."""

    schema_version: int
    index_config: dict[str, object]
    documents: dict[str, ManifestEntry]


def load_manifest(path: Path) -> IngestionManifest | None:
    """Return no manifest on the first run; otherwise validate and load it."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        documents = {
            source_path: ManifestEntry(
                source_path=entry["source_path"],
                document_id=entry.get("document_id"),
                version=entry.get("version"),
                content_sha256=entry["content_sha256"],
                metadata_sha256=entry["metadata_sha256"],
                chunk_ids=tuple(entry["chunk_ids"]),
                indexed_at=entry["indexed_at"],
            )
            for source_path, entry in payload["documents"].items()
        }
        return IngestionManifest(
            schema_version=payload["schema_version"],
            index_config=payload["index_config"],
            documents=documents,
        )
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid ingestion manifest: {path}") from error


def save_manifest(path: Path, manifest: IngestionManifest) -> None:
    """Persist the manifest only after a successful sync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": manifest.schema_version,
        "index_config": manifest.index_config,
        "documents": {
            source_path: asdict(entry)
            for source_path, entry in sorted(manifest.documents.items())
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
