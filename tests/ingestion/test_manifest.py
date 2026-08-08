from rag.ingestion.manifest import IngestionManifest, ManifestEntry, load_manifest, save_manifest


def test_manifest_round_trip(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    manifest = IngestionManifest(
        schema_version=1,
        index_config={"chunk_size": 500},
        documents={
            "data/example.md": ManifestEntry(
                source_path="data/example.md",
                document_id="POL-001",
                version="2.0",
                content_sha256="content",
                metadata_sha256="metadata",
                chunk_ids=("data/example.md:0000",),
                indexed_at="2026-08-08T00:00:00+00:00",
            )
        },
    )

    save_manifest(path, manifest)

    assert load_manifest(path) == manifest
