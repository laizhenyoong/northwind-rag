from rag.ingestion import Document
from rag.ingestion.manifest import IngestionManifest, ManifestEntry
from rag.sync import build_sync_plan


def document(content: str, *, status: str = "current") -> Document:
    return Document(
        content=content,
        metadata={
            "source_path": "data/policy.md",
            "document_key": "data/policy.md",
            "doc_id": "POL-001",
            "status": status,
        },
    )


def test_sync_plan_distinguishes_new_content_metadata_and_unchanged_documents() -> None:
    config = {"embedding_model": "embeddinggemma", "chunk_size": 500}
    first = build_sync_plan([document("first")], None, index_config=config, chunk_size=500)
    assert first.actions[0].kind == "new"

    previous = IngestionManifest(1, config, first.entries)
    unchanged = build_sync_plan([document("first")], previous, index_config=config, chunk_size=500)
    assert unchanged.actions[0].kind == "unchanged"

    metadata = build_sync_plan(
        [document("first", status="superseded")], previous, index_config=config, chunk_size=500
    )
    assert metadata.actions[0].kind == "metadata"

    content = build_sync_plan([document("second")], previous, index_config=config, chunk_size=500)
    assert content.actions[0].kind == "content"


def test_sync_plan_marks_missing_source_as_deleted() -> None:
    previous = IngestionManifest(
        1,
        {"chunk_size": 500},
        {
            "data/deleted.md": ManifestEntry(
                "data/deleted.md", None, None, "a", "b", ("data/deleted.md:0000",), "now"
            )
        },
    )

    plan = build_sync_plan([], previous, index_config={"chunk_size": 500}, chunk_size=500)

    assert plan.actions[0].kind == "deleted"
