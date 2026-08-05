from pathlib import Path

from rag.ingestion import load_corpus


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_corpus_loader_preserves_document_metadata() -> None:
    documents = load_corpus(REPOSITORY_ROOT / "data")

    assert len(documents) == 33

    travel_policy = next(
        document
        for document in documents
        if document.metadata["document_key"]
        == "policies/travel-expense-v2.1.md"
    )
    assert travel_policy.metadata["status"] == "current"
    assert travel_policy.metadata["source_path"] == "policies/travel-expense-v2.1.md"
    assert "per diem" in travel_policy.content.lower()
