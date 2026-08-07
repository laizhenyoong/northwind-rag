from datetime import date

from rag.chunking import Chunk
from rag.indexing import _metadata_for_pinecone, index_corpus
from rag.ingestion import Document


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(index)] for index, _ in enumerate(texts)]


class FakeIndex:
    def __init__(self) -> None:
        self.upserts = []

    def upsert(self, *, vectors, namespace) -> None:
        self.upserts.append({"vectors": vectors, "namespace": namespace})


def test_index_corpus_embeds_and_upserts_chunks_in_batches(monkeypatch, tmp_path) -> None:
    documents = [Document(content="unused", metadata={"document_key": "data/example.md"})]
    chunks = [
        Chunk("data/example.md:0000", "first", {"status": "current"}),
        Chunk("data/example.md:0001", "second", {"status": "current"}),
        Chunk("data/example.md:0002", "third", {"status": "current"}),
    ]
    monkeypatch.setattr("rag.indexing.load_corpus", lambda _: documents)
    monkeypatch.setattr("rag.indexing.chunk_corpus", lambda _, chunk_size: chunks)
    embedder = FakeEmbedder()
    index = FakeIndex()

    result = index_corpus(
        tmp_path, embedder=embedder, vector_index=index, batch_size=2
    )

    assert result.document_count == 1
    assert result.chunk_count == 3
    assert embedder.calls == [["first", "second"], ["third"]]
    assert index.upserts[0]["namespace"] == ""
    assert index.upserts[0]["vectors"][0] == {
        "id": "data/example.md:0000",
        "values": [0.0],
        "metadata": {"text": "first", "status": "current"},
    }


def test_metadata_for_pinecone_makes_dates_filterable_strings() -> None:
    metadata = _metadata_for_pinecone(
        {
            "doc_id": "POL-FIN-004",
            "version": "2.1",
            "effective_date": date(2026, 3, 1),
            "status": "current",
        }
    )

    assert metadata == {
        "doc_id": "POL-FIN-004",
        "version": "2.1",
        "effective_date": "2026-03-01",
        "status": "current",
        "effective_date_ordinal": date(2026, 3, 1).toordinal(),
        "expiry_date_ordinal": date.max.toordinal(),
        "document_family": "POL-FIN-004",
        "is_current_version": True,
    }
