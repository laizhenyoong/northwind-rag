from rag.chunking import Chunk
from rag.retrieval.neighbors import NeighborExpandingRetriever
from rag.retrieval.semantic import RetrievedPassage


class FakeRetriever:
    def retrieve(self, question: str, *, top_k: int, metadata_filter=None) -> list[RetrievedPassage]:
        return [
            RetrievedPassage(
                chunk_id="policy.md:0001",
                text="Ranked evidence",
                score=0.9,
                metadata={"source_path": "data/policy.md", "chunk_index": 1},
            )
        ]


def chunk(index: int) -> Chunk:
    return Chunk(
        id=f"policy.md:{index:04d}",
        text=f"Chunk {index}",
        metadata={"source_path": "data/policy.md", "chunk_index": index},
    )


def test_neighbor_expansion_keeps_ranked_evidence_then_adds_adjacent_chunks() -> None:
    retriever = NeighborExpandingRetriever(FakeRetriever(), tuple(chunk(index) for index in range(3)))

    passages = retriever.retrieve("policy", top_k=1)

    assert [passage.chunk_id for passage in passages] == [
        "policy.md:0001",
        "policy.md:0000",
        "policy.md:0002",
    ]
    assert passages[1].metadata["expanded_from"] == "policy.md:0001"


def test_neighbor_expansion_deduplicates_neighbors_of_multiple_ranked_chunks() -> None:
    class TwoPassageRetriever:
        def retrieve(self, question: str, *, top_k: int, metadata_filter=None) -> list[RetrievedPassage]:
            return [
                RetrievedPassage("policy.md:0001", "One", 0.9, {"source_path": "data/policy.md", "chunk_index": 1}),
                RetrievedPassage("policy.md:0002", "Two", 0.8, {"source_path": "data/policy.md", "chunk_index": 2}),
            ]

    passages = NeighborExpandingRetriever(TwoPassageRetriever(), tuple(chunk(index) for index in range(4))).retrieve("policy", top_k=2)

    assert [passage.chunk_id for passage in passages] == [
        "policy.md:0001",
        "policy.md:0002",
        "policy.md:0000",
        "policy.md:0003",
    ]
