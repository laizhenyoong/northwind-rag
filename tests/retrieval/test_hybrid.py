from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.semantic import RetrievedPassage


class FakeRetriever:
    def __init__(self, passages: list[RetrievedPassage]) -> None:
        self.passages = passages
        self.calls = []

    def retrieve(self, question: str, *, top_k: int) -> list[RetrievedPassage]:
        self.calls.append((question, top_k))
        return self.passages[:top_k]


def passage(chunk_id: str) -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=chunk_id,
        text=f"Text for {chunk_id}",
        score=0.5,
        metadata={"source_path": f"data/{chunk_id}.md"},
    )


def test_hybrid_retriever_rewards_a_chunk_found_by_both_systems() -> None:
    semantic = FakeRetriever([passage("semantic-only"), passage("shared")])
    keyword = FakeRetriever([passage("shared"), passage("keyword-only")])
    retriever = HybridRetriever(semantic, keyword, candidate_k=2, rrf_k=60)

    passages = retriever.retrieve("Which form?", top_k=2)

    assert [passage.chunk_id for passage in passages] == ["shared", "semantic-only"]
    assert semantic.calls == [("Which form?", 2)]
    assert keyword.calls == [("Which form?", 2)]
    assert passages[0].score == 1 / 62 + 1 / 61
