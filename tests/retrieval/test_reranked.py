from rag.retrieval import RetrievedPassage, RerankingRetriever


class FakeCandidateRetriever:
    def __init__(self) -> None:
        self.calls = []

    def retrieve(self, question: str, *, top_k: int) -> list[RetrievedPassage]:
        self.calls.append((question, top_k))
        return [passage("first"), passage("second")]


class FakeReranker:
    def __init__(self) -> None:
        self.calls = []

    def rerank(self, question, passages, *, top_k):
        self.calls.append((question, [passage.chunk_id for passage in passages], top_k))
        return list(reversed(passages))[:top_k]


def passage(chunk_id: str) -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=chunk_id,
        text=chunk_id,
        score=0.1,
        metadata={"source_path": f"data/{chunk_id}.md"},
    )


def test_reranking_retriever_only_reranks_the_wider_candidate_set() -> None:
    candidates = FakeCandidateRetriever()
    reranker = FakeReranker()
    retriever = RerankingRetriever(candidates, reranker, candidate_k=20)

    passages = retriever.retrieve("Which document?", top_k=1)

    assert [passage.chunk_id for passage in passages] == ["second"]
    assert candidates.calls == [("Which document?", 20)]
    assert reranker.calls == [("Which document?", ["first", "second"], 1)]
