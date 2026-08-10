from rag.retrieval import RetrievedPassage, RerankingRetriever


class FakeCandidateRetriever:
    def __init__(self) -> None:
        self.calls = []

    def retrieve(self, question: str, *, top_k: int, metadata_filter=None) -> list[RetrievedPassage]:
        self.calls.append((question, top_k, metadata_filter))
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
    assert candidates.calls == [("Which document?", 20, None)]
    assert reranker.calls == [("Which document?", ["first", "second"], 1)]


def test_reranking_retriever_passes_filter_to_its_candidate_search() -> None:
    candidates = FakeCandidateRetriever()
    reranker = FakeReranker()
    retriever = RerankingRetriever(candidates, reranker, candidate_k=20)
    metadata_filter = {"is_current_version": {"$eq": True}}

    retriever.retrieve("Which document?", top_k=1, metadata_filter=metadata_filter)

    assert candidates.calls == [("Which document?", 20, metadata_filter)]
