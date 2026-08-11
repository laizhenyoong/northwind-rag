from rag.query_transformation.multihop import MultiHopRetriever
from rag.retrieval import RetrievedPassage


def passage(chunk_id: str) -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=chunk_id,
        text=chunk_id,
        score=0.1,
        metadata={"source_path": f"data/{chunk_id}.md"},
    )


class FakeRetriever:
    def __init__(self) -> None:
        self.calls = []
        self.last_queries = ()

    def retrieve(self, question, *, top_k, metadata_filter=None):
        self.calls.append((question, top_k, metadata_filter))
        self.last_queries = (question,)
        return {
            "original": [passage("supplier"), passage("decision")],
            "Tanaka previous versus new lead time": [passage("lead-time"), passage("supplier")],
        }[question]


class FakeFollowupGenerator:
    def __init__(self) -> None:
        self.calls = []

    def generate(self, question, passages):
        self.calls.append((question, [passage.chunk_id for passage in passages]))
        return "Tanaka previous versus new lead time"


def test_multihop_uses_first_pass_evidence_for_a_second_retrieval() -> None:
    candidate_retriever = FakeRetriever()
    generator = FakeFollowupGenerator()
    retriever = MultiHopRetriever(candidate_retriever, generator, candidate_k=2)

    passages = retriever.retrieve("original", top_k=2)

    assert {passage.chunk_id for passage in passages} == {"supplier", "lead-time"}
    assert generator.calls == [("original", ["supplier", "decision"])]
    assert candidate_retriever.calls == [("original", 2, None), ("Tanaka previous versus new lead time", 2, None)]
    assert retriever.last_queries == ("original", "Tanaka previous versus new lead time")
