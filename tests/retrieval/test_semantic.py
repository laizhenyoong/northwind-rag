from rag.retrieval import SemanticRetriever, build_context


class FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        assert texts == ["Where is the policy?"]
        return [[0.1, 0.2]]


class FakeIndex:
    def __init__(self) -> None:
        self.query_arguments = None

    def query(self, **kwargs):
        self.query_arguments = kwargs
        return {
            "matches": [
                {
                    "id": "data/policies/travel.md:0002",
                    "score": 0.93,
                    "metadata": {
                        "text": "Use FIN-EXP-22.",
                        "source_path": "data/policies/travel.md",
                        "status": "current",
                    },
                }
            ]
        }


def test_retrieve_embeds_question_and_queries_pinecone() -> None:
    index = FakeIndex()
    retriever = SemanticRetriever(embedder=FakeEmbedder(), vector_index=index)

    passages = retriever.retrieve(
        "Where is the policy?", top_k=5, metadata_filter={"status": "current"}
    )

    assert index.query_arguments == {
        "vector": [0.1, 0.2],
        "top_k": 5,
        "include_metadata": True,
        "filter": {"status": "current"},
    }
    assert passages[0].chunk_id == "data/policies/travel.md:0002"
    assert passages[0].text == "Use FIN-EXP-22."
    assert passages[0].as_trace_chunk(rank=1).source_path == "data/policies/travel.md"


def test_build_context_keeps_source_and_chunk_identifiers() -> None:
    retriever = SemanticRetriever(embedder=FakeEmbedder(), vector_index=FakeIndex())
    passage = retriever.retrieve("Where is the policy?")[0]

    context = build_context([passage])

    assert "data/policies/travel.md" in context
    assert "data/policies/travel.md:0002" in context
    assert "Use FIN-EXP-22." in context
