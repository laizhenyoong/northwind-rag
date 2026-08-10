from rag.chunking import Chunk
from rag.retrieval.keyword import KeywordRetriever, tokenize


def test_tokenize_preserves_hyphenated_identifiers() -> None:
    assert tokenize("Use FIN-EXP-22, not FIN-EXP-19.") == [
        "use",
        "fin-exp-22",
        "not",
        "fin-exp-19",
    ]


def test_keyword_retriever_ranks_an_exact_rare_token_first() -> None:
    retriever = KeywordRetriever(
        (
            Chunk("a", "Use FIN-EXP-19 for the old process.", {"source_path": "old.md"}),
            Chunk("b", "Use FIN-EXP-22 for the current process.", {"source_path": "new.md"}),
            Chunk("c", "Travel reimbursement guidance.", {"source_path": "travel.md"}),
        )
    )

    passages = retriever.retrieve("Which form is FIN-EXP-22?", top_k=2)

    assert [passage.chunk_id for passage in passages] == ["b"]
    assert passages[0].metadata["source_path"] == "new.md"


def test_keyword_retriever_applies_the_same_version_filter_as_semantic_search() -> None:
    retriever = KeywordRetriever(
        (
            Chunk(
                "old",
                "Domestic per diem rate.",
                {"document_family": "POL-FIN-004", "is_current_version": False},
            ),
            Chunk(
                "current",
                "Domestic per diem rate.",
                {"document_family": "POL-FIN-004", "is_current_version": True},
            ),
        )
    )

    passages = retriever.retrieve(
        "Domestic per diem rate",
        metadata_filter={
            "$and": [
                {"document_family": {"$eq": "POL-FIN-004"}},
                {"is_current_version": {"$eq": True}},
            ]
        },
    )

    assert [passage.chunk_id for passage in passages] == ["current"]
