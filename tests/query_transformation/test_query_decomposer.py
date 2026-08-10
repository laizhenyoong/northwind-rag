from rag.query_transformation.ollama import OllamaQueryDecomposer, parse_subqueries


class FakeChatModel:
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        assert "FKM seals" in user_prompt
        return "1. Who supplies FKM seals for NH-8840-X?\n2. What is the expected FKM seal lead time?"


def test_decomposer_returns_clean_focused_subqueries() -> None:
    queries = OllamaQueryDecomposer(FakeChatModel()).decompose(
        "Who supplies FKM seals for NH-8840-X and what lead time is expected?"
    )

    assert queries == [
        "Who supplies FKM seals for NH-8840-X?",
        "What is the expected FKM seal lead time?",
    ]


def test_parser_removes_duplicates_and_original_question() -> None:
    queries = parse_subqueries(
        "- What is the status?\n- Who owns the decision?\n- Who owns the decision?",
        original_question="What is the status?",
        limit=3,
    )

    assert queries == ["Who owns the decision?"]
