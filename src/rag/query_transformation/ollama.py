"""Use a local chat model to create focused retrieval sub-queries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


SYSTEM_PROMPT = """You rewrite a complex question into focused document-search queries.
Return at most three sub-queries, one per line, with no explanation. Preserve
exact names, product IDs, dates, and numbers. Each sub-query should seek one
fact needed to answer the original question. For a comparison question, write
a comparison-focused query that names both sides of the comparison (for
example, "previous lead time compared with new supplier lead time"). Do not
invent missing values and do not answer the question."""
_LIST_PREFIX = re.compile(r"^(?:[-*]|\d+[.)])\s*")


class ChatModel(Protocol):
    """The one local model operation needed for query decomposition."""

    def complete(self, *, system_prompt: str, user_prompt: str) -> str: ...


@dataclass(slots=True)
class OllamaQueryDecomposer:
    """Create a small, deduplicated list of search sub-queries."""

    chat_model: ChatModel
    max_subqueries: int = 3

    def __post_init__(self) -> None:
        if self.max_subqueries < 1:
            raise ValueError("max_subqueries must be at least 1")

    def decompose(self, question: str) -> list[str]:
        """Return focused queries, excluding the original question itself."""
        if not question.strip():
            raise ValueError("question must not be empty")
        response = self.chat_model.complete(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=f"Original question:\n{question}\n\nSub-queries:",
        )
        return parse_subqueries(response, original_question=question, limit=self.max_subqueries)


def parse_subqueries(response: str, *, original_question: str, limit: int) -> list[str]:
    """Normalize a line-oriented model response into safe retrieval inputs."""
    original_normalized = original_question.strip().casefold()
    subqueries: list[str] = []
    seen = {original_normalized}
    for line in response.splitlines():
        query = _LIST_PREFIX.sub("", line.strip()).strip("` \t\"'")
        if not query or query.startswith("```"):
            continue
        normalized = query.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        subqueries.append(query)
        if len(subqueries) == limit:
            break
    return subqueries
