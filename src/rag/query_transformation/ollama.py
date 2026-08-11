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
FOLLOWUP_SYSTEM_PROMPT = """You create one follow-up document-search query.
Use only names, IDs, and facts visible in the retrieved evidence. The original
question needs another fact that the first search may not contain. Write a
targeted query that combines a discovered name or ID with that unresolved fact.
For example, if the evidence identifies a new supplier and the question asks
about a comparison, search for that supplier's previous versus new value. Do
not answer the question. Return one query only, with no explanation."""
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


@dataclass(slots=True)
class OllamaFollowupQueryGenerator:
    """Derive one second-pass query from the first pass's retrieved evidence."""

    chat_model: ChatModel

    def generate(self, question: str, passages: list[object]) -> str | None:
        evidence = "\n\n".join(
            f"[Evidence {index}]\n{getattr(passage, 'text', '')}"
            for index, passage in enumerate(passages, start=1)
        )
        response = self.chat_model.complete(
            system_prompt=FOLLOWUP_SYSTEM_PROMPT,
            user_prompt=f"Original question:\n{question}\n\nRetrieved evidence:\n{evidence}\n\nFollow-up query:",
        )
        return parse_followup_query(response, original_question=question)


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


def parse_followup_query(response: str, *, original_question: str) -> str | None:
    """Keep one safe single-line follow-up query from the local model."""
    queries = parse_subqueries(response, original_question=original_question, limit=1)
    return queries[0] if queries else None
