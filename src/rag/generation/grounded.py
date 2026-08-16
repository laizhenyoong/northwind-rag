"""Prompt construction and citation validation for grounded answers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from rag.constants import REFUSAL
from rag.generation.ollama import OllamaCompletion, OllamaUsage
from rag.retrieval.semantic import RetrievedPassage


_CITATION_PATTERN = re.compile(r"\[S(\d+)\]")

SYSTEM_PROMPT = f"""Answer only from the supplied context.

- Cite a context label such as [S1] for every factual claim.
- When the context holds several versions or effective dates, use the one that
  applies to the question and name it. Mention superseded values only when the
  question asks for history or a comparison.
- If the context does not support an answer, reply with exactly:
{REFUSAL}"""


class ChatModel(Protocol):
    """The one chat operation needed by the grounded answerer."""

    def complete(self, *, system_prompt: str, user_prompt: str) -> str: ...


class GroundingError(RuntimeError):
    """The model response could not be tied back to retrieved context."""


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    """An answer plus the exact context and chunks it cites."""

    text: str
    context: str
    cited_chunk_ids: tuple[str, ...]
    usage: OllamaUsage | None = None


@dataclass(slots=True)
class GroundedAnswerer:
    """Generate an answer and reject citations outside the retrieved context."""

    chat_model: ChatModel

    def answer(self, question: str, passages: list[RetrievedPassage]) -> GroundedAnswer:
        """Answer a question only from passages and validate its source labels."""
        context, labels = build_labeled_context(passages)
        if not passages:
            return GroundedAnswer(REFUSAL, context, ())

        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Context:\n{context}\n\n"
            "Answer the question now."
        )
        completion = _complete_with_optional_usage(
            self.chat_model, system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt
        )
        response = completion.text
        if response == REFUSAL:
            return GroundedAnswer(response, context, (), completion.usage)

        citation_numbers = [int(match) for match in _CITATION_PATTERN.findall(response)]
        if not citation_numbers:
            raise GroundingError("Model answer has no context citations")
        invalid_numbers = sorted({number for number in citation_numbers if number not in labels})
        if invalid_numbers:
            invalid = ", ".join(f"S{number}" for number in invalid_numbers)
            raise GroundingError(f"Model answer cited context labels that were not provided: {invalid}")

        cited_chunk_ids = tuple(dict.fromkeys(labels[number].chunk_id for number in citation_numbers))
        return GroundedAnswer(response, context, cited_chunk_ids, completion.usage)


def _complete_with_optional_usage(
    chat_model: ChatModel, *, system_prompt: str, user_prompt: str
) -> OllamaCompletion:
    complete_with_usage = getattr(chat_model, "complete_with_usage", None)
    if callable(complete_with_usage):
        return complete_with_usage(system_prompt=system_prompt, user_prompt=user_prompt)
    return OllamaCompletion(
        text=chat_model.complete(system_prompt=system_prompt, user_prompt=user_prompt),
        usage=OllamaUsage(),
    )


def build_labeled_context(
    passages: list[RetrievedPassage],
) -> tuple[str, dict[int, RetrievedPassage]]:
    """Create stable labels that let a generated answer cite real chunks."""
    labels = {number: passage for number, passage in enumerate(passages, start=1)}
    context = "\n\n".join(
        f"[S{number}]\n"
        f"Source: {passage.metadata.get('source_path', 'unknown')}\n"
        f"Chunk: {passage.chunk_id}\n"
        f"{passage.text}"
        for number, passage in labels.items()
    )
    return context, labels
