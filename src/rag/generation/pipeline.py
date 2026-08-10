"""Run retrieval and grounded generation while retaining an inspectable trace."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from rag.evaluation.traces import RunTrace
from rag.generation.grounded import GroundedAnswer
from rag.retrieval.semantic import RetrievedPassage, build_context


class Retriever(Protocol):
    def retrieve(
        self,
        question: str,
        *,
        top_k: int,
        metadata_filter: Mapping[str, object] | None = None,
    ) -> list[RetrievedPassage]: ...


class Answerer(Protocol):
    def answer(self, question: str, passages: list[RetrievedPassage]) -> GroundedAnswer: ...


@dataclass(frozen=True, slots=True)
class AnswerRun:
    """The persisted trace and, when successful, generated citation details."""

    trace: RunTrace
    generated_answer: GroundedAnswer | None


def answer_question(
    *,
    question_id: str,
    question: str,
    retriever: Retriever,
    answerer: Answerer,
    top_k: int,
    pipeline_config: dict[str, Any],
    metadata_filter: Mapping[str, object] | None = None,
) -> AnswerRun:
    """Retrieve, answer, and capture failures without losing the evidence."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    started_at = time.perf_counter()
    passages: list[RetrievedPassage] = []
    generated_answer: GroundedAnswer | None = None
    error: str | None = None
    try:
        retrieve_arguments: dict[str, object] = {"top_k": top_k}
        if metadata_filter is not None:
            retrieve_arguments["metadata_filter"] = metadata_filter
        passages = retriever.retrieve(question, **retrieve_arguments)
        generated_answer = answerer.answer(question, passages)
    except Exception as exception:  # Preserve a debuggable record for evaluation.
        error = f"{type(exception).__name__}: {exception}"

    retrieved_chunks = tuple(
        passage.as_trace_chunk(rank=rank) for rank, passage in enumerate(passages, start=1)
    )
    context = generated_answer.context if generated_answer else build_context(passages)
    trace = RunTrace(
        question_id=question_id,
        question=question,
        pipeline_config={**pipeline_config, "top_k": top_k, "metadata_filter": metadata_filter},
        retrieved_chunks=retrieved_chunks,
        context_sent_to_model=context or None,
        answer=generated_answer.text if generated_answer else None,
        timing_ms=round((time.perf_counter() - started_at) * 1000),
        error=error,
    )
    return AnswerRun(trace=trace, generated_answer=generated_answer)
