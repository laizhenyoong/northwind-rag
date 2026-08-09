"""Deterministic checks for answer traces, citations, and refusal behavior."""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from collections.abc import Iterable
from typing import Sequence

from rag.constants import REFUSAL
from rag.evaluation.questions import GoldQuestion
from rag.evaluation.traces import RunTrace


_CITATION_PATTERN = re.compile(r"\[S(\d+)\]")


@dataclass(frozen=True, slots=True)
class AnswerMetrics:
    """Inspectable deterministic checks for one generated answer."""

    question_id: str
    produced_answer: bool
    refused: bool
    citations_valid: bool
    cites_expected_source: bool
    error: bool


@dataclass(frozen=True, slots=True)
class AnswerSummary:
    """Aggregate answer-trace rates; this is not semantic answer correctness."""

    question_count: int
    answerable_question_count: int
    unanswerable_question_count: int
    error_count: int
    answer_rate: float
    citation_valid_rate: float
    expected_source_citation_rate: float
    refusal_rate: float


@dataclass(frozen=True, slots=True)
class AnswerEvaluation:
    """Per-question checks and their aggregate summary."""

    metrics: tuple[AnswerMetrics, ...]
    summary: AnswerSummary


def score_answer(question: GoldQuestion, trace: RunTrace) -> AnswerMetrics:
    """Check trace evidence without trying to judge natural-language wording."""
    if question.id != trace.question_id:
        raise ValueError(f"Question {question.id} does not match trace {trace.question_id}")

    answer = trace.answer or ""
    refused = answer == REFUSAL
    cited_ranks = [int(match) for match in _CITATION_PATTERN.findall(answer)]
    chunks_by_rank = {chunk.rank: chunk for chunk in trace.retrieved_chunks}
    citations_valid = bool(cited_ranks) and all(rank in chunks_by_rank for rank in cited_ranks)
    cited_sources = {
        chunks_by_rank[rank].source_path for rank in cited_ranks if rank in chunks_by_rank
    }
    cites_expected_source = citations_valid and bool(cited_sources & set(question.source_files))

    return AnswerMetrics(
        question_id=question.id,
        produced_answer=bool(answer) and not refused and trace.error is None,
        refused=refused and trace.error is None,
        citations_valid=citations_valid,
        cites_expected_source=cites_expected_source,
        error=trace.error is not None,
    )


def evaluate_answers(
    questions: Sequence[GoldQuestion], traces: Sequence[RunTrace]
) -> AnswerEvaluation:
    """Score matching question and answer traces without an LLM judge."""
    traces_by_question_id = {trace.question_id: trace for trace in traces}
    missing = [question.id for question in questions if question.id not in traces_by_question_id]
    if missing:
        raise ValueError(f"Missing answer traces for: {', '.join(missing)}")

    metrics = tuple(score_answer(question, traces_by_question_id[question.id]) for question in questions)
    answerable_metrics = [
        metric for question, metric in zip(questions, metrics, strict=True) if question.answerable
    ]
    unanswerable_metrics = [
        metric for question, metric in zip(questions, metrics, strict=True) if not question.answerable
    ]

    return AnswerEvaluation(
        metrics=metrics,
        summary=AnswerSummary(
            question_count=len(questions),
            answerable_question_count=len(answerable_metrics),
            unanswerable_question_count=len(unanswerable_metrics),
            error_count=sum(metric.error for metric in metrics),
            answer_rate=_mean(metric.produced_answer for metric in answerable_metrics),
            citation_valid_rate=_mean(metric.citations_valid for metric in answerable_metrics),
            expected_source_citation_rate=_mean(
                metric.cites_expected_source for metric in answerable_metrics
            ),
            refusal_rate=_mean(metric.refused for metric in unanswerable_metrics),
        ),
    )


def _mean(values: Iterable[bool]) -> float:
    values = tuple(values)
    return statistics.fmean(values) if values else 0.0
