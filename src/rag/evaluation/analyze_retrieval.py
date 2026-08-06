"""Inspect retrieval traces to decide which improvement to try next."""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rag.evaluation.questions import GoldQuestion, load_gold_questions
from rag.evaluation.traces import RunTrace, load_traces


@dataclass(frozen=True, slots=True)
class QuestionOutcome:
    """Where an answerable question's first relevant source appeared."""

    question_id: str
    concept: str
    first_relevant_rank: int | None


@dataclass(frozen=True, slots=True)
class ConceptSummary:
    """A small scorecard for one question concept."""

    concept: str
    answerable_count: int
    hit_at_1: int
    hit_at_k: int
    missing_from_top_k: int
    mean_reciprocal_rank: float


def analyze_retrieval(
    questions: Sequence[GoldQuestion], traces: Sequence[RunTrace], *, top_k: int
) -> tuple[list[ConceptSummary], list[QuestionOutcome]]:
    """Group answerable outcomes by concept without calling any model."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    traces_by_id = {trace.question_id: trace for trace in traces}
    missing_traces = [question.id for question in questions if question.id not in traces_by_id]
    if missing_traces:
        raise ValueError(f"Trace file has no result for: {', '.join(missing_traces)}")

    outcomes = []
    for question in questions:
        if question.answerable:
            outcomes.append(
                QuestionOutcome(
                    question_id=question.id,
                    concept=question.concept,
                    first_relevant_rank=_first_relevant_rank(
                        question, traces_by_id[question.id]
                    ),
                )
            )

    outcomes_by_concept: dict[str, list[QuestionOutcome]] = defaultdict(list)
    for outcome in outcomes:
        outcomes_by_concept[outcome.concept].append(outcome)

    summaries = [
        _summarize_concept(concept, concept_outcomes, top_k)
        for concept, concept_outcomes in outcomes_by_concept.items()
    ]
    return sorted(summaries, key=lambda summary: summary.concept), outcomes


def format_report(summaries: Sequence[ConceptSummary], *, top_k: int) -> str:
    """Format the scorecard as a terminal-friendly table."""
    headers = ("Concept", "Answerable", "Hit@1", f"Hit@{top_k}", "Missing", "MRR")
    rows = [
        (
            summary.concept,
            str(summary.answerable_count),
            str(summary.hit_at_1),
            str(summary.hit_at_k),
            str(summary.missing_from_top_k),
            f"{summary.mean_reciprocal_rank:.3f}",
        )
        for summary in summaries
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    format_row = lambda row: " | ".join(
        value.ljust(widths[index]) for index, value in enumerate(row)
    )
    separator = "-+-".join("-" * width for width in widths)
    return "\n".join([format_row(headers), separator, *(format_row(row) for row in rows)])


def _first_relevant_rank(question: GoldQuestion, trace: RunTrace) -> int | None:
    expected_sources = set(question.source_files)
    for chunk in sorted(trace.retrieved_chunks, key=lambda chunk: chunk.rank):
        if chunk.source_path in expected_sources:
            return chunk.rank
    return None


def _summarize_concept(
    concept: str, outcomes: Sequence[QuestionOutcome], top_k: int
) -> ConceptSummary:
    ranks = [outcome.first_relevant_rank for outcome in outcomes]
    return ConceptSummary(
        concept=concept,
        answerable_count=len(outcomes),
        hit_at_1=sum(rank == 1 for rank in ranks),
        hit_at_k=sum(rank is not None and rank <= top_k for rank in ranks),
        missing_from_top_k=sum(rank is None or rank > top_k for rank in ranks),
        mean_reciprocal_rank=statistics.fmean(
            1 / rank if rank is not None and rank <= top_k else 0 for rank in ranks
        ),
    )


def main() -> None:
    """Print the scorecard and questions whose expected source missed rank 1."""
    parser = argparse.ArgumentParser(description="Analyze saved retrieval traces")
    parser.add_argument("--questions", type=Path, default=Path("eval/gold_questions.jsonl"))
    parser.add_argument("--traces", type=Path, default=Path("results/retrieval-baseline.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    arguments = parser.parse_args()

    summaries, outcomes = analyze_retrieval(
        load_gold_questions(arguments.questions),
        load_traces(arguments.traces),
        top_k=arguments.top_k,
    )
    print(format_report(summaries, top_k=arguments.top_k))
    print("\nQuestions not retrieved at rank 1:")
    for outcome in outcomes:
        if outcome.first_relevant_rank != 1:
            rank = outcome.first_relevant_rank or f">{arguments.top_k}"
            print(f"- {outcome.question_id} | {outcome.concept} | first relevant rank: {rank}")


if __name__ == "__main__":
    main()
