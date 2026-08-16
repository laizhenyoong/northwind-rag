"""Compare two answer runs graded by the same judge.

Judge agreement asks whether two graders read one run the same way. This asks
the other question: one grader, two pipelines, did the change help. The rates
alone cannot answer it, because a change that fixes four questions and breaks
four leaves the rate untouched while altering a tenth of the answers.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Sequence

from rag.evaluation.answer_judge import AnswerJudgement
from rag.evaluation.judge_agreement import COMPARED_LABELS, load_judgements


@dataclass(frozen=True, slots=True)
class LabelChange:
    """How one boolean label moved between a baseline run and a variant."""

    label: str
    compared_count: int
    baseline_rate: float
    comparison_rate: float
    fixed_question_ids: tuple[str, ...]
    broken_question_ids: tuple[str, ...]
    p_value: float

    @property
    def rate_change(self) -> float:
        """The headline movement, which the churn counts qualify."""
        return self.comparison_rate - self.baseline_rate

    @property
    def churn(self) -> int:
        """Questions whose verdict moved in either direction."""
        return len(self.fixed_question_ids) + len(self.broken_question_ids)


@dataclass(frozen=True, slots=True)
class RunComparison:
    """Per-label movement over the questions both runs actually scored."""

    compared_question_ids: tuple[str, ...]
    skipped_question_ids: tuple[str, ...]
    labels: tuple[LabelChange, ...]

    @property
    def compared_count(self) -> int:
        """The number of questions that carried a verdict in both runs."""
        return len(self.compared_question_ids)


def compare_runs(
    baseline: Sequence[AnswerJudgement],
    comparison: Sequence[AnswerJudgement],
    *,
    labels: Sequence[str] = COMPARED_LABELS,
) -> RunComparison:
    """Compare two runs question by question, ignoring unscored questions.

    A question is skipped when either run is missing it or recorded an error,
    for the same reason judge agreement skips them: a failed judgement holds
    placeholder falses rather than an opinion.
    """
    baseline_by_id = {judgement.question_id: judgement for judgement in baseline}
    comparison_by_id = {judgement.question_id: judgement for judgement in comparison}

    compared_ids = []
    skipped_ids = []
    for question_id in sorted(baseline_by_id.keys() | comparison_by_id.keys()):
        left = baseline_by_id.get(question_id)
        right = comparison_by_id.get(question_id)
        if left is None or right is None or left.error is not None or right.error is not None:
            skipped_ids.append(question_id)
        else:
            compared_ids.append(question_id)

    return RunComparison(
        compared_question_ids=tuple(compared_ids),
        skipped_question_ids=tuple(skipped_ids),
        labels=tuple(
            _score_label(label, compared_ids, baseline_by_id, comparison_by_id)
            for label in labels
        ),
    )


def mcnemar_p_value(broken: int, fixed: int) -> float:
    """Return the chance that this much lopsidedness came from noise alone.

    Only the questions that changed carry information about the change, so the
    test ignores every question both runs agreed on and asks whether splitting
    the movers this unevenly is surprising for a coin. Two-sided exact binomial.
    """
    movers = broken + fixed
    if movers == 0:
        return 1.0

    tail = sum(comb(movers, count) for count in range(min(broken, fixed) + 1))
    return min(1.0, 2 * tail / 2**movers)


def _score_label(
    label: str,
    compared_ids: Sequence[str],
    baseline_by_id: dict[str, AnswerJudgement],
    comparison_by_id: dict[str, AnswerJudgement],
) -> LabelChange:
    fixed = tuple(
        question_id
        for question_id in compared_ids
        if not getattr(baseline_by_id[question_id], label)
        and getattr(comparison_by_id[question_id], label)
    )
    broken = tuple(
        question_id
        for question_id in compared_ids
        if getattr(baseline_by_id[question_id], label)
        and not getattr(comparison_by_id[question_id], label)
    )
    compared_count = len(compared_ids)

    return LabelChange(
        label=label,
        compared_count=compared_count,
        baseline_rate=_rate(label, compared_ids, baseline_by_id),
        comparison_rate=_rate(label, compared_ids, comparison_by_id),
        fixed_question_ids=fixed,
        broken_question_ids=broken,
        p_value=mcnemar_p_value(len(broken), len(fixed)),
    )


def _rate(label: str, compared_ids: Sequence[str], by_id: dict[str, AnswerJudgement]) -> float:
    if not compared_ids:
        return 0.0
    return sum(bool(getattr(by_id[question_id], label)) for question_id in compared_ids) / len(
        compared_ids
    )


def main() -> None:
    """Report how a variant run moved against its baseline, label by label."""
    parser = argparse.ArgumentParser(description="Compare two answer runs")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--label", action="append", dest="labels")
    arguments = parser.parse_args()

    comparison = compare_runs(
        load_judgements(arguments.baseline),
        load_judgements(arguments.comparison),
        labels=arguments.labels or COMPARED_LABELS,
    )
    print(
        f"Compared {comparison.compared_count} questions "
        f"({len(comparison.skipped_question_ids)} skipped)."
    )
    for label in comparison.labels:
        print(
            f"\n{label.label}: {label.baseline_rate:.3f} -> {label.comparison_rate:.3f} "
            f"({label.rate_change:+.3f}) churn={label.churn} p={label.p_value:.3f}"
        )
        if label.fixed_question_ids:
            print(f"  fixed  : {', '.join(label.fixed_question_ids)}")
        if label.broken_question_ids:
            print(f"  broken : {', '.join(label.broken_question_ids)}")


if __name__ == "__main__":
    main()
