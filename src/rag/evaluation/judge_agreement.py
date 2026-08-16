"""Measure how far two sets of verdicts agree about the same answers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from rag.evaluation.answer_judge import AnswerJudgement


# The four boolean verdicts every judgement carries.
COMPARED_LABELS = (
    "correct",
    "citation_supported",
    "used_correct_version",
    "proper_refusal",
)


@dataclass(frozen=True, slots=True)
class LabelAgreement:
    """How often two judges gave the same verdict for one boolean label."""

    label: str
    compared_count: int
    agreement_rate: float
    both_true: int
    both_false: int
    only_baseline_true: int
    only_comparison_true: int
    cohen_kappa: float


@dataclass(frozen=True, slots=True)
class AgreementReport:
    """Per-label agreement over the questions both judges actually scored."""

    compared_question_ids: tuple[str, ...]
    skipped_question_ids: tuple[str, ...]
    labels: tuple[LabelAgreement, ...]

    @property
    def compared_count(self) -> int:
        """The number of questions that carried a verdict on both sides."""
        return len(self.compared_question_ids)


def load_judgements(path: Path) -> list[AnswerJudgement]:
    """Load a JSONL judgement file, tolerating files written before failure staging."""
    judgements = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            judgements.append(
                AnswerJudgement(
                    question_id=record["question_id"],
                    correct=record["correct"],
                    citation_supported=record["citation_supported"],
                    used_correct_version=record["used_correct_version"],
                    proper_refusal=record["proper_refusal"],
                    reason=record["reason"],
                    error=record.get("error"),
                    failure_stage=record.get("failure_stage"),
                )
            )
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid judgement on line {line_number} of {path}") from error
    return judgements


def compare_judgements(
    baseline: Sequence[AnswerJudgement],
    comparison: Sequence[AnswerJudgement],
    *,
    labels: Sequence[str] = COMPARED_LABELS,
) -> AgreementReport:
    """Compare two judges question by question, ignoring unscored questions.

    A question is skipped when either side is missing it, or when either side
    recorded an error. An errored judgement holds placeholder falses rather than
    an opinion, so counting it would credit the judges with agreeing about
    something neither of them actually judged.
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

    return AgreementReport(
        compared_question_ids=tuple(compared_ids),
        skipped_question_ids=tuple(skipped_ids),
        labels=tuple(
            _score_label(
                label,
                [getattr(baseline_by_id[question_id], label) for question_id in compared_ids],
                [getattr(comparison_by_id[question_id], label) for question_id in compared_ids],
            )
            for label in labels
        ),
    )


def disagreements(
    baseline: Sequence[AnswerJudgement],
    comparison: Sequence[AnswerJudgement],
    *,
    label: str = "correct",
) -> tuple[tuple[AnswerJudgement, AnswerJudgement], ...]:
    """Return the judgement pairs that differ on one label, for manual reading."""
    comparison_by_id = {judgement.question_id: judgement for judgement in comparison}
    return tuple(
        (left, comparison_by_id[left.question_id])
        for left in baseline
        if left.error is None
        and left.question_id in comparison_by_id
        and comparison_by_id[left.question_id].error is None
        and getattr(left, label) != getattr(comparison_by_id[left.question_id], label)
    )


def _score_label(
    label: str, baseline_values: Sequence[bool], comparison_values: Sequence[bool]
) -> LabelAgreement:
    both_true = sum(1 for left, right in zip(baseline_values, comparison_values) if left and right)
    both_false = sum(
        1 for left, right in zip(baseline_values, comparison_values) if not left and not right
    )
    only_baseline_true = sum(
        1 for left, right in zip(baseline_values, comparison_values) if left and not right
    )
    only_comparison_true = sum(
        1 for left, right in zip(baseline_values, comparison_values) if not left and right
    )
    compared_count = len(baseline_values)

    return LabelAgreement(
        label=label,
        compared_count=compared_count,
        agreement_rate=(both_true + both_false) / compared_count if compared_count else 0.0,
        both_true=both_true,
        both_false=both_false,
        only_baseline_true=only_baseline_true,
        only_comparison_true=only_comparison_true,
        cohen_kappa=_cohen_kappa(baseline_values, comparison_values),
    )


def _cohen_kappa(baseline_values: Sequence[bool], comparison_values: Sequence[bool]) -> float:
    """Return agreement discounted by the agreement two judges would hit by luck.

    Raw agreement flatters a judge whenever one verdict dominates. If 95 of 100
    answers are correct, two judges that always answer true agree 95% of the
    time while sharing no judgement at all. Kappa subtracts that expected
    coincidence: 1.0 is perfect, 0.0 is no better than chance, below 0.0 is
    worse than chance.
    """
    total = len(baseline_values)
    if total == 0:
        return 0.0

    observed = sum(1 for left, right in zip(baseline_values, comparison_values) if left == right) / total
    baseline_true = sum(baseline_values) / total
    comparison_true = sum(comparison_values) / total
    expected = baseline_true * comparison_true + (1 - baseline_true) * (1 - comparison_true)

    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return (observed - expected) / (1 - expected)
