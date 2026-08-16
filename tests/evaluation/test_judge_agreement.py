import json

import pytest

from rag.evaluation import (
    AnswerJudgement,
    compare_judgements,
    disagreements,
    load_judgements,
)


def judgement(
    question_id: str,
    *,
    correct: bool = True,
    citation_supported: bool = True,
    used_correct_version: bool = True,
    proper_refusal: bool = False,
    error: str | None = None,
) -> AnswerJudgement:
    return AnswerJudgement(
        question_id=question_id,
        correct=correct,
        citation_supported=citation_supported,
        used_correct_version=used_correct_version,
        proper_refusal=proper_refusal,
        reason="Because.",
        error=error,
    )


def label(report, name: str):
    return next(entry for entry in report.labels if entry.label == name)


def test_identical_judges_agree_completely() -> None:
    baseline = [judgement("Q001"), judgement("Q002", correct=False)]

    report = compare_judgements(baseline, list(baseline))

    assert report.compared_count == 2
    assert label(report, "correct").agreement_rate == 1.0
    assert label(report, "correct").cohen_kappa == 1.0


def test_agreement_counts_which_side_said_true() -> None:
    baseline = [judgement("Q001", correct=True), judgement("Q002", correct=False)]
    comparison = [judgement("Q001", correct=False), judgement("Q002", correct=False)]

    correct = label(compare_judgements(baseline, comparison), "correct")

    assert correct.agreement_rate == 0.5
    assert correct.both_false == 1
    assert correct.only_baseline_true == 1
    assert correct.only_comparison_true == 0


def test_kappa_is_zero_when_agreement_is_pure_luck() -> None:
    """Two judges that say true at the same rate but never on the same question."""
    baseline = [judgement(f"Q{index:03d}", correct=index < 2) for index in range(4)]
    comparison = [judgement(f"Q{index:03d}", correct=index >= 2) for index in range(4)]

    correct = label(compare_judgements(baseline, comparison), "correct")

    assert correct.agreement_rate == 0.0
    assert correct.cohen_kappa == pytest.approx(-1.0)


def test_kappa_discounts_a_dominant_verdict() -> None:
    """Nine of ten agree, but almost all of that is the shared habit of saying true."""
    baseline = [judgement(f"Q{index:03d}", correct=True) for index in range(10)]
    comparison = [judgement(f"Q{index:03d}", correct=index != 0) for index in range(10)]

    correct = label(compare_judgements(baseline, comparison), "correct")

    assert correct.agreement_rate == pytest.approx(0.9)
    assert correct.cohen_kappa == pytest.approx(0.0)


def test_errored_questions_are_skipped_rather_than_counted_as_agreement() -> None:
    baseline = [judgement("Q001"), judgement("Q002", error="Pipeline error: boom")]
    comparison = [judgement("Q001"), judgement("Q002", error="Pipeline error: boom")]

    report = compare_judgements(baseline, comparison)

    assert report.compared_question_ids == ("Q001",)
    assert report.skipped_question_ids == ("Q002",)
    assert label(report, "correct").compared_count == 1


def test_questions_missing_from_one_side_are_skipped() -> None:
    report = compare_judgements([judgement("Q001"), judgement("Q002")], [judgement("Q001")])

    assert report.compared_question_ids == ("Q001",)
    assert report.skipped_question_ids == ("Q002",)


def test_disagreements_return_both_verdicts_for_reading() -> None:
    baseline = [judgement("Q001"), judgement("Q002", correct=False)]
    comparison = [judgement("Q001"), judgement("Q002", correct=True)]

    pairs = disagreements(baseline, comparison)

    assert len(pairs) == 1
    assert pairs[0][0].question_id == "Q002"
    assert (pairs[0][0].correct, pairs[0][1].correct) == (False, True)


def test_load_judgements_accepts_files_written_before_failure_staging(tmp_path) -> None:
    path = tmp_path / "judgements.jsonl"
    path.write_text(
        json.dumps(
            {
                "question_id": "Q001",
                "correct": True,
                "citation_supported": True,
                "used_correct_version": True,
                "proper_refusal": False,
                "reason": "Matches.",
                "error": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_judgements(path)

    assert loaded[0].question_id == "Q001"
    assert loaded[0].failure_stage is None


def test_load_judgements_reports_the_offending_line(tmp_path) -> None:
    path = tmp_path / "judgements.jsonl"
    path.write_text('{"question_id": "Q001"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        load_judgements(path)
