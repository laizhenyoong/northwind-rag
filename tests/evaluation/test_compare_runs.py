import pytest

from rag.evaluation.answer_judge import AnswerJudgement
from rag.evaluation.compare_runs import compare_runs, mcnemar_p_value


def judgement(question_id: str, correct: bool, *, error: str | None = None) -> AnswerJudgement:
    return AnswerJudgement(
        question_id=question_id,
        correct=correct,
        citation_supported=True,
        used_correct_version=True,
        proper_refusal=False,
        reason="reason",
        error=error,
    )


def test_a_change_reports_which_questions_it_fixed_and_broke() -> None:
    baseline = [judgement("Q1", True), judgement("Q2", False), judgement("Q3", True)]
    comparison = [judgement("Q1", False), judgement("Q2", True), judgement("Q3", True)]

    correct = compare_runs(baseline, comparison, labels=["correct"]).labels[0]

    assert correct.fixed_question_ids == ("Q2",)
    assert correct.broken_question_ids == ("Q1",)
    assert correct.churn == 2


def test_an_unchanged_rate_still_exposes_its_churn() -> None:
    """The failure the rate alone hides: equal traffic in both directions."""
    baseline = [judgement("Q1", True), judgement("Q2", False)]
    comparison = [judgement("Q1", False), judgement("Q2", True)]

    correct = compare_runs(baseline, comparison, labels=["correct"]).labels[0]

    assert correct.rate_change == pytest.approx(0.0)
    assert correct.churn == 2


def test_a_question_either_run_failed_to_judge_is_skipped() -> None:
    baseline = [judgement("Q1", True), judgement("Q2", False, error="timeout")]
    comparison = [judgement("Q1", True), judgement("Q2", False)]

    comparison_report = compare_runs(baseline, comparison, labels=["correct"])

    assert comparison_report.compared_question_ids == ("Q1",)
    assert comparison_report.skipped_question_ids == ("Q2",)


def test_an_even_split_of_movers_is_indistinguishable_from_noise() -> None:
    assert mcnemar_p_value(broken=4, fixed=4) == pytest.approx(1.0)


def test_a_one_sided_run_of_movers_is_significant() -> None:
    assert mcnemar_p_value(broken=0, fixed=8) == pytest.approx(2 / 256)


def test_a_run_that_changed_nothing_carries_no_evidence() -> None:
    assert mcnemar_p_value(broken=0, fixed=0) == pytest.approx(1.0)
