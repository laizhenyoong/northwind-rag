from pathlib import Path

from rag.evaluation import load_gold_questions


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_load_gold_questions_preserves_ground_truth() -> None:
    questions = load_gold_questions(REPOSITORY_ROOT / "eval" / "gold_questions.jsonl")

    assert len(questions) == 87

    first_question = questions[0]
    assert first_question.id == "Q001"
    assert first_question.source_files == ("data/policies/travel-expense-v2.1.md",)

    unanswerable = next(question for question in questions if question.id == "Q079")
    assert unanswerable.answerable is False
    assert unanswerable.source_files == ()
