"""Semantic answer judging against gold answers and retrieved evidence."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
import re
import statistics
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol, Sequence

from rag.constants import REFUSAL
from rag.evaluation.questions import GoldQuestion
from rag.evaluation.traces import RunTrace


FailureStage = Literal["pipeline", "judge"]


JUDGE_SYSTEM_PROMPT = """Judge a RAG answer using only the supplied question, gold
answer, generated answer and context. Return one JSON object and nothing else:
{
  "correct": boolean,
  "citation_supported": boolean,
  "used_correct_version": boolean,
  "proper_refusal": boolean,
  "reason": string
}

correct: the answer reaches the gold answer's conclusion for what the question
asked. Wording may differ. It is false when a number, name, date or version is
wrong, when a fact the question asked for is missing, or when it volunteers a
superseded value the question did not ask about.

citation_supported: the labels the answer cites support the claims it makes.

used_correct_version: the answer uses the version or effective date the gold
answer requires.

proper_refusal: true only when the question is unanswerable and the answer
refuses instead of inventing one. Always false for an answerable question.

Mark a field false and say why when the supplied material cannot support it."""


class ChatModel(Protocol):
    """The chat operation required for semantic answer judging."""

    def complete(self, *, system_prompt: str, user_prompt: str) -> str: ...


@dataclass(frozen=True, slots=True)
class AnswerJudgement:
    """One inspectable semantic verdict for a generated answer trace."""

    question_id: str
    correct: bool
    citation_supported: bool
    used_correct_version: bool
    proper_refusal: bool
    reason: str
    error: str | None = None
    failure_stage: FailureStage | None = None


@dataclass(frozen=True, slots=True)
class AnswerJudgementSummary:
    """Aggregate semantic quality rates, separated by answerability."""

    question_count: int
    answerable_question_count: int
    unanswerable_question_count: int
    error_count: int
    pipeline_error_count: int
    judge_error_count: int
    judged_answerable_question_count: int
    judged_unanswerable_question_count: int
    correctness_rate: float
    citation_supported_rate: float
    correct_version_rate: float
    proper_refusal_rate: float


@dataclass(frozen=True, slots=True)
class AnswerJudgementEvaluation:
    """Per-question semantic verdicts and aggregate rates."""

    judgements: tuple[AnswerJudgement, ...]
    summary: AnswerJudgementSummary


@dataclass(slots=True)
class SemanticAnswerJudge:
    """Use a chat model to compare a generated answer with known evidence."""

    chat_model: ChatModel

    def judge(self, question: GoldQuestion, trace: RunTrace) -> AnswerJudgement:
        """Return a structured verdict without modifying the original trace."""
        if question.id != trace.question_id:
            raise ValueError(f"Question {question.id} does not match trace {trace.question_id}")
        if trace.error is not None:
            return _failed_judgement(question.id, f"Pipeline error: {trace.error}", stage="pipeline")
        if not question.answerable and trace.answer == REFUSAL:
            return _deterministic_refusal_judgement(question.id)

        try:
            response = self.chat_model.complete(
                system_prompt=JUDGE_SYSTEM_PROMPT,
                user_prompt=_build_judge_prompt(question, trace),
            )
            values = _parse_judgement(response)
        except (RuntimeError, ValueError, json.JSONDecodeError) as error:
            return _failed_judgement(question.id, str(error), stage="judge")

        return AnswerJudgement(question_id=question.id, **values)


def evaluate_semantic_answers(
    questions: Sequence[GoldQuestion],
    traces: Sequence[RunTrace],
    *,
    judge: SemanticAnswerJudge,
    workers: int = 1,
) -> AnswerJudgementEvaluation:
    """Judge every trace and compute answerable and refusal quality rates."""
    traces_by_question_id = {trace.question_id: trace for trace in traces}
    missing = [question.id for question in questions if question.id not in traces_by_question_id]
    if missing:
        raise ValueError(f"Missing answer traces for: {', '.join(missing)}")

    if workers > 1:
        # Judging is one independent network call per question, so it parallelises
        # cleanly. pool.map preserves input order, which the aggregation relies on.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            judgements = tuple(
                pool.map(lambda question: judge.judge(question, traces_by_question_id[question.id]), questions)
            )
    else:
        judgements = tuple(
            judge.judge(question, traces_by_question_id[question.id]) for question in questions
        )
    answerable = tuple(
        judgement
        for question, judgement in zip(questions, judgements, strict=True)
        if question.answerable
    )
    unanswerable = tuple(
        judgement
        for question, judgement in zip(questions, judgements, strict=True)
        if not question.answerable
    )
    judged_answerable = _judged(answerable)
    judged_unanswerable = _judged(unanswerable)
    failures: Counter[FailureStage] = Counter(
        judgement.failure_stage for judgement in judgements if judgement.failure_stage is not None
    )

    return AnswerJudgementEvaluation(
        judgements=judgements,
        summary=AnswerJudgementSummary(
            question_count=len(questions),
            answerable_question_count=len(answerable),
            unanswerable_question_count=len(unanswerable),
            error_count=sum(failures.values()),
            pipeline_error_count=failures["pipeline"],
            judge_error_count=failures["judge"],
            judged_answerable_question_count=len(judged_answerable),
            judged_unanswerable_question_count=len(judged_unanswerable),
            correctness_rate=_mean(judgement.correct for judgement in judged_answerable),
            citation_supported_rate=_mean(
                judgement.citation_supported for judgement in judged_answerable
            ),
            correct_version_rate=_mean(
                judgement.used_correct_version for judgement in judged_answerable
            ),
            proper_refusal_rate=_mean(judgement.proper_refusal for judgement in judged_unanswerable),
        ),
    )


def write_judgements(path: Path, judgements: Sequence[AnswerJudgement]) -> None:
    """Write one durable semantic verdict per question as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(asdict(judgement), ensure_ascii=False) for judgement in judgements) + "\n",
        encoding="utf-8",
    )


def _build_judge_prompt(question: GoldQuestion, trace: RunTrace) -> str:
    return (
        f"Question ID: {question.id}\n"
        f"Question: {question.question}\n"
        f"Answerable: {question.answerable}\n"
        f"Gold answer: {question.expected_answer}\n"
        f"Expected source files: {', '.join(question.source_files) or 'none'}\n\n"
        f"Generated answer:\n{trace.answer or '(no answer)'}\n\n"
        f"Retrieved context:\n{trace.context_sent_to_model or '(no context)'}"
    )


def _parse_judgement(response: str) -> dict[str, bool | str]:
    match = re.search(r"\{.*\}", response, flags=re.DOTALL)
    if match is None:
        raise ValueError("Judge did not return a JSON object")
    value = json.loads(match.group())
    required_booleans = (
        "correct",
        "citation_supported",
        "used_correct_version",
        "proper_refusal",
    )
    if not isinstance(value, dict) or any(not isinstance(value.get(key), bool) for key in required_booleans):
        raise ValueError("Judge response is missing one or more boolean verdicts")
    if not isinstance(value.get("reason"), str) or not value["reason"].strip():
        raise ValueError("Judge response is missing a reason")
    return {key: value[key] for key in (*required_booleans, "reason")}


def _failed_judgement(question_id: str, error: str, *, stage: FailureStage) -> AnswerJudgement:
    return AnswerJudgement(
        question_id=question_id,
        correct=False,
        citation_supported=False,
        used_correct_version=False,
        proper_refusal=False,
        reason="No semantic judgement was produced.",
        error=error,
        failure_stage=stage,
    )


def _deterministic_refusal_judgement(question_id: str) -> AnswerJudgement:
    return AnswerJudgement(
        question_id=question_id,
        correct=True,
        citation_supported=False,
        used_correct_version=False,
        proper_refusal=True,
        reason="Deterministic pass: the answer exactly matches the required refusal.",
    )


def _judged(judgements: Sequence[AnswerJudgement]) -> tuple[AnswerJudgement, ...]:
    return tuple(judgement for judgement in judgements if judgement.error is None)


def _mean(values: Iterable[bool]) -> float:
    values = tuple(values)
    return statistics.fmean(values) if values else 0.0
