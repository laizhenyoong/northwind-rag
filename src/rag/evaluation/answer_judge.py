"""Semantic answer judging against gold answers and retrieved evidence."""

from __future__ import annotations

import json
import re
import statistics
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, Sequence

from rag.evaluation.questions import GoldQuestion
from rag.evaluation.traces import RunTrace


JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for a grounded RAG system.
Judge only from the supplied question, gold answer, generated answer, and retrieved
context. Do not use outside knowledge. Return exactly one JSON object, with no
markdown or extra text, matching this schema:
{
  "correct": boolean,
  "citation_supported": boolean,
  "used_correct_version": boolean,
  "proper_refusal": boolean,
  "reason": string
}

For an answerable question, correct means the generated answer reaches the same
material conclusion as the gold answer. Different wording is allowed, but wrong
numbers, names, dates, policy versions, or missing required facts make it false.
An answer is also incorrect when it introduces a conflicting historical, superseded,
or irrelevant value that could mislead the user, unless the question explicitly asks
for a comparison or historical information.
For an unanswerable question, correct and proper_refusal are true only when the
generated answer refuses rather than inventing an answer. For answerable questions,
proper_refusal must be false.

citation_supported is true only when the citations used by the generated answer
actually support its factual claims in the supplied context. used_correct_version is
true only when it uses the version or effective-date interpretation required by the
gold answer. If a judgement cannot be supported from the supplied material, mark it
false and explain why."""


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


@dataclass(frozen=True, slots=True)
class AnswerJudgementSummary:
    """Aggregate semantic quality rates, separated by answerability."""

    question_count: int
    answerable_question_count: int
    unanswerable_question_count: int
    error_count: int
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
            return _failed_judgement(question.id, f"Pipeline error: {trace.error}")

        try:
            response = self.chat_model.complete(
                system_prompt=JUDGE_SYSTEM_PROMPT,
                user_prompt=_build_judge_prompt(question, trace),
            )
            values = _parse_judgement(response)
        except (RuntimeError, ValueError, json.JSONDecodeError) as error:
            return _failed_judgement(question.id, str(error))

        return AnswerJudgement(question_id=question.id, **values)


def evaluate_semantic_answers(
    questions: Sequence[GoldQuestion],
    traces: Sequence[RunTrace],
    *,
    judge: SemanticAnswerJudge,
) -> AnswerJudgementEvaluation:
    """Judge every trace and compute answerable and refusal quality rates."""
    traces_by_question_id = {trace.question_id: trace for trace in traces}
    missing = [question.id for question in questions if question.id not in traces_by_question_id]
    if missing:
        raise ValueError(f"Missing answer traces for: {', '.join(missing)}")

    judgements = tuple(judge.judge(question, traces_by_question_id[question.id]) for question in questions)
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

    return AnswerJudgementEvaluation(
        judgements=judgements,
        summary=AnswerJudgementSummary(
            question_count=len(questions),
            answerable_question_count=len(answerable),
            unanswerable_question_count=len(unanswerable),
            error_count=sum(judgement.error is not None for judgement in judgements),
            correctness_rate=_mean(judgement.correct for judgement in answerable),
            citation_supported_rate=_mean(
                judgement.citation_supported for judgement in answerable
            ),
            correct_version_rate=_mean(
                judgement.used_correct_version for judgement in answerable
            ),
            proper_refusal_rate=_mean(judgement.proper_refusal for judgement in unanswerable),
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


def _failed_judgement(question_id: str, error: str) -> AnswerJudgement:
    return AnswerJudgement(
        question_id=question_id,
        correct=False,
        citation_supported=False,
        used_correct_version=False,
        proper_refusal=False,
        reason="No semantic judgement was produced.",
        error=error,
    )


def _mean(values: Iterable[bool]) -> float:
    values = tuple(values)
    return statistics.fmean(values) if values else 0.0
