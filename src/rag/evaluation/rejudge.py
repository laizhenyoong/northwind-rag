"""Re-judge saved answer traces without re-running retrieval or generation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rag.evaluation.answer_judge import (
    AnswerJudgementEvaluation,
    ChatModel,
    SemanticAnswerJudge,
    evaluate_semantic_answers,
    write_judgements,
)
from rag.evaluation.judge_agreement import (
    AgreementReport,
    compare_judgements,
    load_judgements,
)
from rag.evaluation.questions import GoldQuestion, load_gold_questions
from rag.evaluation.traces import RunTrace, load_traces
from rag.generation import BedrockChatModel, DeepSeekChatModel, OllamaChatModel


def rejudge_traces(
    questions: Sequence[GoldQuestion],
    traces: Sequence[RunTrace],
    *,
    judge: SemanticAnswerJudge,
    workers: int = 1,
) -> AnswerJudgementEvaluation:
    """Judge saved traces, scoring only the questions the trace file covers.

    A trace file written with ``--limit`` holds a subset of the gold set, so the
    questions are narrowed to match rather than treated as missing evidence.
    """
    covered_ids = {trace.question_id for trace in traces}
    covered_questions = [question for question in questions if question.id in covered_ids]
    if not covered_questions:
        raise ValueError("No gold question matches any trace in the file")

    return evaluate_semantic_answers(covered_questions, traces, judge=judge, workers=workers)


def main() -> None:
    """Re-score answers already on disk, optionally against an earlier judge."""
    parser = argparse.ArgumentParser(
        description="Re-judge saved answer traces with a chosen judge model"
    )
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=Path("eval/gold_questions.jsonl"))
    parser.add_argument("--judge-provider", choices=("ollama", "deepseek", "bedrock"), default="ollama")
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-timeout", type=float, default=120.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Judge only the first N saved traces.")
    parser.add_argument(
        "--compare-with",
        type=Path,
        help="An existing judgements file to report agreement against.",
    )
    arguments = parser.parse_args()

    questions = load_gold_questions(arguments.questions)
    traces = load_traces(arguments.traces)
    if arguments.limit is not None:
        if arguments.limit < 1:
            parser.error("--limit must be at least 1")
        traces = traces[: arguments.limit]
    chat_model = _chat_model(arguments)
    evaluation = rejudge_traces(
        questions, traces, judge=SemanticAnswerJudge(chat_model), workers=arguments.workers
    )
    write_judgements(arguments.output, evaluation.judgements)

    summary = evaluation.summary
    print(
        f"Wrote {summary.question_count} judgements from {arguments.traces} "
        f"to {arguments.output} using {chat_model.model}. "
        f"correctness_rate={summary.correctness_rate:.3f} "
        f"citation_supported_rate={summary.citation_supported_rate:.3f} "
        f"correct_version_rate={summary.correct_version_rate:.3f} "
        f"proper_refusal_rate={summary.proper_refusal_rate:.3f} "
        f"judged={summary.judged_answerable_question_count}"
        f"/{summary.judged_unanswerable_question_count} "
        f"errors={summary.error_count} "
        f"(pipeline={summary.pipeline_error_count} judge={summary.judge_error_count})"
    )

    if arguments.compare_with is not None:
        _print_agreement(
            compare_judgements(load_judgements(arguments.compare_with), evaluation.judgements),
            baseline_path=arguments.compare_with,
            comparison_model=chat_model.model,
        )


def _chat_model(arguments: argparse.Namespace) -> ChatModel:
    """Build the judge's chat model for the chosen provider."""
    if arguments.judge_provider == "deepseek":
        return DeepSeekChatModel.from_environment(
            model=arguments.judge_model, timeout_seconds=arguments.judge_timeout
        )
    if arguments.judge_provider == "bedrock":
        return BedrockChatModel(
            model=arguments.judge_model or BedrockChatModel.model,
            timeout_seconds=arguments.judge_timeout,
        )
    return OllamaChatModel(
        model=arguments.judge_model or "gemma4", timeout_seconds=arguments.judge_timeout
    )


def _print_agreement(
    report: AgreementReport, *, baseline_path: Path, comparison_model: str
) -> None:
    print(
        f"\nAgreement between {baseline_path} and {comparison_model} "
        f"over {report.compared_count} questions "
        f"({len(report.skipped_question_ids)} skipped as unscored):"
    )
    for label in report.labels:
        print(
            f"  {label.label:<22} agreement={label.agreement_rate:.3f} "
            f"kappa={label.cohen_kappa:+.3f} "
            f"both_true={label.both_true} both_false={label.both_false} "
            f"baseline_only={label.only_baseline_true} "
            f"comparison_only={label.only_comparison_true}"
        )


if __name__ == "__main__":
    main()
