"""Run and measure the semantic-retrieval baseline over the gold questions."""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from rag.embeddings import OllamaEmbedder
from rag.evaluation.questions import GoldQuestion, load_gold_questions
from rag.evaluation.retrieval_metrics import RetrievalMetrics, score_retrieval
from rag.evaluation.traces import RunTrace, write_traces
from rag.retrieval.semantic import RetrievedPassage, SemanticRetriever
from rag.vector_store import PineconeSettings, ensure_index


class Retriever(Protocol):
    """The portion of a retriever required for evaluation."""

    def retrieve(self, question: str, *, top_k: int) -> list[RetrievedPassage]: ...


@dataclass(frozen=True, slots=True)
class RetrievalSummary:
    """Aggregate document-level scores for one retrieval run."""

    question_count: int
    answerable_question_count: int
    error_count: int
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_reciprocal_rank: float
    mean_ndcg_at_k: float


@dataclass(frozen=True, slots=True)
class RetrievalEvaluation:
    """Inspectable per-question traces and their aggregate summary."""

    traces: tuple[RunTrace, ...]
    summary: RetrievalSummary


def evaluate_retrieval(
    questions: Sequence[GoldQuestion],
    *,
    retriever: Retriever,
    top_k: int,
    pipeline_config: dict[str, Any],
) -> RetrievalEvaluation:
    """Retrieve every question and retain one complete trace per question."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    traces: list[RunTrace] = []
    metrics: list[RetrievalMetrics] = []

    for gold_question in questions:
        started_at = time.perf_counter()
        error: str | None = None
        retrieved_chunks = ()

        try:
            passages = retriever.retrieve(gold_question.question, top_k=top_k)
            retrieved_chunks = tuple(
                passage.as_trace_chunk(rank=rank)
                for rank, passage in enumerate(passages, start=1)
            )
        except Exception as exception:  # Save the failure as an inspectable trace.
            error = f"{type(exception).__name__}: {exception}"

        timing_ms = round((time.perf_counter() - started_at) * 1000)
        traces.append(
            RunTrace(
                question_id=gold_question.id,
                question=gold_question.question,
                pipeline_config={**pipeline_config, "top_k": top_k},
                retrieved_chunks=retrieved_chunks,
                timing_ms=timing_ms,
                error=error,
            )
        )

        retrieval_metrics = score_retrieval(gold_question, retrieved_chunks, k=top_k)
        if retrieval_metrics is not None:
            metrics.append(retrieval_metrics)

    return RetrievalEvaluation(
        traces=tuple(traces),
        summary=_summarize(questions, metrics, traces),
    )


def _summarize(
    questions: Sequence[GoldQuestion],
    metrics: list[RetrievalMetrics],
    traces: list[RunTrace],
) -> RetrievalSummary:
    return RetrievalSummary(
        question_count=len(questions),
        answerable_question_count=sum(question.answerable for question in questions),
        error_count=sum(trace.error is not None for trace in traces),
        mean_precision_at_k=_mean(metric.precision_at_k for metric in metrics),
        mean_recall_at_k=_mean(metric.recall_at_k for metric in metrics),
        mean_reciprocal_rank=_mean(metric.reciprocal_rank for metric in metrics),
        mean_ndcg_at_k=_mean(metric.ndcg_at_k for metric in metrics),
    )


def _mean(values: Any) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def main() -> None:
    """Run the baseline retrieval evaluation from the project root."""
    parser = argparse.ArgumentParser(description="Evaluate Pinecone semantic retrieval")
    parser.add_argument("--questions", type=Path, default=Path("eval/gold_questions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/retrieval-baseline.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    arguments = parser.parse_args()

    embedder = OllamaEmbedder()
    retriever = SemanticRetriever(
        embedder=embedder,
        vector_index=ensure_index(PineconeSettings.from_environment()),
    )
    evaluation = evaluate_retrieval(
        load_gold_questions(arguments.questions),
        retriever=retriever,
        top_k=arguments.top_k,
        pipeline_config={
            "retrieval_strategy": "pinecone-semantic",
            "embedding_model": embedder.model,
            "chunk_size": 500,
            "metadata_filter": None,
        },
    )
    write_traces(arguments.output, list(evaluation.traces))
    summary = evaluation.summary
    print(
        f"Wrote {summary.question_count} traces to {arguments.output}. "
        f"P@{arguments.top_k}={summary.mean_precision_at_k:.3f} "
        f"Recall@{arguments.top_k}={summary.mean_recall_at_k:.3f} "
        f"MRR={summary.mean_reciprocal_rank:.3f} "
        f"nDCG@{arguments.top_k}={summary.mean_ndcg_at_k:.3f} "
        f"errors={summary.error_count}"
    )


if __name__ == "__main__":
    main()
