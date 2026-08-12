"""Run and measure the semantic-retrieval baseline over the gold questions."""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from rag.chunking.semantic import chunk_semantic_corpus
from rag.embeddings import OllamaEmbedder
from rag.evaluation.questions import GoldQuestion, load_gold_questions
from rag.evaluation.retrieval_metrics import RetrievalMetrics, score_retrieval
from rag.evaluation.traces import RunTrace, write_traces
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.keyword import KeywordRetriever
from rag.retrieval.reranked import RerankingRetriever
from rag.retrieval.semantic import RetrievedPassage, SemanticRetriever
from rag.reranking import BGEReranker
from rag.query_transformation import (
    MultiHopRetriever,
    OllamaFollowupQueryGenerator,
    OllamaQueryDecomposer,
    QueryDecompositionRetriever,
)
from rag.generation.ollama import OllamaChatModel
from rag.ingestion import load_corpus
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


def default_output_path(strategy: str) -> Path:
    """Keep independent trace files for each retrieval experiment."""
    paths = {
        "semantic": Path("results/retrieval-baseline.jsonl"),
        "keyword": Path("results/retrieval-keyword.jsonl"),
        "hybrid": Path("results/retrieval-hybrid.jsonl"),
        "hybrid-reranked": Path("results/retrieval-hybrid-reranked.jsonl"),
        "hybrid-decomposed": Path("results/retrieval-hybrid-decomposed.jsonl"),
        "hybrid-decomposed-multihop": Path("results/retrieval-hybrid-decomposed-multihop.jsonl"),
        "hybrid-decomposed-reranked": Path("results/retrieval-hybrid-decomposed-reranked.jsonl"),
    }
    try:
        return paths[strategy]
    except KeyError as error:
        raise ValueError(f"Unknown retrieval strategy: {strategy}") from error


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
            queries_used = tuple(getattr(retriever, "last_queries", ()) or (gold_question.question,))
        except Exception as exception:  # Save the failure as an inspectable trace.
            error = f"{type(exception).__name__}: {exception}"
            queries_used = (gold_question.question,)

        timing_ms = round((time.perf_counter() - started_at) * 1000)
        traces.append(
            RunTrace(
                question_id=gold_question.id,
                question=gold_question.question,
                pipeline_config={**pipeline_config, "top_k": top_k},
                retrieved_chunks=retrieved_chunks,
                queries_used=queries_used,
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
    parser.add_argument(
        "--strategy",
        choices=(
            "semantic",
            "keyword",
            "hybrid",
            "hybrid-reranked",
            "hybrid-decomposed",
            "hybrid-decomposed-multihop",
            "hybrid-decomposed-reranked",
        ),
        default="semantic",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument(
        "--chunking-strategy", choices=("markdown-aware", "semantic"), default="markdown-aware"
    )
    parser.add_argument("--namespace", default="")
    parser.add_argument("--min-chunk-size", type=int, default=200)
    parser.add_argument("--similarity-threshold", type=float, default=0.75)
    parser.add_argument("--coverage-per-query", type=int, default=0)
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--query-model", default="gemma4")
    arguments = parser.parse_args()

    default_output = default_output_path(arguments.strategy)
    output_path = arguments.output or default_output
    if arguments.strategy in {
        "semantic",
        "hybrid",
        "hybrid-reranked",
        "hybrid-decomposed",
        "hybrid-decomposed-multihop",
        "hybrid-decomposed-reranked",
    }:
        embedder = OllamaEmbedder()
        semantic_retriever: Retriever = SemanticRetriever(
            embedder=embedder,
            vector_index=ensure_index(PineconeSettings.from_environment()),
            namespace=arguments.namespace,
        )
        if arguments.strategy == "semantic":
            retriever = semantic_retriever
            pipeline_config = {
                "retrieval_strategy": "pinecone-semantic",
                "embedding_model": embedder.model,
                "chunk_size": arguments.chunk_size,
                "chunking_strategy": arguments.chunking_strategy,
                "namespace": arguments.namespace,
                "metadata_filter": None,
            }
        else:
            hybrid_retriever = HybridRetriever(
                semantic_retriever=semantic_retriever,
                keyword_retriever=_keyword_retriever(arguments, embedder),
            )
            pipeline_config: dict[str, Any] = {
                "retrieval_strategy": "semantic-bm25-rrf",
                "embedding_model": embedder.model,
                "chunk_size": arguments.chunk_size,
                "chunking_strategy": arguments.chunking_strategy,
                "namespace": arguments.namespace,
                "candidate_k": 20,
                "rrf_k": 60,
                "metadata_filter": None,
            }
            if arguments.strategy in {
                "hybrid-decomposed",
                "hybrid-decomposed-multihop",
                "hybrid-decomposed-reranked",
            }:
                hybrid_retriever = QueryDecompositionRetriever(
                    candidate_retriever=hybrid_retriever,
                    decomposer=OllamaQueryDecomposer(OllamaChatModel(model=arguments.query_model)),
                    coverage_per_query=arguments.coverage_per_query,
                )
                pipeline_config.update(
                    {
                        "retrieval_strategy": "semantic-bm25-rrf-query-decomposition",
                        "query_model": arguments.query_model,
                        "coverage_per_query": arguments.coverage_per_query,
                    }
                )
            if arguments.strategy == "hybrid-decomposed-multihop":
                hybrid_retriever = MultiHopRetriever(
                    candidate_retriever=hybrid_retriever,
                    followup_generator=OllamaFollowupQueryGenerator(
                        OllamaChatModel(model=arguments.query_model)
                    ),
                )
                pipeline_config.update(
                    {
                        "retrieval_strategy": "semantic-bm25-rrf-query-decomposition-multihop",
                        "multi_hop": True,
                    }
                )
            if arguments.strategy in {"hybrid-reranked", "hybrid-decomposed-reranked"}:
                retriever = RerankingRetriever(
                    candidate_retriever=hybrid_retriever,
                    reranker=BGEReranker(model_name=arguments.reranker_model),
                )
                pipeline_config.update(
                    {
                        "retrieval_strategy": "semantic-bm25-rrf-bge-reranker",
                        "reranker_model": arguments.reranker_model,
                    }
                )
            else:
                retriever = hybrid_retriever
    else:
        retriever = _keyword_retriever(arguments, OllamaEmbedder())
        pipeline_config = {
            "retrieval_strategy": "bm25-keyword",
            "chunk_size": arguments.chunk_size,
            "chunking_strategy": arguments.chunking_strategy,
        }

    evaluation = evaluate_retrieval(
        load_gold_questions(arguments.questions),
        retriever=retriever,
        top_k=arguments.top_k,
        pipeline_config=pipeline_config,
    )
    write_traces(output_path, list(evaluation.traces))
    summary = evaluation.summary
    print(
        f"Wrote {summary.question_count} traces to {output_path}. "
        f"P@{arguments.top_k}={summary.mean_precision_at_k:.3f} "
        f"Recall@{arguments.top_k}={summary.mean_recall_at_k:.3f} "
        f"MRR={summary.mean_reciprocal_rank:.3f} "
        f"nDCG@{arguments.top_k}={summary.mean_ndcg_at_k:.3f} "
        f"errors={summary.error_count}"
    )


def _keyword_retriever(arguments: argparse.Namespace, embedder: OllamaEmbedder) -> KeywordRetriever:
    """Build BM25 from precisely the same chunking strategy as Pinecone."""
    if arguments.chunking_strategy == "markdown-aware":
        return KeywordRetriever.from_corpus(Path("data"), chunk_size=arguments.chunk_size)
    chunks = chunk_semantic_corpus(
        load_corpus(Path("data")),
        embedder=embedder,
        chunk_size=arguments.chunk_size,
        min_chunk_size=arguments.min_chunk_size,
        similarity_threshold=arguments.similarity_threshold,
    )
    return KeywordRetriever(tuple(chunks))


if __name__ == "__main__":
    main()
