"""Generate and evaluate grounded answers for the gold-question set."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.embeddings import OllamaEmbedder
from rag.evaluation.answer_metrics import AnswerEvaluation, evaluate_answers
from rag.evaluation.questions import GoldQuestion, load_gold_questions
from rag.evaluation.traces import RunTrace, write_traces
from rag.generation import GroundedAnswerer, OllamaChatModel
from rag.generation.pipeline import Answerer, Retriever, answer_question
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.keyword import KeywordRetriever
from rag.retrieval.reranked import RerankingRetriever
from rag.retrieval.semantic import SemanticRetriever
from rag.reranking import BGEReranker
from rag.query_transformation import OllamaQueryDecomposer, QueryDecompositionRetriever
from rag.vector_store import PineconeSettings, ensure_index


@dataclass(frozen=True, slots=True)
class AnswerRunEvaluation:
    """Generated traces and deterministic checks for one answer run."""

    traces: tuple[RunTrace, ...]
    evaluation: AnswerEvaluation


def run_answer_evaluation(
    questions: Sequence[GoldQuestion],
    *,
    retriever: Retriever,
    answerer: Answerer,
    top_k: int,
    pipeline_config: dict[str, Any],
) -> AnswerRunEvaluation:
    """Answer every gold question and evaluate the persisted evidence."""
    traces = tuple(
        answer_question(
            question_id=question.id,
            question=question.question,
            retriever=retriever,
            answerer=answerer,
            top_k=top_k,
            pipeline_config=pipeline_config,
        ).trace
        for question in questions
    )
    return AnswerRunEvaluation(traces=traces, evaluation=evaluate_answers(questions, traces))


def main() -> None:
    """Run the full answer evaluation against Pinecone and local Ollama."""
    parser = argparse.ArgumentParser(description="Evaluate grounded RAG answers")
    parser.add_argument("--questions", type=Path, default=Path("eval/gold_questions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/answer-evaluation.jsonl"))
    parser.add_argument("--model", default="gemma4")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--decompose", action="store_true")
    parser.add_argument("--query-model", default="gemma4")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args()

    questions = load_gold_questions(arguments.questions)
    if arguments.limit is not None:
        if arguments.limit < 1:
            parser.error("--limit must be at least 1")
        questions = questions[: arguments.limit]

    embedder = OllamaEmbedder()
    hybrid_retriever = HybridRetriever(
        semantic_retriever=SemanticRetriever(
            embedder=embedder,
            vector_index=ensure_index(PineconeSettings.from_environment()),
        ),
        keyword_retriever=KeywordRetriever.from_corpus(
            Path("data"), chunk_size=arguments.chunk_size
        ),
        candidate_k=arguments.candidate_k,
    )
    candidate_retriever = (
        QueryDecompositionRetriever(
            candidate_retriever=hybrid_retriever,
            decomposer=OllamaQueryDecomposer(OllamaChatModel(model=arguments.query_model)),
            candidate_k=arguments.candidate_k,
        )
        if arguments.decompose
        else hybrid_retriever
    )
    retriever = (
        RerankingRetriever(
            candidate_retriever=candidate_retriever,
            reranker=BGEReranker(model_name=arguments.reranker_model),
            candidate_k=arguments.candidate_k,
        )
        if arguments.rerank
        else candidate_retriever
    )
    evaluation = run_answer_evaluation(
        questions,
        retriever=retriever,
        answerer=GroundedAnswerer(OllamaChatModel(model=arguments.model)),
        top_k=arguments.top_k,
        pipeline_config={
            "retrieval_strategy": "semantic-bm25-rrf",
            "embedding_model": embedder.model,
            "generation_model": arguments.model,
            "chunk_size": arguments.chunk_size,
            "candidate_k": arguments.candidate_k,
            "rrf_k": 60,
            "reranker_model": arguments.reranker_model if arguments.rerank else None,
            "query_model": arguments.query_model if arguments.decompose else None,
        },
    )
    write_traces(arguments.output, list(evaluation.traces))
    summary = evaluation.evaluation.summary
    print(
        f"Wrote {summary.question_count} answer traces to {arguments.output}. "
        f"answer_rate={summary.answer_rate:.3f} "
        f"citation_valid_rate={summary.citation_valid_rate:.3f} "
        f"expected_source_citation_rate={summary.expected_source_citation_rate:.3f} "
        f"refusal_rate={summary.refusal_rate:.3f} "
        f"errors={summary.error_count}"
    )


if __name__ == "__main__":
    main()
