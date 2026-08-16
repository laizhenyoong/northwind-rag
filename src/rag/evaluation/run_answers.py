"""Generate and evaluate grounded answers for the gold-question set."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.chunking import Chunk, chunk_corpus
from rag.chunking.semantic import chunk_semantic_corpus
from rag.embeddings import OllamaEmbedder
from rag.evaluation.answer_metrics import AnswerEvaluation, evaluate_answers
from rag.evaluation.answer_judge import (
    AnswerJudgementEvaluation,
    SemanticAnswerJudge,
    evaluate_semantic_answers,
    write_judgements,
)
from rag.evaluation.questions import GoldQuestion, load_gold_questions
from rag.evaluation.traces import RunTrace, write_traces
from rag.ingestion import load_corpus
from rag.generation import DeepSeekChatModel, GroundedAnswerer, OllamaChatModel
from rag.generation.pipeline import Answerer, Retriever, answer_question
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.keyword import KeywordRetriever
from rag.retrieval.neighbors import NeighborExpandingRetriever
from rag.retrieval.reranked import RerankingRetriever
from rag.retrieval.semantic import SemanticRetriever
from rag.reranking import BGEReranker
from rag.query_transformation import (
    MultiHopRetriever,
    OllamaFollowupQueryGenerator,
    OllamaQueryDecomposer,
    QueryDecompositionRetriever,
)
from rag.vector_store import PineconeSettings, ensure_index


@dataclass(frozen=True, slots=True)
class AnswerRunEvaluation:
    """Generated traces and deterministic checks for one answer run."""

    traces: tuple[RunTrace, ...]
    evaluation: AnswerEvaluation
    semantic_evaluation: AnswerJudgementEvaluation | None = None


def run_answer_evaluation(
    questions: Sequence[GoldQuestion],
    *,
    retriever: Retriever,
    answerer: Answerer,
    top_k: int,
    pipeline_config: dict[str, Any],
    semantic_judge: SemanticAnswerJudge | None = None,
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
    return AnswerRunEvaluation(
        traces=traces,
        evaluation=evaluate_answers(questions, traces),
        semantic_evaluation=(
            evaluate_semantic_answers(questions, traces, judge=semantic_judge)
            if semantic_judge is not None
            else None
        ),
    )


def chat_model(provider: str, model: str | None) -> Any:
    """Build a chat model for the chosen provider, letting it pick its own default."""
    if provider == "deepseek":
        return DeepSeekChatModel.from_environment(model=model, timeout_seconds=600.0)
    return OllamaChatModel(model=model or "gemma4")


def corpus_chunks(arguments: argparse.Namespace, embedder: OllamaEmbedder) -> tuple[Chunk, ...]:
    """Rebuild precisely the chunk layout the chosen Pinecone namespace was indexed with.

    BM25 and neighbor expansion both read chunks from disk rather than Pinecone,
    so a mismatch here would silently score one chunking strategy against
    another's vectors.
    """
    documents = load_corpus(Path("data"))
    if arguments.chunking_strategy == "semantic":
        return tuple(
            chunk_semantic_corpus(
                documents,
                embedder=embedder,
                chunk_size=arguments.chunk_size,
                min_chunk_size=arguments.min_chunk_size,
                similarity_threshold=arguments.similarity_threshold,
            )
        )
    return tuple(chunk_corpus(documents, chunk_size=arguments.chunk_size))


def main() -> None:
    """Run the answer evaluation against Pinecone, with local or hosted generation."""
    parser = argparse.ArgumentParser(description="Evaluate grounded RAG answers")
    parser.add_argument("--questions", type=Path, default=Path("eval/gold_questions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/answer-evaluation.jsonl"))
    parser.add_argument(
        "--judgements-output", type=Path, default=Path("results/answer-judgements.jsonl")
    )
    parser.add_argument("--provider", choices=("ollama", "deepseek"), default="ollama")
    parser.add_argument("--model")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--decompose", action="store_true")
    parser.add_argument("--multi-hop", action="store_true")
    parser.add_argument("--coverage-per-query", type=int, default=0)
    parser.add_argument("--neighbor-window", type=int, default=0)
    parser.add_argument(
        "--neighbor-source-k",
        type=int,
        help="Expand neighbors only for this many top-ranked chunks.",
    )
    parser.add_argument("--query-model")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument(
        "--chunking-strategy", choices=("markdown-aware", "semantic"), default="markdown-aware"
    )
    parser.add_argument("--namespace", default="")
    parser.add_argument("--min-chunk-size", type=int, default=200)
    parser.add_argument("--similarity-threshold", type=float, default=0.75)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--judge-model")
    parser.add_argument("--skip-semantic-judge", action="store_true")
    arguments = parser.parse_args()

    questions = load_gold_questions(arguments.questions)
    if arguments.limit is not None:
        if arguments.limit < 1:
            parser.error("--limit must be at least 1")
        questions = questions[: arguments.limit]
    if arguments.neighbor_source_k is not None and arguments.neighbor_window < 1:
        parser.error("--neighbor-source-k requires --neighbor-window")

    answer_model = chat_model(arguments.provider, arguments.model)
    query_model = chat_model(arguments.provider, arguments.query_model)

    embedder = OllamaEmbedder()
    chunks = corpus_chunks(arguments, embedder)
    hybrid_retriever = HybridRetriever(
        semantic_retriever=SemanticRetriever(
            embedder=embedder,
            vector_index=ensure_index(PineconeSettings.from_environment()),
            namespace=arguments.namespace,
        ),
        keyword_retriever=KeywordRetriever(chunks),
        candidate_k=arguments.candidate_k,
    )
    candidate_retriever = (
        QueryDecompositionRetriever(
            candidate_retriever=hybrid_retriever,
            decomposer=OllamaQueryDecomposer(query_model),
            candidate_k=arguments.candidate_k,
            coverage_per_query=arguments.coverage_per_query,
        )
        if arguments.decompose
        else hybrid_retriever
    )
    candidate_retriever = (
        MultiHopRetriever(
            candidate_retriever=candidate_retriever,
            followup_generator=OllamaFollowupQueryGenerator(query_model),
            candidate_k=arguments.candidate_k,
        )
        if arguments.multi_hop
        else candidate_retriever
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
    retriever = (
        NeighborExpandingRetriever(
            candidate_retriever=retriever,
            chunks=chunks,
            neighbor_window=arguments.neighbor_window,
            neighbor_source_k=arguments.neighbor_source_k,
        )
        if arguments.neighbor_window
        else retriever
    )
    evaluation = run_answer_evaluation(
        questions,
        retriever=retriever,
        answerer=GroundedAnswerer(answer_model),
        top_k=arguments.top_k,
        pipeline_config={
            "retrieval_strategy": "semantic-bm25-rrf",
            "embedding_model": embedder.model,
            "generation_model": answer_model.model,
            "chunking_strategy": arguments.chunking_strategy,
            "namespace": arguments.namespace,
            "chunk_size": arguments.chunk_size,
            "candidate_k": arguments.candidate_k,
            "rrf_k": 60,
            "reranker_model": arguments.reranker_model if arguments.rerank else None,
            "query_model": query_model.model if arguments.decompose else None,
            "coverage_per_query": arguments.coverage_per_query if arguments.decompose else None,
            "multi_hop": arguments.multi_hop,
            "neighbor_window": arguments.neighbor_window or None,
            "neighbor_source_k": arguments.neighbor_source_k,
            "judge_model": None if arguments.skip_semantic_judge else arguments.judge_model,
        },
        semantic_judge=(
            None
            if arguments.skip_semantic_judge
            else SemanticAnswerJudge(chat_model(arguments.provider, arguments.judge_model))
        ),
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
    if evaluation.semantic_evaluation is not None:
        write_judgements(arguments.judgements_output, evaluation.semantic_evaluation.judgements)
        semantic = evaluation.semantic_evaluation.summary
        print(
            f"Wrote {semantic.question_count} semantic judgements to {arguments.judgements_output}. "
            f"correctness_rate={semantic.correctness_rate:.3f} "
            f"citation_supported_rate={semantic.citation_supported_rate:.3f} "
            f"correct_version_rate={semantic.correct_version_rate:.3f} "
            f"proper_refusal_rate={semantic.proper_refusal_rate:.3f} "
            f"judged={semantic.judged_answerable_question_count}"
            f"/{semantic.judged_unanswerable_question_count} "
            f"errors={semantic.error_count} "
            f"(pipeline={semantic.pipeline_error_count} judge={semantic.judge_error_count})"
        )


if __name__ == "__main__":
    main()
