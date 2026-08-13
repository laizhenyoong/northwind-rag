"""Answer one question with hybrid retrieval, grounding, and source citations."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag.embeddings import OllamaEmbedder
from rag.evaluation import write_traces
from rag.generation import GroundedAnswerer, OllamaChatModel
from rag.generation.pipeline import answer_question
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.keyword import KeywordRetriever
from rag.retrieval.neighbors import NeighborExpandingRetriever
from rag.retrieval.reranked import RerankingRetriever
from rag.retrieval.semantic import SemanticRetriever
from rag.retrieval.version_filters import (
    as_of_date_filter,
    current_version_filter,
    exact_version_filter,
    parse_as_of_date,
)
from rag.reranking import BGEReranker
from rag.query_transformation import (
    MultiHopRetriever,
    OllamaFollowupQueryGenerator,
    OllamaQueryDecomposer,
    QueryDecompositionRetriever,
)
from rag.vector_store import PineconeSettings, ensure_index


def main() -> None:
    """Answer a question and write the complete evidence trace to JSONL."""
    parser = argparse.ArgumentParser(description="Answer from retrieved Northwind documents")
    parser.add_argument("question")
    parser.add_argument("--question-id", default="interactive")
    parser.add_argument("--model", default="gemma4")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--decompose", action="store_true")
    parser.add_argument("--multi-hop", action="store_true")
    parser.add_argument("--coverage-per-query", type=int, default=0)
    parser.add_argument("--neighbor-window", type=int, default=0)
    parser.add_argument("--query-model", default="gemma4")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--document-id")
    version_selection = parser.add_mutually_exclusive_group()
    version_selection.add_argument("--current", action="store_true")
    version_selection.add_argument("--version")
    version_selection.add_argument("--as-of")
    parser.add_argument("--corpus-root", type=Path, default=Path("data"))
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--trace", type=Path, default=Path("results/answer.jsonl"))
    arguments = parser.parse_args()

    if (arguments.current or arguments.version or arguments.as_of) and not arguments.document_id:
        parser.error("--document-id is required with --current, --version, or --as-of")

    metadata_filter = None
    if arguments.current:
        metadata_filter = current_version_filter(arguments.document_id)
    elif arguments.version:
        metadata_filter = exact_version_filter(arguments.document_id, arguments.version)
    elif arguments.as_of:
        try:
            metadata_filter = as_of_date_filter(
                arguments.document_id, parse_as_of_date(arguments.as_of)
            )
        except ValueError as error:
            parser.error(str(error))

    embedder = OllamaEmbedder()
    hybrid_retriever = HybridRetriever(
        semantic_retriever=SemanticRetriever(
            embedder=embedder,
            vector_index=ensure_index(PineconeSettings.from_environment()),
        ),
        keyword_retriever=KeywordRetriever.from_corpus(
            arguments.corpus_root, chunk_size=arguments.chunk_size
        ),
        candidate_k=arguments.candidate_k,
    )
    candidate_retriever = (
        QueryDecompositionRetriever(
            candidate_retriever=hybrid_retriever,
            decomposer=OllamaQueryDecomposer(OllamaChatModel(model=arguments.query_model)),
            candidate_k=arguments.candidate_k,
            coverage_per_query=arguments.coverage_per_query,
        )
        if arguments.decompose
        else hybrid_retriever
    )
    candidate_retriever = (
        MultiHopRetriever(
            candidate_retriever=candidate_retriever,
            followup_generator=OllamaFollowupQueryGenerator(OllamaChatModel(model=arguments.query_model)),
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
        NeighborExpandingRetriever.from_corpus(
            retriever,
            arguments.corpus_root,
            chunk_size=arguments.chunk_size,
            neighbor_window=arguments.neighbor_window,
        )
        if arguments.neighbor_window
        else retriever
    )
    run = answer_question(
        question_id=arguments.question_id,
        question=arguments.question,
        retriever=retriever,
        answerer=GroundedAnswerer(OllamaChatModel(model=arguments.model)),
        top_k=arguments.top_k,
        metadata_filter=metadata_filter,
        pipeline_config={
            "retrieval_strategy": "semantic-bm25-rrf",
            "embedding_model": embedder.model,
            "generation_model": arguments.model,
            "chunk_size": arguments.chunk_size,
            "candidate_k": arguments.candidate_k,
            "rrf_k": 60,
            "reranker_model": arguments.reranker_model if arguments.rerank else None,
            "query_model": arguments.query_model if arguments.decompose else None,
            "coverage_per_query": arguments.coverage_per_query if arguments.decompose else None,
            "multi_hop": arguments.multi_hop,
            "neighbor_window": arguments.neighbor_window or None,
        },
    )
    write_traces(arguments.trace, [run.trace])

    if run.trace.error:
        print(f"Error: {run.trace.error}")
        return

    print(run.trace.answer)
    if run.generated_answer and run.generated_answer.cited_chunk_ids:
        print("\nCited chunks:")
        for chunk_id in run.generated_answer.cited_chunk_ids:
            print(f"- {chunk_id}")
    print(f"\nTrace: {arguments.trace}")


if __name__ == "__main__":
    main()
