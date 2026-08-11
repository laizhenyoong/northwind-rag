"""Index semantic chunks in an isolated namespace for comparison experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from rag.chunking.semantic import chunk_semantic_document
from rag.embeddings import OllamaEmbedder
from rag.sync import synchronize_corpus
from rag.vector_store import PineconeSettings, ensure_index


def main() -> None:
    """Create or update the semantic-chunking experiment index."""
    parser = argparse.ArgumentParser(description="Synchronize semantic chunks to Pinecone")
    parser.add_argument("--corpus-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--manifest", type=Path, default=Path(".rag/semantic-ingestion-manifest.json")
    )
    parser.add_argument("--namespace", default="semantic-v1")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--min-chunk-size", type=int, default=200)
    parser.add_argument("--similarity-threshold", type=float, default=0.75)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    embedder = OllamaEmbedder()
    chunker = lambda document: chunk_semantic_document(
        document,
        embedder=embedder,
        chunk_size=arguments.chunk_size,
        min_chunk_size=arguments.min_chunk_size,
        similarity_threshold=arguments.similarity_threshold,
    )
    vector_index: Any = (
        object() if arguments.dry_run else ensure_index(PineconeSettings.from_environment())
    )
    plan = synchronize_corpus(
        arguments.corpus_root,
        manifest_path=arguments.manifest,
        embedder=embedder,
        vector_index=vector_index,
        index_config={
            "embedding_model": embedder.model,
            "chunking_strategy": "semantic-v1",
            "chunk_size": arguments.chunk_size,
            "min_chunk_size": arguments.min_chunk_size,
            "similarity_threshold": arguments.similarity_threshold,
        },
        chunk_size=arguments.chunk_size,
        batch_size=arguments.batch_size,
        namespace=arguments.namespace,
        dry_run=arguments.dry_run,
        chunker=chunker,
    )
    for action in plan.actions:
        print(f"{action.kind.upper():9} {action.source_path}")
    if arguments.dry_run:
        print("Dry run: no Pinecone records or manifest were changed.")


if __name__ == "__main__":
    main()
