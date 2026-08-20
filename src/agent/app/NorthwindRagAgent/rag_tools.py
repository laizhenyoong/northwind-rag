"""Agent tools for searching, inspecting and reading the Northwind corpus."""

from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from strands import tool


logger = logging.getLogger(__name__)

CORPUS_ROOT_ENV = "NORTHWIND_CORPUS_ROOT"

# Must match the indexing run that populated Pinecone. BM25 chunks are rebuilt
# here at query time, and a different size yields chunk ids that do not line up
# with the semantic ones, which degrades fusion silently.
CHUNK_SIZE = 500
CANDIDATE_K = 20

MAX_TOP_K = 20
DEFAULT_READ_CHARS = 4_000
MAX_READ_CHARS = 20_000


class CorpusUnavailable(RuntimeError):
    """The repository checkout backing these tools could not be located."""


@lru_cache(maxsize=1)
def _repo_root() -> Path | None:
    """Return the checkout that contains the rag package, if there is one."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "rag").is_dir():
            return parent
    return None


@lru_cache(maxsize=1)
def _corpus_root() -> Path:
    """Return the corpus directory named by the environment or the checkout."""
    override = os.environ.get(CORPUS_ROOT_ENV, "").strip()
    if override:
        root = Path(override).expanduser().resolve()
    else:
        repo_root = _repo_root()
        if repo_root is None:
            raise CorpusUnavailable(
                "The Northwind corpus is not reachable from this process. "
                f"Set {CORPUS_ROOT_ENV} to the absolute path of the corpus directory."
            )
        root = repo_root / "data"
    if not root.is_dir():
        raise CorpusUnavailable(f"Corpus root {root} does not exist or is not a directory")
    return root


def _import_rag() -> None:
    """Put the checkout's src/ on sys.path, lazily so a missing one is catchable."""
    repo_root = _repo_root()
    if repo_root is None:
        raise CorpusUnavailable("The rag package is not reachable from this process")
    source_root = str(repo_root / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)


@lru_cache(maxsize=1)
def _retriever() -> Any:
    """Build the hybrid retriever once.

    Construction opens the Pinecone index and reads, chunks and scores the whole
    corpus for BM25, so it is far too expensive to repeat per call.
    """
    _import_rag()
    from rag.embeddings import OllamaEmbedder
    from rag.retrieval.hybrid import HybridRetriever
    from rag.retrieval.keyword import KeywordRetriever
    from rag.retrieval.semantic import SemanticRetriever
    from rag.vector_store import PineconeSettings, ensure_index

    return HybridRetriever(
        semantic_retriever=SemanticRetriever(
            embedder=OllamaEmbedder(),
            vector_index=ensure_index(PineconeSettings.from_environment()),
        ),
        keyword_retriever=KeywordRetriever.from_corpus(_corpus_root(), chunk_size=CHUNK_SIZE),
        candidate_k=CANDIDATE_K,
    )


def _resolve_in_corpus(source_path: str) -> Path:
    """Resolve a source_path, refusing anything outside the corpus.

    The model chooses this argument, so it is untrusted. Resolving before
    checking containment is what defeats ../ traversal and escaping symlinks.
    """
    if not source_path.strip():
        raise ValueError("source_path must not be empty")
    corpus_root = _corpus_root()
    candidate = (corpus_root.parent / source_path).resolve()
    if not candidate.is_relative_to(corpus_root):
        raise ValueError(f"source_path must stay inside {corpus_root.name}/: got {source_path!r}")
    return candidate


def _read_frontmatter(path: Path) -> dict[str, Any]:
    """Return one document's metadata, or an empty mapping if it cannot be read."""
    from rag.ingestion.markdown import load_markdown_document

    try:
        return load_markdown_document(path, corpus_root=_corpus_root()).metadata
    except (ValueError, OSError):
        return {}


def _format_passage(position: int, passage: Any) -> str:
    metadata = passage.metadata
    header = " | ".join(
        f"{key}: {metadata[key]}"
        for key in ("doc_id", "version", "status", "department")
        if metadata.get(key)
    )
    return (
        f"[{position}] chunk_id: {passage.chunk_id}\n"
        f"source_path: {metadata.get('source_path', 'unknown')}\n"
        + (f"{header}\n" if header else "")
        + f"score: {passage.score:.4f}\n"
        f"{passage.text}"
    )


def _format_version(metadata: dict[str, Any]) -> str:
    return (
        f"- version: {metadata.get('version', 'unversioned')}"
        f" | status: {metadata.get('status', 'unknown')}"
        f" | effective_date: {metadata.get('effective_date', 'unknown')}"
        f" | expiry_date: {metadata.get('expiry_date', 'none')}"
        f" | source_path: {metadata.get('source_path')}"
    )


@tool
def search_documents(query: str, top_k: int = 5) -> str:
    """Search the Northwind company documents for passages relevant to a query.

    Use this first for any question about company policies, products, tickets,
    contracts, meetings, onboarding, or staff. Call it more than once with
    different wording when a question has several parts or the first results
    look incomplete.

    Args:
        query: A natural-language search phrase. Specific wording retrieves
            better than a whole question, for example "travel expense meal
            allowance" rather than "how much can I claim for meals".
        top_k: How many passages to return, 1 to 20. Start at 5.

    Returns:
        Numbered passages, each with its chunk_id, source_path, document
        metadata and text. Cite a passage by its chunk_id.
    """
    if not query.strip():
        return "Error: query must not be empty."
    try:
        limit = max(1, min(int(top_k), MAX_TOP_K))
    except (TypeError, ValueError):
        return f"Error: top_k must be a whole number between 1 and {MAX_TOP_K}."

    # A tool returns its failures as text. Raising here would end the turn,
    # while a message lets the model retry or say what it could not do.
    try:
        passages = _retriever().retrieve(query, top_k=limit)
    except Exception as error:  # noqa: BLE001 - tool boundary
        logger.exception("search_documents failed for query %r", query)
        return f"Error: search is unavailable. {error}"

    if not passages:
        return f"No passages matched {query!r}. Try different wording or a broader phrase."
    return "\n\n".join(
        _format_passage(position, passage) for position, passage in enumerate(passages, start=1)
    )


@tool
def list_document_versions(doc_id: str) -> str:
    """List every version of one document family, newest effective date first.

    Use this when a question depends on which revision applies, for example
    "what does the current travel policy say" or "what was the rule in 2025".
    A doc_id such as POL-FIN-004 is shared by all revisions of one document and
    appears in the metadata of every search result.

    Args:
        doc_id: The document family identifier, for example "POL-FIN-004".

    Returns:
        One line per version with its version number, status, effective and
        expiry dates, and source_path.
    """
    wanted = doc_id.strip()
    if not wanted:
        return "Error: doc_id must not be empty."

    try:
        _import_rag()
        corpus_root = _corpus_root()
        versions = [
            metadata
            for path in sorted(corpus_root.rglob("*.md"))
            if (metadata := _read_frontmatter(path)).get("doc_id") == wanted
        ]
    except Exception as error:  # noqa: BLE001 - tool boundary
        logger.exception("list_document_versions failed for doc_id %r", wanted)
        return f"Error: the corpus could not be listed. {error}"

    if not versions:
        return (
            f"No document has doc_id {wanted!r}. Run search_documents first and read the "
            "doc_id from a result's metadata."
        )

    versions.sort(key=lambda metadata: str(metadata.get("effective_date", "")), reverse=True)
    return "\n".join(
        [f"{len(versions)} version(s) of {wanted}:", *(_format_version(m) for m in versions)]
    )


@tool
def read_document(source_path: str, max_chars: int = DEFAULT_READ_CHARS) -> str:
    """Read a whole source document when a retrieved passage looks incomplete.

    Search returns fixed-size chunks, so a table, a numbered list or a handbook
    section can be cut in half. Use this when a passage ends mid-sentence, when
    you need a figure that a table header implies but the chunk does not show,
    or when you need the surrounding context of a chunk. Do not use it to browse.

    Args:
        source_path: The document path exactly as a search result reported it,
            for example "data/policies/travel-expense-v2.1.md".
        max_chars: Maximum characters to return, up to 20000. Raise it only if
            the document was truncated and the part you need is missing.

    Returns:
        The document text, truncated with an explicit marker if it is longer
        than max_chars.
    """
    try:
        limit = max(1, min(int(max_chars), MAX_READ_CHARS))
    except (TypeError, ValueError):
        return f"Error: max_chars must be a whole number between 1 and {MAX_READ_CHARS}."

    try:
        path = _resolve_in_corpus(source_path)
        if not path.is_file():
            return f"Error: no document at {source_path!r}. Use a source_path from a search result."
        text = path.read_text(encoding="utf-8")
    except Exception as error:  # noqa: BLE001 - tool boundary
        logger.exception("read_document failed for source_path %r", source_path)
        return f"Error: {error}"

    if len(text) <= limit:
        return f"{source_path} ({len(text)} characters)\n\n{text}"
    return (
        f"{source_path} (truncated to {limit} of {len(text)} characters)\n\n"
        f"{text[:limit]}\n\n[TRUNCATED. Call read_document again with a larger max_chars "
        "if the part you need is missing.]"
    )


NORTHWIND_TOOLS = [search_documents, list_document_versions, read_document]
