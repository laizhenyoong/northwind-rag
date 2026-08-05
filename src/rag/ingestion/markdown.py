"""Load Markdown source documents and preserve their YAML metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


FRONTMATTER_DELIMITER = "---\n"


@dataclass(frozen=True, slots=True)
class Document:
    """A source document before it has been divided into chunks."""

    content: str
    metadata: dict[str, Any]


def load_markdown_document(path: Path, *, corpus_root: Path) -> Document:
    """Load one Markdown file with YAML frontmatter.

    The content is kept separate from metadata. Retrieval will embed content,
    while filters use metadata such as status, access, and effective dates.
    """
    frontmatter, content = _split_frontmatter(path)
    metadata = yaml.safe_load(frontmatter)

    if not isinstance(metadata, dict):
        raise ValueError(f"{path} frontmatter must be a mapping")

    normalized_metadata = dict(metadata)
    source_path = path.relative_to(corpus_root).as_posix()
    normalized_metadata["source_path"] = source_path
    # `doc_id` can describe a family of versions, and some documents have no
    # version. A path uniquely identifies every file in this fixed corpus.
    normalized_metadata["document_key"] = source_path

    return Document(content=content.strip(), metadata=normalized_metadata)


def load_corpus(corpus_root: Path) -> list[Document]:
    """Load every Markdown file below ``corpus_root`` in stable path order."""
    return [
        load_markdown_document(path, corpus_root=corpus_root)
        for path in sorted(corpus_root.rglob("*.md"))
    ]


def _split_frontmatter(path: Path) -> tuple[str, str]:
    """Return a document's YAML frontmatter and Markdown body."""
    raw_text = path.read_text(encoding="utf-8")

    if not raw_text.startswith(FRONTMATTER_DELIMITER):
        raise ValueError(f"{path} does not start with YAML frontmatter")

    try:
        _, frontmatter, content = raw_text.split(FRONTMATTER_DELIMITER, maxsplit=2)
    except ValueError as error:
        raise ValueError(f"{path} has an unclosed YAML frontmatter block") from error

    return frontmatter, content
