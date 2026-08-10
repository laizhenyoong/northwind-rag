"""Score question-passage pairs with the local BGE cross-encoder reranker."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from rag.retrieval.semantic import RetrievedPassage


class CrossEncoder(Protocol):
    """The scoring operation used by a cross-encoder reranker."""

    def predict(
        self,
        sentences: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> Sequence[float]: ...


@dataclass(slots=True)
class BGEReranker:
    """Rerank already-retrieved chunks using BAAI's local cross-encoder."""

    model_name: str = "BAAI/bge-reranker-v2-m3"
    batch_size: int = 8
    device: str | None = None
    _model: CrossEncoder | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")

    def rerank(
        self,
        question: str,
        passages: Sequence[RetrievedPassage],
        *,
        top_k: int,
    ) -> list[RetrievedPassage]:
        """Return the passages whose joint question-passage score is highest."""
        if not question.strip():
            raise ValueError("question must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not passages:
            return []

        model = self._model or _load_cross_encoder(self.model_name, self.device)
        self._model = model
        scores = model.predict(
            [(question, passage.text) for passage in passages],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        if len(scores) != len(passages):
            raise RuntimeError("Reranker did not return one score for every passage")

        ranked = sorted(
            zip(scores, passages, strict=True),
            key=lambda item: (-float(item[0]), item[1].chunk_id),
        )[:top_k]
        return [
            RetrievedPassage(
                chunk_id=passage.chunk_id,
                text=passage.text,
                score=float(score),
                metadata=dict(passage.metadata),
            )
            for score, passage in ranked
        ]


def _load_cross_encoder(model_name: str, device: str | None) -> CrossEncoder:
    """Load the model lazily so baseline retrieval does not pay this cost."""
    try:
        import torch
        from sentence_transformers import CrossEncoder as SentenceTransformersCrossEncoder
    except ImportError as error:
        raise RuntimeError(
            "The local reranker needs sentence-transformers. Install project dependencies "
            "before using BGEReranker."
        ) from error

    selected_device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    return SentenceTransformersCrossEncoder(model_name, device=selected_device)
