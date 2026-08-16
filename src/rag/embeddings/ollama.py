"""Local embedding client for Ollama's /api/embed endpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True, slots=True)
class OllamaEmbedder:
    """Embed text locally using an Ollama embedding model."""

    model: str = "embeddinggemma"
    host: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 60.0
    batch_size: int = 64

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector for every input text."""
        if not texts:
            return []
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            embeddings.extend(self._embed_batch(texts[start : start + self.batch_size]))
        return embeddings

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed one batch small enough for Ollama to accept."""
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        request = Request(
            f"{self.host}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body: dict[str, Any] = json.load(response)
        except HTTPError as error:
            raise RuntimeError(
                f"Ollama rejected the embedding request with HTTP {error.code}. "
                f"Model {self.model!r}, {len(texts)} input(s)."
            ) from error
        except URLError as error:
            raise RuntimeError(
                "Could not reach Ollama. Start it, then confirm it is available at "
                f"{self.host}."
            ) from error

        embeddings = body.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError("Ollama returned an unexpected embedding response")

        return embeddings
