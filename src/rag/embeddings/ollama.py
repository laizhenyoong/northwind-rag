"""Local embedding client for Ollama's /api/embed endpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True, slots=True)
class OllamaEmbedder:
    """Embed text locally using an Ollama embedding model."""

    model: str = "embeddinggemma"
    host: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 60.0

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector for every input text."""
        if not texts:
            return []

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
        except URLError as error:
            raise RuntimeError(
                "Could not reach Ollama. Start it, then confirm it is available at "
                f"{self.host}."
            ) from error

        embeddings = body.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError("Ollama returned an unexpected embedding response")

        return embeddings
