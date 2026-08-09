"""Local chat client for Ollama's non-streaming /api/chat endpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True, slots=True)
class OllamaChatModel:
    """Ask a locally running Ollama chat model for one complete response."""

    model: str = "gemma4"
    host: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 120.0

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        """Return the model's non-empty response text."""
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"temperature": 0},
            }
        ).encode("utf-8")
        request = Request(
            f"{self.host}/api/chat",
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

        message = body.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Ollama returned an unexpected chat response")
        return content.strip()
