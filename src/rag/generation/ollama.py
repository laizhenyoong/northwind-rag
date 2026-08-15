"""Local chat client for Ollama's non-streaming /api/chat endpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True, slots=True)
class OllamaUsage:
    """Exact token and timing counters returned by Ollama for one chat call."""

    prompt_tokens: int | None = None
    output_tokens: int | None = None
    load_duration_ms: int | None = None
    prompt_eval_duration_ms: int | None = None
    eval_duration_ms: int | None = None
    total_duration_ms: int | None = None


@dataclass(frozen=True, slots=True)
class OllamaCompletion:
    """One chat response together with Ollama's server-side measurements."""

    text: str
    usage: OllamaUsage


@dataclass(frozen=True, slots=True)
class OllamaChatModel:
    """Ask a locally running Ollama chat model for one complete response."""

    model: str = "gemma4"
    host: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 120.0

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        """Return the model's non-empty response text."""
        return self.complete_with_usage(
            system_prompt=system_prompt, user_prompt=user_prompt
        ).text

    def complete_with_usage(
        self, *, system_prompt: str, user_prompt: str
    ) -> OllamaCompletion:
        """Return response text and exact usage counters supplied by Ollama."""
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
        return OllamaCompletion(
            text=content.strip(),
            usage=OllamaUsage(
                prompt_tokens=_optional_int(body.get("prompt_eval_count")),
                output_tokens=_optional_int(body.get("eval_count")),
                load_duration_ms=_nanoseconds_to_milliseconds(body.get("load_duration")),
                prompt_eval_duration_ms=_nanoseconds_to_milliseconds(
                    body.get("prompt_eval_duration")
                ),
                eval_duration_ms=_nanoseconds_to_milliseconds(body.get("eval_duration")),
                total_duration_ms=_nanoseconds_to_milliseconds(body.get("total_duration")),
            ),
        )


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _nanoseconds_to_milliseconds(value: object) -> int | None:
    return round(value / 1_000_000) if isinstance(value, int) else None
