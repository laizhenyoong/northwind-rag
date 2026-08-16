"""Hosted chat client for DeepSeek's OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from rag.generation.ollama import OllamaCompletion, OllamaUsage


@dataclass(frozen=True, slots=True)
class DeepSeekChatModel:
    """Ask DeepSeek for one complete response, mirroring the Ollama client."""

    api_key: str
    model: str = "deepseek-v4-flash"
    host: str = "https://api.deepseek.com"
    timeout_seconds: float = 300.0

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None, **overrides: Any
    ) -> DeepSeekChatModel:
        """Build a client from the local, Git-ignored .env without printing the key."""
        load_dotenv()
        values = os.environ if environment is None else environment
        api_key = values.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("Missing required DeepSeek setting(s): DEEPSEEK_API_KEY")

        model = overrides.pop("model", None) or values.get("DEEPSEEK_MODEL") or "deepseek-v4-flash"
        return cls(api_key=api_key, model=model, **overrides)

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        """Return the model's non-empty response text."""
        return self.complete_with_usage(
            system_prompt=system_prompt, user_prompt=user_prompt
        ).text

    def complete_with_usage(
        self, *, system_prompt: str, user_prompt: str
    ) -> OllamaCompletion:
        """Return response text and the token counters DeepSeek reports."""
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "temperature": 0,
            }
        ).encode("utf-8")
        request = Request(
            f"{self.host}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body: dict[str, Any] = json.load(response)
        except HTTPError as error:
            raise RuntimeError(f"DeepSeek rejected the request with HTTP {error.code}") from error
        except OSError as error:
            raise RuntimeError(f"Could not reach DeepSeek at {self.host}: {error}") from error

        choices = body.get("choices")
        message = choices[0].get("message") if isinstance(choices, list) and choices else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("DeepSeek returned an unexpected chat response")

        usage = body.get("usage") or {}
        return OllamaCompletion(
            text=content.strip(),
            usage=OllamaUsage(
                prompt_tokens=_optional_int(usage.get("prompt_tokens")),
                output_tokens=_optional_int(usage.get("completion_tokens")),
            ),
        )


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None
