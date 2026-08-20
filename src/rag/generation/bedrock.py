"""Hosted chat client for Amazon Bedrock's Converse API."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from rag.generation.ollama import OllamaCompletion, OllamaUsage


DEFAULT_MODEL = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"


@dataclass(slots=True)
class BedrockChatModel:
    """Ask Bedrock for one complete response, mirroring the other chat clients.

    Converse is used rather than InvokeModel so the request shape is the same
    for every model family, and so the token counters come back in one place.
    """

    model: str = DEFAULT_MODEL
    region: str | None = None
    max_tokens: int = 2048
    temperature: float = 0.0
    timeout_seconds: float = 300.0
    _client: Any = field(default=None, init=False, repr=False)

    def _runtime(self) -> Any:
        """Create the boto3 client lazily and reuse it across calls."""
        if self._client is None:
            import boto3
            from botocore.config import Config

            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self.region or os.environ.get("AWS_REGION") or "us-west-2",
                config=Config(
                    read_timeout=self.timeout_seconds,
                    retries={"max_attempts": 5, "mode": "adaptive"},
                ),
            )
        return self._client

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        """Return the model's non-empty response text."""
        return self.complete_with_usage(
            system_prompt=system_prompt, user_prompt=user_prompt
        ).text

    def complete_with_usage(
        self, *, system_prompt: str, user_prompt: str
    ) -> OllamaCompletion:
        """Return response text and the token counters Bedrock reports."""
        response = self._runtime().converse(
            modelId=self.model,
            system=[{"text": system_prompt}],
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            inferenceConfig={"maxTokens": self.max_tokens, "temperature": self.temperature},
        )

        blocks = response.get("output", {}).get("message", {}).get("content", [])
        text = "".join(block.get("text", "") for block in blocks).strip()
        if not text:
            raise RuntimeError(f"Bedrock returned an empty response for model {self.model!r}")

        usage = response.get("usage", {})
        return OllamaCompletion(
            text=text,
            usage=OllamaUsage(
                prompt_tokens=usage.get("inputTokens"),
                output_tokens=usage.get("outputTokens"),
                total_duration_ms=response.get("metrics", {}).get("latencyMs"),
            ),
        )
