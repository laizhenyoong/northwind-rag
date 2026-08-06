"""Create the Pinecone index used by this RAG project."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec


# Ollama's embeddinggemma model returns vectors with this exact length.
EMBEDDING_DIMENSIONS = 768


@dataclass(frozen=True, slots=True)
class PineconeSettings:
    """Connection details loaded from the local, Git-ignored .env file."""

    api_key: str
    index_name: str
    cloud: str
    region: str

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None
    ) -> PineconeSettings:
        """Load Pinecone settings without ever printing the API key."""
        load_dotenv()
        values = os.environ if environment is None else environment
        required = (
            "PINECONE_API_KEY",
            "PINECONE_INDEX_NAME",
            "PINECONE_CLOUD",
            "PINECONE_REGION",
        )
        missing = [name for name in required if not values.get(name)]
        if missing:
            names = ", ".join(missing)
            raise RuntimeError(f"Missing required Pinecone setting(s): {names}")

        return cls(
            api_key=values["PINECONE_API_KEY"],
            index_name=values["PINECONE_INDEX_NAME"],
            cloud=values["PINECONE_CLOUD"],
            region=values["PINECONE_REGION"],
        )


def ensure_index(
    settings: PineconeSettings, *, client: Any | None = None
) -> Any:
    """Return the project index, creating its empty vector cabinet when needed."""
    pinecone = client or Pinecone(api_key=settings.api_key)

    if not pinecone.has_index(settings.index_name):
        pinecone.create_index(
            name=settings.index_name,
            dimension=EMBEDDING_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.cloud, region=settings.region),
            deletion_protection="disabled",
            tags={"project": "northwind-rag"},
        )

    return pinecone.Index(settings.index_name)
