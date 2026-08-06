import pytest

from rag.vector_store import EMBEDDING_DIMENSIONS, PineconeSettings, ensure_index


class FakePinecone:
    def __init__(self, *, exists: bool) -> None:
        self.exists = exists
        self.create_calls = []
        self.opened_name = None

    def has_index(self, name: str) -> bool:
        self.checked_name = name
        return self.exists

    def create_index(self, **kwargs) -> None:
        self.create_calls.append(kwargs)

    def Index(self, name: str) -> str:
        self.opened_name = name
        return f"index:{name}"


def test_settings_require_every_environment_variable() -> None:
    with pytest.raises(RuntimeError, match="PINECONE_API_KEY"):
        PineconeSettings.from_environment({})


def test_ensure_index_creates_a_768_dimension_cosine_index() -> None:
    client = FakePinecone(exists=False)
    settings = PineconeSettings("secret", "northwind-rag", "aws", "us-east-1")

    index = ensure_index(settings, client=client)

    assert index == "index:northwind-rag"
    assert client.checked_name == "northwind-rag"
    assert len(client.create_calls) == 1
    create_call = client.create_calls[0]
    assert create_call["name"] == "northwind-rag"
    assert create_call["dimension"] == EMBEDDING_DIMENSIONS
    assert create_call["metric"] == "cosine"
    assert create_call["deletion_protection"] == "disabled"
    assert create_call["tags"] == {"project": "northwind-rag"}
    assert create_call["spec"].cloud == "aws"
    assert create_call["spec"].region == "us-east-1"


def test_ensure_index_reuses_an_existing_index() -> None:
    client = FakePinecone(exists=True)
    settings = PineconeSettings("secret", "northwind-rag", "aws", "us-east-1")

    ensure_index(settings, client=client)

    assert client.create_calls == []
