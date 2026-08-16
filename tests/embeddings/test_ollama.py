import json
from urllib.error import HTTPError, URLError

import pytest

from rag.embeddings import OllamaEmbedder


class FakeResponse:
    def __init__(self, body: dict) -> None:
        self.body = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self) -> bytes:
        return self.body


def test_embed_posts_all_texts_to_ollama(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse({"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

    monkeypatch.setattr("rag.embeddings.ollama.urlopen", fake_urlopen)

    embeddings = OllamaEmbedder().embed(["first", "second"])

    assert captured == {
        "url": "http://127.0.0.1:11434/api/embed",
        "body": {"model": "embeddinggemma", "input": ["first", "second"]},
        "timeout": 60.0,
    }
    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]


def test_embed_does_not_call_ollama_for_an_empty_batch() -> None:
    assert OllamaEmbedder().embed([]) == []


def test_a_long_input_list_is_split_into_batches_ollama_accepts(monkeypatch) -> None:
    """Ollama 400s on an over-long input list, so one document must not be one request."""
    batches = []

    def fake_urlopen(request, timeout):
        texts = json.loads(request.data)["input"]
        batches.append(len(texts))
        return FakeResponse({"embeddings": [[0.1] for _ in texts]})

    monkeypatch.setattr("rag.embeddings.ollama.urlopen", fake_urlopen)

    embeddings = OllamaEmbedder(batch_size=64).embed([f"text {index}" for index in range(150)])

    assert batches == [64, 64, 22]
    assert len(embeddings) == 150


def test_batched_embeddings_keep_the_order_of_their_inputs(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        texts = json.loads(request.data)["input"]
        return FakeResponse({"embeddings": [[float(text.split()[1])] for text in texts]})

    monkeypatch.setattr("rag.embeddings.ollama.urlopen", fake_urlopen)

    embeddings = OllamaEmbedder(batch_size=2).embed([f"text {index}" for index in range(5)])

    assert embeddings == [[0.0], [1.0], [2.0], [3.0], [4.0]]


def test_a_rejected_request_reports_the_status_instead_of_blaming_a_dead_server(
    monkeypatch,
) -> None:
    """A 400 means Ollama answered, so 'start it' would send the reader the wrong way."""

    def fake_urlopen(request, timeout):
        raise HTTPError(url="http://127.0.0.1:11434/api/embed", code=400, msg="Bad Request",
                        hdrs=None, fp=None)

    monkeypatch.setattr("rag.embeddings.ollama.urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="HTTP 400"):
        OllamaEmbedder().embed(["text"])


def test_an_unreachable_server_still_says_so(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise URLError("connection refused")

    monkeypatch.setattr("rag.embeddings.ollama.urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="Could not reach Ollama"):
        OllamaEmbedder().embed(["text"])
