import json

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
