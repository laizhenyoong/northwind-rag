import json

from rag.generation.ollama import OllamaChatModel, OllamaUsage


class FakeResponse:
    def __init__(self, body: dict) -> None:
        self.body = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self) -> bytes:
        return self.body


def test_chat_posts_a_non_streaming_deterministic_request(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse({"message": {"content": "Supported answer [S1]"}})

    monkeypatch.setattr("rag.generation.ollama.urlopen", fake_urlopen)

    answer = OllamaChatModel().complete(system_prompt="system", user_prompt="question")

    assert answer == "Supported answer [S1]"
    assert captured == {
        "url": "http://127.0.0.1:11434/api/chat",
        "body": {
            "model": "gemma4",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
            ],
            "stream": False,
            "options": {"temperature": 0},
        },
        "timeout": 120.0,
    }


def test_chat_exposes_ollamas_exact_usage_counters(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        return FakeResponse(
            {
                "message": {"content": "Supported answer [S1]"},
                "prompt_eval_count": 120,
                "eval_count": 24,
                "load_duration": 5_100_000,
                "prompt_eval_duration": 10_200_000,
                "eval_duration": 20_300_000,
                "total_duration": 40_400_000,
            }
        )

    monkeypatch.setattr("rag.generation.ollama.urlopen", fake_urlopen)

    completion = OllamaChatModel().complete_with_usage(
        system_prompt="system", user_prompt="question"
    )

    assert completion.text == "Supported answer [S1]"
    assert completion.usage == OllamaUsage(
        prompt_tokens=120,
        output_tokens=24,
        load_duration_ms=5,
        prompt_eval_duration_ms=10,
        eval_duration_ms=20,
        total_duration_ms=40,
    )
