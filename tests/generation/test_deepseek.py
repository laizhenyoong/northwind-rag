import json

import pytest

from rag.generation import DeepSeekChatModel


class FakeResponse:
    def __init__(self, body: dict) -> None:
        self.body = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self) -> bytes:
        return self.body


def verdict_body(content: str = "Supported answer [S1]") -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 3400, "completion_tokens": 150},
    }


def test_chat_posts_an_authenticated_deterministic_request(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        return FakeResponse(verdict_body())

    monkeypatch.setattr("rag.generation.deepseek.urlopen", fake_urlopen)

    answer = DeepSeekChatModel(api_key="secret-key").complete(
        system_prompt="system", user_prompt="question"
    )

    assert answer == "Supported answer [S1]"
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["authorization"] == "Bearer secret-key"
    assert captured["body"] == {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
        ],
        "stream": False,
        "temperature": 0,
    }


def test_chat_reports_the_token_counters_deepseek_returns(monkeypatch) -> None:
    monkeypatch.setattr(
        "rag.generation.deepseek.urlopen",
        lambda request, timeout: FakeResponse(verdict_body()),
    )

    completion = DeepSeekChatModel(api_key="k").complete_with_usage(
        system_prompt="system", user_prompt="question"
    )

    assert completion.usage.prompt_tokens == 3400
    assert completion.usage.output_tokens == 150


def test_an_empty_answer_is_refused_rather_than_returned(monkeypatch) -> None:
    monkeypatch.setattr(
        "rag.generation.deepseek.urlopen",
        lambda request, timeout: FakeResponse(verdict_body("   ")),
    )

    with pytest.raises(RuntimeError, match="unexpected chat response"):
        DeepSeekChatModel(api_key="k").complete(system_prompt="s", user_prompt="q")


def test_settings_load_from_the_environment() -> None:
    model = DeepSeekChatModel.from_environment(
        {"DEEPSEEK_API_KEY": "secret-key", "DEEPSEEK_MODEL": "deepseek-v4-pro"}
    )

    assert model.api_key == "secret-key"
    assert model.model == "deepseek-v4-pro"


def test_an_explicit_model_beats_the_environment() -> None:
    model = DeepSeekChatModel.from_environment(
        {"DEEPSEEK_API_KEY": "k", "DEEPSEEK_MODEL": "deepseek-v4-pro"},
        model="deepseek-v4-flash",
    )

    assert model.model == "deepseek-v4-flash"


def test_a_missing_key_fails_before_any_request_is_sent() -> None:
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        DeepSeekChatModel.from_environment({"DEEPSEEK_MODEL": "deepseek-v4-flash"})
