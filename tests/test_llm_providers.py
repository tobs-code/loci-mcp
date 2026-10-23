import pytest

from inference.llm import (
    build_chat_request,
    classify,
    load_config,
    nli_score,
    parse_nli_json,
)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def _chat_payload(score=0.9, label="contradiction"):
    return {
        "choices": [{
            "message": {
                "content": f'{{"label": "{label}", "score": {score}}}',
            },
        }],
    }


def test_defaults_to_groq_and_is_unconfigured_without_a_key():
    config = load_config({})

    assert config.provider == "groq"
    assert config.model == "openai/gpt-oss-20b"
    assert config.base_url == "https://api.groq.com/openai/v1"
    assert config.style == "openai"
    assert config.configured is False


def test_groq_keeps_the_legacy_env_names():
    config = load_config({
        "GROQ_API_KEY": "gsk-test",
        "GROQ_MODEL": "custom-groq-model",
    })

    assert config.api_key == "gsk-test"
    assert config.model == "custom-groq-model"
    assert config.configured is True


def test_openai_uses_its_own_defaults_and_key():
    config = load_config({"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"})

    assert config.provider == "openai"
    assert config.model == "gpt-4o-mini"
    assert config.base_url == "https://api.openai.com/v1"
    assert config.configured is True


def test_generic_overrides_win():
    config = load_config({
        "LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "provider-key",
        "LLM_API_KEY": "generic-key",
        "LLM_MODEL": "gpt-4.1",
        "LLM_BASE_URL": "https://proxy.example/v1",
    })

    assert config.api_key == "generic-key"
    assert config.model == "gpt-4.1"
    assert config.base_url == "https://proxy.example/v1"


def test_anthropic_uses_the_messages_style():
    config = load_config({
        "LLM_PROVIDER": "anthropic",
        "ANTHROPIC_API_KEY": "ak-test",
    })

    assert config.style == "anthropic"
    assert config.model == "claude-3-5-haiku-latest"
    assert config.configured is True


def test_ollama_needs_no_key_and_normalizes_the_v1_suffix():
    config = load_config({
        "LLM_PROVIDER": "ollama",
        "OLLAMA_BASE_URL": "http://localhost:11434",
    })

    assert config.base_url == "http://localhost:11434/v1"
    assert config.model == "llama3.2"
    assert config.configured is True


def test_openai_compatible_requires_a_base_url():
    missing = load_config({"LLM_PROVIDER": "openai-compatible"})
    configured = load_config({
        "LLM_PROVIDER": "openai-compatible",
        "LLM_BASE_URL": "http://localhost:1234/v1",
        "LLM_MODEL": "local-model",
    })

    assert missing.configured is False
    assert configured.configured is True
    assert configured.api_key == ""


def test_auto_detects_the_first_configured_provider():
    assert load_config({
        "LLM_PROVIDER": "auto",
        "GROQ_API_KEY": "g",
        "OPENAI_API_KEY": "o",
        "ANTHROPIC_API_KEY": "a",
    }).provider == "groq"
    assert load_config({
        "LLM_PROVIDER": "auto",
        "OPENAI_API_KEY": "o",
        "ANTHROPIC_API_KEY": "a",
    }).provider == "openai"
    assert load_config({
        "LLM_PROVIDER": "auto",
        "ANTHROPIC_API_KEY": "a",
    }).provider == "anthropic"
    assert load_config({
        "LLM_PROVIDER": "auto",
        "LLM_BASE_URL": "http://localhost:1234/v1",
    }).provider == "openai-compatible"


def test_none_disables_nli():
    config = load_config({"LLM_PROVIDER": "none"})

    assert config.provider == "none"
    assert config.configured is False


def test_classify_returns_zero_without_a_request_when_disabled(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("disabled provider must not send a request")

    monkeypatch.setattr("inference.llm.requests.post", fail)
    config = load_config({"LLM_PROVIDER": "none"})

    assert classify(config, "A", "B") == 0.0


def test_build_request_for_openai_style():
    config = load_config({"LLM_PROVIDER": "openai", "LLM_API_KEY": "sk-test"})

    url, headers, body = build_chat_request(
        config, "system", "user", {"type": "json_object"},
    )

    assert url == "https://api.openai.com/v1/chat/completions"
    assert headers["Authorization"] == "Bearer sk-test"
    assert body["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    assert body["response_format"] == {"type": "json_object"}


def test_build_request_for_anthropic_style():
    config = load_config({
        "LLM_PROVIDER": "anthropic",
        "ANTHROPIC_API_KEY": "ak-test",
    })

    url, headers, body = build_chat_request(config, "system", "user")

    assert url == "https://api.anthropic.com/v1/messages"
    assert headers["x-api-key"] == "ak-test"
    assert headers["anthropic-version"] == "2023-06-01"
    assert body["system"] == "system"
    assert body["messages"] == [{"role": "user", "content": "user"}]
    assert "response_format" not in body


def test_ollama_requests_have_no_authorization_header():
    config = load_config({"LLM_PROVIDER": "ollama"})

    _, headers, _ = build_chat_request(config, "system", "user")

    assert "Authorization" not in headers


def test_parse_nli_json_handles_fences_and_surrounding_text():
    fenced = '```json\n{"label": "neutral", "score": 0.1}\n```'
    embedded = 'Here is the result: {"label": "contradiction", "score": 0.8} done'

    assert parse_nli_json(fenced) == {"label": "neutral", "score": 0.1}
    assert parse_nli_json(embedded) == {"label": "contradiction", "score": 0.8}
    assert parse_nli_json("not json") is None
    assert parse_nli_json("") is None


def test_classify_returns_the_contradiction_score(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append((url, headers, json))
        return FakeResponse(payload=_chat_payload(score=0.91))

    monkeypatch.setattr("inference.llm.requests.post", fake_post)
    config = load_config({"LLM_PROVIDER": "openai", "LLM_API_KEY": "sk-test"})

    assert classify(config, "Alice likes tea", "Alice does not like tea") == 0.91
    assert len(calls) == 1
    assert calls[0][2]["response_format"]["type"] == "json_schema"


def test_classify_falls_back_to_json_object(monkeypatch):
    formats = []

    def fake_post(url, headers=None, json=None, timeout=None):
        formats.append(json.get("response_format"))
        if len(formats) == 1:
            return FakeResponse(status_code=400, text="unsupported format")
        return FakeResponse(payload=_chat_payload(score=0.7))

    monkeypatch.setattr("inference.llm.requests.post", fake_post)
    config = load_config({"LLM_PROVIDER": "openai", "LLM_API_KEY": "sk-test"})

    assert classify(config, "A", "B") == 0.7
    assert formats[0]["type"] == "json_schema"
    assert formats[1] == {"type": "json_object"}


def test_classify_retries_groq_json_validate_failed(monkeypatch):
    # Groq's strict structured output intermittently rejects the model's own
    # empty generation as 400 json_validate_failed. That is transient, so the
    # SAME format is retried instead of falling back immediately.
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json.get("response_format"))
        if len(calls) == 1:
            return FakeResponse(
                status_code=400,
                text='{"error":{"code":"json_validate_failed",'
                     '"failed_generation":""}}')
        return FakeResponse(payload=_chat_payload(score=0.88))

    monkeypatch.setattr("inference.llm.requests.post", fake_post)
    monkeypatch.setattr("inference.llm.time.sleep", lambda s: None)
    config = load_config({"LLM_PROVIDER": "openai", "LLM_API_KEY": "sk-test"})

    assert classify(config, "A", "B") == 0.88
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert calls[0]["type"] == "json_schema"


def test_classify_supports_anthropic_responses(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        return FakeResponse(payload={
            "content": [{"type": "text",
                         "text": '{"label": "contradiction", "score": 0.8}'}],
        })

    monkeypatch.setattr("inference.llm.requests.post", fake_post)
    config = load_config({
        "LLM_PROVIDER": "anthropic",
        "ANTHROPIC_API_KEY": "ak-test",
    })

    assert classify(config, "A", "B") == 0.8


def test_classify_aborts_when_every_format_fails(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        return FakeResponse(status_code=500, text="provider down")

    monkeypatch.setattr("inference.llm.requests.post", fake_post)
    monkeypatch.setattr("inference.llm.time.sleep", lambda s: None)
    config = load_config({"LLM_PROVIDER": "openai", "LLM_API_KEY": "sk-test"})

    with pytest.raises(RuntimeError, match="LLM request failed"):
        classify(config, "A", "B")


def test_nli_score_normalizes_percentages_instead_of_clamping():
    assert nli_score({"label": "contradiction", "score": 95}) == pytest.approx(0.95)
    # An ambiguous 1.4 is understated, not promoted to a perfect 1.0.
    assert nli_score({"label": "contradiction", "score": 1.4}) == pytest.approx(0.014)
    assert nli_score({"label": "neutral", "score": 0.99}) == 0.0
    assert nli_score({"label": "contradiction", "score": 500}) == 0.0
