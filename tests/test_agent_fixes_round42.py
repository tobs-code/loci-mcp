"""LLM provider defaults: memory content must not leave the machine by accident.

NLI sends the compared facts to the provider, so the default is `none`. A bare
install stays off; configured provider keys keep working for existing
deployments when `LLM_PROVIDER` is omitted (compose passes `none` explicitly).
"""

from inference.llm import load_config


def test_bare_install_keeps_nli_off():
    config = load_config({})

    assert config.provider == "none"
    assert config.configured is False
    assert config.model == ""
    assert config.base_url == ""


def test_legacy_provider_key_still_enables_that_provider():
    config = load_config({"GROQ_API_KEY": "gsk-test"})

    assert config.provider == "groq"
    assert config.configured is True


def test_explicit_none_wins_over_a_configured_key():
    config = load_config({"LLM_PROVIDER": "none", "GROQ_API_KEY": "gsk-test"})

    assert config.provider == "none"
    assert config.configured is False


def test_explicit_provider_is_kept():
    config = load_config({"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"})

    assert config.provider == "openai"
    assert config.configured is True
    assert config.base_url == "https://api.openai.com/v1"
