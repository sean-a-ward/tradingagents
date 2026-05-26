from __future__ import annotations

import importlib
import json

import pytest

from tradingagents.llm_clients.codex_auth import resolve_codex_access_token
from tradingagents.llm_clients.factory import create_llm_client
from tradingagents.llm_clients.model_catalog import get_model_options


def test_codex_token_resolves_from_env(monkeypatch):
    monkeypatch.setenv("CODEX_ACCESS_TOKEN", "env-token")
    result = resolve_codex_access_token()
    assert result.token == "env-token"
    assert result.source == "CODEX_ACCESS_TOKEN"


def test_codex_token_resolves_from_auth_json(monkeypatch, tmp_path):
    monkeypatch.delenv("CODEX_ACCESS_TOKEN", raising=False)
    auth_dir = tmp_path / "codex"
    auth_dir.mkdir()
    (auth_dir / "auth.json").write_text(
        json.dumps({"tokens": {"access_token": "file-token"}}),
        encoding="utf-8",
    )

    result = resolve_codex_access_token(codex_home=auth_dir)
    assert result.token == "file-token"
    assert str(auth_dir / "auth.json") == result.source


def test_codex_token_missing_returns_actionable_error(monkeypatch, tmp_path):
    monkeypatch.delenv("CODEX_ACCESS_TOKEN", raising=False)
    result = resolve_codex_access_token(codex_home=tmp_path / "missing")
    assert result.token is None
    assert "CODEX_ACCESS_TOKEN" in result.error
    assert "auth.json" in result.error


def test_openai_codex_factory_and_catalog():
    client = create_llm_client("openai-codex", "gpt-5.5")
    assert client.provider == "openai-codex"
    assert get_model_options("openai-codex", "quick")[0][1] == "gpt-5.5"
    assert get_model_options("openai-codex", "deep")[0][1] == "gpt-5.5"


def test_openai_codex_client_sets_bearer_header_and_preflights(monkeypatch):
    import tradingagents.llm_clients.openai_client as mod

    mod = importlib.reload(mod)
    monkeypatch.setenv("CODEX_ACCESS_TOKEN", "codex-token-123")
    captured = {}

    class FakeResponse:
        text = "{}"

        def raise_for_status(self):
            return None

    def fake_post(url, **kwargs):
        captured["preflight_url"] = url
        captured["preflight_kwargs"] = kwargs
        return FakeResponse()

    def fake_chat(**kwargs):
        captured["chat_kwargs"] = kwargs
        return kwargs

    monkeypatch.setattr(mod.requests, "post", fake_post)
    monkeypatch.setattr(mod, "NormalizedChatOpenAI", fake_chat)

    llm = mod.OpenAIClient("gpt-5.5", provider="openai-codex").get_llm()

    assert llm["base_url"] == "https://api.openai.com/v1"
    assert llm["use_responses_api"] is True
    assert llm["default_headers"]["Authorization"] == "Bearer codex-token-123"
    assert captured["preflight_url"] == "https://api.openai.com/v1/responses"
    assert captured["preflight_kwargs"]["headers"]["Authorization"] == "Bearer codex-token-123"
    assert captured["preflight_kwargs"]["json"]["model"] == "gpt-5.5"


def test_openai_codex_merges_default_headers_without_losing_bearer(monkeypatch):
    import tradingagents.llm_clients.openai_client as mod

    mod = importlib.reload(mod)
    monkeypatch.setenv("CODEX_ACCESS_TOKEN", "codex-token-123")
    captured = {}

    class FakeResponse:
        text = "{}"

        def raise_for_status(self):
            return None

    def fake_post(url, **kwargs):
        captured["preflight_kwargs"] = kwargs
        return FakeResponse()

    def fake_chat(**kwargs):
        return kwargs

    monkeypatch.setattr(mod.requests, "post", fake_post)
    monkeypatch.setattr(mod, "NormalizedChatOpenAI", fake_chat)

    llm = mod.OpenAIClient(
        "gpt-5.5",
        provider="openai-codex",
        default_headers={
            "Authorization": "Bearer stale-or-wrong",
            "OpenAI-Organization": "org-test",
        },
    ).get_llm()

    assert llm["default_headers"]["Authorization"] == "Bearer codex-token-123"
    assert llm["default_headers"]["OpenAI-Organization"] == "org-test"
    assert (
        captured["preflight_kwargs"]["headers"]["Authorization"]
        == "Bearer codex-token-123"
    )
    assert captured["preflight_kwargs"]["headers"]["OpenAI-Organization"] == "org-test"


def test_openai_codex_preflight_cache_is_model_specific(monkeypatch):
    import tradingagents.llm_clients.openai_client as mod

    mod = importlib.reload(mod)
    monkeypatch.setenv("CODEX_ACCESS_TOKEN", "codex-token-123")
    calls = []

    class FakeResponse:
        text = "{}"

        def raise_for_status(self):
            return None

    def fake_post(url, **kwargs):
        calls.append(kwargs["json"]["model"])
        return FakeResponse()

    def fake_chat(**kwargs):
        return kwargs

    monkeypatch.setattr(mod.requests, "post", fake_post)
    monkeypatch.setattr(mod, "NormalizedChatOpenAI", fake_chat)

    mod.OpenAIClient("gpt-5.5", provider="openai-codex").get_llm()
    mod.OpenAIClient("gpt-5.4-mini", provider="openai-codex").get_llm()
    mod.OpenAIClient("gpt-5.5", provider="openai-codex").get_llm()

    assert calls == ["gpt-5.5", "gpt-5.4-mini"]


def test_openai_codex_preflight_failure_message(monkeypatch):
    import requests
    import tradingagents.llm_clients.openai_client as mod

    mod = importlib.reload(mod)
    monkeypatch.setenv("CODEX_ACCESS_TOKEN", "codex-token-123")

    def fake_post(*args, **kwargs):
        raise requests.Timeout("boom")

    monkeypatch.setattr(mod.requests, "post", fake_post)

    with pytest.raises(ValueError, match="openai-codex preflight failed"):
        mod.OpenAIClient("gpt-5.5", provider="openai-codex").get_llm()


def test_openai_codex_quick_deep_reasoning_effort_fallback():
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
    graph.config = {
        "llm_provider": "openai-codex",
        "openai_quick_reasoning_effort": "low",
        "openai_deep_reasoning_effort": "medium",
        "openai_reasoning_effort": "high",
    }
    assert graph._get_provider_kwargs("quick")["reasoning_effort"] == "low"
    assert graph._get_provider_kwargs("deep")["reasoning_effort"] == "medium"

    graph.config = {
        "llm_provider": "openai-codex",
        "openai_reasoning_effort": "high",
    }
    assert graph._get_provider_kwargs("quick")["reasoning_effort"] == "high"
    assert graph._get_provider_kwargs("deep")["reasoning_effort"] == "high"
