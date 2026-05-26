from __future__ import annotations

import base64
import json
import time

import pytest

from tradingagents.llm_clients.codex_auth import (
    CodexTokenResult,
    extract_codex_account_id,
    openai_codex_headers,
    resolve_openai_codex_token,
)
from tradingagents.llm_clients.factory import create_llm_client
from tradingagents.llm_clients.model_catalog import get_model_options


def _jwt(account_id: str = "acct_test", exp: int | None = None) -> str:
    def enc(data: dict) -> str:
        raw = json.dumps(data, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    payload = {
        "https://api.openai.com/auth": {
            "chatgpt_account_id": account_id,
        }
    }
    if exp is not None:
        payload["exp"] = exp
    return f"{enc({'alg': 'none'})}.{enc(payload)}.signature"


def test_codex_token_resolves_from_env(monkeypatch):
    token = _jwt("acct_env")
    monkeypatch.setenv("CODEX_ACCESS_TOKEN", token)
    result = resolve_openai_codex_token()

    assert result.token == token
    assert result.source == "CODEX_ACCESS_TOKEN"
    assert result.account_id == "acct_env"


def test_codex_token_resolves_from_pi_auth(monkeypatch, tmp_path):
    import tradingagents.llm_clients.codex_auth as codex_auth

    monkeypatch.delenv("CODEX_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_CODEX_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "missing-codex-home"))
    token = _jwt("acct_pi")
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "openai-codex": {
                    "type": "oauth",
                    "access": token,
                    "refresh": "refresh-token",
                    "expires": 4_102_444_800_000,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(codex_auth, "PI_AUTH_PATH", auth_path)

    result = resolve_openai_codex_token()

    assert result.token == token
    assert result.source == str(auth_path)
    assert result.account_id == "acct_pi"


def test_codex_token_resolves_from_codex_cli_auth(monkeypatch, tmp_path):
    monkeypatch.delenv("CODEX_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_CODEX_ACCESS_TOKEN", raising=False)
    token = _jwt("acct_codex")
    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    auth_path = codex_home / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "OPENAI_API_KEY": None,
                "tokens": {
                    "access_token": token,
                    "refresh_token": "refresh-token",
                    "account_id": "acct_codex",
                },
                "last_refresh": "2026-05-26T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    result = resolve_openai_codex_token()

    assert result.token == token
    assert result.source == str(auth_path)
    assert result.account_id == "acct_codex"


def test_codex_cli_auth_refreshes_expired_access_token(monkeypatch, tmp_path):
    import tradingagents.llm_clients.codex_auth as codex_auth

    monkeypatch.delenv("CODEX_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_CODEX_ACCESS_TOKEN", raising=False)
    old_token = _jwt("acct_old", exp=int(time.time()) - 10)
    new_token = _jwt("acct_new", exp=int(time.time()) + 3600)
    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    auth_path = codex_home / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": old_token,
                    "refresh_token": "refresh-old",
                    "account_id": "acct_old",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "access_token": new_token,
                "refresh_token": "refresh-new",
                "expires_in": 3600,
            }

    def fake_post(url, **kwargs):
        assert url == codex_auth._TOKEN_URL
        assert kwargs["data"]["refresh_token"] == "refresh-old"
        return FakeResponse()

    import requests
    monkeypatch.setattr(requests, "post", fake_post)

    result = resolve_openai_codex_token()
    saved = json.loads(auth_path.read_text(encoding="utf-8"))

    assert result.token == new_token
    assert result.account_id == "acct_new"
    assert saved["tokens"]["access_token"] == new_token
    assert saved["tokens"]["refresh_token"] == "refresh-new"
    assert saved["last_refresh"]


def test_codex_token_missing_raises_actionable_error(monkeypatch, tmp_path):
    import tradingagents.llm_clients.codex_auth as codex_auth

    monkeypatch.delenv("CODEX_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_CODEX_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setattr(codex_auth, "PI_AUTH_PATH", tmp_path / "missing.json")

    with pytest.raises(ValueError, match="codex login"):
        resolve_openai_codex_token()


def test_openai_codex_factory_and_catalog():
    client = create_llm_client("openai-codex", "gpt-5.5")
    assert client.provider == "openai-codex"
    assert get_model_options("openai-codex", "quick")[0][1] == "gpt-5.3-codex-spark"
    assert get_model_options("openai-codex", "deep")[0][1] == "gpt-5.5"


def test_openai_codex_headers_match_chatgpt_backend_requirements():
    token = _jwt("acct_headers")
    headers = openai_codex_headers(
        CodexTokenResult(
            token=token,
            source="test",
            account_id=extract_codex_account_id(token),
        )
    )

    assert headers["Authorization"] == f"Bearer {token}"
    assert headers["chatgpt-account-id"] == "acct_headers"
    assert headers["originator"] == "tradingagents"
    assert headers["OpenAI-Beta"] == "responses=experimental"


def test_openai_codex_client_uses_chatgpt_backend(monkeypatch):
    import tradingagents.llm_clients.openai_client as mod

    token = _jwt("acct_client")
    monkeypatch.setenv("CODEX_ACCESS_TOKEN", token)

    def fake_chat(**kwargs):
        return kwargs

    monkeypatch.setattr(mod, "OpenAICodexChatOpenAI", fake_chat)

    llm = mod.OpenAIClient(
        "gpt-5.5",
        base_url="https://api.openai.com/v1",
        provider="openai-codex",
        default_headers={"OpenAI-Organization": "org-test"},
    ).get_llm()

    assert llm["base_url"] == "https://chatgpt.com/backend-api/codex"
    assert llm["use_responses_api"] is True
    assert llm["streaming"] is True
    assert llm["store"] is False
    assert llm["include"] == ["reasoning.encrypted_content"]
    assert llm["default_headers"]["Authorization"] == f"Bearer {token}"
    assert llm["default_headers"]["chatgpt-account-id"] == "acct_client"
    assert llm["default_headers"]["OpenAI-Organization"] == "org-test"


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
