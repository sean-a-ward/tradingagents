from __future__ import annotations

import base64
import json

import pytest

from tradingagents.llm_clients.codex_auth import (
    extract_codex_account_id,
    openai_codex_headers,
    resolve_openai_codex_token,
)
from tradingagents.llm_clients.openai_client import OpenAIClient
from tradingagents.llm_clients.openai_client import OpenAICodexChatOpenAI


def _jwt(account_id: str = "acct_test") -> str:
    def enc(data: dict) -> str:
        raw = json.dumps(data, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    payload = {
        "https://api.openai.com/auth": {
            "chatgpt_account_id": account_id,
        }
    }
    return f"{enc({'alg': 'none'})}.{enc(payload)}.signature"


def test_extract_codex_account_id_from_jwt():
    assert extract_codex_account_id(_jwt("acct_123")) == "acct_123"


def test_resolve_openai_codex_token_prefers_codex_access_token(monkeypatch):
    token = _jwt("acct_env")
    monkeypatch.setenv("CODEX_ACCESS_TOKEN", token)
    monkeypatch.delenv("OPENAI_CODEX_ACCESS_TOKEN", raising=False)

    result = resolve_openai_codex_token()

    assert result.token == token
    assert result.source == "CODEX_ACCESS_TOKEN"
    assert result.account_id == "acct_env"


def test_openai_codex_headers_match_chatgpt_backend_requirements(monkeypatch):
    token = _jwt("acct_headers")
    monkeypatch.setenv("CODEX_ACCESS_TOKEN", token)
    result = resolve_openai_codex_token()

    headers = openai_codex_headers(result)

    assert headers["Authorization"] == f"Bearer {token}"
    assert headers["chatgpt-account-id"] == "acct_headers"
    assert headers["originator"] == "tradingagents"
    assert headers["OpenAI-Beta"] == "responses=experimental"


def test_openai_codex_client_uses_chatgpt_codex_responses_endpoint(monkeypatch):
    token = _jwt("acct_client")
    monkeypatch.setenv("CODEX_ACCESS_TOKEN", token)

    llm = OpenAIClient(
        "gpt-5.3-codex-spark",
        provider="openai-codex",
        reasoning_effort="medium",
    ).get_llm()

    assert str(llm.openai_api_base).rstrip("/") == "https://chatgpt.com/backend-api/codex"
    assert llm.use_responses_api is True
    assert llm.streaming is True
    assert llm.store is False
    assert llm.include == ["reasoning.encrypted_content"]
    assert llm.default_headers["Authorization"] == f"Bearer {token}"
    assert llm.default_headers["chatgpt-account-id"] == "acct_client"


def test_openai_codex_client_does_not_use_public_openai_responses_url(monkeypatch):
    monkeypatch.setenv("CODEX_ACCESS_TOKEN", _jwt("acct_client"))

    llm = OpenAIClient(
        "gpt-5.5",
        base_url="https://api.openai.com/v1",
        provider="openai-codex",
    ).get_llm()

    assert "api.openai.com" not in str(llm.openai_api_base)
    assert str(llm.openai_api_base).rstrip("/") == "https://chatgpt.com/backend-api/codex"


def test_invalid_codex_token_raises_clear_error(monkeypatch):
    monkeypatch.setenv("CODEX_ACCESS_TOKEN", "not-a-jwt")

    with pytest.raises(ValueError, match="extract ChatGPT account id"):
        OpenAIClient("gpt-5.3-codex-spark", provider="openai-codex").get_llm()


def test_openai_codex_payload_lifts_system_messages_to_instructions():
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = OpenAICodexChatOpenAI(
        model="gpt-5.3-codex-spark",
        api_key="test-token",
        base_url="https://chatgpt.com/backend-api/codex",
        use_responses_api=True,
    )

    payload = llm._get_request_payload(
        [
            SystemMessage(content="system one"),
            SystemMessage(content="system two"),
            HumanMessage(content="hello"),
        ]
    )

    assert payload["instructions"] == "system one\n\nsystem two"
    assert all(item.get("role") != "system" for item in payload["input"])
    assert payload["input"] == [
        {"content": "hello", "role": "user", "type": "message"}
    ]


def test_openai_codex_payload_adds_default_instructions_without_system_message():
    from langchain_core.messages import HumanMessage

    llm = OpenAICodexChatOpenAI(
        model="gpt-5.3-codex-spark",
        api_key="test-token",
        base_url="https://chatgpt.com/backend-api/codex",
        use_responses_api=True,
    )

    payload = llm._get_request_payload([HumanMessage(content="hello")])

    assert payload["instructions"] == "You are a helpful assistant."
    assert payload["input"] == [
        {"content": "hello", "role": "user", "type": "message"}
    ]


def test_openai_codex_payload_preserves_existing_instructions():
    from langchain_core.messages import HumanMessage

    llm = OpenAICodexChatOpenAI(
        model="gpt-5.3-codex-spark",
        api_key="test-token",
        base_url="https://chatgpt.com/backend-api/codex",
        use_responses_api=True,
    )

    payload = llm._get_request_payload(
        [HumanMessage(content="hello")],
        instructions="caller instruction",
    )

    assert payload["instructions"] == "caller instruction"
    assert payload["input"] == [
        {"content": "hello", "role": "user", "type": "message"}
    ]


def test_openai_codex_payload_combines_existing_and_system_instructions():
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = OpenAICodexChatOpenAI(
        model="gpt-5.3-codex-spark",
        api_key="test-token",
        base_url="https://chatgpt.com/backend-api/codex",
        use_responses_api=True,
    )

    payload = llm._get_request_payload(
        [
            SystemMessage(content="system instruction"),
            HumanMessage(content="hello"),
        ],
        instructions="caller instruction",
    )

    assert payload["instructions"] == "caller instruction\n\nsystem instruction"
    assert payload["input"] == [
        {"content": "hello", "role": "user", "type": "message"}
    ]
