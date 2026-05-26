"""Authentication helpers for ChatGPT Codex OAuth tokens.

The ``openai-codex`` provider is not the same as OpenAI API-key auth.  Pi's
implementation sends ChatGPT OAuth access tokens to the ChatGPT Codex backend
(``chatgpt.com/backend-api/codex/responses``), with the ChatGPT account id
extracted from the token.  Sending the same token to ``api.openai.com`` fails
with missing API scopes.
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CODEX_PROVIDER = "openai-codex"
CODEX_ACCESS_TOKEN_ENVS = ("CODEX_ACCESS_TOKEN", "OPENAI_CODEX_ACCESS_TOKEN")
PI_AUTH_PATH = Path.home() / ".pi" / "agent" / "auth.json"

_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_TOKEN_URL = "https://auth.openai.com/oauth/token"
_JWT_CLAIM_PATH = "https://api.openai.com/auth"


@dataclass(frozen=True)
class CodexTokenResult:
    token: str
    source: str
    account_id: str


def _decode_jwt_payload(token: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("OpenAI Codex token is not a JWT.")
    payload = parts[1]
    padding = "=" * (-len(payload) % 4)
    decoded = base64.urlsafe_b64decode(payload + padding)
    return json.loads(decoded)


def extract_codex_account_id(token: str) -> str:
    """Extract ChatGPT account id from an OpenAI Codex OAuth access token."""
    try:
        payload = _decode_jwt_payload(token)
        account_id = payload.get(_JWT_CLAIM_PATH, {}).get("chatgpt_account_id")
    except Exception as exc:
        raise ValueError("Failed to extract ChatGPT account id from Codex token.") from exc
    if not isinstance(account_id, str) or not account_id:
        raise ValueError("Codex token does not contain a ChatGPT account id.")
    return account_id


def _read_pi_auth(path: Path | None = None) -> dict[str, Any] | None:
    path = path or PI_AUTH_PATH
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse Pi auth file at {path}.") from exc


def _write_pi_auth(auth: dict[str, Any], path: Path | None = None) -> None:
    path = path or PI_AUTH_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(auth, indent=2) + "\n", encoding="utf-8")


def _refresh_pi_codex_token(
    credentials: dict[str, Any],
    auth: dict[str, Any],
    path: Path | None = None,
) -> str | None:
    import requests

    refresh_token = credentials.get("refresh")
    if not isinstance(refresh_token, str) or not refresh_token:
        return None

    response = requests.post(
        _TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": _CLIENT_ID,
        },
        timeout=15,
    )
    response.raise_for_status()
    refreshed = response.json()
    access = refreshed.get("access_token")
    refresh = refreshed.get("refresh_token")
    expires_in = refreshed.get("expires_in")
    if not access or not refresh or not isinstance(expires_in, (int, float)):
        raise ValueError("OpenAI Codex token refresh response was missing fields.")

    account_id = extract_codex_account_id(access)
    auth[CODEX_PROVIDER] = {
        "type": "oauth",
        "access": access,
        "refresh": refresh,
        "expires": int(time.time() * 1000 + expires_in * 1000),
        "accountId": account_id,
    }
    _write_pi_auth(auth, path)
    return access


def _resolve_pi_codex_token(path: Path | None = None) -> str | None:
    auth = _read_pi_auth(path)
    if not auth:
        return None
    credentials = auth.get(CODEX_PROVIDER)
    if not isinstance(credentials, dict) or credentials.get("type") != "oauth":
        return None

    access = credentials.get("access")
    expires = credentials.get("expires")
    if isinstance(access, str) and (
        not isinstance(expires, (int, float)) or int(time.time() * 1000) < expires
    ):
        return access

    return _refresh_pi_codex_token(credentials, auth, path)


def has_openai_codex_credentials() -> bool:
    """Return True if env or Pi auth has usable-looking Codex credentials."""
    if any(os.environ.get(env_var) for env_var in CODEX_ACCESS_TOKEN_ENVS):
        return True
    auth = _read_pi_auth()
    credentials = auth.get(CODEX_PROVIDER) if auth else None
    return isinstance(credentials, dict) and isinstance(credentials.get("access"), str)


def resolve_openai_codex_token() -> CodexTokenResult:
    """Resolve a ChatGPT Codex OAuth token from env or Pi's auth store."""
    for env_var in CODEX_ACCESS_TOKEN_ENVS:
        token = os.environ.get(env_var)
        if token:
            return CodexTokenResult(token, env_var, extract_codex_account_id(token))

    token = _resolve_pi_codex_token()
    if token:
        return CodexTokenResult(token, str(PI_AUTH_PATH), extract_codex_account_id(token))

    envs = " or ".join(CODEX_ACCESS_TOKEN_ENVS)
    raise ValueError(
        "OpenAI Codex credentials were not found. Set "
        f"{envs}, or run `pi` /login for openai-codex so "
        f"{PI_AUTH_PATH} contains an openai-codex OAuth entry."
    )


def openai_codex_headers(token_result: CodexTokenResult) -> dict[str, str]:
    """Headers required by ChatGPT's Codex Responses endpoint."""
    return {
        "Authorization": f"Bearer {token_result.token}",
        "chatgpt-account-id": token_result.account_id,
        "originator": "tradingagents",
        "OpenAI-Beta": "responses=experimental",
        "User-Agent": "tradingagents",
    }
