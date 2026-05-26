"""Codex access-token discovery for the experimental openai-codex provider."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


CODEX_ACCESS_TOKEN_ENV = "CODEX_ACCESS_TOKEN"

_ACCESS_TOKEN_KEYS = {
    "access_token",
    "accesstoken",
    "codex_access_token",
    "codexaccesstoken",
    "openai_access_token",
    "openaiaccesstoken",
}


@dataclass(frozen=True)
class CodexTokenResult:
    """Result of resolving a Codex Access Token."""

    token: Optional[str]
    source: Optional[str]
    error: Optional[str] = None

    @property
    def found(self) -> bool:
        return bool(self.token)


def get_codex_auth_path(
    *,
    env: Mapping[str, str] | None = None,
    codex_home: str | Path | None = None,
) -> Path:
    """Return the expected file-backed Codex auth path."""
    env = env or os.environ
    home = codex_home or env.get("CODEX_HOME") or Path.home() / ".codex"
    return Path(home).expanduser() / "auth.json"


def _find_access_token(value: Any) -> Optional[str]:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized_key = str(key).replace("-", "_").lower()
            if normalized_key in _ACCESS_TOKEN_KEYS and isinstance(nested, str):
                token = nested.strip()
                if token:
                    return token
        for nested in value.values():
            token = _find_access_token(nested)
            if token:
                return token
    elif isinstance(value, list):
        for nested in value:
            token = _find_access_token(nested)
            if token:
                return token
    return None


def resolve_codex_access_token(
    *,
    env: Mapping[str, str] | None = None,
    codex_home: str | Path | None = None,
) -> CodexTokenResult:
    """Resolve a Codex Access Token from env, then file-backed Codex auth.

    The Codex auth file is intentionally parsed defensively because the
    official credential-store schema is not a stable TradingAgents contract.
    """
    env = env or os.environ
    env_token = env.get(CODEX_ACCESS_TOKEN_ENV, "").strip()
    if env_token:
        return CodexTokenResult(env_token, CODEX_ACCESS_TOKEN_ENV)

    auth_path = get_codex_auth_path(env=env, codex_home=codex_home)
    if not auth_path.exists():
        return CodexTokenResult(
            None,
            None,
            (
                f"No Codex Access Token found. Set {CODEX_ACCESS_TOKEN_ENV} "
                f"or sign in with Codex so {auth_path} exists."
            ),
        )

    try:
        data = json.loads(auth_path.read_text(encoding="utf-8"))
    except OSError as exc:
        return CodexTokenResult(None, str(auth_path), f"Could not read {auth_path}: {exc}")
    except json.JSONDecodeError as exc:
        return CodexTokenResult(None, str(auth_path), f"Could not parse {auth_path}: {exc}")

    token = _find_access_token(data)
    if token:
        return CodexTokenResult(token, str(auth_path))

    return CodexTokenResult(
        None,
        str(auth_path),
        (
            f"Could not find a Codex Access Token in {auth_path}. If Codex stores "
            f"credentials in your OS keyring, export {CODEX_ACCESS_TOKEN_ENV} "
            "before running TradingAgents."
        ),
    )
