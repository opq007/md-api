"""Runtime configuration for the ModelScope → OpenAI image proxy."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_BASE_URL = "https://api-inference.modelscope.cn"
DEFAULT_TASK_TYPE = "image_generation"


def _split_tokens(raw: str | None) -> list[str]:
    """Parse a comma (or whitespace/newline) separated token list.

    Accepts the three shapes people actually paste in practice: a single value,
    a comma separated list, and a newline separated blob copied out of a file.
    """
    if not raw:
        return []
    parts: list[str] = []
    for chunk in raw.replace("\r\n", "\n").replace(",", "\n").split("\n"):
        token = chunk.strip()
        if token and not token.startswith("#"):
            parts.append(token)
    return parts


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class Settings:
    """Everything the proxy needs to know at runtime."""

    tokens: list[str] = field(default_factory=list)
    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    task_type: str = DEFAULT_TASK_TYPE

    # Blocking wait budget for a single image request.
    request_timeout: float = 300.0
    poll_initial_interval: float = 2.0
    poll_max_interval: float = 10.0
    poll_backoff: float = 1.5

    # Upstream HTTP behaviour.
    connect_timeout: float = 15.0
    upstream_timeout: float = 60.0
    max_retries: int = 2

    # How many tokens to try when a submit fails with an auth/quota error.
    max_token_attempts: int = 0  # 0 => try every token at most once

    @classmethod
    def from_env(cls) -> "Settings":
        tokens = _split_tokens(os.getenv("MODELSCOPE_TOKENS"))
        api_key = os.getenv("PROXY_API_KEY") or None
        settings = cls(
            tokens=tokens,
            api_key=api_key,
            base_url=(os.getenv("MODELSCOPE_BASE_URL") or DEFAULT_BASE_URL).rstrip("/"),
            task_type=os.getenv("MODELSCOPE_TASK_TYPE") or DEFAULT_TASK_TYPE,
            request_timeout=_env_float("PROXY_REQUEST_TIMEOUT", 300.0),
            poll_initial_interval=_env_float("PROXY_POLL_INITIAL_INTERVAL", 2.0),
            poll_max_interval=_env_float("PROXY_POLL_MAX_INTERVAL", 10.0),
            poll_backoff=_env_float("PROXY_POLL_BACKOFF", 1.5),
            connect_timeout=_env_float("PROXY_CONNECT_TIMEOUT", 15.0),
            upstream_timeout=_env_float("PROXY_UPSTREAM_TIMEOUT", 60.0),
            max_retries=_env_int("PROXY_MAX_RETRIES", 2),
        )
        settings.max_token_attempts = (
            _env_int("PROXY_MAX_TOKEN_ATTEMPTS", 0) or len(tokens)
        )
        return settings

    @property
    def images_url(self) -> str:
        return f"{self.base_url}/v1/images/generations"

    @property
    def tasks_url(self) -> str:
        return f"{self.base_url}/v1/tasks"

    @property
    def models_url(self) -> str:
        return f"{self.base_url}/v1/models"

    def task_url(self, task_id: str) -> str:
        return f"{self.tasks_url}/{task_id}"
