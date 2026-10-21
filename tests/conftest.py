"""Shared pytest fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MODELSCOPE_TOKENS", "token-a,token-b")
os.environ.setdefault("PROXY_API_KEY", "test-key")


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"
