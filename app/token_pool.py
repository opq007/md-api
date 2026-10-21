"""Round-robin pool of ModelScope access tokens.

The pool hands out one token at a time. A token that fails with an auth, quota
or rate-limit error is marked unusable and skipped for subsequent requests; if
``recheck_interval`` is configured it becomes eligible again after that many
seconds, so a temporarily broken token does not permanently shrink the pool.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass


@dataclass
class _TokenState:
    token: str
    disabled_until: float = 0.0
    failures: int = 0


class TokenPool:
    def __init__(self, tokens: list[str], recheck_interval: float = 0.0) -> None:
        if not tokens:
            raise ValueError("token pool requires at least one token")
        self._states = [_TokenState(token=t) for t in tokens]
        self._recheck_interval = recheck_interval
        self._cursor = 0
        self._lock = asyncio.Lock()

    def __len__(self) -> int:
        return len(self._states)

    @property
    def tokens(self) -> list[str]:
        return [s.token for s in self._states]

    def _is_healthy(self, state: _TokenState, now: float) -> bool:
        if state.disabled_until <= 0:
            return True
        if self._recheck_interval > 0 and now >= state.disabled_until:
            # The quarantine window has elapsed; let it be tried again.
            state.disabled_until = 0.0
            state.failures = 0
            return True
        return False

    async def acquire(self) -> str | None:
        """Return the next usable token, or ``None`` when the pool is exhausted."""
        async with self._lock:
            now = time.monotonic()
            total = len(self._states)
            for _ in range(total):
                state = self._states[self._cursor]
                self._cursor = (self._cursor + 1) % total
                if self._is_healthy(state, now):
                    return state.token
            return None

    async def mark_failed(self, token: str) -> None:
        """Mark a token unusable after an auth / quota / rate-limit failure."""
        async with self._lock:
            for state in self._states:
                if state.token == token:
                    state.failures += 1
                    state.disabled_until = time.monotonic() + max(self._recheck_interval, 1.0)
                    return

    async def mark_healthy(self, token: str) -> None:
        """Clear the failure state of a token that worked."""
        async with self._lock:
            for state in self._states:
                if state.token == token:
                    state.failures = 0
                    state.disabled_until = 0.0
                    return

    def snapshot(self) -> list[dict[str, object]]:
        now = time.monotonic()
        return [
            {
                "token": f"{s.token[:6]}...{s.token[-4:]}" if len(s.token) > 12 else s.token,
                "healthy": self._is_healthy(s, now),
                "failures": s.failures,
            }
            for s in self._states
        ]
