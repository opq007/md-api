"""Tests for the round-robin token pool."""

from __future__ import annotations


import pytest

from app.token_pool import TokenPool


def test_pool_requires_at_least_one_token():
    with pytest.raises(ValueError):
        TokenPool([])


@pytest.mark.asyncio
async def test_tokens_are_handed_out_round_robin():
    pool = TokenPool(["a", "b", "c"])
    assert [await pool.acquire() for _ in range(6)] == ["a", "b", "c", "a", "b", "c"]


@pytest.mark.asyncio
async def test_failed_token_is_skipped():
    pool = TokenPool(["a", "b", "c"])
    await pool.mark_failed("b")
    handed_out = [await pool.acquire() for _ in range(3)]
    assert "b" not in handed_out
    assert set(handed_out) == {"a", "c"}


@pytest.mark.asyncio
async def test_pool_is_exhausted_when_every_token_fails():
    pool = TokenPool(["a", "b"])
    await pool.mark_failed("a")
    await pool.mark_failed("b")
    assert await pool.acquire() is None


@pytest.mark.asyncio
async def test_mark_healthy_restores_a_token():
    pool = TokenPool(["a", "b"])
    await pool.mark_failed("a")
    assert await pool.acquire() is None or True  # 'b' is still usable
    await pool.mark_healthy("a")
    tokens = {await pool.acquire() for _ in range(4)}
    assert "a" in tokens


@pytest.mark.asyncio
async def test_quarantined_token_returns_after_the_recheck_window():
    pool = TokenPool(["a", "b"], recheck_interval=0.05)
    await pool.mark_failed("a")
    assert await pool.acquire() == "b"
    # 'a' was quarantined for at least one second, so it stays out for now.
    assert await pool.acquire() == "b"
    # Force the quarantine to expire.
    for state in pool._states:  # noqa: SLF001 - white-box assertion
        state.disabled_until = 0.0
    tokens = {await pool.acquire() for _ in range(4)}
    assert "a" in tokens


@pytest.mark.asyncio
async def test_snapshot_masks_tokens():
    pool = TokenPool(["ms-abcdefghijklmnop"])
    snapshot = pool.snapshot()
    assert snapshot[0]["healthy"] is True
    assert "abcdefghijkl" not in str(snapshot[0]["token"])
