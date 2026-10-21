"""ModelScope API-Inference upstream client.

ModelScope exposes exactly one image endpoint (``POST /v1/images/generations``)
which serves both text-to-image and image editing. It is asynchronous: the
submit call returns a ``task_id`` and the result is collected by polling
``GET /v1/tasks/{task_id}``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from .config import Settings
from .errors import (
    ConfigurationError,
    TaskFailedError,
    UpstreamError,
    UpstreamTimeoutError,
)
from .token_pool import TokenPool

logger = logging.getLogger(__name__)

# Substrings that mark an upstream rejection as "this token is the problem".
# These are deliberately specific: a bare "token" would also match unrelated
# messages such as "prompt is too long, max 2000 tokens".
_TOKEN_FAULT_MARKERS = (
    "authentication failed",
    "invalid token",
    "invalid access token",
    "token is invalid",
    "token expired",
    "expired token",
    "unauthorized",
    "forbidden",
    "quota",
    "insufficient",
    "not enough",
    "balance",
    "no permission",
    "not authorized",
    "access denied",
    "rate limit",
    "too many requests",
)


class ModelScopeClient:
    """Async client for the ModelScope API-Inference image endpoints."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._owns_client = client is None
        self.http = client or httpx.AsyncClient(
            timeout=httpx.Timeout(
                settings.upstream_timeout,
                connect=settings.connect_timeout,
            )
        )
        self.pool = TokenPool(settings.tokens, recheck_interval=settings.poll_max_interval * 6)

    async def aclose(self) -> None:
        if self._owns_client:
            await self.http.aclose()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    @staticmethod
    def _is_token_fault(message: str) -> bool:
        lowered = message.lower()
        return any(marker in lowered for marker in _TOKEN_FAULT_MARKERS)

    @staticmethod
    def _extract_error_message(payload: Any, fallback: str) -> str:
        if isinstance(payload, dict):
            errors = payload.get("errors")
            if isinstance(errors, dict):
                message = errors.get("message")
                if isinstance(message, str):
                    return message
            if isinstance(errors, list) and errors:
                first = errors[0]
                if isinstance(first, dict) and isinstance(first.get("message"), str):
                    return first["message"]
            error = payload.get("error")
            if isinstance(error, dict) and isinstance(error.get("message"), str):
                return error["message"]
            if isinstance(error, str):
                return error
            message = payload.get("message")
            if isinstance(message, str):
                return message
        return fallback

    def _headers(self, token: str, *, async_mode: bool = False) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        if async_mode:
            headers["X-ModelScope-Async-Mode"] = "true"
        return headers

    def _task_headers(self, token: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {token}",
            "X-ModelScope-Task-Type": self.settings.task_type,
        }

    # ------------------------------------------------------------------
    # submit
    # ------------------------------------------------------------------
    async def submit_image_task(self, payload: dict[str, Any]) -> tuple[str, str]:
        """Submit an image generation/editing task.

        Returns the ``task_id`` together with the token that successfully
        submitted it, so the caller can poll with the same credentials.

        On an auth/quota/rate-limit rejection the token is marked unusable and
        the submission is retried with the next token in the pool.
        """
        attempts = max(1, self.settings.max_token_attempts or len(self.pool))
        last_error: Exception | None = None

        for attempt in range(attempts):
            token = await self.pool.acquire()
            if token is None:
                if last_error is not None:
                    # Every token was rejected; report the upstream reason
                    # rather than a bare "pool exhausted" message.
                    raise last_error
                raise ConfigurationError(
                    "All ModelScope access tokens are currently unavailable; "
                    "the token pool is exhausted."
                )

            try:
                response = await self.http.post(
                    self.settings.images_url,
                    headers=self._headers(token, async_mode=True),
                    json=payload,
                )
            except httpx.HTTPError as exc:
                last_error = UpstreamError(f"Failed to reach ModelScope: {exc}")
                logger.warning("Submit attempt %d failed to reach ModelScope: %s", attempt + 1, exc)
                continue

            if response.status_code < 400:
                await self.pool.mark_healthy(token)
                try:
                    body = response.json()
                except ValueError as exc:
                    raise UpstreamError(f"ModelScope returned a non-JSON response: {exc}") from exc
                task_id = body.get("task_id") if isinstance(body, dict) else None
                if not task_id:
                    raise UpstreamError(
                        f"ModelScope did not return a task_id: {self._extract_error_message(body, 'unknown response')}"
                    )
                return str(task_id), token

            try:
                body = response.json()
            except ValueError:
                body = None
            message = self._extract_error_message(body, f"HTTP {response.status_code}")
            last_error = UpstreamError(f"ModelScope rejected the request: {message}")

            if self._is_token_fault(message) or response.status_code in (401, 403, 429):
                logger.warning("Token fault on submit (%s); trying the next token.", message)
                await self.pool.mark_failed(token)
                continue

            raise last_error

        raise last_error or UpstreamError("Failed to submit the image task to ModelScope.")

    # ------------------------------------------------------------------
    # poll
    # ------------------------------------------------------------------
    async def poll_task(self, task_id: str, token: str) -> list[str]:
        """Poll a task until it finishes and return the output image URLs.

        The task lives on ModelScope's side, so a token fault while polling is
        reported to the caller rather than silently retried with another token.
        """
        deadline = time.monotonic() + self.settings.request_timeout
        interval = self.settings.poll_initial_interval
        last_status: str | None = None

        while True:
            try:
                response = await self.http.get(
                    self.settings.task_url(task_id),
                    headers=self._task_headers(token),
                )
            except httpx.HTTPError as exc:
                raise UpstreamError(f"Failed to poll ModelScope task {task_id}: {exc}") from exc

            if response.status_code >= 400:
                try:
                    body = response.json()
                except ValueError:
                    body = None
                message = self._extract_error_message(body, f"HTTP {response.status_code}")
                raise UpstreamError(f"Failed to poll ModelScope task {task_id}: {message}")

            try:
                body = response.json()
            except ValueError as exc:
                raise UpstreamError(f"ModelScope returned a non-JSON task payload: {exc}") from exc

            status = str(body.get("task_status", "")) if isinstance(body, dict) else ""
            if status != last_status:
                logger.info("Task %s status: %s", task_id, status or "UNKNOWN")
                last_status = status

            if status == "SUCCEED":
                images = body.get("output_images") or []
                if isinstance(images, str):
                    images = [images]
                if not images:
                    raise UpstreamError(f"ModelScope task {task_id} succeeded without output images.")
                return [str(url) for url in images]

            if status in ("FAILED", "CANCELED", "CANCELLED"):
                message = self._extract_error_message(body, f"task {status.lower()}")
                raise TaskFailedError(f"ModelScope task {task_id} failed: {message}")

            if time.monotonic() + interval > deadline:
                raise UpstreamTimeoutError(
                    f"Timed out after {self.settings.request_timeout:.0f}s waiting for ModelScope "
                    f"task {task_id} (last status: {status or 'UNKNOWN'})."
                )

            await asyncio.sleep(interval)
            interval = min(interval * self.settings.poll_backoff, self.settings.poll_max_interval)

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------
    async def generate(self, payload: dict[str, Any]) -> list[str]:
        """Submit a task and wait for it, returning the output image URLs."""
        task_id, token = await self.submit_image_task(payload)
        return await self.poll_task(task_id, token)

    async def fetch_models(self) -> dict[str, Any]:
        """Fetch the list of models exposed by ModelScope."""
        try:
            response = await self.http.get(self.settings.models_url)
        except httpx.HTTPError as exc:
            raise UpstreamError(f"Failed to reach ModelScope: {exc}") from exc
        if response.status_code >= 400:
            try:
                body = response.json()
            except ValueError:
                body = None
            message = self._extract_error_message(body, f"HTTP {response.status_code}")
            raise UpstreamError(f"Failed to list ModelScope models: {message}")
        try:
            return response.json()
        except ValueError as exc:
            raise UpstreamError(f"ModelScope returned a non-JSON model list: {exc}") from exc
