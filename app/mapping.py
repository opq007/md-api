"""Translate between the OpenAI image API and ModelScope's native parameters.

Mapping rules (as agreed during the requirements interview):

* ``model``, ``prompt``, ``size`` map straight across.
* OpenAI fields with no ModelScope equivalent (``n``, ``quality``, ``style``,
  ``user``) are dropped.
* ModelScope-native fields (``negative_prompt``, ``seed``, ``steps``,
  ``guidance``) are passed through untouched.
* ``image_url`` accepts a single URL/base64 string or a list of them; the list
  form is ModelScope's multi-image editing input.
* ``loras`` accepts a single repo id or a ``{repo_id: weight}`` mapping.
"""

from __future__ import annotations

import base64
from typing import Any

from .schemas import ImageGenerationRequest, validate_loras

# OpenAI-only request fields that are accepted for client compatibility but have
# no ModelScope counterpart.
IGNORED_OPENAI_FIELDS = ("n", "quality", "style", "user")

# ModelScope request fields that are forwarded verbatim.
PASSTHROUGH_FIELDS = ("negative_prompt", "seed", "steps", "guidance")


def build_upstream_payload(
    request: ImageGenerationRequest,
    *,
    image_urls: list[str] | None = None,
    loras: str | dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build the JSON body sent to ModelScope's ``/v1/images/generations``.

    ``image_urls`` overrides the request's own ``image_url`` field; it is how
    multipart uploads (already converted to base64 data URLs) are injected.
    """
    payload: dict[str, Any] = {
        "model": request.model,
        "prompt": request.prompt,
    }

    if request.size:
        payload["size"] = request.size

    for field in PASSTHROUGH_FIELDS:
        value = getattr(request, field, None)
        if value is not None:
            payload[field] = value

    # Input images: an explicit list wins over the request's own image_url.
    images = image_urls if image_urls is not None else request.image_url
    if images:
        if isinstance(images, str):
            payload["image_url"] = images
        else:
            cleaned = [url for url in images if url]
            if not cleaned:
                pass
            elif len(cleaned) == 1:
                # A lone image is sent in ModelScope's plain string form.
                payload["image_url"] = cleaned[0]
            else:
                # Multiple images select ModelScope's multi-image editing mode.
                payload["image_url"] = cleaned

    resolved_loras = loras if loras is not None else request.loras
    if resolved_loras is not None:
        payload["loras"] = validate_loras(resolved_loras)

    return payload


def bytes_to_data_url(data: bytes, content_type: str | None, filename: str | None = None) -> str:
    """Encode raw image bytes as a ``data:`` URL for ModelScope."""
    mime = content_type or None
    if not mime or not mime.startswith("image/"):
        mime = _guess_mime(filename)
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _guess_mime(filename: str | None) -> str:
    if filename:
        lowered = filename.lower()
        for extension, mime in (
            (".png", "image/png"),
            (".jpg", "image/jpeg"),
            (".jpeg", "image/jpeg"),
            (".webp", "image/webp"),
            (".gif", "image/gif"),
            (".bmp", "image/bmp"),
        ):
            if lowered.endswith(extension):
                return mime
    return "image/png"
