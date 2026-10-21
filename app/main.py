"""FastAPI application exposing an OpenAI-compatible image API over ModelScope."""

from __future__ import annotations

import base64
import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any, AsyncIterator

import httpx
from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .config import Settings
from .errors import (
    AuthenticationError,
    ConfigurationError,
    InvalidRequestError,
    ProxyError,
)
from .mapping import build_upstream_payload, bytes_to_data_url
from .schemas import (
    ImageData,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ModelInfo,
    ModelListResponse,
    validate_loras,
)
from .upstream import ModelScopeClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("md-api")

settings = Settings.from_env()

_client: ModelScopeClient | None = None


def get_client() -> ModelScopeClient:
    """Return the process-wide upstream client, creating it on first use."""
    global _client
    if _client is None:
        if not settings.tokens:
            raise ConfigurationError(
                "No ModelScope access tokens configured. Set MODELSCOPE_TOKENS to a "
                "comma separated list of tokens."
            )
        _client = ModelScopeClient(settings)
    return _client


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


app = FastAPI(
    title="ModelScope OpenAI-Compatible Image API",
    description=(
        "An OpenAI-compatible facade over the ModelScope API-Inference image "
        "endpoints. Supports text-to-image, image editing and multi-image "
        "editing, LoRA configuration and a multi-account token pool."
    ),
    version="1.0.0",
    lifespan=_lifespan,
)


# ----------------------------------------------------------------------
# auth
# ----------------------------------------------------------------------
def _extract_bearer(request: Request) -> str | None:
    header = request.headers.get("authorization") or ""
    if not header.lower().startswith("bearer "):
        return None
    return header[7:].strip() or None


async def require_api_key(request: Request) -> None:
    """Validate the client-facing API key of the proxy itself."""
    if not settings.api_key:
        # Auth is disabled: the proxy is meant for local/trusted use.
        return
    presented = _extract_bearer(request)
    if presented != settings.api_key:
        raise AuthenticationError(
            "Incorrect API key provided. Pass it as 'Authorization: Bearer <key>'."
        )


# ----------------------------------------------------------------------
# error handling
# ----------------------------------------------------------------------
@app.exception_handler(ProxyError)
async def _proxy_error_handler(_: Request, exc: ProxyError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.to_payload())


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    first = exc.errors()[0] if exc.errors() else {}
    message = first.get("msg", "Invalid request body.")
    if message.startswith("Value error, "):
        message = message[len("Value error, ") :]
    location = ".".join(str(part) for part in first.get("loc", ()) if part not in ("body",))
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": location or None,
                "code": "invalid_request",
            }
        },
    )


@app.exception_handler(ValidationError)
async def _pydantic_validation_handler(_: Request, exc: ValidationError) -> JSONResponse:
    """Validation raised inside an endpoint body (not by FastAPI's own parsing).

    The multipart endpoint builds its request model by hand, so Pydantic
    errors surface here rather than through ``RequestValidationError``.
    """
    first = exc.errors()[0] if exc.errors() else {}
    message = first.get("msg", "Invalid request body.")
    if message.startswith("Value error, "):
        message = message[len("Value error, ") :]
    location = ".".join(str(part) for part in first.get("loc", ()))
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": location or None,
                "code": "invalid_request",
            }
        },
    )


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
async def _resolve_images(
    urls: list[str],
    response_format: str | None,
    client: ModelScopeClient,
) -> list[ImageData]:
    """Turn ModelScope output URLs into OpenAI ``data`` entries."""
    data: list[ImageData] = []
    for url in urls:
        if response_format == "b64_json":
            encoded = await _download_as_base64(url, client)
            data.append(ImageData(b64_json=encoded))
        else:
            # ModelScope's own CDN URL is passed through untouched.
            data.append(ImageData(url=url))
    return data


async def _download_as_base64(url: str, client: ModelScopeClient) -> str:
    try:
        response = await client.http.get(url, timeout=60.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ProxyError(f"Failed to download the generated image: {exc}") from exc
    return base64.b64encode(response.content).decode("ascii")


def _parse_multipart_loras(form: Any) -> str | dict[str, float] | None:
    """Collect LoRA weights from a multipart form.

    The OpenAI Python SDK serialises ``extra_body={"loras": {"repo/id": 0.6}}``
    into multipart form fields named ``loras[repo/id]``, so both the bracketed
    and the plain forms are supported.
    """
    bracketed: dict[str, float] = {}
    plain: str | None = None

    for key, value in form.multi_items():
        if key.startswith("loras[") and key.endswith("]"):
            repo_id = key[len("loras[") : -1]
            try:
                bracketed[repo_id] = float(value)
            except (TypeError, ValueError):
                raise InvalidRequestError(
                    f"LoRA weight for {repo_id!r} must be a number, got {value!r}."
                ) from None
        elif key == "loras":
            plain = str(value)

    if bracketed:
        return bracketed
    if plain:
        return plain
    return None


def _loras_from_form(form: Any) -> str | dict[str, float] | None:
    loras = _parse_multipart_loras(form)
    if loras is None:
        return None
    try:
        return validate_loras(loras)
    except ValueError as exc:
        raise InvalidRequestError(str(exc)) from exc


# ----------------------------------------------------------------------
# endpoints
# ----------------------------------------------------------------------
@app.post("/v1/images/generations", response_model=ImageGenerationResponse)
async def create_image(
    request: ImageGenerationRequest,
    _: None = Depends(require_api_key),
    client: ModelScopeClient = Depends(get_client),
) -> ImageGenerationResponse:
    """OpenAI-compatible text-to-image / image-editing endpoint.

    The OpenAI SDK sends ``extra_body`` contents flattened into the top-level
    JSON body, so a LoRA configuration arrives here as a top-level ``loras``
    field. Passing ``image_url`` here (a URL or a list of URLs) also enables
    image editing without the multipart endpoint.
    """
    payload = build_upstream_payload(request)
    urls = await client.generate(payload)
    data = await _resolve_images(urls, request.response_format, client)
    return ImageGenerationResponse(created=int(time.time()), data=data, model=request.model)


@app.post("/v1/images/edits", response_model=ImageGenerationResponse)
async def edit_image(
    request: Request,
    model: Annotated[str, Form()],
    prompt: Annotated[str, Form()],
    image: Annotated[list[UploadFile], File(alias="image[]")] = None,  # type: ignore[assignment]
    single_image: Annotated[list[UploadFile], File(alias="image")] = None,  # type: ignore[assignment]
    _: None = Depends(require_api_key),
    client: ModelScopeClient = Depends(get_client),
) -> ImageGenerationResponse:
    """OpenAI-compatible image editing endpoint.

    Accepts one or more uploaded images. The OpenAI Python SDK names the part
    ``image[]`` when a list of images is passed and ``image`` when a single
    image is passed, so both are accepted here. Every uploaded image is
    forwarded to ModelScope as a base64 data URL, which is ModelScope's
    multi-image editing input.
    """
    form = await request.form()
    loras = _loras_from_form(form)

    uploads: list[UploadFile] = []
    if image:
        uploads.extend(image)
    if single_image:
        uploads.extend(single_image)
    if not uploads:
        raise InvalidRequestError(
            "At least one image is required. Send the file as 'image' or 'image[]'."
        )

    request_model = ImageGenerationRequest(
        model=model,
        prompt=prompt,
        size=_form_str(form, "size"),
        response_format=_form_str(form, "response_format"),
        negative_prompt=_form_str(form, "negative_prompt"),
        seed=_form_int(form, "seed"),
        steps=_form_int(form, "steps"),
        guidance=_form_float(form, "guidance"),
    )

    images: list[str] = []
    for upload in uploads:
        data = await upload.read()
        if not data:
            raise InvalidRequestError(f"Uploaded image {upload.filename!r} is empty.")
        images.append(bytes_to_data_url(data, upload.content_type, upload.filename))

    payload = build_upstream_payload(request_model, image_urls=images, loras=loras)
    urls = await client.generate(payload)
    data = await _resolve_images(urls, request_model.response_format, client)
    return ImageGenerationResponse(
        created=int(time.time()), data=data, model=request_model.model
    )


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models(
    _: None = Depends(require_api_key),
    client: ModelScopeClient = Depends(get_client),
) -> ModelListResponse:
    """List the models available through ModelScope API-Inference."""
    payload = await client.fetch_models()
    raw = payload.get("data", []) if isinstance(payload, dict) else []
    models: list[ModelInfo] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        identifier = entry.get("id")
        if not identifier:
            continue
        models.append(
            ModelInfo(
                id=str(identifier),
                created=entry.get("created"),
                owned_by=str(entry.get("owned_by") or "system"),
            )
        )
    return ModelListResponse(data=models)


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    """Liveness probe plus a view of the token pool's health."""
    return {
        "status": "ok",
        "tokens_configured": len(settings.tokens),
        "pool": get_client().pool.snapshot() if settings.tokens else [],
    }


# ----------------------------------------------------------------------
# small form helpers
# ----------------------------------------------------------------------
def _form_str(form: Any, name: str) -> str | None:
    value = form.get(name)
    if value is None:
        return None
    return str(value)


def _form_int(form: Any, name: str) -> int | None:
    value = _form_str(form, name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        raise InvalidRequestError(f"{name} must be an integer, got {value!r}.") from None


def _form_float(form: Any, name: str) -> float | None:
    value = _form_str(form, name)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        raise InvalidRequestError(f"{name} must be a number, got {value!r}.") from None
