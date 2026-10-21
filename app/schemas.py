"""Request/response schemas for the OpenAI-compatible surface."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

ResponseFormat = Literal["url", "b64_json"]


class ImageGenerationRequest(BaseModel):
    """OpenAI `POST /v1/images/generations` body.

    OpenAI-only fields that ModelScope has no equivalent for (``n``, ``quality``,
    ``style``, ``user``) are accepted and ignored so that stock OpenAI clients
    work unmodified. ModelScope-native fields (``seed``, ``steps``, ``guidance``,
    ``negative_prompt``, ``image_url``, ``loras``) are passed through as-is.
    """

    model_config = ConfigDict(extra="ignore")

    model: str
    prompt: str

    # OpenAI-standard fields.
    n: int | None = None
    size: str | None = None
    quality: str | None = None
    style: str | None = None
    user: str | None = None
    response_format: ResponseFormat | None = None

    # ModelScope-native fields.
    negative_prompt: str | None = None
    seed: int | None = None
    steps: int | None = None
    guidance: float | None = None
    image_url: str | list[str] | None = None
    loras: str | dict[str, float] | None = None

    @field_validator("prompt")
    @classmethod
    def _prompt_length(cls, value: str) -> str:
        if len(value) >= 2000:
            raise ValueError("prompt must be shorter than 2000 characters")
        return value

    @field_validator("negative_prompt")
    @classmethod
    def _negative_prompt_length(cls, value: str | None) -> str | None:
        if value is not None and len(value) >= 4000:
            raise ValueError("negative_prompt must be shorter than 4000 characters")
        return value

    @field_validator("loras")
    @classmethod
    def _validate_loras(cls, value: str | dict[str, float] | None) -> str | dict[str, float] | None:
        return validate_loras(value)


def validate_loras(value: str | dict[str, float] | None) -> str | dict[str, float] | None:
    """Validate a LoRA spec against ModelScope's rules.

    ModelScope accepts either a single repo id string or a mapping of repo id to
    weight, with at most 6 entries whose weights sum to 1.0.
    """
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("loras must not be empty")
        return value.strip()
    if not isinstance(value, dict):
        raise ValueError("loras must be a repo id string or a {repo_id: weight} object")
    if not value:
        raise ValueError("loras must not be empty")
    if len(value) > 6:
        raise ValueError("at most 6 LoRAs are supported")
    total = 0.0
    for repo_id, weight in value.items():
        if not repo_id.strip():
            raise ValueError("LoRA repo id must not be empty")
        if not isinstance(weight, (int, float)) or isinstance(weight, bool):
            raise ValueError(f"LoRA weight for {repo_id!r} must be a number")
        total += float(weight)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"LoRA weights must sum to 1.0, got {total}")
    return {k: float(v) for k, v in value.items()}


class ImageData(BaseModel):
    """One entry of an OpenAI image response's ``data`` array."""

    url: str | None = None
    b64_json: str | None = None
    revised_prompt: str | None = None


class ImageGenerationResponse(BaseModel):
    """OpenAI `ImagesResponse` shape."""

    created: int
    data: list[ImageData]
    model: str | None = None
    object: str = "list"


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int | None = None
    owned_by: str = "system"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


def openai_error_payload(message: str, err_type: str, code: str | None, param: str | None) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": err_type,
            "param": param,
            "code": code,
        }
    }
