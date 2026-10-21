"""Tests for the OpenAI <-> ModelScope parameter mapping."""

from __future__ import annotations

import pytest

from app.mapping import build_upstream_payload, bytes_to_data_url
from app.schemas import ImageGenerationRequest, validate_loras


def test_openai_only_fields_are_dropped():
    request = ImageGenerationRequest(
        model="Qwen/Qwen-Image",
        prompt="a cat",
        n=4,
        quality="hd",
        style="vivid",
        user="user-1",
    )
    payload = build_upstream_payload(request)

    assert payload == {"model": "Qwen/Qwen-Image", "prompt": "a cat"}
    for dropped in ("n", "quality", "style", "user"):
        assert dropped not in payload


def test_modelscope_native_fields_pass_through():
    request = ImageGenerationRequest(
        model="Qwen/Qwen-Image",
        prompt="a cat",
        size="1024x1024",
        negative_prompt="blurry",
        seed=42,
        steps=30,
        guidance=3.5,
    )
    payload = build_upstream_payload(request)

    assert payload["size"] == "1024x1024"
    assert payload["negative_prompt"] == "blurry"
    assert payload["seed"] == 42
    assert payload["steps"] == 30
    assert payload["guidance"] == 3.5


def test_single_image_url_is_sent_as_string():
    request = ImageGenerationRequest(
        model="m", prompt="p", image_url="https://example.com/a.png"
    )
    payload = build_upstream_payload(request)
    assert payload["image_url"] == "https://example.com/a.png"


def test_multi_image_url_is_sent_as_list():
    request = ImageGenerationRequest(
        model="m",
        prompt="p",
        image_url=["https://example.com/a.png", "https://example.com/b.png"],
    )
    payload = build_upstream_payload(request)
    assert payload["image_url"] == [
        "https://example.com/a.png",
        "https://example.com/b.png",
    ]


def test_uploaded_images_override_request_image_url():
    request = ImageGenerationRequest(model="m", prompt="p", image_url="https://x/a.png")
    payload = build_upstream_payload(
        request, image_urls=["data:image/png;base64,AAA", "data:image/png;base64,BBB"]
    )
    assert payload["image_url"] == [
        "data:image/png;base64,AAA",
        "data:image/png;base64,BBB",
    ]


def test_lora_passthrough():
    request = ImageGenerationRequest(model="m", prompt="p", loras="owner/lora")
    payload = build_upstream_payload(request)
    assert payload["loras"] == "owner/lora"


def test_lora_dict_passthrough():
    request = ImageGenerationRequest(
        model="m", prompt="p", loras={"owner/a": 0.6, "owner/b": 0.4}
    )
    payload = build_upstream_payload(request)
    assert payload["loras"] == {"owner/a": 0.6, "owner/b": 0.4}


def test_lora_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        validate_loras({"owner/a": 0.6, "owner/b": 0.6})


def test_at_most_six_loras():
    loras = {f"owner/l{i}": 1 / 6 for i in range(6)}
    assert validate_loras(loras) is not None
    loras["owner/l6"] = 0.0
    with pytest.raises(ValueError):
        validate_loras(loras)


def test_prompt_length_limit():
    with pytest.raises(ValueError):
        ImageGenerationRequest(model="m", prompt="x" * 2000)


def test_negative_prompt_length_limit():
    with pytest.raises(ValueError):
        ImageGenerationRequest(model="m", prompt="p", negative_prompt="x" * 4000)


def test_bytes_to_data_url_uses_content_type():
    assert bytes_to_data_url(b"abc", "image/jpeg", "a.jpg").startswith(
        "data:image/jpeg;base64,"
    )


def test_bytes_to_data_url_falls_back_to_extension():
    assert bytes_to_data_url(b"abc", None, "a.png").startswith("data:image/png;base64,")
    assert bytes_to_data_url(b"abc", None, "a.jpeg").startswith("data:image/jpeg;base64,")
    assert bytes_to_data_url(b"abc", None, None).startswith("data:image/png;base64,")
