"""End-to-end compatibility test with the real OpenAI Python SDK.

The proxy is exercised through the official ``openai`` client so the exact
wire format the SDK produces (multipart ``image[]`` parts, ``loras[...]`` form
fields, flattened ``extra_body``) is validated against the proxy.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from typing import Iterator

import httpx
import pytest
import uvicorn
from openai import OpenAI

from app.config import Settings
from app.main import app, get_client
from app.upstream import ModelScopeClient

IMAGE_URL = "https://modelscope.oss-cn-beijing.aliyuncs.com/out.png"


class MockTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/v1/images/generations"):
            return httpx.Response(200, json={"task_id": "task-1"})
        if "/v1/tasks/" in request.url.path:
            return httpx.Response(
                200, json={"task_status": "SUCCEED", "output_images": [IMAGE_URL]}
            )
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "Qwen/Qwen-Image"}]})
        return httpx.Response(200, content=b"IMAGEBYTES")


@pytest.fixture()
def transport() -> MockTransport:
    return MockTransport()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture()
def server(transport: MockTransport) -> Iterator[str]:
    """Run the proxy on a real port so the sync OpenAI client can reach it."""
    settings = Settings(
        tokens=["token-a"],
        api_key="test-key",
        request_timeout=5.0,
        poll_initial_interval=0.01,
        poll_max_interval=0.02,
    )
    upstream = ModelScopeClient(settings, client=httpx.AsyncClient(transport=transport))
    app.dependency_overrides[get_client] = lambda: upstream

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("the test server failed to start")

    yield f"http://127.0.0.1:{port}/v1"
    server.should_exit = True
    thread.join(timeout=10)
    app.dependency_overrides.clear()


@pytest.fixture()
def openai_client(server: str) -> OpenAI:
    return OpenAI(base_url=server, api_key="test-key")


def test_sdk_text_to_image(openai_client: OpenAI, transport: MockTransport):
    result = openai_client.images.generate(
        model="Qwen/Qwen-Image",
        prompt="a golden cat",
        size="1024x1024",
        n=1,
    )
    assert result.data[0].url == IMAGE_URL
    assert result.created > 0

    sent = json.loads(transport.requests[0].content)
    assert sent == {"model": "Qwen/Qwen-Image", "prompt": "a golden cat", "size": "1024x1024"}


def test_sdk_lora_via_extra_body(openai_client: OpenAI, transport: MockTransport):
    openai_client.images.generate(
        model="Qwen/Qwen-Image",
        prompt="a golden cat",
        extra_body={"loras": {"owner/a": 0.6, "owner/b": 0.4}},
    )
    sent = json.loads(transport.requests[0].content)
    assert sent["loras"] == {"owner/a": 0.6, "owner/b": 0.4}


def test_sdk_single_image_edit(openai_client: OpenAI, transport: MockTransport):
    result = openai_client.images.edit(
        model="Qwen/Qwen-Image-Edit",
        image=("input.png", b"AAAA", "image/png"),
        prompt="turn the hair blue",
    )
    assert result.data[0].url == IMAGE_URL

    sent = json.loads(transport.requests[0].content)
    assert sent["image_url"] == "data:image/png;base64,QUFBQQ=="
    assert sent["model"] == "Qwen/Qwen-Image-Edit"
    assert sent["prompt"] == "turn the hair blue"


def test_sdk_multi_image_edit(openai_client: OpenAI, transport: MockTransport):
    result = openai_client.images.edit(
        model="Qwen/Qwen-Image-Edit",
        image=[
            ("a.png", b"AAAA", "image/png"),
            ("b.png", b"BBBB", "image/png"),
        ],
        prompt="combine these two",
    )
    assert result.data[0].url == IMAGE_URL

    sent = json.loads(transport.requests[0].content)
    assert isinstance(sent["image_url"], list)
    assert len(sent["image_url"]) == 2
    assert sent["image_url"][0] == "data:image/png;base64,QUFBQQ=="
    assert sent["image_url"][1] == "data:image/png;base64,QkJCQg=="


def test_sdk_multi_image_edit_with_lora(openai_client: OpenAI, transport: MockTransport):
    openai_client.images.edit(
        model="Qwen/Qwen-Image-Edit",
        image=[
            ("a.png", b"AAAA", "image/png"),
            ("b.png", b"BBBB", "image/png"),
        ],
        prompt="combine these two",
        extra_body={"loras": {"owner/a": 0.6, "owner/b": 0.4}},
    )
    sent = json.loads(transport.requests[0].content)
    assert sent["loras"] == {"owner/a": 0.6, "owner/b": 0.4}
    assert len(sent["image_url"]) == 2


def test_sdk_b64_json(openai_client: OpenAI):
    result = openai_client.images.generate(
        model="Qwen/Qwen-Image",
        prompt="a golden cat",
        response_format="b64_json",
    )
    assert result.data[0].b64_json == "SU1BR0VCWVRFUw=="


def test_sdk_list_models(openai_client: OpenAI):
    models = openai_client.models.list()
    assert models.data[0].id == "Qwen/Qwen-Image"
