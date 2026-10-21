"""API-level tests: OpenAI wire compatibility, token failover and error mapping.

The upstream ModelScope service is replaced by a mock transport so the tests
assert on the exact HTTP traffic the proxy produces.
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app, get_client
from app.upstream import ModelScopeClient

IMAGE_URL = "https://modelscope.oss-cn-beijing.aliyuncs.com/out.png"


class MockTransport(httpx.AsyncBaseTransport):
    """Records requests and replays scripted responses."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.post_responses: list[httpx.Response] = []
        self.get_responses: list[httpx.Response] = []
        self.image_responses: list[httpx.Response] = []
        self.sticky_task_status: str | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)

        if request.url.path.endswith("/v1/images/generations"):
            if self.post_responses:
                return self.post_responses.pop(0)
            return httpx.Response(200, json={"task_id": "task-1"})

        if "/v1/tasks/" in request.url.path:
            if self.get_responses:
                return self.get_responses.pop(0)
            if self.sticky_task_status:
                return httpx.Response(200, json={"task_status": self.sticky_task_status})
            return httpx.Response(
                200, json={"task_status": "SUCCEED", "output_images": [IMAGE_URL]}
            )

        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "Qwen/Qwen-Image"}]})

        # Image download for response_format=b64_json.
        if self.image_responses:
            return self.image_responses.pop(0)
        return httpx.Response(200, content=b"IMAGEBYTES")


@pytest.fixture()
def transport() -> MockTransport:
    return MockTransport()


@pytest.fixture()
def client(transport: MockTransport, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    settings = Settings(
        tokens=["token-a", "token-b"],
        api_key="test-key",
        request_timeout=5.0,
        poll_initial_interval=0.01,
        poll_max_interval=0.02,
        poll_backoff=1.0,
    )
    upstream = ModelScopeClient(settings, client=httpx.AsyncClient(transport=transport))
    app.dependency_overrides[get_client] = lambda: upstream
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# ----------------------------------------------------------------------
# auth
# ----------------------------------------------------------------------
def test_missing_api_key_is_rejected(client: TestClient):
    response = client.post("/v1/images/generations", json={"model": "m", "prompt": "p"})
    assert response.status_code == 401
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_wrong_api_key_is_rejected(client: TestClient):
    response = client.post(
        "/v1/images/generations",
        json={"model": "m", "prompt": "p"},
        headers={"Authorization": "Bearer nope"},
    )
    assert response.status_code == 401


# ----------------------------------------------------------------------
# text to image
# ----------------------------------------------------------------------
def test_text_to_image_returns_openai_shape(client: TestClient, transport: MockTransport):
    response = client.post(
        "/v1/images/generations",
        json={"model": "Qwen/Qwen-Image", "prompt": "a golden cat"},
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["data"][0]["url"] == IMAGE_URL
    assert body["created"] > 0

    submit = transport.requests[0]
    assert submit.url.path == "/v1/images/generations"
    assert submit.headers["authorization"] == "Bearer token-a"
    assert submit.headers["x-modelscope-async-mode"] == "true"
    assert json.loads(submit.content) == {"model": "Qwen/Qwen-Image", "prompt": "a golden cat"}


def test_task_polling_uses_task_type_header(client: TestClient, transport: MockTransport):
    client.post(
        "/v1/images/generations",
        json={"model": "m", "prompt": "p"},
        headers={"Authorization": "Bearer test-key"},
    )
    poll = transport.requests[1]
    assert poll.url.path == "/v1/tasks/task-1"
    assert poll.headers["x-modelscope-task-type"] == "image_generation"
    assert poll.headers["authorization"] == "Bearer token-a"


def test_openai_noise_fields_are_ignored(client: TestClient, transport: MockTransport):
    client.post(
        "/v1/images/generations",
        json={
            "model": "m",
            "prompt": "p",
            "n": 4,
            "quality": "hd",
            "style": "vivid",
            "user": "u1",
            "size": "1024x1024",
        },
        headers={"Authorization": "Bearer test-key"},
    )
    sent = json.loads(transport.requests[0].content)
    assert sent == {"model": "m", "prompt": "p", "size": "1024x1024"}


def test_loras_are_forwarded(client: TestClient, transport: MockTransport):
    client.post(
        "/v1/images/generations",
        json={
            "model": "m",
            "prompt": "p",
            "loras": {"owner/a": 0.6, "owner/b": 0.4},
        },
        headers={"Authorization": "Bearer test-key"},
    )
    assert json.loads(transport.requests[0].content)["loras"] == {
        "owner/a": 0.6,
        "owner/b": 0.4,
    }


def test_invalid_lora_weights_are_rejected(client: TestClient):
    response = client.post(
        "/v1/images/generations",
        json={"model": "m", "prompt": "p", "loras": {"owner/a": 0.5, "owner/b": 0.9}},
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 400
    assert "sum to 1.0" in response.json()["error"]["message"]


def test_b64_json_response_format(client: TestClient, transport: MockTransport):
    response = client.post(
        "/v1/images/generations",
        json={"model": "m", "prompt": "p", "response_format": "b64_json"},
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 200
    assert response.json()["data"][0]["b64_json"] == "SU1BR0VCWVRFUw=="


# ----------------------------------------------------------------------
# multi image editing
# ----------------------------------------------------------------------
def test_multi_image_edit_sends_a_list(client: TestClient, transport: MockTransport):
    response = client.post(
        "/v1/images/edits",
        data={"model": "m", "prompt": "make it blue"},
        files=[
            ("image[]", ("a.png", b"AAAA", "image/png")),
            ("image[]", ("b.png", b"BBBB", "image/png")),
        ],
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 200

    sent = json.loads(transport.requests[0].content)
    assert isinstance(sent["image_url"], list)
    assert len(sent["image_url"]) == 2
    assert sent["image_url"][0].startswith("data:image/png;base64,")
    assert sent["image_url"][1].startswith("data:image/png;base64,")


def test_single_image_edit_uses_image_field(client: TestClient, transport: MockTransport):
    response = client.post(
        "/v1/images/edits",
        data={"model": "m", "prompt": "p"},
        files=[("image", ("a.png", b"AAAA", "image/png"))],
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 200
    sent = json.loads(transport.requests[0].content)
    assert sent["image_url"] == "data:image/png;base64,QUFBQQ=="


def test_multipart_loras_are_parsed(client: TestClient, transport: MockTransport):
    response = client.post(
        "/v1/images/edits",
        data={
            "model": "m",
            "prompt": "p",
            "loras[owner/a]": "0.6",
            "loras[owner/b]": "0.4",
        },
        files=[("image[]", ("a.png", b"AAAA", "image/png"))],
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 200
    sent = json.loads(transport.requests[0].content)
    assert sent["loras"] == {"owner/a": 0.6, "owner/b": 0.4}


def test_edit_without_image_is_rejected(client: TestClient):
    response = client.post(
        "/v1/images/edits",
        data={"model": "m", "prompt": "p"},
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 400


def test_edit_rejects_an_oversized_prompt(client: TestClient):
    """A validation error inside the handler must be a 400, not a 500."""
    response = client.post(
        "/v1/images/edits",
        data={"model": "m", "prompt": "x" * 2500},
        files=[("image[]", ("a.png", b"AAAA", "image/png"))],
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 400
    assert "shorter than 2000" in response.json()["error"]["message"]


def test_edit_rejects_a_non_numeric_seed(client: TestClient):
    response = client.post(
        "/v1/images/edits",
        data={"model": "m", "prompt": "p", "seed": "abc"},
        files=[("image[]", ("a.png", b"AAAA", "image/png"))],
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 400
    assert "seed must be an integer" in response.json()["error"]["message"]


# ----------------------------------------------------------------------
# token pool failover
# ----------------------------------------------------------------------
def test_auth_failure_moves_to_the_next_token(client: TestClient, transport: MockTransport):
    transport.post_responses.append(
        httpx.Response(
            401,
            json={"errors": {"message": "Authentication failed, please make sure that a valid ModelScope token is supplied."}},
        )
    )
    response = client.post(
        "/v1/images/generations",
        json={"model": "m", "prompt": "p"},
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 200

    submits = [r for r in transport.requests if r.url.path.endswith("/v1/images/generations")]
    assert len(submits) == 2
    assert submits[0].headers["authorization"] == "Bearer token-a"
    assert submits[1].headers["authorization"] == "Bearer token-b"


def test_pool_exhaustion_returns_502(client: TestClient, transport: MockTransport):
    transport.post_responses.extend(
        [
            httpx.Response(401, json={"errors": {"message": "Authentication failed"}}),
            httpx.Response(401, json={"errors": {"message": "Authentication failed"}}),
        ]
    )
    response = client.post(
        "/v1/images/generations",
        json={"model": "m", "prompt": "p"},
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 502
    # The upstream reason must survive the pool being drained.
    assert "Authentication failed" in response.json()["error"]["message"]


def test_quota_error_is_treated_as_a_token_fault(client: TestClient, transport: MockTransport):
    transport.post_responses.append(
        httpx.Response(429, json={"errors": {"message": "quota exceeded for this account"}})
    )
    response = client.post(
        "/v1/images/generations",
        json={"model": "m", "prompt": "p"},
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 200
    submits = [r for r in transport.requests if r.url.path.endswith("/v1/images/generations")]
    assert submits[1].headers["authorization"] == "Bearer token-b"


def test_non_token_error_is_not_retried(client: TestClient, transport: MockTransport):
    transport.post_responses.append(
        httpx.Response(400, json={"errors": {"message": "model not supported"}})
    )
    response = client.post(
        "/v1/images/generations",
        json={"model": "m", "prompt": "p"},
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 502
    submits = [r for r in transport.requests if r.url.path.endswith("/v1/images/generations")]
    assert len(submits) == 1


def test_a_prompt_length_error_does_not_trigger_failover(client: TestClient, transport: MockTransport):
    """The word "tokens" in a prompt-length error is not a token fault."""
    transport.post_responses.append(
        httpx.Response(400, json={"errors": {"message": "prompt is too long, max 2000 tokens"}})
    )
    response = client.post(
        "/v1/images/generations",
        json={"model": "m", "prompt": "p"},
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 502
    submits = [r for r in transport.requests if r.url.path.endswith("/v1/images/generations")]
    assert len(submits) == 1


def test_task_failure_is_reported(client: TestClient, transport: MockTransport):
    transport.get_responses.append(
        httpx.Response(200, json={"task_status": "FAILED", "errors": {"message": "cuda oom"}})
    )
    response = client.post(
        "/v1/images/generations",
        json={"model": "m", "prompt": "p"},
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 502
    assert "cuda oom" in response.json()["error"]["message"]


def test_polling_timeout_is_reported(client: TestClient, transport: MockTransport):
    transport.sticky_task_status = "RUNNING"
    response = client.post(
        "/v1/images/generations",
        json={"model": "m", "prompt": "p"},
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 504


# ----------------------------------------------------------------------
# models
# ----------------------------------------------------------------------
def test_models_endpoint(client: TestClient):
    response = client.get("/v1/models", headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "Qwen/Qwen-Image"
    assert response.json()["object"] == "list"
