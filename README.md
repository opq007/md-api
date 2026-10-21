# md-api

An OpenAI-compatible image API that runs on top of the [ModelScope
API-Inference](https://www.modelscope.cn/docs/model-service/API-Inference/intro)
service.

ModelScope exposes a single, asynchronous, JSON-only image endpoint. This
proxy reshapes it into the two endpoints the OpenAI Images API defines, so
stock OpenAI clients (the official SDK, ComfyUI, LobeChat, …) work against
ModelScope without modification.

| Capability | ModelScope native | This proxy |
| --- | --- | --- |
| Text to image | ✅ | `POST /v1/images/generations` |
| Image editing | ✅ (same endpoint) | `POST /v1/images/edits` (multipart) |
| Multi-image editing | ✅ (list of `image_url`) | `POST /v1/images/edits` with several files |
| LoRA configuration | `loras` field | `extra_body={"loras": ...}` |
| Sync / async | async only | blocking, transparent polling |
| Multiple accounts | one token | rotating token pool |

---

## Quick start

```bash
pip install -r requirements.txt

export MODELSCOPE_TOKENS="ms-token-one,ms-token-two"
export PROXY_API_KEY="your-proxy-key"

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Or with Docker:

```bash
export MODELSCOPE_TOKENS="ms-token-one,ms-token-two"
export PROXY_API_KEY="your-proxy-key"
docker compose up --build
```

Get your ModelScope access tokens at
<https://modelscope.cn/my/myaccesstoken>. Note that a ModelScope account must
be linked to an Aliyun account and pass real-name verification before
API-Inference can be used.

---

## Usage

Point the OpenAI SDK at the proxy. Only `base_url` and `api_key` change.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="your-proxy-key",
)
```

### Text to image

```python
result = client.images.generate(
    model="Qwen/Qwen-Image",
    prompt="a golden cat in the snow",
    size="1024x1024",
)
print(result.data[0].url)
```

### Image editing

```python
with open("input.png", "rb") as fh:
    result = client.images.edit(
        model="Qwen/Qwen-Image-Edit",
        image=fh,
        prompt="turn the girl's hair blue",
    )
print(result.data[0].url)
```

### Multi-image editing

Pass a list of images; they are forwarded to ModelScope as its multi-image
editing input.

```python
with open("person.png", "rb") as a, open("clothes.png", "rb") as b:
    result = client.images.edit(
        model="Qwen/Qwen-Image-Edit",
        image=[a, b],
        prompt="dress the person in the clothes",
    )
```

### LoRA

LoRA configuration travels in `extra_body`. The OpenAI SDK flattens it into a
top-level `loras` field on the wire, which the proxy reads directly.

```python
result = client.images.generate(
    model="Qwen/Qwen-Image",
    prompt="a golden cat",
    extra_body={"loras": {"owner/lora-a": 0.6, "owner/lora-b": 0.4}},
)
```

A single LoRA may be given as a plain string:

```python
extra_body={"loras": "owner/lora-a"}
```

Rules inherited from ModelScope: at most 6 LoRAs, and the weights must sum to
`1.0`. Violations are rejected with a `400` before anything is sent upstream.

### Binary output

```python
result = client.images.generate(
    model="Qwen/Qwen-Image",
    prompt="a golden cat",
    response_format="b64_json",
)
```

`response_format="url"` (the default) returns ModelScope's own CDN URL
unchanged — the proxy does not copy or re-host the image.

### Listing models

```python
models = client.models.list()
```

---

## Endpoints

### `POST /v1/images/generations`

OpenAI-compatible. JSON body.

| Field | Type | Notes |
| --- | --- | --- |
| `model` | string | **required**, a ModelScope model id |
| `prompt` | string | **required**, under 2000 characters |
| `size` | string | e.g. `1024x1024` |
| `response_format` | `url` \| `b64_json` | default `url` |
| `negative_prompt` | string | ModelScope native, under 4000 characters |
| `seed` | int | ModelScope native |
| `steps` | int | ModelScope native |
| `guidance` | float | ModelScope native |
| `image_url` | string \| string[] | ModelScope native; a list enables image editing |
| `loras` | string \| object | ModelScope native, see above |

`n`, `quality`, `style` and `user` are accepted and ignored — ModelScope has
no equivalent. `n` is *not* honoured: one upstream task yields one image, so
the response always contains a single entry in `data`.

### `POST /v1/images/edits`

OpenAI-compatible. `multipart/form-data`.

The OpenAI SDK names the file part `image` for a single image and `image[]`
for a list; both are accepted. Repeat `image[]` to send several images.

| Field | Type | Notes |
| --- | --- | --- |
| `model` | string | **required** |
| `prompt` | string | **required** |
| `image` / `image[]` | file | **required**, one or more |
| `size` | string | |
| `response_format` | `url` \| `b64_json` | |
| `negative_prompt`, `seed`, `steps`, `guidance` | | ModelScope native |
| `loras[owner/lora]` | number | LoRA weights; the weights must sum to 1.0 |

Uploaded files are encoded as base64 `data:` URLs before being sent upstream.
`mask` is not supported — ModelScope has no equivalent.

### `GET /v1/models`

Proxies ModelScope's model list.

### `GET /healthz`

Liveness probe plus the health of each token in the pool.

---

## The token pool

`MODELSCOPE_TOKENS` holds a comma separated list of ModelScope access tokens.
The pool hands them out round-robin. When a submission is rejected for an
auth, quota or rate-limit reason, that token is marked unhealthy and the
submission is retried with the next token, up to the size of the pool. A
healthy token is returned to rotation after a short quarantine.

The token that successfully submits a task is the one used to poll it, so a
task is never abandoned mid-flight. A token fault *while polling* is reported
to the client rather than silently retried, because the task itself lives on
ModelScope's side.

`GET /healthz` shows which tokens are currently usable.

---

## Errors

Failures use the OpenAI error envelope:

```json
{
  "error": {
    "message": "ModelScope rejected the request: ...",
    "type": "api_error",
    "param": null,
    "code": "upstream_error"
  }
}
```

| Status | When |
| --- | --- |
| `400` | the request body is invalid (bad LoRA weights, empty image, …) |
| `401` | the client did not present the proxy's API key |
| `502` | ModelScope rejected the request or the task failed |
| `504` | the blocking wait budget elapsed before the task finished |

---

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODELSCOPE_TOKENS` | — | comma separated ModelScope access tokens |
| `PROXY_API_KEY` | — | key clients must present; empty disables auth |
| `MODELSCOPE_BASE_URL` | `https://api-inference.modelscope.cn` | upstream base URL |
| `MODELSCOPE_TASK_TYPE` | `image_generation` | task type header when polling |
| `PROXY_REQUEST_TIMEOUT` | `300` | total seconds a request may block |
| `PROXY_POLL_INITIAL_INTERVAL` | `2` | first poll delay, in seconds |
| `PROXY_POLL_MAX_INTERVAL` | `10` | maximum poll delay, in seconds |
| `PROXY_POLL_BACKOFF` | `1.5` | poll delay multiplier |
| `PROXY_CONNECT_TIMEOUT` | `15` | upstream connect timeout, in seconds |
| `PROXY_UPSTREAM_TIMEOUT` | `60` | upstream read timeout, in seconds |
| `PROXY_MAX_TOKEN_ATTEMPTS` | `0` (all tokens) | tokens to try per submission |

---

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

The suite includes an end-to-end test that drives the proxy with the real
`openai` Python SDK against a live local server, asserting on the exact HTTP
traffic — including the multipart `image[]` parts and the `loras[...]` form
fields the SDK produces.

---

## Notes and limitations

- ModelScope's free API-Inference is rate limited and metered in 魔粒
  (magic credits). A multi-token pool raises the ceiling but does not remove
  it.
- The models available depend on what ModelScope currently exposes; the list
  changes as models are added and retired.
- `n > 1` is not implemented — one task yields one image.
- `mask` (inpainting) is not supported.
- Streaming is not supported; the proxy blocks until the image is ready.
