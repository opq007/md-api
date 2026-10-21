"""Error types translated into OpenAI-shaped error payloads."""

from __future__ import annotations

from typing import Any


class ProxyError(Exception):
    """Base class for errors that map onto an HTTP status + OpenAI error body."""

    status_code = 500
    error_type = "server_error"
    code: str | None = None

    def __init__(self, message: str, *, code: str | None = None, param: str | None = None):
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        self.param = param

    def to_payload(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


class ConfigurationError(ProxyError):
    """The proxy itself is misconfigured (e.g. no tokens configured)."""

    status_code = 500
    error_type = "server_error"
    code = "configuration_error"


class AuthenticationError(ProxyError):
    """The client presented no/invalid credentials to the proxy."""

    status_code = 401
    error_type = "invalid_request_error"
    code = "invalid_api_key"


class InvalidRequestError(ProxyError):
    """The client's request body is malformed for the OpenAI image API."""

    status_code = 400
    error_type = "invalid_request_error"
    code = "invalid_request"


class UpstreamError(ProxyError):
    """ModelScope returned an error that is not attributable to the client."""

    status_code = 502
    error_type = "api_error"
    code = "upstream_error"


class UpstreamTimeoutError(ProxyError):
    """The blocking wait budget elapsed before the task finished."""

    status_code = 504
    error_type = "api_error"
    code = "upstream_timeout"


class TaskFailedError(ProxyError):
    """The ModelScope task reached FAILED state."""

    status_code = 502
    error_type = "api_error"
    code = "task_failed"
