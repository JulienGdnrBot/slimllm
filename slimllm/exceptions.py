"""
Exception hierarchy — mirrors LiteLLM / OpenAI SDK names so callers
can swap libraries without changing error handling code.
"""
from __future__ import annotations


class SlimLLMError(Exception):
    """Base exception for all slimllm errors."""

    def __init__(self, message: str, status_code: int = 0, provider: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.provider = provider
        self.message = message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(status={self.status_code}, provider={self.provider!r}, message={self.message!r})"


class AuthenticationError(SlimLLMError):
    """401 / missing API key."""


class PermissionDeniedError(SlimLLMError):
    """403."""


class NotFoundError(SlimLLMError):
    """404 — wrong model name or endpoint."""


class RateLimitError(SlimLLMError):
    """429."""


class InternalServerError(SlimLLMError):
    """5xx from the provider."""


class BadRequestError(SlimLLMError):
    """400 — invalid request payload."""


class ProviderError(SlimLLMError):
    """Unexpected non-HTTP error from the provider (e.g. malformed SSE)."""


class UnsupportedProviderError(SlimLLMError):
    """Model string doesn't map to any known provider."""


def raise_for_status(status_code: int, body: str, provider: str) -> None:
    """Convert HTTP status → typed exception."""
    if status_code == 400:
        raise BadRequestError(body, status_code, provider)
    if status_code == 401:
        raise AuthenticationError(body, status_code, provider)
    if status_code == 403:
        raise PermissionDeniedError(body, status_code, provider)
    if status_code == 404:
        raise NotFoundError(body, status_code, provider)
    if status_code == 429:
        raise RateLimitError(body, status_code, provider)
    if status_code >= 500:
        raise InternalServerError(body, status_code, provider)
    if status_code >= 400:
        raise SlimLLMError(body, status_code, provider)
