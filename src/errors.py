"""Maps SDK and transport exceptions to a controlled LLMError."""

from __future__ import annotations

import anthropic
import httpx2
import openai

from .schemas import LLMError

# 503 (service unavailable) and 529 (Anthropic "overloaded") are capacity limits:
# reported like rate limiting so the caller knows waiting and retrying may help.
_OVERLOADED_STATUS = {503, 529}

HANDLED_ERRORS: tuple[type[BaseException], ...] = (
    openai.APIError,
    anthropic.APIError,
    # Raw transport failures can escape the SDK while a stream is being consumed.
    httpx2.TransportError,
    TimeoutError,
    ConnectionError,
)


def retry_after_seconds(exc: BaseException) -> float | None:
    """Read the Retry-After header (seconds) from a status error, if present."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    for name in ("retry-after-ms", "retry-after"):
        value = headers.get(name)
        if value is None:
            continue
        try:
            seconds = float(value)
        except ValueError:
            continue
        if name == "retry-after-ms":
            seconds /= 1000
        if seconds >= 0:
            return seconds
    return None


def to_llm_error(exc: BaseException, provider: str) -> LLMError:
    status = getattr(exc, "status_code", None)

    def error(code, message, retry_after=None) -> LLMError:
        return LLMError(code=code, message=message, provider=provider, status_code=status, retry_after=retry_after)

    if isinstance(exc, (openai.AuthenticationError, anthropic.AuthenticationError)) or status in (401, 403):
        return error("authentication", "The API key is invalid or lacks permission. Check your .env file.")

    if status == 429 or isinstance(exc, (openai.RateLimitError, anthropic.RateLimitError)):
        # OpenAI uses 429 both for rate limits and for an exhausted billing quota;
        # only the former goes away by waiting.
        if getattr(exc, "code", None) == "insufficient_quota":
            return error("quota_exceeded", "The account has no quota left. Check the provider's billing settings.")
        wait = retry_after_seconds(exc)
        hint = f" Retry in {wait:g} s." if wait is not None else " Wait a moment and try again."
        return error("rate_limit", "The provider's rate limit was reached." + hint, wait)

    if status in _OVERLOADED_STATUS:
        return error(
            "rate_limit",
            "The provider is overloaded right now. Wait a moment and try again.",
            retry_after_seconds(exc),
        )

    if status is not None:
        return error("api_error", f"The provider returned an error (HTTP {status}).")

    if isinstance(
        exc,
        (openai.APITimeoutError, anthropic.APITimeoutError, httpx2.TimeoutException, TimeoutError),
    ):
        return error("timeout", "The request to the provider took too long and was cancelled.")

    if isinstance(
        exc,
        (openai.APIConnectionError, anthropic.APIConnectionError, httpx2.TransportError, ConnectionError),
    ):
        return error("network", "Could not reach the provider. Check your connection and try again.")

    return error("api_error", f"Unexpected provider error: {exc}")
