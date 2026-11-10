import unittest

import anthropic
import httpx2
import openai

from src.clients import AnthropicClient, BaseLLMClient, OpenAIClient
from src.schemas import LLMError, ModelInputParams

PARAMS = ModelInputParams.from_prompt(model="test-model", prompt="hello")


def _request() -> httpx2.Request:
    return httpx2.Request("POST", "https://api.example.com/v1")


def _response(status: int, headers: dict | None = None) -> httpx2.Response:
    return httpx2.Response(status, headers=headers or {}, request=_request())


def _failing(exc: BaseException, chunks: tuple[str, ...] = ()):
    async def stream(_params):
        for chunk in chunks:
            yield chunk
        raise exc

    async def generate(_params):
        raise exc

    return generate, stream


class ErrorHandlingTests(unittest.IsolatedAsyncioTestCase):
    async def _client(self, provider: str, exc: BaseException, chunks: tuple[str, ...] = ()) -> BaseLLMClient:
        client = (OpenAIClient if provider == "openai" else AnthropicClient)(api_key="test-key")
        client._generate, client._stream = _failing(exc, chunks)
        self.addAsyncCleanup(client.aclose)
        return client

    async def _stream(self, provider: str, exc: BaseException, chunks: tuple[str, ...] = ()) -> list:
        client = await self._client(provider, exc, chunks)
        return [item async for item in client.stream(PARAMS)]

    async def _generate(self, provider: str, exc: BaseException):
        client = await self._client(provider, exc)
        return await client.generate(PARAMS)

    async def test_openai_rate_limit_with_retry_after(self):
        exc = openai.RateLimitError("rate limited", response=_response(429, {"retry-after": "7"}), body=None)
        result = await self._generate("openai", exc)
        self.assertIsInstance(result, LLMError)
        self.assertEqual(result.code, "rate_limit")
        self.assertEqual(result.retry_after, 7)
        self.assertEqual(result.provider, "openai")

    async def test_openai_insufficient_quota(self):
        exc = openai.RateLimitError(
            "quota", response=_response(429), body={"code": "insufficient_quota", "message": "quota"}
        )
        result = await self._generate("openai", exc)
        self.assertEqual(result.code, "quota_exceeded")

    async def test_invalid_api_key(self):
        exc = anthropic.AuthenticationError("invalid x-api-key", response=_response(401), body=None)
        result = await self._generate("anthropic", exc)
        self.assertEqual(result.code, "authentication")
        self.assertEqual(result.status_code, 401)

    async def test_anthropic_rate_limit_in_stream(self):
        exc = anthropic.RateLimitError("rate limited", response=_response(429), body=None)
        items = await self._stream("anthropic", exc)
        self.assertEqual(items[0].code, "rate_limit")
        self.assertIsNone(items[0].retry_after)

    async def test_anthropic_overloaded_is_rate_limit(self):
        exc = anthropic.APIStatusError("overloaded", response=_response(529), body=None)
        result = await self._generate("anthropic", exc)
        self.assertEqual(result.code, "rate_limit")

    async def test_connection_error(self):
        result = await self._generate("openai", openai.APIConnectionError(request=_request()))
        self.assertEqual(result.code, "network")

    async def test_sdk_timeout(self):
        result = await self._generate("anthropic", anthropic.APITimeoutError(request=_request()))
        self.assertEqual(result.code, "timeout")

    async def test_raw_transport_error_mid_stream_keeps_partial_output(self):
        exc = httpx2.ReadError("connection reset", request=_request())
        items = await self._stream("openai", exc, chunks=("Hola", " mundo"))
        self.assertEqual(items[:2], ["Hola", " mundo"])
        self.assertEqual(items[2].code, "network")

    async def test_raw_read_timeout(self):
        items = await self._stream("anthropic", httpx2.ReadTimeout("read timed out", request=_request()))
        self.assertEqual(items[0].code, "timeout")

    async def test_os_connection_error(self):
        items = await self._stream("openai", ConnectionResetError("reset by peer"))
        self.assertEqual(items[0].code, "network")

    async def test_other_status_error_is_controlled(self):
        exc = openai.InternalServerError("boom", response=_response(500), body=None)
        result = await self._generate("openai", exc)
        self.assertEqual(result.code, "api_error")

    async def test_unrelated_bug_still_raises(self):
        with self.assertRaises(KeyError):
            await self._stream("openai", KeyError("bug"))


if __name__ == "__main__":
    unittest.main()
