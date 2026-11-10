"""End-to-end tests of both clients against a fake HTTP server (no network, no API keys)."""

import json
import unittest

import httpx2
import pydantic

from src.clients import AnthropicClient, BaseLLMClient, OpenAIClient, create_client
from src.config import LLMConfig
from src.schemas import ChatMessage, LLMError, ModelInputParams, ModelResponse

PARAMS = ModelInputParams.from_prompt(model="test-model", prompt="¿Qué es la entropía?", system_prompt="Be brief.")


def _sse(events: list[tuple[str | None, dict | str]]) -> bytes:
    lines = []
    for event, data in events:
        if event:
            lines.append(f"event: {event}")
        lines.append(f"data: {data if isinstance(data, str) else json.dumps(data)}")
        lines.append("")
    return ("\n".join(lines) + "\n").encode()


def _openai_handler(request: httpx2.Request) -> httpx2.Response:
    body = json.loads(request.content)
    _openai_handler.last_body = body
    if body.get("stream"):
        chunks = [
            {"id": "1", "object": "chat.completion.chunk", "created": 0, "model": "test-model",
             "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}]}
            for token in ("La ", "entropía")
        ]
        events = [(None, chunk) for chunk in chunks] + [(None, "[DONE]")]
        return httpx2.Response(200, content=_sse(events), headers={"content-type": "text/event-stream"})
    return httpx2.Response(200, json={
        "id": "1", "object": "chat.completion", "created": 0, "model": "test-model",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "La entropía mide el desorden."}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
    })


def _anthropic_handler(request: httpx2.Request) -> httpx2.Response:
    body = json.loads(request.content)
    _anthropic_handler.last_body = body
    message = {"id": "msg_1", "type": "message", "role": "assistant", "model": "test-model",
               "content": [], "stop_reason": None, "stop_sequence": None,
               "usage": {"input_tokens": 10, "output_tokens": 0}}
    if body.get("stream"):
        events = [("message_start", {"type": "message_start", "message": message}),
                  ("content_block_start", {"type": "content_block_start", "index": 0,
                                           "content_block": {"type": "text", "text": ""}})]
        events += [("content_block_delta", {"type": "content_block_delta", "index": 0,
                                             "delta": {"type": "text_delta", "text": token}})
                   for token in ("La ", "entropía")]
        events += [("content_block_stop", {"type": "content_block_stop", "index": 0}),
                   ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                                      "usage": {"output_tokens": 2}}),
                   ("message_stop", {"type": "message_stop"})]
        return httpx2.Response(200, content=_sse(events), headers={"content-type": "text/event-stream"})
    message.update(content=[{"type": "text", "text": "La entropía mide el desorden."}],
                   stop_reason="end_turn", usage={"input_tokens": 10, "output_tokens": 6})
    return httpx2.Response(200, json=message)


def _status_handler(status: int, body: dict):
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json=body)
    return handler


def _make(provider: str, handler) -> BaseLLMClient:
    client = (OpenAIClient if provider == "openai" else AnthropicClient)(api_key="test-key", max_retries=0)
    client.client = client.client.with_options(http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    return client


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def _check_generate_and_stream(self, provider: str, handler) -> None:
        async with _make(provider, handler) as client:
            response = await client.generate(PARAMS)
            self.assertIsInstance(response, ModelResponse)
            self.assertEqual(response.content, "La entropía mide el desorden.")
            self.assertEqual(response.provider, provider)
            self.assertEqual(response.usage.output_tokens, 6)

            tokens = [token async for token in client.stream(PARAMS)]
            self.assertEqual(tokens, ["La ", "entropía"])

    async def test_openai_generate_and_stream(self):
        await self._check_generate_and_stream("openai", _openai_handler)
        self.assertEqual(_openai_handler.last_body["messages"][0], {"role": "system", "content": "Be brief."})

    async def test_anthropic_generate_and_stream(self):
        await self._check_generate_and_stream("anthropic", _anthropic_handler)
        body = _anthropic_handler.last_body
        self.assertEqual(body["system"], "Be brief.")
        self.assertEqual([m["role"] for m in body["messages"]], ["user"])

    async def test_invalid_api_key_over_http(self):
        handler = _status_handler(401, {"type": "error", "error": {"type": "authentication_error",
                                                                   "message": "invalid x-api-key"}})
        async with _make("anthropic", handler) as client:
            result = await client.generate(PARAMS)
        self.assertIsInstance(result, LLMError)
        self.assertEqual(result.code, "authentication")

    async def test_rate_limit_over_http_in_stream(self):
        handler = _status_handler(429, {"error": {"message": "slow down", "type": "requests", "code": "rate_limit"}})
        async with _make("openai", handler) as client:
            items = [item async for item in client.stream(PARAMS)]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].code, "rate_limit")

    async def test_dict_input_is_validated(self):
        async with _make("openai", _openai_handler) as client:
            response = await client.generate({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
            self.assertIsInstance(response, ModelResponse)
            with self.assertRaises(pydantic.ValidationError):
                await client.generate({"model": "m", "messages": [{"role": "robot", "content": "hi"}]})


class FactoryAndSchemaTests(unittest.TestCase):
    def test_factory_returns_common_interface(self):
        config = LLMConfig(openai_api_key="sk-test", anthropic_api_key="sk-ant-test")
        self.assertIsInstance(create_client(config, "openai"), OpenAIClient)
        self.assertIsInstance(create_client(config, "anthropic"), AnthropicClient)

    def test_factory_missing_key(self):
        with self.assertRaises(ValueError):
            create_client(LLMConfig(provider="anthropic"))

    def test_base_client_is_abstract(self):
        with self.assertRaises(TypeError):
            BaseLLMClient()

    def test_chat_message_validation(self):
        with self.assertRaises(pydantic.ValidationError):
            ChatMessage(role="user", content="   ")

    def test_params_need_a_non_system_message(self):
        with self.assertRaises(pydantic.ValidationError):
            ModelInputParams(model="m", messages=[ChatMessage(role="system", content="x")])


if __name__ == "__main__":
    unittest.main()
