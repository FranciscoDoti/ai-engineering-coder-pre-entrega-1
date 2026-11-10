from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from anthropic import AsyncAnthropic

from ..schemas import ModelInputParams, ModelResponse, TokenUsage
from .base import BaseLLMClient


class AnthropicClient(BaseLLMClient):
    provider = "anthropic"

    def __init__(self, api_key: str, *, timeout: float = 60.0, max_retries: int = 2) -> None:
        self.client = AsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=max_retries)

    def _request(self, params: ModelInputParams) -> dict[str, Any]:
        # Anthropic takes system instructions as a separate field, not as a message.
        # The current SDK/models no longer accept temperature or top_p, so those
        # settings only apply to OpenAI.
        system = "\n\n".join(m.content for m in params.messages if m.role == "system")
        request: dict[str, Any] = {
            "model": params.model,
            "max_tokens": params.max_tokens,
            "messages": [m.model_dump() for m in params.messages if m.role != "system"],
        }
        if system:
            request["system"] = system
        if params.stop is not None:
            request["stop_sequences"] = params.stop
        return request

    async def _generate(self, params: ModelInputParams) -> ModelResponse:
        response = await self.client.messages.create(**self._request(params))
        return ModelResponse(
            content="".join(block.text for block in response.content if getattr(block, "text", None)),
            model=response.model,
            provider=self.provider,
            finish_reason=response.stop_reason,
            usage=TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
        )

    async def _stream(self, params: ModelInputParams) -> AsyncIterator[str]:
        async with self.client.messages.stream(**self._request(params)) as stream:
            async for text in stream.text_stream:
                yield text

    async def aclose(self) -> None:
        await self.client.close()
