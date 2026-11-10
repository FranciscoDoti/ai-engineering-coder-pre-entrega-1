from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from ..schemas import ModelInputParams, ModelResponse, TokenUsage
from .base import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    provider = "openai"

    def __init__(self, api_key: str, *, timeout: float = 60.0, max_retries: int = 2) -> None:
        self.client = AsyncOpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)

    def _request(self, params: ModelInputParams) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": params.model,
            "messages": [message.model_dump() for message in params.messages],
            "temperature": params.temperature,
            "max_tokens": params.max_tokens,
            "top_p": params.top_p,
            "frequency_penalty": params.frequency_penalty,
            "presence_penalty": params.presence_penalty,
        }
        if params.stop is not None:
            request["stop"] = params.stop
        if params.seed is not None:
            request["seed"] = params.seed
        return request

    async def _generate(self, params: ModelInputParams) -> ModelResponse:
        response = await self.client.chat.completions.create(**self._request(params))
        choice = response.choices[0]
        usage = response.usage
        return ModelResponse(
            content=choice.message.content or "",
            model=response.model,
            provider=self.provider,
            finish_reason=choice.finish_reason,
            usage=TokenUsage(input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens) if usage else None,
        )

    async def _stream(self, params: ModelInputParams) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(**self._request(params), stream=True)
        async for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta.content
            if delta:
                yield delta

    async def aclose(self) -> None:
        await self.client.close()
