from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import ClassVar, Union

from ..errors import HANDLED_ERRORS, to_llm_error
from ..schemas import LLMError, ModelInputParams, ModelResponse

StreamItem = Union[str, LLMError]


class BaseLLMClient(ABC):
    """Common async interface for every provider.

    Subclasses implement only the provider-specific calls (`_generate`, `_stream`).
    The public methods validate the input and turn network, rate-limit and auth
    failures into an `LLMError`, so a provider error never crashes the caller.
    """

    provider: ClassVar[str]

    async def generate(self, params: ModelInputParams | dict) -> ModelResponse | LLMError:
        """Return the full answer at once."""
        params = _validate(params)
        try:
            return await self._generate(params)
        except HANDLED_ERRORS as exc:
            return to_llm_error(exc, self.provider)

    async def stream(self, params: ModelInputParams | dict) -> AsyncIterator[StreamItem]:
        """Yield the answer token by token; on failure yield one final LLMError."""
        params = _validate(params)
        try:
            async for token in self._stream(params):
                yield token
        except HANDLED_ERRORS as exc:
            # Tokens already yielded are kept; the error is the last item.
            yield to_llm_error(exc, self.provider)

    @abstractmethod
    async def _generate(self, params: ModelInputParams) -> ModelResponse: ...

    @abstractmethod
    def _stream(self, params: ModelInputParams) -> AsyncIterator[str]: ...

    @abstractmethod
    async def aclose(self) -> None: ...

    async def __aenter__(self) -> "BaseLLMClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()


def _validate(params: ModelInputParams | dict) -> ModelInputParams:
    if isinstance(params, ModelInputParams):
        return params
    return ModelInputParams.model_validate(params)
