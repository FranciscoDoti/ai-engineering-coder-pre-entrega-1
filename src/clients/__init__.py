from ..config import LLMConfig, Provider
from .anthropic_client import AnthropicClient
from .base import BaseLLMClient, StreamItem
from .openai_client import OpenAIClient

_CLIENTS: dict[str, type[BaseLLMClient]] = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
}


def create_client(config: LLMConfig | None = None, provider: Provider | None = None) -> BaseLLMClient:
    """Build the client for `provider` (or the configured one) behind the common interface."""
    config = config or LLMConfig.from_env()
    provider = provider or config.provider
    client_cls = _CLIENTS[provider]
    return client_cls(config.api_key_for(provider), timeout=config.timeout, max_retries=config.max_retries)


__all__ = ["AnthropicClient", "BaseLLMClient", "OpenAIClient", "StreamItem", "create_client"]
