"""Validation script: asks one short question to every configured provider,
first in normal mode and then in streaming mode."""

import asyncio
import sys

from src.clients import BaseLLMClient, create_client
from src.config import LLMConfig
from src.schemas import LLMError, ModelInputParams

QUESTION = "¿Qué es la entropía?"


def build_params(model: str) -> ModelInputParams:
    return ModelInputParams.from_prompt(
        model=model,
        prompt=QUESTION,
        system_prompt="Answer in two or three sentences, in the user's language.",
        temperature=0.3,
        max_tokens=300,
    )


async def run_normal(client: BaseLLMClient, params: ModelInputParams) -> None:
    print(f"\n[{client.provider}] normal mode")
    result = await client.generate(params)
    if isinstance(result, LLMError):
        print(f"Error ({result.code}): {result.message}")
        return
    print(result.content)
    if result.usage:
        print(f"(tokens: {result.usage.input_tokens} in / {result.usage.output_tokens} out)")


async def run_streaming(client: BaseLLMClient, params: ModelInputParams) -> None:
    print(f"\n[{client.provider}] streaming mode")
    async for token in client.stream(params):
        if isinstance(token, LLMError):
            print(f"\nError ({token.code}): {token.message}")
            break
        print(token, end="", flush=True)
    print()


async def main() -> None:
    config = LLMConfig.from_env()  # loads .env
    providers = config.available_providers()
    if not providers:
        print("Configuration error: set OPENAI_API_KEY and/or ANTHROPIC_API_KEY in .env.")
        return

    print(f"Question: {QUESTION}")
    for provider in providers:
        async with create_client(config, provider) as client:
            params = build_params(config.model_for(provider))
            await run_normal(client, params)
            await run_streaming(client, params)


if __name__ == "__main__":
    # Avoid garbled accents on Windows terminals that default to a legacy code page.
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
