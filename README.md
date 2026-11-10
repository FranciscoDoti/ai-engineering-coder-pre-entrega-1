# Unified Async LLM Client

Python 3.12 async client for **OpenAI** and **Anthropic** behind one common interface:

- **Interchangeable** — `OpenAIClient` and `AnthropicClient` both extend the abstract
  `BaseLLMClient`; `create_client()` returns either one from configuration.
- **Async** — built on the official async SDKs (`AsyncOpenAI`, `AsyncAnthropic`); every call is
  `await`ed, so the event loop is never blocked.
- **Streaming** — `stream()` is an async generator that yields tokens as they arrive.
- **Validated** — Pydantic models for messages (`ChatMessage`), request settings
  (`ModelInputParams`), answers (`ModelResponse`) and errors (`LLMError`).
- **No exception leaks** — invalid API key, quota, rate limit, timeout and network failures are
  returned as an `LLMError` instead of crashing the caller. Transient failures are retried by
  the SDK with backoff first (`LLM_MAX_RETRIES`).

## Usage

```python
from src.clients import create_client
from src.schemas import ChatMessage, LLMError, ModelInputParams

params = ModelInputParams(
    model="gpt-4o-mini",
    messages=[ChatMessage(role="user", content="¿Qué es la entropía?")],
)

async with create_client(provider="openai") as client:
    result = await client.generate(params)          # ModelResponse | LLMError
    async for token in client.stream(params):       # str tokens, or a final LLMError
        ...
```

## Requirements

- Python 3.12+
- An API key for at least one provider (OpenAI or Anthropic)

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Environment variables

Copy `.env.example` to `.env` in the project root and fill it in. The app reads `.env` from the
project root and from `src/`; variables already set in the shell take precedence.

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | At least one of the two keys | OpenAI API key. |
| `ANTHROPIC_API_KEY` | At least one of the two keys | Anthropic API key. |
| `LLM_PROVIDER` | No (default `openai`) | Provider `create_client()` uses when none is passed. |
| `OPENAI_MODEL` | No (default `gpt-4o-mini`) | Model used by `main.py` for OpenAI. |
| `ANTHROPIC_MODEL` | No (default `claude-sonnet-5-5`) | Model used by `main.py` for Anthropic. |
| `LLM_TIMEOUT` | No (default `60`) | Per-request timeout, in seconds. |
| `LLM_MAX_RETRIES` | No (default `2`) | SDK retries with backoff before returning an error. |

## Running the validation script

`main.py` loads `.env`, then asks *"¿Qué es la entropía?"* to **every provider that has an API
key**, first in normal mode (full answer at once) and then in streaming mode (token by token):

```bash
python main.py
```

Expected output (shortened):

```
Question: ¿Qué es la entropía?

[openai] normal mode
La entropía es una medida del desorden o la aleatoriedad en un sistema...
(tokens: 30 in / 67 out)

[openai] streaming mode
La entropía es una medida del desorden...
```

If a call fails, the script prints a controlled error and keeps going:

```
[openai] normal mode
Error (authentication): The API key is invalid or lacks permission. Check your .env file.
```

## Running the unit tests

The tests use a fake HTTP server, so they need **no API keys and no internet connection**:

```bash
python -m unittest discover -s tests -t . -v
```

## Controlled error codes

`generate()` returns a `ModelResponse` or an `LLMError`. `stream()` yields text tokens and, if
something fails, one final `LLMError` (tokens already received are kept).

| `code` | When |
|---|---|
| `authentication` | Invalid API key or missing permission (HTTP 401 / 403). |
| `quota_exceeded` | The account has no quota/credit left. |
| `rate_limit` | HTTP 429, or the provider is overloaded (HTTP 503 / 529). `retry_after` holds the suggested wait in seconds, when the provider sends it. |
| `timeout` | The request or the stream took too long. |
| `network` | Connection could not be established or dropped mid-stream. |
| `api_error` | Any other HTTP error returned by the provider (e.g. 400, 500). |

Programming errors (bugs) and invalid input (`pydantic.ValidationError`) are still raised on
purpose, so they are not hidden.

## Notes

- `temperature`, `top_p`, `frequency_penalty`, `presence_penalty` and `seed` are only sent to
  OpenAI: the installed Anthropic SDK no longer accepts sampling parameters.
- System messages are sent as OpenAI `system` messages and as Anthropic's separate `system` field.

## Project structure

```
main.py                         # Validation script (normal + streaming)
src/schemas.py                  # ChatMessage, ModelInputParams, ModelResponse, LLMError
src/config.py                   # Loads keys, models, timeout and retries from .env
src/errors.py                   # Maps SDK/network exceptions to LLMError
src/clients/base.py             # BaseLLMClient (abstract): generate() and stream()
src/clients/openai_client.py    # OpenAIClient (AsyncOpenAI)
src/clients/anthropic_client.py # AnthropicClient (AsyncAnthropic)
src/clients/__init__.py         # create_client() factory
tests/                          # Unit tests (no network needed)
```
