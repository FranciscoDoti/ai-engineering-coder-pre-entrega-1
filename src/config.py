import os
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent

Provider = Literal["openai", "anthropic"]

_FALLBACK_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5-5",
}


def load_env() -> None:
    """Load .env from the project root and from src/ (shell variables win)."""
    load_dotenv(_PROJECT_ROOT / ".env")
    load_dotenv(_SRC_DIR / ".env")


class LLMConfig(BaseModel):
    """Provider settings, loaded from environment variables."""

    provider: Provider = Field(default="openai", description="Provider to use: openai or anthropic.")
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    timeout: float = Field(default=60.0, gt=0, description="Per-request timeout in seconds.")
    max_retries: int = Field(
        default=2,
        ge=0,
        le=10,
        description="SDK retries (with backoff) for rate limits, timeouts and network errors.",
    )

    @field_validator("provider", mode="before")
    @classmethod
    def normalize_provider(cls, value: str) -> str:
        return str(value).strip().lower()

    @classmethod
    def from_env(cls) -> "LLMConfig":
        load_env()
        return cls(
            provider=os.getenv("LLM_PROVIDER", "openai"),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
            timeout=os.getenv("LLM_TIMEOUT") or 60.0,
            max_retries=os.getenv("LLM_MAX_RETRIES") or 2,
        )

    def api_key_for(self, provider: Provider | None = None) -> str:
        provider = provider or self.provider
        key = self.openai_api_key if provider == "openai" else self.anthropic_api_key
        if not key:
            raise ValueError(f"Missing {provider.upper()}_API_KEY to use the {provider} provider.")
        return key

    def model_for(self, provider: Provider | None = None) -> str:
        """Model from OPENAI_MODEL / ANTHROPIC_MODEL, or a sensible default."""
        provider = provider or self.provider
        return os.getenv(f"{provider.upper()}_MODEL") or _FALLBACK_MODELS[provider]

    def available_providers(self) -> list[Provider]:
        return [p for p in ("openai", "anthropic") if (self.openai_api_key if p == "openai" else self.anthropic_api_key)]
