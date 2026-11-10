from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    """A single message in a conversation."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    role: Role
    content: str = Field(..., min_length=1)


class ModelInputParams(BaseModel):
    """Input for a model call: the conversation plus the generation settings."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    model: str = Field(..., min_length=1, description="Model identifier.")
    messages: list[ChatMessage] = Field(..., min_length=1, description="Conversation, oldest first.")
    temperature: float = Field(
        default=0.7,
        ge=0,
        le=2,
        description="Sampling randomness. 0 is deterministic. OpenAI only.",
    )
    max_tokens: int = Field(default=256, gt=0, le=8192, description="Maximum tokens to generate.")
    top_p: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Nucleus sampling. 1 disables it. OpenAI only.",
    )
    frequency_penalty: float = Field(default=0.0, ge=-2, le=2, description="OpenAI only.")
    presence_penalty: float = Field(default=0.0, ge=-2, le=2, description="OpenAI only.")
    stop: Optional[list[str]] = Field(default=None, description="Sequences that stop generation.")
    seed: Optional[int] = Field(default=None, ge=0, description="OpenAI only, for reproducibility.")

    @classmethod
    def from_prompt(cls, model: str, prompt: str, system_prompt: str | None = None, **settings) -> "ModelInputParams":
        """Shortcut for the common single-question case."""
        messages = [ChatMessage(role="system", content=system_prompt)] if system_prompt else []
        messages.append(ChatMessage(role="user", content=prompt))
        return cls(model=model, messages=messages, **settings)

    @field_validator("stop")
    @classmethod
    def validate_stop_sequences(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        if value is None:
            return value
        if len(value) == 0:
            raise ValueError("stop cannot be an empty list.")
        if len(value) > 4:
            raise ValueError("stop accepts at most 4 sequences.")
        if any(not sequence.strip() for sequence in value):
            raise ValueError("stop sequences cannot be blank.")
        return value

    @model_validator(mode="after")
    def require_conversation_turn(self) -> "ModelInputParams":
        if all(message.role == "system" for message in self.messages):
            raise ValueError("messages needs at least one user or assistant message.")
        return self


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class ModelResponse(BaseModel):
    """Normalized answer from any provider."""

    content: str
    model: str
    provider: str
    finish_reason: Optional[str] = None
    usage: Optional[TokenUsage] = None


ErrorCode = Literal["rate_limit", "quota_exceeded", "authentication", "network", "timeout", "api_error"]


class LLMError(BaseModel):
    """Controlled error returned instead of raising (network, rate limit, auth, ...)."""

    code: ErrorCode
    message: str
    provider: str
    status_code: Optional[int] = None
    retry_after: Optional[float] = Field(
        default=None,
        description="Seconds the provider asked to wait before retrying, if known.",
    )
