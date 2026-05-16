#!/usr/bin/env python3
"""Central provider, model, and language configuration."""
from typing import Final, Literal, Optional


LLMProvider = Literal["openai", "anthropic", "gemini"]

DEFAULT_PROVIDER: Final[LLMProvider] = "openai"

SUPPORTED_PROVIDERS: Final[tuple[LLMProvider, ...]] = (
    "openai",
    "anthropic",
    "gemini",
)

VALID_LANGUAGES: Final[tuple[str, ...]] = (
    "mingrelian",
    "georgian",
    "english",
)

DEFAULT_SOURCE_LANGUAGE: Final[str] = "mingrelian"
DEFAULT_TARGET_LANGUAGE: Final[str] = "english"

DEFAULT_MODEL_BY_PROVIDER: Final[dict[LLMProvider, str]] = {
    "openai": "gpt-5.5",
    "anthropic": "claude-sonnet-4-5-20250929",
    "gemini": "gemini-3.1-flash-lite-preview",
}

SERVER_KEY_MODELS: Final[dict[LLMProvider, frozenset[str]]] = {
    "openai": frozenset({"gpt-5.5", "gpt-5.4-nano"}),
    "gemini": frozenset({"gemini-3.1-flash-lite-preview"}),
}

DEFAULT_REASONING_EFFORT_BY_MODEL: Final[dict[str, str]] = {
    "gpt-5.5": "none",
}

PROVIDER_API_KEY_ENV: Final[dict[LLMProvider, str]] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def get_default_model_for_provider(provider: str) -> Optional[str]:
    """Resolve the provider's runtime default model."""
    return DEFAULT_MODEL_BY_PROVIDER.get(provider)


def get_default_reasoning_effort_for_model(model: Optional[str]) -> Optional[str]:
    """Resolve the runtime default reasoning effort for a model."""
    if model is None:
        return None
    return DEFAULT_REASONING_EFFORT_BY_MODEL.get(model)


def get_api_key_env_var(provider: str) -> Optional[str]:
    """Return the environment variable used for a provider API key."""
    return PROVIDER_API_KEY_ENV.get(provider)


def is_server_key_model_allowed(provider: str, model: str) -> bool:
    """Check whether this provider/model pair may use the server-side key."""
    return model in SERVER_KEY_MODELS.get(provider, frozenset())
