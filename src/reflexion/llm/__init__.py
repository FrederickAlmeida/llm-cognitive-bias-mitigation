import os

from .anthropic_client import AnthropicClient
from .base import LLMClient, LLMResponse, TokenUsage, calculate_cost
from .deepinfra_client import DeepInfraClient
from .groq_client import GroqClient
from .openai_client import OpenAIClient

__all__ = [
    "LLMClient",
    "LLMResponse",
    "TokenUsage",
    "calculate_cost",
    "make_client",
]


def make_client(provider: str, model: str, api_key: str | None = None) -> LLMClient:
    if provider == "anthropic":
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        return AnthropicClient(model=model, api_key=key)
    elif provider == "openai":
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        return OpenAIClient(model=model, api_key=key)
    elif provider == "groq":
        key = api_key or os.environ.get("GROQ_API_KEY", "")
        return GroqClient(model=model, api_key=key)
    elif provider == "deepinfra":
        key = api_key or os.environ.get("DEEPINFRA_API_KEY", "")
        return DeepInfraClient(model=model, api_key=key)
    raise ValueError(f"Unknown LLM provider: {provider!r}. Choose 'anthropic', 'openai', 'groq', or 'deepinfra'.")
