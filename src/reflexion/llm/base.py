import time
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TokenUsage:
    input_tokens: int
    cached_input_tokens: int  # prompt cache hits (Anthropic) or cached_tokens (OpenAI)
    output_tokens: int
    model: str
    cost_usd: float  # calculated from known per-model pricing


@dataclass
class LLMResponse:
    content: str
    usage: TokenUsage


# Pricing in USD per million tokens: {model: {input, cached, output}}
PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-6": {"input": 15.00, "cached": 1.50, "output": 75.00},
    "claude-sonnet-4-6": {"input": 3.00, "cached": 0.30, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "cached": 0.08, "output": 4.00},
    "gpt-4o": {"input": 2.50, "cached": 1.25, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "cached": 0.075, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "cached": 0.50, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "cached": 0.10, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "cached": 0.025, "output": 0.40},
    # Groq — prices per million tokens (no prompt caching)
    "llama-3.1-8b-instant": {"input": 0.05, "cached": 0.0, "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "cached": 0.0, "output": 0.79},
    "llama-3.1-70b-versatile": {"input": 0.59, "cached": 0.0, "output": 0.79},
    "mixtral-8x7b-32768": {"input": 0.24, "cached": 0.0, "output": 0.24},
    "openai/gpt-oss-20b": {"input": 0.075, "cached": 0.0, "output": 0.30},
    # DeepInfra
    "openai/gpt-oss-120b": {"input": 0.039, "cached": 0.0, "output": 0.19},
    "Qwen/Qwen3-235B-A22B-Instruct-2507": {"input": 0.071, "cached": 0.0, "output": 0.10},
}


def calculate_cost(model: str, input_tokens: int, cached_input_tokens: int, output_tokens: int) -> float:
    pricing = PRICING.get(model)
    if pricing is None:
        return 0.0
    return (
        input_tokens * pricing["input"] / 1_000_000
        + cached_input_tokens * pricing["cached"] / 1_000_000
        + output_tokens * pricing["output"] / 1_000_000
    )


class LLMClient(ABC):
    max_retries: int = 10
    retry_delay: float = 60.0  # seconds

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> LLMResponse:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._complete(system_prompt, user_prompt, temperature, max_tokens, json_mode)
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    print(f"[retry {attempt}/{self.max_retries}] {exc} — waiting {self.retry_delay:.0f}s")
                    time.sleep(self.retry_delay)
        raise last_exc

    @abstractmethod
    def _complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> LLMResponse:
        ...
