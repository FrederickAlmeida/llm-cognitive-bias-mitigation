import anthropic

from .base import LLMClient, LLMResponse, TokenUsage, calculate_cost


class AnthropicClient(LLMClient):
    def __init__(self, model: str, api_key: str) -> None:
        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key)

    def _complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> LLMResponse:
        message = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        usage = message.usage
        input_tokens = usage.input_tokens
        # cache_read_input_tokens is present when prompt caching is active
        cached_input_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
        output_tokens = usage.output_tokens

        cost = calculate_cost(self.model, input_tokens, cached_input_tokens, output_tokens)

        return LLMResponse(
            content=message.content[0].text,
            usage=TokenUsage(
                input_tokens=input_tokens,
                cached_input_tokens=cached_input_tokens,
                output_tokens=output_tokens,
                model=self.model,
                cost_usd=cost,
            ),
        )
