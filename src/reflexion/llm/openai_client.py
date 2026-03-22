from openai import OpenAI

from .base import LLMClient, LLMResponse, TokenUsage, calculate_cost


class OpenAIClient(LLMClient):
    def __init__(self, model: str, api_key: str) -> None:
        self.model = model
        self._client = OpenAI(api_key=api_key)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        usage = response.usage
        input_tokens = usage.prompt_tokens
        # cached_tokens lives under prompt_tokens_details when available
        cached_input_tokens = 0
        if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
            cached_input_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
        output_tokens = usage.completion_tokens

        cost = calculate_cost(self.model, input_tokens, cached_input_tokens, output_tokens)

        return LLMResponse(
            content=response.choices[0].message.content,
            usage=TokenUsage(
                input_tokens=input_tokens,
                cached_input_tokens=cached_input_tokens,
                output_tokens=output_tokens,
                model=self.model,
                cost_usd=cost,
            ),
        )
