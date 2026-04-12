import sys

from openai import OpenAI

from .base import LLMClient, LLMResponse, TokenUsage, calculate_cost

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqClient(LLMClient):
    def __init__(self, model: str, api_key: str) -> None:
        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    def _complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> LLMResponse:
        kwargs: dict = dict(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(**kwargs)

        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens

        choice = response.choices[0]
        content = choice.message.content or ""
        finish_reason = choice.finish_reason

        if not content:
            prompt_snippet = user_prompt[:120].replace("\n", " ")
            print(
                f"[groq] EMPTY RESPONSE  finish_reason={finish_reason!r}"
                f"  input_tokens={input_tokens}  output_tokens={output_tokens}"
                f"  prompt_snippet={prompt_snippet!r}",
                file=sys.stderr,
            )

        cost = calculate_cost(self.model, input_tokens, 0, output_tokens)

        return LLMResponse(
            content=content,
            usage=TokenUsage(
                input_tokens=input_tokens,
                cached_input_tokens=0,
                output_tokens=output_tokens,
                model=self.model,
                cost_usd=cost,
            ),
        )
