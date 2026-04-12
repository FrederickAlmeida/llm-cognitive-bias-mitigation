"""Self-Help debiaser.

Implements the two debiasing strategies from the paper:
  1. Prompt-based: LLM rewrites the biased prompt to remove cognitive bias cues.
  2. Anchoring: LLM is shown all its previous decisions and asked to revise any
     that may have been influenced by anchoring bias.

Reference: Echterhoff et al., EMNLP 2024
"""
from __future__ import annotations

from string import Template

import yaml

from src.reflexion.llm.base import LLMResponse
from src.reflexion.llm import LLMClient


_REVISED_PROMPT_START = "[start of revised prompt]"
_REVISED_PROMPT_END = "[end of revised prompt]"


class SelfHelpDebiaser:
    """Rewrites biased prompts (or decisions) to remove cognitive bias."""

    def __init__(self, llm: LLMClient, prompts_path: str) -> None:
        self._llm = llm
        with open(prompts_path, encoding="utf-8") as f:
            self._prompts: dict = yaml.safe_load(f)

    # ── Public API ─────────────────────────────────────────────────────────────

    def debias_prompt(self, prompt: str) -> tuple[str, LLMResponse]:
        """Rewrite a biased prompt to remove cognitive bias cues.

        Args:
            prompt: the original biased prompt text.

        Returns:
            (debiased_prompt, llm_response) — the rewritten prompt and the raw response.
        """
        system = self._prompts["debiaser_prompt_based"]["system"]
        user_template = self._prompts["debiaser_prompt_based"]["user"]
        user = Template(user_template).safe_substitute({"prompt": prompt})

        response = self._llm.complete(system, user, temperature=0.0, max_tokens=2048)
        debiased = _extract_between_tags(response.content, _REVISED_PROMPT_START, _REVISED_PROMPT_END)
        return debiased, response

    def debias_decisions(
        self, conversation_history: str, n_students: int,
    ) -> tuple[str, LLMResponse]:
        """Ask the LLM to revise its anchoring-biased sequential decisions.

        Args:
            conversation_history: full text of the sequential session including
                all student profiles and the model's previous decisions.
            n_students: number of students in the session (for the prompt and
                to validate the JSON output).

        Returns:
            (revised_json, llm_response).
        """
        system = self._prompts["debiaser_anchoring"]["system"]
        user_template = self._prompts["debiaser_anchoring"]["user"]
        user = Template(user_template).safe_substitute(
            {"conversation_history": conversation_history, "n_students": n_students}
        )

        response = self._llm.complete(
            system, user, temperature=0.0, max_tokens=2048, json_mode=True,
        )
        return response.content, response


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_between_tags(text: str, start_tag: str, end_tag: str) -> str:
    """Return text between start_tag and end_tag, stripped.
    Falls back to everything after start_tag if end_tag is missing,
    or the full text if start_tag is also missing.
    """
    start_idx = text.find(start_tag)
    if start_idx == -1:
        return text.strip()
    content_start = start_idx + len(start_tag)
    end_idx = text.find(end_tag, content_start)
    if end_idx == -1:
        return text[content_start:].strip()
    return text[content_start:end_idx].strip()
