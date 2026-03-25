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


_REVISED_PROMPT_TAG = "[start of revised prompt]"
_REVISED_DECISIONS_TAG = "[start of revised decisions]"


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
        debiased = _extract_after_tag(response.content, _REVISED_PROMPT_TAG)
        return debiased, response

    def debias_decisions(self, conversation_history: str) -> tuple[str, LLMResponse]:
        """Ask the LLM to revise its anchoring-biased sequential decisions.

        Args:
            conversation_history: full text of the sequential session including
                all student profiles and the model's previous decisions.

        Returns:
            (revised_decisions, llm_response).
        """
        system = self._prompts["debiaser_anchoring"]["system"]
        user_template = self._prompts["debiaser_anchoring"]["user"]
        user = Template(user_template).safe_substitute(
            {"conversation_history": conversation_history}
        )

        response = self._llm.complete(system, user, temperature=0.0, max_tokens=2048)
        revised = _extract_after_tag(response.content, _REVISED_DECISIONS_TAG)
        return revised, response


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_after_tag(text: str, tag: str) -> str:
    """Return everything after `tag` in `text`, stripped. Falls back to full text."""
    idx = text.find(tag)
    if idx == -1:
        return text.strip()
    return text[idx + len(tag):].strip()
