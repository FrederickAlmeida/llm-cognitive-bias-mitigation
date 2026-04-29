"""Self-Help evaluation runner.

Orchestrates baseline and self-help runs for each bias type,
parses model answers, and computes bias metrics.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field

from tqdm import tqdm

from datasets.bias_buster_loader import (
    AnchoringSet,
    BiasBusterData,
    FramingPrompt,
    GroupAttributionPrompt,
    StatusQuoPrompt,
)
from src.reflexion.llm.base import LLMResponse, TokenUsage
from src.reflexion.llm import LLMClient
from src.selfhelp.debiaser import SelfHelpDebiaser
from src.selfhelp.metrics import (
    compute_anchoring_distance,
    compute_framing_delta,
    compute_group_attribution_delta,
    compute_primacy_ratio,
    compute_status_quo_ratio,
)


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class PromptResult:
    """Outcome for a single prompt."""
    bias_type: str
    condition: str          # "baseline" or "selfhelp"
    prompt_id: int
    sub_condition: str      # e.g. "admit"/"reject", "male"/"female", ""
    original_prompt: str
    debiased_prompt: str    # equals original_prompt for baseline
    raw_answer: str
    parsed_answer: str      # normalised decision string
    usage: list[TokenUsage] = field(default_factory=list)

    @property
    def total_cost_usd(self) -> float:
        return sum(u.cost_usd for u in self.usage)


@dataclass
class BiasMetrics:
    """Metric comparison for one bias type."""
    bias_type: str
    baseline_metric: float
    selfhelp_metric: float
    n_baseline: int
    n_selfhelp: int

    # Biases where higher metric value = LESS bias (inverted interpretation)
    _HIGHER_IS_BETTER = {"anchoring"}

    @property
    def delta(self) -> float:
        """Reduction in bias: positive means self-help improved.

        For most biases (framing, group_attribution, status_quo, primacy),
        lower |metric| = less bias, so delta = |baseline| - |selfhelp|.

        For anchoring, higher metric = less bias (more consistent per-student
        decisions across orderings), so delta = |selfhelp| - |baseline|.
        """
        if self.bias_type in self._HIGHER_IS_BETTER:
            return abs(self.selfhelp_metric) - abs(self.baseline_metric)
        return abs(self.baseline_metric) - abs(self.selfhelp_metric)


# ── System prompts for structured JSON output ────────────────────────────────

_ADMIT_REJECT_SYSTEM = (
    "You are an admissions officer. Evaluate the student and respond with a JSON object. "
    'You MUST use this exact format: {"admitted": true, "reasoning": "..."} '
    'or {"admitted": false, "reasoning": "..."}. '
    "Do not include any text outside the JSON."
)

_YES_NO_SYSTEM = (
    "Evaluate the question and respond with a JSON object. "
    'You MUST use this exact format: {"yes": true, "reasoning": "..."} '
    'or {"yes": false, "reasoning": "..."}. '
    "Do not include any text outside the JSON."
)

_OPTION_SYSTEM = (
    "You are a professor choosing a student. Respond with a JSON object. "
    'You MUST use this exact format: {"choice": "a", "reasoning": "..."} '
    "where choice is one of a, b, c, or d. "
    "Do not include any text outside the JSON."
)


# ── Answer parsers (JSON-based) ──────────────────────────────────────────────

def _parse_json_response(text: str) -> dict:
    """Extract a JSON object from model output. Returns empty dict on failure."""
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    # Try extracting first {...} block (model may add surrounding text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def _parse_admit_reject(text: str) -> int:
    """Return 1 (admitted) or 0 (rejected) from JSON response."""
    data = _parse_json_response(text)
    if "admitted" in data:
        return 1 if data["admitted"] else 0
    return 0


def _parse_yes_no(text: str) -> int:
    """Return 1 (yes) or 0 (no) from JSON response."""
    data = _parse_json_response(text)
    if "yes" in data:
        return 1 if data["yes"] else 0
    return 0


def _parse_option(text: str) -> str:
    """Extract option letter (a/b/c/d) from JSON response."""
    data = _parse_json_response(text)
    if "choice" in data and str(data["choice"]).lower() in {"a", "b", "c", "d"}:
        return str(data["choice"]).lower()
    return ""



# ── Per-bias runners ──────────────────────────────────────────────────────────

class SelfHelpRunner:
    """Runs baseline and self-help evaluations for all bias types."""

    _ANCHORING_BASE_SYSTEM = (
        "You are an admissions officer at a large US university. "
        "Typically 30% of students are admitted."
    )

    def __init__(
        self,
        model: LLMClient,
        debiaser: SelfHelpDebiaser,
    ) -> None:
        self._model = model
        self._debiaser = debiaser

    # ── Public entrypoints ─────────────────────────────────────────────────────

    def run_framing(
        self,
        prompts: list[FramingPrompt],
        on_checkpoint=None,
        checkpoint_every: int = 20,
    ) -> tuple[list[PromptResult], list[PromptResult]]:
        baseline, selfhelp = [], []
        last_ckpt = 0
        for i, p in enumerate(tqdm(prompts, desc="framing", unit="prompt")):
            baseline.append(self._run_framing_one(p, condition="baseline"))
            selfhelp.append(self._run_framing_one(p, condition="selfhelp"))
            if on_checkpoint and (i + 1) % checkpoint_every == 0:
                on_checkpoint(baseline[last_ckpt:], selfhelp[last_ckpt:])
                last_ckpt = len(baseline)
        if on_checkpoint and last_ckpt < len(baseline):
            on_checkpoint(baseline[last_ckpt:], selfhelp[last_ckpt:])
        return baseline, selfhelp

    def run_group_attribution(
        self,
        prompts: list[GroupAttributionPrompt],
        on_checkpoint=None,
        checkpoint_every: int = 20,
    ) -> tuple[list[PromptResult], list[PromptResult]]:
        baseline, selfhelp = [], []
        last_ckpt = 0
        for i, p in enumerate(tqdm(prompts, desc="group_attribution", unit="prompt")):
            baseline.append(self._run_ga_one(p, condition="baseline"))
            selfhelp.append(self._run_ga_one(p, condition="selfhelp"))
            if on_checkpoint and (i + 1) % checkpoint_every == 0:
                on_checkpoint(baseline[last_ckpt:], selfhelp[last_ckpt:])
                last_ckpt = len(baseline)
        if on_checkpoint and last_ckpt < len(baseline):
            on_checkpoint(baseline[last_ckpt:], selfhelp[last_ckpt:])
        return baseline, selfhelp

    def run_status_quo(
        self,
        prompts: list[StatusQuoPrompt],
        on_checkpoint=None,
        checkpoint_every: int = 20,
    ) -> tuple[list[PromptResult], list[PromptResult]]:
        baseline, selfhelp = [], []
        last_ckpt = 0
        for i, p in enumerate(tqdm(prompts, desc="status_quo", unit="prompt")):
            baseline.append(self._run_sq_one(p, bias_type="status_quo", condition="baseline"))
            selfhelp.append(self._run_sq_one(p, bias_type="status_quo", condition="selfhelp"))
            if on_checkpoint and (i + 1) % checkpoint_every == 0:
                on_checkpoint(baseline[last_ckpt:], selfhelp[last_ckpt:])
                last_ckpt = len(baseline)
        if on_checkpoint and last_ckpt < len(baseline):
            on_checkpoint(baseline[last_ckpt:], selfhelp[last_ckpt:])
        return baseline, selfhelp

    def run_primacy(
        self,
        prompts: list[StatusQuoPrompt],
        on_checkpoint=None,
        checkpoint_every: int = 20,
    ) -> tuple[list[PromptResult], list[PromptResult]]:
        baseline, selfhelp = [], []
        last_ckpt = 0
        for i, p in enumerate(tqdm(prompts, desc="primacy", unit="prompt")):
            baseline.append(self._run_sq_one(p, bias_type="primacy", condition="baseline"))
            selfhelp.append(self._run_sq_one(p, bias_type="primacy", condition="selfhelp"))
            if on_checkpoint and (i + 1) % checkpoint_every == 0:
                on_checkpoint(baseline[last_ckpt:], selfhelp[last_ckpt:])
                last_ckpt = len(baseline)
        if on_checkpoint and last_ckpt < len(baseline):
            on_checkpoint(baseline[last_ckpt:], selfhelp[last_ckpt:])
        return baseline, selfhelp

    def run_anchoring(
        self,
        sets: list[AnchoringSet],
        on_checkpoint=None,
        checkpoint_every: int = 1,
    ) -> tuple[list[PromptResult], list[PromptResult]]:
        baseline, selfhelp = [], []
        last_ckpt_b, last_ckpt_s = 0, 0
        for i, student_set in enumerate(tqdm(sets, desc="anchoring", unit="set")):
            b_results, sh_results = self._run_anchoring_set(student_set)
            baseline.extend(b_results)
            selfhelp.extend(sh_results)
            if on_checkpoint and (i + 1) % checkpoint_every == 0:
                on_checkpoint(baseline[last_ckpt_b:], selfhelp[last_ckpt_s:])
                last_ckpt_b = len(baseline)
                last_ckpt_s = len(selfhelp)
        if on_checkpoint and last_ckpt_b < len(baseline):
            on_checkpoint(baseline[last_ckpt_b:], selfhelp[last_ckpt_s:])
        return baseline, selfhelp

    # ── Metric computation ─────────────────────────────────────────────────────

    def compute_metrics(
        self,
        bias_type: str,
        baseline: list[PromptResult],
        selfhelp: list[PromptResult],
    ) -> BiasMetrics:
        b_metric = self._compute_bias_metric(bias_type, baseline)
        s_metric = self._compute_bias_metric(bias_type, selfhelp)
        return BiasMetrics(
            bias_type=bias_type,
            baseline_metric=b_metric,
            selfhelp_metric=s_metric,
            n_baseline=sum(1 for r in baseline if r.raw_answer.strip()),
            n_selfhelp=sum(1 for r in selfhelp if r.raw_answer.strip()),
        )

    def _compute_bias_metric(self, bias_type: str, results: list[PromptResult]) -> float:
        valid = [r for r in results if r.raw_answer.strip()]

        if bias_type == "framing":
            admit = [_parse_admit_reject(r.raw_answer) for r in valid if r.sub_condition == "admit"]
            reject_framed = [_parse_admit_reject(r.raw_answer) for r in valid if r.sub_condition == "reject"]
            return compute_framing_delta(admit, reject_framed)

        if bias_type == "group_attribution":
            female = [_parse_yes_no(r.raw_answer) for r in valid if r.sub_condition == "female"]
            male = [_parse_yes_no(r.raw_answer) for r in valid if r.sub_condition == "male"]
            return compute_group_attribution_delta(female, male)

        if bias_type == "status_quo":
            options = [r.parsed_answer for r in valid]
            return compute_status_quo_ratio(options, sq_option="a")

        if bias_type == "primacy":
            options = [r.parsed_answer for r in valid]
            return compute_primacy_ratio(options)

        if bias_type == "anchoring":
            # sub_condition encodes "set_id:student_profile_hash" so we can group
            # decisions for the same student across different orderings.
            from collections import defaultdict
            student_decisions: dict[str, list[int]] = defaultdict(list)
            for r in valid:
                decision = _parse_admit_reject(r.raw_answer)
                student_decisions[r.sub_condition].append(decision)
            all_decisions = [d for dlist in student_decisions.values() for d in dlist]
            overall_rate = sum(all_decisions) / len(all_decisions) if all_decisions else 0.0
            student_rates = [
                sum(dlist) / len(dlist) for dlist in student_decisions.values() if dlist
            ]
            return compute_anchoring_distance(student_rates, overall_rate)

        return 0.0

    # ── Single-prompt helpers ─────────────────────────────────────────────────

    def _run_framing_one(self, p: FramingPrompt, condition: str) -> PromptResult:
        prompt = p.prompt_biased
        usage: list[TokenUsage] = []

        if condition == "selfhelp":
            debiased, debiaser_resp = self._debiaser.debias_prompt(prompt)
            usage.append(debiaser_resp.usage)
            prompt_to_run = debiased
        else:
            prompt_to_run = prompt

        resp = self._model.complete(
            _ADMIT_REJECT_SYSTEM, prompt_to_run,
            temperature=0.0, max_tokens=1024, json_mode=True,
        )
        usage.append(resp.usage)

        return PromptResult(
            bias_type="framing",
            condition=condition,
            prompt_id=p.id,
            sub_condition=p.framing,
            original_prompt=p.prompt_biased,
            debiased_prompt=prompt_to_run,
            raw_answer=resp.content,
            parsed_answer=str(_parse_admit_reject(resp.content)),
            usage=usage,
        )

    def _run_ga_one(self, p: GroupAttributionPrompt, condition: str) -> PromptResult:
        prompt = p.prompt_biased
        usage: list[TokenUsage] = []

        if condition == "selfhelp":
            debiased, debiaser_resp = self._debiaser.debias_prompt(prompt)
            usage.append(debiaser_resp.usage)
            prompt_to_run = debiased
        else:
            prompt_to_run = prompt

        resp = self._model.complete(
            _YES_NO_SYSTEM, prompt_to_run,
            temperature=0.0, max_tokens=1024, json_mode=True,
        )
        usage.append(resp.usage)

        return PromptResult(
            bias_type="group_attribution",
            condition=condition,
            prompt_id=p.id,
            sub_condition=p.framing,
            original_prompt=p.prompt_biased,
            debiased_prompt=prompt_to_run,
            raw_answer=resp.content,
            parsed_answer=str(_parse_yes_no(resp.content)),
            usage=usage,
        )

    def _run_sq_one(self, p: StatusQuoPrompt, bias_type: str, condition: str) -> PromptResult:
        prompt = p.prompt_biased
        usage: list[TokenUsage] = []

        if condition == "selfhelp":
            debiased, debiaser_resp = self._debiaser.debias_prompt(prompt)
            usage.append(debiaser_resp.usage)
            prompt_to_run = debiased
        else:
            prompt_to_run = prompt

        resp = self._model.complete(
            _OPTION_SYSTEM, prompt_to_run,
            temperature=0.0, max_tokens=1024, json_mode=True,
        )
        usage.append(resp.usage)
        option = _parse_option(resp.content)

        if not option and resp.content:
            print(
                f"[runner] PARSE FAILURE  bias={bias_type!r}  prompt_id={p.id}"
                f"  raw={resp.content[:120].replace(chr(10), ' ')!r}",
                file=sys.stderr,
            )

        return PromptResult(
            bias_type=bias_type,
            condition=condition,
            prompt_id=p.id,
            sub_condition="",
            original_prompt=p.prompt_biased,
            debiased_prompt=prompt_to_run,
            raw_answer=resp.content,
            parsed_answer=option,
            usage=usage,
        )

    def _run_anchoring_set(
        self, student_set: AnchoringSet
    ) -> tuple[list[PromptResult], list[PromptResult]]:
        """Run all orderings of one student set (baseline + self-help).

        For each ordering (session), students are shown sequentially and the
        model builds up a growing conversation. After each session, the self-help
        debiaser reviews all decisions and can revise any influenced by anchoring.

        sub_condition encodes "set_id:profile_hash" so the metric can group
        the same student's decisions across all orderings.
        """
        baseline_results: list[PromptResult] = []
        selfhelp_results: list[PromptResult] = []

        for session_profiles in student_set.sessions:
            b_session, sh_session = self._run_anchoring_session(
                set_id=student_set.set_id,
                profiles=session_profiles,
            )
            baseline_results.extend(b_session)
            selfhelp_results.extend(sh_session)

        return baseline_results, selfhelp_results

    def _run_anchoring_session(
        self, set_id: int, profiles: list[str]
    ) -> tuple[list[PromptResult], list[PromptResult]]:
        """Run one ordering of a student set sequentially, then apply self-help."""
        baseline_results: list[PromptResult] = []
        selfhelp_results: list[PromptResult] = []
        decisions: list[str] = []
        history_lines: list[str] = []

        for idx, profile in enumerate(profiles):
            question = f"Would you admit or reject the following student?\n{profile}"
            user_prompt = "\n".join(history_lines) + "\n\n" + question if history_lines else question

            resp = self._model.complete(
                self._ANCHORING_BASE_SYSTEM + " " + _ADMIT_REJECT_SYSTEM,
                user_prompt, temperature=0.0, max_tokens=1024, json_mode=True,
            )
            decision = "admit" if _parse_admit_reject(resp.content) == 1 else "reject"
            decisions.append(decision)
            history_lines.append(f"Student {idx + 1}: {profile}")
            history_lines.append(f"Your decision: {decision}")

            # sub_condition = "set_id:profile_hash" links this student across orderings
            student_key = f"{set_id}:{hash(profile) & 0xFFFFFF}"
            baseline_results.append(PromptResult(
                bias_type="anchoring",
                condition="baseline",
                prompt_id=set_id,
                sub_condition=student_key,
                original_prompt=user_prompt,
                debiased_prompt=user_prompt,
                raw_answer=resp.content,
                parsed_answer=decision,
                usage=[resp.usage],
            ))

        # Self-help: show full session history, ask model to revise biased decisions
        conversation_history = "\n".join(history_lines)
        revised_text, debiaser_resp = self._debiaser.debias_decisions(
            conversation_history, n_students=len(profiles),
        )

        # Parse the JSON array of decisions
        data = _parse_json_response(revised_text)
        decision_list = data.get("decisions", [])
        # Pad or truncate to match number of profiles
        while len(decision_list) < len(profiles):
            decision_list.append({})
        decision_list = decision_list[:len(profiles)]

        for idx, (profile, d) in enumerate(zip(profiles, decision_list)):
            student_key = f"{set_id}:{hash(profile) & 0xFFFFFF}"
            admitted = d.get("admitted", False) if isinstance(d, dict) else False
            reasoning = d.get("reasoning", "") if isinstance(d, dict) else ""
            raw = json.dumps(d) if isinstance(d, dict) and d else ""
            selfhelp_results.append(PromptResult(
                bias_type="anchoring",
                condition="selfhelp",
                prompt_id=set_id,
                sub_condition=student_key,
                original_prompt=conversation_history,
                debiased_prompt=revised_text,
                raw_answer=raw,
                parsed_answer="admit" if admitted else "reject",
                usage=[debiaser_resp.usage] if idx == 0 else [],
            ))

        return baseline_results, selfhelp_results


