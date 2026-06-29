"""Reflexion-based cognitive bias debiasing runner.

Loads prior-step CSV results (baseline or reflexion step N-1), applies the
Reflexion reflect-then-re-answer loop, and produces new CSVs for the next step.

Evaluator: rule-based pairwise consistency check (no LLM judge).
  - Framing/Group attribution: FAIL if same prompt_id gets different decisions
    under the two framings (admit/reject or female/male).
  - Anchoring: FAIL if per-student consistency score |rate-0.5|/0.5 < threshold
    for any student in the set.

Consistent pairs/sets are carried forward as-is (zero cost).
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from tqdm import tqdm

from src.reflexion.agent import PromptLoader
from src.reflexion.llm import LLMClient
from src.reflexion.llm.base import TokenUsage
from src.reflexion.memory import MemoryStore
from src.reflexion.reflection import SelfReflection
from src.selfhelp.metrics import (
    compute_anchoring_distance,
    compute_framing_delta,
    compute_group_attribution_delta,
    compute_primacy_ratio,
    compute_status_quo_ratio,
)
from src.selfhelp.runner import (
    BiasMetrics,
    SelfHelpRunner,
    _ADMIT_REJECT_SYSTEM,
    _OPTION_SYSTEM,
    _YES_NO_SYSTEM,
    _parse_admit_reject,
    _parse_json_response,
    _parse_option,
    _parse_yes_no,
)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ReflexionStepResult:
    """Outcome for one prompt/row in a single Reflexion step."""
    bias_type: str
    step: int
    prompt_id: int
    sub_condition: str
    original_prompt: str
    prior_raw_answer: str       # the answer this step reflected on
    reflection_text: str        # empty string if consistent (PASS, no reflection)
    raw_answer: str             # equals prior_raw_answer if consistent
    parsed_answer: str
    usage: list[TokenUsage] = field(default_factory=list)

    @property
    def total_cost_usd(self) -> float:
        return sum(u.cost_usd for u in self.usage)


# ── Bias-specific SelfReflection ──────────────────────────────────────────────

class _BiasSelfReflection(SelfReflection):
    """Uses the 'bias_reflection' prompt key instead of 'self_reflection'."""

    def generate(self, question, prior_answer, feedback, memory):  # type: ignore[override]
        system = self.prompt_loader.render("bias_reflection", "system", {})
        user = self.prompt_loader.render(
            "bias_reflection",
            "user",
            {
                "question": question,
                "prior_answer": prior_answer,
                "feedback": feedback,
                "memory": memory.format_for_prompt(),
            },
        )
        response = self.llm.complete(system, user, temperature=0.0, max_tokens=2048)
        return response.content.strip(), response


# ── Procedural feedback ─────────────────────────────────────────────────────────
#
# Strictly procedural reflection trigger shared by all bias-driven loops. It
# carries NO directional information: it never names the bias under test, the
# manipulated condition (framing, gender, ordering, status quo), the previously
# chosen option, or any direction to move toward or away from. It only instructs
# the model to re-evaluate on individual academic merit with full contextual
# neutrality.
#
# This removes the data leakage flagged in review: any change in the consistency
# metric must now come from genuine re-evaluation, not from the actor avoiding a
# label that the evaluator flagged as suspicious. The rule-based consistency
# check still decides *whether* to reflect (pairwise/per-set gating), but that
# decision is internal and is never surfaced to the model.
#
# IMPORTANT — purely POSITIVE phrasing. Earlier wording told the model what to
# ignore ("do not let any contextual or presentational feature influence you").
# Larger models (gpt-oss, Qwen) helpfully enumerated those features back —
# "without reference to presentation order", "ignoring narrative framing",
# "sequence of information" — which names the manipulated condition and, for
# primacy, the bias mechanism itself. That enumerated text then lands in
# episodic memory. To prevent it, the signal now states ONLY what to do (judge
# on stated academic merit) and never names anything to avoid.
_PROCEDURAL_FEEDBACK = (
    "Re-evaluate this decision carefully and from scratch, considering only the "
    "academic merit and qualifications described for each candidate."
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _memory_suffix(memory: MemoryStore) -> str:
    if memory.is_empty():
        return ""
    return "\n\nPrevious reflections on this decision:\n" + memory.format_for_prompt()


def _consistency_score(rate: float) -> float:
    """Per-student consistency score in [0, 1]. 1=perfectly consistent, 0=maximally inconsistent."""
    return abs(rate - 0.5) / 0.5


def _compute_metric(
    bias_type: str,
    raw_answers: list[str],
    sub_conditions: list[str],
) -> float:
    """Shared metric computation for both baseline dicts and ReflexionStepResult lists."""
    pairs = [(sub, raw) for sub, raw in zip(sub_conditions, raw_answers) if raw.strip()]

    if bias_type == "framing":
        admit = [_parse_admit_reject(raw) for sub, raw in pairs if sub == "admit"]
        reject_framed = [_parse_admit_reject(raw) for sub, raw in pairs if sub == "reject"]
        return compute_framing_delta(admit, reject_framed)

    if bias_type == "group_attribution":
        female = [_parse_yes_no(raw) for sub, raw in pairs if sub == "female"]
        male = [_parse_yes_no(raw) for sub, raw in pairs if sub == "male"]
        return compute_group_attribution_delta(female, male)

    if bias_type == "anchoring":
        student_decisions: dict[str, list[int]] = defaultdict(list)
        for sub, raw in pairs:
            student_decisions[sub].append(_parse_admit_reject(raw))
        all_decisions = [d for dlist in student_decisions.values() for d in dlist]
        if not all_decisions:
            return 0.0
        overall_rate = sum(all_decisions) / len(all_decisions)
        student_rates = [sum(dlist) / len(dlist) for dlist in student_decisions.values() if dlist]
        return compute_anchoring_distance(student_rates, overall_rate)

    if bias_type == "status_quo":
        options = [_parse_option(raw) for _, raw in pairs]
        return compute_status_quo_ratio(options, sq_option="a")

    if bias_type == "primacy":
        options = [_parse_option(raw) for _, raw in pairs]
        return compute_primacy_ratio(options)

    return 0.0


# ── Runner ────────────────────────────────────────────────────────────────────

class ReflexionBiasRunner:
    """Applies one Reflexion step to bias evaluation results loaded from CSV."""

    def __init__(
        self,
        actor_llm: LLMClient,
        reflection_llm: LLMClient,
        prompts_path: str,
    ) -> None:
        self._actor_llm = actor_llm
        self._reflection = _BiasSelfReflection(reflection_llm, PromptLoader(prompts_path))

    # ── Unit driver (serial or threaded) ───────────────────────────────────────

    def _run_units(
        self, items, process, desc, on_checkpoint, checkpoint_every, max_workers,
    ) -> list[ReflexionStepResult]:
        """Apply ``process(item) -> list[ReflexionStepResult]`` over ``items``.

        Units are independent (each pair/row reflects on its own), so they may run
        concurrently on ``max_workers`` threads (LLM calls are I/O-bound; the LLM
        clients are thread-safe and carry their own retry). Results are collected
        and checkpointed in the main thread, so ``on_checkpoint`` needs no lock.
        Order is not preserved when threaded (irrelevant: resume keys on prompt_id).
        """
        results: list[ReflexionStepResult] = []
        last = 0

        def maybe_ckpt(force=False):
            nonlocal last
            if on_checkpoint and (force or len(results) - last >= checkpoint_every):
                on_checkpoint(results[last:])
                last = len(results)

        if max_workers <= 1:
            for it in tqdm(items, desc=desc):
                results.extend(process(it))
                maybe_ckpt()
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(process, it) for it in items]
                for fut in tqdm(as_completed(futs), total=len(futs), desc=desc):
                    results.extend(fut.result())
                    maybe_ckpt()
        maybe_ckpt(force=True)
        return results

    # ── Public entrypoints ─────────────────────────────────────────────────────

    def run_framing(
        self,
        rows: list[dict],
        step: int,
        prior_reflections: dict[tuple[int, str], str] | None = None,
        on_checkpoint=None,
        checkpoint_every: int = 20,
        max_workers: int = 1,
    ) -> list[ReflexionStepResult]:
        """Run one Reflexion step for framing bias.

        Groups rows by prompt_id into (admit, reject) pairs. Consistent pairs
        (same parsed_answer) are carried forward. For inconsistent pairs, each
        side is reflected on INDEPENDENTLY — its own (prompt, prior answer) — and
        re-answered using only its own reflection (design "b"). This mirrors
        SelfHelp's per-prompt independence and avoids any cross-condition
        contamination. ``prior_reflections`` is keyed by ``(prompt_id,
        sub_condition)`` so the per-side chain is preserved across steps.
        """
        pairs: dict[int, dict[str, dict]] = defaultdict(dict)
        for row in rows:
            pairs[int(row["prompt_id"])][row["sub_condition"]] = row
        prior = prior_reflections or {}

        def process(item):
            pid, pair = item
            admit_row = pair.get("admit")
            reject_row = pair.get("reject")
            if not admit_row or not reject_row:
                return [self._carry_forward(r, step, "framing") for r in pair.values()]
            if admit_row.get("parsed_answer", "") == reject_row.get("parsed_answer", ""):
                return [self._carry_forward(admit_row, step, "framing"),
                        self._carry_forward(reject_row, step, "framing")]
            out = []
            for row in (admit_row, reject_row):
                memory = MemoryStore()
                seed = prior.get((pid, row["sub_condition"]))
                if seed:
                    memory.add(seed)
                refl_text, refl_resp = self._reflect(
                    row["original_prompt"], row.get("raw_answer", ""),
                    _PROCEDURAL_FEEDBACK, memory,
                )
                memory.add(refl_text)
                resp = self._actor_llm.complete(
                    _ADMIT_REJECT_SYSTEM,
                    row["original_prompt"] + _memory_suffix(memory),
                    temperature=0.0, max_tokens=2048, json_mode=True,
                )
                out.append(ReflexionStepResult(
                    bias_type="framing", step=step, prompt_id=pid,
                    sub_condition=row["sub_condition"],
                    original_prompt=row["original_prompt"],
                    prior_raw_answer=row.get("raw_answer", ""),
                    reflection_text=refl_text,
                    raw_answer=resp.content,
                    parsed_answer=str(_parse_admit_reject(resp.content)),
                    usage=[refl_resp.usage, resp.usage],
                ))
            return out

        return self._run_units(list(pairs.items()), process, f"framing step {step}",
                               on_checkpoint, checkpoint_every, max_workers)

    def run_group_attribution(
        self,
        rows: list[dict],
        step: int,
        prior_reflections: dict[tuple[int, str], str] | None = None,
        on_checkpoint=None,
        checkpoint_every: int = 20,
        max_workers: int = 1,
    ) -> list[ReflexionStepResult]:
        """Run one Reflexion step for group attribution bias (female/male pairs).

        Each side of an inconsistent pair is reflected on INDEPENDENTLY and
        re-answered using only its own reflection (design "b"); ``prior_reflections``
        is keyed by ``(prompt_id, sub_condition)``.
        """
        pairs: dict[int, dict[str, dict]] = defaultdict(dict)
        for row in rows:
            pairs[int(row["prompt_id"])][row["sub_condition"]] = row
        prior = prior_reflections or {}

        def process(item):
            pid, pair = item
            female_row = pair.get("female")
            male_row = pair.get("male")
            if not female_row or not male_row:
                return [self._carry_forward(r, step, "group_attribution") for r in pair.values()]
            if female_row.get("parsed_answer", "") == male_row.get("parsed_answer", ""):
                return [self._carry_forward(female_row, step, "group_attribution"),
                        self._carry_forward(male_row, step, "group_attribution")]
            out = []
            for row in (female_row, male_row):
                memory = MemoryStore()
                seed = prior.get((pid, row["sub_condition"]))
                if seed:
                    memory.add(seed)
                refl_text, refl_resp = self._reflect(
                    row["original_prompt"], row.get("raw_answer", ""),
                    _PROCEDURAL_FEEDBACK, memory,
                )
                memory.add(refl_text)
                resp = self._actor_llm.complete(
                    _YES_NO_SYSTEM,
                    row["original_prompt"] + _memory_suffix(memory),
                    temperature=0.0, max_tokens=2048, json_mode=True,
                )
                out.append(ReflexionStepResult(
                    bias_type="group_attribution", step=step, prompt_id=pid,
                    sub_condition=row["sub_condition"],
                    original_prompt=row["original_prompt"],
                    prior_raw_answer=row.get("raw_answer", ""),
                    reflection_text=refl_text,
                    raw_answer=resp.content,
                    parsed_answer=str(_parse_yes_no(resp.content)),
                    usage=[refl_resp.usage, resp.usage],
                ))
            return out

        return self._run_units(list(pairs.items()), process, f"group_attribution step {step}",
                               on_checkpoint, checkpoint_every, max_workers)

    def run_anchoring(
        self,
        rows: list[dict],
        step: int,
        prior_reflections: dict[int, str] | None = None,
        consistency_threshold: float = 1.0,
        on_checkpoint=None,
        checkpoint_every: int = 1,
    ) -> list[ReflexionStepResult]:
        """Run one Reflexion step for anchoring bias.

        Groups rows by set_id. Computes per-student consistency score
        |rate-0.5|/0.5 across orderings. If any student falls below
        consistency_threshold, the whole set is reflected and re-answered.
        """
        sets: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            set_id = int(row["sub_condition"].split(":")[0])
            sets[set_id].append(row)

        results: list[ReflexionStepResult] = []
        prior = prior_reflections or {}
        last_ckpt = 0
        system = SelfHelpRunner._ANCHORING_BASE_SYSTEM + " " + _ADMIT_REJECT_SYSTEM

        for i, (set_id, set_rows) in enumerate(tqdm(sets.items(), desc=f"anchoring step {step}", unit="set")):
            student_decisions: dict[str, list[int]] = defaultdict(list)
            for row in set_rows:
                decision = 1 if row.get("parsed_answer") == "admit" else 0
                student_decisions[row["sub_condition"]].append(decision)

            scores = {
                key: _consistency_score(sum(dlist) / len(dlist) if dlist else 0.5)
                for key, dlist in student_decisions.items()
            }

            if all(s >= consistency_threshold for s in scores.values()):
                for row in set_rows:
                    results.append(self._carry_forward(row, step, "anchoring"))
            else:
                feedback = _PROCEDURAL_FEEDBACK
                memory = MemoryStore()
                if set_id in prior:
                    memory.add(prior[set_id])

                rep_row = set_rows[0]
                # Pass only the decision, not the full reasoning JSON — passing
                # the actor's reasoning causes the reflection model to echo
                # credential-specific language (e.g. "GPA below average") back
                # as boilerplate, which leaks evaluative hints to the actor.
                rep_decision = rep_row.get("parsed_answer", "reject")
                refl_text, refl_resp = self._reflect(
                    rep_row["original_prompt"], rep_decision,
                    feedback, memory,
                )
                memory.add(refl_text)

                for j, row in enumerate(set_rows):
                    resp = self._actor_llm.complete(
                        system,
                        row["original_prompt"] + _memory_suffix(memory),
                        temperature=0.0, max_tokens=2048, json_mode=True,
                    )
                    decision = "admit" if _parse_admit_reject(resp.content) == 1 else "reject"
                    usage = ([refl_resp.usage] if j == 0 else []) + [resp.usage]
                    results.append(ReflexionStepResult(
                        bias_type="anchoring", step=step,
                        prompt_id=int(row["prompt_id"]),
                        sub_condition=row["sub_condition"],
                        original_prompt=row["original_prompt"],
                        prior_raw_answer=row.get("raw_answer", ""),
                        reflection_text=refl_text,
                        raw_answer=resp.content,
                        parsed_answer=decision,
                        usage=usage,
                    ))

            if on_checkpoint and (i + 1) % checkpoint_every == 0:
                on_checkpoint(results[last_ckpt:])
                last_ckpt = len(results)

        if on_checkpoint and last_ckpt < len(results):
            on_checkpoint(results[last_ckpt:])
        return results

    def run_primacy(
        self,
        rows: list[dict],
        step: int,
        prior_reflections: dict[int, str] | None = None,
        on_checkpoint=None,
        checkpoint_every: int = 20,
        max_workers: int = 1,
    ) -> list[ReflexionStepResult]:
        """Run one Reflexion step for primacy bias.

        Every row is reflected on unconditionally — primacy cannot be detected
        from a single response. The evaluator always prompts the model to
        reconsider whether option order influenced its choice.
        """
        prior = prior_reflections or {}

        def process(row):
            pid = int(row["prompt_id"])
            prior_raw = row.get("raw_answer", "")
            memory = MemoryStore()
            if pid in prior:
                memory.add(prior[pid])
            refl_text, refl_resp = self._reflect(
                row["original_prompt"], prior_raw, _PROCEDURAL_FEEDBACK, memory)
            memory.add(refl_text)
            resp = self._actor_llm.complete(
                _OPTION_SYSTEM,
                row["original_prompt"] + _memory_suffix(memory),
                temperature=0.0, max_tokens=2048, json_mode=True,
            )
            return [ReflexionStepResult(
                bias_type="primacy", step=step, prompt_id=pid,
                sub_condition=row.get("sub_condition", ""),
                original_prompt=row["original_prompt"],
                prior_raw_answer=prior_raw,
                reflection_text=refl_text,
                raw_answer=resp.content,
                parsed_answer=_parse_option(resp.content),
                usage=[refl_resp.usage, resp.usage],
            )]

        return self._run_units(list(rows), process, f"primacy step {step}",
                               on_checkpoint, checkpoint_every, max_workers)

    def run_status_quo(
        self,
        rows: list[dict],
        step: int,
        prior_reflections: dict[int, str] | None = None,
        on_checkpoint=None,
        checkpoint_every: int = 20,
    ) -> list[ReflexionStepResult]:
        """Run one Reflexion step for status quo bias.

        Every row is reflected on unconditionally — status quo bias cannot be
        detected from a single response. The evaluator always prompts the model
        to reconsider whether preference for the existing state influenced its choice.
        """
        results: list[ReflexionStepResult] = []
        prior = prior_reflections or {}
        last_ckpt = 0

        for i, row in enumerate(tqdm(rows, desc=f"status_quo step {step}", unit="prompt")):
            pid = int(row["prompt_id"])
            prior_raw = row.get("raw_answer", "")

            data = _parse_json_response(prior_raw)
            prior_choice = data.get("choice", row.get("parsed_answer", "?"))
            prior_reasoning = data.get("reasoning", "").strip()
            reasoning_clause = f' Your reasoning was: "{prior_reasoning}".' if prior_reasoning else ""

            feedback = (
                f"Your previous choice was option '{prior_choice}'.{reasoning_clause} "
                "Consider whether you favored option 'a' because it represents the STATUS QUO "
                "(the current/existing candidate already working in your lab). "
                "Status quo bias causes people to prefer maintaining the existing state even when "
                "alternatives may be equally or more suitable. "
                "Re-evaluate all candidates purely on their individual qualifications."
            )

            memory = MemoryStore()
            if pid in prior:
                memory.add(prior[pid])

            refl_text, refl_resp = self._reflect(row["original_prompt"], prior_raw, feedback, memory)
            memory.add(refl_text)

            resp = self._actor_llm.complete(
                _OPTION_SYSTEM,
                row["original_prompt"] + _memory_suffix(memory),
                temperature=0.0, max_tokens=2048, json_mode=True,
            )
            results.append(ReflexionStepResult(
                bias_type="status_quo", step=step, prompt_id=pid,
                sub_condition=row.get("sub_condition", ""),
                original_prompt=row["original_prompt"],
                prior_raw_answer=prior_raw,
                reflection_text=refl_text,
                raw_answer=resp.content,
                parsed_answer=_parse_option(resp.content),
                usage=[refl_resp.usage, resp.usage],
            ))

            if on_checkpoint and (i + 1) % checkpoint_every == 0:
                on_checkpoint(results[last_ckpt:])
                last_ckpt = len(results)

        if on_checkpoint and last_ckpt < len(results):
            on_checkpoint(results[last_ckpt:])
        return results

    # ── Metric computation ─────────────────────────────────────────────────────

    def compute_metrics(
        self,
        bias_type: str,
        baseline_rows: list[dict],
        reflexion_results: list[ReflexionStepResult],
    ) -> BiasMetrics:
        # Filter baseline to only rows present in reflexion output for a fair comparison.
        # For anchoring, key is sub_condition (set_id:hash); for others, (prompt_id, sub_condition).
        refl_keys: set[tuple[int, str]] = {
            (r.prompt_id, r.sub_condition) for r in reflexion_results
        }
        matched_baseline = [
            r for r in baseline_rows
            if (int(r["prompt_id"]), r["sub_condition"]) in refl_keys
        ]

        b_metric = _compute_metric(
            bias_type,
            [r.get("raw_answer", "") for r in matched_baseline],
            [r.get("sub_condition", "") for r in matched_baseline],
        )
        r_metric = _compute_metric(
            bias_type,
            [r.raw_answer for r in reflexion_results],
            [r.sub_condition for r in reflexion_results],
        )
        return BiasMetrics(
            bias_type=bias_type,
            baseline_metric=b_metric,
            selfhelp_metric=r_metric,
            n_baseline=sum(1 for r in matched_baseline if r.get("raw_answer", "").strip()),
            n_selfhelp=sum(1 for r in reflexion_results if r.raw_answer.strip()),
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _carry_forward(self, row: dict, step: int, bias_type: str) -> ReflexionStepResult:
        raw = row.get("raw_answer", "")
        return ReflexionStepResult(
            bias_type=bias_type,
            step=step,
            prompt_id=int(row["prompt_id"]),
            sub_condition=row["sub_condition"],
            original_prompt=row["original_prompt"],
            prior_raw_answer=raw,
            reflection_text="",
            raw_answer=raw,
            parsed_answer=row.get("parsed_answer", ""),
            usage=[],
        )

    def _reflect(
        self,
        question: str,
        prior_raw_answer: str,
        feedback: str,
        memory: MemoryStore,
    ):
        return self._reflection.generate(question, prior_raw_answer, feedback, memory)
