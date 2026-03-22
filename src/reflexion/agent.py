from __future__ import annotations

from dataclasses import dataclass, field
from string import Template

import yaml

from .actor import BaseActor, Trajectory
from .evaluator import BaseEvaluator, EvalResult
from .llm.base import TokenUsage
from .memory import MemoryStore
from .reflection import SelfReflection


# ── Prompt Loader ─────────────────────────────────────────────────────────────

class PromptLoader:
    def __init__(self, prompts_path: str) -> None:
        with open(prompts_path, encoding="utf-8") as f:
            self._prompts: dict = yaml.safe_load(f)

    def render(self, prompt_key: str, turn: str, variables: dict) -> str:
        """Render a prompt template. `turn` is 'system' or 'user'."""
        try:
            template_str = self._prompts[prompt_key][turn]
        except KeyError:
            raise KeyError(f"Prompt '{prompt_key}.{turn}' not found in prompts YAML.")
        return Template(template_str).safe_substitute(variables)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ReflexionResult:
    question: str
    final_trajectory: Trajectory
    final_eval: EvalResult
    reflections: list[str]
    trials_used: int
    solved: bool
    usage: list[TokenUsage] = field(default_factory=list)
    total_cost_usd: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Solved:      {self.solved}",
            f"Trials used: {self.trials_used}",
            f"Final answer: {self.final_trajectory.final_answer}",
            f"Total cost:  ${self.total_cost_usd:.6f} USD",
            f"Token usage: " + ", ".join(
                f"{u.model} in={u.input_tokens} cached={u.cached_input_tokens} out={u.output_tokens}"
                for u in self.usage
            ),
        ]
        for i, r in enumerate(self.reflections, 1):
            lines.append(f"\n--- Reflection {i} ---\n{r}")
        return "\n".join(lines)


# ── Agent ─────────────────────────────────────────────────────────────────────

class ReflexionAgent:
    def __init__(
        self,
        actor: BaseActor,
        evaluator: BaseEvaluator,
        self_reflection: SelfReflection,
        memory: MemoryStore,
        max_trials: int = 3,
    ) -> None:
        self.actor = actor
        self.evaluator = evaluator
        self.self_reflection = self_reflection
        self.memory = memory
        self.max_trials = max_trials

    def run(self, question: str, ground_truth: str) -> ReflexionResult:
        self.memory.clear()
        all_usage: list[TokenUsage] = []
        reflections: list[str] = []
        trajectory = Trajectory(question=question)
        result = EvalResult(passed=False, score=0.0)

        for trial in range(self.max_trials):
            trajectory = self.actor.run(question, self.memory)
            all_usage.extend(trajectory.usage)

            result = self.evaluator.score(trajectory, ground_truth)
            if result.usage:
                all_usage.append(result.usage)

            if result.passed:
                return _build_result(
                    question, trajectory, result, reflections,
                    trials_used=trial + 1, solved=True, usage=all_usage,
                )

            reflection_text, refl_response = self.self_reflection.generate(
                question, trajectory, result, self.memory
            )
            all_usage.append(refl_response.usage)
            self.memory.add(reflection_text)
            reflections.append(reflection_text)

        return _build_result(
            question, trajectory, result, reflections,
            trials_used=self.max_trials, solved=False, usage=all_usage,
        )


def _build_result(
    question: str,
    trajectory: Trajectory,
    result: EvalResult,
    reflections: list[str],
    trials_used: int,
    solved: bool,
    usage: list[TokenUsage],
) -> ReflexionResult:
    total_cost = sum(u.cost_usd for u in usage)
    return ReflexionResult(
        question=question,
        final_trajectory=trajectory,
        final_eval=result,
        reflections=reflections,
        trials_used=trials_used,
        solved=solved,
        usage=usage,
        total_cost_usd=total_cost,
    )
