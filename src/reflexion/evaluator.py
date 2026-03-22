from abc import ABC, abstractmethod
from dataclasses import dataclass

from .actor import Trajectory
from .llm.base import LLMClient, TokenUsage


@dataclass
class EvalResult:
    passed: bool
    score: float  # 0.0–1.0
    feedback: str = ""
    usage: TokenUsage | None = None  # only set by LLM-based evaluators


class BaseEvaluator(ABC):
    @abstractmethod
    def score(self, trajectory: Trajectory, ground_truth: str) -> EvalResult:
        ...


class ExactMatchEvaluator(BaseEvaluator):
    """Normalises both strings (lowercase, strip) and compares them."""

    def score(self, trajectory: Trajectory, ground_truth: str) -> EvalResult:
        pred = trajectory.final_answer.strip().lower()
        gt = ground_truth.strip().lower()
        passed = pred == gt
        return EvalResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            feedback=f"Expected '{ground_truth}', got '{trajectory.final_answer}'." if not passed else "Correct.",
        )


class LLMJudgeEvaluator(BaseEvaluator):
    """Uses an LLM to judge whether the answer is correct or semantically equivalent."""

    def __init__(self, llm: LLMClient, prompt_loader) -> None:
        self.llm = llm
        self.prompt_loader = prompt_loader

    def score(self, trajectory: Trajectory, ground_truth: str) -> EvalResult:
        system = self.prompt_loader.render("evaluator_judge", "system", {})
        user = self.prompt_loader.render(
            "evaluator_judge",
            "user",
            {
                "question": trajectory.question,
                "answer": trajectory.final_answer,
                "ground_truth": ground_truth,
            },
        )
        response = self.llm.complete(system, user)
        content = response.content.strip()

        passed = content.upper().startswith("PASS")
        return EvalResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            feedback=content,
            usage=response.usage,
        )


# ── factory ──────────────────────────────────────────────────────────────────

def make_evaluator(
    evaluator_type: str,
    llm: LLMClient | None = None,
    prompt_loader=None,
) -> BaseEvaluator:
    if evaluator_type == "exact_match":
        return ExactMatchEvaluator()
    elif evaluator_type == "llm_judge":
        if llm is None or prompt_loader is None:
            raise ValueError("LLMJudgeEvaluator requires an llm and prompt_loader.")
        return LLMJudgeEvaluator(llm, prompt_loader)
    raise ValueError(f"Unknown evaluator type: {evaluator_type!r}. Choose 'exact_match' or 'llm_judge'.")
