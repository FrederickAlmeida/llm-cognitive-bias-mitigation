from .actor import Trajectory
from .evaluator import EvalResult
from .llm.base import LLMClient, LLMResponse
from .memory import MemoryStore


class SelfReflection:
    def __init__(self, llm: LLMClient, prompt_loader) -> None:
        self.llm = llm
        self.prompt_loader = prompt_loader

    def generate(
        self,
        question: str,
        trajectory: Trajectory,
        eval_result: EvalResult,
        memory: MemoryStore,
    ) -> tuple[str, LLMResponse]:
        system = self.prompt_loader.render("self_reflection", "system", {})
        user = self.prompt_loader.render(
            "self_reflection",
            "user",
            {
                "question": question,
                "trajectory": trajectory.format_for_reflection(),
                "feedback": eval_result.feedback,
                "memory": memory.format_for_prompt(),
            },
        )
        response = self.llm.complete(system, user)
        return response.content.strip(), response
