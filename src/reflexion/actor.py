import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .llm.base import LLMClient, TokenUsage
from .memory import MemoryStore
from .tools import ToolRegistry


@dataclass
class Trajectory:
    question: str
    steps: list[dict] = field(default_factory=list)  # {"type": str, "content": str}
    final_answer: str = ""
    usage: list[TokenUsage] = field(default_factory=list)

    def append_step(self, step_type: str, content: str) -> None:
        self.steps.append({"type": step_type, "content": content})

    def format_for_reflection(self) -> str:
        lines = [f"Question: {self.question}"]
        for step in self.steps:
            lines.append(f"{step['type'].upper()}: {step['content']}")
        lines.append(f"Final Answer: {self.final_answer}")
        return "\n".join(lines)


class BaseActor(ABC):
    def __init__(self, llm: LLMClient, prompt_loader) -> None:
        self.llm = llm
        self.prompt_loader = prompt_loader

    @abstractmethod
    def run(self, question: str, memory: MemoryStore) -> Trajectory:
        ...


class CoTActor(BaseActor):
    def run(self, question: str, memory: MemoryStore) -> Trajectory:
        system = self.prompt_loader.render("actor_cot", "system", {"memory": memory.format_for_prompt()})
        user = self.prompt_loader.render("actor_cot", "user", {"question": question})

        response = self.llm.complete(system, user)

        trajectory = Trajectory(question=question)
        trajectory.append_step("answer", response.content)
        trajectory.usage.append(response.usage)

        # Extract the "Answer: ..." line if present, otherwise use the full response
        answer = _extract_answer(response.content)
        trajectory.final_answer = answer

        return trajectory


class ReActActor(BaseActor):
    def __init__(self, llm: LLMClient, prompt_loader, tool_registry: ToolRegistry, max_steps: int = 10) -> None:
        super().__init__(llm, prompt_loader)
        self.tools = tool_registry
        self.max_steps = max_steps

    def run(self, question: str, memory: MemoryStore) -> Trajectory:
        trajectory = Trajectory(question=question)
        context = ""

        for _ in range(self.max_steps):
            system = self.prompt_loader.render(
                "actor_react",
                "system",
                {
                    "tool_descriptions": self.tools.descriptions_for_prompt(),
                    "memory": memory.format_for_prompt(),
                },
            )
            user = self.prompt_loader.render(
                "actor_react",
                "user",
                {"question": question, "context": context},
            )

            response = self.llm.complete(system, user)
            trajectory.usage.append(response.usage)
            raw = response.content.strip()

            # Parse Thought and Action from LLM output
            thought, action_name, action_input = _parse_react_output(raw)

            if thought:
                trajectory.append_step("thought", thought)
                context += f"\nThought: {thought}"

            if action_name is None:
                # Parse failed — inject corrective observation so the LLM can retry
                obs = "Parse error: please use the exact format 'Action: ToolName[input]'."
                trajectory.append_step("observation", obs)
                context += f"\nObservation: {obs}"
                continue

            trajectory.append_step("action", f"{action_name}[{action_input}]")

            if action_name == "Finish":
                trajectory.final_answer = action_input
                break

            # Execute the tool
            try:
                result = self.tools.get(action_name).run(action_input)
                obs = result.observation
            except KeyError as e:
                obs = str(e)

            trajectory.append_step("observation", obs)
            context += f"\nAction: {action_name}[{action_input}]\nObservation: {obs}"

        return trajectory


# ── helpers ──────────────────────────────────────────────────────────────────

def _extract_answer(text: str) -> str:
    for line in text.splitlines():
        if line.strip().lower().startswith("answer:"):
            return line.split(":", 1)[1].strip()
    return text.strip()


_ACTION_RE = re.compile(r"Action:\s*(\w+)\[(.+?)\]", re.DOTALL)
_THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?=\nAction:|\Z)", re.DOTALL)


def _parse_react_output(text: str) -> tuple[str | None, str | None, str]:
    thought_match = _THOUGHT_RE.search(text)
    thought = thought_match.group(1).strip() if thought_match else None

    action_match = _ACTION_RE.search(text)
    if not action_match:
        return thought, None, ""

    return thought, action_match.group(1), action_match.group(2).strip()


# ── factory ──────────────────────────────────────────────────────────────────

def make_actor(
    strategy: str,
    llm: LLMClient,
    prompt_loader,
    tool_registry: ToolRegistry | None = None,
    config: dict | None = None,
) -> BaseActor:
    config = config or {}
    if strategy == "cot":
        return CoTActor(llm, prompt_loader)
    elif strategy == "react":
        registry = tool_registry or ToolRegistry()
        max_steps = config.get("react_max_steps", 10)
        return ReActActor(llm, prompt_loader, registry, max_steps=max_steps)
    raise ValueError(f"Unknown actor strategy: {strategy!r}. Choose 'cot' or 'react'.")
