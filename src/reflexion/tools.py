from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    observation: str
    raw: Any = None


class Tool(ABC):
    name: str
    description: str

    @abstractmethod
    def run(self, input: str) -> ToolResult:
        ...


class FinishTool(Tool):
    """Special tool that signals the agent has a final answer. Handled by the actor loop."""

    name = "Finish"
    description = "Signal that you have reached a final answer. Input: your final answer string."

    def run(self, input: str) -> ToolResult:
        return ToolResult(observation=input, raw=input)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self.register(FinishTool())

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered. Available: {list(self._tools)}")
        return self._tools[name]

    def descriptions_for_prompt(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values())

    def has(self, name: str) -> bool:
        return name in self._tools
