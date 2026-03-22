from dataclasses import dataclass, field


@dataclass
class MemoryStore:
    max_entries: int = 3
    entries: list[str] = field(default_factory=list)

    def add(self, reflection: str) -> None:
        self.entries.append(reflection)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]

    def format_for_prompt(self) -> str:
        if not self.entries:
            return "No previous reflections."
        return "\n".join(f"Reflection {i + 1}: {r}" for i, r in enumerate(self.entries))

    def is_empty(self) -> bool:
        return len(self.entries) == 0

    def clear(self) -> None:
        self.entries = []
