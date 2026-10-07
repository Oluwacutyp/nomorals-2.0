"""Loop working memory — compact state for the ReAct loop.

Keeps conversation history, tool results, and the current plan in a form
small enough for weaker models. Summarizes aggressively: only the latest
N exchanges are kept verbatim, older ones are compressed to one-line
summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepRecord:
    """One think → act → observe cycle."""

    step: int
    thought: str
    action: str  # "tool" | "respond" | "ask"
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    failed: bool = False


@dataclass
class LoopMemory:
    """Working memory for one agentic loop run.

    Attributes:
        user_message: The original owner message that started the loop.
        history: Prior conversation turns (role, text) for context.
        steps: The think/act/observe records so far this run.
        plan: The model's current plan, updated as it learns.
        max_history_turns: How many recent turns to keep verbatim.
    """

    user_message: str = ""
    history: list[tuple[str, str]] = field(default_factory=list)
    steps: list[StepRecord] = field(default_factory=list)
    plan: str = ""
    max_history_turns: int = 6
    max_observation_chars: int = 2000

    # ── recording ────────────────────────────────────────────────────

    def record_step(self, record: StepRecord) -> None:
        self.steps.append(record)

    def set_plan(self, plan: str) -> None:
        self.plan = (plan or "").strip()[:500]

    def add_history(self, role: str, text: str) -> None:
        self.history.append((role, text))

    # ── rendering for the model ──────────────────────────────────────

    def _recent_history(self) -> list[tuple[str, str]]:
        return self.history[-self.max_history_turns :]

    def _compact_observation(self, text: str) -> str:
        text = str(text)
        if len(text) > self.max_observation_chars:
            return text[: self.max_observation_chars] + "…[truncated]"
        return text

    def render(self) -> str:
        """Render the full loop state as prompt context.

        Structured for weaker models: clear sections, no ambiguity about
        what each part is.
        """
        lines = [f"USER REQUEST: {self.user_message}", ""]

        recent = self._recent_history()
        if recent:
            lines.append("RECENT CONVERSATION:")
            for role, text in recent:
                snippet = text[:300] + ("…" if len(text) > 300 else "")
                lines.append(f"  {role}: {snippet}")
            lines.append("")

        if self.plan:
            lines.append(f"CURRENT PLAN: {self.plan}")
            lines.append("")

        if self.steps:
            lines.append("WHAT YOU HAVE DONE SO FAR:")
            for s in self.steps:
                status = "FAILED" if s.failed else "ok"
                lines.append(f"  Step {s.step} [{status}] thought: {s.thought[:200]}")
                if s.action == "tool":
                    lines.append(
                        f"    → called {s.tool_name} with {s.tool_args}"
                    )
                    obs = self._compact_observation(s.observation)
                    lines.append(f"    → result: {obs}")
                elif s.action == "ask":
                    lines.append(f"    → asked user: {s.thought[:200]}")
            lines.append("")
            # Failure hints — help weaker models recover instead of looping
            failed = [s for s in self.steps if s.failed]
            if failed:
                tried = ", ".join(s.tool_name for s in failed[-3:])
                lines.append(
                    f"NOTE: these tools already failed ({tried}). "
                    "Try a DIFFERENT tool or approach — do not retry the same call."
                )
                lines.append("")

        return "\n".join(lines)

    def step_count(self) -> int:
        return len(self.steps)

    def failed_tools(self) -> list[str]:
        """Names of tools that failed this run (to avoid blind retries)."""
        return [s.tool_name for s in self.steps if s.failed and s.tool_name]
