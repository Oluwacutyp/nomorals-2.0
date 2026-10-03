"""ContextEngine: token-budgeted assembly of model-call context.

The engine assembles the canonical sections — system, user profile, project,
mission state, artifacts (summaries + provenance links), tools (manifests),
history — then fits them to a :class:`ContextBudget` and compresses each
survivor.  Load-bearing content (acceptance criteria, required artifacts,
active mission state) is pinned and never dropped.

Every input is duck-typed: missions may be dataclasses or dicts, artifacts
may be :class:`nomorals.storage.artifacts.Artifact` or plain dicts, tools may
be a :class:`nomorals.tools.registry.ToolRegistry` or a plain list.  The
package never imports above its own layer; shapes are structural, not nominal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.text import approx_token_count
from .budget import ContextBudget
from .compress import Summarizer, compress_section
from .sections import Section, priority_for

__all__ = [
    "BuiltContext",
    "ContextEngine",
    "DEFAULT_PREAMBLE",
]

DEFAULT_PREAMBLE = (
    "You are Devon, a universal agent OS. Be direct, capable, and precise. "
    "Use the tools and context below; never invent facts the context does not support."
)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Attribute-or-key access for duck-typed mission/step/artifact shapes."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


@dataclass
class BuiltContext:
    """The result of one :meth:`ContextEngine.build` call."""

    text: str
    sections: list[Section] = field(default_factory=list)
    total_tokens: int = 0
    over_budget: bool = False
    budget_total: int = 0
    dropped: list[str] = field(default_factory=list)
    truncated: list[str] = field(default_factory=list)

    def report(self) -> dict[str, Any]:
        return {
            "total_tokens": self.total_tokens,
            "budget_total": self.budget_total,
            "over_budget": self.over_budget,
            "sections": [
                {
                    "name": s.name,
                    "tokens": s.tokens,
                    "priority": s.priority,
                    "load_bearing": s.load_bearing,
                    "truncated": s.truncated,
                    "dropped": s.dropped,
                }
                for s in self.sections
            ],
            "dropped": self.dropped,
            "truncated": self.truncated,
        }

    def section(self, name: str) -> Section | None:
        for section in self.sections:
            if section.name == name:
                return section
        return None


class ContextEngine:
    """Assemble token-budgeted context for a model call."""

    def __init__(
        self,
        budget: ContextBudget | None = None,
        *,
        summarizer: Summarizer | None = None,
        system_preamble: str | None = None,
    ) -> None:
        self.budget = budget or ContextBudget.default()
        self.summarizer = summarizer
        self.system_preamble = system_preamble or DEFAULT_PREAMBLE

    # ── public API ──────────────────────────────────────────────────────────
    def build(
        self,
        *,
        system: str | None = None,
        user_profile: Any = None,
        project: Any = None,
        mission: Any = None,
        artifacts: Any = None,
        tools: Any = None,
        history: Any = None,
        memory: Any = None,
        memory_query: str = "",
        budget: ContextBudget | None = None,
        extra_sections: list[Section] | None = None,
    ) -> BuiltContext:
        """Assemble, budget, compress, and render every requested section."""
        budget = budget or self.budget
        sections: list[Section] = [self._system_section(system)]
        if mission is not None:
            sections.append(self._mission_section(mission))
        if memory is not None:
            sections.append(self._memory_section(memory, memory_query))
        if artifacts:
            sections.append(self._artifacts_section(artifacts))
        if tools is not None:
            sections.append(self._tools_section(tools))
        if project is not None:
            sections.append(self._project_section(project))
        if user_profile is not None:
            sections.append(self._user_section(user_profile))
        if history:
            sections.append(self._history_section(history))
        if extra_sections:
            sections.extend(extra_sections)

        survivors = budget.fit(sections)
        for section in survivors:
            cap = budget.allocation_for(section.name)
            compress_section(
                section,
                cap,
                summarizer=self.summarizer,
                keep_tail=section.name == "history",
            )

        dropped = [s.name for s in sections if s.dropped]
        truncated = [s.name for s in survivors if s.truncated]
        text = "\n\n".join(s.render() for s in survivors)
        total = approx_token_count(text)
        return BuiltContext(
            text=text,
            sections=survivors,
            total_tokens=total,
            over_budget=budget.last_overrun > 0,
            budget_total=budget.total,
            dropped=dropped,
            truncated=truncated,
        )

    def build_step_prompt(self, mission: Any, step: Any, *, rich: bool = False,
                          **kwargs: Any) -> str:
        """Prompt for one mission step.

        Default (``rich=False``) is the legacy-compatible assembly the
        missions runner has always used — byte-identical output.  With
        ``rich=True`` the step is assembled through the full budgeted engine
        (mission state, acceptance criteria, artifacts, tools, history).
        """
        if not rich:
            return _legacy_step_prompt(mission, step)
        outputs = (_get(mission, "state") or {}).get("outputs") or {}
        history = [(str(name), str(value)) for name, value in outputs.items()]
        step_goal = _get(step, "goal") or _get(step, "name") or ""
        system = (
            f"Current step: {_get(step, 'name') or 'step'}\n"
            f"Step goal: {step_goal}\n"
            "Execute this step now, using the mission context below."
        )
        built = self.build(
            system=system, mission=mission, history=history, **kwargs,
        )
        return built.text

    # ── sections ────────────────────────────────────────────────────────────
    def _system_section(self, system: str | None) -> Section:
        text = system or self.system_preamble
        return Section(
            name="system",
            content=text,
            priority=priority_for("system"),
            load_bearing=True,
            keep=(text,),
        )

    def _mission_section(self, mission: Any) -> Section:
        goal = _get(mission, "goal") or ""
        mid = _get(mission, "id") or ""
        status = _get(mission, "status") or ""
        criteria = _get(mission, "acceptance_criteria") or _get(mission, "acceptance") or []
        if isinstance(criteria, str):
            criteria = [criteria]
        required = _get(mission, "required_artifacts") or []
        if isinstance(required, str):
            required = [required]
        state = _get(mission, "state") or {}
        outputs = state.get("outputs") or {} if isinstance(state, dict) else {}

        header = f"Mission {mid} [{status}]: {goal}".strip()
        lines = [header]
        keep: list[str] = [header]
        if criteria:
            crit_block = "Acceptance criteria:\n" + "\n".join(
                f"- {c}" for c in criteria)
            lines.append(crit_block)
            keep.append(crit_block)
        if required:
            req_line = "Required artifacts: " + ", ".join(
                r if str(r).startswith("artifact://") else f"artifact://{r}"
                for r in required)
            lines.append(req_line)
            keep.append(req_line)
        plan = state.get("plan") if isinstance(state, dict) else None
        if plan:
            done = sum(1 for p in plan if isinstance(p, dict) and p.get("done"))
            lines.append(f"Plan: {len(plan)} steps, {done} done")
        if outputs:
            lines.append(f"Prior outputs: {len(outputs)} produced")
            last = list(outputs.items())[-3:]
            for name, value in last:
                lines.append(f"- {name}: {str(value)[:200]}")
        return Section(
            name="mission",
            content="\n".join(lines),
            priority=priority_for("mission"),
            load_bearing=True,
            keep=tuple(keep),
            meta={"mission_id": mid, "status": status},
        )

    def _memory_section(self, memory: Any, query: str) -> Section:
        """Build a memory section from a MemoryManager.

        Uses MemoryManager.build_context() so there's one unified path
        for memory → context (not a parallel implementation).
        """
        try:
            text = memory.build_context(query or "current task")
        except Exception:  # noqa: BLE001 - memory is best-effort, never breaks context
            text = ""
        return Section(
            name="memory",
            content=text,
            priority=priority_for("memory"),
            load_bearing=False,
        )

    def _artifacts_section(self, artifacts: Any) -> Section:
        lines: list[str] = []
        for art in artifacts:
            uri = _get(art, "uri") or f"artifact://{_get(art, 'id') or '?'}"
            atype = _get(art, "type") or ""
            creator = _get(art, "creator") or ""
            meta = _get(art, "metadata") or {}
            summary = (
                _get(art, "summary")
                or (meta.get("summary") if isinstance(meta, dict) else "")
                or (meta.get("description") if isinstance(meta, dict) else "")
                or ""
            )
            line = f"- {uri} ({atype}) by {creator or 'unknown'}"
            if summary:
                line += f": {str(summary)[:240]}"
            prov = _get(art, "provenance")
            derived = _get(prov, "derived_from") if prov is not None else None
            if derived:
                line += f" [derived from: {', '.join(str(d) for d in derived)}]"
            lines.append(line)
        return Section(
            name="artifacts",
            content="Artifact summaries (full content via artifact:// URIs):\n"
                    + "\n".join(lines),
            priority=priority_for("artifacts"),
        )

    def _tools_section(self, tools: Any) -> Section:
        content = ""
        if hasattr(tools, "prompt_listing"):
            try:
                content = str(tools.prompt_listing())
            except Exception:  # noqa: BLE001 - fall back to schemas below
                content = ""
        if not content and hasattr(tools, "schemas"):
            try:
                schemas = tools.schemas()
            except Exception:  # noqa: BLE001
                schemas = []
            content = "\n".join(
                f"- {s.get('name')}: {s.get('description', '')}"
                for s in schemas if isinstance(s, dict))
        if not content and isinstance(tools, (list, tuple)):
            parts = []
            for tool in tools:
                if isinstance(tool, dict):
                    parts.append(
                        f"- {tool.get('name')}: {tool.get('description', '')}")
                elif isinstance(tool, str):
                    parts.append(f"- {tool}")
                else:
                    parts.append(
                        f"- {_get(tool, 'name')}: {_get(tool, 'description', '')}")
            content = "\n".join(parts)
        if not content:
            content = str(tools)
        return Section(
            name="tools",
            content="Available tools:\n" + content,
            priority=priority_for("tools"),
        )

    def _project_section(self, project: Any) -> Section:
        if isinstance(project, str):
            content = project
        elif isinstance(project, dict):
            lines = [f"Project {project.get('name', '')}: "
                     f"{project.get('goal', '')}".strip()]
            state = project.get("state") or {}
            if isinstance(state, dict):
                for key, value in list(state.items())[:8]:
                    lines.append(f"- {key}: {str(value)[:160]}")
            content = "\n".join(lines)
        else:
            name = _get(project, "name") or ""
            goal = _get(project, "goal") or ""
            content = f"Project {name}: {goal}".strip()
        return Section(
            name="project",
            content=content,
            priority=priority_for("project"),
        )

    def _user_section(self, user_profile: Any) -> Section:
        if isinstance(user_profile, str):
            content = user_profile
        elif isinstance(user_profile, dict):
            lines = ["User profile:"]
            for key, value in user_profile.items():
                lines.append(f"- {key}: {value}")
            content = "\n".join(lines)
        else:
            content = str(user_profile)
        return Section(
            name="user_profile",
            content=content,
            priority=priority_for("user_profile"),
        )

    def _history_section(self, history: Any) -> Section:
        lines = ["Recent history (oldest to newest):"]
        if isinstance(history, str):
            lines.append(history)
        else:
            for item in history:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    role, text = item
                    lines.append(f"- [{role}] {text}")
                elif isinstance(item, dict):
                    role = item.get("role", "?")
                    text = item.get("content", item.get("text", ""))
                    lines.append(f"- [{role}] {text}")
                else:
                    lines.append(f"- {item}")
        return Section(
            name="history",
            content="\n".join(lines),
            priority=priority_for("history"),
        )


def _legacy_step_prompt(mission: Any, step: Any) -> str:
    """The missions runner's original ad-hoc assembly, preserved exactly.

    ``Mission goal: ...`` / ``Current step: ...`` / ``Already produced: ...``
    — this is the compatibility path the runner's ``_step_prompt`` wrapper
    delegates to, so existing callers see byte-identical output.
    """
    state = _get(mission, "state") or {}
    outputs = state.get("outputs") or {}
    prior = "\n".join(
        f"- {name}: {str(value)[:300]}" for name, value in list(outputs.items())[-4:]
    )
    parts = [
        f"Mission goal: {_get(mission, 'goal')}",
        f"Current step: {_get(step, 'goal') or _get(step, 'name')}",
    ]
    if prior:
        parts.append(f"Already produced:\n{prior}")
    return "\n\n".join(parts)
