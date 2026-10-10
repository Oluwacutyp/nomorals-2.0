"""ContextEngine: token-budgeted assembly of model-call context.

The engine assembles the canonical sections — system, user profile, project,
mission state, artifacts (summaries + provenance links), tools (manifests),
history — then fits them to a :class:`ContextBudget` and compresses each
survivor.  Load-bearing content (acceptance criteria, required artifacts,
active mission state) is pinned and never dropped.

Assembly ordering modes (``ordering``):

* ``"priority"`` — canonical order (system, mission, memory, artifacts,
  tools, project, user, history).  The default; byte-stable with history.
* ``"cache"`` — stable sections first, volatile tail last.  Keeps the
  provider prompt-cache prefix byte-identical across calls so cache hits
  survive turn-to-turn growth.
* ``"rot"`` — load-bearing sections first, then the rest by priority, with
  the pinned key facts echoed in a compact footer.  Mitigates the
  "lost in the middle" attention decay (primacy + recency dominate).

Every input is duck-typed: missions may be dataclasses or dicts, artifacts
may be :class:`nomorals.storage.artifacts.Artifact` or plain dicts, tools may
be a :class:`nomorals.tools.registry.ToolRegistry` or a plain list.  The
package never imports above its own layer; shapes are structural, not nominal.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .budget import ContextBudget
from .cache import CachePlan, cache_plan, stable_fingerprint
from .compress import Summarizer, compress_section, extractive_summary
from .sections import Section, priority_for, set_token_counter, token_count, try_tiktoken_counter

__all__ = [
    "BuiltContext",
    "ContextEngine",
    "DEFAULT_PREAMBLE",
    "ORDERING_MODES",
]

DEFAULT_PREAMBLE = (
    "You are Devon, a universal agent OS. Be direct, capable, and precise. "
    "Use the tools and context below; never invent facts the context does not support."
)

#: Assembly ordering modes accepted by :meth:`ContextEngine.build`.
ORDERING_MODES = ("priority", "cache", "rot")


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Attribute-or-key access for duck-typed mission/step/artifact shapes."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _sparkbar(ratio: float, width: int = 16) -> str:
    filled = max(0, min(width, int(round(ratio * width))))
    return "█" * filled + "░" * (width - filled)


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

    #: Free-form build metadata: ordering mode, cache fingerprints,
    #: compaction boundary info, rot analysis, ...
    meta: dict[str, Any] = field(default_factory=dict)

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
                    "volatile": s.volatile,
                    "truncated": s.truncated,
                    "dropped": s.dropped,
                    "fingerprint": s.fingerprint(),
                }
                for s in self.sections
            ],
            "dropped": self.dropped,
            "truncated": self.truncated,
            "meta": self.meta,
        }

    def section(self, name: str) -> Section | None:
        for section in self.sections:
            if section.name == name:
                return section
        return None

    def render_dashboard(self, *, style: str = "plain") -> str:
        """Styled budget dashboard for this built context.

        ``style`` is ``"plain"`` (ASCII) or ``"fancy"`` (unicode bars and
        markers; still no color codes, safe on any terminal).
        """
        fancy = style == "fancy"
        lines = []
        title = "Context dashboard" if not fancy else "◆ Context dashboard"
        lines.append(title)
        status = "OVER BUDGET" if self.over_budget else "within budget"
        lines.append(
            f"  tokens: ~{self.total_tokens} / {self.budget_total} ({status})"
        )
        ordering = self.meta.get("ordering", "?")
        lines.append(f"  ordering: {ordering}")
        lines.append("  " + "-" * 56)
        peak = max((s.tokens for s in self.sections), default=1)
        for section in self.sections:
            ratio = section.tokens / peak if peak else 0.0
            bar = _sparkbar(ratio) if fancy else "#" * max(1, int(round(ratio * 16)))
            flags: list[str] = []
            if section.load_bearing:
                flags.append("pinned")
            if section.truncated:
                flags.append("truncated")
            if section.volatile:
                flags.append("volatile")
            flag_text = f" [{', '.join(flags)}]" if flags else ""
            lines.append(
                f"  {section.name:<14} {bar} ~{section.tokens:>6}{flag_text}"
            )
        if self.dropped:
            lines.append(f"  dropped: {', '.join(self.dropped)}")
        rot = self.meta.get("rot") or {}
        buried = rot.get("buried") or []
        if buried:
            lines.append(f"  buried load-bearing: {', '.join(buried)}")
        fp = self.meta.get("stable_fingerprint")
        if fp:
            lines.append(f"  stable prefix fingerprint: {fp}")
        return "\n".join(lines)


class ContextEngine:
    """Assemble token-budgeted context for a model call."""

    def __init__(
        self,
        budget: ContextBudget | None = None,
        *,
        summarizer: Summarizer | None = None,
        system_preamble: str | None = None,
        token_counter: Callable[[str], int] | None = None,
        use_tiktoken: bool = False,
        ordering: str = "priority",
    ) -> None:
        self.budget = budget or ContextBudget.default()
        self.summarizer = summarizer
        self.system_preamble = system_preamble or DEFAULT_PREAMBLE
        if token_counter is not None:
            set_token_counter(token_counter)
        elif use_tiktoken:
            exact = try_tiktoken_counter()
            if exact is not None:
                set_token_counter(exact)
        if ordering not in ORDERING_MODES:
            raise ValueError(
                f"unknown ordering {ordering!r} (modes: {', '.join(ORDERING_MODES)})")
        self.ordering = ordering

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
        ordering: str | None = None,
        tagged: bool = False,
    ) -> BuiltContext:
        """Assemble, budget, compress, and render every requested section."""
        budget = budget or self.budget
        ordering = ordering or self.ordering
        if ordering not in ORDERING_MODES:
            raise ValueError(
                f"unknown ordering {ordering!r} (modes: {', '.join(ORDERING_MODES)})")

        sections = self._assemble_sections(
            system=system,
            user_profile=user_profile,
            project=project,
            mission=mission,
            artifacts=artifacts,
            tools=tools,
            history=history,
            memory=memory,
            memory_query=memory_query,
            extra_sections=extra_sections,
        )
        ordered = self._apply_ordering(sections, ordering)

        survivors = budget.fit(ordered)
        for section in survivors:
            cap = budget.allocation_for(section.name)
            compress_section(
                section,
                cap,
                summarizer=self.summarizer,
                keep_tail=section.name == "history",
            )

        dropped = [s.name for s in ordered if s.dropped]
        truncated = [s.name for s in survivors if s.truncated]
        if tagged:
            text = "\n\n".join(s.render_tagged() for s in survivors)
        else:
            text = "\n\n".join(s.render() for s in survivors)
        if ordering == "rot":
            echo = self._echo_footer(survivors)
            if echo:
                text += echo
        total = token_count(text)
        built = BuiltContext(
            text=text,
            sections=survivors,
            total_tokens=total,
            over_budget=budget.last_overrun > 0,
            budget_total=budget.total,
            dropped=dropped,
            truncated=truncated,
        )
        built.meta["ordering"] = ordering
        built.meta["tagged"] = tagged
        built.meta["stable_fingerprint"] = stable_fingerprint(survivors)
        built.meta["cache_plan"] = cache_plan(survivors).to_dict()
        built.meta["rot"] = self.rot_report_for(survivors, text)
        return built

    def preview(
        self,
        *,
        budget: ContextBudget | None = None,
        **kwargs: Any,
    ) -> str:
        """Show the budget table and cache plan *before* building.

        Assembles sections without fitting or compressing — the operator
        sees where tokens would go and whether the stable prefix is
        cache-safe, then calls :meth:`build` for real.
        """
        budget = budget or self.budget
        sections = self._assemble_sections(**kwargs)
        ordered = self._apply_ordering(sections, self.ordering)
        parts = [budget.render_table(ordered, style="fancy"), "",
                 cache_plan(ordered).render(style="fancy")]
        return "\n".join(parts)

    def cache_plan_for(self, built: BuiltContext) -> CachePlan:
        """Compute the prompt-cache plan for an already-built context."""
        return cache_plan(built.sections)

    def rot_report_for(
        self, sections: list[Section], text: str | None = None
    ) -> dict[str, Any]:
        """Lost-in-the-middle analysis for assembled sections.

        For each load-bearing section, reports the percentile position of
        its content in the rendered text (0.0 = start, 1.0 = end).
        Sections landing in the 0.25–0.75 band are flagged ``buried`` —
        that is where attention decays.
        """
        rendered = text if text is not None else "\n\n".join(
            s.render() for s in sections if not s.dropped)
        total = max(1, len(rendered))
        positions: dict[str, float] = {}
        cursor = 0
        for section in sections:
            if section.dropped:
                continue
            block = section.render()
            idx = rendered.find(block, cursor)
            if idx >= 0:
                cursor = idx + len(block)
                mid = idx + len(block) / 2
                positions[section.name] = round(mid / total, 3)
        buried = [
            name for name, pos in positions.items()
            if 0.25 <= pos <= 0.75
            and (next((s for s in sections if s.name == name), None)
                 or Section(name=name)).load_bearing
        ]
        return {"positions": positions, "buried": buried}

    def compact_history(
        self,
        history: Any,
        *,
        keep_last_n: int = 6,
        focus: str = "",
        boundary: dict[str, Any] | None = None,
    ) -> tuple[list[tuple[str, str]], dict[str, Any]]:
        """Incrementally compact a history list with a boundary marker.

        Keeps the most recent ``keep_last_n`` entries verbatim and condenses
        the older span into one summary entry via the engine's summarizer
        (or :func:`extractive_summary` by default).  ``boundary`` is the
        ``boundary_info`` returned by a previous call: entries it already
        covered are never re-summarized — the new compaction resumes after
        the previous boundary, exactly like Claude Code's compact boundary.

        Returns ``(compacted_entries, boundary_info)``; the entries feed
        straight back into :meth:`build` as ``history=``.

        The boundary marker embedded in the returned entries is
        authoritative: a later call scans for the last ``[compact boundary]``
        marker and resumes after it, so an already-summarized span is never
        re-summarized.  ``boundary_info["end_index"]`` is the marker's
        position in the *returned* list.
        """
        entries = self._normalize_history(history)
        start = int((boundary or {}).get("end_index", 0) or 0)
        # An embedded boundary marker is authoritative: resume after it.
        for idx, (role, text) in enumerate(entries):
            if role == "system" and "[compact boundary]" in text:
                start = idx + 1
        fresh = entries[max(0, start):]
        if len(fresh) <= keep_last_n:
            info = {
                "end_index": len(entries),
                "compacted": False,
                "summary_tokens": 0,
                "at": time.time(),
            }
            return entries, info

        older = fresh[: len(fresh) - keep_last_n]
        recent = fresh[len(fresh) - keep_last_n:]
        older_text = "\n".join(f"[{role}] {text}" for role, text in older)
        room = max(64, sum(token_count(t) for _, t in older) // 4)
        summarizer = self.summarizer or extractive_summary
        try:
            summary = summarizer(older_text, room)
        except Exception:  # noqa: BLE001 - summarizer must never break this
            summary = extractive_summary(older_text, room)

        header = (
            f"[compact boundary] {len(older)} older entr"
            f"{'y' if len(older) == 1 else 'ies'} summarized "
            f"(~{token_count(summary)} tokens)"
        )
        if focus:
            header += f"; focus: {focus}"
        marker = ("system", f"{header}\n{summary}")
        compacted = entries[: max(0, start)] + [marker] + recent
        info = {
            "end_index": len(compacted) - len(recent),
            "compacted": True,
            "summary_tokens": token_count(summary),
            "entries_summarized": len(older),
            "focus": focus,
            "at": time.time(),
        }
        return compacted, info

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

    # ── assembly ────────────────────────────────────────────────────────────
    def _assemble_sections(
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
        extra_sections: list[Section] | None = None,
    ) -> list[Section]:
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
        return sections

    def _apply_ordering(
        self, sections: list[Section], ordering: str
    ) -> list[Section]:
        if ordering == "cache":
            # Stable prefix first (canonical order), volatile tail last:
            # byte-identical prefixes across calls keep prompt caches hot.
            stable = [s for s in sections if not s.volatile]
            volatile = [s for s in sections if s.volatile]
            return stable + volatile
        if ordering == "rot":
            # Load-bearing first (primacy), then the rest by priority;
            # the pinned key facts are echoed again at the tail (recency).
            bearing = [s for s in sections if s.load_bearing]
            rest = sorted(
                (s for s in sections if not s.load_bearing),
                key=lambda s: s.priority,
                reverse=True,
            )
            return bearing + rest
        return list(sections)  # "priority": canonical assembly order

    def _echo_footer(self, survivors: list[Section]) -> str:
        """Compact recency echo of pinned load-bearing facts."""
        pins: list[str] = []
        seen: set[str] = set()
        for section in survivors:
            if not section.load_bearing or section.dropped:
                continue
            for keep_str in section.keep:
                keep_str = keep_str.strip()
                if keep_str and keep_str not in seen:
                    seen.add(keep_str)
                    pins.append(keep_str)
        if not pins:
            return ""
        echo = "\n".join(pins)
        # Keep the echo cheap: cap at ~200 tokens.
        words = echo.split()
        if token_count(echo) > 200:
            echo = " ".join(words[:150]) + " …"
        return "\n\n## Key facts (echo — do not lose sight of these)\n" + echo

    @staticmethod
    def _normalize_history(history: Any) -> list[tuple[str, str]]:
        if isinstance(history, str):
            return [("history", history)]
        entries: list[tuple[str, str]] = []
        for item in history:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                entries.append((str(item[0]), str(item[1])))
            elif isinstance(item, dict):
                role = str(item.get("role", "?"))
                text = str(item.get("content", item.get("text", "")))
                entries.append((role, text))
            else:
                entries.append(("history", str(item)))
        return entries

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
            volatile=True,
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
