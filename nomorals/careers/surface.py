"""Career pipeline surface: the board and reference resolution.

The presentation layer over :class:`~nomorals.careers.store.CareerStore`,
shared by the chat ``/careers`` surface (agents/partner) and the
``/goals list careers`` drill-down.  Rendering is pure text over a
store the caller supplies.

The board's honesty rule (the class system's OFF-vs-BROKEN rule
applied to a pipeline): an application that needs attention is
CALLED OUT — a past-due next action and a stale application get
their own section at the top and markers inline, never a silent
line in a list.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

from .store import (
    ACTIVE_STAGES,
    STAGES,
    TERMINAL_STAGES,
    Application,
    CareerStore,
)

__all__ = [
    "render_pipeline_board",
    "render_stages",
    "resolve_application",
    "unknown_application_text",
]


def _fmt_day(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d")


def _days(n: float) -> str:
    days = int(n // 86400)
    if days <= 0:
        return "today"
    if days == 1:
        return "1d"
    return f"{days}d"


def resolve_application(
    store: CareerStore, ref: str,
) -> tuple[Application | None, list[Application]]:
    """Resolve a chat application reference to one application.

    Accepts an application id (exact or unique prefix), ``#n``
    (1-based into the board order — the numbers the board prints),
    or a unique substring of "company role".  Returns
    ``(application, [])`` on a hit, ``(None, candidates)`` when the
    reference is ambiguous or matches nothing (mirrors the goal
    surface's resolve contract).
    """
    ref = (ref or "").strip()
    apps = store.list()
    if not ref:
        return None, apps
    exact = store.get(ref)
    if exact is not None:
        return exact, []
    if ref.startswith("#") and ref[1:].isdigit():
        idx = int(ref[1:]) - 1
        if 0 <= idx < len(apps):
            return apps[idx], []
        return None, apps
    by_prefix = [a for a in apps if a.id.startswith(ref)]
    if len(by_prefix) == 1:
        return by_prefix[0], []
    needle = ref.casefold()
    by_text = [a for a in apps
               if needle in f"{a.company} {a.role}".casefold()]
    if len(by_text) == 1:
        return by_text[0], []
    if len(by_text) > 1:
        return None, by_text
    return None, apps


def unknown_application_text(
    ref: str, candidates: list[Application],
) -> str:
    """The honest unknown-ref reply: name the miss, show what exists."""
    lines = [f"no single application matches {ref!r}."]
    if candidates:
        lines.append("applications on file:")
        for i, app in enumerate(candidates[:10], 1):
            lines.append(
                f"  #{i} {app.company} — {app.role} ({app.stage}) · {app.id}")
    else:
        lines.append(
            "no applications yet — /careers add <company> | <role> "
            "starts the pipeline.")
    return "\n".join(lines)


def _attention_lines(store: CareerStore, *, now: float) -> list[str]:
    """The call-outs: overdue next actions first, then stale."""
    lines: list[str] = []
    overdue = store.overdue_actions(now=now)
    stale = store.stale(now=now)
    if not overdue and not stale:
        return lines
    lines.append("⚠️ **Needs attention:**")
    for app in overdue:
        assert app.next_action_due is not None
        lines.append(
            f"  ⛔ {app.company} — {app.role}: next action "
            f"“{app.next_action}” was due {_fmt_day(app.next_action_due)} "
            f"({_days(now - app.next_action_due)} overdue)")
    for app in stale:
        lines.append(
            f"  ⚠ {app.company} — {app.role}: untouched for "
            f"{_days(now - app.updated_at)}, no next action set")
    lines.append("")
    return lines


def _app_line(app: Application, index: int, *, now: float) -> str:
    line = (f"  #{index} **{app.company}** — {app.role} · "
            f"{_days(now - app.stage_since())} in stage")
    if app.next_action:
        due = (f" (due {_fmt_day(app.next_action_due)})"
               if app.next_action_due is not None else "")
        flag = ""
        if (app.next_action_due is not None
                and app.next_action_due < now):
            flag = " ⛔ OVERDUE"
        line += f" · next: {app.next_action}{due}{flag}"
    if app.cv_artifact_id:
        line += " · 📎 CV attached"
    line += f" · `{app.id}`"
    return line


def render_pipeline_board(
    store: CareerStore, *, now: float | None = None,
) -> str:
    """The pipeline board: attention call-outs, then stage sections.

    Applications are numbered in board order (filed order) — the
    ``#n`` the other verbs accept.  An empty pipeline renders a
    designed empty state, never a bare blank.
    """
    now = time.time() if now is None else now
    apps = store.list()
    if not apps:
        return (
            "💼 **Career pipeline**\n"
            "  🗒 no applications yet — this is where the job hunt "
            "lives: applications by stage, each with its next action.\n"
            "  start: /careers add <company> | <role>")
    counts = store.funnel_counts()
    funnel = " · ".join(
        f"{stage} {counts[stage]}" for stage in STAGES if counts[stage])
    lines = [f"💼 **Career pipeline** — {len(apps)} application(s)",
             f"  {funnel}", ""]
    lines.extend(_attention_lines(store, now=now))
    by_stage: dict[str, list[tuple[int, Application]]] = {}
    for i, app in enumerate(apps, 1):
        by_stage.setdefault(app.stage, []).append((i, app))
    for stage in ACTIVE_STAGES:
        group = by_stage.get(stage)
        if not group:
            continue
        lines.append(f"**{stage.title()}**")
        for i, app in group:
            lines.append(_app_line(app, i, now=now))
        lines.append("")
    closed = [(i, a) for stage in TERMINAL_STAGES
              for i, a in by_stage.get(stage, [])]
    if closed:
        lines.append("**Closed**")
        for i, app in sorted(closed):
            mark = {"accepted": "✅", "rejected": "✖",
                    "withdrawn": "↩"}[app.stage]
            lines.append(
                f"  #{i} {mark} {app.company} — {app.role} ({app.stage})"
                f" · `{app.id}`")
        lines.append("")
    return "\n".join(lines).rstrip()


def render_stages() -> str:
    """The stage vocabulary as text (the /careers stages verb)."""
    from .store import STAGE_PROGRESS

    lines = ["💼 **Pipeline stages** (in funnel order):"]
    for stage in STAGES:
        pct = STAGE_PROGRESS[stage]
        note = " · terminal — never moves again" \
            if stage in TERMINAL_STAGES else ""
        lines.append(f"  `{stage}` — {pct:.0f}% of the pipeline{note}")
    lines.append("")
    lines.append("move one: /careers move <app> <stage>")
    return "\n".join(lines)
