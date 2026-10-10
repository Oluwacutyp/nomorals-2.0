"""Mission progress: chat-visible status, milestone pushes, stall tracking.

This module is the chat-visible face of missions. It answers three questions
the owner asks in chat:

1. **How far along is it?** — ``MissionStore.detail`` + ``render_status_text``
   (% complete, current step, an honest ETA, budget burn).
2. **Tell me when something happens.** — ``MissionMilestones`` pushes
   started / step / stalled / done milestones through the existing
   :class:`nomorals.agents.notifier.Notifier`. No parallel notifier: the
   Wave D choke point (dedupe, feature flag, quiet-hours holds, persisted
   delivery states) applies to every mission push.
3. **Why is it stuck?** — stalled-reason tracking. A stalled mission
   records a *concrete* blocker (``StallCode`` + message + step + since),
   never a bare "waiting". The runner records these automatically
   (consecutive-failure budget, wall/token budget exhaustion) and exposes
   ``mark_stalled`` / ``clear_stalled`` for the concrete external reasons
   only an operator knows: "waiting on provider X", "blocked on approval Y",
   "dependency Z missing".

Anti-spam policy (the Wave D discipline, applied to missions):

- ``started`` / ``stalled`` / ``terminal`` (done/failed/cancelled) pushes are
  *transition*-gated: each fires once per mission, the marker persisted in
  ``mission.state["milestones"]`` so a restart/resume never re-sends.
- ``step`` pushes are *cooldown*-gated (default 600s, the same window as the
  Notifier dedupe) and *quiet-hours*-held: a step update during quiet hours
  is stored as ``held-quiet-hours`` (visible in ``/notify``, redeliverable by
  the existing ``Notifier.redeliver``) instead of waking the owner with a
  transient "%" note.
- The Notifier's own (kind, title) dedupe is the backstop for all of it.

Every method on ``MissionMilestones`` is telemetry-safe: it never raises.
A milestone push must not be able to break a mission run.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from ..agents.notifier import (
    STATE_HELD_QUIET,
    Notifier,
    _in_quiet_hours_now,
)
from ..core.errors import ValidationError
from ..core.logging_setup import get_logger
from ..storage.kv import KVStore

__all__ = [
    "StallCode",
    "clear_stall",
    "estimate_eta",
    "eta_breakdown",
    "fmt_duration",
    "MissionMilestones",
    "MissionWatchers",
    "real_plan_steps",
    "box_lines",
    "record_stall",
    "render_mission_card",
    "render_mission_table",
    "render_progress_bar",
    "render_result_card",
    "render_sparkline",
    "render_status_card",
    "render_status_text",
    "STATUS_STYLES",
]

_log = get_logger(__name__)

#: after this many consecutive step failures the runner declares the
#: mission stalled instead of silently grinding through the plan.
STALL_AFTER_FAILURES = 3


class StallCode:
    """Concrete reasons a mission can be stuck. Never a bare 'waiting'."""

    WAITING_ON_PROVIDER = "waiting_on_provider"
    BLOCKED_ON_APPROVAL = "blocked_on_approval"
    RETRY_BUDGET_EXHAUSTED = "retry_budget_exhausted"
    BUDGET_EXHAUSTED = "budget_exhausted"
    DEPENDENCY_MISSING = "dependency_missing"

    ALL = frozenset({
        WAITING_ON_PROVIDER,
        BLOCKED_ON_APPROVAL,
        RETRY_BUDGET_EXHAUSTED,
        BUDGET_EXHAUSTED,
        DEPENDENCY_MISSING,
    })

    #: human-readable one-liners for the status command.
    LABELS = {
        WAITING_ON_PROVIDER: "waiting on provider",
        BLOCKED_ON_APPROVAL: "blocked on approval",
        RETRY_BUDGET_EXHAUSTED: "retry budget exhausted",
        BUDGET_EXHAUSTED: "budget exhausted",
        DEPENDENCY_MISSING: "dependency missing",
    }

    #: what would unblock the mission for each stall code. The status
    #: command and the stall reply always ship these, so a stall never
    #: reads as a bare "stalled" with no way out.
    UNBLOCK_HINTS = {
        WAITING_ON_PROVIDER: "unblocks when the provider recovers, "
            "or switch providers and /mission clear it",
        BLOCKED_ON_APPROVAL: "unblocks when the owner approves — reply to "
            "the approval request, then /mission clear it",
        RETRY_BUDGET_EXHAUSTED: "unblocks with /mission clear after fixing "
            "the failing step, or /mission retry for a fresh attempt",
        BUDGET_EXHAUSTED: "unblocks when the budget is raised, or "
            "/mission retry for a fresh attempt with a bigger budget",
        DEPENDENCY_MISSING: "unblocks when the missing dependency is "
            "installed, then /mission clear it",
    }


# ── stall state (pure helpers on the Mission object) ─────────────────────────

_STALL_KEY = "stall"


def record_stall(
    mission: Any,
    code: str,
    message: str,
    *,
    step: str = "",
    extra: dict[str, Any] | None = None,
) -> bool:
    """Record a concrete stall reason on the mission's state.

    Returns ``True`` when this is a *new or changed* stall (a push-worthy
    transition); ``False`` when the identical stall is already recorded, so
    callers can avoid re-notifying.  Raises :class:`ValidationError` for an
    unknown code — fail fast, no silent generic strings.
    """
    if code not in StallCode.ALL:
        raise ValidationError(
            f"unknown stall code {code!r} — one of: {sorted(StallCode.ALL)}",
            field="code",
        )
    if not message or not message.strip():
        raise ValidationError("a stall reason needs a message", field="message")
    prev = mission.state.get(_STALL_KEY) or {}
    mission.state[_STALL_KEY] = {
        "code": code,
        "message": message.strip(),
        "step": step or "",
        "since": time.time(),
        "extra": dict(extra or {}),
    }
    return prev.get("code") != code or prev.get("message") != message.strip()


def clear_stall(mission: Any) -> bool:
    """Drop the stall record (called when the mission makes progress again)."""
    return mission.state.pop(_STALL_KEY, None) is not None


# ── plan / ETA ───────────────────────────────────────────────────────────────

def _step_name(entry: Any) -> str:
    if isinstance(entry, dict):
        return str(entry.get("name") or "")
    return str(entry)


def real_plan_steps(plan: Any) -> list[Any]:
    """Plan entries minus the persisted ``__plan_error__`` degradation marker."""
    return [
        s for s in (plan or [])
        if not (isinstance(s, dict) and "__plan_error__" in s)
    ]


def fmt_duration(seconds: float | None) -> str:
    """'3661' -> '1h 1m'. Never raises; None -> 'unknown'."""
    if seconds is None:
        return "unknown"
    try:
        total = max(0, int(seconds))
    except (TypeError, ValueError):
        return "unknown"
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s" if secs else f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m" if minutes else f"{hours}h"


#: EMA smoothing factor for the per-step rate (gmailarchiver's
#: ProgressTracker practice: α=0.3 reacts to recent steps without spike
#: whiplash from one slow outlier).
ETA_EMA_ALPHA = 0.3


def _historical_step_rate(db: Any) -> tuple[float | None, int]:
    """Average wall-seconds per step across recently finished missions.

    Used as an honest prior when the current mission has no step timing
    yet. Returns ``(seconds_per_step, mission_count)``; ``(None, 0)``
    when there is no history. Never raises — no db, no history.
    """
    if db is None:
        return None, 0
    try:
        row = db.query_one(
            "SELECT AVG(spent_wall / NULLIF(iterations, 0)) AS avg_s,"
            " COUNT(*) AS n FROM missions"
            " WHERE status = 'done' AND iterations > 0 AND spent_wall > 0"
            " AND COALESCE(finished_at, updated_at, 0) > ?",
            (time.time() - 30 * 86400,),
        )
    except Exception:  # noqa: BLE001 - history is a nice-to-have
        return None, 0
    if row is None:
        return None, 0
    try:
        avg = float(row["avg_s"]) if row["avg_s"] is not None else None
        count = int(row["n"] or 0)
    except (TypeError, ValueError, KeyError):
        return None, 0
    if not avg or avg <= 0 or count <= 0:
        return None, 0
    return avg, count


def eta_breakdown(mission: Any, db: Any = None) -> dict[str, Any]:
    """Structured ETA: method, per-step rate, and the honest note.

    Method is one of ``"ema"`` (exponential moving average over this
    mission's measured step durations), ``"run-average"`` (whole-run
    average fallback), ``"history"`` (prior from recently finished
    missions — no timing on this mission yet), or ``"none"``.
    """
    steps = real_plan_steps(mission.state.get("plan"))
    if not steps:
        return {"eta_seconds": None, "note": "no plan stored yet",
                "method": "none", "per_step_s": None, "remaining": 0}
    completed = set(mission.state.get("completed_steps") or [])
    done = sum(1 for s in steps if _step_name(s) in completed)
    remaining = len(steps) - done
    if remaining <= 0:
        return {"eta_seconds": 0.0, "note": "all steps complete",
                "method": "ema", "per_step_s": 0.0, "remaining": 0}
    durations = (mission.state or {}).get("step_durations") or {}
    series: list[float] = []
    if isinstance(durations, dict):
        # trailing window (insertion-ordered: last entries = most recent
        # steps), so one ancient outlier cannot yank the ETA — then an
        # exponential moving average inside the window, so recent steps
        # weigh most without spike whiplash from a single slow step.
        for value in list(durations.values())[-5:]:
            try:
                seconds = float(value)
            except (TypeError, ValueError):
                continue
            if seconds >= 0:
                series.append(seconds)
    per_step: float | None = None
    method = "none"
    note = ""
    if series:
        ema = series[0]
        for value in series[1:]:
            ema = ETA_EMA_ALPHA * value + (1.0 - ETA_EMA_ALPHA) * ema
        per_step = ema
        method = "ema"
        note = (f"based on the last {len(series)} step(s), "
                f"EMA-smoothed (α={ETA_EMA_ALPHA})")
    elif done > 0 and (mission.spent_wall or 0) > 0:
        per_step = mission.spent_wall / done
        method = "run-average"
    else:
        hist_rate, hist_n = _historical_step_rate(db)
        if hist_rate is not None:
            per_step = hist_rate
            method = "history"
            note = (f"no timing on this mission yet — based on {hist_n} "
                    f"recently finished mission(s)")
        else:
            return {"eta_seconds": None, "note": "no step timing yet",
                    "method": "none", "per_step_s": None,
                    "remaining": remaining}
    assert per_step is not None
    eta = per_step * remaining
    budget_wall = float(mission.budget_wall or 0.0)
    if budget_wall:
        wall_left = max(0.0, budget_wall - mission.spent_wall)
        if wall_left < eta:
            note = (f"{note}; " if note else "") + \
                f"wall budget runs out first (~{fmt_duration(wall_left)} left)"
    return {"eta_seconds": eta, "note": note, "method": method,
            "per_step_s": round(per_step, 2), "remaining": remaining}


def estimate_eta(mission: Any, db: Any = None) -> tuple[float | None, str]:
    """Honest ETA from measured per-step wall time.

    Returns ``(seconds, note)``. ``seconds`` is ``None`` when there is not
    enough measured data — the note then says *why* ("no step timing yet",
    "no plan stored") instead of inventing a number. When the wall budget
    would run out first, the note says so explicitly.

    The rate is an exponential moving average (α=0.3) over this mission's
    measured step durations — recent steps predict the near future better
    than a whole-run average on a heterogeneous plan, and the EMA keeps
    one slow outlier from yanking the number. Falls back to the whole-run
    average, then to the historical per-step rate of recently finished
    missions (marked as such so it never masquerades as measured data).
    """
    info = eta_breakdown(mission, db)
    return info["eta_seconds"], info["note"]


# ── chat rendering ───────────────────────────────────────────────────────────

#: Styles for :func:`render_status_text`.
STATUS_STYLES = ("full", "compact", "card", "plain")


def render_progress_bar(percent: float, width: int = 20,
                        *, ascii_only: bool = False) -> str:
    """``62.0`` -> ``████████████░░░░░░░░ 62%`` (tqdm-style, one glance).

    ``ascii_only`` swaps the unicode blocks for ``#``/``-`` (dumb
    terminals, logs). Never raises; clamps out-of-range input.
    """
    try:
        pct = max(0.0, min(100.0, float(percent)))
    except (TypeError, ValueError):
        pct = 0.0
    width = max(4, min(60, int(width or 20)))
    filled = int(round(pct / 100.0 * width))
    if ascii_only:
        bar = "#" * filled + "-" * (width - filled)
    else:
        bar = "█" * filled + "░" * (width - filled)
    return f"{bar} {pct:.0f}%"


def render_sparkline(values: list[float] | tuple[float, ...],
                     *, width: int = 0) -> str:
    """Tiny bar chart of a value series: per-step durations at a glance.

    ``[1, 2, 8, 4]`` -> ``▁▂█▄``. Empty input -> ``""``.
    """
    glyphs = "▁▂▃▄▅▆▇█"
    try:
        series = [float(v) for v in (values or [])]
    except (TypeError, ValueError):
        return ""
    if not series:
        return ""
    if width and len(series) > width:
        # downsample: keep the shape, drop the noise
        stride = len(series) / width
        series = [series[int(i * stride)] for i in range(width)]
    lo, hi = min(series), max(series)
    span = hi - lo
    if span <= 0:
        return glyphs[3] * len(series)
    return "".join(
        glyphs[min(len(glyphs) - 1, int((v - lo) / span * (len(glyphs) - 1)))]
        for v in series)


def _box_lines(title: str, lines: list[str], *, width: int = 62) -> list[str]:
    """Wrap lines in a unicode box. Pure presentation, no logic."""
    width = max(20, min(100, width))
    inner = width - 4
    out = [f"┌─ {title[:inner]} " + "─" * max(0, inner - len(title) - 1) + "┐"]
    for line in lines:
        for chunk in [line[i:i + inner] for i in range(0, max(1, len(line)), inner)] or [""]:
            out.append(f"│ {chunk.ljust(inner)} │")
    out.append("└" + "─" * (width - 2) + "┘")
    return out


#: Public alias for the boxed-card wrapper (used by golden reports and
#: other renderers that want the same card chrome).
box_lines = _box_lines


_STATUS_GLYPH = {
    "pending": "⏳", "running": "⚙️", "paused": "⏸️",
    "done": "✅", "failed": "❌", "cancelled": "🚫",
}


def _status_line(detail: dict[str, Any], *, plain: bool = False) -> str:
    m = detail.get("mission") or {}
    p = detail.get("progress") or {}
    status = str(m.get("status") or "?")
    if plain:
        head = f"{m.get('name') or 'mission'} [{status}]"
    else:
        head = f"{_STATUS_GLYPH.get(status, '🎯')} {m.get('name') or 'mission'} [{status}]"
    pct = float(p.get("percent") or 0.0)
    line = (f"{head} — {p.get('steps_done', 0)}/{p.get('total_steps', 0)} "
            f"steps ({pct:.0f}%)")
    if p.get("current_step"):
        line += f" · now: {p['current_step']}"
    return line


def _eta_line(detail: dict[str, Any]) -> str:
    eta = detail.get("eta_seconds")
    note = str(detail.get("eta_note") or "")
    if eta is None:
        return f"eta: unknown" + (f" — {note}" if note else "")
    return f"eta: ~{fmt_duration(eta)}" + (f" ({note})" if note else "")


def _spend_line(detail: dict[str, Any]) -> str:
    m = detail.get("mission") or {}
    spent = (f"spent: {float(m.get('spent_wall') or 0.0):.0f}s wall, "
             f"{int(m.get('spent_tokens') or 0)} tokens")
    budget_wall = float(m.get("budget_wall") or 0.0)
    budget_tokens = int(m.get("budget_tokens") or 0)
    if budget_wall or budget_tokens:
        spent += (f" (budget: {fmt_duration(budget_wall) if budget_wall else '∞'} / "
                  f"{budget_tokens if budget_tokens else '∞'} tokens)")
    return spent + f" · {int(m.get('iterations') or 0)} iteration(s)"


def _stall_lines(detail: dict[str, Any], *, indent: str = "") -> list[str]:
    stall = detail.get("stall")
    if not stall:
        return []
    code = stall.get("code")
    label = StallCode.LABELS.get(code, code)
    line = f"{indent}⚠️ stalled [{code}]: {label} — {stall.get('message')}"
    if stall.get("step"):
        line += f" (step: {stall['step']})"
    since = stall.get("since")
    if since:
        line += f" [since {time.strftime('%H:%M', time.localtime(since))}]"
    lines = [line]
    hint = StallCode.UNBLOCK_HINTS.get(code)
    if hint:
        lines.append(f"{indent}   → unblocks: {hint}")
    return lines


def render_status_text(detail: dict[str, Any], *, style: str = "full") -> str:
    """One chat-visible block: % complete, current step, ETA, stall reason.

    Takes the dict from ``MissionStore.detail``. Every line is real,
    persisted state — nothing is guessed.

    Styles: ``"full"`` (the historical multi-line block, byte-identical
    default), ``"compact"`` (one line), ``"card"`` (boxed card),
    ``"plain"`` (no emoji — logs, dumb terminals).
    """
    if style not in STATUS_STYLES:
        raise ValueError(f"unknown status style {style!r} — one of: {STATUS_STYLES}")
    if style == "compact":
        parts = [_status_line(detail), _eta_line(detail), _spend_line(detail)]
        stall = detail.get("stall") or {}
        if stall:
            code = stall.get("code")
            parts.append(f"stalled [{code}]: {stall.get('message')}")
        elif (detail.get("progress") or {}).get("last_error"):
            parts.append(f"last error: {str((detail['progress']['last_error'])[:120])}")
        return " · ".join(p for p in parts if p)
    if style == "card":
        return render_status_card(detail)
    if style == "plain":
        lines = [_status_line(detail, plain=True)]
        m = detail.get("mission") or {}
        goal = str(m.get("goal") or "")
        if goal:
            lines.append(f"goal: {goal[:160]}")
        p = detail.get("progress") or {}
        lines.append(f"progress: {p.get('steps_done', 0)}/{p.get('total_steps', 0)} "
                     f"({float(p.get('percent') or 0.0):.0f}%)")
        lines.append(_eta_line(detail))
        lines.append(_spend_line(detail))
        lines.extend(_stall_lines(detail))
        if p.get("last_error"):
            lines.append(f"last error: {str(p['last_error'])[:160]}")
        return "\n".join(ln for ln in lines if ln)
    # "full": the historical rendering, unchanged.
    m = detail.get("mission") or {}
    p = detail.get("progress") or {}
    stall = detail.get("stall")
    lines = [f"🎯 {m.get('name') or 'mission'} [{m.get('status') or '?'}]"]
    goal = str(m.get("goal") or "")
    if goal:
        lines.append(f"goal: {goal[:160]}")
    pct = float(p.get("percent") or 0.0)
    progress_line = (
        f"progress: {p.get('steps_done', 0)}/{p.get('total_steps', 0)} "
        f"steps ({pct:.0f}%)"
    )
    if p.get("current_step"):
        progress_line += f" — current: {p['current_step']}"
    lines.append(progress_line)
    lines.append(_eta_line(detail))
    lines.append(_spend_line(detail))
    lines.extend(_stall_lines(detail))
    if not stall and str(m.get("status") or "") in {"running", "paused", "pending"} \
            and not p.get("last_error"):
        lines.append("state: making progress")
    if p.get("last_error"):
        lines.append(f"last error: {str(p['last_error'])[:160]}")
    return "\n".join(ln for ln in lines if ln)


def render_status_card(detail: dict[str, Any], *, width: int = 62) -> str:
    """Boxed one-glance mission card: header, bars, ETA, stall, attempts."""
    m = detail.get("mission") or {}
    p = detail.get("progress") or {}
    status = str(m.get("status") or "?")
    title = f"{_STATUS_GLYPH.get(status, '🎯')} {m.get('name') or 'mission'} [{status}]"
    pct = float(p.get("percent") or 0.0)
    lines = [f"progress  {render_progress_bar(pct, width=24)}"
             f"  {p.get('steps_done', 0)}/{p.get('total_steps', 0)} steps"]
    if p.get("current_step"):
        lines.append(f"now       {p['current_step'][:44]}")
    lines.append(_eta_line(detail))
    # budget burn bar
    budget_wall = float(m.get("budget_wall") or 0.0)
    if budget_wall:
        burn = 100.0 * float(m.get("spent_wall") or 0.0) / budget_wall
        lines.append(f"budget    {render_progress_bar(burn, width=24)}"
                     f"  {fmt_duration(float(m.get('spent_wall') or 0.0))}"
                     f"/{fmt_duration(budget_wall)} wall")
    else:
        lines.append(_spend_line(detail))
    attempts = detail.get("attempts") or {}
    if attempts:
        retried = {k: v for k, v in attempts.items() if int(v or 0) > 1}
        if retried:
            lines.append("retries   " + ", ".join(
                f"{k}×{v}" for k, v in sorted(retried.items())[:6]))
    durations = ((detail.get("mission") or {}).get("state") or {}).get("step_durations")
    spark = render_sparkline(list((durations or {}).values()))
    if spark:
        lines.append(f"step pace {spark}")
    lines.extend(_stall_lines(detail))
    if p.get("last_error"):
        lines.append(f"last error: {str(p['last_error'])[:120]}")
    return "\n".join(_box_lines(title, lines, width=width))


def render_mission_card(mission: Any, *, width: int = 62) -> str:
    """Boxed card from a Mission object (no store round-trip needed)."""
    state = mission.state or {}
    plan = real_plan_steps(state.get("plan"))
    completed = set(state.get("completed_steps") or [])
    total = len(plan)
    done = (sum(1 for s in plan if _step_name(s) in completed) if total
            else len(completed))
    pct = (100.0 * done / total) if total else 0.0
    status = str(getattr(mission, "status", "?"))
    title = (f"{_STATUS_GLYPH.get(status, '🎯')} "
             f"{getattr(mission, 'name', '') or 'mission'} [{status}]")
    lines = [f"goal      {str(getattr(mission, 'goal', ''))[:52]}"]
    lines.append(f"progress  {render_progress_bar(pct, width=24)}"
                 f"  {done}/{total} steps")
    eta, note = estimate_eta(mission)
    lines.append(f"eta: {'~' + fmt_duration(eta) if eta is not None else 'unknown'}"
                 + (f" ({note})" if note else ""))
    priority = int(getattr(mission, "priority", 0) or 0)
    tags = getattr(mission, "tags", []) or []
    meta_bits = []
    if priority:
        meta_bits.append(f"priority {priority}")
    if tags:
        meta_bits.append("tags: " + ", ".join(tags[:5]))
    if getattr(mission, "parent_id", ""):
        meta_bits.append(f"child of {mission.parent_id[:8]}")
    if meta_bits:
        lines.append(" · ".join(meta_bits))
    stall = state.get("stall")
    if stall:
        code = stall.get("code")
        lines.append(f"⚠️ stalled [{code}]: {stall.get('message')}")
    return "\n".join(_box_lines(title, lines, width=width))


def render_mission_table(missions: list[Any], *, width: int = 0) -> str:
    """Aligned one-line-per-mission table for ``/mission list``."""
    if not missions:
        return "(no missions)"
    rows: list[tuple[str, str, str, str, str]] = []
    for mission in missions:
        mid = str(getattr(mission, "id", ""))[:8]
        name = str(getattr(mission, "name", "") or "mission")[:30]
        status = str(getattr(mission, "status", "?"))
        glyph = _STATUS_GLYPH.get(status, "·")
        state = getattr(mission, "state", None) or {}
        plan = real_plan_steps(state.get("plan"))
        completed = set(state.get("completed_steps") or [])
        total = len(plan)
        done = (sum(1 for s in plan if _step_name(s) in completed) if total
                else len(completed))
        pct = (100.0 * done / total) if total else 0.0
        bar = render_progress_bar(pct, width=12)
        eta, _ = estimate_eta(mission)
        eta_s = f"~{fmt_duration(eta)}" if eta is not None else "—"
        prio = int(getattr(mission, "priority", 0) or 0)
        rows.append((mid, name, f"{glyph} {status}", bar, eta_s, f"p{prio}"))
    header = ("ID", "NAME", "STATUS", "PROGRESS", "ETA", "PRI")
    widths = [max(len(r[i]) for r in rows + [header]) for i in range(6)]
    fmt_row = lambda r: "  ".join(c.ljust(widths[i]) for i, c in enumerate(r))
    out = [fmt_row(header), "  ".join("─" * w for w in widths)]
    out.extend(fmt_row(r) for r in rows)
    return "\n".join(out)


def render_result_card(result: dict[str, Any], *, width: int = 62) -> str:
    """Terminal MissionResult card: status, steps table, cost, lessons."""
    status = str(result.get("status") or "?")
    ok = bool(result.get("ok"))
    title = (f"{'✅' if ok else '❌'} mission {status}: "
             f"{result.get('mission_id', '')[:8]}")
    steps = result.get("steps") or []
    done = sum(1 for s in steps if (s.get("ok") if isinstance(s, dict) else False))
    lines = [f"steps     {render_progress_bar(100.0 * done / max(1, len(steps)), width=24)}"
             f"  {done}/{len(steps)} ok"]
    for s in steps[:12]:
        if not isinstance(s, dict):
            continue
        glyph = "✓" if s.get("ok") else "✗"
        sec = float(s.get("seconds") or 0.0)
        lines.append(f"  {glyph} {str(s.get('step'))[:34].ljust(34)} {sec:7.1f}s")
    if len(steps) > 12:
        lines.append(f"  … and {len(steps) - 12} more")
    lines.append(f"took      {fmt_duration(result.get('seconds'))} wall · "
                 f"{result.get('iterations', 0)} iterations")
    if result.get("resumed_from"):
        lines.append(f"resumed   from {result['resumed_from']}")
    lessons = result.get("lessons") or []
    for lesson in lessons[:4]:
        lines.append(f"lesson    {str(lesson)[:52]}")
    if result.get("error"):
        lines.append(f"error     {str(result['error'])[:52]}")
    return "\n".join(_box_lines(title, lines, width=width))


# ── milestone pushes ─────────────────────────────────────────────────────────

class MissionWatchers:
    """Chat-key subscriptions for mission milestone pushes.

    ``/mission watch <id>`` subscribes the current chat; ``MissionMilestones``
    fans every milestone push (started / step / stalled / terminal) out to
    the subscribed chats on top of the normal Notifier publish to the owner
    channels. Subscriptions persist in ``kv_store`` as JSON::

        mission.watch.<mission_id> -> ["telegram:12345", "console:main"]

    All methods are telemetry-safe: with no db they are silent no-ops, and
    every db touch is best-effort so a subscription failure never breaks a
    mission run or a chat command.
    """

    PREFIX = "mission.watch."

    def __init__(self, db: Any) -> None:
        self.db = db

    def _key(self, mission_id: str) -> str:
        return f"{self.PREFIX}{mission_id}"

    def _read(self, mission_id: str) -> list[str]:
        if self.db is None:
            return []
        try:
            data = KVStore(self.db).get(self._key(mission_id))
        except Exception:  # noqa: BLE001 - subscriptions are best-effort
            return []
        if not data:
            return []
        return [str(c) for c in data if c] if isinstance(data, list) else []

    def _write(self, mission_id: str, chats: list[str]) -> None:
        if self.db is None:
            return
        try:
            KVStore(self.db).set(self._key(mission_id), chats)
        except Exception:  # noqa: BLE001 - subscriptions are best-effort
            pass

    def subscribe(self, mission_id: str, chat_key: str) -> bool:
        """Subscribe a chat; returns True when it was newly added."""
        chats = self._read(mission_id)
        if chat_key in chats:
            return False
        chats.append(chat_key)
        self._write(mission_id, chats)
        return True

    def unsubscribe(self, mission_id: str, chat_key: str) -> bool:
        """Unsubscribe a chat; returns True when one was removed."""
        chats = self._read(mission_id)
        if chat_key not in chats:
            return False
        chats = [c for c in chats if c != chat_key]
        self._write(mission_id, chats)
        return True

    def watchers(self, mission_id: str) -> list[str]:
        """Chat keys subscribed to this mission's milestones."""
        return self._read(mission_id)


class MissionMilestones:
    """Proactive chat pushes for mission milestones, via the Notifier.

    Events: ``started`` (once), ``step`` (cooldown-gated progress),
    ``stalled`` (on transition), ``terminal`` (done/failed/cancelled, once
    each). Every push funnels through ``Notifier.publish(kind="mission")``
    so the Wave D choke point — dedupe, the ``notifier`` feature flag, and
    persisted delivery states — governs mission pushes exactly like every
    other alert. Nothing here may raise: milestone reporting is telemetry,
    and telemetry never breaks a run.
    """

    KIND = "mission"
    MILESTONE_LOG_KEY = "milestones"
    LOG_LIMIT = 30

    #: What gets pushed. ``"all"`` (historical: started/step/stalled/
    #: terminal), ``"milestones"`` (started/stalled/terminal — no per-step
    #: progress notes), ``"terminal"`` (only the final verdict). Long
    #: missions stay quiet until they matter.
    NOTIFY_LEVELS = ("all", "milestones", "terminal")

    def __init__(
        self,
        context: Any,
        *,
        store: Any = None,
        clock: Callable[[], float] | None = None,
        step_cooldown_seconds: float = 600.0,
        watch_store: "MissionWatchers | None" = None,
        notify_level: str = "all",
    ) -> None:
        self.context = context
        if store is None:
            from .mission import MissionStore

            store = MissionStore(context.db)
        self.store = store
        self._clock = clock or time.time
        self.step_cooldown = max(0.0, float(step_cooldown_seconds))
        self._notifier = Notifier(context)
        self.notify_level = (notify_level or "all").strip().lower()
        if self.notify_level not in self.NOTIFY_LEVELS:
            self.notify_level = "all"
        # Chat-key subscriptions fanned out on top of the Notifier publish.
        # Default: backed by the context db (silent no-op when db is None).
        if watch_store is None:
            watch_store = MissionWatchers(getattr(context, "db", None))
        self.watch_store = watch_store

    def set_notify_level(self, level: str) -> str:
        """Change the push verbosity. Returns the effective level."""
        level = (level or "").strip().lower()
        if level not in self.NOTIFY_LEVELS:
            raise ValueError(f"unknown notify level {level!r} — one of: "
                             f"{self.NOTIFY_LEVELS}")
        self.notify_level = level
        return level

    def digest(self, mission: Any, *, limit: int = 10) -> str:
        """Collapse the persisted milestone log into one digest block.

        For ``notify_level="milestones"`` owners: what happened, when —
        without the per-step noise. Reads ``mission.state["milestones"]``
        (written by ``_mark``), so it survives restarts.
        """
        log = mission.state.get(self.MILESTONE_LOG_KEY) or []
        entries = [e for e in log if isinstance(e, dict)][-max(1, limit):]
        if not entries:
            return f"no milestones yet for {getattr(mission, 'name', 'mission')}"
        lines = [f"📰 digest: {getattr(mission, 'name', 'mission')} "
                 f"({len(log)} milestone(s))"]
        for entry in entries:
            at = entry.get("at")
            when = time.strftime("%H:%M", time.localtime(at)) if at else "??:??"
            lines.append(f"  {when}  {entry.get('event', '?')} — "
                         f"{str(entry.get('title') or '')[:80]}")
        return "\n".join(lines)

    # ── events ───────────────────────────────────────────────────────────────

    def _fanout(self, mission: Any, title: str, body: str) -> None:
        """Push the same milestone to subscribed chats (``/mission watch``).

        This rides on top of the normal Notifier publish, which still goes
        to the owner channels with all Wave D discipline intact. Chats that
        are already owner channels are skipped (the Notifier reached them).
        Never raises: fan-out is telemetry.
        """
        try:
            store = self.watch_store
            if store is None:
                return
            chats = store.watchers(getattr(mission, "id", ""))
        except Exception:  # noqa: BLE001 - telemetry never breaks a run
            return
        if not chats:
            return
        gateway = getattr(self._notifier, "gateway", None)
        if gateway is None:
            return
        from ..social.chat.base import ChatRef

        owner_keys = set()
        settings = getattr(self.context, "settings", None)
        partner = getattr(settings, "partner", None)
        raw = getattr(partner, "owner_chats", "") or ""
        for key in str(raw).split(","):
            key = key.strip()
            if key:
                owner_keys.add(key)
        text = f"🔔 {title}" + (f"\n{body}" if body else "")
        text = text[:3900]
        for key in chats:
            plat, _, cid = str(key).partition(":")
            if not (plat and cid) or key in owner_keys:
                continue
            try:
                gateway.send(plat, ChatRef(platform=plat, chat_id=cid), text)
            except Exception:  # noqa: BLE001 - one dead chat must not kill the rest
                continue

    def on_started(self, mission: Any) -> dict[str, Any]:
        """Push once when a mission starts running. Resume-safe."""
        try:
            if self.notify_level == "terminal":
                return {"suppressed": "notify-level", "event": "started"}
            if self._marked(mission, "started"):
                return {"suppressed": "already-notified", "event": "started"}
            title = f"mission started: {mission.name}"
            body = (
                f"goal: {mission.goal[:200]}\n"
                + self._budget_line(mission)
            )
            res = self._notifier.publish(self.KIND, title, body)
            self._fanout(mission, title, body)
            self._mark(mission, "started", title)
            # a fresh start push counts against the step cooldown, so the
            # first completing step doesn't double-notify seconds later.
            mission.state["last_progress_push"] = self._clock()
            self.store.save(mission)
            return res
        except Exception as exc:  # noqa: BLE001 - telemetry never breaks a run
            _log.debug("mission milestone started failed: %s", exc)
            return {"suppressed": "error", "event": "started"}

    def on_step(self, mission: Any, outcome: Any) -> dict[str, Any]:
        """Cooldown-gated progress push for a completed step.

        Suppressed entirely unless ``notify_level == "all"`` — digest
        mode owners get started/stalled/terminal only.
        """
        try:
            if self.notify_level != "all":
                return {"suppressed": "notify-level", "event": "step"}
            return self._on_step(mission, outcome)
        except Exception as exc:  # noqa: BLE001 - telemetry never breaks a run
            _log.debug("mission milestone step failed: %s", exc)
            return {"suppressed": "error", "event": "step"}

    def _on_step(self, mission: Any, outcome: Any) -> dict[str, Any]:
        now = self._clock()
        last = float(mission.state.get("last_progress_push") or 0.0)
        if now - last < self.step_cooldown:
            return {
                "suppressed": "cooldown",
                "event": "step",
                "cooldown_remaining": round(self.step_cooldown - (now - last), 1),
            }
        detail = self.store.detail(mission.id)
        progress = detail.get("progress") or {}
        pct = float(progress.get("percent") or 0.0)
        title = f"mission update: {mission.name} — {pct:.0f}%"
        body = render_status_text(detail)
        if _in_quiet_hours_now(self.context):
            # Non-critical progress note during quiet hours: held, not
            # sent. Stored as held-quiet-hours so /notify shows it and the
            # existing Notifier.redeliver() can send it when quiet ends.
            res = self._notifier._store(  # noqa: SLF001 - deliberate: held send
                self.KIND, title, body, 0, delivery_state=STATE_HELD_QUIET
            )
            res["held"] = "quiet-hours"
        else:
            res = self._notifier.publish(self.KIND, title, body)
        self._fanout(mission, title, body)
        mission.state["last_progress_push"] = now
        self.store.save(mission)
        return res

    def on_stalled(self, mission: Any) -> dict[str, Any]:
        """Push immediately when a *new* stall reason is recorded."""
        try:
            if self.notify_level == "terminal":
                return {"suppressed": "notify-level", "event": "stalled"}
            stall = mission.state.get(_STALL_KEY) or {}
            if not stall:
                return {"suppressed": "no-stall", "event": "stalled"}
            key = f"stalled:{stall.get('code')}"
            if self._marked(mission, key):
                return {"suppressed": "already-notified", "event": "stalled"}
            # the stall code rides in the title so the Notifier's
            # (kind, title) dedupe treats a *different* blocker as a new
            # alert while collapsing repeats of the same one.
            title = f"mission stalled [{stall.get('code')}]: {mission.name}"
            body = render_status_text(self.store.detail(mission.id))
            res = self._notifier.publish(self.KIND, title, body)
            self._fanout(mission, title, body)
            self._mark(mission, key, title)
            return res
        except Exception as exc:  # noqa: BLE001 - telemetry never breaks a run
            _log.debug("mission milestone stalled failed: %s", exc)
            return {"suppressed": "error", "event": "stalled"}

    def on_terminal(self, mission: Any, status: str, *, error: str = "") -> dict[str, Any]:
        """Push once per terminal transition (done / failed / cancelled)."""
        try:
            key = f"terminal:{status}"
            if self._marked(mission, key):
                return {"suppressed": "already-notified", "event": "terminal"}
            word = {"done": "done ✅", "failed": "failed ❌",
                    "cancelled": "cancelled"}.get(status, status)
            title = f"mission {word}: {mission.name}"
            detail = self.store.detail(mission.id)
            body = render_status_text(detail)
            if error:
                body += f"\nerror: {error[:200]}"
            res = self._notifier.publish(self.KIND, title, body)
            self._fanout(mission, title, body)
            self._mark(mission, key, title)
            return res
        except Exception as exc:  # noqa: BLE001 - telemetry never breaks a run
            _log.debug("mission milestone terminal failed: %s", exc)
            return {"suppressed": "error", "event": "terminal"}

    # ── milestone log (persisted markers: resume-safe, process-safe) ─────────

    def _marked(self, mission: Any, key: str) -> bool:
        log = mission.state.get(self.MILESTONE_LOG_KEY) or []
        return any(isinstance(e, dict) and e.get("event") == key for e in log)

    def _mark(self, mission: Any, key: str, title: str) -> None:
        log = list(mission.state.get(self.MILESTONE_LOG_KEY) or [])
        log.append({"event": key, "title": title, "at": self._clock()})
        mission.state[self.MILESTONE_LOG_KEY] = log[-self.LOG_LIMIT:]
        self.store.save(mission)

    @staticmethod
    def _budget_line(mission: Any) -> str:
        wall = float(getattr(mission, "budget_wall", 0.0) or 0.0)
        tokens = int(getattr(mission, "budget_tokens", 0) or 0)
        if not wall and not tokens:
            return "budget: unlimited"
        return (f"budget: {fmt_duration(wall) if wall else '∞'} wall / "
                f"{tokens if tokens else '∞'} tokens")
