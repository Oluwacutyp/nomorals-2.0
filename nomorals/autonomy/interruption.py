"""One interruption policy — every interruption goes through here.

Previously, interruption decisions were scattered: presence gated
surfaces on interruptibility, the coordinator gated ticks on budget,
the autonomy agent applied its own safety vetoes, and each caller
picked its own threshold. Two callers could disagree about the same
ping. This module is the single policy:

    assess_worth(interruption, context) -> {score, decision, reasons, ...}

* ``interruption`` — what wants the owner's attention: a dict (or
  dataclass-like) with ``priority``, ``urgency`` (0–1), ``importance``
  (0–1), ``kind`` / ``title`` and an optional ``deadline_ts``.
* ``context`` — the moment: ``now`` (unix ts), ``quiet_hours``
  (``"23:00-07:00"`` / ``("23:00", "07:00")`` / list of hour ints /
  ``None``), ``interruptibility`` (0–1, the presence model), and
  ``tz_offset`` hours when no timezone is available.
* Returns ``score`` (0–1, higher = more worth interrupting for) and
  ``decision``: ``interrupt`` (surface now), ``defer`` (hold for a
  better moment / the next digest), or ``drop`` (not worth the ping
  at all).

Rules, in order:

1. **Safety always interrupts.** ``priority="safety"`` (and
   ``"owner-direct"`` — the owner is already talking to us) bypasses
   quiet hours and thresholds.
2. **Quiet hours hold everything else.** Inside quiet hours,
   non-safety interruptions lose score and can at best ``defer`` —
   except ``urgent`` ones past a high bar, which may still interrupt.
   Quiet hours are computed from the wall clock in ``context`` —
   never from a prompt.
3. **Thresholds are per priority.** ``urgent`` needs less to interrupt
   than ``normal``; ``background`` almost never interrupts alone —
   it defers to the digest or drops.
4. **The function is pure.** Same inputs → same outputs. No DB, no
   randomness, no hidden state — so the policy is testable and two
   callers can never disagree.

Note: ``nomorals.research.pipeline.assess_worth`` is a different
function (it gates research findings for delivery). This one gates
*interruptions*. The name is shared deliberately — "is this worth
the owner's attention?" is one question — but the signatures differ
(``(finding, rctx)`` vs ``(interruption, context)``) and neither
imports the other.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

__all__ = [
    "DECISIONS",
    "PRIORITIES",
    "THRESHOLDS",
    "Interruption",
    "assess_worth",
    "in_quiet_hours",
    "normalize_priority",
]

#: Decisions, in order of assertiveness.
DECISIONS = ("interrupt", "defer", "drop")

#: Priorities this policy understands (aliases normalized in
#: :func:`normalize_priority`).
PRIORITIES = ("safety", "owner-direct", "urgent", "normal", "background",
              "scheduled", "ambient")

#: Per-priority (interrupt_threshold, defer_threshold). Score >=
#: interrupt_threshold → interrupt; >= defer_threshold → defer;
#: else drop. ``safety`` / ``owner-direct`` short-circuit before
#: thresholds are consulted (see :func:`assess_worth`).
THRESHOLDS: dict[str, tuple[float, float]] = {
    "safety": (0.0, 0.0),
    "owner-direct": (0.0, 0.0),
    "urgent": (0.45, 0.20),
    "normal": (0.65, 0.35),
    "background": (0.88, 0.55),
    # Coordinator-ladder names map onto the same tiers.
    "scheduled": (0.65, 0.35),
    "ambient": (0.88, 0.55),
}

#: Quiet-hours penalty: non-safety score multiplier inside quiet hours.
QUIET_HOURS_MULTIPLIER = 0.35

#: Deadline pressure: full weight when the deadline is within this long.
DEADLINE_PRESSURE_WINDOW_S = 3600.0

_PRIORITY_ALIASES = {
    "owner_direct": "owner-direct",
    "owner": "owner-direct",
    "critical": "safety",
    "emergency": "safety",
    "high": "urgent",
    "medium": "normal",
    "low": "background",
}


def normalize_priority(priority: Any) -> str:
    """Normalize a priority string; unknown → ``"normal"``."""
    key = str(priority or "normal").strip().lower().replace("-", "_")
    # Canonical form uses hyphens for owner-direct.
    key = key.replace("_", "-") if key in ("owner-direct",) else key
    alias = _PRIORITY_ALIASES.get(str(priority or "").strip().lower())
    if alias:
        return alias
    if str(priority or "").strip().lower() in THRESHOLDS:
        return str(priority).strip().lower()
    return "normal"


def in_quiet_hours(now_ts: float, quiet_hours: Any,
                   *, tz_offset: float = 0.0) -> bool:
    """True when ``now_ts`` falls inside ``quiet_hours``.

    ``quiet_hours`` accepts: ``None`` (never quiet),
    ``"HH:MM-HH:MM"`` (overnight windows like ``"23:00-07:00"`` wrap
    midnight), a ``(start, end)`` pair of ``"HH:MM"`` strings, or a
    list/set of hour ints (the presence model's learned-quiet-hours
    shape). Hours are wall-clock at ``tz_offset`` from UTC.
    """
    if quiet_hours is None:
        return False
    dt = datetime.fromtimestamp(now_ts, tz=timezone.utc)
    hour = (dt.hour + int(tz_offset)) % 24
    minute = dt.minute
    now_minutes = hour * 60 + minute
    if isinstance(quiet_hours, (list, tuple, set, frozenset)):
        items = list(quiet_hours)
        if items and all(isinstance(i, (int, float)) for i in items):
            return hour in {int(i) for i in items}
        if len(items) == 2 and all(isinstance(i, str) for i in items):
            start_s, end_s = str(items[0]), str(items[1])
        else:
            return False
    elif isinstance(quiet_hours, str):
        parts = quiet_hours.split("-")
        if len(parts) != 2:
            return False
        start_s, end_s = parts[0].strip(), parts[1].strip()
    else:
        return False

    def _to_min(s: str) -> int | None:
        try:
            h, m = s.split(":")
            h_i, m_i = int(h), int(m)
        except (ValueError, AttributeError):
            return None
        if not (0 <= h_i <= 23 and 0 <= m_i <= 59):
            return None
        return h_i * 60 + m_i

    start = _to_min(start_s)
    end = _to_min(end_s)
    if start is None or end is None:
        return False
    if start == end:
        return False
    if start < end:
        return start <= now_minutes < end
    # Overnight window wraps midnight.
    return now_minutes >= start or now_minutes < end


@dataclass
class Interruption:
    """What wants the owner's attention."""

    priority: str = "normal"
    kind: str = ""
    title: str = ""
    urgency: float = 0.5   # 0–1: how time-sensitive
    importance: float = 0.5  # 0–1: how much it matters
    deadline_ts: float | None = None
    # Optional extra signal: the presence interruptibility model
    # already scored this moment (0–1). When present it nudges the
    # score; when absent the context-level value is used.
    interruptibility: float | None = None


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if hasattr(value, "_asdict"):
        return dict(value._asdict())
    out: dict[str, Any] = {}
    for attr in ("priority", "kind", "title", "urgency", "importance",
                 "deadline_ts", "interruptibility", "now", "quiet_hours",
                 "tz_offset"):
        if hasattr(value, attr):
            out[attr] = getattr(value, attr)
    return out


def _clamp01(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def assess_worth(interruption: Any, context: Any = None) -> dict[str, Any]:
    """Score an interruption and decide: interrupt / defer / drop.

    Pure function: same ``(interruption, context)`` → same result.
    ``interruption`` is a dict / :class:`Interruption` with
    ``priority`` / ``urgency`` / ``importance`` / ``deadline_ts``.
    ``context`` is a dict with ``now`` (unix ts, default: now),
    ``quiet_hours`` (see :func:`in_quiet_hours`), ``interruptibility``
    (0–1, default 0.5) and ``tz_offset`` (hours, default 0).
    """
    intr = _as_dict(interruption)
    ctx = _as_dict(context)
    now = float(ctx.get("now") or time.time())
    tz_offset = float(ctx.get("tz_offset") or 0.0)
    priority = normalize_priority(intr.get("priority", "normal"))
    urgency = _clamp01(intr.get("urgency", 0.5))
    importance = _clamp01(intr.get("importance", 0.5))
    interruptibility = _clamp01(
        intr.get("interruptibility", ctx.get("interruptibility", 0.5)))
    quiet = in_quiet_hours(now, ctx.get("quiet_hours"),
                           tz_offset=tz_offset)
    reasons: list[str] = [f"priority={priority}"]

    # 1. Safety / owner-direct short-circuit: always interrupt.
    if priority in ("safety", "owner-direct"):
        return {
            "score": 1.0,
            "decision": "interrupt",
            "priority": priority,
            "quiet_hours": quiet,
            "reasons": reasons + ["safety/owner-direct always interrupts"],
        }

    # 2. Base score: urgency + importance, nudged by interruptibility.
    # Deadline pressure: ramps from 0 (deadline >= 1h out / none) to
    # +0.20 (deadline now or past).
    score = 0.45 * urgency + 0.35 * importance + 0.10 * interruptibility
    deadline_ts = intr.get("deadline_ts")
    if deadline_ts is not None:
        try:
            seconds_left = float(deadline_ts) - now
        except (TypeError, ValueError):
            seconds_left = math.inf
        if seconds_left <= 0:
            score += 0.20
            reasons.append("deadline passed")
        elif seconds_left < DEADLINE_PRESSURE_WINDOW_S:
            pressure = 0.20 * (1.0 - seconds_left / DEADLINE_PRESSURE_WINDOW_S)
            score += pressure
            reasons.append(f"deadline in {seconds_left:.0f}s")
    # Priority base: urgent items start higher, background lower.
    score += {"urgent": 0.10, "normal": 0.0, "background": -0.10,
              "scheduled": 0.0, "ambient": -0.10}.get(priority, 0.0)

    # 3. Quiet hours: crush non-safety scores; cap the decision.
    # ``pre_quiet`` is kept for the urgent-breakthrough rule below: a
    # genuinely urgent page (pre-penalty score >= 0.90) still interrupts
    # at 2am; everything else waits.
    pre_quiet = score
    if quiet:
        score *= QUIET_HOURS_MULTIPLIER
        reasons.append(
            f"quiet hours (x{QUIET_HOURS_MULTIPLIER}) — non-safety held")

    score = round(max(0.0, min(1.0, score)), 4)
    interrupt_at, defer_at = THRESHOLDS[priority]
    if quiet:
        # Inside quiet hours, non-safety interruptions can at best
        # defer — unless an urgent one was overwhelmingly strong
        # before the penalty (a genuinely urgent 2am page still gets
        # through; everything else waits).
        if priority == "urgent" and pre_quiet >= 0.90:
            decision = "interrupt"
        elif score >= defer_at:
            decision = "defer"
        else:
            decision = "drop"
    elif score >= interrupt_at:
        decision = "interrupt"
    elif score >= defer_at:
        decision = "defer"
    else:
        decision = "drop"
    reasons.append(
        f"score={score} vs interrupt>={interrupt_at} defer>={defer_at} "
        f"→ {decision}")
    return {
        "score": score,
        "decision": decision,
        "priority": priority,
        "quiet_hours": quiet,
        "reasons": reasons,
    }
