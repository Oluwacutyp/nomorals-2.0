"""Periodic synthesis — weekly + monthly reviews.

Composes (never rebuilds) the existing week layer:

* :mod:`nomorals.focus.sessions` — ``build_weekly_review`` supplies
  wins, open commitments, decisions, focus totals, calibration and
  the summarizer sections (health / learn / finance).  This module
  imports and reuses that builder for its sections; it does NOT
  re-implement focus elapsed / overlap / summarizer logic.
* :mod:`nomorals.commitments.ledger` / :mod:`nomorals.decisions.journal`
  — consumed duck-typed through the focus builder (ledgers, lists of
  rows, or plain dicts), exactly as the focus builder accepts them.

On top of the focus review this module adds:

* ``slipped`` — open commitments whose deadline is past (overdue).
* ``blocked`` — decisions still open + focus sessions with
  interruptions (the things standing in the way).
* ``next_week`` — carry-overs: slipped + blocked + the focus
  builder's next-week plan, de-duplicated.
* ``build_monthly_review`` — aggregates weekly reviews (totals,
  per-week reliability trend, most-frequent slipped / blocked items).

Honesty rules (inherited from the focus builder, kept here too):

* Missing summarizer → "not connected".  Failing summarizer →
  "unavailable".  Never invent data.
* No resolved commitments/decisions in a window → reliability is
  ``None``, never 0%.

Only the standard library is imported.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "WeeklyReview",
    "MonthlyReview",
    "build_weekly_review",
    "build_monthly_review",
    "render_markdown",
]

# ---------------------------------------------------------------------------
# Helpers — reading rows without re-implementing focus logic.
# These only *read* interruptions / deadlines from caller-supplied
# objects; elapsed-time, overlap and summarizer handling stay in
# nomorals.focus.sessions.
# ---------------------------------------------------------------------------


def _get(row: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(row, dict):
            if name in row and row[name] is not None:
                return row[name]
        elif hasattr(row, name):
            value = getattr(row, name)
            if value is not None:
                return value
    return default


def _rows_of(source: Any, attr: str) -> list[Any]:
    """Normalise a source to a list of rows (ledger / list / dict / None)."""
    if source is None:
        return []
    if isinstance(source, list):
        return list(source)
    if isinstance(source, dict):
        return list(source.values())
    for name in (attr, "entries", "rows"):
        value = getattr(source, name, None)
        if isinstance(value, dict):
            return list(value.values())
        if isinstance(value, list):
            return list(value)
    for name in (f"open_{attr}", f"all_{attr}", "list"):
        method = getattr(source, name, None)
        if callable(method):
            try:
                result = method()
            except TypeError:
                continue
            if isinstance(result, list):
                return result
    return []


def _run_summarizer(name: str, fn: Any) -> str:
    """Fallback summarizer runner, mirroring focus.sessions honesty."""
    if fn is None or not callable(fn):
        return f"No data — {name} summary not connected."
    try:
        result = fn()
    except Exception:  # noqa: BLE001 — a summarizer must not kill the review
        return f"{name.capitalize()} summary unavailable (summarizer failed)."
    if result is None or result == "":
        return f"No data — no {name} activity recorded this week."
    if isinstance(result, dict):
        parts = [f"{k}: {v}" for k, v in result.items()]
        return "; ".join(parts) if parts else f"No {name} data this week."
    return str(result)


def _focus_weekly_review(*, commitments: Any, decisions: Any, focus: Any,
                         summarizers: dict | None, now: float,
                         week_start: float | None):
    """Delegate to focus.sessions.build_weekly_review when available."""
    try:
        from ..focus.sessions import build_weekly_review as _focus_build
    except Exception:  # noqa: BLE001 — fallback below composes directly
        return None
    try:
        kwargs: dict[str, Any] = {
            "commitments": commitments,
            "decisions": decisions,
            "focus": focus,
            "summarizers": summarizers,
            "now": now,
        }
        if week_start is not None:
            kwargs["week_start"] = week_start
        return _focus_build(**kwargs)
    except TypeError:
        # Older signature without week_start — retry without it.
        kwargs.pop("week_start", None)
        return _focus_build(**kwargs)


def _fallback_focus_review(*, commitments: Any, decisions: Any, focus: Any,
                           summarizers: dict | None, now: float):
    """Minimal fallback if focus.sessions is unavailable.

    Composes directly from the supplied rows only — no invented data.
    Used only when the focus builder cannot be imported; in normal
    operation the delegated builder above is used.
    """
    import datetime as _dt

    start_dt = _dt.datetime.fromtimestamp(now, tz=_dt.timezone.utc)
    monday = start_dt - _dt.timedelta(days=start_dt.weekday())
    start = _dt.datetime(monday.year, monday.month, monday.day,
                         tzinfo=_dt.timezone.utc).timestamp()
    end = start + 7 * 86400
    week_label = (
        f"{_dt.datetime.fromtimestamp(start, tz=_dt.timezone.utc).strftime('%d %b')} – "
        f"{_dt.datetime.fromtimestamp(end - 1, tz=_dt.timezone.utc).strftime('%d %b %Y')}"
    )
    summaries = summarizers or {}
    return {
        "week_start": start, "week_end": end, "week_label": week_label,
        "wins": [],
        "open_commitments": [
            {
                "id": _get(r, "id", default=""),
                "what": _get(r, "what", default=""),
                "promisee": _get(r, "promisee"),
                "deadline": _get(r, "deadline"),
                "status": _get(r, "status", default="open"),
                "deadline_label": None,
            }
            for r in _rows_of(commitments, "commitments")
            if _get(r, "status", default="open") == "open"
        ],
        "decisions_resolved": [],
        "decisions_open": [
            {
                "id": _get(r, "id", default=""),
                "decision": _get(r, "decision", default=""),
                "chosen": _get(r, "chosen", default=""),
                "status": _get(r, "status", default="open"),
            }
            for r in _rows_of(decisions, "decisions")
            if _get(r, "status", default="open") == "open"
        ],
        "calibration_sentence": None,
        "focus_summary": {"sessions": 0, "total_seconds": 0.0, "by_label": []},
        "health_summary": _run_summarizer("health", summaries.get("health")),
        "learn_summary": _run_summarizer("learn", summaries.get("learn")),
        "finance_summary": _run_summarizer("finance", summaries.get("finance")),
        "next_week_plan": [],
    }


# ---------------------------------------------------------------------------
# Weekly review
# ---------------------------------------------------------------------------


@dataclass
class WeeklyReview:
    """One week's synthesis, built on top of the focus weekly review."""

    week_start: float = 0.0
    week_end: float = 0.0
    week_label: str = ""
    # What happened: wins + summarizer lines (health/learn/finance).
    happened: list[str] = field(default_factory=list)
    wins: list[str] = field(default_factory=list)
    summaries: dict[str, str] = field(default_factory=dict)
    # Overdue open commitments (subset of the focus open_commitments).
    slipped: list[dict[str, Any]] = field(default_factory=list)
    # Decisions still open + focus sessions with interruptions.
    blocked: list[dict[str, Any]] = field(default_factory=list)
    # Focus totals, straight from the focus builder.
    focus_totals: dict[str, Any] = field(default_factory=dict)
    # Carry-overs: slipped + blocked + focus next-week plan, deduped.
    next_week: list[str] = field(default_factory=list)
    # Reliability for the week: kept / (kept + broken) or None.
    kept: int = 0
    broken: int = 0
    reliability: float | None = None
    calibration_sentence: str | None = None
    # Raw focus-builder payload, kept for callers that need the full detail.
    focus_review: dict[str, Any] = field(default_factory=dict)

    # Dict-like access for callers used to the focus WeeklyReview dict.
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "week_start": self.week_start, "week_end": self.week_end,
            "week_label": self.week_label, "happened": list(self.happened),
            "wins": list(self.wins), "summaries": dict(self.summaries),
            "slipped": [dict(r) for r in self.slipped],
            "blocked": [dict(r) for r in self.blocked],
            "focus_totals": dict(self.focus_totals),
            "next_week": list(self.next_week),
            "kept": self.kept, "broken": self.broken,
            "reliability": self.reliability,
            "calibration_sentence": self.calibration_sentence,
        }

    def render_markdown(self) -> str:
        return render_markdown(self)


def _focus_interruptions(focus: Any, start: float | None = None,
                         end: float | None = None) -> list[dict[str, Any]]:
    """Per-session interruption rows, read (not recomputed) from focus.

    Scoped to [start, end) when bounds are given: a session outside
    the reviewed week is not this week's blocker.
    """
    out: list[dict[str, Any]] = []
    for row in _rows_of(focus, "sessions"):
        if start is not None and end is not None:
            started = _get(row, "started_at")
            ended = _get(row, "ended_at")
            if started is not None and ended is not None:
                if not (started < end and ended > start):
                    continue
            elif started is not None and started >= end:
                continue
        interruptions = _get(row, "interruptions", default=0) or 0
        if interruptions:
            out.append({
                "kind": "focus",
                "label": _get(row, "label", default="") or "(untitled)",
                "id": _get(row, "id", default=""),
                "interruptions": int(interruptions),
                "notes": list(_get(row, "notes", default=[]) or []),
            })
    return out


def build_weekly_review(*, commitments: Any = None, decisions: Any = None,
                        focus: Any = None,
                        summarizers: dict | None = None,
                        now: float | None = None,
                        week_start: float | None = None) -> WeeklyReview:
    """Build the week's review.

    Delegates to ``nomorals.focus.sessions.build_weekly_review`` for
    wins / open commitments / decisions / focus totals / calibration /
    summarizer sections, then adds slipped / blocked / next-week on top.
    ``commitments`` / ``decisions`` / ``focus`` accept the same shapes
    the focus builder accepts (ledger/journal/timer, list, or None).
    """
    now_f = float(now if now is not None else time.time())
    base = _focus_weekly_review(
        commitments=commitments, decisions=decisions, focus=focus,
        summarizers=summarizers, now=now_f, week_start=week_start)
    if base is None:
        base = _fallback_focus_review(
            commitments=commitments, decisions=decisions, focus=focus,
            summarizers=summarizers, now=now_f)
    # base is a dict (focus WeeklyReview) — read it, don't rebuild it.
    wins: list[str] = list(base.get("wins") or [])
    summaries = {
        "health": base.get("health_summary") or "No data — health summary not connected.",
        "learn": base.get("learn_summary") or "No data — learn summary not connected.",
        "finance": base.get("finance_summary") or "No data — finance summary not connected.",
    }
    happened: list[str] = list(wins)
    for name in ("health", "learn", "finance"):
        happened.append(f"{name}: {summaries[name]}")

    # ── slipped: overdue open commitments ──
    slipped: list[dict[str, Any]] = []
    for row in base.get("open_commitments") or []:
        deadline = row.get("deadline") if isinstance(row, dict) else None
        if deadline is not None and deadline < now_f:
            slipped.append(dict(row) if isinstance(row, dict) else {"what": str(row)})

    # ── blocked: open decisions + focus interruptions ──
    blocked: list[dict[str, Any]] = []
    for row in base.get("decisions_open") or []:
        if isinstance(row, dict):
            blocked.append({
                "kind": "decision",
                "id": row.get("id", ""),
                "decision": row.get("decision", ""),
                "chosen": row.get("chosen", ""),
                "created_at": row.get("created_at"),
            })
        else:
            blocked.append({"kind": "decision", "decision": str(row)})
    week_start_f = float(base.get("week_start") or 0.0)
    week_end_f = float(base.get("week_end") or 0.0)
    blocked.extend(_focus_interruptions(focus, week_start_f, week_end_f))
    # If the focus source was already consumed via the builder and the
    # caller passed only a summary, fall back to by_label interruption
    # counts from the focus totals so interruptions are not lost.
    if not _focus_interruptions(focus, week_start_f, week_end_f):
        for bucket in (base.get("focus_summary") or {}).get("by_label") or []:
            if bucket.get("interruptions"):
                blocked.append({
                    "kind": "focus",
                    "label": bucket.get("label", "(untitled)"),
                    "interruptions": int(bucket.get("interruptions") or 0),
                    "notes": [],
                })

    focus_totals = dict(base.get("focus_summary") or {})

    # ── reliability: kept / (kept + broken) from the commitment rows ──
    # Count from the focus builder's normalised view where possible:
    # wins carry kept promises, but broken is not in the focus review,
    # so count from the raw commitment rows when available.
    kept = broken = 0
    seen_ids: set[str] = set()
    for row in _rows_of(commitments, "commitments"):
        status = _get(row, "status", default="open")
        if status == "kept":
            kept += 1
        elif status == "broken":
            broken += 1
        seen_ids.add(str(_get(row, "id", default="")))
    # If the caller passed a ledger object, _rows_of(commitments) may
    # have returned only open rows via open_commitments(); also try the
    # full mapping so kept/broken are not undercounted.
    if hasattr(commitments, "commitments") and isinstance(
            getattr(commitments, "commitments"), dict):
        for row in commitments.commitments.values():  # type: ignore[union-attr]
            rid = str(_get(row, "id", default=""))
            if rid in seen_ids:
                continue
            status = _get(row, "status", default="open")
            if status == "kept":
                kept += 1
            elif status == "broken":
                broken += 1
            seen_ids.add(rid)
    resolved = kept + broken
    reliability = (kept / resolved) if resolved else None

    # ── next week: carry-overs, de-duplicated, slipped first ──
    next_week: list[str] = []
    for row in slipped:
        who = f" to {row.get('promisee')}" if row.get("promisee") else ""
        due = row.get("deadline_label") or "no deadline set"
        next_week.append(f"Keep (overdue): {row.get('what', '')}{who} (due {due})")
    for entry in blocked:
        if entry.get("kind") == "decision":
            next_week.append(f"Decide: {entry.get('decision', '')}")
    for item in base.get("next_week_plan") or []:
        if item not in next_week:
            next_week.append(item)

    return WeeklyReview(
        week_start=float(base.get("week_start") or 0.0),
        week_end=float(base.get("week_end") or 0.0),
        week_label=str(base.get("week_label") or ""),
        happened=happened,
        wins=wins,
        summaries=summaries,
        slipped=slipped,
        blocked=blocked,
        focus_totals=focus_totals,
        next_week=next_week,
        kept=kept,
        broken=broken,
        reliability=reliability,
        calibration_sentence=base.get("calibration_sentence"),
        focus_review=dict(base),
    )


# ---------------------------------------------------------------------------
# Monthly review
# ---------------------------------------------------------------------------


@dataclass
class MonthlyReview:
    """Aggregation of weekly reviews (typically 4)."""

    week_count: int = 0
    period_label: str = ""
    weeks: list[WeeklyReview] = field(default_factory=list)
    totals: dict[str, Any] = field(default_factory=dict)
    focus_totals: dict[str, Any] = field(default_factory=dict)
    reliability_trend: list[dict[str, Any]] = field(default_factory=list)
    top_slipped: list[dict[str, Any]] = field(default_factory=list)
    top_blocked: list[dict[str, Any]] = field(default_factory=list)
    next_month: list[str] = field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "week_count": self.week_count,
            "period_label": self.period_label,
            "totals": dict(self.totals),
            "focus_totals": dict(self.focus_totals),
            "reliability_trend": [dict(r) for r in self.reliability_trend],
            "top_slipped": [dict(r) for r in self.top_slipped],
            "top_blocked": [dict(r) for r in self.top_blocked],
            "next_month": list(self.next_month),
        }

    def render_markdown(self) -> str:
        return render_markdown(self)


def _coerce_weekly(item: Any) -> WeeklyReview | None:
    """Accept our WeeklyReview, a focus WeeklyReview dict, or a plain dict."""
    if isinstance(item, WeeklyReview):
        return item
    if isinstance(item, dict):
        # Focus-builder dict shape.
        wins = list(item.get("wins") or [])
        slipped: list[dict[str, Any]] = []
        for row in item.get("open_commitments") or []:
            if isinstance(row, dict):
                label = str(row.get("deadline_label") or "")
                if "OVERDUE" in label or (
                        row.get("deadline") is not None):
                    # Only count as slipped when we can prove overdue;
                    # deadline_label OVERDUE is the focus builder's proof.
                    if "OVERDUE" in label:
                        slipped.append(dict(row))
        blocked: list[dict[str, Any]] = []
        for row in item.get("decisions_open") or []:
            blocked.append({"kind": "decision",
                            "decision": row.get("decision", "") if isinstance(row, dict) else str(row),
                            "chosen": row.get("chosen", "") if isinstance(row, dict) else ""})
        focus_totals = dict(item.get("focus_summary") or {})
        return WeeklyReview(
            week_start=float(item.get("week_start") or 0.0),
            week_end=float(item.get("week_end") or 0.0),
            week_label=str(item.get("week_label") or ""),
            happened=list(wins),
            wins=wins,
            summaries={
                "health": item.get("health_summary") or "",
                "learn": item.get("learn_summary") or "",
                "finance": item.get("finance_summary") or "",
            },
            slipped=slipped,
            blocked=blocked,
            focus_totals=focus_totals,
            next_week=list(item.get("next_week_plan") or []),
            calibration_sentence=item.get("calibration_sentence"),
            focus_review=dict(item),
        )
    return None


def build_monthly_review(*, commitments: Any = None, decisions: Any = None,
                         focus: Any = None,
                         summarizers: dict | None = None,
                         now: float | None = None,
                         weeks: list[Any] | None = None,
                         week_start: float | None = None) -> MonthlyReview:
    """Aggregate weekly reviews into a monthly review.

    ``weeks`` is a list of weekly reviews (our :class:`WeeklyReview`
    or focus-builder dicts).  When ``weeks`` is empty/None, a single
    weekly review is built from the supplied inputs so the monthly
    review still reflects real data — it reports ``week_count`` 1 in
    that case and never fabricates the missing weeks.
    """
    weekly_list: list[WeeklyReview] = []
    if weeks:
        for item in weeks:
            coerced = _coerce_weekly(item)
            if coerced is not None:
                weekly_list.append(coerced)
    if not weekly_list:
        weekly_list = [build_weekly_review(
            commitments=commitments, decisions=decisions, focus=focus,
            summarizers=summarizers, now=now, week_start=week_start)]

    total_sessions = sum(int((w.focus_totals or {}).get("sessions", 0) or 0)
                         for w in weekly_list)
    total_seconds = sum(float((w.focus_totals or {}).get("total_seconds", 0.0) or 0.0)
                        for w in weekly_list)
    total_wins = sum(len(w.wins) for w in weekly_list)
    total_slipped = sum(len(w.slipped) for w in weekly_list)
    total_blocked = sum(len(w.blocked) for w in weekly_list)
    total_interruptions = sum(
        int(b.get("interruptions", 0) or 0)
        for w in weekly_list
        for b in w.blocked if b.get("kind") == "focus")
    # by_label focus totals across weeks
    by_label: dict[str, dict[str, Any]] = {}
    for w in weekly_list:
        for bucket in (w.focus_totals or {}).get("by_label") or []:
            label = bucket.get("label", "(untitled)")
            acc = by_label.setdefault(label, {"seconds": 0.0, "sessions": 0,
                                              "interruptions": 0})
            acc["seconds"] += float(bucket.get("seconds", 0.0) or 0.0)
            acc["sessions"] += int(bucket.get("sessions", 0) or 0)
            acc["interruptions"] += int(bucket.get("interruptions", 0) or 0)

    # Reliability trend: one entry per week (None when no resolved data).
    reliability_trend: list[dict[str, Any]] = []
    for w in weekly_list:
        reliability_trend.append({
            "week_label": w.week_label,
            "kept": w.kept, "broken": w.broken,
            "reliability": w.reliability,
        })

    # Top slipped / blocked: frequency across weeks, most frequent first.
    def _top(rows: list[dict[str, Any]], key_fn: Any) -> list[dict[str, Any]]:
        counts: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = key_fn(row)
            if not key:
                continue
            entry = counts.setdefault(key, {"item": key, "count": 0,
                                            "example": row})
            entry["count"] += 1
        return sorted(counts.values(), key=lambda e: e["count"], reverse=True)

    all_slipped = [r for w in weekly_list for r in w.slipped]
    all_blocked = [r for w in weekly_list for r in w.blocked]
    top_slipped = _top(
        all_slipped,
        lambda r: str(r.get("what") or r.get("id") or ""))
    top_blocked = _top(
        all_blocked,
        lambda r: str(r.get("decision") or r.get("label") or ""))

    # Next month: carry-overs from the latest week's next_week, plus
    # any slipped/blocked that recurred (top items) — deduped.
    next_month: list[str] = []
    if weekly_list:
        for item in weekly_list[-1].next_week:
            if item not in next_month:
                next_month.append(item)
    for entry in top_slipped:
        line = f"Keep (recurring): {entry['item']}"
        if line not in next_month:
            next_month.append(line)
    for entry in top_blocked:
        line = f"Unblock: {entry['item']}"
        if line not in next_month:
            next_month.append(line)

    period_label = ""
    if weekly_list:
        first = weekly_list[0].week_label
        last = weekly_list[-1].week_label
        period_label = f"{first} → {last}" if first != last else first

    totals = {
        "weeks": len(weekly_list),
        "wins": total_wins,
        "slipped": total_slipped,
        "blocked": total_blocked,
        "focus_sessions": total_sessions,
        "focus_seconds": total_seconds,
        "focus_interruptions": total_interruptions,
    }
    focus_totals = {
        "sessions": total_sessions,
        "total_seconds": total_seconds,
        "by_label": [{"label": label, **stats}
                     for label, stats in sorted(
                         by_label.items(),
                         key=lambda kv: kv[1]["seconds"], reverse=True)],
    }
    return MonthlyReview(
        week_count=len(weekly_list),
        period_label=period_label,
        weeks=weekly_list,
        totals=totals,
        focus_totals=focus_totals,
        reliability_trend=reliability_trend,
        top_slipped=top_slipped,
        top_blocked=top_blocked,
        next_month=next_month,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _format_duration(seconds: float) -> str:
    try:
        from ..focus.sessions import format_duration as _fmt
        return _fmt(seconds)
    except Exception:  # noqa: BLE001
        total = max(0, int(seconds or 0))
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours}h {minutes:02d}m"
        if minutes:
            return f"{minutes}m"
        return f"{secs}s"


def _render_weekly(review: WeeklyReview) -> str:
    lines = [f"# Weekly Review — {review.week_label}"]
    lines.append("")
    lines.append("## What happened")
    if review.happened:
        lines.extend(f"- {line}" for line in review.happened)
    else:
        lines.append("- Nothing recorded this week.")
    lines.append("")
    lines.append("## What slipped")
    if review.slipped:
        for row in review.slipped:
            who = f" → {row.get('promisee')}" if row.get("promisee") else ""
            due = row.get("deadline_label") or "no deadline set"
            lines.append(f"- {row.get('what', '')}{who} (due {due})")
    else:
        lines.append("- Nothing slipped this week.")
    lines.append("")
    lines.append("## What's blocked")
    if review.blocked:
        for entry in review.blocked:
            if entry.get("kind") == "decision":
                lines.append(f"- ⏳ Decision: {entry.get('decision', '')}"
                             f" → {entry.get('chosen') or 'undecided'}")
            else:
                lines.append(f"- 🛑 Focus interruptions: "
                             f"{entry.get('label', '')} "
                             f"({entry.get('interruptions', 0)} interruptions)")
    else:
        lines.append("- Nothing blocked.")
    lines.append("")
    lines.append("## Focus")
    ft = review.focus_totals or {}
    lines.append(f"- {ft.get('sessions', 0)} sessions, "
                 f"{_format_duration(ft.get('total_seconds', 0.0))} focused")
    for bucket in ft.get("by_label") or []:
        lines.append(f"  - {bucket.get('label')}: "
                     f"{_format_duration(bucket.get('seconds', 0.0))} "
                     f"({bucket.get('sessions', 0)} sessions, "
                     f"{bucket.get('interruptions', 0)} interruptions)")
    if review.calibration_sentence:
        lines.append(f"- {review.calibration_sentence}")
    if review.reliability is not None:
        lines.append(f"- Reliability: {review.reliability:.0%} "
                     f"({review.kept} kept / {review.kept + review.broken} resolved)")
    else:
        lines.append("- Reliability: no resolved commitments to measure yet.")
    lines.append("")
    lines.append("## Next week")
    if review.next_week:
        lines.extend(f"- {item}" for item in review.next_week)
    else:
        lines.append("- Nothing carried over. A clean week.")
    return "\n".join(lines)


def _render_monthly(review: MonthlyReview) -> str:
    lines = [f"# Monthly Review — {review.period_label or 'aggregated weeks'}"]
    lines.append("")
    totals = review.totals or {}
    lines.append("## What happened")
    lines.append(f"- {totals.get('weeks', 0)} weeks reviewed, "
                 f"{totals.get('wins', 0)} wins recorded.")
    lines.append(f"- Focus: {totals.get('focus_sessions', 0)} sessions, "
                 f"{_format_duration(totals.get('focus_seconds', 0.0))} focused, "
                 f"{totals.get('focus_interruptions', 0)} interruptions.")
    lines.append("")
    lines.append("## What slipped")
    if review.top_slipped:
        for entry in review.top_slipped:
            lines.append(f"- {entry['item']} (×{entry['count']})")
    elif totals.get("slipped", 0):
        lines.append(f"- {totals.get('slipped')} slipped items across the month.")
    else:
        lines.append("- Nothing slipped this month.")
    lines.append("")
    lines.append("## What's blocked")
    if review.top_blocked:
        for entry in review.top_blocked:
            lines.append(f"- {entry['item']} (×{entry['count']})")
    elif totals.get("blocked", 0):
        lines.append(f"- {totals.get('blocked')} blocked items across the month.")
    else:
        lines.append("- Nothing blocked.")
    lines.append("")
    lines.append("## Focus")
    ft = review.focus_totals or {}
    lines.append(f"- {ft.get('sessions', 0)} sessions, "
                 f"{_format_duration(ft.get('total_seconds', 0.0))} focused")
    for bucket in ft.get("by_label") or []:
        lines.append(f"  - {bucket.get('label')}: "
                     f"{_format_duration(bucket.get('seconds', 0.0))} "
                     f"({bucket.get('sessions', 0)} sessions)")
    lines.append("")
    lines.append("## Reliability trend")
    if review.reliability_trend:
        for entry in review.reliability_trend:
            rel = entry.get("reliability")
            text = f"{rel:.0%}" if rel is not None else "no data"
            lines.append(f"- {entry.get('week_label')}: {text} "
                         f"({entry.get('kept', 0)} kept / "
                         f"{entry.get('broken', 0)} broken)")
    else:
        lines.append("- No reliability data.")
    lines.append("")
    lines.append("## Next month")
    if review.next_month:
        lines.extend(f"- {item}" for item in review.next_month)
    else:
        lines.append("- Nothing carried over. A clean month.")
    return "\n".join(lines)


def render_markdown(review: Any) -> str:
    """Render a weekly or monthly review as markdown.

    Accepts our :class:`WeeklyReview` / :class:`MonthlyReview`, or a
    focus-builder dict (rendered as a weekly review via coercion).
    Sections: What happened / What slipped / What's blocked / Focus /
    Next week (or month).  Missing summarizer → "not connected",
    failing → "unavailable" — those strings come from the focus
    builder's summarizer handling and are preserved verbatim.
    """
    if isinstance(review, MonthlyReview):
        return _render_monthly(review)
    if isinstance(review, WeeklyReview):
        return _render_weekly(review)
    if isinstance(review, dict):
        coerced = _coerce_weekly(review)
        if coerced is not None:
            return _render_weekly(coerced)
        return "No review data."
    return "No review data."
