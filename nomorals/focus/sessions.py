"""Focus sessions + weekly review — the week layer.

Pulses (``nomorals.autonomy.presence``) cover the *day*: briefings,
nudges, serendipity. This module covers the *week*: deep-work sessions
and a review that pulls the whole week together — wins, open
commitments, decisions, health / learn / finance — and turns it into
next week's plan.

Design rules:

* **Sessions are a ledger, not a stopwatch.** Every session is
  recorded with pauses, interruptions, and notes. Elapsed time
  subtracts paused time; a session paused at review time counts only
  the un-paused span. Nothing is estimated.
* **The review aggregates, never invents.** Wins come from the
  commitment ledger (kept this week) and the decision journal
  (successful resolutions this week). Health / learn / finance sections
  come from *injected summarizer callables* — if a summarizer is not
  connected, the section says so honestly instead of fabricating a
  summary. The review takes ledger/journal/timer objects duck-typed
  (or plain lists/dicts), so it never imports sibling packages.
* **Next week is derived from this week.** Carry-overs are the open
  commitments and open decisions; focus targets continue the week's
  top label. No generic "do better" filler.
* **Universal.** Week starts Monday (ISO), labels are free text in any
  language, times are epoch seconds — the caller's timezone decides
  what "Monday" means (see ``nomorals.pulses.timezone``).

Example::

    >>> timer = FocusTimer()
    >>> s = timer.start("codebeast training")
    >>> timer.stop(s.id, note="smoke test green").status
    'done'
    >>> review = build_weekly_review(focus=timer)
    >>> review["focus_summary"]["sessions"]
    1

Only the standard library is imported (see the layering note in
:mod:`nomorals.commitments.ledger`).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = [
    "FocusSession",
    "FocusTimer",
    "WeeklyReview",
    "build_weekly_review",
    "week_bounds",
    "format_duration",
]

#: Summarizer sections the weekly review asks for. Keys of the
#: ``summarizers`` mapping passed to :func:`build_weekly_review`.
SUMMARY_SECTIONS = ("health", "learn", "finance")


# ── helpers ──────────────────────────────────────────────────────────

def format_duration(seconds: float) -> str:
    """``7380`` → ``2h 3m``; ``900`` → ``15m``; ``45`` → ``45s``."""
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m"
    return f"{secs}s"


def week_bounds(now: float | None = None,
                week_start: float | None = None) -> tuple[float, float]:
    """(week_start, week_end) epoch seconds, Monday 00:00 UTC → +7d.

    Pass ``week_start`` explicitly to review a different week (e.g. a
    caller in another timezone who already resolved "Monday" locally).
    """
    import datetime as _dt
    if week_start is not None:
        start = float(week_start)
        return start, start + 7 * 86400
    ref = _dt.datetime.fromtimestamp(
        now if now is not None else time.time(), tz=_dt.timezone.utc)
    monday = ref - _dt.timedelta(days=ref.weekday())
    start = _dt.datetime(monday.year, monday.month, monday.day,
                         tzinfo=_dt.timezone.utc).timestamp()
    return start, start + 7 * 86400


def _rows_of(source: Any, attr: str) -> list[Any]:
    """Normalise a source to a list of rows.

    Accepts a ledger/journal/timer object (``.commitments`` /
    ``.decisions`` / ``.sessions`` mapping or ``.open_*`` list), a
    plain list, or ``None``. Duck-typed on purpose: this module never
    imports the sibling packages.
    """
    if source is None:
        return []
    if isinstance(source, list):
        return source
    if isinstance(source, dict):
        return list(source.values())
    for name in (attr, "entries", "rows"):
        value = getattr(source, name, None)
        if isinstance(value, dict):
            return list(value.values())
        if isinstance(value, list):
            return list(value)
    # Objects exposing only an accessor (e.g. open_commitments()).
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


def _get(row: Any, *names: str, default: Any = None) -> Any:
    """Attribute-or-key read: works for dataclasses and plain dicts."""
    for name in names:
        if isinstance(row, dict):
            if name in row and row[name] is not None:
                return row[name]
        elif hasattr(row, name):
            value = getattr(row, name)
            if value is not None:
                return value
    return default


# ── focus sessions ───────────────────────────────────────────────────

@dataclass
class FocusSession:
    """One deep-work block: start → pauses → stop, with the receipts."""

    id: str
    label: str = ""
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    pauses: list[list[float]] = field(default_factory=list)
    # pauses: [pause_ts, resume_ts | None] pairs; an open pair means
    # the session is currently paused.
    status: str = "active"  # active | paused | done
    interruptions: int = 0
    notes: list[str] = field(default_factory=list)

    def elapsed(self, now: float | None = None) -> float:
        """Focused seconds: wall span minus paused spans."""
        now = time.time() if now is None else now
        end = self.ended_at if self.ended_at is not None else now
        paused = 0.0
        for pair in self.pauses:
            pause_at = pair[0]
            resume_at = pair[1] if len(pair) > 1 and pair[1] is not None else end
            paused += max(0.0, min(resume_at, end) - pause_at)
        return max(0.0, end - self.started_at - paused)

    def is_paused(self) -> bool:
        return bool(self.pauses) and self.pauses[-1][1] is None

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "label": self.label,
                "started_at": self.started_at, "ended_at": self.ended_at,
                "pauses": [list(p) for p in self.pauses],
                "status": self.status,
                "interruptions": self.interruptions,
                "notes": list(self.notes)}

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> FocusSession:
        return cls(
            id=str(row.get("id", "")),
            label=str(row.get("label", "")),
            started_at=float(row.get("started_at") or 0.0),
            ended_at=row.get("ended_at"),
            pauses=[list(p) for p in (row.get("pauses") or [])],
            status=str(row.get("status", "active")),
            interruptions=int(row.get("interruptions") or 0),
            notes=list(row.get("notes") or []))


class FocusTimer:
    """Start / pause / resume / stop a focus session. JSON-backed."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self.sessions: dict[str, FocusSession] = {}
        if path:
            self._load()

    # ── persistence ──────────────────────────────────────────────
    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, ValueError):
            return
        for row in raw.get("sessions") or []:
            try:
                session = FocusSession.from_dict(row)
                self.sessions[session.id] = session
            except (TypeError, ValueError):
                continue

    def _save(self) -> None:
        if not self.path:
            return
        try:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"sessions": [s.to_dict()
                                       for s in self.sessions.values()]},
                          fh, ensure_ascii=False)
            os.replace(tmp, self.path)
        except OSError:
            pass

    # ── lifecycle ────────────────────────────────────────────────
    def active(self) -> FocusSession | None:
        for session in self.sessions.values():
            if session.status in ("active", "paused"):
                return session
        return None

    def start(self, label: str = "", ts: float | None = None) -> FocusSession:
        """Start a session. Fail fast if one is already running — two
        "active" sessions would double-count the same minutes."""
        if self.active() is not None:
            raise ValueError("a focus session is already running — "
                             "stop or pause it first")
        now = float(ts if ts is not None else time.time())
        sid = "fc-" + hashlib.sha1(
            f"{label}|{now}".encode("utf-8")).hexdigest()[:12]
        session = FocusSession(id=sid, label=label, started_at=now)
        self.sessions[sid] = session
        self._save()
        return session

    def pause(self, session_id: str | None = None,
              ts: float | None = None) -> FocusSession:
        session = self._resolve(session_id)
        if session.status != "active":
            raise ValueError(f"session {session.id} is {session.status}, "
                             "cannot pause")
        now = float(ts if ts is not None else time.time())
        session.pauses.append([now, None])
        session.status = "paused"
        self._save()
        return session

    def resume(self, session_id: str | None = None,
               ts: float | None = None) -> FocusSession:
        session = self._resolve(session_id)
        if session.status != "paused":
            raise ValueError(f"session {session.id} is {session.status}, "
                             "cannot resume")
        now = float(ts if ts is not None else time.time())
        session.pauses[-1][1] = now
        session.status = "active"
        self._save()
        return session

    def interrupt(self, session_id: str | None = None,
                  note: str = "") -> FocusSession:
        """Log an interruption without stopping the session."""
        session = self._resolve(session_id)
        if session.status == "done":
            raise ValueError(f"session {session.id} is done")
        session.interruptions += 1
        if note:
            session.notes.append(f"interruption: {note}")
        self._save()
        return session

    def stop(self, session_id: str | None = None, note: str = "",
             ts: float | None = None) -> FocusSession:
        session = self._resolve(session_id)
        if session.status == "done":
            raise ValueError(f"session {session.id} already stopped")
        now = float(ts if ts is not None else time.time())
        if session.status == "paused" and session.pauses:
            session.pauses[-1][1] = now
        session.ended_at = now
        session.status = "done"
        if note:
            session.notes.append(note)
        self._save()
        return session

    def _resolve(self, session_id: str | None) -> FocusSession:
        if session_id is None:
            current = self.active()
            if current is None:
                raise ValueError("no focus session is running")
            return current
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(f"unknown focus session {session_id!r}")
        return session

    # ── queries ──────────────────────────────────────────────────
    def total_seconds(self, start: float | None = None,
                      end: float | None = None,
                      now: float | None = None) -> float:
        """Focused seconds overlapping [start, end). Unbounded when a
        bound is ``None``. Pauses are subtracted per session."""
        now = time.time() if now is None else now
        total = 0.0
        for session in self.sessions.values():
            total += _overlap_seconds(session, start, end, now)
        return total

    def summary(self, start: float | None = None,
                end: float | None = None,
                now: float | None = None) -> dict[str, Any]:
        """Per-label totals + session count for the window."""
        now = time.time() if now is None else now
        by_label: dict[str, dict[str, Any]] = {}
        count = 0
        for session in self.sessions.values():
            seconds = _overlap_seconds(session, start, end, now)
            if seconds <= 0:
                continue
            count += 1
            bucket = by_label.setdefault(
                session.label or "(untitled)",
                {"seconds": 0.0, "sessions": 0, "interruptions": 0})
            bucket["seconds"] += seconds
            bucket["sessions"] += 1
            bucket["interruptions"] += session.interruptions
        ordered = sorted(by_label.items(),
                         key=lambda kv: kv[1]["seconds"], reverse=True)
        return {
            "sessions": count,
            "total_seconds": sum(b["seconds"] for _, b in ordered),
            "by_label": [{"label": label, **stats}
                         for label, stats in ordered],
        }


def _overlap_seconds(session: FocusSession, start: float | None,
                     end: float | None, now: float) -> float:
    """Focused seconds of one session inside [start, end)."""
    wall_end = session.ended_at if session.ended_at is not None else now
    lo = max(session.started_at, start) if start is not None else session.started_at
    hi = min(wall_end, end) if end is not None else wall_end
    if hi <= lo:
        return 0.0
    paused = 0.0
    for pair in session.pauses:
        pause_at = pair[0]
        resume_at = pair[1] if len(pair) > 1 and pair[1] is not None else wall_end
        overlap_lo = max(pause_at, lo)
        overlap_hi = min(resume_at, hi)
        if overlap_hi > overlap_lo:
            paused += overlap_hi - overlap_lo
    return max(0.0, (hi - lo) - paused)


# ── weekly review ────────────────────────────────────────────────────

def _normalise_commitments(source: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _rows_of(source, "commitments"):
        rows.append({
            "id": _get(row, "id", default=""),
            "promisee": _get(row, "promisee"),
            "what": _get(row, "what", default=""),
            "deadline": _get(row, "deadline"),
            "status": _get(row, "status", default="open"),
            "created_at": _get(row, "created_at", default=0.0) or 0.0,
            "closed_at": _get(row, "closed_at"),
        })
    return rows


def _normalise_decisions(source: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _rows_of(source, "decisions"):
        success = _get(row, "outcome_success", "success")
        if success is None and isinstance(row, dict):
            history = row.get("history") or []
            if history:
                success = history[-1].get("success")
        rows.append({
            "id": _get(row, "id", default=""),
            "decision": _get(row, "decision", default=""),
            "chosen": _get(row, "chosen", default=""),
            "outcome": _get(row, "outcome", default=""),
            "success": success,
            "confidence": _get(row, "confidence"),
            "status": _get(row, "status", default="open"),
            "created_at": _get(row, "created_at", default=0.0) or 0.0,
            "resolved_at": _get(row, "resolved_at"),
        })
    return rows


def _normalise_sessions(source: Any) -> list[FocusSession]:
    rows: list[FocusSession] = []
    for row in _rows_of(source, "sessions"):
        if isinstance(row, FocusSession):
            rows.append(row)
        elif isinstance(row, dict):
            try:
                rows.append(FocusSession.from_dict(row))
            except (TypeError, ValueError):
                continue
    return rows


def _run_summarizer(name: str, fn: Any) -> str:
    """Run one injected summarizer. Missing / failing → honest text,
    never a fabricated summary."""
    if fn is None:
        return f"No data — {name} summary not connected."
    if not callable(fn):
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


class WeeklyReview(dict):
    """A weekly review dict with a markdown renderer."""

    def render_markdown(self) -> str:
        lines = [f"# Weekly Review — {self['week_label']}"]
        lines.append("")
        wins = self.get("wins") or []
        lines.append("## Wins")
        if wins:
            lines.extend(f"- {win}" for win in wins)
        else:
            lines.append("- No recorded wins this week — the ledger and "
                         "journal have nothing closed as kept/successful.")
        lines.append("")
        lines.append("## Open commitments")
        open_rows = self.get("open_commitments") or []
        if open_rows:
            for row in open_rows:
                due = row.get("deadline_label") or "no deadline set"
                who = f" → {row['promisee']}" if row.get("promisee") else ""
                lines.append(f"- {row['what']}{who} (due {due})")
        else:
            lines.append("- None. Nothing owed to anyone.")
        lines.append("")
        lines.append("## Decisions")
        cal = self.get("calibration_sentence")
        lines.append(cal if cal else "No resolved decisions to calibrate.")
        for row in self.get("decisions_resolved") or []:
            mark = ("✅" if row.get("success") is True
                    else "❌" if row.get("success") is False else "❔")
            lines.append(f"- {mark} {row['decision']} → {row['chosen']}")
        for row in self.get("decisions_open") or []:
            lines.append(f"- ⏳ {row['decision']} → {row['chosen'] or 'undecided'}")
        if not (self.get("decisions_resolved") or self.get("decisions_open")):
            lines.append("- No decisions recorded.")
        lines.append("")
        focus = self.get("focus_summary") or {}
        lines.append("## Focus")
        lines.append(f"- {focus.get('sessions', 0)} sessions, "
                     f"{format_duration(focus.get('total_seconds', 0.0))} focused")
        for bucket in focus.get("by_label") or []:
            lines.append(f"  - {bucket['label']}: "
                         f"{format_duration(bucket['seconds'])} "
                         f"({bucket['sessions']} sessions, "
                         f"{bucket['interruptions']} interruptions)")
        lines.append("")
        for section in SUMMARY_SECTIONS:
            lines.append(f"## {section.capitalize()}")
            lines.append(self.get(f"{section}_summary")
                         or f"No data — {section} summary not connected.")
            lines.append("")
        lines.append("## Next week")
        plan = self.get("next_week_plan") or []
        if plan:
            lines.extend(f"- {item}" for item in plan)
        else:
            lines.append("- Nothing carried over. A clean week.")
        return "\n".join(lines)


def build_weekly_review(*, commitments: Any = None, decisions: Any = None,
                        focus: Any = None,
                        summarizers: dict[str, Callable[[], Any]] | None = None,
                        now: float | None = None,
                        week_start: float | None = None) -> WeeklyReview:
    """Build the week's review from the three ledgers + summarizers.

    ``commitments`` / ``decisions`` / ``focus`` accept a ledger/journal/
    timer object, a plain list of rows, or ``None``. ``summarizers`` maps
    ``"health"`` / ``"learn"`` / ``"finance"`` to zero-arg callables
    returning ``str`` / ``dict`` / ``None``. Missing summarizers yield
    honest "not connected" text, never invented content.
    """
    import datetime as _dt
    now = float(now if now is not None else time.time())
    start, end = week_bounds(now, week_start)
    label_start = _dt.datetime.fromtimestamp(start, tz=_dt.timezone.utc)
    label_end = _dt.datetime.fromtimestamp(end - 1, tz=_dt.timezone.utc)
    week_label = (f"{label_start.strftime('%d %b')} – "
                  f"{label_end.strftime('%d %b %Y')}")

    commitment_rows = _normalise_commitments(commitments)
    decision_rows = _normalise_decisions(decisions)
    session_rows = _normalise_sessions(focus)

    # ── wins: kept promises + successful decisions closed this week ──
    wins: list[str] = []
    kept = [c for c in commitment_rows
            if c["status"] == "kept"
            and c["closed_at"] is not None and start <= c["closed_at"] < end]
    for row in kept:
        who = f" to {row['promisee']}" if row.get("promisee") else ""
        wins.append(f"Kept promise{who}: {row['what']}")
    won_decisions = [d for d in decision_rows
                     if d["status"] == "resolved" and d["success"] is True
                     and d["resolved_at"] is not None
                     and start <= d["resolved_at"] < end]
    for row in won_decisions:
        wins.append(f"Decision paid off: {row['decision']} → {row['chosen']}")

    # ── open commitments (all, not just this week's) ──
    open_rows: list[dict[str, Any]] = []
    for row in sorted(
            [c for c in commitment_rows if c["status"] == "open"],
            key=lambda c: (c["deadline"] is None, c["deadline"] or 0.0)):
        deadline_label = None
        if row["deadline"] is not None:
            dt = _dt.datetime.fromtimestamp(row["deadline"],
                                            tz=_dt.timezone.utc)
            deadline_label = dt.strftime("%a, %d %b")
            if row["deadline"] < now:
                deadline_label += " (OVERDUE)"
        open_rows.append({**row, "deadline_label": deadline_label})

    # ── decisions: resolved this week + still open ──
    resolved_rows = [d for d in decision_rows
                     if d["status"] == "resolved"
                     and d["resolved_at"] is not None
                     and start <= d["resolved_at"] < end]
    open_decisions = [d for d in decision_rows if d["status"] == "open"]
    verdicts = [d for d in resolved_rows if d["success"] is not None]
    calibration_sentence = None
    if verdicts:
        accuracy = (sum(1 for d in verdicts if d["success"]) / len(verdicts))
        calibration_sentence = (
            f"Your predictions have been {accuracy:.0%} accurate across "
            f"{len(verdicts)} resolved decision"
            f"{'s' if len(verdicts) != 1 else ''} this week.")

    # ── focus: sessions overlapping the week ──
    by_label: dict[str, dict[str, Any]] = {}
    session_count = 0
    for session in session_rows:
        seconds = _overlap_seconds(session, start, end, now)
        if seconds <= 0:
            continue
        session_count += 1
        bucket = by_label.setdefault(session.label or "(untitled)",
                                     {"seconds": 0.0, "sessions": 0,
                                      "interruptions": 0})
        bucket["seconds"] += seconds
        bucket["sessions"] += 1
        bucket["interruptions"] += session.interruptions
    focus_summary = {
        "sessions": session_count,
        "total_seconds": sum(b["seconds"] for b in by_label.values()),
        "by_label": [{"label": label, **stats}
                     for label, stats in sorted(
                         by_label.items(),
                         key=lambda kv: kv[1]["seconds"], reverse=True)],
    }

    # ── injected summaries ──
    summarizers = summarizers or {}
    section_text = {name: _run_summarizer(name, summarizers.get(name))
                    for name in SUMMARY_SECTIONS}

    # ── next week: carry-overs + continue the top focus label ──
    plan: list[str] = []
    for row in open_rows:
        due = row.get("deadline_label") or "no deadline set"
        who = f" to {row['promisee']}" if row.get("promisee") else ""
        plan.append(f"Keep: {row['what']}{who} (due {due})")
    for row in open_decisions:
        plan.append(f"Decide: {row['decision']}")
    if focus_summary["by_label"]:
        top = focus_summary["by_label"][0]
        plan.append(f"Focus: continue '{top['label']}' "
                    f"({format_duration(top['seconds'])} this week)")
    else:
        plan.append("Focus: start one deep-work block.")

    return WeeklyReview({
        "week_start": start, "week_end": end, "week_label": week_label,
        "wins": wins,
        "open_commitments": open_rows,
        "decisions_resolved": resolved_rows,
        "decisions_open": open_decisions,
        "calibration_sentence": calibration_sentence,
        "focus_summary": focus_summary,
        "health_summary": section_text["health"],
        "learn_summary": section_text["learn"],
        "finance_summary": section_text["finance"],
        "next_week_plan": plan,
    })
