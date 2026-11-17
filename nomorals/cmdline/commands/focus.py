"""``nm focus`` — focus sessions from the console.

The top-level twin of chat /focus (``agents/partner/
runtime_reach.py: _control_focus``): same verbs, same rendering —
start/pause/resume/stop/status/today/week/month — over the same
FocusTimer at ``<home>/data/focus.json``. Week/month render through
``reviews.review.build_weekly_review`` / ``build_monthly_review``, the
same composer chat uses, never a second copy. Only the command names
in usage strings differ (``nm focus`` for ``/focus``); the Telegram
button wrappers have no console equivalent, so the text lines print
bare.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path
from typing import Any

__all__ = ["_cmd_focus"]


def _home_path(context: Any) -> Path | None:
    settings = getattr(context, "settings", None)
    if settings is None:
        return None
    home = getattr(settings, "home_path", None) or getattr(settings, "home", None)
    return Path(home) if home else None


def _timer(context: Any) -> Any:
    from ...focus.sessions import FocusTimer

    home = _home_path(context)
    if home is None:
        raise ValueError("no home path on context")
    return FocusTimer(str(Path(home) / "data" / "focus.json"))


def _ledger(context: Any) -> Any:
    from ...commitments.ledger import CommitmentLedger

    home = _home_path(context)
    if home is None:
        raise ValueError("no home path on context")
    return CommitmentLedger(str(Path(home) / "data" / "commitments.json"))


def _journal(context: Any) -> Any:
    from ...decisions.journal import DecisionJournal

    home = _home_path(context)
    if home is None:
        raise ValueError("no home path on context")
    return DecisionJournal(str(Path(home) / "data" / "decisions.json"))


def _usage_error(message: str) -> int:
    print(message, file=sys.stderr)
    return 2


def _today_bounds() -> tuple[float, float]:
    now = time.time()
    start = _dt.datetime.fromtimestamp(now).replace(
        hour=0, minute=0, second=0, microsecond=0).timestamp()
    return start, now


def _cmd_focus(args: argparse.Namespace, context: Any) -> int:
    """``nm focus [start [label]|pause|resume|stop [note]|status|today|week|month]``."""
    from ...focus.sessions import format_duration

    action = (getattr(args, "focus_action", None) or "status").lower()
    rest = list(getattr(args, "focus_rest", None) or [])
    as_json = bool(getattr(args, "json", False))
    try:
        timer = _timer(context)
    except Exception as exc:  # noqa: BLE001 - timer path is the whole command
        return _usage_error(f"focus unavailable: {exc}")

    if action == "start":
        try:
            s = timer.start(label=" ".join(rest).strip())
        except ValueError as exc:
            return _usage_error(str(exc))
        label = f" ({s.label})" if s.label else ""
        if as_json:
            print(json.dumps({"session": s.to_dict()}, indent=2))
            return 0
        print(f"focus started{label}.")
        return 0

    if action == "pause":
        try:
            s = timer.pause()
        except (KeyError, ValueError) as exc:
            return _usage_error(str(exc))
        if as_json:
            print(json.dumps({"session": s.to_dict()}, indent=2))
            return 0
        print(f"paused ({s.label or 'untitled'}).")
        return 0

    if action == "resume":
        try:
            s = timer.resume()
        except (KeyError, ValueError) as exc:
            return _usage_error(str(exc))
        if as_json:
            print(json.dumps({"session": s.to_dict()}, indent=2))
            return 0
        print(f"resumed ({s.label or 'untitled'}).")
        return 0

    if action == "stop":
        try:
            s = timer.stop(note=" ".join(rest).strip())
        except (KeyError, ValueError) as exc:
            return _usage_error(str(exc))
        if as_json:
            print(json.dumps({"session": s.to_dict()}, indent=2))
            return 0
        total = format_duration(s.elapsed())
        print(f"stopped ({s.label or 'untitled'}) — {total} focused.")
        return 0

    if action == "today":
        start, now = _today_bounds()
        summary = timer.summary(start=start, end=now)
        if as_json:
            print(json.dumps(summary, indent=2))
            return 0
        lines = [f"today: {format_duration(summary['total_seconds'])} "
                 f"across {summary['sessions']} session(s)"]
        for row in summary["by_label"]:
            lines.append(f"  {row['label']}: {format_duration(row['seconds'])} "
                         f"({row['sessions']}×, {row['interruptions']} interruptions)")
        print("\n".join(lines))
        return 0

    if action == "week":
        from ...reviews.review import build_weekly_review

        try:
            review = build_weekly_review(
                commitments=_ledger(context), decisions=_journal(context),
                focus=timer)
        except Exception as exc:  # noqa: BLE001 - a broken ledger never fails a command
            return _usage_error(f"weekly review unavailable: {exc}")
        if as_json:
            print(json.dumps(review.to_dict(), indent=2))
            return 0
        print(review.render_markdown())
        return 0

    if action == "month":
        from ...reviews.review import (
            build_monthly_review,
            build_weekly_review,
        )

        try:
            ledger = _ledger(context)
            journal = _journal(context)
            now = time.time()
            weeks = []
            for back in range(4):
                wk_now = now - back * 7 * 86400.0
                weeks.append(build_weekly_review(
                    commitments=ledger, decisions=journal, focus=timer,
                    now=wk_now))
            review = build_monthly_review(weeks=weeks, now=now)
        except Exception as exc:  # noqa: BLE001 - a broken ledger never fails a command
            return _usage_error(f"monthly review unavailable: {exc}")
        if as_json:
            print(json.dumps(review.to_dict(), indent=2))
            return 0
        print(review.render_markdown())
        return 0

    if action == "status":
        active = timer.active()
        start, now = _today_bounds()
        today = timer.summary(start=start, end=now)
        today_line = (f"today: {format_duration(today['total_seconds'])} "
                      f"across {today['sessions']} session(s)")
        if as_json:
            print(json.dumps({
                "active": active.to_dict() if active is not None else None,
                "today": today}, indent=2))
            return 0
        if active is None:
            print(f"no focus session running.\n{today_line}\n"
                  "nm focus start [label] to begin.")
            return 0
        state = "paused" if active.is_paused() else "running"
        label = f" ({active.label})" if active.label else ""
        print(f"focus {state}{label} — {format_duration(active.elapsed())} so far.\n"
              f"{today_line}")
        return 0

    return _usage_error(
        "usage: nm focus start [label] · pause · resume · stop [note]\n"
        "       nm focus status · today · week · month")
