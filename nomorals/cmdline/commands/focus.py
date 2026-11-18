"""``nm focus`` — focus sessions from the console.

The top-level twin of chat /focus (``agents/partner/
runtime_reach.py: _control_focus``): same verbs, same rendering —
start/pause/resume/stop/status/today/week/month — over the same
FocusTimer at ``<home>/data/focus.json``. Week/month render through
``reviews.review.build_weekly_review`` / ``build_monthly_review``, the
same composer chat uses, never a second copy. Only the command names
in usage strings differ (``nm focus`` for ``/focus``); the Telegram
button wrappers have no console equivalent, so the text lines print
bare. ``mix`` mirrors chat /focus mix (MusicForProgramming #31):
list/show/play/assemble/refresh over the same FocusMixStore; play
starts the session with the mix attached — audio itself goes
through the chat play spine, so on the console the mix is planned
and the session runs.
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
        # nm focus start <label> --mix <id|title|episode N>: start the
        # session with a mix attached (MusicForProgramming #31).
        joined = " ".join(rest).strip()
        label, _, mix_ref = joined.partition("--mix")
        label = label.strip()
        mix_ref = mix_ref.strip()
        if mix_ref:
            from ...music_sources.focus_mixes import (
                FocusMixStore,
                find_mix,
                format_mix_reply,
                start_focus_session_with_mix,
            )

            home = _home_path(context)
            if home is None:
                return _usage_error("focus unavailable: no home path")
            store = FocusMixStore(home)
            mix = find_mix(store, mix_ref)
            if mix is None:
                return _usage_error(
                    f"no focus mix matches {mix_ref!r} — nm focus mix "
                    "list to see them")
            result = start_focus_session_with_mix(
                timer, store, mix, label=label)
            if as_json:
                print(json.dumps({
                    "session": result["session"].to_dict(),
                    "mix": mix.to_dict(), "play": result["play"]},
                    indent=2))
                return 0
            print(format_mix_reply(mix, result["session"], result["play"]))
            return 0
        try:
            s = timer.start(label=label)
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

    if action == "mix":
        # MusicForProgramming (#31): the store half of chat /focus mix
        # — list/show/play/assemble/refresh over the same store. Play
        # starts the session with the mix attached; audio itself goes
        # through the chat play spine, so on the console the mix is
        # planned and the session runs.
        from ...music_sources.focus_mixes import (
            MOOD_TAGS,
            FocusMixStore,
            assemble_mix,
            find_mix,
            format_mix_reply,
            refresh_from_series,
            render_mix,
            start_focus_session_with_mix,
        )
        from ...tools.focus_mix import dj_planner, read_owner_taste

        home = _home_path(context)
        if home is None:
            return _usage_error("focus unavailable: no home path")
        store = FocusMixStore(home)
        subverb = (rest[0].lower() if rest else "list")
        subrest = " ".join(rest[1:]).strip()
        if subverb in ("list", ""):
            mixes = store.list()
            if as_json:
                print(json.dumps({"mixes": [m.to_dict() for m in mixes]},
                                 indent=2))
                return 0
            if not mixes:
                print("no focus mixes yet — nm focus mix assemble builds "
                      "one from your library, nm focus mix refresh pulls "
                      "the Music For Programming series.")
                return 0
            print("focus mixes:")
            for mix in mixes:
                print(f"  {mix.id} — {mix.title} "
                      f"({len(mix.tracks)} tracks, "
                      f"{format_duration(mix.duration_s)}, played in "
                      f"{mix.sessions_played} session(s))")
            return 0
        if subverb == "show":
            mix = find_mix(store, subrest)
            if mix is None:
                return _usage_error("no focus mix matches that id or title")
            if as_json:
                print(json.dumps({"mix": mix.to_dict()}, indent=2))
                return 0
            print(render_mix(mix))
            return 0
        if subverb == "play":
            mix = find_mix(store, subrest)
            if mix is None:
                return _usage_error("no focus mix matches that id or title")
            result = start_focus_session_with_mix(timer, store, mix)
            if as_json:
                print(json.dumps({
                    "session": result["session"].to_dict(),
                    "mix": mix.to_dict(), "play": result["play"]},
                    indent=2))
                return 0
            print(format_mix_reply(mix, result["session"], result["play"]))
            return 0
        if subverb == "assemble":
            mood = subrest.strip().lower()
            tags = dict(MOOD_TAGS[mood]) if mood in MOOD_TAGS else {}
            if mood and mood not in MOOD_TAGS:
                return _usage_error(
                    f"unknown mood {mood!r} — darker | lighter | deep | "
                    "driving | calm | ambient")
            library = None
            if getattr(context, "db", None) is not None:
                try:
                    from ...media.library import MusicLibrary

                    library = MusicLibrary(context)
                except Exception:  # noqa: BLE001 - no library, honest
                    library = None
            try:
                mix = assemble_mix(
                    store, library=library,
                    feedback_rows=read_owner_taste(), tags=tags,
                    planner=dj_planner)
            except ValueError as exc:
                return _usage_error(str(exc))
            if as_json:
                print(json.dumps({"mix": mix.to_dict()}, indent=2))
                return 0
            print(render_mix(mix))
            return 0
        if subverb == "refresh":
            result = refresh_from_series(store)
            if result.get("error"):
                return _usage_error(
                    f"series refresh failed: {result['error']}")
            if as_json:
                print(json.dumps(result, indent=2))
                return 0
            print(f"series refreshed: {result['added']} new episode(s) "
                  f"of {result['total']} in the feed.")
            return 0
        return _usage_error(
            "usage: nm focus mix [list] · show <id> · play <id> · "
            "assemble [mood] · refresh")

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
        "usage: nm focus start [label] [--mix <id>] · pause · resume · stop [note]\n"
        "       nm focus status · today · week · month · mix [list|show|play|assemble|refresh]")
