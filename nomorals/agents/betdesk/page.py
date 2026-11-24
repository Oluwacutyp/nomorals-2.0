"""The bet desk signal page — Devon's HTML page for her signal calls.

There was no signal page: the calls lived only as data (the
``SignalEngine`` ledger, the desk's ``status()`` dict, chat text
via ``render_status()``, and Notifier pushes).  This page is the
first of the HTML redesign — the page the owner looks at, rendered
live from the desk on every request, in the shared shell
(:mod:`nomorals.web.shell`) the rest of the pages follow.

Order is the design: what needs a decision first (live and
upcoming calls, with the countdown the call is about), then the
record that grades the calls in the open (scored calls and the
hit rate against the lab's base rate), then what the lab says per
game (verdict, evidence, what would change it), then desk health
(sources, keys, the lead floor).  Every section has an honest
empty state — the page says "nothing here yet, and why" instead
of dressing an empty desk up as a quiet one.

Serving: ``GET /signals`` on the API server (the ``/pair``
precedent — bearer-gated, ``db.read`` like the betdesk tool).
The page is a snapshot: it stamps the moment its data was read,
and reload re-renders live.  It never places a bet and never
manages anything — that stays in chat with ``/desk``.
"""
from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from ...web.shell import esc, page_shell

__all__ = ["signal_page_html", "signal_page_for_context"]


def _field(row: Any, name: str, default: Any = "") -> Any:
    """One field off a call/record that may be a dataclass or a dict."""
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _clock(epoch: float) -> str:
    """Epoch as ``HH:MM UTC``."""
    return datetime.fromtimestamp(
        float(epoch), tz=UTC).strftime("%H:%M UTC")


def _mins(seconds: float) -> str:
    """A duration as whole minutes, never negative, never a lie about
    seconds the owner cannot act on."""
    return str(max(0, int(seconds // 60)))


def _pct(value: Any) -> str:
    """A rate as a whole percent, or an honest dash when unknown."""
    if value is None:
        return "—"
    try:
        return str(int(round(float(value) * 100))) + "%"
    except (TypeError, ValueError):
        return "—"


def _call_state(call: Any, now: float) -> str:
    """Where one call stands against the clock: the state the page
    leads with, because the call is a timed alert."""
    start = float(_field(call, "window_start", 0) or 0)
    end = float(_field(call, "window_end", 0) or 0)
    if now < start:
        return "opens in " + _mins(start - now) + "m"
    if now < end:
        return "LIVE NOW — closes in " + _mins(end - now) + "m"
    return "window passed"


def _call_rows(calls: list[Any], now: float) -> str:
    """The live/upcoming calls table, soonest window first."""
    live = []
    for call in calls:
        end = float(_field(call, "window_end", 0) or 0)
        if end > now:
            live.append(call)
    live.sort(key=lambda c: float(_field(c, "window_start", 0) or 0))
    if not live:
        return ("<p>No live or upcoming calls. When the lab confirms "
                "a time structure, the next call appears here at "
                "least 2 minutes before its window opens.</p>")
    rows = ["<table><thead><tr><th>Call</th><th>Window</th>"
            "<th>State</th><th>Target</th>"
            "<th>Expected vs base</th></tr></thead><tbody>"]
    for call in live:
        provider = esc(_field(call, "provider"))
        game = esc(_field(call, "game"))
        target = _field(call, "target", 0)
        try:
            target_text = format(float(target), ".2f") + "x"
        except (TypeError, ValueError):
            target_text = esc(target)
        start = float(_field(call, "window_start", 0) or 0)
        end = float(_field(call, "window_end", 0) or 0)
        expected = _pct(_field(call, "expected_rate"))
        base = _pct(_field(call, "base_rate"))
        rows.append(
            "<tr><td>" + provider + " " + game + "</td>"
            "<td>" + _clock(start) + " – " + _clock(end) + "</td>"
            "<td><b>" + esc(_call_state(call, now)) + "</b></td>"
            "<td>" + target_text + "</td>"
            "<td>" + expected + " vs " + base + "</td></tr>")
    rows.append("</tbody></table>")
    return "".join(rows)


def _scored_rows(calls: list[Any]) -> str:
    """The scored calls table and the calibration line — the calls
    grading themselves in the open."""
    done = [c for c in calls if _field(c, "scored") is not None]
    if not done:
        return ("<p>No calls scored yet. A call scores once its "
                "window has fully passed — hit means some round in "
                "the window reached the target.</p>")
    hits = sum(1 for c in done if _field(c, "scored"))
    rate = hits / len(done) if done else 0.0
    parts = [
        "<p><b>" + str(hits) + "/" + str(len(done)) + " scored calls "
        "hit (" + str(int(round(rate * 100))) + "%)</b> — against "
        "the lab's base rate on the same games, below.</p>",
        "<table><thead><tr><th>Call</th><th>Window</th>"
        "<th>Target</th><th>Outcome</th></tr></thead><tbody>",
    ]
    for call in sorted(
            done,
            key=lambda c: float(_field(c, "window_start", 0) or 0),
            reverse=True):
        provider = esc(_field(call, "provider"))
        game = esc(_field(call, "game"))
        target = _field(call, "target", 0)
        try:
            target_text = format(float(target), ".2f") + "x"
        except (TypeError, ValueError):
            target_text = esc(target)
        start = float(_field(call, "window_start", 0) or 0)
        end = float(_field(call, "window_end", 0) or 0)
        outcome = "HIT" if _field(call, "scored") else "MISS"
        parts.append(
            "<tr><td>" + provider + " " + game + "</td>"
            "<td>" + _clock(start) + " – " + _clock(end) + "</td>"
            "<td>" + target_text + "</td>"
            "<td><b>" + outcome + "</b></td></tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _lab_rows(status: dict[str, Any],
              library: dict[str, Any]) -> str:
    """Per-game lab verdicts: what the desk believes, the evidence,
    and what would change it."""
    games = status.get("games") or []
    if not games and not library:
        return ("<p>No games studied yet — collect rounds first: "
                "in Devon's DM, <code>/desk study</code>.</p>")
    rows = ["<table><thead><tr><th>Game</th><th>Rounds</th>"
            "<th>Verdict</th><th>Evidence</th>"
            "<th>What would change it</th></tr></thead><tbody>"]
    keys = sorted({str(g.get("provider", "")) + "/" + str(g.get("game", ""))
                   for g in games} | set(library.keys()))
    for key in keys:
        rec = library.get(key) or {}
        game_row = next(
            (g for g in games
             if str(g.get("provider", "")) + "/"
             + str(g.get("game", "")) == key),
            {})
        rounds = game_row.get("rounds", rec.get("n_rounds", 0))
        verdict = esc(game_row.get(
            "verdict", rec.get("verdict", "NOT_STUDIED")))
        structure = game_row.get(
            "callable_structure", rec.get("callable_structure", ""))
        verdict_text = verdict
        if structure:
            verdict_text = verdict_text + " (callable: " + esc(structure) + ")"
        evidence = esc(game_row.get("evidence", rec.get("evidence", "")))
        change = esc(rec.get("what_would_change", ""))
        rows.append(
            "<tr><td>" + esc(key) + "</td>"
            "<td>" + esc(rounds) + "</td>"
            "<td>" + verdict_text + "</td>"
            "<td>" + (evidence if evidence else "—") + "</td>"
            "<td>" + (change if change else "—") + "</td></tr>")
    rows.append("</tbody></table>")
    return "".join(rows)


def _health_rows(status: dict[str, Any]) -> str:
    """Desk health: the lead floor, the ledger, sources, and keys —
    presence only, never values."""
    parts = ["<ul>"]
    lead = status.get("min_lead_seconds", 120)
    parts.append("<li>Signal lead floor: " + esc(lead) + "s — a call "
                 "leads its window by at least this long, or it is "
                 "not called rather than called late.</li>")
    parts.append("<li>Crash rounds in the ledger: "
                 + esc(status.get("rounds_total", 0)) + " · library "
                 "records: " + esc(status.get("library_records", 0))
                 + "</li>")
    fixtures = status.get("sports_fixtures")
    if fixtures is not None:
        parts.append("<li>Sports fixtures in the analyst store: "
                     + esc(fixtures) + "</li>")
    sources = (status.get("sources") or {}).get("sources", [])
    if sources:
        health = ", ".join(
            esc(s.get("domain", "?")) + " ["
            + esc(s.get("health", "?")) + "]" for s in sources)
        parts.append("<li>Sources: " + health + "</li>")
    for row in status.get("provisioning", []):
        key = esc(row.get("env_key", ""))
        if not row.get("key_present"):
            parts.append("<li>Key " + key + ": MISSING — provision "
                         + esc(row.get("service", "")) + "</li>")
        else:
            source = esc(row.get("source", ""))
            line = "<li>Key " + key + ": present"
            if source:
                line += " (" + source + ")"
            parts.append(line + "</li>")
    book = status.get("session_book") or {}
    if book.get("imports_recorded"):
        parts.append("<li>Session books: "
                     + esc(book.get("files_booked", 0))
                     + " export files booked (+"
                     + esc(book.get("rounds_landed", 0))
                     + " rounds landed, "
                     + esc(book.get("duplicates_skipped", 0))
                     + " duplicates skipped)</li>")
    parts.append("</ul>")
    return "".join(parts)


def signal_page_html(status: dict[str, Any],
                     calls: list[Any],
                     library: dict[str, Any],
                     *, now: float | None = None) -> str:
    """Render the signal page from live desk state.

    ``status`` is ``BettingDesk.status()``, ``calls`` is
    ``SignalLog.calls()`` (``SignalCall`` objects or plain dicts),
    ``library`` is ``BettingDesk.library()``.  Pure: no desk, no
    network, no writes — a broken store still renders, because the
    page must render especially then."""
    moment = time.time() if now is None else float(now)
    scoring = status.get("signals") or {}
    total = scoring.get("calls_total", len(calls))
    scored = scoring.get("calls_scored")
    hit_rate = scoring.get("hit_rate")
    if scored is None:
        scored = sum(1 for c in calls if _field(c, "scored") is not None)
    if hit_rate is None:
        done = [c for c in calls if _field(c, "scored") is not None]
        hit_rate = (sum(1 for c in done if _field(c, "scored"))
                    / len(done)) if done else None
    body = [
        "<p>Timed signal calls from the betting desk — published "
        "at least 2 minutes before their window opens, off "
        "lab-confirmed time structure only. "
        + str(total) + " calls published, "
        + str(scored) + " scored, hit rate "
        + _pct(hit_rate) + ".</p>",
        "<h2>Live and upcoming</h2>",
        _call_rows(calls, moment),
        "<h2>Scored calls — calibration</h2>",
        _scored_rows(calls),
        "<h2>What the lab says</h2>",
        _lab_rows(status, library),
        "<h2>Desk health</h2>",
        _health_rows(status),
        "<p>The desk studies, signals and proposes — it never "
        "places a bet.</p>",
    ]
    return page_shell(
        title="Devon — bet desk signals",
        heading="Bet desk signals",
        body="".join(body),
        updated_at=moment,
        manage_line="Manage the desk in Devon's DM with /desk — "
                    "this page never places anything by itself.")


def signal_page_for_context(context: Any) -> str:
    """Render the page for an API/runtime context: build the desk
    the way the betdesk tool does, read its live state, render.
    ``status()`` scores due calls the same way the tool's status
    action does, so the page and the chat surface never disagree
    about the record."""
    from .desk import BettingDesk

    desk = BettingDesk(context)
    status = desk.status()
    calls = desk.signals.log.calls()
    library = desk.library()
    return signal_page_html(status, calls, library)
