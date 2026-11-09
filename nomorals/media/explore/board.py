"""The ``/explore`` board — WS4 Pass 3's render layer over the
shipped substrate (P1's store + engine, P2's claims).

One store, four doors (census §3.4): the morning briefing reads
claims through its own provider; this module renders the chat
doors — the bare board (last sweep's truth, what's current, the
candidate standings, when the next sweep is due), ``trends``
(the full claim set with each claim's basis, through P2's render
seam) and ``candidates`` (the pool's standings). Every number
here comes off a stored record; a failed sweep is named failed,
an empty field is stated, and no internal record id ever
renders — beats, names, timestamps and counts only.

The proxy line is deliberate and stays until the dependency
lands: sweeps run direct + the existing proxy pool because the
proxy-source catalogue refresh (``refresh_sources``, doc 17)
has no production caller yet (census §7's named dependency).
Exploration degrades honestly; the board says so rather than
letting a thin pool read as a quiet field.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from ..failures import record_failure
from .engine import default_explore_store
from .jobs import next_sweep_due, owner_tz
from .store import CANDIDATE_PROMOTED, CANDIDATE_WATCHING
from .tool import render_claims_text, render_run_text
from .trends import current_claims, evaluate_trends

__all__ = [
    "PROXY_DEPENDENCY_LINE",
    "render_explore_board",
    "render_explore_candidates",
    "render_explore_trends",
    "render_sweep_cycle",
]

#: The board's standing honesty line (census §7): the free pool's
#: catalogue activation is queued work, not this system's to
#: rebuild — until it lands, sweeps run on what the pool holds.
PROXY_DEPENDENCY_LINE = (
    "🛰 proxy: sweeps run direct + the existing proxy pool — the "
    "proxy-source refresh (refresh_sources) has no production "
    "caller yet (doc 17, queued), so the pool is only what it "
    "already holds")


def _when(ts: float, tz: Any = None) -> str:
    if not ts:
        return "never"
    if tz is not None:
        return datetime.fromtimestamp(float(ts), tz).strftime(
            "%Y-%m-%d %H:%M")
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def resolve_store(context: Any = None, store: Any = None) -> Any:
    """The ExploreStore a surface reads: an injected store wins;
    otherwise the shared default bound to the context's KV store —
    the same bind step :func:`~.engine.explore_for_context`
    performs, so board, briefing and tool all read one state."""
    if store is not None:
        return store
    store = default_explore_store()
    db = getattr(context, "db", None) if context is not None else None
    if db is not None:
        try:
            from ...storage.kv import KVStore

            store.bind_store(KVStore(db))
        except Exception as exc:  # noqa: BLE001 — persistence garnish
            record_failure("explore.board.bind_store", exc)
    return store


def _last_sweep_lines(store: Any, tz: Any) -> list[str]:
    latest = store.latest_sweep()
    if not latest:
        return ["last sweep: none yet — /explore run starts one"]
    beats = latest.get("beats") or []
    when = _when(latest.get("finished_at")
                 or latest.get("started_at") or 0, tz)
    lines = [
        f"last sweep: {when} ({latest.get('trigger') or 'manual'}) — "
        f"{len(beats)} beat(s) · "
        f"{latest.get('observations') or 0} observations · "
        f"{latest.get('results') or 0} results · "
        f"{latest.get('failed') or 0} failed · "
        f"{latest.get('new_candidates') or 0} new candidates"]
    if latest.get("status") == "failed":
        lines.append(
            "⚠️ that sweep FAILED — it did not finish; the "
            "observations it recorded before dying are on record "
            "below, and the failure is in the job history")
    elif latest.get("status") == "running":
        lines.append("… that sweep is still running")
    return lines


def render_explore_board(context: Any = None, *,
                         store: Any = None,
                         scheduler: Any = None,
                         now: float = 0.0) -> str:
    """The bare ``/explore`` board: last sweep truth, the next
    sweep due, current claims (fresh first), candidate standings,
    the proxy-dependency line, and the verbs."""
    now = now or time.time()
    store = resolve_store(context, store)
    _tz_name, tz = owner_tz()
    lines = ["🔭 **Exploration** — the video field Devon fetches "
             "from, watched"]
    lines.extend(_last_sweep_lines(store, tz))
    due = next_sweep_due(scheduler)
    if due:
        slot = ("morning" if "morning" in due["name"] else "evening")
        lines.append(
            f"next sweep: {_when(due['at'], tz)} ({slot} slot, "
            "owner-local) — sweeps run morning + evening on the "
            "durable scheduler")
    else:
        lines.append(
            "next sweep: not scheduled — the sweep jobs register "
            "at boot (morning + evening, owner-local)")
    claims = current_claims(store, now=now)
    fresh = [c for c in claims if c.get("standing") == "fresh"]
    lines.append("")
    if fresh:
        lines.append("moving now:")
        for claim in fresh[:5]:
            lines.append(
                f"· [{claim.get('kind')}] {claim.get('text')} — "
                f"fresh, last evidence "
                f"{_when(claim.get('observed_at') or 0, tz)}")
        if len(fresh) > 5:
            lines.append(
                f"· … and {len(fresh) - 5} more — /explore trends "
                "has them all, with their basis")
    elif claims:
        lines.append(
            f"claims: {len(claims)} on record, all stale — nothing "
            "current; the next sweep refreshes the field")
    else:
        lines.append(
            "claims: none yet — one sweep alone can only ever be "
            "a spike or unverified, never a trend")
    pool = store.candidates()
    watching = [c for c in pool if c.get("status") == CANDIDATE_WATCHING]
    promoted = [c for c in pool if c.get("status") == CANDIDATE_PROMOTED]
    shed = store.shed_records()
    lines.append("")
    lines.append(
        f"candidates: {len(watching)} watching · "
        f"{len(promoted)} promoted · {len(shed)} shed")
    for cand in watching[:3]:
        lines.append(
            f"· watching: {cand.get('name')} ({cand.get('kind')}) — "
            f"seen {cand.get('sightings')}×, last "
            f"{_when(cand.get('last_seen') or 0, tz)}")
    lines.append("")
    lines.append(PROXY_DEPENDENCY_LINE)
    lines.append("")
    lines.append(
        "commands: /explore run — sweep now · /explore trends — "
        "the claims with their basis · /explore candidates — the "
        "candidate pool")
    return "\n".join(lines)


def render_explore_trends(context: Any = None, *,
                          store: Any = None,
                          now: float = 0.0,
                          evaluate: bool = True) -> str:
    """``/explore trends``: re-evaluate the observation history
    (idempotent — P2's supersede rule) and render the current
    claim set through the tool's render seam: kind, basis in
    plain words, freshness."""
    now = now or time.time()
    store = resolve_store(context, store)
    report = evaluate_trends(store, now=now) if evaluate else None
    claims = current_claims(store, now=now)
    return render_claims_text(claims, report)


def render_explore_candidates(context: Any = None, *,
                              store: Any = None,
                              now: float = 0.0) -> str:
    """``/explore candidates``: the pool's standings — watching
    and promoted with their sighting counts and evidence
    standing, shed with the reason they left."""
    store = resolve_store(context, store)
    _tz_name, tz = owner_tz()
    pool = store.candidates()
    lines = ["🔭 **Exploration candidates**"]
    if not pool and not store.shed_records():
        lines.append(
            "none yet — candidates are noted from sweep results "
            "and from what the owner actually downloads")
        return "\n".join(lines)
    for status, label in ((CANDIDATE_WATCHING, "watching"),
                          (CANDIDATE_PROMOTED, "promoted")):
        rows = [c for c in pool if c.get("status") == status]
        if not rows:
            continue
        lines.append(f"\n{label}:")
        for cand in rows[:15]:
            standing = cand.get("standing") or {}
            seen = (f" across {len(cand.get('sweeps') or [])} "
                    "sweep(s)") if cand.get("sweeps") else ""
            score = standing.get("score")
            score_s = (f" · standing score {score} (inputs on "
                       "record)" if score is not None else "")
            lines.append(
                f"· {cand.get('name')} ({cand.get('kind')}) — seen "
                f"{cand.get('sightings')}×{seen}, last "
                f"{_when(cand.get('last_seen') or 0, tz)}{score_s}")
    shed = store.shed_records()
    if shed:
        lines.append("\nshed:")
        for row in shed[-10:]:
            reason = row.get("reason") or "no reason recorded"
            lines.append(
                f"· {row.get('name')} — shed "
                f"{_when(row.get('at') or 0, tz)}: {reason}")
    lines.append(
        "\npromotion is evidence standing in this pool only — "
        "nothing enters a production fetch path or a niche "
        "without the owner's act")
    return "\n".join(lines)


def render_sweep_cycle(cycle: dict[str, Any]) -> str:
    """A cycle report as chat text (``/explore run`` + the sweep
    tool action): deferrals say why in plain words; a completed
    cycle shows the sweep's own report plus the evaluation
    outcome — claims current, written, verdicts."""
    if cycle.get("deferred") == "quiet_hours":
        return ("🌙 **Exploration sweep deferred** — inside the "
                "partner quiet window; the field keeps until the "
                "next slot. Nothing was swept.")
    if cycle.get("deferred") == "daily_cap":
        return (f"⏸ **Exploration sweep deferred** — "
                f"{cycle.get('note') or 'the daily cap is spent'}. "
                "Nothing was swept.")
    sweep = cycle.get("sweep") or {}
    lines = [render_run_text(sweep)]
    evaluation = cycle.get("evaluation") or {}
    if evaluation:
        lines.append(
            f"trends re-evaluated: "
            f"{evaluation.get('claims_current', 0)} claim(s) "
            f"current ({evaluation.get('claims_written', 0)} "
            f"written this pass · "
            f"{evaluation.get('promoted', 0)} candidate(s) "
            f"promoted · {evaluation.get('shed', 0)} shed · "
            f"{evaluation.get('declined', 0)} declined)")
        if evaluation.get("error"):
            lines.append(
                f"⚠️ trend evaluation hit a problem: "
                f"{evaluation['error']} — the sweep itself "
                "completed and its observations are on record")
    return "\n".join(lines)
