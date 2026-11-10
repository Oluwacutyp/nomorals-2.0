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

The proxy line renders the catalogue-activation record (doc 17
P1 resolved census §7's named dependency): ``refresh_sources``
now has a production caller — every proxy refresh activates the
catalogue first — so the line states the recorded activation
state (epoch, last merge, registry size), or, before the first
activation has run on the host, that it has not run yet.
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
from .store import (
    BRANCH_NICHE_CLUSTER,
    CANDIDATE_PROMOTED,
    CANDIDATE_WATCHING,
    NICHE_ADOPTED,
    NICHE_DISMISSED,
    NICHE_PROPOSED,
)
from .tool import render_claims_text, render_run_text
from .trends import current_claims, eliminated_branches, evaluate_trends

__all__ = [
    "PROXY_DEPENDENCY_LINE",
    "proxy_line",
    "render_explore_board",
    "render_explore_candidates",
    "render_explore_niches",
    "render_explore_trends",
    "render_niche_act",
    "render_sweep_cycle",
]

#: The board's standing proxy line, pre-activation form (census
#: §7, resolved by doc 17 P1): catalogue activation IS wired —
#: it rides every proxy refresh — so this wording is the truth
#: only until the first activation has run on the host; after
#: that, :func:`proxy_line` renders the recorded state instead.
PROXY_DEPENDENCY_LINE = (
    "🛰 proxy: sweeps run direct + the existing proxy pool — "
    "catalogue activation (refresh_sources, doc 17) is wired "
    "into every proxy refresh but has not run yet on this host, "
    "so the pool is only what it already holds")


def _activation_state(context: Any = None) -> dict[str, Any] | None:
    """The host's catalogue-activation record (doc 17 P1), or
    None when activation has never completed here. Read-only;
    a broken read is a ledger entry + None, never a board crash.
    The home resolves exactly as the ProxyLab resolves it —
    the context's settings home — so board and lab read the
    same store."""
    try:
        from ...tools.proxy_sources import activation_state_for_home

        settings = getattr(context, "settings", None) \
            if context is not None else None
        home = str(getattr(settings, "home", "~/.nomorals")) \
            if settings is not None else "~/.nomorals"
        return activation_state_for_home(home)
    except Exception as exc:  # noqa: BLE001 — the board never dies on it
        record_failure("explore.board.proxy_activation_state", exc)
        return None


def proxy_line(context: Any = None) -> str:
    """The board's proxy line, from the activation record:
    before the first activation it says activation has not run
    yet (PROXY_DEPENDENCY_LINE); after it, the recorded epoch,
    last merge and registry size — the "has activation ever
    run, and what did it do" answer, on the board that used to
    name the missing caller."""
    state = _activation_state(context)
    if not state or not int(state.get("epoch") or 0):
        return PROXY_DEPENDENCY_LINE
    last = state.get("last_run") or {}
    return (
        "🛰 proxy: sweeps run direct + the existing proxy pool — "
        "the source catalogue is activated (doc 17): "
        f"epoch {int(state.get('epoch') or 0)} · last merge "
        f"{_when(float(last.get('at') or 0))} added "
        f"{len(last.get('added') or [])} source(s) · registry "
        f"holds {int(last.get('registry_after') or 0)} sources · "
        "every proxy refresh re-merges the catalogue")


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


def _fields_watched_lines(context: Any) -> list[str]:
    """The fields a sweep would watch, grouped by origin — the
    resolution :func:`~.beats.load_beats` itself makes, so the
    board can never drift from the sweep. The trading pack's
    topics are named outright: the trading field is the owner's
    dominant watching field, and its findings must be tellable
    from his interest beats at a glance."""
    try:
        from .beats import load_beats

        db = getattr(context, "db", None) if context is not None \
            else None
        settings = getattr(context, "settings", None) \
            if context is not None else None
        beats = load_beats(db, settings)
    except Exception as exc:  # noqa: BLE001 — a board never dies
        record_failure("explore.board.fields_watched", exc)
        return []
    if not beats:
        return []
    groups: dict[str, list[str]] = {}
    for beat in beats:
        groups.setdefault(str(beat.get("origin") or "seed"),
                          []).append(str(beat.get("topic") or ""))
    parts = [f"{len(topics)} {origin}"
             for origin, topics in groups.items()]
    lines = [f"fields watched: {len(beats)} beat(s) — "
             + " · ".join(parts)]
    trading = groups.get("trading") or []
    if trading:
        named = ", ".join(trading[:3])
        if len(trading) > 3:
            named += f", +{len(trading) - 3} more"
        lines.append(f"· trading field: {named}")
    return lines


def render_explore_board(context: Any = None, *,
                         store: Any = None,
                         scheduler: Any = None,
                         now: float = 0.0) -> str:
    """The bare ``/explore`` board: last sweep truth, the next
    sweep due, the fields watched by origin, current claims
    (fresh first), candidate standings, the proxy-dependency
    line, and the verbs."""
    now = now or time.time()
    store = resolve_store(context, store)
    _tz_name, tz = owner_tz()
    lines = ["🔭 **Exploration** — the video field Devon fetches "
             "from, watched"]
    lines.extend(_last_sweep_lines(store, tz))
    lines.extend(_fields_watched_lines(context))
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
            grade = str((claim.get("proof_grade") or {}).get(
                "grade") or "G0")
            lines.append(
                f"· [{claim.get('kind')}] {claim.get('text')} — "
                f"proof {grade}, fresh, last evidence "
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
    preserved = eliminated_branches(store, limit=1000)
    if preserved:
        lines.append(
            f"eliminated branches: {len(preserved)} preserved "
            "(superseded / shed / declined — labelled on "
            "/explore trends, never live claims)")
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
    proposals = store.niche_proposals(status=NICHE_PROPOSED)
    if proposals:
        lines.append("")
        lines.append(
            f"niche proposals: {len(proposals)} awaiting your tap "
            "— /explore niches lists them with their evidence; "
            "adoption is your act, nothing discovered goes live "
            "on its own")
    lines.append("")
    lines.append(proxy_line(context))
    lines.append("")
    lines.append(
        "commands: /explore run — sweep now · /explore trends — "
        "the claims with their basis · /explore candidates — the "
        "candidate pool · /explore niches — niche proposals + "
        "the registry")
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
    return render_claims_text(
        claims, report, eliminated_branches(store))


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


def render_explore_niches(context: Any = None, *,
                          store: Any = None,
                          now: float = 0.0) -> str:
    """``/explore niches`` — the shared niche registry (census
    §4): the shipped code floor (production, always in force),
    the discovered overlay's proposals (NOT live — adoption is
    the owner's act, numbered as the act verbs read them), the
    adopted overlay entries, and the dismissed count. A proposal
    is rendered as a proposal everywhere; nothing here calls one
    a live niche."""
    from .niches import (
        effective_niches,
        floor_niches,
        proposal_board_order,
    )

    store = resolve_store(context, store)
    lines = ["🔭 **Exploration niches** — the shared registry: "
             "the shipped floor + the discovered overlay"]
    lines.append("\nproduction floor (hand-written, always in "
                 "force):")
    for row in floor_niches():
        lines.append(
            f"· {row['slug']} — {row['thesis']} "
            f"({row['cadence']:g}/day)")
    adopted = store.niche_proposals(status=NICHE_ADOPTED)
    if adopted:
        lines.append("\nadopted (live via the overlay — the "
                     "draft plugin is a review artifact, not the "
                     "thing serving):")
        for rec in adopted:
            lines.append(
                f"· {rec.get('slug')} — {rec.get('thesis')} "
                f"(draft: {rec.get('draft_path') or 'not recorded'})")
    proposals = proposal_board_order(store)
    lines.append("\nproposals — discovered, NOT live; adoption "
                 "is your act:")
    if not proposals:
        lines.append(
            "· none awaiting — discovery proposes only what the "
            "evidence carries (two+ members, corroborated across "
            "sweeps)")
    for i, rec in enumerate(proposals, 1):
        pack = rec.get("evidence_pack") or {}
        member_names = ", ".join(
            str(m.get("name") or "")
            for m in (rec.get("members") or [])[:4])
        lines.append(
            f"#{i} {rec.get('slug')} — "
            f"{len(rec.get('members') or [])} member(s) · "
            f"{pack.get('sightings') or 0} sighting(s) across "
            f"{len(pack.get('sweeps') or [])} sweep(s) · score "
            f"{pack.get('score')}")
        lines.append(f"   members: {member_names}")
        lines.append(f"   basis: {rec.get('basis')}")
        lines.append(
            f"   adopt: /explore adopt {i} · dismiss: "
            f"/explore dismiss {i}")
    dismissed = store.niche_proposals(status=NICHE_DISMISSED)
    if dismissed:
        lines.append(
            f"\ndismissed: {len(dismissed)} — the same evidence "
            "won't re-propose them; new sightings can")
    declined = eliminated_branches(
        store, branch_kind=BRANCH_NICHE_CLUSTER, limit=5)
    if declined:
        lines.append(
            "\ndeclined clusters (preserved as eliminated "
            "branches — not proposals):")
        for branch in declined:
            grade = str((branch.get("proof_grade") or {}).get(
                "grade") or "G0")
            lines.append(
                f"· {branch.get('subject')} — "
                f"{branch.get('reason')} · proof {grade} "
                "(its own evidence)")
    effective = effective_niches(store)
    lines.append(
        f"\neffective niche set (floor ∪ adopted): "
        f"{len(effective)} — proposals are not in it until you "
        "adopt them")
    return "\n".join(lines)


def render_niche_act(result: dict[str, Any], verb: str) -> str:
    """An adopt/dismiss outcome as chat text: what happened, and
    — for adoption — the draft plugin's exact standing (a review
    artifact in the drafts folder; going live is a separate,
    human step)."""
    if not result.get("ok"):
        return (f"⚠️ couldn't {verb} that niche: "
                f"{result.get('reason') or 'unknown reason'}")
    slug = result.get("slug") or ""
    if verb == "adopt":
        note = result.get("note") or ""
        note_s = f" ({note})" if note else ""
        return (
            f"✅ **{slug}** adopted into the overlay — it is in "
            "the effective niche set now, served from its "
            f"record{note_s}.\n"
            f"a DRAFT plugin was generated at "
            f"{result.get('draft_path')} — it is not a production "
            "plugin and nothing imports it: it goes live only "
            "when it's reviewed, its TODO sections are filled, "
            "it's moved into the niches package and its slug is "
            "added to the shipped list in niches/registry.py.")
    return (f"🗑 **{slug}** dismissed — recorded with its "
            "evidence fingerprint; the same evidence won't "
            "re-propose it. New sightings can.")


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
    niches = cycle.get("niches") or {}
    if niches:
        if niches.get("error"):
            lines.append(
                f"⚠️ niche discovery hit a problem: "
                f"{niches['error']} — the sweep and its claims "
                "are on record; discovery is a proposal layer "
                "over them, not part of the sweep")
        elif niches.get("proposed"):
            lines.append(
                f"niche discovery: {niches['proposed']} new "
                "proposal(s) awaiting your tap — /explore niches")
    return "\n".join(lines)
