"""Setup lifecycle + shadow driver — Devon trading system, Pass 2.

Setups are the funnel between the playbooks (agents/playbooks.py)
and everything downstream: a setup is born ``detected`` when a
playbook's premise forms, becomes ``armed`` when its arming rule
completes the plan (entry zone, invalidation and targets fixed from
the entry timeframe's structure), and ends ``triggered`` (price
traded into the zone — the shadow entry), ``invalidated`` (a close
beyond the invalidation, or the premise un-formed) or ``expired``
(its declared TTL ran out, a fresher setup superseded it, or its
own geometry floor rejected it at the break).

SHADOW ONLY: the driver detects, arms, tracks and journals. It
never sizes, never calls the desk or any executor (Pass 4), and
installs no scheduling (Pass 6).

Storage: the Pass 1 append-only trading journal IS the store.
Every transition is one journal event (``setup_detected`` /
``setup_armed`` / ``setup_triggered`` / ``setup_invalidated`` /
``setup_expired``, plus one ``shadow_pass`` summary per pass), and
current state is the fold of those events — no new file, no storage
migration, and Pass 3/5 read exactly the records this pass wrote.
Each setup event carries its reason from the fixed 10-category
taxonomy below and a reference to the read snapshot that produced
it (``read_taken_at``), so validation can grade a setup against the
read it was taken from.

Placement: ``agents/`` beside market_read/playbooks — the layer
from which the TA library, the finance journal and the market-data
integrations are all legal static imports (Pass 1's note applies).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..finance import trading_journal
from ..integrations import market_data
from . import market_read
from .playbooks import (
    PLAYBOOKS,
    TF_LABELS,
    TF_SECONDS,
    Candidate,
    DetectContext,
    Playbook,
    get_playbook,
    ts_epoch,
)

_log = logging.getLogger(__name__)

#: The fixed 10-category reason taxonomy. Transition categories
#: (journaled on the transition): pattern_formed, premise_confirmed,
#: zone_touched, invalidation_hit, premise_lost, ttl_expired,
#: superseded, target_invalid_geometry. Pass-accounting categories
#: (counted in the pass summary, not transitions): already_live,
#: data_gap.
REASONS: tuple[str, ...] = (
    "pattern_formed", "premise_confirmed", "zone_touched",
    "invalidation_hit", "premise_lost", "ttl_expired", "superseded",
    "target_invalid_geometry", "already_live", "data_gap")

LIVE_STATES = ("detected", "armed")
TERMINAL_STATES = ("triggered", "invalidated", "expired")
_EVENT_FOR_STATE = {"detected": "setup_detected", "armed": "setup_armed",
                    "triggered": "setup_triggered",
                    "invalidated": "setup_invalidated",
                    "expired": "setup_expired"}


# ── the store: state is a fold of the journal ────────────────────────

def _apply(setup: dict[str, Any], rec: dict[str, Any]) -> None:
    state = rec.get("to_state")
    if state:
        setup["state"] = state
    if rec.get("plan"):
        setup["plan"] = rec["plan"]
    if rec.get("armed_at") is not None:
        setup["armed_at"] = rec["armed_at"]
        setup["armed_bar_ts"] = rec.get("armed_bar_ts")
    if rec.get("expires_at") is not None:
        setup["expires_at"] = rec["expires_at"]
    if rec.get("trigger"):
        setup["trigger"] = rec["trigger"]
    setup["updated_at"] = rec.get("ts", setup.get("updated_at"))
    setup.setdefault("history", []).append({
        "ts": rec.get("ts"), "from": rec.get("from_state"),
        "to": state, "reason_category": rec.get("reason_category"),
        "reason": rec.get("reason")})


def load_setups(*, settings: Any = None, journal_path: Any = None,
                symbol: str | None = None) -> list[dict[str, Any]]:
    """Current setup states, folded from the trading journal.

    Oldest first (creation order). ``symbol`` filters by instrument.
    """
    records = trading_journal.read_events(
        limit=0, settings=settings, path=journal_path)
    setups: dict[str, dict[str, Any]] = {}
    for rec in reversed(records):  # read_events is newest-first
        event = rec.get("event") or ""
        if not event.startswith("setup_"):
            continue
        sid = rec.get("setup_id")
        if not sid:
            continue
        if event == "setup_detected":
            setups[sid] = {
                "setup_id": sid, "symbol": rec.get("symbol"),
                "market": rec.get("market"),
                "playbook": rec.get("playbook"),
                "route": rec.get("route"),
                "family": rec.get("family"),
                "entry_timeframe": rec.get("entry_timeframe"),
                "direction": rec.get("direction"),
                "pattern_key": rec.get("pattern_key"),
                "state": "detected",
                "detected_at": rec.get("ts"),
                "detected_bar_ts": rec.get("detected_bar_ts"),
                "armed_at": None, "armed_bar_ts": None,
                "expires_at": rec.get("expires_at"),
                "plan": rec.get("plan"),
                "evidence": rec.get("evidence") or {},
                "read_taken_at": rec.get("read_taken_at"),
                "trigger": None, "history": []}
        setup = setups.get(sid)
        if setup is not None:
            _apply(setup, rec)
    out = list(setups.values())
    if symbol is not None:
        want = symbol.strip().upper()
        out = [s for s in out if (s.get("symbol") or "") == want]
    return out


def _journal_transition(setup: dict[str, Any], to_state: str,
                        category: str, reason: str, *,
                        settings: Any, journal_path: Any,
                        read_taken_at: float | None,
                        ts: float | None = None,
                        extra: dict[str, Any] | None = None) -> None:
    fields: dict[str, Any] = {
        "setup_id": setup["setup_id"], "symbol": setup["symbol"],
        "market": setup.get("market"), "playbook": setup["playbook"],
        "route": setup.get("route"), "family": setup.get("family"),
        "entry_timeframe": setup.get("entry_timeframe"),
        "direction": setup.get("direction"),
        "pattern_key": setup.get("pattern_key"),
        "from_state": setup["state"], "to_state": to_state,
        "reason_category": category, "reason": reason,
        "read_taken_at": read_taken_at}
    if extra:
        fields.update(extra)
    if ts is not None:
        fields["ts"] = ts  # the pass's logical time stamps the record
    trading_journal.append_event(
        _EVENT_FOR_STATE[to_state], settings=settings,
        path=journal_path, **fields)
    rec = dict(fields)
    rec.setdefault("ts", time.time())
    _apply(setup, rec)


# ── matching candidates to live setups ───────────────────────────────

def _zone_overlap(a_top: float, a_bottom: float, b_top: float,
                  b_bottom: float) -> bool:
    lo = max(min(a_top, a_bottom), min(b_top, b_bottom))
    hi = min(max(a_top, a_bottom), max(b_top, b_bottom))
    if hi < lo:
        return False
    width = max(max(a_top, a_bottom) - min(a_top, a_bottom), 1e-9)
    return (hi - lo) >= 0.5 * width


def _matches(setup: dict[str, Any], cand: Candidate) -> bool:
    if setup.get("pattern_key") == cand.pattern_key:
        return True
    if setup.get("direction") != cand.direction:
        return False
    plan = setup.get("plan") or {}
    zone = plan.get("entry_zone") or {}
    if zone.get("top") is None:
        return False
    return _zone_overlap(float(zone["top"]), float(zone["bottom"]),
                         cand.entry_zone_top, cand.entry_zone_bottom)


def _last_bar_ts(frame: Any) -> float | None:
    if frame is None or len(frame) == 0:
        return None
    return ts_epoch(frame.index[-1])


# ── the shadow pass ──────────────────────────────────────────────────

def shadow_pass(symbol: str, market: str = "crypto", *,
                settings: Any = None, journal_path: Any = None,
                now: float | None = None,
                read: dict[str, Any] | None = None,
                bars: dict[str, Any] | None = None,
                playbooks: tuple[Playbook, ...] = PLAYBOOKS
                ) -> dict[str, Any]:
    """One shadow pass: fresh read -> detect -> advance -> journal.

    Takes a fresh whole-market read (journaled, unless a read is
    supplied), fetches the entry-timeframe bars the registry needs
    (unless supplied), runs every playbook's detector, folds the
    candidates into the setup store, advances live setups against
    the new bars, and journals every transition. Returns the pass
    summary. Nothing executes; nothing is scheduled.
    """
    taken = float(now if now is not None else time.time())
    sym = (symbol or "").strip().upper()
    if read is None:
        read = market_read.whole_market_read(
            sym, market=market, journal=True, settings=settings,
            journal_path=journal_path, now=taken)
    read_taken_at = read.get("taken_at")
    if bars is None:
        bars = {}
        tfs = {pb.entry_timeframe for pb in playbooks}
        for tf in market_read.TIMEFRAMES:
            if tf not in tfs:
                continue
            try:
                bars[tf] = market_data.get_ohlcv(
                    sym, market=market, timeframe=tf,
                    bars=market_read.DEFAULT_BARS[tf])
            except Exception as exc:  # noqa: BLE001 - marked, not fatal
                _log.info("setups: %s bars unavailable: %s", tf, exc)
                bars[tf] = None
    prior = load_setups(settings=settings, journal_path=journal_path,
                        symbol=sym)
    live = [s for s in prior if s["state"] in LIVE_STATES]
    by_id = {s["setup_id"]: s for s in prior}
    ctx = DetectContext(symbol=sym, market=market, read=read,
                        bars=bars, now=taken)
    by_name = {pb.name: pb for pb in playbooks}
    counts = {r: 0 for r in REASONS}
    made: dict[str, list[str]] = {"created": [], "armed": [],
                                  "triggered": [], "invalidated": [],
                                  "expired": []}
    detector_errors: list[str] = []
    candidates: list[Candidate] = []
    for pb in playbooks:
        try:
            candidates.extend(pb.detector(ctx, pb))
        except Exception as exc:  # noqa: BLE001 - one form down ≠ pass down
            _log.warning("setups: detector %s failed: %s", pb.name, exc)
            detector_errors.append(f"{pb.name}: {type(exc).__name__}")
    touched: set[str] = set()
    matched_ids: set[str] = set()

    def _expire(setup: dict[str, Any], category: str,
                reason: str) -> None:
        _journal_transition(setup, "expired", category, reason,
                            settings=settings, journal_path=journal_path,
                            read_taken_at=read_taken_at, ts=taken)
        counts[category] += 1
        made["expired"].append(setup["setup_id"])
        touched.add(setup["setup_id"])

    def _create(cand: Candidate, pb: Playbook) -> dict[str, Any]:
        sid = f"{pb.name}:{sym}:{cand.pattern_key}"
        tf_sec = TF_SECONDS[pb.entry_timeframe]
        setup: dict[str, Any] = {
            "setup_id": sid, "symbol": sym, "market": market,
            "playbook": pb.name, "route": pb.name, "family": pb.family,
            "entry_timeframe": pb.entry_timeframe,
            "direction": cand.direction,
            "pattern_key": cand.pattern_key, "state": "candidate",
            "detected_at": taken,
            "detected_bar_ts": _last_bar_ts(bars.get(pb.entry_timeframe)),
            "armed_at": None, "armed_bar_ts": None,
            "expires_at": taken + pb.detected_ttl_bars * tf_sec,
            "plan": cand.plan(), "evidence": cand.evidence,
            "read_taken_at": read_taken_at, "trigger": None,
            "history": []}
        _journal_transition(
            setup, "detected", "pattern_formed", cand.reason,
            settings=settings, journal_path=journal_path,
            read_taken_at=read_taken_at, ts=taken,
            extra={"plan": cand.plan(), "evidence": cand.evidence,
                   "detected_bar_ts": setup["detected_bar_ts"],
                   "expires_at": setup["expires_at"]})
        counts["pattern_formed"] += 1
        made["created"].append(sid)
        by_id[sid] = setup
        live.append(setup)
        touched.add(sid)
        return setup

    def _arm(setup: dict[str, Any], cand: Candidate,
             pb: Playbook) -> None:
        tf_sec = TF_SECONDS[pb.entry_timeframe]
        bar_ts = _last_bar_ts(bars.get(pb.entry_timeframe))
        _journal_transition(
            setup, "armed", "premise_confirmed", cand.reason,
            settings=settings, journal_path=journal_path,
            read_taken_at=read_taken_at, ts=taken,
            extra={"plan": cand.plan(),
                   "armed_at": taken, "armed_bar_ts": bar_ts,
                   "expires_at": taken + pb.armed_ttl_bars * tf_sec})
        counts["premise_confirmed"] += 1
        made["armed"].append(setup["setup_id"])
        touched.add(setup["setup_id"])

    for cand in candidates:
        pb = by_name.get(cand.playbook)
        if pb is None:
            continue
        match = next((s for s in live
                      if s["playbook"] == cand.playbook
                      and s["state"] in LIVE_STATES
                      and _matches(s, cand)), None)
        if match is not None:
            matched_ids.add(match["setup_id"])
            if cand.maturity == "rejected" \
                    and match["state"] == "detected":
                _expire(match, "target_invalid_geometry", cand.reason)
            elif cand.maturity == "armed" \
                    and match["state"] == "detected":
                _arm(match, cand, pb)
            else:
                counts["already_live"] += 1
            continue
        if cand.maturity == "rejected":
            continue  # never seen live: nothing to kill, nothing to arm
        # One live setup per (playbook, direction): a fresher armed
        # candidate supersedes; a detected-only candidate waits.
        rivals = [s for s in live
                  if s["playbook"] == cand.playbook
                  and s["direction"] == cand.direction
                  and s["state"] in LIVE_STATES]
        if rivals:
            if cand.maturity == "armed":
                for rival in rivals:
                    _expire(rival, "superseded",
                            f"superseded by {cand.pattern_key}: "
                            f"{cand.reason}")
                live[:] = [s for s in live if s["state"] in LIVE_STATES]
            else:
                counts["already_live"] += 1
                continue
        setup = _create(cand, pb)
        if cand.maturity == "armed":
            _arm(setup, cand, pb)

    # Advance the setups that were live before this pass and were
    # not themselves transitioned by a candidate above.
    for setup in prior:
        sid = setup["setup_id"]
        if setup["state"] not in LIVE_STATES or sid in touched:
            continue
        pb = by_name.get(setup["playbook"]) or get_playbook(
            setup["playbook"])
        if pb is None:
            continue
        tf = setup["entry_timeframe"]
        block = (read.get("timeframes") or {}).get(tf) or {}
        frame = bars.get(tf)
        if block.get("status") == "missing" or frame is None \
                or len(frame) == 0:
            counts["data_gap"] += 1
            continue  # a data gap pauses a setup; it never kills it
        plan = setup.get("plan") or {}
        invalidation = plan.get("invalidation")
        direction = setup["direction"]
        # Chronological scan of the bars since the setup's stage
        # began: a close beyond invalidation kills the setup on
        # that bar; for an armed setup a zone touch on an earlier
        # bar triggers first. (On a single bar that does both, the
        # invalidation close wins — the premise died on the entry
        # bar.) Detected setups only scan for the killing close.
        zone = plan.get("entry_zone") or {}
        since = setup.get("armed_bar_ts") if setup["state"] == "armed" \
            else setup.get("detected_bar_ts")
        outcome: tuple[str, float, float | None] | None = None
        if invalidation is not None and since is not None:
            for ts, row in frame.iterrows():
                if ts_epoch(ts) <= since:
                    continue
                close = float(row["close"])
                if (direction == 1 and close < float(invalidation)) \
                        or (direction == -1
                            and close > float(invalidation)):
                    outcome = ("invalidated", ts_epoch(ts), None)
                    break
                if setup["state"] == "armed" \
                        and zone.get("top") is not None \
                        and float(row["low"]) <= float(zone["top"]) \
                        and float(row["high"]) >= float(zone["bottom"]):
                    entry_price = float(zone["top"]) if direction == 1 \
                        else float(zone["bottom"])
                    outcome = ("triggered", ts_epoch(ts), entry_price)
                    break
        if outcome is not None and outcome[0] == "invalidated":
            _journal_transition(
                setup, "invalidated", "invalidation_hit",
                f"close beyond invalidation "
                f"{float(invalidation):.2f} on {TF_LABELS[tf]}",
                settings=settings, journal_path=journal_path,
                read_taken_at=read_taken_at, ts=taken)
            counts["invalidation_hit"] += 1
            made["invalidated"].append(sid)
            continue
        if outcome is not None and outcome[0] == "triggered":
            _journal_transition(
                setup, "triggered", "zone_touched",
                f"price traded into the entry zone "
                f"{float(zone['bottom']):.2f}–{float(zone['top']):.2f}"
                f" on {TF_LABELS[tf]} (shadow entry "
                f"{outcome[2]:.2f})",
                settings=settings, journal_path=journal_path,
                read_taken_at=read_taken_at, ts=taken,
                extra={"trigger": {"bar_ts": outcome[1],
                                   "price": outcome[2]}})
            counts["zone_touched"] += 1
            made["triggered"].append(sid)
            continue
        if setup.get("expires_at") is not None \
                and taken > float(setup["expires_at"]):
            _expire(setup, "ttl_expired",
                    f"TTL ran out in state {setup['state']} "
                    f"({pb.name} allows "
                    f"{pb.detected_ttl_bars if setup['state'] == 'detected' else pb.armed_ttl_bars}"
                    f" {TF_LABELS[tf]} bars)")
            continue
        if setup["state"] == "detected" and sid not in matched_ids:
            _journal_transition(
                setup, "invalidated", "premise_lost",
                "the playbook's premise no longer reads on the "
                "entry timeframe",
                settings=settings, journal_path=journal_path,
                read_taken_at=read_taken_at, ts=taken)
            counts["premise_lost"] += 1
            made["invalidated"].append(sid)

    after = load_setups(settings=settings, journal_path=journal_path,
                        symbol=sym)
    summary: dict[str, Any] = {
        "kind": "shadow_pass", "symbol": sym, "market": market,
        "taken_at": taken, "read_taken_at": read_taken_at,
        "created": made["created"], "armed": made["armed"],
        "triggered": made["triggered"],
        "invalidated": made["invalidated"], "expired": made["expired"],
        "counts": counts, "detector_errors": detector_errors,
        "live": [s for s in after if s["state"] in LIVE_STATES],
        "setups": after}
    trading_journal.append_event(
        "shadow_pass", settings=settings, path=journal_path,
        symbol=sym, market=market, read_taken_at=read_taken_at,
        created=len(made["created"]), armed=len(made["armed"]),
        triggered=len(made["triggered"]),
        invalidated=len(made["invalidated"]),
        expired=len(made["expired"]), counts=counts)
    return summary


# ── surfaces ─────────────────────────────────────────────────────────

def recent_setups(symbol: str | None = None, *, limit: int = 20,
                  settings: Any = None, journal_path: Any = None,
                  now: float | None = None) -> list[dict[str, Any]]:
    """Live setups first, then the most recent terminal ones."""
    taken = float(now if now is not None else time.time())
    setups = load_setups(settings=settings, journal_path=journal_path,
                         symbol=symbol)
    for s in setups:
        s["age_s"] = round(max(0.0, taken - float(s["detected_at"])), 1) \
            if s.get("detected_at") else None
    live = sorted((s for s in setups if s["state"] in LIVE_STATES),
                  key=lambda s: s["detected_at"] or 0, reverse=True)
    done = sorted((s for s in setups if s["state"] in TERMINAL_STATES),
                  key=lambda s: s.get("updated_at") or 0, reverse=True)
    return (live + done)[:limit] if limit > 0 else live + done


def _fmt(value: Any) -> str:
    if value is None:
        return "?"
    return f"{float(value):,.2f}"


def _age_text(age_s: float | None) -> str:
    if age_s is None:
        return "?"
    if age_s < 3600:
        return f"{int(age_s // 60)}m"
    if age_s < 86400:
        return f"{age_s / 3600:.1f}h"
    return f"{age_s / 86400:.1f}d"


def render_setups(setups: list[dict[str, Any]], *,
                  symbol: str | None = None,
                  now: float | None = None) -> str:
    """Human rendering of the setup view — the chat/CLI surface."""
    taken = float(now if now is not None else time.time())
    live_n = sum(1 for s in setups if s["state"] in LIVE_STATES)
    head = f"{symbol or 'all symbols'} shadow setups — {live_n} live" \
        f" of {len(setups)} shown"
    lines = [head]
    if not setups:
        lines.append("no setups — the funnel is empty")
        return "\n".join(lines)
    for s in setups:
        plan = s.get("plan") or {}
        zone = plan.get("entry_zone") or {}
        targets = ", ".join(_fmt(t) for t in plan.get("targets") or [])
        age = s.get("age_s")
        if age is None and s.get("detected_at"):
            age = taken - float(s["detected_at"])
        last = (s.get("history") or [{}])[-1]
        side = "LONG" if s.get("direction") == 1 else "SHORT"
        lines.append(
            f"{s['playbook']} {side} {s['state']} · "
            f"zone {_fmt(zone.get('bottom'))}–{_fmt(zone.get('top'))} · "
            f"inv {_fmt(plan.get('invalidation'))} · T {targets or '?'}"
            f" · {_age_text(age)} old")
        if last.get("reason"):
            lines.append(f"  {last.get('reason_category')}: "
                         f"{last['reason']}")
    return "\n".join(lines)


def render_pass(summary: dict[str, Any]) -> str:
    """Human rendering of one shadow pass + the live funnel."""
    counts = summary.get("counts") or {}
    lines = [
        f"shadow pass {summary.get('symbol')} — "
        f"created {len(summary.get('created') or [])}, "
        f"armed {len(summary.get('armed') or [])}, "
        f"triggered {len(summary.get('triggered') or [])}, "
        f"invalidated {len(summary.get('invalidated') or [])}, "
        f"expired {len(summary.get('expired') or [])}",
        "reasons: " + ", ".join(f"{k}={v}" for k, v in counts.items()
                                if v),
    ]
    if summary.get("detector_errors"):
        lines.append("detector errors: "
                     + "; ".join(summary["detector_errors"]))
    lines.append(render_setups(summary.get("setups") or [],
                               symbol=summary.get("symbol"),
                               now=summary.get("taken_at")))
    return "\n".join(lines)
