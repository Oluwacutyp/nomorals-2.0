"""Presence — the "alive" layer. She notices, prepares, acts.

This is what makes the system feel alive instead of command-driven:

* **Time-aware** — knows the time of day, day of week, owner's timezone.
  Morning hits different from 2am.
* **Notices** — rising interests, recurring patterns, upcoming active
  windows, weaknesses clustering. Turns observations into organ events.
* **Prepares** — before a predicted active window, pre-fetches what's
  likely wanted (news before the morning check, research on rising
  interests). Anticipation, not reaction. The pattern model
  (``likely_next``) tells her what usually follows a trigger, so idle
  time prepares the *right* thing.
* **Serendipity** — surfaces genuinely interesting findings without
  being asked. Candidates are scored on novelty × relevance ×
  unexpectedness (the RecSys serendipity decomposition), gated by a
  learned **interruptibility** score (InterruptMe-style: daypart +
  activity recency + learned quiet hours), and delivered as one
  **batched digest** — not scattered pings. Quiet hours are learned
  from the routine model and enforced in code, not in a prompt.
* **Learns taste** — every surface records the owner's reaction
  (engaged / dismissed / ignored) and re-weights that surface kind.
  Surfacing that annoys gets quieter on its own.

Presence never sends directly — it emits events and proposals. The
existing autonomy agent (``nomorals.agents.autonomy``) owns the actual
sending with its safety vetoes. Presence is the mind; the autonomy
agent is the hands.

Runs on the idle tick and on a slow heartbeat (every ~30 min).
"""

from __future__ import annotations

import re
import time
from typing import Any

from ..core.events import Event, global_bus
from ..core.logging_setup import get_logger
from .. import organs as _organs
from . import patterns as _patterns
from . import weakness as _weakness

_log = get_logger(__name__)

#: Minimum seconds between serendipitous surfaces. Enforced in code.
SERENDIPITY_COOLDOWN = 6 * 3600  # 6 hours

#: Heartbeat interval for the presence check (seconds).
HEARTBEAT_INTERVAL = 1800  # 30 minutes

#: Minimum seconds between research nudges for the same topic.
NUDGE_COOLDOWN = 24 * 3600  # 24 hours

#: Interruptibility below this holds non-urgent surfaces.
INTERRUPTIBILITY_FLOOR = 0.35

#: Surface priority tiers (proactive-assistant hierarchy: urgent push,
#: normal batched, background digest-only).
PRIORITIES = ("urgent", "normal", "background")
_PRIORITY_RANK = {"urgent": 0, "normal": 1, "background": 2}

#: Base interruptibility by daypart (before context adjustments).
_DAYPART_INTERRUPTIBILITY = {
    "morning": 0.75,
    "afternoon": 0.65,
    "evening": 0.55,
    "night": 0.15,
}

#: Daypart-aware voice for digests — how she sounds, not just what.
_DAYPART_TONE = {
    "morning": ("☀️", "morning briefing", "Quick and useful — the day's ahead."),
    "afternoon": ("🌤️", "afternoon notes", "A few things worth your eyes."),
    "evening": ("🌙", "evening roundup", "Winding down? A couple of finds."),
    "night": ("🌌", "night notes", "Only the important stuff — it's late."),
}


def ensure_schema(db: Any) -> None:
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS presence_surface (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            kind TEXT NOT NULL DEFAULT '',
            title TEXT NOT NULL DEFAULT '',
            detail TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'pending'
        )
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS presence_nudge (
            key TEXT PRIMARY KEY,
            last_nudged REAL NOT NULL DEFAULT 0
        )
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS presence_trigger (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            trigger_sig TEXT NOT NULL DEFAULT '',
            ts REAL NOT NULL DEFAULT 0
        )
        """
    )
    db.execute(
        "INSERT OR IGNORE INTO presence_trigger (id) VALUES (1)"
    )
    # Candidate queue: bounded-deferral surfacing.
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS presence_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            kind TEXT NOT NULL DEFAULT '',
            title TEXT NOT NULL DEFAULT '',
            detail TEXT NOT NULL DEFAULT '',
            priority TEXT NOT NULL DEFAULT 'normal',
            deadline_ts REAL NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'queued'
        )
        """
    )
    # Learned per-kind surfacing weights (feedback loop).
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS presence_kind_weights (
            kind TEXT PRIMARY KEY,
            weight REAL NOT NULL DEFAULT 1.0,
            engaged INTEGER NOT NULL DEFAULT 0,
            dismissed INTEGER NOT NULL DEFAULT 0,
            ignored INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    # Surface reactions for the feedback loop.
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS presence_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            surface_id INTEGER NOT NULL DEFAULT 0,
            kind TEXT NOT NULL DEFAULT '',
            reaction TEXT NOT NULL DEFAULT '',
            ts REAL NOT NULL DEFAULT 0
        )
        """
    )


def note_trigger(db: Any, trigger_sig: str,
                 ts: float | None = None) -> None:
    """Remember the last system trigger (e.g. 'morning-pulse',
    'idle-heartbeat'). Cheap. Lets ``sense()`` pull ``likely_next``
    anticipation for what usually follows."""
    ensure_schema(db)
    now = ts if ts is not None else time.time()
    db.execute(
        "UPDATE presence_trigger SET trigger_sig = ?, ts = ? WHERE id = 1",
        (str(trigger_sig or ""), now),
    )
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        pass


def last_trigger(db: Any) -> dict[str, Any]:
    ensure_schema(db)
    row = db.execute(
        "SELECT trigger_sig, ts FROM presence_trigger WHERE id = 1"
    ).fetchone()
    if not row:
        return {"trigger_sig": "", "ts": 0.0}
    return {"trigger_sig": row[0] or "", "ts": float(row[1] or 0.0)}


def _nudge_due(db: Any, key: str, cooldown: float = NUDGE_COOLDOWN,
               ts: float | None = None) -> bool:
    """True when this nudge key hasn't fired within the cooldown.

    Claim-style: a due nudge marks itself immediately, so concurrent
    heartbeats can't double-emit.
    """
    ensure_schema(db)
    now = ts if ts is not None else time.time()
    row = db.execute(
        "SELECT last_nudged FROM presence_nudge WHERE key = ?",
        (key,)).fetchone()
    last = float(row[0]) if row else 0.0
    if now - last < cooldown:
        return False
    db.execute(
        "INSERT INTO presence_nudge (key, last_nudged) VALUES (?, ?) "
        "ON CONFLICT (key) DO UPDATE SET last_nudged = excluded.last_nudged",
        (key, now),
    )
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        pass
    return True


def _last_surface_ts(db: Any) -> float:
    ensure_schema(db)
    row = db.execute(
        "SELECT MAX(ts) FROM presence_surface").fetchone()
    return float(row[0]) if row and row[0] else 0.0


def _now_in_tz(tz_offset: float = 0.0,
               ts: float | None = None) -> dict[str, Any]:
    import datetime as _dt
    base = (_dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc)
            if ts is not None
            else _dt.datetime.now(_dt.timezone.utc))
    now = base + _dt.timedelta(hours=tz_offset)
    hour = now.hour
    if 5 <= hour < 12:
        part = "morning"
    elif 12 <= hour < 17:
        part = "afternoon"
    elif 17 <= hour < 22:
        part = "evening"
    else:
        part = "night"
    return {
        "hour": hour, "dow": now.weekday(),
        "dow_name": now.strftime("%A"),
        "daypart": part,
        "iso": now.isoformat(timespec="minutes"),
    }


# ── interruptibility ─────────────────────────────────────────────────

def interruptibility(db: Any, tz_offset: float = 0.0,
                     ts: float | None = None) -> dict[str, Any]:
    """How interruptible is the owner right now? 0.0–1.0.

    InterruptMe-style contextual model, learned locally:
    daypart base × learned quiet hours × recent-activity boost.
    A score below ``INTERRUPTIBILITY_FLOOR`` holds non-urgent surfaces
    for a better moment (bounded deferral) instead of pinging into the
    void — or into sleep.
    """
    ensure_schema(db)
    now = ts if ts is not None else time.time()
    t = _now_in_tz(tz_offset, ts)
    score = _DAYPART_INTERRUPTIBILITY.get(t["daypart"], 0.5)
    reasons = [f"daypart={t['daypart']} base={score}"]

    # Learned quiet hours crush the score — never ping at 3am just
    # because the cooldown elapsed.
    try:
        quiet = _patterns.quiet_hours(db, tz_offset=tz_offset)
        if t["hour"] in quiet:
            score = min(score, 0.05)
            reasons.append(f"quiet hour ({t['hour']:02d}:00)")
    except Exception:  # noqa: BLE001
        quiet = []

    # Recently active → highly interruptible (they're right here).
    # (No activity history at all is NOT "recently active".)
    try:
        from . import idle as _idle
        st = _idle.idle_state(db)
        idle_for = st.get("idle_seconds", 0.0)
        last_ts = st.get("last_activity_ts", 0.0)
        if last_ts > 0 and idle_for < 900:  # active within 15 min
            score = max(score, 0.9)
            reasons.append("active <15m ago")
        elif last_ts > 0 and idle_for > 8 * 3600:
            score = min(score, 0.2)
            reasons.append("quiet >8h (probably away/asleep)")
    except Exception:  # noqa: BLE001
        pass

    score = max(0.0, min(1.0, score))
    return {
        "score": round(score, 2),
        "daypart": t["daypart"],
        "hour": t["hour"],
        "quiet_hours": quiet,
        "interruptible": score >= INTERRUPTIBILITY_FLOOR,
        "reasons": reasons,
        "ts": now,
    }


# ── candidate queue (bounded deferral) ───────────────────────────────

def queue_candidate(db: Any, kind: str, title: str, detail: str = "",
                    priority: str = "normal",
                    ttl_hours: float = 24.0,
                    ts: float | None = None) -> int:
    """Queue a serendipity candidate with a bounded deferral deadline.

    The candidate waits for a high-interruptibility moment instead of
    firing immediately; past ``deadline_ts`` it becomes due regardless
    (bounded — never dropped silently, never held forever).
    Returns the candidate id.
    """
    ensure_schema(db)
    now = ts if ts is not None else time.time()
    if priority not in _PRIORITY_RANK:
        priority = "normal"
    cur = db.execute(
        "INSERT INTO presence_candidates "
        "(ts, kind, title, detail, priority, deadline_ts, status) "
        "VALUES (?, ?, ?, ?, ?, ?, 'queued')",
        (now, kind, title, detail, priority, now + ttl_hours * 3600),
    )
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        pass
    cid = cur.lastrowid if cur is not None else 0
    _log.info("presence candidate #%s [%s/%s]: %s", cid, priority, kind,
              title)
    return int(cid or 0)


def due_candidates(db: Any, ts: float | None = None,
                   limit: int = 10) -> list[dict[str, Any]]:
    """Queued candidates, best first (priority, then serendipity score).

    A candidate is *due* when its deadline is near (< 1h left) or
    interruptibility is high — otherwise it keeps waiting. Expired
    candidates (deadline passed, still queued) are always due.
    """
    ensure_schema(db)
    now = ts if ts is not None else time.time()
    rows = db.execute(
        "SELECT id, ts, kind, title, detail, priority, deadline_ts FROM "
        "presence_candidates WHERE status = 'queued' ORDER BY id "
        "LIMIT ?", (max(1, int(limit)) * 2,)).fetchall()
    scored = []
    for r in rows:
        cid, cts, kind, title, detail, priority, deadline = r
        time_left = deadline - now
        if time_left <= 0:
            due, why = True, "deadline passed"
        elif time_left < 3600:
            due, why = True, "deadline <1h"
        else:
            due, why = False, "waiting for a better moment"
        scored.append({
            "id": cid, "ts": cts, "kind": kind, "title": title,
            "detail": detail, "priority": priority,
            "deadline_ts": deadline,
            "serendipity": serendipity_score(
                db, kind, title, detail)["score"],
            "due": due, "due_why": why,
        })
    scored.sort(key=lambda c: (_PRIORITY_RANK.get(c["priority"], 1),
                               -c["serendipity"]))
    return scored[:max(1, int(limit))]


def drop_candidate(db: Any, candidate_id: int) -> bool:
    """Withdraw a queued candidate (superseded, no longer relevant)."""
    ensure_schema(db)
    cur = db.execute(
        "UPDATE presence_candidates SET status = 'dropped' "
        "WHERE id = ? AND status = 'queued'", (int(candidate_id),))
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        pass
    return bool(cur.rowcount)


# ── serendipity scoring ──────────────────────────────────────────────

_WORD_RE = re.compile(r"[a-z][a-z\-']{2,}")


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower())
            if w not in _patterns._STOPWORDS}


def serendipity_score(db: Any, kind: str, title: str,
                      detail: str = "") -> dict[str, Any]:
    """Score a candidate: novelty × relevance × unexpectedness.

    The RecSys decomposition (Ge et al., RecSys 2010): serendipity is
    the intersection of surprising AND useful. "You're deep into X
    lately" is a tautology — this scores whether it's actually worth
    surfacing, multiplied by the learned per-kind taste weight.
    """
    ensure_schema(db)
    words = _content_words(f"{title} {detail}")
    # Novelty: dissimilar to recent surfaces.
    try:
        rows = db.execute(
            "SELECT title FROM presence_surface ORDER BY id DESC "
            "LIMIT 10").fetchall()
        recent = [_content_words(r[0]) for r in rows]
        if recent and words:
            overlap = max(
                len(words & rw) / max(len(words | rw), 1) for rw in recent)
        else:
            overlap = 0.0
        novelty = 1.0 - overlap
    except Exception:  # noqa: BLE001
        novelty = 0.7
    # Relevance: ties to real current interests.
    try:
        interests = {i["topic"] for i in
                     _patterns.current_interests(db, limit=15)}
        if words and interests:
            hits = sum(1 for w in words
                       if any(w in t or t in w for t in interests))
            relevance = min(1.0, hits / max(len(words), 1) * 3.0)
        else:
            relevance = 0.3
    except Exception:  # noqa: BLE001
        relevance = 0.3
    # Unexpectedness: not the obvious next thing.
    if kind == "weakness_proposal":
        unexpectedness = 0.85  # a found fix is genuinely surprising
    elif kind == "rising_interest":
        unexpectedness = 0.45  # adjacent to known interests
    elif kind == "digest":
        unexpectedness = 0.3
    else:
        unexpectedness = 0.6
    # Learned taste weight for this kind.
    weight = kind_weight(db, kind)
    score = (novelty * 0.4 + relevance * 0.35 + unexpectedness * 0.25)
    score *= weight
    return {
        "score": round(max(0.0, min(1.5, score)), 3),
        "novelty": round(novelty, 2),
        "relevance": round(relevance, 2),
        "unexpectedness": round(unexpectedness, 2),
        "kind_weight": round(weight, 2),
    }


# ── feedback loop (learned taste) ────────────────────────────────────

def kind_weight(db: Any, kind: str) -> float:
    """Learned surfacing weight for a surface kind (default 1.0)."""
    ensure_schema(db)
    row = db.execute(
        "SELECT weight FROM presence_kind_weights WHERE kind = ?",
        (str(kind or ""),)).fetchone()
    return float(row[0]) if row else 1.0


def record_surface_feedback(db: Any, surface_id: int,
                            reaction: str) -> dict[str, Any]:
    """Record the owner's reaction to a surface.

    ``reaction``: ``engaged`` (opened/acted), ``dismissed`` (swiped
    away), ``ignored`` (never opened). Tunes the per-kind weight so
    annoying surface kinds get quieter on their own and welcome ones
    get bolder. Returns the new weight.
    """
    ensure_schema(db)
    reaction = str(reaction or "").lower()
    if reaction not in ("engaged", "dismissed", "ignored"):
        return {"ok": False, "reason": "unknown reaction"}
    row = db.execute(
        "SELECT kind FROM presence_surface WHERE id = ?",
        (int(surface_id),)).fetchone()
    if not row:
        return {"ok": False, "reason": "unknown surface"}
    kind = row[0] or ""
    db.execute(
        "INSERT INTO presence_feedback (surface_id, kind, reaction, ts) "
        "VALUES (?, ?, ?, ?)",
        (int(surface_id), kind, reaction, time.time()))
    db.execute(
        "INSERT INTO presence_kind_weights (kind, weight) VALUES (?, 1.0) "
        "ON CONFLICT (kind) DO NOTHING", (kind,))
    col = {"engaged": "engaged", "dismissed": "dismissed",
           "ignored": "ignored"}[reaction]
    db.execute(
        f"UPDATE presence_kind_weights SET {col} = {col} + 1 "
        "WHERE kind = ?", (kind,))
    wrow = db.execute(
        "SELECT weight FROM presence_kind_weights WHERE kind = ?",
        (kind,)).fetchone()
    weight = float(wrow[0]) if wrow else 1.0
    if reaction == "engaged":
        weight = min(2.0, weight * 1.15)
    elif reaction == "dismissed":
        weight = max(0.2, weight * 0.7)
    else:
        weight = max(0.3, weight * 0.9)
    db.execute(
        "UPDATE presence_kind_weights SET weight = ? WHERE kind = ?",
        (weight, kind))
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        pass
    _log.info("surface #%s %s → kind '%s' weight %.2f", surface_id,
              reaction, kind, weight)
    return {"ok": True, "kind": kind, "weight": round(weight, 2)}


def kind_stats(db: Any) -> list[dict[str, Any]]:
    """Per-kind surfacing stats: how each kind is landing."""
    ensure_schema(db)
    rows = db.execute(
        "SELECT kind, weight, engaged, dismissed, ignored FROM "
        "presence_kind_weights ORDER BY weight DESC").fetchall()
    return [{"kind": r[0], "weight": round(r[1], 2), "engaged": r[2],
             "dismissed": r[3], "ignored": r[4]} for r in rows]


# ── sensing ──────────────────────────────────────────────────────────

def sense(db: Any, tz_offset: float = 0.0) -> dict[str, Any]:
    """Take a snapshot of what the system knows right now.

    Returns the full presence context: time, predicted windows,
    interests, rising topics, open weaknesses, and anticipation
    (what the pattern model says usually follows the last trigger).
    """
    now = _now_in_tz(tz_offset)
    trig = last_trigger(db)
    anticipation: list[dict[str, Any]] = []
    if trig["trigger_sig"]:
        try:
            anticipation = _patterns.likely_next(
                db, trig["trigger_sig"])[:3]
        except Exception:  # noqa: BLE001
            anticipation = []
    try:
        interrupt = interruptibility(db, tz_offset)
    except Exception:  # noqa: BLE001
        interrupt = {"score": 0.5, "interruptible": True}
    return {
        "time": now,
        "active_windows": _patterns.predict_active_windows(
            db, tz_offset=tz_offset),
        "next_activity": _patterns.next_activity(
            db, tz_offset=tz_offset),
        "interests": _patterns.current_interests(db, limit=8),
        "rising": _patterns.rising_interests(db, limit=5),
        "clusters": _patterns.topic_clusters(db)[:3],
        "weaknesses": _weakness.open_weaknesses(db)[:5],
        "last_trigger": trig,
        "anticipation": anticipation,
        "interruptibility": interrupt,
        "ts": time.time(),
    }


def _ledger(db: Any, kind: str, ref_id: str, summary: str,
            *, metadata: dict[str, Any] | None = None) -> None:
    try:
        from ..agents.autonomy_ledger import record_ledger
        record_ledger(db, "presence", kind, ref_id, summary,
                      metadata=metadata)
    except Exception:  # noqa: BLE001 - telemetry is fail-open
        _log.debug("presence ledger write failed", exc_info=True)


# ── surfacing ────────────────────────────────────────────────────────

def surface(db: Any, kind: str, title: str, detail: str = "",
            *, force: bool = False,
            priority: str = "normal",
            ts: float | None = None) -> dict[str, Any]:
    """Propose one serendipitous surface.

    Records it in ``presence_surface`` (audit) and publishes a
    ``presence.serendipity`` bus event. The actual delivery is owned by
    the autonomy agent / runtime with its safety vetoes — presence only
    proposes. The cooldown is enforced in code: surfaces more often than
    ``SERENDIPITY_COOLDOWN`` are held (``force=True`` bypasses, for
    owner-triggered surfaces only). ``priority="urgent"`` also bypasses
    the interruptibility gate (but never the delivery owner's vetoes).
    """
    ensure_schema(db)
    now = ts if ts is not None else time.time()
    if priority not in _PRIORITY_RANK:
        priority = "normal"
    last = _last_surface_ts(db)
    if not force and priority != "urgent" and \
            now - last < SERENDIPITY_COOLDOWN:
        return {"ok": False, "held": True,
                "reason": "cooldown",
                "next_due_in": round(SERENDIPITY_COOLDOWN - (now - last), 1)}
    cur = db.execute(
        "INSERT INTO presence_surface (ts, kind, title, detail, status) "
        "VALUES (?, ?, ?, ?, 'pending')",
        (now, kind, title, detail),
    )
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        pass
    surface_id = cur.lastrowid if cur is not None else 0
    _log.info("presence surface #%s [%s/%s]: %s", surface_id, priority,
              kind, title)
    _ledger(db, "surfaced", str(surface_id),
            f"serendipity surfaced [{kind}]: {title}",
            metadata={"detail": detail[:300], "priority": priority})
    try:
        global_bus.publish(Event(
            topic="presence.serendipity",
            data={"surface_id": surface_id, "kind": kind,
                  "title": title, "detail": detail[:500],
                  "priority": priority},
            source="nomorals.autonomy.presence",
        ))
    except Exception:  # noqa: BLE001
        _log.debug("presence.serendipity publish failed", exc_info=True)
    return {"ok": True, "surface_id": surface_id}


def heartbeat(db: Any, context: Any = None,
              tz_offset: float = 0.0) -> dict[str, Any]:
    """Slow heartbeat — notice, prepare, surface. Returns what it did.

    Called every ~30 min by the scheduler and on every idle tick.
    Surfaces are interruptibility-gated and batched: the best due
    candidate goes out as one surface; the rest wait for a better
    moment or the digest. Never raises; never sends directly.
    """
    did: dict[str, Any] = {"noticed": [], "prepared": [], "surfaced": []}
    try:
        ensure_schema(db)
        ctx = sense(db, tz_offset)
        interrupt = ctx.get("interruptibility", {})
        int_score = float(interrupt.get("score", 0.5))

        # ── notice: rising interests → research organ (deduped) ──
        for item in ctx["rising"]:
            topic = item["topic"]
            if not _nudge_due(db, f"interest:{topic}"):
                continue
            _log.info("presence: rising interest '%s' — nudging research",
                      topic)
            try:
                _organs.emit(
                    db, src="autonomy.presence", dst="research",
                    kind="interest.watch",
                    payload={"topic": topic,
                             "reason": "rising interest",
                             "score": item["score"]})
                did["noticed"].append(f"rising interest: {topic}")
            except Exception:  # noqa: BLE001
                _log.debug("interest→research emit failed", exc_info=True)

        # ── notice: open weaknesses → already handled by weakness module,
        # but presence keeps them visible in the snapshot ──
        for w in ctx["weaknesses"]:
            if w["status"] == "proposed":
                did["noticed"].append(
                    f"weakness proposal pending: {w['subject']}")

        # ── prepare: upcoming active window → pre-fetch ──
        windows = ctx["active_windows"]
        if windows:
            top = windows[0]
            _log.info("presence: predicted active window %02d:00 (dow %d)",
                      top["hour"], top["dow"])
            try:
                _organs.emit(
                    db, src="autonomy.presence", dst="research",
                    kind="window.prepare",
                    payload={"hour": top["hour"], "dow": top["dow"],
                             "interests": [i["topic"]
                                           for i in ctx["interests"][:3]]})
                did["prepared"].append(
                    f"pre-fetch for {top['hour']:02d}:00 window")
            except Exception:  # noqa: BLE001
                _log.debug("window→research emit failed", exc_info=True)

        # ── prepare: anticipation from the pattern model ──
        for ant in ctx["anticipation"]:
            _log.info("presence: pattern '%s' → usually '%s' (%dx)",
                      ctx["last_trigger"]["trigger_sig"],
                      ant["action"], ant["count"])
            try:
                _organs.emit(
                    db, src="autonomy.presence", dst="research",
                    kind="pattern.prepare",
                    payload={"trigger": ctx["last_trigger"]["trigger_sig"],
                             "likely_action": ant["action"],
                             "count": ant["count"]})
                did["prepared"].append(
                    f"pattern prep: {ant['action']} "
                    f"after {ctx['last_trigger']['trigger_sig']}")
                break  # one anticipation prep per heartbeat is enough
            except Exception:  # noqa: BLE001
                _log.debug("pattern→research emit failed", exc_info=True)

        # ── surface: one batched, interruptibility-gated surface ──
        # (Actual sends are owned by nomorals.agents.autonomy with its
        # safety vetoes. Presence only proposes.)
        surfaced = _maybe_surface(db, ctx, int_score)
        if surfaced:
            did["surfaced"].append(surfaced)
        did["interruptibility"] = int_score

        try:
            global_bus.publish(Event(
                topic="presence.heartbeat",
                data={"daypart": ctx["time"]["daypart"],
                      "rising": [r["topic"] for r in ctx["rising"]],
                      "noticed": did["noticed"],
                      "interruptibility": int_score},
                source="nomorals.autonomy.presence",
            ))
        except Exception:  # noqa: BLE001
            pass

        _ledger(db, "heartbeat", f"hb-{int(ctx['ts'])}",
                f"heartbeat: {len(did['noticed'])} noticed, "
                f"{len(did['prepared'])} prepared, "
                f"{len(did['surfaced'])} surfaced",
                metadata={"daypart": ctx["time"]["daypart"],
                          "interruptibility": int_score,
                          "noticed": did["noticed"][:5],
                          "prepared": did["prepared"][:5],
                          "surfaced": did["surfaced"][:3]})
    except Exception as exc:  # noqa: BLE001
        _log.warning("presence heartbeat failed: %s", exc, exc_info=True)
    return did


def _auto_candidates(db: Any, ctx: dict[str, Any]) -> list[dict[str, Any]]:
    """Build serendipity candidates from the current snapshot.

    Priority order: pending weakness proposals (the owner should know
    something was found), then the top rising interest, then topic
    clusters ("you're into X+Y lately").
    """
    cands: list[dict[str, Any]] = []
    for w in ctx["weaknesses"]:
        if w["status"] == "proposed":
            cands.append({
                "kind": "weakness_proposal",
                "title": f"Fix proposal ready: {w['subject']}",
                "detail": (w.get("proposal") or "")[:300],
                "priority": "urgent",
            })
    rising = ctx["rising"]
    if rising:
        top = rising[0]
        cands.append({
            "kind": "rising_interest",
            "title": f"You're deep into {top['topic']} lately",
            "detail": f"sightings={top['sightings']} "
                      f"score={top['score']}",
            "priority": "normal",
        })
    clusters = ctx.get("clusters") or []
    if clusters:
        c = clusters[0]
        cands.append({
            "kind": "interest_cluster",
            "title": f"Noticing a thread: {' + '.join(c[:3])}",
            "detail": "these topics keep appearing together",
            "priority": "background",
        })
    return cands


def _maybe_surface(db: Any, ctx: dict[str, Any],
                   int_score: float) -> str | None:
    """Pick one surface candidate, at most one per heartbeat.

    Queued candidates (bounded deferral) compete with fresh auto
    candidates on serendipity score. Non-urgent winners only go out
    when interruptibility clears the floor — otherwise they stay
    queued for a better moment or the digest. Returns the title
    surfaced, or None.
    """
    now = time.time()
    pool: list[dict[str, Any]] = []
    for c in due_candidates(db, limit=5):
        pool.append({
            "kind": c["kind"], "title": c["title"],
            "detail": c["detail"], "priority": c["priority"],
            "candidate_id": c["id"], "due": c["due"],
        })
    for c in _auto_candidates(db, ctx):
        # Fresh candidates carry no deadline pressure — they wait for a
        # good moment like everything else.
        pool.append({**c, "candidate_id": None, "due": False})
    if not pool:
        return None

    scored = []
    for c in pool:
        s = serendipity_score(db, c["kind"], c["title"], c["detail"])
        scored.append((s["score"], c, s))
    scored.sort(key=lambda t: (
        _PRIORITY_RANK.get(t[1]["priority"], 1), -t[0]))
    best_score, best, _ = scored[0]

    # Gate: urgent always goes; others need interruptibility or a blown
    # deadline. Background-tier never fires alone — digest only.
    if best["priority"] == "background":
        queue_candidate(db, best["kind"], best["title"], best["detail"],
                        priority="background", ttl_hours=72)
        return None
    if best["priority"] != "urgent" and int_score < INTERRUPTIBILITY_FLOOR:
        if not best["due"]:
            # Hold for a better moment.
            if best["candidate_id"] is None:
                queue_candidate(db, best["kind"], best["title"],
                                best["detail"], priority=best["priority"])
            _log.info("presence: holding '%s' (interruptibility %.2f)",
                      best["title"], int_score)
            return None
    res = surface(db, best["kind"], best["title"], best["detail"],
                  priority=best["priority"])
    if res.get("ok"):
        if best.get("candidate_id"):
            db.execute(
                "UPDATE presence_candidates SET status = 'surfaced' "
                "WHERE id = ?", (best["candidate_id"],))
            try:
                db.commit()
            except Exception:  # noqa: BLE001
                pass
        return best["title"]
    return None


def pending_surfaces(db: Any, limit: int = 20) -> list[dict[str, Any]]:
    """Surfaces proposed but not yet delivered/consumed."""
    ensure_schema(db)
    rows = db.execute(
        "SELECT id, ts, kind, title, detail, status FROM presence_surface "
        "WHERE status = 'pending' ORDER BY ts DESC LIMIT ?",
        (max(1, int(limit)),)).fetchall()
    return [{"id": r[0], "ts": r[1], "kind": r[2], "title": r[3],
             "detail": r[4], "status": r[5]} for r in rows]


def mark_surface(db: Any, surface_id: int, status: str) -> bool:
    """Mark a surface delivered/dismissed by the delivery owner."""
    ensure_schema(db)
    cur = db.execute(
        "UPDATE presence_surface SET status = ? WHERE id = ?",
        (str(status or "pending"), int(surface_id)),
    )
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        pass
    return bool(cur.rowcount)


# ── batched digest ───────────────────────────────────────────────────

def digest(db: Any, tz_offset: float = 0.0,
           theme: str = "plain") -> str:
    """Compile one batched briefing from everything pending.

    Queued candidates + pending surfaces + open weakness proposals,
    rendered in a daypart-aware voice. This is the product shape that
    works (ChatGPT Pulse lesson): one briefing, not scattered pings.
    The digest itself is audited as a surface (kind ``digest``).
    """
    ensure_schema(db)
    ctx = sense(db, tz_offset)
    daypart = ctx["time"]["daypart"]
    icon, title, tagline = _DAYPART_TONE.get(daypart, ("✨", "notes", ""))
    items: list[tuple[str, str, str]] = []  # (icon, head, sub)

    for w in ctx["weaknesses"]:
        if w["status"] == "proposed":
            items.append(("🛠️", f"Fix proposal: {w['subject']}",
                          (w.get("proposal") or "")[:160]))
    for c in due_candidates(db, limit=5):
        items.append(("💡", c["title"], c["detail"][:160]))
    for s in pending_surfaces(db, limit=5):
        if s["kind"] == "digest":
            continue
        items.append(("🔔", s["title"], s["detail"][:160]))
    rising = ctx.get("rising") or []
    if rising and len(items) < 6:
        top = rising[0]
        items.append(("📈", f"You're deep into {top['topic']} lately",
                      f"{top['sightings']} sightings this week"))

    if theme == "rich":
        lines = [f"╭─ {icon} Devon {title} ─╮",
                 f"│ {tagline}"]
        if not items:
            lines.append("│ nothing pending — enjoy the quiet.")
        for ic, head, sub in items[:8]:
            lines.append(f"│ {ic} {head[:52]}")
            if sub:
                lines.append(f"│   {sub[:54]}")
        lines.append("╰─────────────────────────╯")
        text = "\n".join(lines)
    else:
        lines = [f"{icon} Devon {title} — {tagline}"]
        if not items:
            lines.append("nothing pending — enjoy the quiet.")
        for ic, head, sub in items[:8]:
            lines.append(f"{ic} {head}")
            if sub:
                lines.append(f"   {sub}")
        text = "\n".join(lines)

    # Audit the digest itself.
    try:
        db.execute(
            "INSERT INTO presence_surface (ts, kind, title, detail, "
            "status) VALUES (?, 'digest', ?, ?, 'pending')",
            (time.time(), f"{title} ({len(items)} items)",
             text[:500]),
        )
        db.commit()
    except Exception:  # noqa: BLE001
        pass
    return text


def _heartbeat_fire(scheduler: Any, job: Any) -> None:
    """Scheduler action: run the presence heartbeat."""
    try:
        ws_dir = None
        try:
            ws_dir = job.get("parameters", {}).get("workspace_dir")
        except Exception:  # noqa: BLE001
            pass
        if not ws_dir:
            # Fall back to cwd-adjacent workspace.
            from pathlib import Path as _P
            ws_dir = str(_P.cwd())
        from .idle import workspace_db
        from .presence import heartbeat as _hb
        db = workspace_db(ws_dir)
        did = _hb(db)
        _log.info("presence heartbeat fired: %s", did)
    except Exception as exc:  # noqa: BLE001
        _log.warning("presence heartbeat action failed: %s", exc)


def ensure_heartbeat_job(scheduler: Any) -> None:
    """Register the durable 30-min presence heartbeat (idempotent)."""
    try:
        if hasattr(scheduler, "register_action"):
            scheduler.register_action(
                "__presence_heartbeat__", _heartbeat_fire)
        have = [j for j in scheduler.list_jobs()
                if j.get("name") == "presence-heartbeat"]
        if not have:
            scheduler.add(
                name="presence-heartbeat",
                spec="*/30 * * * *",
                payload_kind="tool",
                payload={"tool": "__presence_heartbeat__", "args": {}},
                metadata={"organ": "autonomy.presence"},
            )
            _log.info("presence heartbeat job registered")
    except Exception as exc:  # noqa: BLE001
        _log.warning("presence heartbeat registration failed: %s", exc)
