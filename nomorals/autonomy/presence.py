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
  being asked, but respects quiet hours and a hard cooldown (enforced
  in code, not in a prompt). Never spammy. Every surface is audited in
  ``presence_surface``.

Presence never sends directly — it emits events and proposals. The
existing autonomy agent (``nomorals.agents.autonomy``) owns the actual
sending with its safety vetoes. Presence is the mind; the autonomy
agent is the hands.

Runs on the idle tick and on a slow heartbeat (every ~30 min).
"""

from __future__ import annotations

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


def _now_in_tz(tz_offset: float = 0.0) -> dict[str, Any]:
    import datetime as _dt
    now = _dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(
        hours=tz_offset)
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
    return {
        "time": now,
        "active_windows": _patterns.predict_active_windows(
            db, tz_offset=tz_offset),
        "interests": _patterns.current_interests(db, limit=8),
        "rising": _patterns.rising_interests(db, limit=5),
        "weaknesses": _weakness.open_weaknesses(db)[:5],
        "last_trigger": trig,
        "anticipation": anticipation,
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


def surface(db: Any, kind: str, title: str, detail: str = "",
            *, force: bool = False,
            ts: float | None = None) -> dict[str, Any]:
    """Propose one serendipitous surface.

    Records it in ``presence_surface`` (audit) and publishes a
    ``presence.serendipity`` bus event. The actual delivery is owned by
    the autonomy agent / runtime with its safety vetoes — presence only
    proposes. The cooldown is enforced in code: surfaces more often than
    ``SERENDIPITY_COOLDOWN`` are held (``force=True`` bypasses, for
    owner-triggered surfaces only).
    """
    ensure_schema(db)
    now = ts if ts is not None else time.time()
    last = _last_surface_ts(db)
    if not force and now - last < SERENDIPITY_COOLDOWN:
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
    _log.info("presence surface #%s [%s]: %s", surface_id, kind, title)
    _ledger(db, "surfaced", str(surface_id),
            f"serendipity surfaced [{kind}]: {title}",
            metadata={"detail": detail[:300]})
    try:
        global_bus.publish(Event(
            topic="presence.serendipity",
            data={"surface_id": surface_id, "kind": kind,
                  "title": title, "detail": detail[:500]},
            source="nomorals.autonomy.presence",
        ))
    except Exception:  # noqa: BLE001
        _log.debug("presence.serendipity publish failed", exc_info=True)
    return {"ok": True, "surface_id": surface_id}


def heartbeat(db: Any, context: Any = None,
              tz_offset: float = 0.0) -> dict[str, Any]:
    """Slow heartbeat — notice, prepare, surface. Returns what it did.

    Called every ~30 min by the scheduler and on every idle tick.
    Never raises; never sends directly.
    """
    did: dict[str, Any] = {"noticed": [], "prepared": [], "surfaced": []}
    try:
        ensure_schema(db)
        ctx = sense(db, tz_offset)

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

        # ── surface: serendipity via the cooldown-guarded path ──
        # (Actual sends are owned by nomorals.agents.autonomy with its
        # safety vetoes. Presence only proposes.)
        surfaced = _maybe_surface(db, ctx)
        if surfaced:
            did["surfaced"].append(surfaced)

        try:
            global_bus.publish(Event(
                topic="presence.heartbeat",
                data={"daypart": ctx["time"]["daypart"],
                      "rising": [r["topic"] for r in ctx["rising"]],
                      "noticed": did["noticed"]},
                source="nomorals.autonomy.presence",
            ))
        except Exception:  # noqa: BLE001
            pass

        _ledger(db, "heartbeat", f"hb-{int(ctx['ts'])}",
                f"heartbeat: {len(did['noticed'])} noticed, "
                f"{len(did['prepared'])} prepared, "
                f"{len(did['surfaced'])} surfaced",
                metadata={"daypart": ctx["time"]["daypart"],
                          "noticed": did["noticed"][:5],
                          "prepared": did["prepared"][:5],
                          "surfaced": did["surfaced"][:3]})
    except Exception as exc:  # noqa: BLE001
        _log.warning("presence heartbeat failed: %s", exc, exc_info=True)
    return did


def _maybe_surface(db: Any, ctx: dict[str, Any]) -> str | None:
    """Pick one serendipity candidate, at most one per heartbeat.

    Candidates in priority order: pending weakness proposals (the owner
    should know something was found), then the top rising interest.
    Returns the title surfaced, or None.
    """
    for w in ctx["weaknesses"]:
        if w["status"] == "proposed":
            title = f"Fix proposal ready: {w['subject']}"
            res = surface(
                db, "weakness_proposal", title,
                (w.get("proposal") or "")[:300])
            return title if res.get("ok") else None
    rising = ctx["rising"]
    if rising:
        top = rising[0]
        res = surface(
            db, "rising_interest",
            f"You're deep into {top['topic']} lately",
            f"sightings={top['sightings']} score={top['score']}")
        if res.get("ok"):
            return f"You're deep into {top['topic']} lately"
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
