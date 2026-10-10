"""Presence — the "alive" layer. She notices, prepares, acts.

This is what makes the system feel alive instead of command-driven:

* **Time-aware** — knows the time of day, day of week, owner's timezone.
  Morning hits different from 2am.
* **Notices** — rising interests, recurring patterns, upcoming active
  windows, weaknesses clustering. Turns observations into organ events.
* **Prepares** — before a predicted active window, pre-fetches what's
  likely wanted (news before the morning check, research on rising
  interests). Anticipation, not reaction.
* **Serendipity** — surfaces genuinely interesting findings without
  being asked, but respects quiet hours and rate limits. Never spammy.

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

#: Minimum seconds between serendipitous surfaces.
SERENDIPITY_COOLDOWN = 6 * 3600  # 6 hours

#: Heartbeat interval for the presence check (seconds).
HEARTBEAT_INTERVAL = 1800  # 30 minutes


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
    interests, rising topics, open weaknesses, likely-next actions.
    """
    now = _now_in_tz(tz_offset)
    return {
        "time": now,
        "active_windows": _patterns.predict_active_windows(
            db, tz_offset=tz_offset),
        "interests": _patterns.current_interests(db, limit=8),
        "rising": _patterns.rising_interests(db, limit=5),
        "weaknesses": _weakness.open_weaknesses(db)[:5],
        "ts": time.time(),
    }


def heartbeat(db: Any, context: Any = None,
              tz_offset: float = 0.0) -> dict[str, Any]:
    """Slow heartbeat — notice, prepare, surface. Returns what it did.

    Called every ~30 min by the scheduler and on every idle tick.
    Never raises; never sends directly.
    """
    did: dict[str, Any] = {"noticed": [], "prepared": [], "surfaced": []}
    try:
        ctx = sense(db, tz_offset)

        # ── notice: rising interests → research organ ──
        for item in ctx["rising"]:
            topic = item["topic"]
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

        # ── surface: serendipity via the autonomy agent's proposal path ──
        # (Actual sends are owned by nomorals.agents.autonomy with its
        # safety vetoes. Presence only proposes.)
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

    except Exception as exc:  # noqa: BLE001
        _log.warning("presence heartbeat failed: %s", exc, exc_info=True)
    return did


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
        from ...storage.db import Database
        from .presence import heartbeat as _hb
        db = Database(ws_dir)
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
