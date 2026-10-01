"""Notifier: durable alerts, delivered to the owner on every live channel.

Every subsystem that produces something the owner should see (arena build
reviews, research suggestions, news digests, finished directives, backup
results) funnels through here instead of each wiring its own send loop.

* **Durable first** — the row lands in ``notifications`` before any send, so
  a dead platform never loses an alert; undelivered rows can be re-sent.
* **Multi-channel** — WhatsApp and/or Telegram (or Discord/local) whichever
  are running in the session; the owner is on at least one.
* **Feature-gated** — the ``notifier`` flag (``/features notifier on|off``).
"""

from __future__ import annotations

import time
from typing import Any

from ..core.ids import new_id

__all__ = ["Notifier", "notify", "resolve_gateway"]


def resolve_gateway(context: Any) -> Any | None:
    """Find the live chat gateway for a context.

    The runtime stores it in ``context.extras["gateway"]`` (see
    ``PartnerRuntime``); older/CLI code may set ``context.gateway``
    directly.  Returns ``None`` when there is no live gateway (CLI
    contexts) — delivery then degrades to persist-only.
    """
    gw = getattr(context, "gateway", None)
    if gw is not None:
        return gw
    try:
        extras = getattr(context, "extras", None)
        if isinstance(extras, dict):
            return extras.get("gateway")
    except Exception:  # noqa: BLE001 - gateway resolution is best-effort
        pass
    return None


#: proactive notification kinds -> PartnerSettings toggle each one obeys.
#: Only these kinds go through the proactive gate; every other kind keeps
#: the historical behavior (feature flag ``notifier`` only).
PROACTIVE_KIND_SETTINGS: dict[str, str] = {
    "briefing": "proactive_briefing",
    "watcher": "proactive_watchers",
}

#: delivery states recorded in ``notifications.delivery_state``
STATE_SENT = "sent"
STATE_FAILED = "failed"            # gateway live, but no owner channel reached
STATE_PENDING = "pending"          # stored; no live gateway (redeliver later)
STATE_HELD_QUIET = "held-quiet-hours"
STATE_DISABLED = "disabled"        # proactive master/kind switch off
STATE_MUTED = "muted"              # feature flag ``notifier`` off
STATE_DEDUPED = "deduped"

#: kinds exempt from Notifier-level quiet hours.  ``briefing`` is a
#: scheduled send the owner explicitly asked for at a chosen time (like an
#: alarm); ``watcher`` alerts carry their own per-watcher quiet-hours
#: logic in the alerting pipeline (watchers.py) — holding them twice
#: would delay urgent hits for no reason.
QUIET_HOURS_EXEMPT = ("briefing", "watcher")


def _partner(context: Any) -> Any | None:
    return getattr(getattr(context, "settings", None), "partner", None)


def _in_quiet_hours_now(context: Any, now: float | None = None) -> bool:
    """True when ``now`` falls in the partner's proactive quiet window."""
    partner = _partner(context)
    if partner is None:
        return False
    try:
        start = int(getattr(partner, "quiet_start", 22) or 0)
        end = int(getattr(partner, "quiet_end", 8) or 0)
    except (TypeError, ValueError):
        return False
    if start == end:
        return False
    # owner's timezone, same resolution order as the briefing
    tzname = None
    settings = getattr(context, "settings", None)
    for attr in ("timezone", "tz", "owner_timezone"):
        tzname = getattr(settings, attr, None) if settings else None
        if tzname:
            break
    if not tzname and partner is not None:
        for attr in ("timezone", "tz"):
            tzname = getattr(partner, attr, None)
            if tzname:
                break
    try:
        from zoneinfo import ZoneInfo
        from datetime import datetime
        tz = ZoneInfo(str(tzname)) if tzname else None
        hour = datetime.now(tz).hour if tz else datetime.now().hour
    except Exception:  # noqa: BLE001 - tz data missing -> wall clock
        from datetime import datetime
        hour = (datetime.fromtimestamp(now).hour if now
                else datetime.now().hour)
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


def proactive_gate(context: Any, kind: str) -> str | None:
    """Decide whether a proactive ``kind`` may send right now.

    Returns ``None`` when sending is allowed, otherwise a
    ``delivery_state`` explaining the hold (``"disabled"`` /
    ``"held-quiet-hours"``).  ``critical`` sends bypass the gate in
    ``publish`` — this helper only encodes the non-critical policy.
    """
    setting = PROACTIVE_KIND_SETTINGS.get(kind)
    if setting is None:
        return None  # not a proactive kind — historical behavior
    partner = _partner(context)
    if partner is None:
        return None  # no settings -> defaults allow (offline tests, CLI)
    if not bool(getattr(partner, "proactive_enabled", True)):
        return STATE_DISABLED
    if not bool(getattr(partner, setting, True)):
        return STATE_DISABLED
    if kind not in QUIET_HOURS_EXEMPT and _in_quiet_hours_now(context):
        return STATE_HELD_QUIET
    return None


class Notifier:
    def __init__(self, context: Any, gateway: Any = None) -> None:
        self.context = context
        self.db = getattr(context, "db", None)
        self.settings = getattr(context, "settings", None)
        #: live gateway when one exists; resolved from the context when not
        #: passed explicitly (the runtime keeps it in context.extras).
        #: None in CLI contexts (delivery = persist only).
        self.gateway = gateway if gateway is not None else resolve_gateway(context)

    #: identical (kind, title) within this window collapses to one alert
    DEDUPE_WINDOW_SECONDS = 600.0

    # ── core ─────────────────────────────────────────────────────────────────
    def publish(
        self,
        kind: str,
        title: str,
        body: str = "",
        *,
        force: bool = False,
        critical: bool = False,
        channels: list[str] | None = None,
        proactive_check: bool = True,
    ) -> dict[str, Any]:
        """Persist an alert and best-effort deliver it. Never raises.

        ``critical`` bypasses every gate: build failures and
        security-relevant events must get through even when the owner has
        the notifier muted.  ``force`` bypasses dedupe and the ``notifier``
        feature flag, but still respects the proactive master/kind
        switches — an explicit "don't push to me" always wins.

        ``channels`` optionally restricts delivery to a subset of the
        owner's platforms (e.g. ``["telegram"]``); ``None`` (default)
        delivers to every live owner channel, preserving the historical
        behavior for all existing callers.

        ``proactive_check`` (default on) applies the proactive gate to
        proactive kinds (``briefing``, ``watcher``): the master
        ``proactive_enabled`` switch, the per-kind toggle, and quiet
        hours.  A held send is still stored — with ``delivery_state``
        ``"disabled"`` / ``"held-quiet-hours"`` — so the owner can see
        what didn't go out and why.
        """
        if self.db is not None and not critical:
            if not force and self._is_duplicate(kind, title):
                # same alert already went out in the window — skip, don't re-spam
                return {"id": "", "kind": kind, "title": title,
                        "delivered": False, "deduped": True,
                        "delivery_state": STATE_DEDUPED}
            from .features import feature_enabled

            if not force and not feature_enabled(self.context, "notifier"):
                # still record it — the owner can read the queue with /notify
                return self._store(kind, title, body, delivered=0,
                                   delivery_state=STATE_MUTED)
            if proactive_check:
                gate = proactive_gate(self.context, kind)
                if gate is not None:
                    return self._store(kind, title, body, delivered=0,
                                       delivery_state=gate)
        if self.gateway is None:
            return self._store(kind, title, body, delivered=0,
                               delivery_state=STATE_PENDING)
        reached = self._deliver(kind, title, body, channels=channels)
        state = STATE_SENT if reached else STATE_FAILED
        return self._store(kind, title, body,
                           delivered=1 if reached else 0,
                           delivery_state=state)

    def _is_duplicate(self, kind: str, title: str) -> bool:
        """True when the same (kind, title) alerted within the dedupe window."""
        try:
            cutoff = time.time() - self.DEDUPE_WINDOW_SECONDS
            row = self.db.query_one(
                "SELECT id FROM notifications WHERE kind = ? AND title = ? AND created_at > ?",
                (kind, title, cutoff),
            )
            return row is not None
        except Exception:  # noqa: BLE001 - dedupe is best-effort
            return False

    def _store(self, kind: str, title: str, body: str, delivered: int,
               delivery_state: str = "") -> dict[str, Any]:
        nid = new_id()
        if self.db is not None:
            try:
                with self.db.transaction():
                    self.db.execute(
                        "INSERT INTO notifications (id, kind, title, body, delivered, "
                        "delivery_state, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (nid, kind, title, body, delivered, delivery_state,
                         time.time()),
                    )
            except Exception:  # noqa: BLE001 - the alert content is the point
                try:
                    # pre-migration DBs lack delivery_state — store anyway
                    with self.db.transaction():
                        self.db.execute(
                            "INSERT INTO notifications (id, kind, title, body, delivered, "
                            "created_at) VALUES (?, ?, ?, ?, ?, ?)",
                            (nid, kind, title, body, delivered, time.time()),
                        )
                except Exception:  # noqa: BLE001
                    pass
        return {"id": nid, "kind": kind, "title": title,
                "delivered": bool(delivered),
                "delivery_state": delivery_state}

    def _deliver(self, kind: str, title: str, body: str,
                 channels: list[str] | None = None) -> int:
        """Send to the owner on every live channel. Returns channels reached.

        ``channels`` restricts delivery to the named platforms
        (``["telegram"]``); ``None`` delivers to all live owner channels.
        """
        if self.gateway is None:
            return 0
        from ..social.chat.base import ChatRef

        text = f"🔔 {title}" + (f"\n{body}" if body else "")
        text = text[:3900]
        partner = getattr(self.settings, "partner", None) if self.settings else None
        raw = getattr(partner, "owner_chats", "") or ""
        reached = 0
        want = {c.strip().lower() for c in (channels or []) if c.strip()}
        for key in raw.split(","):
            plat, _, cid = key.strip().partition(":")
            if not (plat and cid):
                continue
            if want and plat.strip().lower() not in want:
                continue
            try:
                status = self.gateway.status()
                if not status.get(plat, {}).get("running_in_session"):
                    continue
                result = self.gateway.send(plat, ChatRef(platform=plat, chat_id=cid), text)
                if getattr(result, "ok", False):
                    reached += 1
            except Exception:  # noqa: BLE001 - one dead channel must not kill the rest
                continue
        return reached

    # ── read side ────────────────────────────────────────────────────────────
    def recent(self, limit: int = 10, kind: str = "") -> list[dict[str, Any]]:
        if self.db is None:
            return []
        try:
            if kind:
                rows = self.db.query(
                    "SELECT * FROM notifications WHERE kind = ? ORDER BY created_at DESC LIMIT ?",
                    (kind, limit),
                )
            else:
                rows = self.db.query(
                    "SELECT * FROM notifications ORDER BY created_at DESC LIMIT ?", (limit,)
                )
            return rows
        except Exception:  # noqa: BLE001
            return []

    def pending(self) -> list[dict[str, Any]]:
        """Undelivered alerts (platforms were down when they fired, or the
        send was held).  Intentionally-unsent rows (``disabled``,
        ``deduped``, ``muted``) are excluded — redeliver must not
        resurrect what the owner turned off."""
        if self.db is None:
            return []
        try:
            return self.db.query(
                "SELECT * FROM notifications WHERE delivered = 0 "
                "AND COALESCE(delivery_state, '') NOT IN "
                "('disabled', 'deduped', 'muted') "
                "ORDER BY created_at ASC LIMIT 20"
            )
        except Exception:  # noqa: BLE001
            try:  # pre-migration: no delivery_state column
                return self.db.query(
                    "SELECT * FROM notifications WHERE delivered = 0 "
                    "ORDER BY created_at ASC LIMIT 20"
                )
            except Exception:  # noqa: BLE001
                return []

    def redeliver(self) -> int:
        """Re-send pending alerts now (call after a platform hot-starts).

        Rows held for quiet hours are re-checked first — they only go
        out once quiet hours end.
        """
        sent = 0
        for row in self.pending():
            state = (row.get("delivery_state") or "")
            if state == STATE_HELD_QUIET:
                gate = proactive_gate(self.context, row.get("kind") or "")
                if gate == STATE_HELD_QUIET:
                    continue  # still quiet — keep holding
            if self.gateway is None:
                continue
            if self._deliver(row["kind"], row.get("title", ""),
                             row.get("body", "")):
                try:
                    self.db.execute(
                        "UPDATE notifications SET delivered = 1, "
                        "delivery_state = ? WHERE id = ?",
                        (STATE_SENT, row["id"]),
                    )
                    sent += 1
                except Exception:  # noqa: BLE001
                    pass
        return sent

    def delivery_summary(self, limit: int = 10,
                         kinds: tuple[str, ...] = ("briefing", "watcher")
                         ) -> list[dict[str, Any]]:
        """Recent proactive sends with their delivery states, newest first.

        Powers ``nm briefing status`` and the chat ``/notify`` view.
        """
        rows = self.recent(limit * 3)
        out = []
        for row in rows:
            if kinds and (row.get("kind") or "") not in kinds:
                continue
            out.append({
                "id": row.get("id", ""),
                "kind": row.get("kind", ""),
                "title": row.get("title", ""),
                "created_at": row.get("created_at", 0),
                "delivered": bool(row.get("delivered")),
                "delivery_state": (row.get("delivery_state") or
                                   ("sent" if row.get("delivered") else "pending")),
            })
            if len(out) >= limit:
                break
        return out


def notify(context: Any, kind: str, title: str, body: str = "", **kw: Any) -> dict[str, Any]:
    """One-call helper: ``notify(context, "news", "Headlines", digest)``.

    Pass ``critical=True`` to bypass the feature gate and dedupe window.
    """
    return Notifier(context).publish(kind, title, body, **kw)
