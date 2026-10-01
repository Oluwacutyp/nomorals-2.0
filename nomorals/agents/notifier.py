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

__all__ = ["Notifier", "notify"]


class Notifier:
    def __init__(self, context: Any, gateway: Any = None) -> None:
        self.context = context
        self.db = getattr(context, "db", None)
        self.settings = getattr(context, "settings", None)
        #: injected by the runtime; None in CLI contexts (delivery = persist only)
        self.gateway = gateway

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
    ) -> dict[str, Any]:
        """Persist an alert and best-effort deliver it. Never raises.

        ``critical`` bypasses the feature gate AND dedupe: build failures and
        security-relevant events must get through even when the owner has the
        notifier muted.

        ``channels`` optionally restricts delivery to a subset of the
        owner's platforms (e.g. ``["telegram"]``); ``None`` (default)
        delivers to every live owner channel, preserving the historical
        behavior for all existing callers.
        """
        if self.db is not None and not force and not critical:
            if self._is_duplicate(kind, title):
                # same alert already went out in the window — skip, don't re-spam
                return {"id": "", "kind": kind, "title": title,
                        "delivered": False, "deduped": True}
            from .features import feature_enabled

            if not feature_enabled(self.context, "notifier"):
                # still record it — the owner can read the queue with /notify
                return self._store(kind, title, body, delivered=0)
        return self._store(kind, title, body, delivered=self._deliver(
            kind, title, body, channels=channels))

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

    def _store(self, kind: str, title: str, body: str, delivered: int) -> dict[str, Any]:
        nid = new_id()
        if self.db is not None:
            try:
                with self.db.transaction():
                    self.db.execute(
                        "INSERT INTO notifications (id, kind, title, body, delivered, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (nid, kind, title, body, delivered, time.time()),
                    )
            except Exception:  # noqa: BLE001 - the alert content is the point
                pass
        return {"id": nid, "kind": kind, "title": title, "delivered": bool(delivered)}

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
        """Undelivered alerts (platforms were down when they fired)."""
        if self.db is None:
            return []
        try:
            return self.db.query(
                "SELECT * FROM notifications WHERE delivered = 0 ORDER BY created_at ASC LIMIT 20"
            )
        except Exception:  # noqa: BLE001
            return []

    def redeliver(self) -> int:
        """Re-send pending alerts now (call after a platform hot-starts)."""
        sent = 0
        for row in self.pending():
            if self._deliver(row["kind"], row.get("title", ""), row.get("body", "")):
                try:
                    self.db.execute(
                        "UPDATE notifications SET delivered = 1 WHERE id = ?", (row["id"],)
                    )
                    sent += 1
                except Exception:  # noqa: BLE001
                    pass
        return sent


def notify(context: Any, kind: str, title: str, body: str = "", **kw: Any) -> dict[str, Any]:
    """One-call helper: ``notify(context, "news", "Headlines", digest)``.

    Pass ``critical=True`` to bypass the feature gate and dedupe window.
    """
    return Notifier(context).publish(kind, title, body, **kw)
