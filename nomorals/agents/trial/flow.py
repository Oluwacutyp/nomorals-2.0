"""Single-account trial flow: research a platform, save ONE account's
credentials, and deliver them to the owner on whatever channels are live.

This is deliberately the *single-account, owner-driven* shape the owner
asked for — "create a single account to try it out, save the credentials,
and send it to me on WhatsApp if linked or Telegram, or both if both are
active". The bot does NOT mass-create accounts or automate signups on
third-party sites (that's spam/ToS territory); it researches the platform,
helps the owner through one real signup, encrypts the one credential pair,
and re-delivers it on request.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from ...core.errors import ToolError
from ...core.ids import new_short_id
from .vault import TrialVault

__all__ = ["TrialFlow", "active_delivery_platforms", "new_id"]

#: delivery order — the owner said "WhatsApp if linked or Telegram, or both".
_DELIVERY_ORDER = ("whatsapp", "telegram")


def active_delivery_platforms(gateway: Any) -> list[str]:
    """Platforms that are running in this session, in delivery order.

    ``local`` is excluded (the owner is already there); unknown keys are
    skipped. Returns e.g. ``["whatsapp", "telegram"]``.
    """
    if gateway is None:
        return []
    try:
        status = gateway.status()
    except Exception:  # noqa: BLE001
        return []
    out = []
    for name in _DELIVERY_ORDER:
        info = status.get(name)
        if isinstance(info, dict) and info.get("running_in_session"):
            out.append(name)
    return out


class TrialFlow:
    """Owns the vault and the delivery of one trial account's credentials."""

    def __init__(self, context: Any, *, sender: Callable[[str, str, str], Any] | None = None) -> None:
        self.context = context
        self.settings = getattr(context, "settings", None)
        self.db = getattr(context, "db", None)
        home = getattr(self.settings, "home", "~/.nomorals") if self.settings else "~/.nomorals"
        self.vault = TrialVault(home)
        # sender(platform, chat_key, text) -> SendResult-like; injected for tests.
        self._sender = sender

    # ── research (what a single signup needs) ────────────────────────────────
    def start(self, platform: str) -> str:
        """Research a platform so the owner can complete ONE real signup."""
        platform = (platform or "").strip()
        if not platform:
            raise ToolError("usage: /trial start <platform>")
        engine = self._search_engine()
        lines = [f"trial account plan for: {platform}"]
        lines.append("one account, your details, you finish the signup in the browser.")
        try:
            report = engine.run(f"{platform} create account sign up required details",
                                mode="quick", pages=2)
            summary = (report.get("summary") or "").strip()
            if summary:
                lines.append("")
                lines.append("what they usually ask for:")
                lines.append(summary[:1200])
        except Exception as exc:  # noqa: BLE001 - research is best-effort
            lines.append(f"(research note: {exc})")
        lines.append("")
        lines.append("when you've signed up, store it with:")
        lines.append(f"  /trial save {platform} <login> <password>")
        lines.append("then I'll send it to you on WhatsApp and/or Telegram.")
        return "\n".join(lines)

    # ── save + deliver ───────────────────────────────────────────────────────
    def save(self, platform: str, login: str, secret: str, note: str = "") -> dict:
        if not (platform and login and secret):
            raise ToolError("usage: /trial save <platform> <login> <password>")
        self.vault.store(platform, login, secret, note)
        return {"platform": platform.lower(), "login": login}

    def deliver(self, platform: str, gateway: Any = None, via: str = "") -> str:
        """Send a stored credential pair to the owner on live channels.

        ``via`` is the chat the command came from (e.g. ``telegram:123``) —
        used as a fallback destination per platform.
        """
        entry = self.vault.get(platform)
        if entry is None:
            raise ToolError(f"no stored trial account for {platform!r} (try /trial list)")
        if entry.get("unreadable"):
            raise ToolError(f"{platform}: stored credential is unreadable (key/tamper)")
        secret = entry["secret"]
        body = (
            f"trial account — {entry['platform']}\n"
            f"login: {entry['login']}\n"
            f"password: {secret}"
            + (f"\nnote: {entry['note']}" if entry.get("note") else "")
        )
        targets = self._delivery_targets(gateway, via)
        sent = []
        for plat, key in targets:
            ok = self._send(plat, key, body, gateway)
            if ok:
                sent.append(plat)
        if not sent:
            # No live channel (or dry run): hand the text back so the caller
            # can show it — the credential still exists, just undelivered.
            return "(no live WhatsApp/Telegram in this session — shown here)\n" + body
        return "sent on: " + ", ".join(sent)

    def _delivery_targets(self, gateway: Any, via: str) -> list[tuple[str, str]]:
        """(platform, chat_key) pairs to try, deduped, in delivery order."""
        targets: list[tuple[str, str]] = []
        for plat in active_delivery_platforms(gateway):
            key = self._owner_chat_key(plat) or self._fallback_key(plat, via)
            if key:
                targets.append((plat, key))
        # If the command came from a delivery platform, make sure it's covered
        # even if the gateway report is stale.
        if via and ":" in via:
            plat, _, cid = via.partition(":")
            if plat in _DELIVERY_ORDER and not any(t[0] == plat for t in targets):
                targets.append((plat, f"{plat}:{cid}"))
        return targets

    def _owner_chat_key(self, platform: str) -> str:
        partner = getattr(self.settings, "partner", None) if self.settings else None
        raw = getattr(partner, "owner_chats", "") or ""
        for key in raw.split(","):
            key = key.strip()
            if key.startswith(f"{platform}:"):
                return key
        return ""

    def _fallback_key(self, platform: str, via: str) -> str:
        if via and via.startswith(f"{platform}:"):
            return via
        return ""

    def _send(self, platform: str, chat_key: str, text: str, gateway: Any = None) -> bool:
        if self._sender is not None:
            result = self._sender(platform, chat_key, text)
            return bool(getattr(result, "ok", result))
        if gateway is None:
            gateway = getattr(self.context, "gateway", None)
        if gateway is None:
            return False
        from ...social.chat.base import ChatRef

        plat, _, cid = chat_key.partition(":")
        try:
            result = gateway.send(plat, ChatRef(platform=plat, chat_id=cid), text)
            return bool(getattr(result, "ok", False))
        except Exception:  # noqa: BLE001 - delivery is best-effort
            return False

    # ── listing / removal ────────────────────────────────────────────────────
    def list(self) -> str:
        rows = self.vault.list()
        if not rows:
            return "no stored trial accounts yet."
        lines = ["stored trial accounts:"]
        for row in rows:
            when = time.strftime("%Y-%m-%d", time.localtime(row.get("saved_at", 0)))
            lines.append(f"  {row['platform']}: {row['login']}  (saved {when})")
        return "\n".join(lines)

    def remove(self, platform: str) -> str:
        if self.vault.delete(platform):
            return f"deleted trial account for {platform.lower()}."
        return f"no stored trial account for {platform.lower()}."

    def _search_engine(self) -> Any:
        from ..search.engine import SearchEngine

        return SearchEngine(self.context)


def new_id() -> str:
    return new_short_id(length=12)
