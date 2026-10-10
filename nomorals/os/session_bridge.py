"""Bridge: one OS Session per chat, across every surface.

The problem: Telegram, WhatsApp, Discord, CLI, TUI, and the API each
managed their own ad-hoc conversation state. A message on Telegram and a
command in the CLI from the same owner hit different brains with different
memories.

The fix: :class:`SessionBridge` maps every (platform, chat_id) to a single
:class:`~nomorals.os.session.Session`. The chat gateway calls
:meth:`session_for_message` on inbound; the CLI already attaches its own
session via ``cmdline/dispatch.py``. Everything downstream — memory,
persona, gating, missions — keys off the Session, not the platform.

Conversation IDs are deterministic: ``"{platform}:{chat_id}"``. A chat that
already has an active session reuses it; otherwise a new one is created
with the platform, chat kind, and gating mode in its ``state`` dict.

Layering: this module lives in ``os/`` (L6, control plane) and reaches
down to ``social`` (L4) only for the message types. The chat gateway (L4)
never imports this — the bridge is injected as an optional dependency.
"""

from __future__ import annotations

from typing import Any

from ..core.logging_setup import get_logger
from ..social.chat.base import ChatKind, ChatMessage
from .session import Session, SessionStore

__all__ = ["SessionBridge", "conversation_id_for"]

_log = get_logger(__name__)


def _now() -> float:
    import time

    return time.time()


def conversation_id_for(platform: str, chat_id: str) -> str:
    """Deterministic conversation ID for a (platform, chat) pair."""
    return f"{platform}:{chat_id}"


class SessionBridge:
    """Get-or-create OS Sessions for chat messages, on every platform.

    Parameters
    ----------
    db:
        Database (or path) for the underlying :class:`SessionStore`.
    gating_fn:
        Optional callable ``(chat, is_owner) -> mode`` used to record the
        gating mode (owner/private/group) in the session state. When omitted
        the mode is derived from ``is_owner`` and ``chat.kind`` with the
        same rules as ``partner.gating.classify_chat``.
    """

    def __init__(self, db: Any = None, gating_fn: Any = None) -> None:
        self.store = SessionStore(db=db)
        self._gating_fn = gating_fn

    # ── public API ────────────────────────────────────────────────────
    def session_for_message(self, message: ChatMessage,
                            *, is_owner: bool) -> Session:
        """Return the active OS Session for this message's chat.

        Reuses the latest active session with the matching conversation ID;
        creates one (with platform/kind/gating in ``state``) when none
        exists. Never raises — on store failure a transient in-memory
        session is returned so the message is never dropped.
        """
        platform = message.chat.platform
        chat_id = message.chat.chat_id
        conv_id = conversation_id_for(platform, chat_id)
        try:
            for session in reversed(self.store.list_active()):
                if session.conversation_id == conv_id:
                    session.touch()
                    self.store.update(session)
                    return session
            return self._create(platform, chat_id, conv_id, message, is_owner)
        except Exception:  # noqa: BLE001 — session attach never drops messages
            _log.debug("SessionBridge fallback for %s", conv_id, exc_info=True)
            return Session(
                id=f"transient-{conv_id}",
                principal="owner" if is_owner else "guest",
                frontend=platform,
                conversation_id=conv_id,
                state=self._state_for(message, is_owner),
            )

    def session_for_cli(self, *, principal: str = "owner") -> Session:
        """Get-or-create the CLI owner's session (same path as dispatch)."""
        try:
            active = [s for s in self.store.list_active()
                      if s.frontend == "cli" and s.principal == principal]
            if active:
                session = active[-1]
                session.touch()
                self.store.update(session)
                return session
            return self.store.create(frontend="cli", principal=principal,
                                     conversation_id="cli:console")
        except Exception:  # noqa: BLE001 — never break the caller
            _log.debug("SessionBridge CLI fallback", exc_info=True)
            return Session(id="transient-cli:console", principal=principal,
                           frontend="cli", conversation_id="cli:console")

    def end_session(self, session_id: str) -> bool:
        """End a session by ID. Returns False when it did not exist."""
        return self.store.end(session_id)

    def handoff_session(self, session_id: str, platform: str, chat_id: str,
                        *, is_owner: bool | None = None) -> Session | None:
        """Move a conversation to a different (platform, chat) — phone to
        desktop, Telegram to CLI — keeping its session, history and state.

        The conversation id becomes ``"{platform}:{chat_id}"``; the old
        binding is recorded in ``state["handoff_history"]``.  Returns the
        updated session, or None when the session is unknown.
        """
        session = self.store.get(session_id)
        if session is None:
            return None
        old_conv = session.conversation_id
        new_conv = conversation_id_for(platform, chat_id)
        history = list(session.state.get("handoff_history") or [])
        history.append({
            "from": old_conv, "to": new_conv, "ts": _now(),
            "from_frontend": session.frontend,
        })
        session.state["handoff_history"] = history
        session.conversation_id = new_conv
        session.frontend = platform
        if is_owner is not None:
            session.principal = "owner" if is_owner else "guest"
        session.touch(activity=True)
        self.store.update(session)
        _log.info("session %s handed off %s -> %s", session_id, old_conv,
                  new_conv)
        return session

    def sessions_for_principal(self, principal: str) -> list[Session]:
        """Every active session belonging to ``principal``."""
        return [s for s in self.store.list_active()
                if s.principal == principal]

    def session_counts(self) -> dict[str, Any]:
        """Active sessions by frontend and by principal."""
        by_frontend: dict[str, int] = {}
        by_principal: dict[str, int] = {}
        active = self.store.list_active()
        for s in active:
            by_frontend[s.frontend] = by_frontend.get(s.frontend, 0) + 1
            by_principal[s.principal] = by_principal.get(s.principal, 0) + 1
        return {"total": len(active), "by_frontend": by_frontend,
                "by_principal": by_principal}

    def render(self) -> str:
        """Plain-text bridge overview."""
        counts = self.session_counts()
        lines = [f"session bridge — {counts['total']} active"]
        for frontend, n in sorted(counts["by_frontend"].items()):
            lines.append(f"  {frontend}: {n}")
        return "\n".join(lines)

    # ── internals ─────────────────────────────────────────────────────
    def _create(self, platform: str, chat_id: str, conv_id: str,
                message: ChatMessage, is_owner: bool) -> Session:
        state = self._state_for(message, is_owner)
        state["platform_chat_id"] = chat_id
        session = self.store.create(
            frontend=platform,
            principal="owner" if is_owner else "guest",
            conversation_id=conv_id,
            state=state,
        )
        _log.info("session %s opened for %s chat %s (mode=%s)",
                  session.id, platform, chat_id,
                  state.get("gating_mode", "?"))
        return session

    def _state_for(self, message: ChatMessage, is_owner: bool) -> dict[str, Any]:
        chat = message.chat
        if self._gating_fn is not None:
            try:
                mode = self._gating_fn(chat, is_owner=is_owner)
            except Exception:  # noqa: BLE001 — gating must not break sessions
                _log.debug("gating_fn failed", exc_info=True)
                mode = self._default_mode(chat, is_owner)
        else:
            mode = self._default_mode(chat, is_owner)
        return {
            "platform": chat.platform,
            "chat_kind": getattr(chat.kind, "value", str(chat.kind)),
            "gating_mode": mode,
            "peer": getattr(chat, "peer", ""),
            "title": getattr(chat, "title", ""),
        }

    @staticmethod
    def _default_mode(chat: Any, is_owner: bool) -> str:
        if is_owner:
            return "owner"
        kind = getattr(chat, "kind", None)
        if kind == ChatKind.DM:
            return "private"
        return "group"
