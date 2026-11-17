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

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger
from ..social.chat.base import ChatKind, ChatMessage
from .adapters import ADAPTERS
from .project import ProjectStore
from .session import Session, SessionStore

__all__ = ["SessionBridge", "conversation_id_for",
           "try_register_account_source"]

_log = get_logger(__name__)


def _now() -> float:
    import time

    return time.time()


def conversation_id_for(platform: str, chat_id: str) -> str:
    """Deterministic conversation ID for a (platform, chat) pair."""
    return f"{platform}:{chat_id}"


def try_register_account_source(bridge: SessionBridge, db: Any, *,
                                context: Any = None) -> bool:
    """Best-effort: open the accounts vault and register its sessions
    as the bridge's ``"account"`` adapter source (audit gap R2-37).

    The vault is opened the same way the ``nm account`` CLI opens it
    (any owner credential — power key, seal passphrase, legacy env).
    Returns True when the source registered; False when the vault is
    locked/absent or anything else fails — the bridge keeps working
    with its own rows only. Never raises.
    """
    try:
        from ..accounts.sessions import SessionManager
        from ..accounts.vault import open_vault

        vault = open_vault(db, context=context, actor="session-bridge")
        bridge.register_account_manager(SessionManager(vault))
    except Exception:  # noqa: BLE001 — vault locked is normal, not fatal
        _log.debug("account session source unavailable", exc_info=True)
        return False
    return True


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
    adapter_sources:
        Optional mapping ``kind -> source`` registering adapter sources
        up front (see :meth:`register_adapter_source`). Sources are
        callables yielding the raw session concepts for their kind;
        :meth:`list_unified` projects each through the ``ADAPTERS``
        class for that kind.
    """

    def __init__(self, db: Any = None, gating_fn: Any = None, *,
                 adapter_sources: dict[str, Any] | None = None) -> None:
        self.store = SessionStore(db=db)
        # OS projects (AA-28): the scoping shells sessions and artifacts
        # attach to, on the SAME database as the session store — a
        # project summary can then count this bridge's sessions via
        # SessionStore.list_for_project without a second store wiring.
        self.projects = ProjectStore(db=self.store.db)
        self._gating_fn = gating_fn
        # Adapter sources (audit gap R2-37): kind -> [(source, adapter)].
        # A source is a zero-arg callable yielding raw session concepts;
        # the adapter for the kind projects each concept into a Session.
        self._adapter_sources: dict[str, list[tuple[Any, Any]]] = {}
        for kind, source in (adapter_sources or {}).items():
            self.register_adapter_source(kind, source)

    def project_summary(self, project_id: str, *,
                        artifact_store: Any = None) -> dict[str, Any] | None:
        """Dashboard for one OS project: missions by status, this
        bridge's session count, and (when an artifact store is given)
        the project's scoped artifact count. None for an unknown id."""
        return self.projects.summary(
            project_id, session_store=self.store,
            artifact_store=artifact_store)

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
        # The chat KEY is the deterministic conversation id — identical to
        # conversation_id_for(platform, chat_id) for plain chats, and it
        # additionally separates threaded chats (forum topics): two topics
        # in one group are different conversations and must never share a
        # session scope (their histories would bleed into one prompt).
        conv_id = message.chat.key
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
        # History is keyed by conversation id, so the persisted rows must
        # move with the binding — otherwise the handed-off chat starts
        # with amnesia while its history sits orphaned under the old id.
        if new_conv != old_conv:
            self._rekey_conversation(old_conv, new_conv)
        self.store.update(session)
        _log.info("session %s handed off %s -> %s", session_id, old_conv,
                  new_conv)
        return session

    def _rekey_conversation(self, old_conv: str, new_conv: str) -> None:
        """Move persisted conversation rows old -> new (best effort).

        Order matters (foreign keys are enforced): the target
        ``conversations`` row is created first (an existing row for the
        new id wins), then the messages re-key, then the old row goes.
        A store without the chat tables (a standalone bridge DB) simply
        has nothing to move; failures are logged, never fatal — the
        session move itself has already been decided by the caller.
        """
        db = getattr(self.store, "db", None)
        if db is None:
            return
        try:
            from ..storage.conversations import ConversationLog
            ConversationLog(db).rekey(old_conv, new_conv)
        except Exception:  # noqa: BLE001 — history merge must not break handoff
            _log.warning("handoff history re-key %s -> %s failed",
                         old_conv, new_conv, exc_info=True)

    def sessions_for_principal(self, principal: str) -> list[Session]:
        """Every active session belonging to ``principal``."""
        return [s for s in self.store.list_active()
                if s.principal == principal]

    # ── adapter enumeration (audit gap R2-37) ─────────────────────────
    def register_adapter_source(self, kind: str, source: Any, *,
                                adapter: Any = None) -> None:
        """Register a source of raw session concepts for ``kind``.

        ``kind`` must be one of the ``ADAPTERS`` kinds (``account``,
        ``voice``, ``generic``); ``source`` is a zero-arg callable
        returning an iterable of raw concepts — ``(service, username)``
        pairs for ``account``, ``VoiceSession`` objects for ``voice``,
        payload mappings for ``generic``. ``adapter`` is an optional
        already-bound adapter instance (e.g. an
        ``AccountSessionAdapter`` holding a live ``SessionManager``);
        when omitted the adapter class for the kind is constructed
        with no arguments. Several sources may register per kind.
        Unknown kinds and non-callable sources raise ``ValueError``
        here, at registration — enumeration itself never raises.
        """
        kinds = dict(ADAPTERS)
        if kind not in kinds:
            raise ValueError(
                f"unknown session adapter kind: {kind!r} "
                f"(known: {sorted(kinds)})")
        if not callable(source):
            raise ValueError("adapter source must be callable")
        if adapter is None:
            adapter = kinds[kind]()
        self._adapter_sources.setdefault(kind, []).append((source, adapter))

    def register_account_manager(self, manager: Any) -> None:
        """Register a live accounts ``SessionManager`` as the
        ``"account"`` source: its stored ``(service, username)`` pairs
        (see ``SessionManager.list_pairs``) are projected through an
        ``AccountSessionAdapter`` bound to that manager."""
        from .adapters import AccountSessionAdapter

        adapter = AccountSessionAdapter(manager)
        self.register_adapter_source(
            "account", manager.list_pairs, adapter=adapter)

    def list_unified(self) -> list[Session]:
        """The unified session view (audit gap R2-37): the bridge's own
        active sessions first, then one adapter-projected view per
        session concept from every registered source, in ``ADAPTERS``
        order.

        Each view carries its adapter kind in ``state["kind"]``
        (``account`` / ``voice`` / ``generic``); bridge rows carry no
        kind. A concept claimed by two sources yields ONE row (dedup
        by conversation id, first wins). Never raises: a failing
        source is logged and skipped, and a failing projection
        becomes a degraded view (``state["degraded"] is True`` with
        the error named) instead of a fabricated session — the
        bridge's own rows are always returned.
        """
        rows = self.store.list_active()
        seen: set[tuple[str, str]] = set()
        for row in rows:
            if row.conversation_id:
                seen.add(("conv", row.conversation_id))
        for kind, _cls in ADAPTERS:
            for source, adapter in self._adapter_sources.get(kind, []):
                try:
                    concepts = list(source())
                except Exception:  # noqa: BLE001 — one bad source skips
                    _log.warning("session adapter source %r failed", kind,
                                 exc_info=True)
                    continue
                for concept in concepts:
                    try:
                        view = self._project_concept(kind, adapter, concept)
                    except Exception as exc:  # noqa: BLE001 — degrade, don't drop
                        _log.warning(
                            "session adapter %r failed on %r", kind, concept,
                            exc_info=True)
                        state: dict[str, Any] = {
                            "kind": kind,
                            "degraded": True,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        if (isinstance(concept, (tuple, list))
                                and len(concept) == 2):
                            state["service"] = str(concept[0])
                            state["username"] = str(concept[1])
                        view = Session(
                            id=new_short_id("sess_"),
                            principal="owner",
                            frontend=getattr(adapter, "frontend", "api"),
                            state=state,
                        )
                    if view.conversation_id:
                        key = ("conv", view.conversation_id)
                    else:
                        key = ("err", f"{kind}:{view.state.get('error', '')}")
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(view)
        return rows

    @staticmethod
    def _project_concept(kind: str, adapter: Any, concept: Any) -> Session:
        """Project one raw concept through its kind's adapter."""
        if kind == "account":
            if not (isinstance(concept, (tuple, list))
                    and len(concept) == 2):
                raise ValueError(
                    "account concept must be a (service, username) pair")
            return adapter.to_os_session(concept[0], concept[1])
        return adapter.to_os_session(concept)

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
