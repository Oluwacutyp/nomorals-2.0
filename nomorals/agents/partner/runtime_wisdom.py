"""RuntimeWisdomMixin: PartnerRuntime command group (wisdom keeper)."""

from __future__ import annotations

import logging

from ...wisdom import PracticeError, WisdomKeeper

_log = logging.getLogger(__name__)

_WISDOM_USAGE = (
    "usage:\n"
    "  /wisdom ask <query>                  ask the corpus (answer + provenance)\n"
    "  /wisdom practice list                list guided sessions\n"
    "  /wisdom practice <session-id>        start a guided session in this chat\n"
    "  /wisdom practice stop                stop the running session\n"
    "  /wisdom timeline [tradition]         history timeline\n"
    "  /wisdom compare <topic>              cross-tradition comparison\n"
    "  /wisdom status                       corpus status"
)


class RuntimeWisdomMixin:
    """RuntimeWisdomMixin for :class:`PartnerRuntime`."""

    # ── chat practice sessions (one manager per runtime) ──────────────
    def _wisdom_chat_manager(self):
        """Lazily-built :class:`WisdomChatManager` for this runtime."""
        mgr = self.__dict__.get("_wisdom_chat_manager")
        if mgr is None:
            from ...wisdom.chat_session import WisdomChatManager
            mgr = WisdomChatManager(self.context)
            self.__dict__["_wisdom_chat_manager"] = mgr
        return mgr

    def _wisdom_incoming(self, chat_key: str, text: str):
        """Route a plain incoming message to a live practice session.

        Returns the reply to send, or None when no session owns this
        message (normal flow continues). Never raises — a routing
        failure must not eat the owner's message.
        """
        try:
            return self._wisdom_chat_manager().handle_incoming(chat_key, text)
        except Exception:  # noqa: BLE001
            _log.warning("wisdom incoming routing failed", exc_info=True)
            return None

    # ── /wisdom ───────────────────────────────────────────────────────
    def _control_wisdom(self, tail: str, *, chat_key: str,
                        message: object = None) -> str:
        """Dispatch ``/wisdom`` (and its ``/wis`` alias)."""
        parts = (tail or "").strip().split(None, 1)
        verb = parts[0].lower() if parts else ""
        rest = parts[1].strip() if len(parts) > 1 else ""
        keeper = WisdomKeeper(self.context)
        chat = (getattr(message, "chat", None)
                if message is not None else None)
        if chat is None:
            chat = self._ref_from_key(chat_key)

        if verb in ("", "help"):
            return _WISDOM_USAGE
        if verb == "status":
            return self._control_wisdom_status(keeper)
        if verb == "ask":
            return self._control_wisdom_ask(keeper, rest, chat)
        if verb == "practice":
            return self._control_wisdom_practice(keeper, rest,
                                                 chat_key=chat_key, chat=chat)
        if verb == "timeline":
            return self._control_wisdom_timeline(keeper, rest, chat)
        if verb == "compare":
            return self._control_wisdom_compare(keeper, rest, chat)
        return f"unknown /wisdom verb {verb!r}.\n{_WISDOM_USAGE}"

    def _control_wisdom_status(self, keeper) -> str:
        try:
            st = keeper.status()["corpus"]
        except Exception as exc:  # noqa: BLE001 - fail fast, honestly
            return f"⚠️ wisdom status failed: {type(exc).__name__}: {exc}"
        lines = [f"📚 corpus: {st['ingested']}/{st['texts']} texts ingested"]
        for trad, n in sorted(st["by_tradition"].items()):
            lines.append(f"  {trad}: {n}")
        return "\n".join(lines)

    def _control_wisdom_ask(self, keeper, query: str, chat) -> str:
        if not query:
            return "usage: /wisdom ask <query>"
        try:
            ans = keeper.ask(query, top=5)
        except Exception as exc:  # noqa: BLE001
            return f"⚠️ ask failed: {type(exc).__name__}: {exc}"
        lines = [ans.synthesis, ""]
        for p in ans.passages[:5]:
            lines.append(f"[{p.work} — {p.section}]")
            lines.append(f"  {p.snippet.strip()}")
            src = p.url or "(no url)"
            lines.append(f"  source: {src} [{p.canon_status}]")
            lines.append("")
        return self._send_long_checked(
            chat.platform, chat, "\n".join(lines).strip()[:6000])

    def _control_wisdom_practice(self, keeper, rest: str, *,
                                 chat_key: str, chat) -> str:
        sub = (rest or "").strip().split(None, 1)
        subverb = sub[0].lower() if sub else "list"
        manager = self._wisdom_chat_manager()
        if subverb == "list":
            try:
                sessions = keeper.practice.list_sessions()
            except Exception as exc:  # noqa: BLE001
                return f"⚠️ practice list failed: {type(exc).__name__}: {exc}"
            lines = ["🧘 practice sessions:"]
            for s in sessions:
                tag = "beginner" if s["beginner"] else "advanced"
                total = int(s["total_seconds"])
                lines.append(
                    f"  /wisdom practice {s['id']} — {s['name']} "
                    f"({total // 60}m{total % 60:02d}s, {tag})")
            lines.append("start one in this chat: /wisdom practice <session-id>")
            return "\n".join(lines)
        if subverb in ("stop", "cancel"):
            return manager.stop_session(chat_key)
        if subverb == "status":
            st = manager.status(chat_key)
            if st["active"]:
                a = st["active"]
                return (f"🧘 running: {a['name']} ({a['session_id']}) "
                        f"— state {a['state']}")
            if st["journal_await"]:
                return (f"no active session — journal reply awaited for "
                        f"{st['journal_await']}.")
            return "no practice session is running here."
        session_id = sub[0]
        try:
            return manager.start_session(
                chat_key, session_id, platform=chat.platform)
        except PracticeError as exc:
            # unknown session id — the message already lists what's available
            return f"⚠️ {exc}"

    def _control_wisdom_timeline(self, keeper, tradition: str, chat) -> str:
        try:
            events = keeper.history.timeline(tradition, -3000, 2100)
        except Exception as exc:  # noqa: BLE001
            return f"⚠️ timeline failed: {type(exc).__name__}: {exc}"
        if not events:
            return "no timeline events found."
        head = (f"📜 timeline — {tradition}"
                if tradition else "📜 timeline")
        lines = [f"{head} ({len(events)} events):"]
        for e in events[:25]:
            yr = (f"{e['start']}" if e["start"] == e["end"]
                  else f"{e['start']}…{e['end']}")
            lines.append(f"  {yr}: {e['title']} [{e['tradition']}]")
        if len(events) > 25:
            lines.append(f"  …and {len(events) - 25} more")
        return self._send_long_checked(chat.platform, chat, "\n".join(lines))

    def _control_wisdom_compare(self, keeper, topic: str, chat) -> str:
        if not topic:
            return "usage: /wisdom compare <topic>"
        try:
            out = keeper.history.compare(topic, corpus=keeper)
        except Exception as exc:  # noqa: BLE001
            return f"⚠️ compare failed: {type(exc).__name__}: {exc}"
        lines = [f"⚖️ topic: {out['topic']}", ""]
        for p in out["passages"][:5]:
            lines.append(f"[{p.get('work', '?')} — {p.get('canon_status', '')}]")
            lines.append(f"  {p.get('snippet', '')[:200]}")
            lines.append("")
        lines.append("— timeline context —")
        for e in out["timeline_context"][:10]:
            lines.append(f"  {e['start']}: {e['title']}")
        return self._send_long_checked(
            chat.platform, chat, "\n".join(lines).strip()[:6000])
