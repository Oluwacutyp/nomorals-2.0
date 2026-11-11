"""Session wiring: bridge injection, brain session scoping, persistence.

Verifies:
- PartnerRuntime accepts session_bridge and passes it to ChatGateway.
- Brain reads message.meta["os_session_id"] and scopes history,
  persistence, and memory extraction by it.
- Falls back to chat.key when no session is attached.
"""

from __future__ import annotations

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

from nomorals.agents.partner.runtime import PartnerRuntime
from nomorals.os.session_bridge import SessionBridge
from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef
from nomorals.social.chat.gateway import ChatGateway
from nomorals.social.chat.local import LocalAdapter


def _chat(platform: str = "telegram", chat_id: str = "123") -> ChatRef:
    return ChatRef(platform=platform, chat_id=chat_id, kind=ChatKind.DM,
                   title="t", peer="p")


def _message(text: str = "hello", **kw) -> ChatMessage:
    return ChatMessage(chat=_chat(**kw), incoming=True, text=text,
                       sender="owner")


def _faithful_db() -> MagicMock:
    """A db double shaped like a production Database on the default rung.

    Production's Database carries NO ``document_store`` attribute unless
    boot attached a non-SQLite store (storage.docstore.attach_document_store
    sets it only then), and the ConversationLog seam dispatches on exactly
    that attribute (storage.rowstore.attached_rows). A bare MagicMock
    fabricates a truthy ``document_store``, so every write silently routes
    into a phantom row table and db.execute is never called — pin the
    faithful default explicitly.
    """
    db = MagicMock()
    db.document_store = None
    return db


class _RecordingLog:
    """A ConversationLog double that records what the brain hands it."""

    last = None

    def __init__(self, db):
        self.db = db
        self.ensured = []
        self.appended = []
        _RecordingLog.last = self

    def transaction(self):
        return nullcontext()

    def ensure_conversation(self, conv_id, **kw):
        self.ensured.append(conv_id)

    def append_message(self, **kw):
        self.appended.append(kw)


class BridgeInjectionTests(unittest.TestCase):
    def test_runtime_accepts_and_forwards_bridge(self):
        ctx = SimpleNamespace(
            settings=SimpleNamespace(
                partner=SimpleNamespace(
                    owner_chats=set(), us_chats=set(),
                    max_parallel_chats=2, platforms="local",
                ),
            ),
            db=MagicMock(),
            extras={},
        )
        bridge = SessionBridge(db=None)
        # Pass an explicit gateway so no adapters are built (no stdin).
        gw = ChatGateway({"local": LocalAdapter()})
        rt = PartnerRuntime.__new__(PartnerRuntime)
        PartnerRuntime.__init__(rt, ctx, gateway=gw, brain=MagicMock(),
                                session_bridge=bridge)
        self.assertIs(rt._session_bridge, bridge)
        self.assertEqual(ctx.extras.get("session_bridge"), bridge)

    def test_gateway_created_with_bridge(self):
        ctx = SimpleNamespace(
            settings=SimpleNamespace(
                partner=SimpleNamespace(
                    owner_chats=set(), us_chats=set(),
                    max_parallel_chats=2, platforms="local",
                ),
            ),
            db=MagicMock(),
            extras={},
        )
        bridge = SessionBridge(db=None)
        rt = PartnerRuntime.__new__(PartnerRuntime)
        # Provide a local adapter so the gateway gets built internally.
        import nomorals.social.chat as chat_pkg
        orig = chat_pkg.build_adapters
        chat_pkg.build_adapters = lambda settings, **kw: ({"local": LocalAdapter()}, [])
        try:
            PartnerRuntime.__init__(rt, ctx, brain=MagicMock(),
                                    session_bridge=bridge)
        finally:
            chat_pkg.build_adapters = orig
        self.assertIsNotNone(rt.gateway)
        self.assertIs(rt.gateway.session_bridge, bridge)

    def test_gateway_without_bridge_still_works(self):
        gw = ChatGateway({"local": LocalAdapter()})
        self.assertIsNone(gw.session_bridge)
        msg = _message()
        # No bridge: meta untouched, handler still runs.
        seen = []
        gw.start(seen.append)
        gw._on_inbound(msg)
        self.assertEqual(len(seen), 1)
        self.assertNotIn("os_session_id", msg.meta)


class BrainSessionScopeTests(unittest.TestCase):
    def _brain(self):
        from nomorals.agents.partner.brain import PartnerBrain
        brain = PartnerBrain.__new__(PartnerBrain)
        brain.context = SimpleNamespace(
            db=_faithful_db(), memory=None, metrics=MagicMock(),
            settings=SimpleNamespace(
                partner=SimpleNamespace(history_window=10),
                memory=SimpleNamespace(extract_enabled=False),
            ),
        )
        brain.persona = SimpleNamespace(name="Test")
        brain._last_user_seen = {}
        brain.presence_rng = __import__("random").Random(0)
        return brain

    def test_handle_message_adopts_session_id(self):
        brain = self._brain()
        # Stub out the heavy path: presence says reply, generation stubbed.
        brain.mood = MagicMock()
        brain.mood.tick.return_value = None
        brain._chat_flags = lambda chat: {"is_owner": True, "in_us": False,
                                          "last_active": 0}
        brain._generate_and_persist = MagicMock(return_value=(["hi"], False, ""))
        import nomorals.agents.partner.brain as brain_mod
        orig_presence = brain_mod.decide_presence
        brain_mod.decide_presence = lambda *a, **k: SimpleNamespace(
            reply=True, delay_seconds=0, reason="test")
        try:
            msg = _message()
            msg.meta["os_session_id"] = "telegram:123"
            brain.handle_message(msg)
        finally:
            brain_mod.decide_presence = orig_presence
        self.assertEqual(msg.meta["os_session_id"], "telegram:123")
        brain._generate_and_persist.assert_called_once()

    def test_handle_message_falls_back_to_chat_key(self):
        brain = self._brain()
        brain.mood = MagicMock()
        brain._chat_flags = lambda chat: {"is_owner": True, "in_us": False,
                                          "last_active": 0}
        brain._generate_and_persist = MagicMock(return_value=(["hi"], False, ""))
        import nomorals.agents.partner.brain as brain_mod
        orig_presence = brain_mod.decide_presence
        brain_mod.decide_presence = lambda *a, **k: SimpleNamespace(
            reply=True, delay_seconds=0, reason="test")
        try:
            msg = _message()  # no os_session_id in meta
            brain.handle_message(msg)
        finally:
            brain_mod.decide_presence = orig_presence
        # Falls back to chat.key with identical scoping.
        self.assertEqual(msg.meta["os_session_id"], msg.chat.key)

    def test_persist_uses_session_conversation_id(self):
        brain = self._brain()
        db = _faithful_db()
        brain.context.db = db
        msg = _message()
        msg.meta["os_session_id"] = "telegram:123"
        brain._persist_inbound(msg)
        # The INSERT used the session id as conversation_id.
        calls = [str(c) for c in db.execute.call_args_list]
        self.assertTrue(any("telegram:123" in c for c in calls),
                        f"session id not in persist calls: {calls}")

    def test_persist_hands_session_id_to_the_conversation_log(self):
        # Seam pin: whatever storage rung is active, _persist_inbound must
        # hand the ConversationLog the SESSION's conversation id — not the
        # chat key, not an empty scope. The session id here deliberately
        # differs from the chat key, so only the session id can satisfy it.
        import nomorals.storage.conversations as conv_mod
        brain = self._brain()
        msg = _message()
        msg.meta["os_session_id"] = "os-session:42"
        orig = conv_mod.ConversationLog
        conv_mod.ConversationLog = _RecordingLog
        try:
            brain._persist_inbound(msg)
        finally:
            conv_mod.ConversationLog = orig
        log = _RecordingLog.last
        self.assertIsNotNone(log)
        self.assertEqual(log.ensured, ["os-session:42"])
        self.assertEqual([a["conversation_id"] for a in log.appended],
                         ["os-session:42"])

    def test_persist_hands_chat_key_to_the_log_without_a_session(self):
        # Seam pin, fallback half: with no session attached, the log
        # receives the chat key — the scope is never empty.
        import nomorals.storage.conversations as conv_mod
        brain = self._brain()
        msg = _message()  # no os_session_id in meta
        orig = conv_mod.ConversationLog
        conv_mod.ConversationLog = _RecordingLog
        try:
            brain._persist_inbound(msg)
        finally:
            conv_mod.ConversationLog = orig
        log = _RecordingLog.last
        self.assertIsNotNone(log)
        self.assertEqual(log.ensured, [msg.chat.key])
        self.assertEqual([a["conversation_id"] for a in log.appended],
                         [msg.chat.key])


if __name__ == "__main__":
    unittest.main()
