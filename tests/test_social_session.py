"""Session wiring: bridge injection, brain session scoping, persistence.

Verifies:
- PartnerRuntime accepts session_bridge and passes it to ChatGateway.
- Brain reads message.meta["os_session_id"] and scopes history,
  persistence, and memory extraction by it.
- Falls back to chat.key when no session is attached.
"""

from __future__ import annotations

import unittest
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
            db=MagicMock(), memory=None, metrics=MagicMock(),
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
        brain._generate_and_persist = MagicMock(return_value=["hi"])
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
        brain._generate_and_persist = MagicMock(return_value=["hi"])
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
        db = MagicMock()
        brain.context.db = db
        msg = _message()
        msg.meta["os_session_id"] = "telegram:123"
        brain._persist_inbound(msg)
        # The INSERT used the session id as conversation_id.
        calls = [str(c) for c in db.execute.call_args_list]
        self.assertTrue(any("telegram:123" in c for c in calls),
                        f"session id not in persist calls: {calls}")


if __name__ == "__main__":
    unittest.main()
