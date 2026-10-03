"""Session unification: one OS Session per chat, across every surface."""

from __future__ import annotations

import unittest

from nomorals.os.session_bridge import SessionBridge, conversation_id_for
from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef


def _msg(platform="telegram", chat_id="123", kind=ChatKind.DM,
         text="hello"):
    chat = ChatRef(platform=platform, chat_id=chat_id, kind=kind,
                   title="Test", peer="tester")
    return ChatMessage(chat=chat, incoming=True, text=text, sender="tester")


class ConversationIdTests(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(conversation_id_for("telegram", "123"),
                         "telegram:123")

    def test_platform_scoped(self):
        self.assertNotEqual(conversation_id_for("telegram", "123"),
                            conversation_id_for("whatsapp", "123"))


class SessionBridgeTests(unittest.TestCase):
    def setUp(self):
        self.bridge = SessionBridge(db=":memory:")

    def test_creates_session_for_chat(self):
        s = self.bridge.session_for_message(_msg(), is_owner=True)
        self.assertEqual(s.frontend, "telegram")
        self.assertEqual(s.principal, "owner")
        self.assertEqual(s.conversation_id, "telegram:123")
        self.assertEqual(s.state["gating_mode"], "owner")

    def test_reuses_session_for_same_chat(self):
        s1 = self.bridge.session_for_message(_msg(), is_owner=True)
        s2 = self.bridge.session_for_message(_msg(text="again"), is_owner=True)
        self.assertEqual(s1.id, s2.id)

    def test_different_chats_different_sessions(self):
        s1 = self.bridge.session_for_message(_msg(chat_id="1"), is_owner=True)
        s2 = self.bridge.session_for_message(_msg(chat_id="2"), is_owner=True)
        self.assertNotEqual(s1.id, s2.id)

    def test_different_platforms_different_sessions(self):
        s1 = self.bridge.session_for_message(_msg(platform="telegram"),
                                             is_owner=True)
        s2 = self.bridge.session_for_message(_msg(platform="whatsapp"),
                                             is_owner=True)
        self.assertNotEqual(s1.id, s2.id)

    def test_gating_mode_private(self):
        s = self.bridge.session_for_message(_msg(), is_owner=False)
        self.assertEqual(s.state["gating_mode"], "private")
        self.assertEqual(s.principal, "guest")

    def test_gating_mode_group(self):
        s = self.bridge.session_for_message(
            _msg(kind=ChatKind.GROUP), is_owner=False)
        self.assertEqual(s.state["gating_mode"], "group")

    def test_state_carries_platform_info(self):
        s = self.bridge.session_for_message(_msg(), is_owner=True)
        self.assertEqual(s.state["platform"], "telegram")
        self.assertEqual(s.state["platform_chat_id"], "123")

    def test_cli_session(self):
        s1 = self.bridge.session_for_cli()
        s2 = self.bridge.session_for_cli()
        self.assertEqual(s1.id, s2.id)
        self.assertEqual(s1.frontend, "cli")

    def test_end_session(self):
        s = self.bridge.session_for_message(_msg(), is_owner=True)
        self.assertTrue(self.bridge.end_session(s.id))
        self.assertFalse(self.bridge.end_session(s.id))
        # After ending, a new message creates a fresh session
        s2 = self.bridge.session_for_message(_msg(), is_owner=True)
        self.assertNotEqual(s.id, s2.id)

    def test_never_raises(self):
        # Even with a broken store, we get a transient session, not an exception
        bridge = SessionBridge(db=":memory:")

        def _boom(*a, **k):
            raise RuntimeError("store is down")
        bridge.store.list_active = _boom
        s = bridge.session_for_message(_msg(), is_owner=True)
        self.assertTrue(s.id.startswith("transient-"))


class GatewayInjectionTests(unittest.TestCase):
    def test_gateway_attaches_session(self):
        from nomorals.social.chat.gateway import ChatGateway

        seen = {}

        class FakeAdapter:
            def start(self, handler):
                seen["handler"] = handler
                return True

        bridge = SessionBridge(db=":memory:")
        gw = ChatGateway(adapters={"telegram": FakeAdapter()},
                         db=None, session_bridge=bridge,
                         owner_chats={"telegram:123"})
        # Simulate inbound without starting threads
        msg = _msg()
        gw._on_inbound(msg)
        self.assertIn("os_session_id", msg.meta)
        self.assertEqual(msg.meta["os_gating_mode"], "owner")

    def test_gateway_without_bridge_still_works(self):
        from nomorals.social.chat.gateway import ChatGateway

        class FakeAdapter:
            def start(self, handler):
                return True

        gw = ChatGateway(adapters={"telegram": FakeAdapter()}, db=None)
        msg = _msg()
        gw._on_inbound(msg)  # must not raise
        self.assertNotIn("os_session_id", msg.meta)


if __name__ == "__main__":
    unittest.main()
