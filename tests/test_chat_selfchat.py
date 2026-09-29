"""Self-chat support: the owner talks to the companion by texting themselves.

Telegram: Saved Messages (outgoing self-chat events are treated as inbound,
with a loop guard against the adapter's own replies).
WhatsApp: "Message Yourself" (the bridge forwards self-chat upserts).

These tests cover the pure decision logic — no network, no Telethon session.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.social.chat.telegram import TelegramAdapter


def _event(chat_id, msg_id, action=None):
    """A stand-in for a Telethon NewMessage event (the attributes the
    self-chat gate reads)."""
    return SimpleNamespace(
        chat_id=chat_id,
        message=SimpleNamespace(id=msg_id, action=action),
    )


class TelegramSelfChatGateTests(unittest.TestCase):
    def _adapter(self, me_id=12345) -> TelegramAdapter:
        a = TelegramAdapter(api_id="1", api_hash="h")
        a._me_id = me_id
        return a

    def test_self_message_is_inbound(self):
        a = self._adapter()
        self.assertTrue(a._outgoing_is_self_chat(_event(12345, 999)))

    def test_message_to_someone_else_is_ignored(self):
        a = self._adapter()
        self.assertFalse(a._outgoing_is_self_chat(_event(777, 999)))

    def test_own_reply_is_ignored_loop_guard(self):
        a = self._adapter()
        a._remember_sent(999)
        self.assertFalse(a._outgoing_is_self_chat(_event(12345, 999)))

    def test_service_message_is_ignored(self):
        a = self._adapter()
        self.assertFalse(a._outgoing_is_self_chat(_event(12345, 999, action="MessagePinned")))

    def test_no_me_yet_is_ignored(self):
        a = self._adapter(me_id=None)
        self.assertFalse(a._outgoing_is_self_chat(_event(12345, 999)))


class TelegramSentIdGuardTests(unittest.TestCase):
    def _adapter(self) -> TelegramAdapter:
        return TelegramAdapter(api_id="1", api_hash="h")

    def test_remember_sent_keeps_ids(self):
        a = self._adapter()
        a._remember_sent(42)
        self.assertIn(42, a._sent_ids)

    def test_remember_sent_bounded(self):
        a = self._adapter()
        for i in range(2500):
            a._remember_sent(i + 1)
        self.assertLessEqual(len(a._sent_ids), 2000)
        self.assertIn(2500, a._sent_ids)  # the newest is always kept

    def test_remember_sent_ignores_bad_ids(self):
        a = self._adapter()
        a._remember_sent(None)
        a._remember_sent("not-a-number")
        a._remember_sent(0)
        self.assertEqual(a._sent_ids, set())


class _FakeTelegramClient:
    """Stand-in for telethon.TelegramClient — records calls, no network."""

    authorized = True
    fail_check = False
    started = False

    def __init__(self, *args, **kwargs) -> None:
        self._auth = type(self).authorized
        self._fail = type(self).fail_check

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def is_user_authorized(self) -> bool:
        if self._fail:
            raise RuntimeError("no network")
        return self._auth

    async def start(self) -> None:
        type(self).started = True


class MeIdExtractionTest(unittest.TestCase):
    def test_extracts_int_id(self) -> None:
        self.assertEqual(TelegramAdapter._me_id_from(SimpleNamespace(id=12345)), 12345)

    def test_none_when_missing(self) -> None:
        self.assertIsNone(TelegramAdapter._me_id_from(None))

    def test_none_when_bad(self) -> None:
        self.assertIsNone(TelegramAdapter._me_id_from(SimpleNamespace(id="abc")))


class TelegramPreflightTest(unittest.TestCase):
    def setUp(self) -> None:
        import sys
        import types

        self._fake_mod = types.ModuleType("telethon")
        self._fake_mod.TelegramClient = _FakeTelegramClient
        self._saved = sys.modules.get("telethon")
        sys.modules["telethon"] = self._fake_mod
        _FakeTelegramClient.started = False
        _FakeTelegramClient.authorized = True
        _FakeTelegramClient.fail_check = False
        self._tmp = tempfile.TemporaryDirectory(prefix="nm-tg-")

    def tearDown(self) -> None:
        import sys

        if self._saved is None:
            sys.modules.pop("telethon", None)
        else:
            sys.modules["telethon"] = self._saved
        self._tmp.cleanup()

    def _adapter(self) -> TelegramAdapter:
        return TelegramAdapter(
            api_id="1", api_hash="h",
            session_path=str(Path(self._tmp.name) / "telegram.session"),
        )

    def test_authorized_session_is_silent(self) -> None:
        _FakeTelegramClient.authorized = True
        self._adapter().preflight()
        self.assertFalse(_FakeTelegramClient.started, "no login should be attempted")

    def test_unauthenticated_stub_triggers_login(self) -> None:
        # The real-world case: an interrupted first run left a session file
        # that is NOT authorized — the old file-exists check skipped login.
        (Path(self._tmp.name) / "telegram.session").write_text("stub bytes")
        _FakeTelegramClient.authorized = False
        self._adapter().preflight()
        self.assertTrue(_FakeTelegramClient.started, "interactive login should run")

    def test_check_failure_leaves_startup_to_run(self) -> None:
        _FakeTelegramClient.fail_check = True
        self._adapter().preflight()  # must not raise
        self.assertFalse(_FakeTelegramClient.started)


class _FakeEvent:
    """The *Event* shape: Telethon wraps the message; .message + .chat_id."""

    def __init__(self, chat_id: int, text: str = "hello") -> None:
        self.chat_id = chat_id
        self.message = SimpleNamespace(
            raw_text=text,
            media=None,
            reply_to=None,
            id=1,
            date=None,
        )


def _raw_message(chat_id: int, text: str = "hello") -> SimpleNamespace:
    """The *Message* shape: some Telethon builds hand the handler the raw
    message itself (no .message attribute). This is the shape that used to
    be dropped silently on the phone."""
    return SimpleNamespace(
        chat_id=chat_id,
        raw_text=text,
        media=None,
        reply_to=None,
        id=1,
        date=None,
    )


class _FakeInboundClient:
    """Resolves entities by numeric chat id; unknown ids raise (like a real
    lookup failure)."""

    def __init__(self, entities: dict) -> None:
        self._entities = entities

    async def get_entity(self, ident):
        key = int(str(ident))
        if key not in self._entities:
            raise RuntimeError(f"could not resolve entity {key}")
        return self._entities[key]


class InboundDeliveryTest(unittest.TestCase):
    """The Telegram inbound funnel: entity resolution via the client,
    allowlist gating, and support for BOTH Telethon object shapes."""

    def _dm_entity(self, chat_id: int) -> SimpleNamespace:
        return SimpleNamespace(
            id=chat_id,
            title=None,
            first_name=f"peer-{chat_id}",
            username=None,
            megagroup=False,
            gigagroup=False,
            channel=False,
        )

    def _adapter(self, chat_allow: str) -> TelegramAdapter:
        return TelegramAdapter(
            api_id="123",
            api_hash="hash",
            session_path="data/telegram.session",
            chat_allow=chat_allow,
            media_dir="/tmp/nm-allow-media",
        )

    def _run(self, adapter, client, event) -> list:
        import asyncio

        delivered: list = []
        asyncio.run(adapter._handle_inbound(event, client, delivered.append, allow_check=True))
        return delivered

    def test_event_shape_is_delivered(self) -> None:
        entity = self._dm_entity(5478650254)
        client = _FakeInboundClient({5478650254: entity})
        adapter = self._adapter("")
        delivered = self._run(adapter, client, _FakeEvent(5478650254))
        self.assertEqual(len(delivered), 1)
        self.assertEqual(delivered[0].text, "hello")

    def test_raw_message_shape_is_delivered(self) -> None:
        # Regression: the phone's Telethon passes the Message itself, and the
        # old code died with "'Message' object has no attribute 'get_entity'".
        entity = self._dm_entity(5478650254)
        client = _FakeInboundClient({5478650254: entity})
        adapter = self._adapter("")
        delivered = self._run(adapter, client, _raw_message(5478650254, "hey there"))
        self.assertEqual(len(delivered), 1)
        self.assertEqual(delivered[0].text, "hey there")
        self.assertEqual(delivered[0].chat.chat_id, "5478650254")

    def test_unresolvable_entity_is_dropped_not_fatal(self) -> None:
        client = _FakeInboundClient({})  # nothing resolves
        adapter = self._adapter("")
        delivered = self._run(adapter, client, _FakeEvent(777))
        self.assertEqual(delivered, [])

    def test_outside_allowlist_is_dropped(self) -> None:
        entity = self._dm_entity(999999)
        client = _FakeInboundClient({999999: entity})
        adapter = self._adapter("5478650254")
        delivered = self._run(adapter, client, _FakeEvent(999999))
        self.assertEqual(delivered, [], "a chat outside the allowlist must not be delivered")

    def test_inside_allowlist_is_delivered(self) -> None:
        entity = self._dm_entity(5478650254)
        client = _FakeInboundClient({5478650254: entity})
        adapter = self._adapter("5478650254")
        delivered = self._run(adapter, client, _FakeEvent(5478650254))
        self.assertEqual(len(delivered), 1)

    def test_empty_allowlist_delivers_everyone(self) -> None:
        entity = self._dm_entity(42)
        client = _FakeInboundClient({42: entity})
        adapter = self._adapter("")
        delivered = self._run(adapter, client, _FakeEvent(42))
        self.assertEqual(len(delivered), 1)

    def test_channel_shape_resolves_and_delivers(self) -> None:
        # -100... ids are channels; the entity id is the positive channel id.
        channel = SimpleNamespace(
            id=1971671774,
            title="Some Channel",
            first_name=None,
            username=None,
            megagroup=False,
            gigagroup=False,
            channel=True,
        )
        client = _FakeInboundClient({-1001971671774: channel})
        adapter = self._adapter("")
        delivered = self._run(adapter, client, _FakeEvent(-1001971671774, "post text"))
        self.assertEqual(len(delivered), 1)
        self.assertEqual(delivered[0].chat.kind, "channel")


class MentionDetectionTest(unittest.TestCase):
    """Tags/mentions of the account itself (@username / first name)."""

    def _adapter(self, username="Peacethefirst", first_name="Vrede") -> TelegramAdapter:
        a = TelegramAdapter(api_id="1", api_hash="h")
        a._me_username = username
        a._me_first_name = first_name
        return a

    def test_username_tag_matches(self) -> None:
        a = self._adapter()
        self.assertTrue(a._mentions_me("hey @Peacethefirst are you there?"))
        self.assertTrue(a._mentions_me("@peacethefirst"))

    def test_longer_username_is_not_a_match(self) -> None:
        a = self._adapter()
        self.assertFalse(a._mentions_me("@peacethefirst123 nope"))

    def test_first_name_matches(self) -> None:
        a = self._adapter()
        self.assertTrue(a._mentions_me("vrede, what's up"))

    def test_first_name_substring_does_not_match(self) -> None:
        a = self._adapter()
        self.assertFalse(a._mentions_me("the vredeless approach"))

    def test_no_identity_no_mention(self) -> None:
        a = self._adapter(username="", first_name="")
        self.assertFalse(a._mentions_me("@Peacethefirst vrede"))

    def test_empty_text(self) -> None:
        self.assertFalse(self._adapter()._mentions_me("   "))


if __name__ == "__main__":
    unittest.main()
