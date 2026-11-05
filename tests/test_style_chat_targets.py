"""Register #35: /style targets resolve like /chatmode targets.

The /style command's optional remote-chat target used to be
validated by a local format check (``_valid_target``): anything
that was not literally ``platform:chat_id`` was refused with
"chat keys look like telegram:123456", so tuning another chat's
style meant pasting a raw machine key — while its sibling
/chatmode (register #30/#35) accepted a chat NAME, a phone
number, or the key.  The gate now resolves through
``partner/chat_targets.resolve_chat_target`` exactly like
/chatmode: raw keys keep working verbatim, human forms resolve
to the same canonical key, and an unknown target refuses with
the resolver's exhaustion line — nothing applied anywhere.

The pre-fix failures these tests pin (both refused before the
fix, for format reasons only):

- ``/style emoji_cap 1 Sentinel`` — a one-word chat name died
  in ``_valid_target`` ("chat keys look like telegram:123456").
- ``/style emoji_cap 1 XAUUSD Sentinel`` — a multi-word name
  died even earlier: CONTROL_COMMANDS capped /style at 3 args,
  so parse_control rejected it before the handler ran (the
  register-#30 lesson: the cap made group names untypeable).
"""

from __future__ import annotations

import os
import tempfile
import unittest
from contextlib import suppress
from unittest import mock

from nomorals.partner.chat_targets import note_chat_title
from nomorals.partner.style_overrides import get_overrides, list_scopes
from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef
from nomorals.social.chat.control import parse_control

OWNER_KEY = "telegram:5478650254"
GROUP_KEY = "telegram:-100123456"
GROUP_TITLE = "XAUUSD Sentinel"
WA_GROUP_KEY = "whatsapp:120363@g.us"
PHONE_KEY = "whatsapp:2348012345678@s.whatsapp.net"


def _msg(text: str, *, chat_key: str = OWNER_KEY, title: str = ""):
    platform, _, chat_id = chat_key.partition(":")
    return ChatMessage(
        chat=ChatRef(platform=platform or "telegram",
                     chat_id=chat_id or "5478650254", kind=ChatKind.DM,
                     title=title),
        incoming=True, text=text,
        sender="owner", sender_id="5478650254",
        meta={"is_owner": True},
    )


class StyleChatTargetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(
            prefix="nm-style-targets-",
            dir=os.environ.get("NM_TEST_TMPDIR") or None)
        self.path = os.path.join(self.tmp.name, "overrides.json")
        self.env = mock.patch.dict(
            os.environ, {"NOMORALS_STYLE_OVERRIDES_PATH": self.path})
        self.env.start()
        self.rt = self._runtime()

    def tearDown(self):
        with suppress(Exception):
            self.rt.stop()
        with suppress(Exception):
            self.rt.context.close()
        self.env.stop()
        self.tmp.cleanup()

    def _runtime(self):
        from nomorals.agents.context import build_context
        from nomorals.agents.partner_runtime import PartnerRuntime
        from nomorals.core.config import load_settings
        from nomorals.social.chat.base import ChatAdapter, SendResult
        from nomorals.social.chat.gateway import ChatGateway

        class FakeAdapter(ChatAdapter):
            def __init__(self):
                super().__init__(media_dir="/tmp/nm-test-media")

            def run(self, handler):
                pass

            def send(self, chat, text, *, reply_to="", buttons=None,
                     **kwargs):
                return SendResult(ok=True, platform="telegram",
                                  message_id="m1")

        settings = load_settings(overrides={
            "home": self.tmp.name,
            "partner.platforms": "telegram",
            "partner.owner_chats": "5478650254",
            "partner.owner_sender_ids": "5478650254",
        })
        context = build_context(settings, with_executor=False,
                                with_tools=False)
        gateway = ChatGateway({"telegram": FakeAdapter()}, db=context.db)
        return PartnerRuntime(context, gateway=gateway)

    def _seed_title(self, key: str, title: str) -> None:
        platform, _, chat_id = key.partition(":")
        note_chat_title(self.rt.context.db, ChatRef(
            platform=platform, chat_id=chat_id, kind=ChatKind.GROUP,
            title=title))

    def _style(self, text: str) -> str:
        return self.rt.handle_control(text, OWNER_KEY, message=_msg(text))

    # ── parsing: the arg cap no longer makes names untypeable ──

    def test_parse_allows_multiword_name_targets(self):
        cmd = parse_control("/style emoji_cap 1 XAUUSD Sentinel")
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertEqual(cmd.kind, "style")
        cmd = parse_control("/style reset XAUUSD Sentinel")
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertEqual(cmd.kind, "style")

    # ── (a)/(e) human chat names — the exact pre-fix failures ──

    def test_set_by_single_word_name(self):
        # Pre-fix: refused by _valid_target with "chat keys look
        # like telegram:123456" — a format complaint, nothing set.
        self._seed_title(GROUP_KEY, "Sentinel")
        out = self._style("/style emoji_cap 1 Sentinel")
        self.assertIn("emoji_cap = 1", out)
        self.assertIn(f"Sentinel ({GROUP_KEY})", out)
        self.assertEqual(get_overrides(GROUP_KEY), {"emoji_cap": 1})
        self.assertEqual(get_overrides(OWNER_KEY), {})

    def test_set_by_multiword_name(self):
        # Pre-fix: rejected by parse_control itself (4 args > the
        # old (0, 3) cap) before the handler ever ran.
        self._seed_title(GROUP_KEY, GROUP_TITLE)
        out = self._style(f"/style emoji_cap 1 {GROUP_TITLE}")
        self.assertIn("emoji_cap = 1", out)
        self.assertIn(f"{GROUP_TITLE} ({GROUP_KEY})", out)
        self.assertEqual(get_overrides(GROUP_KEY), {"emoji_cap": 1})

    def test_reset_by_multiword_name(self):
        self._seed_title(GROUP_KEY, GROUP_TITLE)
        self._style(f"/style emoji_cap 1 {GROUP_KEY}")
        self.assertEqual(get_overrides(GROUP_KEY), {"emoji_cap": 1})
        out = self._style(f"/style reset {GROUP_TITLE}")
        self.assertIn("overrides cleared", out)
        self.assertIn(f"{GROUP_TITLE} ({GROUP_KEY})", out)
        self.assertEqual(get_overrides(GROUP_KEY), {})

    # ── (b) a phone number names that person's WhatsApp DM ──

    def test_set_by_phone_number(self):
        out = self._style("/style emoji_cap 2 +234 801 234 5678")
        self.assertIn("emoji_cap = 2", out)
        self.assertIn(PHONE_KEY, out)
        self.assertEqual(get_overrides(PHONE_KEY), {"emoji_cap": 2})

    # ── (c) raw keys keep working, byte-for-byte ──

    def test_set_and_reset_by_raw_key_unchanged(self):
        out = self._style(f"/style emoji_cap 1 {GROUP_KEY}")
        self.assertIn(f"style override for {GROUP_KEY}: emoji_cap = 1",
                      out)
        self.assertEqual(get_overrides(GROUP_KEY), {"emoji_cap": 1})
        out = self._style(f"/style reset {GROUP_KEY}")
        self.assertIn(f"overrides cleared for {GROUP_KEY}", out)
        self.assertEqual(get_overrides(GROUP_KEY), {})

    # ── (d) unknown targets refuse honestly, nothing applied ──

    def test_unknown_name_is_honest_refusal(self):
        out = self._style("/style emoji_cap 1 No Such Chat")
        self.assertIn("no chat matches 'No Such Chat'", out)
        self.assertIn("I tried", out)
        self.assertNotIn("chat keys look like", out)
        self.assertEqual(list_scopes(), [])

    def test_unknown_reset_target_is_honest_refusal(self):
        out = self._style("/style reset No Such Chat")
        self.assertIn("no chat matches 'No Such Chat'", out)
        self.assertIn("I tried", out)
        self.assertEqual(list_scopes(), [])

    def test_ambiguous_name_lists_candidates_never_guesses(self):
        self._seed_title(GROUP_KEY, "Sentinel")
        self._seed_title(WA_GROUP_KEY, "Sentinel")
        out = self._style("/style emoji_cap 1 Sentinel")
        self.assertIn("matches more than one chat", out)
        self.assertIn(GROUP_KEY, out)
        self.assertIn(WA_GROUP_KEY, out)
        self.assertEqual(list_scopes(), [])

    # ── the no-target and this-chat forms are untouched ──

    def test_current_chat_forms_unchanged(self):
        out = self._style("/style emoji_cap 0")
        self.assertIn("style override for this chat: emoji_cap = 0", out)
        self.assertEqual(get_overrides(OWNER_KEY), {"emoji_cap": 0})
        out = self._style("/style reset")
        self.assertIn("overrides cleared for this chat", out)
        self.assertEqual(get_overrides(OWNER_KEY), {})

    def test_this_keyword_targets_current_chat(self):
        out = self._style("/style emoji_cap 1 this")
        self.assertIn("style override for this chat: emoji_cap = 1", out)
        self.assertEqual(get_overrides(OWNER_KEY), {"emoji_cap": 1})


if __name__ == "__main__":
    unittest.main()
