"""Part 2b: confirm-flow for CONFIRMABLE commands (quit/forget).

* Bare request -> confirm prompt (with the ⚠️ confirm: /<kind> line
  the Bot API adapter turns into Confirm/Cancel buttons), pending stored.
* ``/<kind> confirm`` -> the ORIGINAL request runs (stored tail).
* ``/<kind> cancel`` -> dropped, honest message.
* Expired / missing pending -> honest message, nothing runs.
* Non-owner never gets a prompt — the grant deny fires first.
* The pending store is keyed by chat_key and single-use.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.social.chat.tgbot_buttons import buttons_for_text
from nomorals.storage.db import Database

OWNER_CHAT = SimpleNamespace(key="telegram:1")
STRANGER_CHAT = SimpleNamespace(key="telegram:999")
OWNER_DENY = "that one's just for the owner."
CHAT = "telegram:1"


def _bare_runtime():
    rt = PartnerRuntime.__new__(PartnerRuntime)
    rt.context = SimpleNamespace(db=Database(":memory:"))
    rt._owner_chats = {"telegram:1"}
    return rt


def _msg(chat, text):
    return SimpleNamespace(chat=chat, text=text, meta={})


class ConfirmFlowTests(unittest.TestCase):
    def setUp(self):
        self.rt = _bare_runtime()
        self.calls: list = []
        # Intercept dispatch: no real shutdown, no real forgetting.
        self.rt.stop = lambda: self.calls.append(("stop",))
        self.rt._control_forget = lambda tail: self.calls.append(
            ("forget", tail)) or f"forgot {tail}"

    def test_quit_prompt_then_confirm_runs(self):
        prompt = self.rt.handle_control("/quit", CHAT,
                                        message=_msg(OWNER_CHAT, "/quit"))
        self.assertIn("⚠️ confirm: /quit", prompt)
        self.assertIn("/quit confirm", prompt)
        self.assertIn("/quit cancel", prompt)
        self.assertEqual(self.calls, [])
        out = self.rt.handle_control("/quit confirm", CHAT,
                                     message=_msg(OWNER_CHAT, "/quit confirm"))
        self.assertIn("shutting down", out)
        self.assertEqual(self.calls, [("stop",)])

    def test_quit_cancel_drops(self):
        self.rt.handle_control("/quit", CHAT,
                               message=_msg(OWNER_CHAT, "/quit"))
        out = self.rt.handle_control("/quit cancel", CHAT,
                                     message=_msg(OWNER_CHAT, "/quit cancel"))
        self.assertIn("cancelled", out)
        self.assertEqual(self.calls, [])
        # Single-use: a second confirm finds nothing pending.
        out2 = self.rt.handle_control("/quit confirm", CHAT,
                                      message=_msg(OWNER_CHAT, "/quit confirm"))
        self.assertIn("nothing to confirm", out2)
        self.assertEqual(self.calls, [])

    def test_forget_confirm_runs_original_tail(self):
        prompt = self.rt.handle_control("/forget abc-123", CHAT,
                                        message=_msg(OWNER_CHAT, "/forget abc-123"))
        self.assertIn("⚠️ confirm: /forget", prompt)
        self.assertIn("'abc-123'", prompt)
        out = self.rt.handle_control("/forget confirm", CHAT,
                                     message=_msg(OWNER_CHAT, "/forget confirm"))
        self.assertEqual(self.calls, [("forget", "abc-123")])
        self.assertIn("forgot abc-123", out)

    def test_forget_cancel_drops(self):
        self.rt.handle_control("/forget abc-123", CHAT,
                               message=_msg(OWNER_CHAT, "/forget abc-123"))
        out = self.rt.handle_control("/forget cancel", CHAT,
                                     message=_msg(OWNER_CHAT, "/forget cancel"))
        self.assertIn("cancelled", out)
        self.assertEqual(self.calls, [])

    def test_confirm_without_pending_is_honest(self):
        out = self.rt.handle_control("/quit confirm", CHAT,
                                     message=_msg(OWNER_CHAT, "/quit confirm"))
        self.assertIn("nothing to confirm", out)
        self.assertEqual(self.calls, [])

    def test_expired_pending_is_honest(self):
        self.rt.handle_control("/quit", CHAT,
                               message=_msg(OWNER_CHAT, "/quit"))
        # Age the pending entry past the TTL.
        import time as _time
        kind, tail, _ = self.rt._pending_confirms[CHAT]
        self.rt._pending_confirms[CHAT] = (kind, tail, _time.time() - 1)
        out = self.rt.handle_control("/quit confirm", CHAT,
                                     message=_msg(OWNER_CHAT, "/quit confirm"))
        self.assertIn("expired", out)
        self.assertEqual(self.calls, [])

    def test_stranger_never_gets_prompt(self):
        out = self.rt.handle_control("/quit", "telegram:999",
                                     message=_msg(STRANGER_CHAT, "/quit"))
        self.assertEqual(out, OWNER_DENY)
        self.assertNotIn("telegram:999", self.rt._pending_confirms
                         if hasattr(self.rt, "_pending_confirms") else {})

    def test_quit_with_garbage_tail_is_usage_error(self):
        out = self.rt.handle_control("/quit frobnicate", CHAT,
                                     message=_msg(OWNER_CHAT, "/quit frobnicate"))
        self.assertIn("usage: /quit", out)
        self.assertEqual(self.calls, [])

    def test_pending_is_per_chat(self):
        self.rt.handle_control("/quit", "telegram:1",
                               message=_msg(OWNER_CHAT, "/quit"))
        # Another OWNER chat's confirm finds nothing (its own key has no
        # pending) — pending is keyed by chat_key.
        self.rt._owner_chats = {"telegram:1", "telegram:2"}
        other = SimpleNamespace(key="telegram:2")
        out = self.rt.handle_control("/quit confirm", "telegram:2",
                                     message=_msg(other, "/quit confirm"))
        self.assertIn("nothing to confirm", out)
        self.assertEqual(self.calls, [])


class ConfirmButtonsTests(unittest.TestCase):
    def test_prompt_text_yields_confirm_cancel_buttons(self):
        kb = buttons_for_text(
            "⚠️ confirm: /quit — this will shut Devon down.\n"
            "Send /quit confirm to go ahead, or /quit cancel to drop it.")
        self.assertEqual(kb, [[("❌ Cancel", "quit cancel"),
                               ("✅ Confirm", "quit confirm")]])

    def test_forget_prompt_buttons(self):
        kb = buttons_for_text(
            "⚠️ confirm: /forget — this will forget 'abc'.\n"
            "Send /forget confirm to go ahead, or /forget cancel to drop it.")
        self.assertEqual(kb, [[("❌ Cancel", "forget cancel"),
                               ("✅ Confirm", "forget confirm")]])

    def test_plain_text_has_no_confirm_buttons(self):
        self.assertIsNone(buttons_for_text("hello world"))
