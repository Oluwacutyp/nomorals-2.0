"""Infra spec P3 (§3): style — mine -> store -> apply -> tune.

- Tune: /style inspects + overrides (deltas, provenance, reversible).
- Privacy: style is mined from the OWNER's messages only (§3.2).
- Overrides win over the computed profile, never rewrite the miner.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.partner.style import observed_owner_texts, style_for_chat
from nomorals.partner.style_overrides import (
    apply_overrides,
    clear_overrides,
    get_overrides,
    get_overrides_detailed,
    set_override,
)


class TestStyleOverrideStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-test-style-")
        self.path = os.path.join(self.tmp.name, "overrides.json")
        self.env = mock.patch.dict(
            os.environ, {"NOMORALS_STYLE_OVERRIDES_PATH": self.path})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_round_trip_and_provenance(self):
        set_override("telegram:1", "emoji_cap", 0)
        self.assertEqual(get_overrides("telegram:1"), {"emoji_cap": 0})
        detailed = get_overrides_detailed("telegram:1")
        self.assertEqual(detailed["emoji_cap"]["value"], 0)
        self.assertEqual(detailed["emoji_cap"]["set_by"], "owner")
        self.assertGreater(detailed["emoji_cap"]["set_at"], 0)

    def test_global_inherits_chat_wins(self):
        set_override("global", "emoji_cap", 1)
        set_override("telegram:1", "target_chars", 200)
        merged = get_overrides("telegram:1")
        self.assertEqual(merged, {"emoji_cap": 1, "target_chars": 200})
        set_override("telegram:1", "emoji_cap", 0)
        self.assertEqual(get_overrides("telegram:1")["emoji_cap"], 0)

    def test_clear_is_reversible(self):
        set_override("telegram:1", "emoji_cap", 0)
        self.assertTrue(clear_overrides("telegram:1"))
        self.assertEqual(get_overrides("telegram:1"), {})
        self.assertFalse(clear_overrides("telegram:1"))

    def test_invalid_field_and_value_rejected(self):
        with self.assertRaises(ValueError):
            set_override("telegram:1", "tone", "kind")
        with self.assertRaises(ValueError):
            set_override("telegram:1", "emoji_cap", 99)
        with self.assertRaises(ValueError):
            set_override("telegram:1", "allow_sections", "maybe")

    def test_corrupt_store_loads_as_defaults(self):
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write("{not json")
        self.assertEqual(get_overrides("telegram:1"), {})


class TestApplyOverrides(unittest.TestCase):
    def test_overrides_win(self):
        base = style_for_chat(audience="owner", platform="telegram")
        tuned = apply_overrides(base, {"emoji_cap": 0, "target_chars": 120,
                                       "allow_sections": True})
        self.assertEqual(tuned.emoji_budget, 0)
        self.assertEqual(tuned.target_chars, 120)
        self.assertTrue(tuned.allow_sections)
        self.assertEqual(tuned.audience, base.audience)  # only deltas change

    def test_unknown_fields_ignored(self):
        base = style_for_chat(audience="owner", platform="telegram")
        self.assertIs(apply_overrides(base, {"tone": "kind"}), base)
        self.assertIs(apply_overrides(base, {}), base)

    def test_style_for_chat_applies_overrides(self):
        profile = style_for_chat(audience="owner", platform="telegram",
                                 overrides={"emoji_cap": 0})
        self.assertEqual(profile.emoji_budget, 0)


class TestObservedOwnerOnly(unittest.TestCase):
    def _msg(self, role, content, name):
        return SimpleNamespace(role=role, content=content, name=name)

    def test_owner_history_mined(self):
        history = [self._msg("user", "hey lol", "12345"),
                   self._msg("user", "someone else", "99999"),
                   self._msg("assistant", "reply", "")]
        observed = observed_owner_texts(history, gate_mode="owner",
                                        sender_name="12345",
                                        user_text="yo")
        self.assertEqual(observed, ["hey lol", "yo"])

    def test_owner_echo_marker_mined(self):
        history = [self._msg("user", "echo text", "__owner__")]
        observed = observed_owner_texts(history, gate_mode="owner",
                                        sender_name="12345")
        self.assertEqual(observed, ["echo text"])

    def test_stranger_history_never_mined(self):
        history = [self._msg("user", "hi stranger here", "99999")]
        for mode in ("private", "group"):
            observed = observed_owner_texts(history, gate_mode=mode,
                                            sender_name="99999",
                                            user_text="hi stranger here")
            self.assertEqual(observed, [], mode)
        # Even in owner mode, another sender's history is not the owner's.
        observed = observed_owner_texts(history, gate_mode="owner",
                                        sender_name="12345",
                                        user_text="owner says")
        self.assertEqual(observed, ["owner says"])


class TestStyleCommand(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-test-style-cmd-")
        self.path = os.path.join(self.tmp.name, "overrides.json")
        self.env = mock.patch.dict(
            os.environ, {"NOMORALS_STYLE_OVERRIDES_PATH": self.path})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def _runtime(self):
        from nomorals.agents.context import build_context
        from nomorals.agents.partner_runtime import PartnerRuntime
        from nomorals.core.config import load_settings
        from nomorals.social.chat.base import ChatAdapter, ChatRef, SendResult
        from nomorals.social.chat.gateway import ChatGateway

        class FakeAdapter(ChatAdapter):
            def __init__(self):
                super().__init__(media_dir="/tmp/nm-test-media")
            def run(self, handler):
                pass
            def send(self, chat, text, *, reply_to="", buttons=None, **kwargs):
                return SendResult(ok=True, platform="telegram", message_id="m1")

        settings = load_settings(overrides={
            "home": self.tmp.name,
            "partner.platforms": "telegram",
            "partner.owner_chats": "5478650254",
            "partner.owner_sender_ids": "5478650254",
        })
        context = build_context(settings, with_executor=False, with_tools=False)
        gateway = ChatGateway({"telegram": FakeAdapter()}, db=context.db)
        return PartnerRuntime(context, gateway=gateway)

    def test_style_shows_profile(self):
        rt = self._runtime()
        out = rt.handle_control("/style", "telegram:5478650254")
        self.assertIn("emoji budget", out)
        self.assertIn("no overrides", out)

    def test_style_set_and_reset(self):
        rt = self._runtime()
        out = rt.handle_control("/style emoji_cap 0", "telegram:5478650254")
        self.assertIn("emoji_cap = 0", out)
        out = rt.handle_control("/style", "telegram:5478650254")
        self.assertIn("emoji_cap = 0", out)
        out = rt.handle_control("/style reset", "telegram:5478650254")
        self.assertIn("cleared", out)
        out = rt.handle_control("/style", "telegram:5478650254")
        self.assertIn("no overrides", out)

    def test_style_unknown_field(self):
        rt = self._runtime()
        out = rt.handle_control("/style tone kind", "telegram:5478650254")
        self.assertIn("unknown style field", out)

    def test_style_is_owner_only(self):
        from nomorals.social.chat.control import caller_may_run
        self.assertTrue(caller_may_run("style", True))
        self.assertFalse(caller_may_run("style", False))


if __name__ == "__main__":
    unittest.main()
