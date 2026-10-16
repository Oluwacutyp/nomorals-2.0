"""Part 2a: capability gating for the public Telegram bot.

* ``caller_may_run`` — the grant table: owner runs anything, non-owner
  runs PUBLIC_COMMANDS + any game command, nothing else.
* ``handle_control`` — a stranger's owner-only command is denied BEFORE
  dispatch (no handler runs); a stranger's public command dispatches.
* The public-bot gate in ``_process`` forwards PUBLIC_COMMANDS kinds to
  ``handle_control`` and redirects owner-only kinds.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.social.chat.control import (
    GAME_COMMANDS, PUBLIC_COMMANDS, PUBLIC_GAME_COMMANDS, caller_may_run)
from nomorals.storage.db import Database

OWNER_CHAT = SimpleNamespace(key="telegram:1")
STRANGER_CHAT = SimpleNamespace(key="telegram:999")
OWNER_DENY = "that one's just for the owner."


def _bare_runtime():
    rt = PartnerRuntime.__new__(PartnerRuntime)
    rt.context = SimpleNamespace(db=Database(":memory:"))
    rt._owner_chats = {"telegram:1"}
    return rt


def _msg(chat, text):
    return SimpleNamespace(chat=chat, text=text, meta={})


class CallerMayRunTests(unittest.TestCase):
    def test_owner_may_run_anything(self):
        for kind in ("quit", "exec", "status", "db", "help", "poker"):
            self.assertTrue(caller_may_run(kind, True), kind)

    def test_stranger_may_run_public_commands(self):
        for kind in PUBLIC_COMMANDS:
            self.assertTrue(caller_may_run(kind, False), kind)

    def test_stranger_may_run_any_game_command(self):
        # Games are social: even games NOT on the public menu (pvp, raid,
        # casino, inbox, puzzles) run for everyone via game dispatch.
        for kind in GAME_COMMANDS:
            self.assertTrue(caller_may_run(kind, False), kind)

    def test_stranger_denied_owner_only(self):
        for kind in ("quit", "exec", "status", "db", "power", "say",
                     "trial", "upgrade", "arena", "profile", "platforms"):
            self.assertFalse(caller_may_run(kind, False), kind)

    def test_public_list_shape(self):
        # 52 = 19 named + /game + 32 game commands.
        self.assertEqual(len(PUBLIC_GAME_COMMANDS), 32)
        self.assertEqual(len(PUBLIC_COMMANDS), 52)
        self.assertIn("game", PUBLIC_COMMANDS)
        for kind in PUBLIC_GAME_COMMANDS:
            self.assertIn(kind, PUBLIC_COMMANDS)


class HandleControlGrantTests(unittest.TestCase):
    def setUp(self):
        self.rt = _bare_runtime()

    def test_stranger_owner_only_denied_before_dispatch(self):
        for text in ("/quit", "/exec print(1)", "/status", "/db tables",
                     "/power status", "/trial list"):
            out = self.rt.handle_control(
                text, "telegram:999", message=_msg(STRANGER_CHAT, text))
            self.assertEqual(out, OWNER_DENY, text)

    def test_stranger_public_command_dispatches(self):
        out = self.rt.handle_control(
            "/help", "telegram:999",
            message=_msg(STRANGER_CHAT, "/help"))
        self.assertNotEqual(out, OWNER_DENY)
        self.assertTrue(out.strip())

    def test_internal_call_runs_as_owner(self):
        # message=None = internal tour/heal/self-heal probe → owner grant.
        # /help with no message dispatches (no deny).
        out = self.rt.handle_control("/help", "local:console")
        self.assertNotEqual(out, OWNER_DENY)
        self.assertTrue(out.strip())

    def test_stranger_game_command_not_denied_by_grant(self):
        # Game commands are not dispatched by handle_control (the any-chat
        # game dispatch in _process owns them), so this falls through to
        # the "unknown command" fallback — the point is the grant itself
        # does not deny it.
        out = self.rt.handle_control(
            "/poker", "telegram:999",
            message=_msg(STRANGER_CHAT, "/poker"))
        self.assertNotEqual(out, OWNER_DENY)
        self.assertEqual(out, "unknown command /poker")
