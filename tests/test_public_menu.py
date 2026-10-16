"""Part 2d: public bottom menu — bare /start and bare /menu.

* The keyboard is 13 rows, every button a PUBLIC_COMMAND, no
  owner-only command anywhere on it.
* The menu text lists the public commands (grouped names, no
  descriptions) and never owner-only ones.
* handle_control: bare /start and bare /menu return the menu text
  (before the grant check — the menu is public). /menu <group> and
  /start <platform> are NOT bare: they go through the normal path
  (grant deny for strangers; the telegram-bot gate redirects
  /menu <group> so list_catalog can't leak the owner catalog).
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.social.chat.control import PUBLIC_COMMANDS, public_menu_text
from nomorals.social.chat.tgbot_buttons import public_menu_keyboard
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


class PublicMenuKeyboardTests(unittest.TestCase):
    def test_shape(self):
        kb = public_menu_keyboard()
        self.assertEqual(len(kb), 13)  # 5 named rows + 8 game rows
        flat = [b for row in kb for b in row]
        self.assertEqual(len(flat), 52)
        self.assertEqual(len(set(flat)), 52)  # no duplicates
        for button in flat:
            self.assertTrue(button.startswith("/"), button)
            self.assertIn(button[1:], PUBLIC_COMMANDS, button)

    def test_named_rows(self):
        kb = public_menu_keyboard()
        self.assertEqual(kb[0], ["/help", "/search", "/news", "/book"])
        self.assertEqual(kb[1], ["/music", "/play", "/video", "/podcast"])
        self.assertEqual(kb[2], ["/tts", "/stt", "/look", "/image"])
        self.assertEqual(kb[3], ["/lens", "/decode", "/dns", "/whois"])
        self.assertEqual(kb[4], ["/cipher", "/wisdom", "/think", "/game"])
        # Games: 32 across 8 rows of 4.
        self.assertEqual([len(row) for row in kb[5:]], [4] * 8)

    def test_no_owner_only_on_menu(self):
        flat = {b for row in public_menu_keyboard() for b in row}
        for owner_only in ("/quit", "/exec", "/status", "/db", "/power",
                           "/say", "/trial", "/arena", "/start", "/menu",
                           "/list", "/pvp", "/raid", "/blackjack"):
            self.assertNotIn(owner_only, flat, owner_only)


class PublicMenuTextTests(unittest.TestCase):
    def test_lists_public_commands(self):
        text = public_menu_text()
        for kind in PUBLIC_COMMANDS:
            self.assertIn(f"/{kind}", text, kind)
        self.assertIn("tap a button below", text)

    def test_no_owner_only_in_text(self):
        text = public_menu_text()
        for owner_only in ("/quit", "/exec", "/status", "/db", "/power",
                           "/say", "/trial", "/arena", "/platforms"):
            self.assertNotIn(f"{owner_only} ", text + " ", owner_only)


class HandleControlMenuTests(unittest.TestCase):
    def setUp(self):
        self.rt = _bare_runtime()

    def test_bare_start_returns_menu_for_stranger(self):
        out = self.rt.handle_control(
            "/start", "telegram:999",
            message=_msg(STRANGER_CHAT, "/start"))
        self.assertEqual(out, public_menu_text())

    def test_bare_menu_returns_menu_for_stranger(self):
        out = self.rt.handle_control(
            "/menu", "telegram:999",
            message=_msg(STRANGER_CHAT, "/menu"))
        self.assertEqual(out, public_menu_text())

    def test_bare_start_returns_menu_for_owner(self):
        out = self.rt.handle_control(
            "/start", "telegram:1",
            message=_msg(OWNER_CHAT, "/start"))
        self.assertEqual(out, public_menu_text())

    def test_menu_with_group_is_not_the_public_menu(self):
        # /menu <group> goes to the normal path: a stranger is denied
        # (menu is not public), so list_catalog never leaks.
        out = self.rt.handle_control(
            "/menu games", "telegram:999",
            message=_msg(STRANGER_CHAT, "/menu games"))
        self.assertEqual(out, OWNER_DENY)

    def test_start_with_platform_is_not_the_menu(self):
        # /start <platform> is the owner-only hot-start.
        out = self.rt.handle_control(
            "/start telegram", "telegram:999",
            message=_msg(STRANGER_CHAT, "/start telegram"))
        self.assertEqual(out, OWNER_DENY)
