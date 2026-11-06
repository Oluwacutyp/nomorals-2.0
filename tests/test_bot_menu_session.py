"""Bot-DM menu session (2026-11-06 menufix) — the owner screenshot fix.

The bot-DM menu is ONE live message:

* bare /menu in a DM posts the session home (the section index
  rendered from the canonical ``menu_sections`` structure, category
  buttons riding it);
* category taps and page turns EDIT that message through the
  live-session machinery — the catalog never lands as a raw text
  wall and never as a second giant message;
* the persistent catalog reply-keyboard grid no longer attaches on
  the menu flow, and a chat still holding one gets exactly one
  removal carrier (empty ``reply_keyboard`` = ReplyKeyboardRemove);
* groups and WhatsApp keep the catalog text paths.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.social.chat.base import ChatKind, ChatRef, SendResult
from nomorals.social.chat.control import (
    list_catalog,
    menu_sections,
    menu_session_render,
    parse_control,
)
from nomorals.storage.db import Database

PAGE_CEILING = 3600  # budget 3200 + card/stamp framing, under the 4096 cap


def _walk_pages(platform="telegram-personal", public_only=False):
    """Every rendered page of the linear catalog walk, in order,
    reached exactly the way the buttons lead: home → first section →
    Next until the walk ends."""
    pages = []
    render = menu_session_render(platform=platform,
                                 public_only=public_only)
    pages.append(render)
    seen_next = True
    guard = 0
    while seen_next and guard < 200:
        guard += 1
        seen_next = False
        for row in render.buttons:
            for _label, data in row:
                if _label.startswith("Next"):
                    parts = data.split()
                    render = menu_session_render(
                        parts[1], int(parts[2]),
                        platform=platform, public_only=public_only)
                    pages.append(render)
                    seen_next = True
    return pages


class MenuSessionRenderTests(unittest.TestCase):
    def test_home_is_a_session_opener_with_category_buttons(self):
        home = menu_session_render(platform="telegram-personal",
                                   home=True)
        self.assertIsNotNone(home)
        self.assertEqual(home.session, "menu")
        self.assertFalse(home.session_edit)  # the opener posts
        flat = [data for row in home.buttons for _l, data in row]
        self.assertIn("list games", flat)
        self.assertIn("list search", flat)
        self.assertIn("power status", flat)  # personal-menu parity
        # All commands = the start of the linear walk (first section).
        self.assertEqual(flat[-1], flat[0])

    def test_home_lists_every_section_with_counts(self):
        from nomorals.social.chat.style import escape
        home = menu_session_render(platform="telegram-personal",
                                   home=True)
        for _emoji, title, items in menu_sections("telegram-personal"):
            if items:
                self.assertIn(escape(title), home.text)
                self.assertIn(f"{len(items)} commands", home.text)

    def test_every_page_within_budget_and_an_edit(self):
        for render in _walk_pages():
            self.assertLessEqual(len(render.text), PAGE_CEILING)
            self.assertTrue(render.session_edit)
            self.assertEqual(render.session, "menu")

    def test_walk_covers_the_whole_catalog_exactly_once(self):
        seen = []
        for render in _walk_pages():
            for line in render.text.splitlines():
                line = line.strip()
                if line.startswith("/") and " — " in line:
                    seen.append(line.split(" — ")[0])
        expected = [cmd for _e, _t, items
                    in menu_sections("telegram-personal")
                    for cmd, _d in items]
        self.assertEqual(sorted(seen), sorted(expected))
        self.assertEqual(len(seen), len(set(seen)))

    def test_personal_category_topics_all_resolve(self):
        from nomorals.social.chat.personal_commands import PERSONAL_HELP_CATEGORIES
        for _label, callback in PERSONAL_HELP_CATEGORIES:
            if not callback.startswith("list"):
                continue
            topic = callback.partition(" ")[2]
            self.assertIsNotNone(
                menu_session_render(topic,
                                    platform="telegram-personal"),
                callback)

    def test_public_only_hides_owner_commands_and_power_row(self):
        home = menu_session_render(platform="telegram-bot",
                                   public_only=True, home=True)
        flat = [data for row in home.buttons for _l, data in row]
        self.assertNotIn("power status", flat)
        walked = "\n".join(r.text for r in _walk_pages(
            platform="telegram-bot", public_only=True))
        self.assertNotIn("/trial", walked)
        self.assertIn("/wordchain", walked)

    def test_unknown_topic_returns_none(self):
        self.assertIsNone(menu_session_render(
            "nosuchthing", platform="telegram-personal"))

    def test_parse_accepts_the_page_word(self):
        cmd = parse_control("/list games 2")
        self.assertEqual(cmd.kind, "list")
        self.assertEqual(cmd.arg, "games")
        self.assertEqual(cmd.tail, "games 2")
        self.assertEqual(parse_control("/list a b c").kind, "error")


# ── the runtime funnel on real bot-DM shapes ─────────────────────

class _FakeAdapter:
    def __init__(self):
        self.edits = []

    def edit_message(self, chat, message_id, text, *, buttons=None,
                     parse_mode=""):
        self.edits.append((message_id, text, buttons))
        return SendResult(ok=True, platform=chat.platform,
                          message_id=message_id)


class _FakeGateway:
    def __init__(self, adapter, platform="telegram-personal"):
        self.adapters = {platform: adapter}
        self.sends = []
        self._n = 0

    def send(self, platform, chat, text, **kwargs):
        self._n += 1
        self.sends.append(SimpleNamespace(
            platform=platform, chat=chat, text=text, kwargs=kwargs))
        return SendResult(ok=True, platform=platform,
                          message_id=str(100 + self._n))


def _dm_runtime(platform="telegram-personal"):
    rt = PartnerRuntime.__new__(PartnerRuntime)
    db = Database(":memory:")
    db.migrate()
    rt.context = SimpleNamespace(db=db)
    chat = ChatRef(platform=platform, chat_id="42", kind=ChatKind.DM)
    rt._owner_chats = {chat.key}
    adapter = _FakeAdapter()
    rt.gateway = _FakeGateway(adapter, platform)
    rt._typing_for = lambda chat_, text: None
    return rt, adapter, chat


def _msg(chat, text, meta=None):
    return SimpleNamespace(chat=chat, text=text, meta=meta or {},
                           sender="owner")


class BotDmMenuFlowTests(unittest.TestCase):
    def test_menu_post_then_category_tap_edits_the_same_message(self):
        rt, adapter, chat = _dm_runtime()
        opener = _msg(chat, "/menu")
        reply = rt.handle_control("/menu", chat.key, message=opener)
        self.assertEqual(getattr(reply, "session", ""), "menu")
        rt._send_control_reply(opener, reply)
        self.assertEqual(len(rt.gateway.sends), 1)
        menu_mid = "101"  # the fake gateway's first assigned id

        tap = _msg(chat, "/list games",
                   {"tapped_message_id": "101"})
        page = rt.handle_control("/list games", chat.key, message=tap)
        rt._send_control_reply(tap, page)
        self.assertEqual(len(rt.gateway.sends), 1)  # no new message
        self.assertEqual(len(adapter.edits), 1)
        mid, text, buttons = adapter.edits[0]
        self.assertEqual(mid, menu_mid)
        self.assertIn("/wordchain", text)
        self.assertLessEqual(len(text), PAGE_CEILING)
        self.assertTrue(
            any(data == "list games 2" for row in buttons
                for _l, data in row))

        turn = _msg(chat, "/list games 2",
                    {"tapped_message_id": "101"})
        page2 = rt.handle_control("/list games 2", chat.key,
                                  message=turn)
        rt._send_control_reply(turn, page2)
        self.assertEqual(len(rt.gateway.sends), 1)  # still one message
        self.assertEqual(len(adapter.edits), 2)
        self.assertEqual(adapter.edits[1][0], "101")

    def test_all_commands_tap_walks_inside_the_menu_message(self):
        rt, adapter, chat = _dm_runtime()
        opener = _msg(chat, "/menu")
        rt._send_control_reply(
            opener, rt.handle_control("/menu", chat.key,
                                      message=opener))
        tap = _msg(chat, "/list", {"tapped_message_id": "101"})
        first = rt.handle_control("/list", chat.key, message=tap)
        rt._send_control_reply(tap, first)
        self.assertEqual(len(rt.gateway.sends), 1)
        self.assertEqual(len(adapter.edits), 1)
        _mid, text, _buttons = adapter.edits[0]
        # A real catalog page (command lines), not the home index and
        # never the raw wall.
        self.assertIn(" — ", text)
        self.assertLessEqual(len(text), PAGE_CEILING)
        wall = list_catalog()
        self.assertGreater(len(wall), 20000)
        for send in rt.gateway.sends:
            self.assertNotIn(wall[:500], send.text)

    def test_grid_clear_carrier_fires_exactly_once(self):
        rt, _adapter, chat = _dm_runtime()
        rt._clear_menu_grid_once(_msg(chat, "/menu"))
        self.assertEqual(len(rt.gateway.sends), 1)
        carrier = rt.gateway.sends[0]
        self.assertEqual(carrier.kwargs.get("reply_keyboard"), [])
        rt._clear_menu_grid_once(_msg(chat, "/menu"))
        self.assertEqual(len(rt.gateway.sends), 1)  # KV-flagged once

    def test_menu_send_itself_carries_no_reply_keyboard(self):
        rt, _adapter, chat = _dm_runtime()
        opener = _msg(chat, "/menu")
        rt._send_control_reply(
            opener, rt.handle_control("/menu", chat.key,
                                      message=opener))
        self.assertIsNone(
            rt.gateway.sends[0].kwargs.get("reply_keyboard"))

    def test_gated_bot_home_is_public_only(self):
        rt, _adapter, chat = _dm_runtime(platform="telegram-bot")
        reply = rt.handle_control("/menu", chat.key,
                                  message=_msg(chat, "/menu"))
        flat = [data for row in reply.buttons for _l, data in row]
        self.assertNotIn("power status", flat)
        self.assertNotIn("trial", reply.text)

    def test_group_list_stays_the_text_catalog(self):
        rt, _adapter, _chat = _dm_runtime()
        group = ChatRef(platform="telegram-personal", chat_id="99",
                        kind=ChatKind.GROUP)
        rt._owner_chats.add(group.key)  # the owner, in his group
        out = rt.handle_control("/list games", group.key,
                                message=_msg(group, "/list games"))
        self.assertIsInstance(out, str)
        self.assertIn("/wordchain", out)

    def test_whatsapp_list_stays_the_text_catalog(self):
        rt, _adapter, _chat = _dm_runtime()
        wa = ChatRef(platform="whatsapp", chat_id="42",
                     kind=ChatKind.DM)
        rt._owner_chats.add(wa.key)  # the owner's own WhatsApp chat
        out = rt.handle_control("/list games", wa.key,
                                message=_msg(wa, "/list games"))
        self.assertIsInstance(out, str)
        self.assertIn("/wordchain", out)


# ── the adapter remove verb ──────────────────────────────────────

def _capture_send():
    """A real TelegramBotAdapter whose _api records the request
    params instead of touching the network."""
    from nomorals.social.chat.telegram import TelegramBotAdapter

    adapter = TelegramBotAdapter(token="123:abc")
    sent = {}

    def _api(method, **params):
        sent.update(params)
        return {"message_id": 1}

    adapter._api = _api
    return adapter, sent


class BotAdapterRemoveKeyboardTests(unittest.TestCase):
    def test_empty_reply_keyboard_sends_remove(self):
        adapter, sent = _capture_send()
        chat = ChatRef(platform="telegram-bot", chat_id="7",
                       kind=ChatKind.DM)
        adapter.send(chat, "carrier", reply_keyboard=[])
        self.assertEqual(sent["reply_markup"],
                         {"remove_keyboard": True})

    def test_grid_still_renders_when_nonempty(self):
        adapter, sent = _capture_send()
        chat = ChatRef(platform="telegram-bot", chat_id="7",
                       kind=ChatKind.DM)
        adapter.send(chat, "menu", reply_keyboard=[["/status"]])
        markup = sent["reply_markup"]
        self.assertTrue(markup["is_persistent"])
        self.assertEqual(markup["keyboard"], [[{"text": "/status"}]])


if __name__ == "__main__":
    unittest.main()
