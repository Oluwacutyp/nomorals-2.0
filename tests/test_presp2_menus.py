"""Presentation Phase 2 — designed menus everywhere (2026-11-11).

Pins:
* the session home is the ONE designed composition — masthead of
  live fields, categories clustered by intent with divider rules
  (flat) / cluster boxes with the XMD row rhythm (WhatsApp
  framed) — with the stamp policy unchanged and no 📌 fallback
  anywhere;
* /start and /menu are the same front door on Telegram (the
  gated/public split preserved);
* /pair home is a Card (the prose tail is gone) whose toggle
  buttons flip real toggles;
* the /boards index and the /pack listing compose as Cards;
* a long PLAIN control reply rides the Doc-16 output route —
  no (i/n) chunk labels — while markup-bearing replies keep the
  gateway path.
"""

from __future__ import annotations

import re
import unittest
from types import SimpleNamespace

from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.social.chat.base import ChatKind, ChatRef, SendResult
from nomorals.social.chat.control import (
    menu_home_text,
    menu_sections,
    menu_session_render,
)
from nomorals.social.chat.typography import DIVIDER
from nomorals.storage.db import Database

# ── the designed session home ────────────────────────────────────

class SessionHomeCompositionTests(unittest.TestCase):
    def test_telegram_home_carries_masthead_clusters_dividers(self):
        home = menu_session_render(platform="telegram-personal",
                                   home=True)
        text = home.text
        # Masthead: live fields, owner form.
        self.assertIn("👋", text)
        self.assertIn("👑 owner: peacethefirst", text)
        self.assertIn("⚡ status: Online", text)
        self.assertIn("📚 total commands:", text)
        self.assertIn("🔐 mode: owner-gated", text)
        # Cluster heads + divider rules between clusters.
        for cluster in ("talk to me", "play", "know", "run"):
            self.assertIn(cluster, text)
        self.assertIn(DIVIDER, text)
        # Every section still listed with its count.
        from nomorals.social.chat.style import escape

        for _emoji, title, items in menu_sections("telegram"):
            if items:
                self.assertIn(escape(title), text)
                self.assertIn(f"{len(items)} commands", text)
        # Stamp policy unchanged; no pin fallback anywhere.
        self.assertIn("Powered by peacethefirst", text)
        self.assertNotIn("📌", text)

    def test_public_home_omits_owner_field_and_names_its_mode(self):
        home = menu_session_render(platform="telegram-bot",
                                   public_only=True, home=True)
        self.assertNotIn("👑 owner:", home.text)
        self.assertIn("🔐 mode: public", home.text)
        self.assertNotIn("📌", home.text)

    def test_whatsapp_home_is_the_xmd_frame(self):
        home = menu_session_render(platform="whatsapp", home=True)
        text = home.text
        self.assertIn("╭━━〔", text)          # framed boxes
        self.assertIn("╰━━━━━━━━━━━━┈⊷", text)
        self.assertIn("┃ ✫ 👋", text)         # masthead fields ride ✫
        self.assertIn("┃ ✫✧│ 🎬 media", text)  # rows ride ✫✧│
        self.assertIn("ᴛᴀʟᴋ ᴛᴏ ᴍᴇ", text)     # cluster titles small-capped
        self.assertNotIn("📌", text)
        self.assertIn("_Powered by peacethefirst_", text)

    def test_no_section_falls_back_to_the_pin_emoji(self):
        for platform in ("telegram", "telegram-bot", "whatsapp"):
            for emoji, _title, _items in menu_sections(platform):
                self.assertNotEqual(emoji, "📌", (platform, _title))

    def test_text_form_is_the_same_composition(self):
        text = menu_home_text("telegram", public_only=True)
        self.assertIn("📚 total commands:", text)
        self.assertIn(DIVIDER, text)
        self.assertIn("tap a button below", text)
        self.assertNotIn("📌", text)


# ── /start == /menu on Telegram ──────────────────────────────────

class _FakeGateway:
    def __init__(self):
        self.adapters = {}
        self.sends = []

    def send(self, platform, chat, text, **kwargs):
        self.sends.append(SimpleNamespace(
            platform=platform, chat=chat, text=text, kwargs=kwargs))
        return SendResult(ok=True, platform=platform,
                          message_id="1")


def _gated_runtime():
    rt = PartnerRuntime.__new__(PartnerRuntime)
    db = Database(":memory:")
    db.migrate()
    rt.context = SimpleNamespace(db=db)
    rt._owner_chats = set()  # the caller is a stranger
    rt.gateway = _FakeGateway()
    rt._typing_for = lambda chat_, text: None
    return rt


class StartMenuUnificationTests(unittest.TestCase):
    def test_start_and_menu_render_the_same_home_for_a_stranger(self):
        rt = _gated_runtime()
        chat = ChatRef(platform="telegram-bot", chat_id="999",
                       kind=ChatKind.DM)
        msg = SimpleNamespace(chat=chat, text="/start", meta={},
                              sender="stranger")
        start = rt.handle_control("/start", chat.key, message=msg)
        menu = rt.handle_control("/menu", chat.key, message=msg)
        self.assertEqual(getattr(start, "session", ""), "menu")
        self.assertEqual(start.text, menu.text)
        self.assertEqual(start.buttons, menu.buttons)
        # The gated/public split is intact: no Power row, no
        # owner-only content in the composed public home.
        flat = [data for row in start.buttons for _l, data in row]
        self.assertNotIn("power status", flat)
        self.assertNotIn("/trial", start.text)


# ── /pair home: Card + toggle buttons ────────────────────────────

class _FakeRec:
    def __init__(self, rid, platform, toggles):
        self.id = rid
        self.platform = platform
        self.status = "linked"
        self.primary = True
        self._toggles = toggles

    def toggle(self, name):
        return self._toggles.get(name, False)


class _FakeRegistry:
    def __init__(self, recs):
        self._recs = {r.id: r for r in recs}

    def get(self, rid):
        return self._recs.get(rid)

    def all(self):
        return list(self._recs.values())


class _FakeSupervisor:
    def __init__(self, registry):
        self.registry = registry

    def counts(self):
        return {"registered": len(self.registry.all()),
                "connected": len(self.registry.all()), "dropped": 0}

    def status_all(self):
        return [{"id": r.id, "label": "", "status": "linked",
                 "process": "go:1", "platform": r.platform,
                 "last_error": ""} for r in self.registry.all()]


def _pair_runtime(rec, chat_settings=None):
    from nomorals.agents.partner.runtime_provision import RuntimeProvisionMixin

    registry = _FakeRegistry([rec])
    sup = _FakeSupervisor(registry)
    rt = object.__new__(RuntimeProvisionMixin)
    rt._pairing_supervisor = lambda: sup
    rt._pairing_personal = lambda: {"pairer": None}
    rt._pair_toggle_live = lambda rec_, svc=None, sup=None: {}
    rt.context = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir="/tmp"))
    return rt


class PairHomeCardTests(unittest.TestCase):
    def test_home_is_a_card_and_the_prose_tail_is_gone(self):
        rec = _FakeRec("telegram:123456", "telegram",
                       {"view_once": True})
        rt = _pair_runtime(rec)
        out = rt._control_pair("", chat_key="telegram-personal:1")
        text = out.text if hasattr(out, "text") else out
        self.assertIn("paired accounts", text)
        self.assertIn("Registered: 1 · Connected: 1 · Dropped: 0",
                      text)
        self.assertIn("**personal surfaces**", text)
        self.assertIn("/statusviewers — shows who viewed it", text)
        self.assertIn("never on the public bot", text)
        # The old single-paragraph enumeration is gone.
        self.assertNotIn("personal surfaces on these accounts:",
                         text)

    def test_toggle_buttons_flip_real_toggles(self):
        rec = _FakeRec("telegram:123456", "telegram",
                       {"view_once": True})
        rt = _pair_runtime(rec)
        out = rt._control_pair("", chat_key="telegram-personal:1")
        self.assertTrue(hasattr(out, "buttons"))
        flat = {data: label for row in out.buttons
                for label, data in row}
        # view_once is ON, so its button turns it OFF — and the
        # label carries the live state.
        self.assertEqual(flat["pair toggle telegram:123456 "
                              "view_once off"], "view_once ✓")
        self.assertEqual(flat["pair toggle telegram:123456 "
                              "always_online on"], "always_online ✗")


# ── /boards index + /pack listing are Cards ──────────────────────

class _FakeBoardSurface:
    def __init__(self):
        self.store = SimpleNamespace(
            settings=lambda key: {"boards": {}})

    def render_board(self, board_id, *, group_key="",
                     games_rows=None):
        return None


class BoardsIndexCardTests(unittest.TestCase):
    def test_index_composes_as_a_card_with_working_buttons(self):
        from nomorals.agents.partner.runtime_group import RuntimeGroupMixin

        class _RT(RuntimeGroupMixin):
            def __init__(self):
                self._surface = _FakeBoardSurface()

            def _group_surface(self):
                return self._surface

            def _group_key_or_none(self, message):
                return "g1"

            def _board_games_rows(self):
                return None

        rt = _RT()
        msg = SimpleNamespace(
            chat=ChatRef(platform="telegram", chat_id="-1",
                         kind=ChatKind.GROUP))
        reply = rt._control_boards("", msg)
        self.assertIn("🏆 <b>boards</b>", reply.text)
        self.assertIn("<b>scopes</b>", reply.text)
        self.assertIn("this group: sparks", reply.text)
        self.assertIn("platform:&lt;platform&gt;:&lt;metric&gt;",
                      reply.text)
        flat = [data for row in reply.buttons for _l, data in row]
        self.assertIn("boards group:sparks", flat)


class PackListingCardTests(unittest.TestCase):
    def test_listing_card_carries_rows_and_grammar(self):
        from nomorals.social.chat import packs
        from nomorals.social.chat.typography import card_text

        class _Store:
            def is_enabled(self, pid):
                return pid != "makers"

        text = card_text(packs.listing_card(_Store()))
        self.assertIn("**experience packs**", text)
        self.assertIn("**installed**", text)
        for pack in packs.PACKS:
            self.assertIn(pack.title, text)
        self.assertIn("open: /pack open <id>", text)
        # The plain-lines form keeps its exact shape for its
        # other callers.
        lines = packs.listing_lines(_Store())
        self.assertEqual(lines[0], "🧩 experience packs — installed:")
        self.assertEqual(lines[-1], packs._LISTING_FOOTER)


# ── long control replies ride the output route ───────────────────

class _RecordingGateway:
    def __init__(self):
        self.adapters = {}
        self.sends = []

    def send(self, platform, chat, text, **kwargs):
        self.sends.append(SimpleNamespace(
            platform=platform, chat=chat, text=text, kwargs=kwargs))
        return SendResult(ok=True, platform=platform,
                          message_id=str(len(self.sends)))


def _control_runtime(tmp):
    rt = PartnerRuntime.__new__(PartnerRuntime)
    db = Database(":memory:")
    db.migrate()
    rt.context = SimpleNamespace(
        db=db, settings=SimpleNamespace(workspace_dir=tmp))
    rt._owner_chats = set()
    rt.gateway = _RecordingGateway()
    rt._typing_for = lambda chat_, text: None
    return rt


def _tmp_root():
    import os
    import tempfile

    return tempfile.mkdtemp(
        prefix="nm-presp2-", dir=os.environ.get("NM_TEST_TMPDIR")
        or None)


class LongControlReplyRouteTests(unittest.TestCase):
    def test_long_plain_reply_rides_the_router_without_labels(self):
        rt = _control_runtime(_tmp_root())
        chat = ChatRef(platform="telegram-personal",
                       chat_id="42", kind=ChatKind.DM)
        msg = SimpleNamespace(chat=chat, text="/status",
                              meta={}, sender="owner")
        body = ("status line worth reading\n" * 220).strip()
        self.assertGreater(len(body), 4096)
        result = rt._send_control_reply(msg, body)
        self.assertTrue(result.ok)
        self.assertGreaterEqual(len(rt.gateway.sends), 1)
        for send in rt.gateway.sends:
            self.assertIsNone(
                re.search(r"\(\d+/\d+\)", send.text),
                send.text[:80])
        # The routed delivery is NOT the raw wall in one send.
        self.assertNotIn(body, [s.text for s in rt.gateway.sends])

    def test_short_reply_keeps_the_gateway_path(self):
        rt = _control_runtime(_tmp_root())
        chat = ChatRef(platform="telegram-personal",
                       chat_id="42", kind=ChatKind.DM)
        msg = SimpleNamespace(chat=chat, text="/ping",
                              meta={}, sender="owner")
        result = rt._send_control_reply(msg, "pong")
        self.assertTrue(result.ok)
        self.assertEqual(len(rt.gateway.sends), 1)
        self.assertIn("pong", rt.gateway.sends[0].text)

    def test_markup_reply_is_not_routed(self):
        rt = _control_runtime(_tmp_root())
        chat = ChatRef(platform="telegram-personal",
                       chat_id="42", kind=ChatKind.DM)
        msg = SimpleNamespace(chat=chat, text="/boards",
                              meta={}, sender="owner")
        body = "<b>head</b> " + ("x" * 5000)
        result = rt._send_control_reply(msg, body)
        self.assertTrue(result.ok)
        # The gateway path owns markup (its splitter is
        # markup-balanced): one send carrying the full text.
        self.assertEqual(len(rt.gateway.sends), 1)
        self.assertIn(body, rt.gateway.sends[0].text)


if __name__ == "__main__":
    unittest.main()
