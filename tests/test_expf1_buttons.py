"""Expression repair F1 — buttons (2026-11-11).

Pins for the expression audit's button break points, each at its
seam, with the owner's meaningful-buttons doctrine applied: a
button's test asserts the EFFECT of its callback (the tap runs the
real command and the operation happens), never just that a
keyboard exists.

* WhatsApp bare /menu and /start dispatch to the live menu
  session (the button-bearing frame), not the text-only frame —
  which remains only as the no-message fallback;
* the WhatsApp button-form preference persists across restarts;
* Telegram send() derives the KIND keyboards when the producer's
  output_kind rides the send path (gateway + control funnel
  included), on BOTH bot roles, and public commands on the gated
  bot earn the rich personal keyboards while owner-only origins
  do not;
* board renders carry navigation buttons whose taps actually
  move the board;
* send_poll passes reply_markup through when a caller provides
  buttons.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from nomorals.agents.partner.runtime import PartnerRuntime, _control_output_kind
from nomorals.agents.partner.runtime_group import RuntimeGroupMixin
from nomorals.social.chat.base import ChatKind, ChatRef, SendResult
from nomorals.social.chat.control import parse_control
from nomorals.social.chat.gateway import ChatGateway
from nomorals.social.chat.outbox import Outbox
from nomorals.social.chat.telegram import TelegramBotAdapter
from nomorals.social.chat.whatsapp import WhatsAppAdapter
from nomorals.storage.db import Database

# ── shared rigs ─────────────────────────────────────────────────

def _dm_runtime(platform: str):
    """The bot-menu-session rig (test_bot_menu_session's shape):
    a bare PartnerRuntime over an in-memory db with a recording
    fake gateway."""
    rt = PartnerRuntime.__new__(PartnerRuntime)
    db = Database(":memory:")
    db.migrate()
    rt.context = SimpleNamespace(db=db)
    chat = ChatRef(platform=platform, chat_id="42", kind=ChatKind.DM)
    rt._owner_chats = {chat.key}

    class _FakeAdapter:
        def __init__(self):
            self.edits = []

    class _FakeGateway:
        def __init__(self, adapter):
            self.adapters = {platform: adapter}
            self.sends = []
            self._n = 0

        def send(self, platform_, chat_, text, **kwargs):
            self._n += 1
            self.sends.append(SimpleNamespace(
                platform=platform_, chat=chat_, text=text,
                kwargs=kwargs))
            return SendResult(ok=True, platform=platform_,
                              message_id=str(100 + self._n))

    adapter = _FakeAdapter()
    rt.gateway = _FakeGateway(adapter)
    rt._typing_for = lambda chat_, text: None
    return rt, adapter, chat


def _msg(chat, text, meta=None):
    return SimpleNamespace(chat=chat, text=text, meta=meta or {},
                           sender="owner")


def _tg_adapter(tmp: str, *, owner_only: bool):
    calls: list = []

    def fake_api(name, **kw):
        calls.append((name, kw))
        return {"message_id": 7}

    adapter = TelegramBotAdapter(
        token="<redacted>" * 20, media_dir=tmp,
        owner_only=owner_only,
        adapter_name="telegram-personal" if owner_only
        else "telegram-bot")
    adapter._api = fake_api  # type: ignore[method-assign]
    return adapter, calls


def _keyboard_labels(calls) -> list[str]:
    for name, kw in calls:
        if name == "sendMessage" and "reply_markup" in kw:
            markup = kw["reply_markup"]
            return [btn["text"] for row in markup["inline_keyboard"]
                    for btn in row]
    return []


def _keyboard_buttons(calls) -> list[dict]:
    for name, kw in calls:
        if name == "sendMessage" and "reply_markup" in kw:
            markup = kw["reply_markup"]
            return [btn for row in markup["inline_keyboard"]
                    for btn in row]
    return []


TG_CHAT = ChatRef(platform="telegram-bot", chat_id="42",
                  kind=ChatKind.DM)


# ── 1 · WhatsApp /menu + /start reach the session menu ─────────

class WhatsAppMenuDispatchTests(unittest.TestCase):
    def test_bare_menu_on_whatsapp_returns_the_session_frame(self):
        rt, _adapter, chat = _dm_runtime("whatsapp")
        reply = rt.handle_control("/menu", chat.key,
                                  message=_msg(chat, "/menu"))
        # The session render: an ActionableMessage with real
        # buttons — NOT the text-only whatsapp frame.
        self.assertEqual(getattr(reply, "session", ""), "menu")
        self.assertTrue(getattr(reply, "buttons", None))
        self.assertNotIn("text-first", str(reply.text))

    def test_bare_start_on_whatsapp_returns_the_session_frame(self):
        rt, _adapter, chat = _dm_runtime("whatsapp")
        reply = rt.handle_control("/start", chat.key,
                                  message=_msg(chat, "/start"))
        self.assertEqual(getattr(reply, "session", ""), "menu")
        self.assertTrue(getattr(reply, "buttons", None))

    def test_menu_button_tap_actually_moves_the_menu(self):
        # Effect: a category button on the WhatsApp menu home,
        # tapped (re-dispatched as its command), renders that
        # category inside the menu session.
        rt, _adapter, chat = _dm_runtime("whatsapp")
        home = rt.handle_control("/menu", chat.key,
                                 message=_msg(chat, "/menu"))
        flat = [data for row in home.buttons for _l, data in row]
        self.assertIn("list games", flat)
        tap = _msg(chat, "/list games", {"tapped_message_id": "101"})
        page = rt.handle_control("/list games", chat.key, message=tap)
        self.assertEqual(getattr(page, "session", ""), "menu")
        self.assertIn("/wordchain", str(page.text))

    def test_text_frame_remains_only_as_the_no_message_fallback(self):
        rt, _adapter, chat = _dm_runtime("whatsapp")
        reply = rt.handle_control("/menu", chat.key, message=None)
        self.assertIsInstance(reply, str)
        # RE-PINNED (presentation Phase 2, 2026-11-11): the old
        # fallback line claimed "no buttons" — false since F1 made
        # the buttons real, and the task removed it. The fallback
        # now names itself the text form of the menu.
        self.assertIn("the text form of the menu", reply)


# ── 2 · WhatsApp button-form preference persists ────────────────

class WhatsAppFormPersistenceTests(unittest.TestCase):
    def test_owner_pin_survives_a_restart(self):
        import tempfile
        with tempfile.TemporaryDirectory(prefix="nm-wa-") as tmp:
            first = WhatsAppAdapter(outbox_dir=tmp)
            self.assertEqual(first.buttons_form_state()["preference"],
                             "quick")
            first.set_buttons_form_preference("list")
            second = WhatsAppAdapter(outbox_dir=tmp)
            self.assertEqual(
                second.buttons_form_state()["preference"], "list")

    def test_degradation_evidence_survives_a_restart(self):
        import tempfile
        with tempfile.TemporaryDirectory(prefix="nm-wa-") as tmp:
            first = WhatsAppAdapter(outbox_dir=tmp)
            first._note_buttons_form("text", "234801@s.whatsapp.net")
            second = WhatsAppAdapter(outbox_dir=tmp)
            state = second.buttons_form_state()
            self.assertEqual(state["preference"], "list")
            self.assertEqual(state["last_form"], "text")

    def test_unprobed_account_stays_on_the_default(self):
        import tempfile
        with tempfile.TemporaryDirectory(prefix="nm-wa-") as tmp:
            adapter = WhatsAppAdapter(outbox_dir=tmp)
            self.assertEqual(
                adapter.buttons_form_state(),
                {"preference": "quick", "last_form": ""})


# ── 3 · Telegram derivation scope ───────────────────────────────

class TelegramKindKeyboardTests(unittest.TestCase):
    def test_kind_keyboard_on_the_gated_bot(self):
        import tempfile
        with tempfile.TemporaryDirectory(prefix="nm-tg-") as tmp:
            adapter, calls = _tg_adapter(tmp, owner_only=False)
            adapter.send(TG_CHAT, "your 6am job finished",
                         output_kind="scheduled",
                         output_meta={"job": "memory-consolidation"})
            labels = _keyboard_labels(calls)
            self.assertTrue(any("Re-run" in label for label in labels),
                            labels)
            self.assertTrue(any("Recall" in label for label in labels),
                            labels)

    def test_kind_keyboard_on_the_personal_bot(self):
        import tempfile
        with tempfile.TemporaryDirectory(prefix="nm-tg-") as tmp:
            adapter, calls = _tg_adapter(tmp, owner_only=True)
            adapter.send(TG_CHAT, "your 6am job finished",
                         output_kind="scheduled",
                         output_meta={"job": "memory-consolidation"})
            labels = _keyboard_labels(calls)
            self.assertTrue(any("Re-run" in label for label in labels),
                            labels)

    def test_search_kind_keyboard_tap_runs_deep_research(self):
        # Effect (doctrine): the Deep research button's callback,
        # delivered as a tap, produces the synthetic command that
        # runs the deep research — not a status line.
        import tempfile
        with tempfile.TemporaryDirectory(prefix="nm-tg-") as tmp:
            adapter, calls = _tg_adapter(tmp, owner_only=False)
            adapter._last_command["42"] = "/search cats"
            adapter._last_inbound_was_command["42"] = True
            adapter.send(TG_CHAT, "3 results about cats",
                         output_kind="search",
                         output_meta={"query": "cats"})
            buttons = _keyboard_buttons(calls)
            deep = [b for b in buttons
                    if "Deep research" in b["text"]]
            self.assertTrue(deep, buttons)
            received: list = []
            adapter._handle_callback(
                {"id": "q1",
                 "from": {"id": 7, "is_bot": False},
                 "message": {"chat": {"id": 42, "type": "private"}},
                 "data": deep[0]["callback_data"]},
                received.append)
            self.assertEqual(len(received), 1)
            self.assertEqual(received[0].text, "/searchdeep cats")

    def test_public_origin_earns_the_rich_keyboard_on_gated(self):
        import tempfile
        with tempfile.TemporaryDirectory(prefix="nm-tg-") as tmp:
            adapter, calls = _tg_adapter(tmp, owner_only=False)
            adapter._last_command["42"] = "/search cats"
            adapter._last_inbound_was_command["42"] = True
            adapter.send(TG_CHAT, "3 results about cats")
            labels = _keyboard_labels(calls)
            # The personal search keyboard: re-run + deepen + save.
            self.assertTrue(any("Search again" in label for label in labels),
                            labels)
            self.assertTrue(any("Save query" in label for label in labels),
                            labels)

    def test_owner_only_origin_gets_no_rich_keyboard_on_gated(self):
        import tempfile
        with tempfile.TemporaryDirectory(prefix="nm-tg-") as tmp:
            adapter, calls = _tg_adapter(tmp, owner_only=False)
            adapter._last_command["42"] = "/power"
            adapter._last_inbound_was_command["42"] = True
            adapter.send(TG_CHAT, "power mode noted")
            labels = _keyboard_labels(calls)
            self.assertFalse(any("Power Off" in label for label in labels),
                             labels)
            self.assertFalse(any("Power On" in label for label in labels),
                             labels)


class ControlOutputKindTests(unittest.TestCase):
    def test_mapping_from_the_parsed_command(self):
        self.assertEqual(
            _control_output_kind(parse_control("/search cats")),
            ("search", {"query": "cats"}))
        self.assertEqual(
            _control_output_kind(parse_control("/searchdeep dogs")),
            ("search", {"query": "dogs"}))
        self.assertEqual(
            _control_output_kind(parse_control("/game trivia")),
            ("game", {"game": "trivia"}))
        self.assertEqual(
            _control_output_kind(parse_control("/trivia")),
            ("game", {"game": "trivia"}))
        self.assertEqual(
            _control_output_kind(parse_control("/recall")),
            ("memory", None))
        self.assertEqual(
            _control_output_kind(parse_control("/power status")),
            (None, None))
        self.assertEqual(_control_output_kind(None), (None, None))

    def test_funnel_forwards_the_kind_to_the_gateway(self):
        rt, _adapter, chat = _dm_runtime("telegram-bot")
        msg = _msg(chat, "/search cats")
        rt._send_control_reply(msg, "plain results",
                               output_kind="search",
                               output_meta={"query": "cats"})
        self.assertEqual(len(rt.gateway.sends), 1)
        kwargs = rt.gateway.sends[0].kwargs
        self.assertEqual(kwargs.get("output_kind"), "search")
        self.assertEqual(kwargs.get("output_meta"),
                         {"query": "cats"})


class _KindDeclaringAdapter:
    name = "telegram-bot"

    def __init__(self):
        self.calls: list = []

    def send(self, chat, text, *, reply_to="", buttons=None,
             reply_keyboard=None, parse_mode="", fresh=False,
             output_kind=None, output_meta=None):
        self.calls.append({"text": text, "output_kind": output_kind,
                           "output_meta": output_meta})
        return SendResult(ok=True, platform="telegram-bot",
                          message_id="1")


class _LegacyAdapter:
    name = "telegram-bot"

    def __init__(self):
        self.calls: list = []

    def send(self, chat, text, *, reply_to="", buttons=None,
             reply_keyboard=None, parse_mode="", fresh=False):
        self.calls.append({"text": text})
        return SendResult(ok=True, platform="telegram-bot",
                          message_id="1")


class GatewayKindThreadingTests(unittest.TestCase):
    def test_gateway_threads_kind_to_declaring_adapters(self):
        adapter = _KindDeclaringAdapter()
        gateway = ChatGateway(adapters={"telegram-bot": adapter})
        out = gateway.send("telegram-bot", TG_CHAT, "results",
                           output_kind="search",
                           output_meta={"query": "cats"})
        self.assertTrue(out.ok)
        self.assertEqual(adapter.calls[0]["output_kind"], "search")
        self.assertEqual(adapter.calls[0]["output_meta"],
                         {"query": "cats"})

    def test_gateway_never_forces_kind_on_legacy_adapters(self):
        adapter = _LegacyAdapter()
        gateway = ChatGateway(adapters={"telegram-bot": adapter})
        out = gateway.send("telegram-bot", TG_CHAT, "results",
                           output_kind="search",
                           output_meta={"query": "cats"})
        self.assertTrue(out.ok)
        self.assertEqual(adapter.calls[0]["text"], "results")

    def test_durable_row_carries_the_kind_to_the_drain(self):
        db = Database(":memory:")
        db.migrate()
        outbox = Outbox(db)
        job = outbox.submit(
            platform="telegram-bot", chat=TG_CHAT, text="results",
            output_kind="search", output_meta={"query": "cats"})
        self.assertTrue(job)
        seen: list = []

        def _capture(payload):
            seen.append(payload)
            return SendResult(ok=True, platform="telegram-bot",
                              message_id="1"), 1

        outbox.deliver_chat(TG_CHAT.key, "telegram-bot", _capture)
        self.assertTrue(seen)
        self.assertEqual(seen[0].get("output_kind"), "search")
        self.assertEqual(seen[0].get("output_meta"),
                         {"query": "cats"})


# ── 4 · Boards carry buttons whose taps move the board ─────────

class _FakeSurface:
    def __init__(self):
        self.store = SimpleNamespace(
            settings=lambda key: {"boards": {}})

    def render_board(self, board_id, *, group_key="",
                     games_rows=None):
        if board_id in ("group:sparks", "group:gems",
                        "global:sparks"):
            return f"BOARD {board_id}"
        return None


class _BoardRT(RuntimeGroupMixin):
    def __init__(self):
        self._surface = _FakeSurface()

    def _group_surface(self):
        return self._surface

    def _group_key_or_none(self, message):
        return "g1"

    def _board_games_rows(self):
        return None


class BoardButtonTests(unittest.TestCase):
    def setUp(self):
        self.rt = _BoardRT()
        self.msg = SimpleNamespace(
            chat=ChatRef(platform="telegram", chat_id="-1",
                         kind=ChatKind.GROUP))

    def test_board_render_carries_navigation_buttons(self):
        reply = self.rt._control_boards("sparks", self.msg)
        self.assertTrue(hasattr(reply, "buttons"))
        flat = [data for row in reply.buttons for _l, data in row]
        self.assertIn("boards group:gems", flat)     # ▶ next metric
        self.assertIn("boards global:sparks", flat)  # scope hop
        self.assertIn("boards", flat)                # the index

    def test_next_metric_tap_actually_moves_the_board(self):
        # Effect: dispatching the ▶ button's command renders the
        # neighbouring metric's board, not the same one again.
        reply = self.rt._control_boards("sparks", self.msg)
        nxt = [data for row in reply.buttons for _l, data in row
               if data == "boards group:gems"]
        self.assertTrue(nxt)
        moved = self.rt._control_boards(
            nxt[0].partition(" ")[2], self.msg)
        self.assertEqual(moved.text, "BOARD group:gems")
        back = [data for row in moved.buttons
                for _l, data in row if data == "boards group:sparks"]
        self.assertTrue(back)  # ◀ walks back

    def test_index_buttons_open_real_boards(self):
        reply = self.rt._control_boards("", self.msg)
        self.assertTrue(hasattr(reply, "buttons"))
        flat = [data for row in reply.buttons for _l, data in row]
        self.assertIn("boards group:sparks", flat)
        opened = self.rt._control_boards("group:sparks", self.msg)
        self.assertEqual(opened.text, "BOARD group:sparks")

    def test_unknown_board_offers_the_index_and_it_opens(self):
        reply = self.rt._control_boards("bogus:zz", self.msg)
        self.assertTrue(hasattr(reply, "buttons"))
        flat = [data for row in reply.buttons for _l, data in row]
        self.assertIn("boards", flat)
        index = self.rt._control_boards("", self.msg)
        self.assertIn("boards", str(index.text))


# ── 5 · send_poll carries reply_markup ──────────────────────────

class SendPollButtonsTests(unittest.TestCase):
    def test_poll_with_buttons_passes_reply_markup(self):
        import tempfile
        with tempfile.TemporaryDirectory(prefix="nm-tg-") as tmp:
            adapter, calls = _tg_adapter(tmp, owner_only=False)
            out = adapter.send_poll(
                TG_CHAT, "Lunch?", ["a", "b"],
                buttons=[[("📊 Results", "pollres 1")]])
            self.assertTrue(out.ok)
            poll = [kw for name, kw in calls if name == "sendPoll"]
            self.assertTrue(poll)
            markup = poll[0].get("reply_markup")
            self.assertIsNotNone(markup)
            labels = [b["text"] for row in markup["inline_keyboard"]
                      for b in row]
            self.assertEqual(labels, ["📊 Results"])

    def test_poll_without_buttons_stays_markup_free(self):
        import tempfile
        with tempfile.TemporaryDirectory(prefix="nm-tg-") as tmp:
            adapter, calls = _tg_adapter(tmp, owner_only=False)
            out = adapter.send_poll(TG_CHAT, "Lunch?", ["a", "b"])
            self.assertTrue(out.ok)
            poll = [kw for name, kw in calls if name == "sendPoll"]
            self.assertTrue(poll)
            self.assertNotIn("reply_markup", poll[0])


if __name__ == "__main__":
    unittest.main()
