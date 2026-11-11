"""Button-fix lane 2 — the WhatsApp demotion class, tap-tested
BY FORM + PAYLOAD SURVIVAL (audit §5 harness, send side).

Lane 1 fixed the draft card by making the WhatsApp adapter pick
the bridges' list form when a keyboard carries more buttons than
the native quick_reply cap (3) — in ``_send_text_outcome`` and
``edit_message``.  This suite pins the GENERALISATION (audit §6
lane 2, findings S4/S5):

* the form decision is ONE adapter policy
  (``WhatsAppAdapter._buttons_list_form``): the persisted "list"
  preference wins, any >3-button keyboard lists, ≤3 keyboards
  keep the native form byte-identically;
* every delivery door decides through it — live text sends, the
  outbox flush, keyboard edits, and MEDIA sends.  A ListMessage
  cannot head media on either bridge (their send_media paths
  never read ``buttons_style``), so a keyboard that cannot ride
  the media's own interactive header — too wide, list-pinned, or
  on a headless form (voice note, video note, view-once) — goes
  out as a follow-up text message carrying the caption + the
  full keyboard, where every row is a real tappable button;
* keyboards are HARVESTED from the real emitters (the session
  menu composition, the draft review card, the poll-card spec
  builder, the adapter's own derivation) and the assertion is
  payload SET equality into tappable rows — never mere text
  presence, which is exactly how the demotion hid.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from nomorals.social.chat.base import ChatKind, ChatRef, MediaRef
from nomorals.social.chat.control import menu_session_render
from nomorals.social.chat.expression import buttons_from_spec
from nomorals.social.chat.tgbot_buttons import grid_layout
from nomorals.social.chat.whatsapp import (
    UNCONFIRMED_SEND_ERROR,
    WhatsAppAdapter,
)
from nomorals.social.drafts import DraftQueue, review_buttons


def _wa_chat() -> ChatRef:
    return ChatRef(platform="whatsapp",
                   chat_id="234801@s.whatsapp.net", kind=ChatKind.DM)


def _wa_adapter(tmp: str, responder) -> tuple[WhatsAppAdapter, list]:
    """Lane-1 rig: a connected adapter whose bridge is a recorder."""
    adapter = WhatsAppAdapter(outbox_dir=str(Path(tmp) / "outbox"))
    adapter.connected.set()
    calls: list = []

    def fake_send_cmd(cmd, timeout=25.0):
        calls.append(dict(cmd))
        return responder(cmd)

    adapter._send_cmd = fake_send_cmd  # type: ignore[method-assign]
    return adapter, calls


def _ok_responder(cmd) -> dict[str, Any]:
    return {"ok": True, "message_id": "M1"}


def _flat(keyboard) -> list[str]:
    return [str(data) for row in keyboard if row
            for _label, data in row]


def _sent_payloads(calls, cmd_name: str = "send") -> list[str]:
    """Payloads on the FIRST button-carrying bridge request of
    the given kind — split sends put the keyboard on chunk one."""
    for call in calls:
        if call.get("cmd") == cmd_name and call.get("buttons"):
            return [str(data) for row in call["buttons"]
                    for _label, data in row]
    return []


def _first(calls, cmd_name: str):
    for call in calls:
        if call.get("cmd") == cmd_name:
            return call
    return None


# ── harvests: the real emitters ────────────────────────────────────


def _menu_home():
    """The S4 surface itself: the WhatsApp session-menu home,
    media head + its real section keyboard."""
    render = menu_session_render(
        platform="whatsapp", home=True,
        media={"path": "/tmp/menu-banner.png", "kind": "image"})
    assert render is not None
    return render


def _poll_keyboard(options: int) -> list:
    """The house poll card's builder + grammar (runtime_group
    ``_poll_card``): one row per option, ``vote <id> <i>``."""
    spec = [[("poll", f"{i}. option {i}", f"vote poll42 {i}")]
            for i in range(1, options + 1)]
    return buttons_from_spec(spec)


class HarvestTests(unittest.TestCase):
    def test_menu_home_keyboard_is_wide_and_real(self) -> None:
        render = _menu_home()
        flat = _flat(render.buttons)
        # The menu home carries a section row per catalog section
        # (plus the owner Power row) — far past the native cap,
        # which is why it was the demotion class's worst surface.
        self.assertGreater(len(flat), 3)
        self.assertTrue(all(
            p.startswith("list ") or p == "power status"
            for p in flat))
        self.assertEqual(render.media,
                         {"path": "/tmp/menu-banner.png",
                          "kind": "image"})

    def test_menu_section_page_keyboard_harvests(self) -> None:
        home = _menu_home()
        key = next(p.split(None, 1)[1] for p in _flat(home.buttons)
                   if p.startswith("list "))
        page = menu_session_render(
            key, 1, platform="whatsapp", continuation=True)
        assert page is not None
        self.assertTrue(_flat(page.buttons))


class PolicySeamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-btnfix2-pol-")
        self.adapter = WhatsAppAdapter(
            outbox_dir=str(Path(self.tmp.name) / "outbox"))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_shape_rule(self) -> None:
        one = [[("A", "a")]]
        three = [[("A", "a"), ("B", "b"), ("C", "c")]]
        four = [three[0] + [("D", "d")]]
        self.assertFalse(self.adapter._buttons_list_form(None))
        self.assertFalse(self.adapter._buttons_list_form([]))
        self.assertFalse(self.adapter._buttons_list_form(one))
        self.assertFalse(self.adapter._buttons_list_form(three))
        self.assertTrue(self.adapter._buttons_list_form(four))

    def test_persisted_list_preference_wins_at_any_width(self) -> None:
        self.adapter.set_buttons_form_preference("list")
        self.assertTrue(self.adapter._buttons_list_form(
            [[("A", "a")]]))
        self.adapter.set_buttons_form_preference("quick")
        self.assertFalse(self.adapter._buttons_list_form(
            [[("A", "a")]]))


class TextFormTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-btnfix2-txt-")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_menu_keyboard_goes_out_as_list_with_every_row(self):
        adapter, calls = _wa_adapter(self.tmp.name, _ok_responder)
        render = _menu_home()
        out = adapter.send(_wa_chat(), render.text,
                           buttons=render.buttons)
        self.assertTrue(out.ok)
        send_cmd = _first(calls, "send")
        assert send_cmd is not None
        self.assertEqual(send_cmd.get("buttons_style"), "list")
        self.assertEqual(set(_sent_payloads(calls)),
                         set(_flat(render.buttons)))

    def test_menu_section_page_rides_the_same_policy(self) -> None:
        adapter, calls = _wa_adapter(self.tmp.name, _ok_responder)
        home = _menu_home()
        key = next(p.split(None, 1)[1] for p in _flat(home.buttons)
                   if p.startswith("list "))
        page = menu_session_render(
            key, 1, platform="whatsapp", continuation=True)
        assert page is not None
        adapter.send(_wa_chat(), page.text, buttons=page.buttons)
        send_cmd = _first(calls, "send")
        assert send_cmd is not None
        if len(_flat(page.buttons)) > 3:
            self.assertEqual(send_cmd.get("buttons_style"), "list")
        else:
            self.assertNotIn("buttons_style", send_cmd)
        self.assertEqual(set(_sent_payloads(calls)),
                         set(_flat(page.buttons)))

    def test_five_option_poll_card_lists(self) -> None:
        adapter, calls = _wa_adapter(self.tmp.name, _ok_responder)
        keyboard = _poll_keyboard(5)
        adapter.send(_wa_chat(), "🗳️ Poll", buttons=keyboard)
        send_cmd = _first(calls, "send")
        assert send_cmd is not None
        self.assertEqual(send_cmd.get("buttons_style"), "list")
        self.assertEqual(set(_sent_payloads(calls)),
                         {f"vote poll42 {i}" for i in range(1, 6)})

    def test_three_option_poll_card_keeps_native_form(self) -> None:
        adapter, calls = _wa_adapter(self.tmp.name, _ok_responder)
        keyboard = _poll_keyboard(3)
        adapter.send(_wa_chat(), "🗳️ Poll", buttons=keyboard)
        send_cmd = _first(calls, "send")
        assert send_cmd is not None
        self.assertNotIn("buttons_style", send_cmd)
        self.assertEqual(set(_sent_payloads(calls)),
                         {f"vote poll42 {i}" for i in range(1, 4)})

    def test_pair_toggle_rows_list_with_every_payload(self) -> None:
        # The /pair home emitter's grammar (runtime_provision):
        # one row per toggle per account — always far past 3.
        adapter, calls = _wa_adapter(self.tmp.name, _ok_responder)
        names = ("view_once", "status_viewing", "status_react",
                 "antidelete", "status_saving", "antitag")
        keyboard = [[(f"🔘 {name}", f"pair toggle acct1 {name} off")]
                    for name in names]
        adapter.send(_wa_chat(), "pair home", buttons=keyboard)
        send_cmd = _first(calls, "send")
        assert send_cmd is not None
        self.assertEqual(send_cmd.get("buttons_style"), "list")
        self.assertEqual(set(_sent_payloads(calls)),
                         set(_flat(keyboard)))

    def test_derived_keyboard_rides_the_same_seam(self) -> None:
        # A reply whose producer declared no keyboard: the
        # adapter derives one (personal/adaptive) and the derived
        # keyboard goes out under the same policy.
        adapter, calls = _wa_adapter(self.tmp.name, _ok_responder)
        chat = _wa_chat()
        adapter._last_inbound_was_command[chat.chat_id] = True
        adapter._last_command[chat.chat_id] = "searchdeep cats"
        derived = adapter._derive_buttons(chat, "results for cats")
        self.assertTrue(derived)
        adapter.send(chat, "results for cats")
        send_cmd = _first(calls, "send")
        assert send_cmd is not None
        if len(_flat(derived)) > 3:
            self.assertEqual(send_cmd.get("buttons_style"), "list")
        else:
            self.assertNotIn("buttons_style", send_cmd)
        self.assertEqual(set(_sent_payloads(calls)),
                         set(_flat(derived)))

    def test_edit_message_uses_the_policy(self) -> None:
        adapter, calls = _wa_adapter(self.tmp.name, _ok_responder)
        wide = [[(f"B{i}", f"cmd {i}")] for i in range(5)]
        verdict = adapter.edit_message(_wa_chat(), "M9", "edited",
                                       buttons=wide)
        self.assertTrue(verdict.get("ok"))
        edit_cmd = _first(calls, "edit")
        assert edit_cmd is not None
        self.assertEqual(edit_cmd.get("buttons_style"), "list")
        calls.clear()
        narrow = [[("A", "a"), ("B", "b")]]
        adapter.edit_message(_wa_chat(), "M9", "edited",
                             buttons=narrow)
        edit_cmd = _first(calls, "edit")
        assert edit_cmd is not None
        self.assertNotIn("buttons_style", edit_cmd)


# ── media deliveries: the S4 class ────────────────────────────────


class MediaFormTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-btnfix2-med-")
        queue = DraftQueue(db_path=Path(self.tmp.name) / "d.db")
        try:
            self.draft = queue.create_draft("media card draft", ["x"])
            queue.propose(self.draft.id)
        finally:
            queue.close()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _media(self, kind: str = "image", **kw) -> MediaRef:
        path = Path(self.tmp.name) / "asset.bin"
        path.write_bytes(b"asset")
        return MediaRef(path=str(path), kind=kind, **kw)

    def test_menu_on_its_banner_rides_a_followup_list(self) -> None:
        # S4 verbatim: the menu home is a media-head send; its
        # ~section-count keyboard must arrive FULLY tappable.
        adapter, calls = _wa_adapter(self.tmp.name, _ok_responder)
        render = _menu_home()
        out = adapter.send_media(
            _wa_chat(), self._media("image"),
            caption=render.text, buttons=render.buttons)
        self.assertTrue(out.ok)
        media_cmd = _first(calls, "send_media")
        assert media_cmd is not None
        # The media goes out bare — no keyboard is asked of a
        # form that would demote all but three rows to caption
        # lines, and the caption rides the follow-up instead.
        self.assertNotIn("buttons", media_cmd)
        self.assertEqual(media_cmd.get("caption"), "")
        follow = _first(calls, "send")
        assert follow is not None
        self.assertEqual(follow.get("buttons_style"), "list")
        self.assertEqual(set(_sent_payloads(calls)),
                         set(_flat(render.buttons)))
        # The menu text itself is the follow-up's body (its
        # masthead renders in fancy glyphs; the plain field
        # lines survive the WhatsApp formatting conversion).
        self.assertIn("total commands",
                      str(follow.get("text") or "").lower())

    def test_draft_card_on_a_photo_rides_a_followup_list(self):
        adapter, calls = _wa_adapter(self.tmp.name, _ok_responder)
        keyboard = grid_layout([review_buttons(self.draft)])
        out = adapter.send_media(
            _wa_chat(), self._media("image"),
            caption="📝 draft review", buttons=keyboard)
        self.assertTrue(out.ok)
        self.assertNotIn("buttons", _first(calls, "send_media"))
        follow = _first(calls, "send")
        assert follow is not None
        self.assertEqual(follow.get("buttons_style"), "list")
        payloads = set(_sent_payloads(calls))
        self.assertEqual(payloads, set(_flat(keyboard)))
        # The two buttons WhatsApp used to print as dead text:
        self.assertIn(f"draft:discard:{self.draft.id}", payloads)
        self.assertIn(f"draft:edit:{self.draft.id}", payloads)

    def test_three_buttons_on_a_photo_stay_on_the_media(self):
        # The on-media interactive header is the right form when
        # the whole keyboard fits the native set — wire-identical
        # to the pre-lane shape: buttons + caption on send_media,
        # no follow-up message at all.
        adapter, calls = _wa_adapter(self.tmp.name, _ok_responder)
        keyboard = _poll_keyboard(3)
        out = adapter.send_media(
            _wa_chat(), self._media("image"),
            caption="🗳️ Poll", buttons=keyboard)
        self.assertTrue(out.ok)
        media_cmd = _first(calls, "send_media")
        assert media_cmd is not None
        self.assertEqual(media_cmd.get("caption"), "🗳️ Poll")
        self.assertEqual(
            {str(d) for row in media_cmd["buttons"]
             for _l, d in row},
            {f"vote poll42 {i}" for i in range(1, 4)})
        self.assertIsNone(_first(calls, "send"))

    def test_list_preference_moves_even_a_small_media_keyboard(self):
        adapter, calls = _wa_adapter(self.tmp.name, _ok_responder)
        adapter.set_buttons_form_preference("list")
        out = adapter.send_media(
            _wa_chat(), self._media("image"),
            caption="🗳️ Poll", buttons=_poll_keyboard(2))
        self.assertTrue(out.ok)
        self.assertNotIn("buttons", _first(calls, "send_media"))
        follow = _first(calls, "send")
        assert follow is not None
        self.assertEqual(follow.get("buttons_style"), "list")
        self.assertEqual(set(_sent_payloads(calls)),
                         {"vote poll42 1", "vote poll42 2"})

    def test_voice_note_keyboard_rides_a_native_followup(self):
        # A voice note has no interactive head on either bridge:
        # today its buttons printed as dead text lines behind it.
        adapter, calls = _wa_adapter(self.tmp.name, _ok_responder)
        keyboard = [[("▶️ Play", "play pick t1 1"),
                     ("⏭ Next", "play next")]]
        out = adapter.send_media(
            _wa_chat(), self._media("voice"),
            caption="your voice note", buttons=keyboard)
        self.assertTrue(out.ok)
        media_cmd = _first(calls, "send_media")
        assert media_cmd is not None
        self.assertTrue(media_cmd.get("ptt"))
        self.assertNotIn("buttons", media_cmd)
        follow = _first(calls, "send")
        assert follow is not None
        self.assertNotIn("buttons_style", follow)  # ≤3 → native
        self.assertEqual(set(_sent_payloads(calls)),
                         {"play pick t1 1", "play next"})
        self.assertIn("your voice note",
                      str(follow.get("text") or ""))

    def test_view_once_keyboard_rides_a_followup_list(self) -> None:
        # The view-once wrap must stay a view-once; its keyboard
        # is a separate, fully tappable message.
        adapter, calls = _wa_adapter(self.tmp.name, _ok_responder)
        keyboard = [[(f"B{i}", f"cmd {i}")] for i in range(4)]
        out = adapter.send_media(
            _wa_chat(), self._media("image", view_once=True),
            caption="one look", buttons=keyboard)
        self.assertTrue(out.ok)
        media_cmd = _first(calls, "send_media")
        assert media_cmd is not None
        self.assertTrue(media_cmd.get("view_once"))
        self.assertNotIn("buttons", media_cmd)
        follow = _first(calls, "send")
        assert follow is not None
        self.assertEqual(follow.get("buttons_style"), "list")
        self.assertEqual(set(_sent_payloads(calls)),
                         set(_flat(keyboard)))

    def test_video_note_keyboard_rides_a_native_followup(self):
        adapter, calls = _wa_adapter(self.tmp.name, _ok_responder)
        keyboard = _poll_keyboard(3)
        out = adapter.send_media(
            _wa_chat(), self._media("video_note"),
            caption="🗳️ Poll", buttons=keyboard)
        self.assertTrue(out.ok)
        media_cmd = _first(calls, "send_media")
        assert media_cmd is not None
        self.assertTrue(media_cmd.get("ptv"))
        self.assertNotIn("buttons", media_cmd)
        follow = _first(calls, "send")
        assert follow is not None
        self.assertNotIn("buttons_style", follow)
        self.assertEqual(set(_sent_payloads(calls)),
                         {f"vote poll42 {i}" for i in range(1, 4)})
        self.assertIn("Poll", str(follow.get("text") or ""))

    def test_captionless_media_keyboard_still_lands(self) -> None:
        adapter, calls = _wa_adapter(self.tmp.name, _ok_responder)
        keyboard = [[("A", "a"), ("B", "b")]]
        out = adapter.send_media(
            _wa_chat(), self._media("voice"), buttons=keyboard)
        self.assertTrue(out.ok)
        follow = _first(calls, "send")
        assert follow is not None
        self.assertTrue(str(follow.get("text") or "").strip())
        self.assertEqual(set(_sent_payloads(calls)), {"a", "b"})

    def test_unconfirmed_media_never_sprouts_a_followup(self):
        # The send-once guard's contract: an ack-lost dispatch is
        # held unconfirmed — a follow-up would be an unguarded
        # second message the reconciliation cannot account for.
        def responder(cmd):
            if cmd.get("cmd") == "send_media":
                return {"ok": False, "ambiguous": True,
                        "error": "bridge response timeout"}
            return {"ok": True, "message_id": "M1"}

        adapter, calls = _wa_adapter(self.tmp.name, responder)
        out = adapter.send_media(
            _wa_chat(), self._media("image"),
            caption="📝 draft review",
            buttons=grid_layout([review_buttons(self.draft)]))
        self.assertFalse(out.ok)
        self.assertEqual(out.error, UNCONFIRMED_SEND_ERROR)
        self.assertIsNone(_first(calls, "send"))

    def test_failed_followup_is_journaled_never_fatal(self) -> None:
        # The media landed; the keyboard follow-up was refused.
        # The send verdict stays the media's (a failure verdict
        # would make the gateway retry and duplicate the media) —
        # and the miss is in the failure ledger by name.
        def responder(cmd):
            if cmd.get("cmd") == "send":
                return {"ok": False, "error": "bridge rejected send"}
            return {"ok": True, "message_id": "M1"}

        adapter, calls = _wa_adapter(self.tmp.name, responder)
        failures: list = []
        with mock.patch(
                "nomorals.social.chat.whatsapp.record_failure",
                lambda site, exc: failures.append((site, exc))):
            out = adapter.send_media(
                _wa_chat(), self._media("image"),
                caption="📝 draft review",
                buttons=grid_layout([review_buttons(self.draft)]))
        self.assertTrue(out.ok)
        sites = [site for site, _exc in failures]
        self.assertIn("whatsapp.WhatsAppAdapter.send_media.buttons",
                      sites)

    def test_queued_media_flushes_with_a_tappable_keyboard(self):
        adapter = WhatsAppAdapter(
            outbox_dir=str(Path(self.tmp.name) / "outbox"))
        render = _menu_home()
        out = adapter.send_media(
            _wa_chat(), self._media("image"),
            caption=render.text, buttons=render.buttons)
        self.assertFalse(out.ok)  # bridge down — queued, not sent
        entries = adapter.outbox.pending()
        self.assertTrue(entries)
        calls: list = []
        adapter.connected.set()
        adapter._send_cmd = (  # type: ignore[method-assign]
            lambda cmd, timeout=25.0: (calls.append(dict(cmd)),
                                       {"ok": True,
                                        "message_id": "M9"})[1])
        self.assertTrue(adapter._deliver_entry(dict(entries[0])))
        self.assertNotIn("buttons", _first(calls, "send_media"))
        follow = _first(calls, "send")
        assert follow is not None
        self.assertEqual(follow.get("buttons_style"), "list")
        self.assertEqual(set(_sent_payloads(calls)),
                         set(_flat(render.buttons)))


if __name__ == "__main__":
    unittest.main()
