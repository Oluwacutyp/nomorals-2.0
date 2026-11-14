"""Card delivery format (owner screenshot): the first send is
formatted, the edit lands as plain text — and the stamp is basic.

The card render is house dialect; the gateway converts it to
Telegram HTML with parse_mode="HTML" on the SEND path, but the
edit path calls the adapter directly and never carries a parse
mode — so the first edit strips the code block, bold and italic.
The fix threads the render's parse mode through the host into
both the send and the edit, so the edit formats exactly like the
send it replaces.

The stamp is plain italic; the brandmark policy (Montserrat /
Inter / Poppins, weight 700) is a strong face — for chat text
that is the Unicode sans-bold alphabet.
"""
from __future__ import annotations

import unittest

from nomorals.games.engine import GameEngine
from nomorals.games.games.base import MultiGame
from nomorals.games.players import Player
from nomorals.social.chat.base import ChatRef
from nomorals.social.chat.living_cards import CardHost, CardStore
from nomorals.storage.db import Database


class Ctx:
    def __init__(self, db):
        self.db = db


ADA = Player.from_sender("telegram", "111", "Ada")
BOLA = Player.from_sender("telegram", "222", "Bola")


class DuelGame(MultiGame):
    name = "dueltest"
    description = "delivery rig duel"
    min_players = 2
    max_players = 2
    ai_seats = 0
    move_timeout = 30.0

    def new_state(self, rng, **kw):
        return {"moves": [], "done": False, "winner_key": ""}

    def setup(self, room, mind):
        return "dueltest — ready."

    def on_move(self, room, player, text, mind):
        room.state["moves"].append(text)
        return [f"{player.name} plays {text}."]

    def is_over(self, room):
        return bool(room.state.get("done"))

    def describe_options(self, room):
        return ["strike", "win"]

    def describe_state(self, room):
        return f"moves: {len(room.state.get('moves') or [])}"


def make_duel():
    db = Database(":memory:")
    db.migrate()
    engine = GameEngine(Ctx(db), send=lambda chat, text: None)
    engine.games["dueltest"] = DuelGame()
    room, _ = engine.start("telegram:g1", "dueltest", ADA, kind="group")
    engine.join("telegram:g1", BOLA)
    return engine, db, room


STAMP_STYLED = "𝗣𝗼𝘄𝗲𝗿𝗲𝗱 𝗯𝘆 𝗽𝗲𝗮𝗰𝗲𝘁𝗵𝗲𝗳𝗶𝗿𝘀𝘁"


class RecordingTransport:
    """A fake chat surface that declares — and records — parse_mode."""

    def __init__(self) -> None:
        self.sends: list[dict] = []
        self.edits: list[dict] = []

    def send(self, platform, chat, text, buttons=None, parse_mode=""):
        self.sends.append({"text": text, "buttons": buttons,
                           "parse_mode": parse_mode})
        return True, "m1"

    def edit(self, chat, message_id, text, buttons=None, parse_mode=""):
        self.edits.append({"text": text, "buttons": buttons,
                           "parse_mode": parse_mode})
        return True


class CardDeliveryFormatTests(unittest.TestCase):
    def _publish(self):
        from nomorals.games import table as table_mod

        engine, db, room = make_duel()
        store = CardStore(db)
        transport = RecordingTransport()
        host = CardHost(store, send=transport.send, edit=transport.edit)
        chat = ChatRef(platform="telegram", chat_id="g1")
        card = table_mod.sync_table_card(host, engine, room, chat)
        return engine, db, room, store, transport, host, chat, card

    def test_edit_carries_the_same_parse_mode_as_the_send(self):
        """Publish then re-deliver the same card: the edit must
        carry the same parse mode and the same formatted text as
        the send — code block, bold and italic survive the edit."""
        (_engine, _db, _room, _store, transport, host, chat,
         card) = self._publish()
        self.assertEqual(len(transport.sends), 1)
        send = transport.sends[0]
        self.assertEqual(send["parse_mode"], "HTML")
        self.assertIn("<pre>", send["text"])
        self.assertIn("<b>", send["text"])

        host.deliver(card, chat)
        self.assertEqual(len(transport.edits), 1)
        edit = transport.edits[0]
        self.assertEqual(edit["parse_mode"], "HTML")
        self.assertEqual(edit["text"], send["text"])
        self.assertIn("<pre>", edit["text"])
        self.assertIn("<b>", edit["text"])

    def test_revise_after_a_move_keeps_the_format(self):
        """A real revise (a move happened) also edits formatted:
        the new countdown renders with its code block intact."""
        from nomorals.games import table as table_mod

        (engine, _db, room, _store, transport, host, chat,
         card) = self._publish()
        engine.move("telegram:g1", "strike", ADA, kind="group")
        table_mod.sync_table_card(host, engine, room, chat)
        self.assertEqual(len(transport.edits), 1)
        edit = transport.edits[0]
        self.assertEqual(edit["parse_mode"], "HTML")
        self.assertIn("<pre>", edit["text"])
        self.assertIn("moves: 1", edit["text"])

    def test_stamp_renders_in_the_strong_face(self):
        """The stamp on a real card is the brandmark face (sans
        bold), not plain italic."""
        from nomorals.social.chat.living_cards import render_card

        (_engine, _db, _room, _store, _transport, _host, _chat,
         card) = self._publish()
        rendered = render_card(card, "telegram")
        self.assertIn(STAMP_STYLED, rendered.text)
        self.assertNotIn("Powered by peacethefirst", rendered.text)

    def test_stamp_once_on_rerender(self):
        """Re-rendering a stamped card never double-stamps, and a
        text that already closes with the stamp passes through
        untouched — old plain form included."""
        from nomorals.social.chat.living_cards import render_card
        from nomorals.social.chat.typography import apply_stamp

        (_engine, _db, _room, _store, _transport, _host, _chat,
         card) = self._publish()
        first = render_card(card, "telegram").text
        second = render_card(card, "telegram").text
        self.assertEqual(first.count(STAMP_STYLED), 1)
        self.assertEqual(second.count(STAMP_STYLED), 1)
        self.assertEqual(apply_stamp(first), first)
        legacy = "hello\n\n<i>Powered by peacethefirst</i>"
        self.assertEqual(apply_stamp(legacy), legacy)
        self.assertEqual(apply_stamp("hello").count(STAMP_STYLED), 1)

    def test_stamp_light_form_keeps_one_stamp(self):
        """The conversational light form also carries exactly one
        styled stamp."""
        from nomorals.social.chat.typography import apply_stamp

        out = apply_stamp("hello", light=True)
        self.assertEqual(out.count(STAMP_STYLED), 1)
        self.assertEqual(apply_stamp(out, light=True), out)

    def test_to_telegram_converts_both_stamp_faces_to_the_styled_face(self):
        """The dialect converter must recognise the stamp in either
        face (plain legacy or styled) and emit the styled HTML
        form — otherwise a canonical-dialect render (the paged
        help) prints the stamp as literal underscores on
        Telegram."""
        from nomorals.social.chat.platforms import to_telegram
        from nomorals.social.chat.typography import DEFAULT_STAMP_TEXT, STAMP_TEXT_STYLED

        for face in (DEFAULT_STAMP_TEXT, STAMP_TEXT_STYLED):
            out = to_telegram("body\n\n_" + face + "_")
            self.assertEqual(
                out, "body\n\n<i>" + STAMP_TEXT_STYLED + "</i>")
        # a non-stamp underscore pair stays literal prose
        self.assertEqual(to_telegram("body _hello_ end"),
                         "body _hello_ end")


if __name__ == "__main__":
    unittest.main()
