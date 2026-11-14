"""Table-card ticker (owner screenshot): the live table's countdown
must keep ticking between game events.

The table card's "59s left" only re-rendered on game events
(start/join/leave/move/timeout/finish) — between events the
displayed countdown was frozen.  A periodic reviser re-renders the
card on a clock (~5s) while the table is live, through the same
table sink the events use.
"""
from __future__ import annotations

import time
import unittest
from unittest import mock

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
    """Deterministic two-human duel with a 30s move clock."""

    name = "dueltest"
    description = "ticker rig duel"
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


class RecordingTransport:
    """A fake chat surface: records sends and edits."""

    def __init__(self) -> None:
        self.sends: list[dict] = []
        self.edits: list[dict] = []

    def send(self, platform, chat, text, buttons=None):
        self.sends.append({"text": text, "buttons": buttons})
        return True, "m1"

    def edit(self, chat, message_id, text, buttons=None):
        self.edits.append({"text": text, "buttons": buttons})
        return True


def _deadline_row(card):
    for label, value in (card.rows or ()):
        if label == "deadline":
            return value
    return ""


class TableCardTickerTests(unittest.TestCase):
    def test_countdown_decreases_across_ticks_with_no_move(self):
        """The card's countdown ticks down on the clock alone —
        no move, no event, just time passing."""
        from nomorals.games import table as table_mod

        engine, db, room = make_duel()
        store = CardStore(db)
        transport = RecordingTransport()
        host = CardHost(store, send=transport.send, edit=transport.edit)
        chat = ChatRef(platform="telegram", chat_id="g1")
        ticker_cls = table_mod.TableCardTicker

        t0 = 1_700_000_000.0
        room.turn_started = t0
        clock = {"now": t0}

        def _sink(r, event):
            return table_mod.sync_table_card(host, engine, r, chat)

        with mock.patch("time.time", side_effect=lambda: clock["now"]):
            # Publish at T0: the card reads 30s left.
            card = table_mod.sync_table_card(host, engine, room, chat)
            self.assertEqual(_deadline_row(card), "30s left")
            self.assertEqual(room.state.get("moves"), [])

            ticker = ticker_cls(engine, _sink, host, "telegram:g1")
            # Same instant: nothing changed, no delivery.
            self.assertTrue(ticker.tick())
            self.assertEqual(transport.edits, [])
            # Ten seconds pass with NO move: the tick re-renders.
            clock["now"] = t0 + 10.0
            self.assertTrue(ticker.tick())
            self.assertEqual(len(transport.edits), 1)
            stored = store.get(card.card_id)
            self.assertEqual(_deadline_row(stored), "20s left")
            self.assertEqual(room.state.get("moves"), [])

    def test_identical_tick_sends_nothing(self):
        """A tick that would render byte-identical content sends
        nothing — an identical edit would make the adapter refuse
        and the host fall back to a duplicate fresh send."""
        from nomorals.games import table as table_mod

        engine, db, room = make_duel()
        store = CardStore(db)
        transport = RecordingTransport()
        host = CardHost(store, send=transport.send, edit=transport.edit)
        chat = ChatRef(platform="telegram", chat_id="g1")

        t0 = 1_700_000_000.0
        room.turn_started = t0
        clock = {"now": t0}

        def _sink(r, event):
            return table_mod.sync_table_card(host, engine, r, chat)

        with mock.patch("time.time", side_effect=lambda: clock["now"]):
            card = table_mod.sync_table_card(host, engine, room, chat)
            self.assertEqual(_deadline_row(card), "30s left")
            ticker = table_mod.TableCardTicker(engine, _sink, host,
                                               "telegram:g1")
            self.assertTrue(ticker.tick())
            self.assertTrue(ticker.tick())
            # The ticks changed nothing: no edit at all — and no
            # second send either (the one send is the publish).
            self.assertEqual(transport.edits, [])
            self.assertEqual(len(transport.sends), 1)

    def test_tick_after_close_stops(self):
        """Once no live room remains the tick reports stop and
        sends nothing — the ticker never haunts a closed table."""
        from nomorals.games import table as table_mod

        sent = []

        class _DeadEngine:
            def live(self, chat_key):
                return None

        def _sink(room, event):
            sent.append(event)
            return None

        ticker = table_mod.TableCardTicker(_DeadEngine(), _sink, None,
                                           "telegram:g1")
        self.assertFalse(ticker.tick())
        self.assertEqual(sent, [])

    def test_thread_runs_and_stops(self):
        """The daemon thread ticks on its clock and exits when
        stopped — no leak, no second card path."""
        from nomorals.games import table as table_mod

        engine, db, room = make_duel()
        store = CardStore(db)
        transport = RecordingTransport()
        host = CardHost(store, send=transport.send, edit=transport.edit)
        chat = ChatRef(platform="telegram", chat_id="g1")

        def _sink(r, event):
            return table_mod.sync_table_card(host, engine, r, chat)

        table_mod.sync_table_card(host, engine, room, chat)
        ticker = table_mod.TableCardTicker(engine, _sink, host,
                                           "telegram:g1", interval=0.05)
        thread = ticker.start()
        try:
            self.assertTrue(thread.is_alive())
            time.sleep(0.15)
            self.assertTrue(thread.is_alive())
        finally:
            ticker.stop()
            thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
