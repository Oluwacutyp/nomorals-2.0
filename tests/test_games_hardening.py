"""Games hardening (doc 8 Priority 4): live table, fenced controls,
atomic settlement, visible persistence health, resign/timeout law,
private hidden information, event-built replays, family identity.

Every test here was written against the hardened behavior; on the
pre-hardening tree each fails for the reason its docstring names.
"""
from __future__ import annotations

import time
import unittest

from nomorals.games.engine import GameEngine
from nomorals.games.games.base import MultiGame
from nomorals.games.players import Player
from nomorals.storage.db import Database


class Ctx:
    def __init__(self, db):
        self.db = db


def make_engine(db=None):
    db = db if db is not None else Database(":memory:")
    db.migrate()
    sent: list[str] = []
    engine = GameEngine(Ctx(db), send=lambda chat, text: sent.append(text))
    return engine, db, sent


ADA = Player.from_sender("telegram", "111", "Ada")
BOLA = Player.from_sender("telegram", "222", "Bola")
CAROL = Player.from_sender("telegram", "333", "Carol")


class DuelGame(MultiGame):
    """A deterministic two-human duel for the hardening rig.

    Seats exactly the humans who join (no AI), carries a per-player
    secret hand, and ends when someone plays "win".
    """

    name = "dueltest"
    description = "hardening rig duel"
    min_players = 2
    max_players = 2
    ai_seats = 0
    move_timeout = 30.0
    secret_state = {"hands": "per_player"}

    def new_state(self, rng, **kw):
        return {"moves": [], "done": False, "winner_key": "",
                "hands": {}}

    def setup(self, room, mind):
        for p in room.players:
            room.state["hands"][p.key] = f"SECRET-{p.name.upper()}-7"
        return "dueltest — ready."

    def on_join(self, room, player, mind):
        room.state["hands"][player.key] = \
            f"SECRET-{player.name.upper()}-7"
        return None

    def on_move(self, room, player, text, mind):
        room.state["moves"].append(text)
        if text.strip().lower() == "win":
            room.state["done"] = True
            room.state["winner_key"] = player.key
            return [f"{player.name} takes the duel."]
        return [f"{player.name} plays {text}."]

    def is_over(self, room):
        return bool(room.state.get("done"))

    def winner(self, room):
        key = room.state.get("winner_key")
        return room.player(key) if key else "draw"

    def score(self, room, player):
        return len(room.state.get("moves") or [])

    def describe_options(self, room):
        return ["strike", "win"]

    def describe_state(self, room):
        return f"moves: {len(room.state.get('moves') or [])}"


def make_duel(db=None):
    """A live group duel: Ada (host) + Bola, Ada to move."""
    engine, db, sent = make_engine(db)
    engine.games["dueltest"] = DuelGame()
    room, _ = engine.start("telegram:g1", "dueltest", ADA, kind="group")
    engine.join("telegram:g1", BOLA)
    return engine, db, room


class RevisionFencingTests(unittest.TestCase):
    def test_room_revision_advances_and_fences_moves(self):
        engine, db, room = make_duel()
        rev = room.revision
        self.assertGreaterEqual(rev, 1)

        stale = engine.move_fenced("telegram:g1", "strike", ADA,
                                   revision=rev - 1)
        self.assertEqual(stale.status, "stale")
        self.assertEqual(room.state["moves"], [])  # nothing applied
        self.assertTrue(any("dueltest" in m for m in stale.messages),
                        stale.messages)  # current state shown

        not_player = engine.move_fenced("telegram:g1", "strike", CAROL,
                                        revision=rev)
        self.assertEqual(not_player.status, "not_player")
        self.assertEqual(room.state["moves"], [])

        not_turn = engine.move_fenced("telegram:g1", "strike", BOLA,
                                      revision=rev)
        self.assertEqual(not_turn.status, "not_turn")
        self.assertEqual(room.state["moves"], [])

        ok = engine.move_fenced("telegram:g1", "strike", ADA,
                                revision=rev)
        self.assertEqual(ok.status, "ok")
        self.assertEqual(room.state["moves"], ["strike"])
        self.assertGreater(room.revision, rev)
        # the pre-move revision is now stale
        again = engine.move_fenced("telegram:g1", "strike", BOLA,
                                   revision=rev)
        self.assertEqual(again.status, "stale")

    def test_revision_and_deadline_survive_restart(self):
        engine, db, room = make_duel()
        rev = room.revision
        room.turn_started = time.time() - 100.0  # clock long expired
        engine._persist(room)
        engine.shutdown()  # its ticker must not sweep the old copy

        engine2, _, _ = make_engine(db)
        engine2.games["dueltest"] = DuelGame()
        restored = engine2.live("telegram:g1")
        self.assertIsNotNone(restored)
        self.assertEqual(restored.revision, rev + 1)  # the persist bumped
        # the deadline was NOT refreshed by the restart: the expired
        # turn times out on the first sweep instead of getting 30s more
        self.assertLess(restored.turn_started, time.time() - 90.0)
        engine2._sweep_timeouts()
        journal = restored.state.get("_clock_events") or []
        self.assertTrue(any(e.get("event") == "timeout"
                            for e in journal), journal)


class ResignTests(unittest.TestCase):
    def test_resign_is_a_recorded_loss(self):
        engine, db, room = make_duel()
        out = engine.move("telegram:g1", "/resign", BOLA)
        self.assertTrue(any("resign" in m.lower() for m in out), out)
        self.assertIsNone(engine.live("telegram:g1"))  # table closed
        decided = room.state.get("decided") or {}
        self.assertEqual(decided.get("how"), "resign")
        self.assertEqual(decided.get("loser"), BOLA.key)
        self.assertEqual(decided.get("winner"), ADA.key)
        # the settle honored the resignation: Ada won, Bola lost
        self.assertEqual(engine.store.get_for(ADA).wins, 1)
        self.assertEqual(engine.store.get_for(BOLA).losses, 1)
        journal = room.state.get("_clock_events") or []
        self.assertTrue(any(e.get("event") == "resign"
                            for e in journal), journal)

    def test_non_player_cannot_resign(self):
        engine, db, room = make_duel()
        engine.move("telegram:g1", "/resign", CAROL)
        self.assertIsNotNone(engine.live("telegram:g1"))
        self.assertFalse(room.state.get("decided"))


class SettlementAtomicityTests(unittest.TestCase):
    def test_mid_settlement_failure_pays_nobody_and_retries_once(self):
        engine, db, room = make_duel()
        ada_before = engine.store.get_for(ADA)
        bola_before = engine.store.get_for(BOLA)

        real = engine.store.record_outcome
        calls = {"n": 0}
        gate = {"fail": True}

        def flaky(player, **kw):
            calls["n"] += 1
            # every attempt dies on its SECOND credit while the
            # gate is on — including a ticker-driven retry, so the
            # mid-settlement state below is deterministic
            if gate["fail"] and calls["n"] % 2 == 0:
                raise RuntimeError("injected settle failure")
            return real(player, **kw)

        engine.store.record_outcome = flaky
        out = engine.move("telegram:g1", "win", ADA)
        # the failure is VISIBLE, not a silent half-pay
        self.assertTrue(any("settle" in m.lower() and "fail" in m.lower()
                            for m in out), out)
        # nobody was paid: the whole settle rolled back
        self.assertEqual(engine.store.get_for(ADA).games_played,
                         ada_before.games_played)
        self.assertEqual(engine.store.get_for(ADA).coins,
                         ada_before.coins)
        self.assertEqual(engine.store.get_for(BOLA).games_played,
                         bola_before.games_played)
        # the table survives for the retry
        self.assertIsNotNone(engine.live("telegram:g1"))
        row = db.query_one(
            "SELECT status FROM game_settlements WHERE room_id = ?",
            (room.id,))
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "planned")

        # heal and let the sweep retry: settles exactly once
        gate["fail"] = False
        for _ in range(5):  # the recovery scan is throttled
            engine._sweep_timeouts()
        self.assertIsNone(engine.live("telegram:g1"))
        self.assertEqual(engine.store.get_for(ADA).games_played,
                         ada_before.games_played + 1)
        self.assertEqual(engine.store.get_for(BOLA).games_played,
                         bola_before.games_played + 1)
        row = db.query_one(
            "SELECT status FROM game_settlements WHERE room_id = ?",
            (room.id,))
        self.assertEqual(row["status"], "applied")

        # a further sweep cannot pay twice
        coins = engine.store.get_for(ADA).coins
        for _ in range(5):
            engine._sweep_timeouts()
        self.assertEqual(engine.store.get_for(ADA).coins, coins)
        self.assertEqual(engine.store.get_for(ADA).games_played,
                         ada_before.games_played + 1)

    def test_crash_after_commit_replays_receipts_without_repaying(self):
        engine, db, room = make_duel()
        engine.move("telegram:g1", "win", ADA)
        ada = engine.store.get_for(ADA)
        paid_coins, paid_games = ada.coins, ada.games_played
        # simulate the crash window: settlement committed, but the
        # room row never got its 'finished' write
        db.execute("UPDATE game_rooms SET status = 'active' WHERE id = ?",
                   (room.id,))

        engine2, _, _ = make_engine(db)
        engine2.games["dueltest"] = DuelGame()
        restored = engine2.live("telegram:g1")
        self.assertIsNotNone(restored)
        out = engine2._finish(restored)
        # the original receipt lines come back…
        self.assertTrue(any("🪙" in m and "Ada" in m for m in out), out)
        # …and not one coin or game count moves again
        ada2 = engine2.store.get_for(ADA)
        self.assertEqual(ada2.coins, paid_coins)
        self.assertEqual(ada2.games_played, paid_games)


class PersistHealthTests(unittest.TestCase):
    def test_persist_failure_is_visible_on_the_surface(self):
        engine, db, sent = make_engine()
        engine.games["dueltest"] = DuelGame()
        db.execute("DROP TABLE game_rooms")
        room, _ = engine.start("telegram:g9", "dueltest", ADA,
                               kind="group")
        health = engine.persist_health()
        self.assertGreaterEqual(health["failed"], 1, health)
        self.assertTrue(health["last_error"], health)
        text = engine.describe(room)
        self.assertIn("saves failing", text)


class HiddenInfoTests(unittest.TestCase):
    def test_views_carry_only_the_viewers_own_secret(self):
        engine, db, room = make_duel()
        ada_view = engine.table_view(room, viewer_key=ADA.key)
        self.assertEqual(
            ada_view["state"]["hands"], {ADA.key: "SECRET-ADA-7"})
        bola_view = engine.table_view(room, viewer_key=BOLA.key)
        self.assertNotIn(ADA.key, bola_view["state"].get("hands", {}))
        self.assertNotIn("SECRET-ADA-7", str(bola_view))
        shared = engine.table_view(room)
        self.assertNotIn("SECRET-ADA-7", str(shared))
        self.assertNotIn("SECRET-BOLA-7", str(shared))

    def test_mafia_roles_never_reach_another_players_view(self):
        engine, db, sent = make_engine()
        room, _ = engine.start("telegram:g2", "mafia", ADA, kind="group")
        engine.join("telegram:g2", BOLA)
        roles = room.state.get("roles") or {}
        self.assertTrue(roles)  # the rig really dealt roles
        self.assertTrue(room.state.get("mafia"))  # mafia ids in state
        for viewer in (ADA, BOLA):
            view = engine.table_view(room, viewer_key=viewer.key)
            seen = view["state"].get("roles", {})
            # a viewer sees at most their OWN entry — never another
            # seat's, and never who the mafia is
            self.assertLessEqual(set(seen), {viewer.key})
            self.assertNotIn("mafia", seen.values())
            self.assertNotIn("mafia", view["state"])
        shared = engine.table_view(room)
        self.assertNotIn("roles", shared["state"])
        self.assertNotIn("mafia", shared["state"])


class TableCardTests(unittest.TestCase):
    def test_table_card_tracks_the_room_and_fences_taps(self):
        from nomorals.games import table as table_mod
        from nomorals.social.chat.base import ChatRef
        from nomorals.social.chat.living_cards import (
            CardHost,
            CardStore,
            TapStatus,
            render_card,
            resolve_card_tap,
        )

        engine, db, room = make_duel()
        store = CardStore(db)
        sent, edits = [], []

        def send(platform, chat, text, buttons=None):
            sent.append((text, buttons))
            return True, "m1"

        def edit(chat, message_id, text, buttons=None):
            edits.append((text, buttons))
            return True

        host = CardHost(store, send=send, edit=edit)
        chat = ChatRef(platform="telegram", chat_id="g1")
        card = table_mod.sync_table_card(host, engine, room, chat)
        self.assertEqual(card.family, "game")
        self.assertTrue(sent)
        rendered = render_card(card, "telegram")
        self.assertIn("dueltest", rendered.text)
        self.assertTrue(rendered.keyboard)  # legal actions ride buttons
        payloads = [data for row in rendered.keyboard for _, data in row]
        self.assertTrue(any("cardtap" in p for p in payloads))

        # a tap at the current card revision releases the fenced move
        out = resolve_card_tap(store, card.card_id, card.revision, 1,
                               chat_key="telegram:g1")
        self.assertEqual(out.status, TapStatus.OK)
        self.assertTrue(out.command.startswith("game move "), out.command)
        self.assertIn(str(room.revision), out.command)

        # play the move: the room advances, the card revises in place
        played = engine.move("telegram:g1", "strike", ADA)
        self.assertTrue(played)
        card2 = table_mod.sync_table_card(host, engine, room, chat)
        self.assertGreater(card2.revision, card.revision)
        self.assertTrue(edits)  # revised, not re-sent

        # the OLD tap is now stale and releases nothing
        stale = resolve_card_tap(store, card.card_id, card.revision, 1,
                                 chat_key="telegram:g1")
        self.assertEqual(stale.status, TapStatus.STALE)
        self.assertEqual(stale.command, "")

    def test_two_families_render_distinct_identities(self):
        from nomorals.games import table as table_mod
        from nomorals.social.chat.living_cards import render_card

        engine, db, sent = make_engine()
        engine.store.add_coins(ADA, 100, reason="test funding")
        ttt_room, _ = engine.start("telegram:f1", "ttt", ADA, kind="dm")
        ttt_card = table_mod.build_table_card(engine, ttt_room)
        slots_room, _ = engine.start("telegram:f2", "slots", ADA,
                                     kind="dm")
        slots_card = table_mod.build_table_card(engine, slots_room)

        self.assertEqual(table_mod.game_family(engine.games["ttt"]),
                         "strategy")
        self.assertEqual(table_mod.game_family(engine.games["slots"]),
                         "casino")
        ttt_text = render_card(ttt_card, "telegram").text
        slots_text = render_card(slots_card, "telegram").text
        self.assertNotEqual(ttt_text, slots_text)
        self.assertIn("♟", ttt_text)      # strategy banner
        self.assertIn("🎰", slots_text)   # casino banner
        # both carry the live-table facts through the same machinery
        for text in (ttt_text, slots_text):
            self.assertIn("turn", text.lower())
        self.assertIn("1️⃣", ttt_text)    # the ttt board itself

    def test_timeout_policy_line_names_the_clock_law(self):
        from nomorals.games import table as table_mod

        engine, db, sent = make_engine()
        engine.games["dueltest"] = DuelGame()
        line = table_mod.timeout_policy(engine.games["dueltest"])
        self.assertIn("30s", line)
        self.assertIn("forfeit", line)


class ReplayTests(unittest.TestCase):
    def test_replay_is_built_from_the_real_events(self):
        engine, db, room = make_duel()
        # force a stall on Ada, then let Bola… Ada still to move after
        # the pass: sweep the expired clock once
        room.turn_started = time.time() - 100.0
        engine._sweep_timeouts()
        engine.move("telegram:g1", "win",
                    engine.live("telegram:g1").current)
        text = engine.replay_text("telegram:g1")
        self.assertIsNotNone(text)
        self.assertIn("replay", text.lower())
        self.assertIn("Ada", text)
        self.assertTrue("⏰" in text or "stall" in text.lower(), text)
        self.assertIn("🪙", text)  # the settle, from the receipts
        self.assertIsNone(engine.replay_text("telegram:nowhere"))


if __name__ == "__main__":
    unittest.main()
