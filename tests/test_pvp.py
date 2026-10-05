"""PvP duel + raid boss: combat, timeouts, forfeits, boss AI, loot split."""
from __future__ import annotations

import random
import unittest

from nomorals.games.combat import new_fighter, strike, tick_fighter
from nomorals.games.engine import GameEngine
from nomorals.games.games.pvp import DuelGame, RaidGame
from nomorals.games.players import Player
from nomorals.storage.db import Database


class Ctx:
    def __init__(self, db):
        self.db = db


def make_engine():
    db = Database(":memory:")
    db.migrate()
    sent: list[str] = []
    engine = GameEngine(Ctx(db), send=lambda chat, text: sent.append(text))
    return engine, db, sent


ADA = Player.from_sender("telegram", "111", "Ada")
BOB = Player.from_sender("telegram", "222", "Bob")
CAT = Player.from_sender("whatsapp", "333", "Cat")


def start_duel(engine, host=ADA, kind="group"):
    room, msgs = engine.start("test:pvp", "pvp", host, kind=kind)
    return room, msgs


def join_duel(engine, player):
    return engine.join("test:pvp", player)


def start_raid(engine, host=ADA, kind="group", key="test:raid"):
    room, msgs = engine.start(key, "raid", host, kind=kind)
    return room, msgs


# ── combat primitives ─────────────────────────────────────────────────

class CombatTests(unittest.TestCase):
    def test_strike_basic(self):
        rng = random.Random(7)
        a, d = new_fighter(), new_fighter()
        rep = strike(a, d, rng)
        self.assertGreater(rep["dmg"], 0)
        self.assertEqual(d["hp"], d["max_hp"] - rep["dmg"])

    def test_strike_dodge(self):
        rng = random.Random(7)
        a, d = new_fighter(), new_fighter()
        d["dodge_next"] = True
        rep = strike(a, d, rng)
        self.assertTrue(rep["dodged"])
        self.assertEqual(d["hp"], d["max_hp"])
        self.assertFalse(d["dodge_next"])

    def test_strike_focus_spent(self):
        rng = random.Random(7)
        a, d = new_fighter(), new_fighter()
        a["focused"] = True
        rep = strike(a, d, rng)
        self.assertTrue(rep["focused"])
        self.assertFalse(a["focused"])

    def test_strike_defend_halves(self):
        d1, d2 = new_fighter(), new_fighter()
        d2["defending"] = True
        r1 = strike(new_fighter(), d1, random.Random(7))
        r2 = strike(new_fighter(), d2, random.Random(7))
        self.assertLessEqual(r2["dmg"], r1["dmg"])
        self.assertFalse(d2["defending"])

    def test_strike_shield_saves(self):
        rng = random.Random(7)
        a = new_fighter(atk=999)
        d = new_fighter(hp=5)
        d["shield"] = True
        rep = strike(a, d, rng)
        self.assertTrue(rep["shielded"])
        self.assertEqual(d["hp"], 1)
        self.assertFalse(d["shield"])

    def test_strike_ignore_def(self):
        rng = random.Random(7)
        a, d = new_fighter(), new_fighter()
        d["def"] = 100
        rep = strike(a, d, rng, ignore_def=1.0)
        rep2 = strike(new_fighter(), new_fighter(hp=50, atk=10, dfn=100),
                     random.Random(7))
        self.assertGreaterEqual(rep["dmg"], rep2["dmg"])

    def test_tick_fighter(self):
        f = new_fighter()
        f["fury_cd"] = 2
        f["atk"] = 13
        f["warcry_turns"] = 1
        notes = tick_fighter(f, {"dragon_punch": 2})
        self.assertEqual(f["fury_cd"], 1)
        self.assertEqual(f["atk"], 10)  # war cry wore off (-3)
        self.assertTrue(any("war cry" in n for n in notes))


# ── duel ──────────────────────────────────────────────────────────────

class DuelTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.db, self.sent = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def test_registered(self):
        self.assertIn("pvp", self.engine.games)
        self.assertIn("raid", self.engine.games)

    def test_lobby_waits(self):
        room, msgs = start_duel(self.engine)
        self.assertTrue(room.state["lobby"])
        out = self.engine.move("test:pvp", "attack", ADA, kind="group")
        self.assertTrue(any("challenger" in m for m in out))

    def test_join_starts_fight(self):
        room, _ = start_duel(self.engine)
        msgs = join_duel(self.engine, BOB)
        self.assertFalse(room.state["lobby"])
        self.assertIn(ADA.key, room.state["fighters"])
        self.assertIn(BOB.key, room.state["fighters"])
        self.assertTrue(any("DUEL" in m for m in msgs))

    def test_attack_trades_turns(self):
        room, _ = start_duel(self.engine)
        join_duel(self.engine, BOB)
        hp_before = room.state["fighters"][BOB.key]["hp"]
        out = self.engine.move("test:pvp", "attack", ADA, kind="group")
        hp_after = room.state["fighters"][BOB.key]["hp"]
        self.assertLess(hp_after, hp_before)
        # turn advanced to Bob
        self.assertEqual(room.current.key, BOB.key)
        self.assertTrue(any("Bob" in m for m in out))

    def test_wrong_turn_blocked(self):
        room, _ = start_duel(self.engine)
        join_duel(self.engine, BOB)
        out = self.engine.move("test:pvp", "attack", BOB, kind="group")
        self.assertTrue(any("waiting on" in m for m in out))
        # Bob's HP untouched
        self.assertEqual(room.state["fighters"][BOB.key]["hp"],
                         room.state["fighters"][BOB.key]["max_hp"])

    def test_kill_wins(self):
        room, _ = start_duel(self.engine)
        join_duel(self.engine, BOB)
        room.state["fighters"][BOB.key]["hp"] = 1
        out = self.engine.move("test:pvp", "attack", ADA, kind="group")
        self.assertTrue(any("takes the duel" in m for m in out))
        # room finished + winner credited
        self.assertIsNone(self.engine.live("test:pvp"))
        self.assertTrue(any("Ada" in m and "wins" in m for m in self.sent
                            if "wins the duel" in m) or
                        any("wins the duel" in m for m in out))

    def test_timeout_autoguard_then_forfeit(self):
        room, _ = start_duel(self.engine)
        join_duel(self.engine, BOB)
        game = self.engine.games["pvp"]
        # miss 1: auto-guard
        out = game.on_timeout(room, ADA, self.engine._mind)
        self.assertTrue(any("auto-guard" in m for m in out))
        self.assertTrue(room.state["fighters"][ADA.key]["defending"])
        # miss 2: auto-guard again
        game.on_timeout(room, BOB, self.engine._mind)
        out = game.on_timeout(room, ADA, self.engine._mind)
        self.assertTrue(any("auto-guard" in m for m in out))
        # miss 3: forfeit
        out = game.on_timeout(room, ADA, self.engine._mind)
        self.assertTrue(any("forfeit" in m for m in out))
        self.assertTrue(game.is_over(room))
        self.assertEqual(game.winner(room).key, BOB.key)

    def test_leave_walkover(self):
        room, _ = start_duel(self.engine)
        join_duel(self.engine, BOB)
        msgs = self.engine.leave("test:pvp", BOB)
        self.assertTrue(any("walkover" in m for m in msgs))
        self.assertIsNone(self.engine.live("test:pvp"))

    def test_duel_score_is_damage(self):
        room, _ = start_duel(self.engine)
        join_duel(self.engine, BOB)
        game = self.engine.games["pvp"]
        self.engine.move("test:pvp", "attack", ADA, kind="group")
        self.assertGreater(game.score(room, ADA), 0)
        self.assertEqual(game.score(room, BOB), 0)

    def test_duel_xp(self):
        game = self.engine.games["pvp"]
        room, _ = start_duel(self.engine)
        join_duel(self.engine, BOB)
        self.assertEqual(game.xp_reward(True, room, ADA), 75)  # 60 + 15 clean
        self.assertEqual(game.xp_reward(False, room, BOB), 20)

    def test_progression_applies(self):
        # a leveled player hits harder than base
        room, _ = start_duel(self.engine)
        room.state["progression"][ADA.key] = {
            "level": 10, "max_hp": 40, "atk": 8, "def": 4}
        game = self.engine.games["pvp"]
        game._build_fighter(room, ADA)
        f = room.state["fighters"][ADA.key]
        self.assertEqual(f["max_hp"], 90)
        self.assertEqual(f["atk"], 18)
        self.assertEqual(f["def"], 9)


# ── raid ──────────────────────────────────────────────────────────────

class RaidTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.db, self.sent = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def test_boss_scales_with_party(self):
        room, _ = start_raid(self.engine)
        b1 = room.state["fighters"]["boss"]
        self.assertEqual(b1["max_hp"], 500)  # 300 + 200*1
        self.engine.join("test:raid", BOB)
        b2 = room.state["fighters"]["boss"]
        self.assertEqual(b2["max_hp"], 700)  # +200 reinforcement
        self.assertIn(BOB.key, room.state["fighters"])

    def test_boss_has_name(self):
        room, _ = start_raid(self.engine)
        from nomorals.games.games.pvp import BOSS_NAMES
        self.assertIn(room.state["fighters"]["boss"]["name"], BOSS_NAMES)

    def test_player_hits_boss(self):
        room, _ = start_raid(self.engine)
        hp_before = room.state["fighters"]["boss"]["hp"]
        self.engine.move("test:raid", "attack", ADA, kind="group")
        self.assertLess(room.state["fighters"]["boss"]["hp"], hp_before)
        dmg = room.state["dmg"][ADA.key]
        self.assertGreater(dmg, 0)

    def test_boss_acts_after_full_round(self):
        room, _ = start_raid(self.engine)
        self.engine.join("test:raid", BOB)
        ada_hp = room.state["fighters"][ADA.key]["hp"]
        bob_hp = room.state["fighters"][BOB.key]["hp"]
        self.engine.move("test:raid", "attack", ADA, kind="group")
        # not a full round yet — boss hasn't acted
        self.assertEqual(room.state["fighters"][ADA.key]["hp"], ada_hp)
        self.engine.move("test:raid", "attack", BOB, kind="group")
        # round wrapped — boss slammed somebody
        after = (room.state["fighters"][ADA.key]["hp"] +
                 room.state["fighters"][BOB.key]["hp"])
        self.assertLess(after, ada_hp + bob_hp)

    def test_boss_acts_when_teammate_down(self):
        # a dead hunter's missing turn must not stall the boss forever:
        # the round completes on living hunters' actions alone
        room, _ = start_raid(self.engine)
        self.engine.join("test:raid", BOB)
        room.state["fighters"][BOB.key]["hp"] = 0
        ada_hp = room.state["fighters"][ADA.key]["hp"]
        self.engine.move("test:raid", "attack", ADA, kind="group")
        # only Ada is alive → her single move completes the round
        self.assertLess(room.state["fighters"][ADA.key]["hp"], ada_hp)

    def test_cleave_every_third_round(self):
        room, _ = start_raid(self.engine)
        self.engine.join("test:raid", BOB)
        # force round 3 via the move counter: one hunter already acted,
        # and it's Bob's turn to complete the round
        room.state["round"] = 2
        room.state["round_moves"] = 1
        room.turn = 1
        ada_hp = room.state["fighters"][ADA.key]["hp"]
        bob_hp = room.state["fighters"][BOB.key]["hp"]
        self.engine.move("test:raid", "attack", BOB, kind="group")
        self.assertEqual(room.state["round"], 3)
        # cleave hit BOTH players
        self.assertLess(room.state["fighters"][ADA.key]["hp"], ada_hp)
        self.assertLess(room.state["fighters"][BOB.key]["hp"], bob_hp)

    def test_enrage(self):
        room, _ = start_raid(self.engine)
        game = self.engine.games["raid"]
        b = room.state["fighters"]["boss"]
        atk_before = b["atk"]
        b["hp"] = int(b["max_hp"] * 0.2)
        out: list[str] = []
        game._boss_turn(room, self.engine._mind, out)
        self.assertTrue(b["enraged"])
        self.assertGreater(b["atk"], atk_before)
        self.assertTrue(any("ENRAGES" in m for m in out))

    def test_victory_all_win(self):
        room, _ = start_raid(self.engine)
        self.engine.join("test:raid", BOB)
        game = self.engine.games["raid"]
        room.state["fighters"]["boss"]["hp"] = 1
        out = self.engine.move("test:raid", "attack", ADA, kind="group")
        self.assertTrue(any("victorious" in m for m in out))
        self.assertEqual(game.winner(room), "all")
        self.assertIsNone(self.engine.live("test:raid"))
        # both players got win credit in the ledger output
        joined = " ".join(self.sent)
        self.assertIn("Ada", joined)
        self.assertIn("Bob", joined)

    def test_defeat(self):
        room, _ = start_raid(self.engine)
        game = self.engine.games["raid"]
        room.state["fighters"][ADA.key]["hp"] = 1
        b = room.state["fighters"]["boss"]
        b["atk"] = 999
        out: list[str] = []
        game._boss_turn(room, self.engine._mind, out)
        # Ada is dead; check the defeat path via _after_player
        out2 = game._after_player(room, ADA, self.engine._mind, [])
        self.assertTrue(game.is_over(room))
        w = game.winner(room)
        self.assertTrue(w.is_ai)

    def test_timeout_auto_attacks(self):
        room, _ = start_raid(self.engine)
        game = self.engine.games["raid"]
        hp_before = room.state["fighters"]["boss"]["hp"]
        out = game.on_timeout(room, ADA, self.engine._mind)
        self.assertTrue(any("auto-attack" in m for m in out))
        self.assertLess(room.state["fighters"]["boss"]["hp"], hp_before)

    def test_loot_shares(self):
        room, _ = start_raid(self.engine)
        self.engine.join("test:raid", BOB)
        game = self.engine.games["raid"]
        room.state["dmg"][ADA.key] = 75
        room.state["dmg"][BOB.key] = 25
        shares = game._shares(room)
        self.assertEqual(shares[ADA.key], 75)
        self.assertEqual(shares[BOB.key], 25)
        self.assertEqual(game.score(room, ADA), 75)
        # MVP earns more XP
        self.assertGreater(game.xp_reward(True, room, ADA),
                           game.xp_reward(True, room, BOB))

    def test_dead_player_sits_out(self):
        room, _ = start_raid(self.engine)
        room.state["fighters"][ADA.key]["hp"] = 0
        out = self.engine.move("test:raid", "attack", ADA, kind="group")
        self.assertTrue(any("down" in m for m in out))

    def test_down_player_skipped_by_boss(self):
        room, _ = start_raid(self.engine)
        self.engine.join("test:raid", BOB)
        room.state["fighters"][ADA.key]["hp"] = 0
        bob_hp = room.state["fighters"][BOB.key]["hp"]
        game = self.engine.games["raid"]
        out: list[str] = []
        game._boss_turn(room, self.engine._mind, out)
        # only Bob could be hit
        self.assertLessEqual(room.state["fighters"][BOB.key]["hp"], bob_hp)


# ── relay: duel over two DMs ────────────────────────────────────────────

class RelayDuelTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.db, self.sent = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def test_invite_accept_turn_order(self):
        relay = self.engine.relay
        inv = relay.create_invite("telegram:111", ADA, "pvp",
                                  to_label="bob")
        room_info = relay.accept_invite(inv.code, "telegram:222", BOB)
        vroom = self.engine.live(room_info.virtual_chat)
        self.assertIsNotNone(vroom)
        # turn order is enforced (group kind), not free-for-all
        self.assertEqual(vroom.kind, "group")
        self.assertEqual(len(vroom.humans), 2)
        # both fighters built (B's mirror ran on accept)
        self.assertIn(ADA.key, vroom.state["fighters"])
        self.assertIn(BOB.key, vroom.state["fighters"])
        # Bob can't jump Ada's turn
        out = self.engine.move(room_info.virtual_chat, "attack", BOB)
        self.assertTrue(any("waiting on" in m for m in out))

    def test_relay_move_roundtrip(self):
        relay = self.engine.relay
        inv = relay.create_invite("telegram:111", ADA, "pvp")
        room_info = relay.accept_invite(inv.code, "telegram:222", BOB)
        bob_hp = room_info and self.engine.live(
            room_info.virtual_chat).state["fighters"][BOB.key]["hp"]
        msgs = relay.relay_move("telegram:111", "attack", ADA)
        self.assertTrue(msgs)
        vroom = self.engine.live(room_info.virtual_chat)
        self.assertLess(vroom.state["fighters"][BOB.key]["hp"], bob_hp)

    def test_status_text_live(self):
        relay = self.engine.relay
        inv = relay.create_invite("telegram:111", ADA, "pvp")
        room_info = relay.accept_invite(inv.code, "telegram:222", BOB)
        text = relay.status_text("telegram:111")
        self.assertIsNotNone(text)
        self.assertIn("pvp", text)
        self.assertIn("Ada", text)
        self.assertIn("Bob", text)

    def test_status_text_none(self):
        relay = self.engine.relay
        self.assertIsNone(relay.status_text("telegram:999"))

    def test_touch_refreshes_activity(self):
        import time
        relay = self.engine.relay
        inv = relay.create_invite("telegram:111", ADA, "pvp")
        room_info = relay.accept_invite(inv.code, "telegram:222", BOB)
        relay.relays[room_info.room_id].last_activity = 0.0
        relay.touch(room_info.room_id)
        self.assertGreater(
            relay.relays[room_info.room_id].last_activity, 0.0)

    def test_rematch_invite(self):
        relay = self.engine.relay
        inv = relay.create_invite("telegram:111", ADA, "pvp")
        room_info = relay.accept_invite(inv.code, "telegram:222", BOB)
        rematch = relay.rematch_invite("telegram:111")
        self.assertIsNotNone(rematch)
        self.assertEqual(rematch.game_name, "pvp")
        self.assertNotEqual(rematch.code, room_info.room_id)
        # original relay still intact
        self.assertIsNotNone(
            relay.get_relay_for_chat("telegram:111"))

    def test_virtual_fanout_lookup(self):
        relay = self.engine.relay
        inv = relay.create_invite("telegram:111", ADA, "pvp")
        room_info = relay.accept_invite(inv.code, "telegram:222", BOB)
        found = relay.get_relay_for_virtual(room_info.virtual_chat)
        self.assertIsNotNone(found)
        self.assertEqual(found.room_id, room_info.room_id)
        self.assertIsNone(relay.get_relay_for_virtual("relay:nope"))


class MoveIntentTests(unittest.TestCase):
    """Casual chat during a duel must not trigger game responses."""

    def setUp(self):
        self.engine, self.db, self.sent = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def test_combat_verbs_are_moves(self):
        from nomorals.games.games.pvp import DuelGame
        game = DuelGame()
        for verb in ("attack", "focus", "fury", "defend", "potion",
                     "item shield", "skill fireball", "combo a + b",
                     "Attack", "  FURY  "):
            self.assertTrue(game.is_move_text(verb), verb)

    def test_chat_is_not_a_move(self):
        from nomorals.games.games.pvp import DuelGame
        game = DuelGame()
        for chat in ("say that again? i spaced",
                     "brain's buffering, one sec",
                     "that lands. go on",
                     "noted. and i mean that in a good way",
                     "hello", ""):
            self.assertFalse(game.is_move_text(chat), chat)

    def test_base_default_is_permissive(self):
        from nomorals.games.games.base import MultiGame
        self.assertTrue(MultiGame().is_move_text("anything at all"))

    def test_chat_while_waiting_stays_silent(self):
        relay = self.engine.relay
        inv = relay.create_invite("telegram:111", ADA, "pvp")
        room_info = relay.accept_invite(inv.code, "telegram:222", BOB)
        # Bob's turn is not now — casual chat gets silence, not
        # "waiting on" spam
        out = self.engine.move(room_info.virtual_chat,
                               "say that again? i spaced", BOB)
        self.assertEqual(out, [])
        # ...but a real move attempt still gets the nudge
        out = self.engine.move(room_info.virtual_chat, "attack", BOB)
        self.assertTrue(any("waiting on" in m for m in out))

    def test_lobby_nudges_capped(self):
        from nomorals.games.games.pvp import DuelGame
        game = DuelGame()
        # start a pvp lobby with only Ada — never becomes ready
        self.engine.start("telegram:111", "pvp", ADA)
        vroom = self.engine.live("telegram:111")
        self.assertIsNotNone(vroom)
        # 5 timeouts → only 3 "still waiting" messages, then quiet
        nudges = 0
        for _ in range(5):
            out = game.on_timeout(vroom, ADA, self.engine._mind)
            nudges += sum("still waiting" in m for m in out)
        self.assertEqual(nudges, 3)


if __name__ == "__main__":
    unittest.main()
