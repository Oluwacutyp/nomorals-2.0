"""One-command wager starts: /slots <amount> and /blackjack <amount>.

Owner ask (live, 2026-11-04): starting slots was a multi-step
dance — open the game, read the card, type "bet 20000", type
"spin".  Now the amount rides the start command itself: one
command opens the table, escrows the wager through the same purse
path the "bet" move uses, and pulls the first spin; the result is
the game's normal settle card.  Blackjack's start shape is the
same wager-then-deal, so "/blackjack <amount>" deals a hand at
that stake (its stake is chosen at setup, so the amount threads
into new_state through the engine's opt-in kwargs).

Pinned here: the fresh-table quickstart (exact purse math, win and
loss), the already-open table (re-bet + spin in place), the
table-full honesty for a second caller, insufficient balance
(refusal only — no debit, no half-open room), junk amounts falling
back to the plain start, the no-amount path untouched, the §33
gate refusing before anything is created, and blackjack's amount
form at both the wallet and the group estate.
"""
from __future__ import annotations

import random
import unittest
from types import SimpleNamespace



from nomorals.agents.partner.runtime_games import RuntimeGamesMixin
from nomorals.games.engine import GameEngine
from nomorals.games.games.casino import SlotsGame
from nomorals.games.players import Player
from nomorals.social.chat.control import parse_control
from nomorals.social.chat.group_surface import GroupStore, GroupSurface
from nomorals.storage.db import Database

KEY = "telegram-bot:-1001"
ADA = Player.from_sender("telegram", "111", "Ada")
BOB = Player.from_sender("telegram", "222", "Bob")


class Ctx:
    def __init__(self, db):
        self.db = db


def make_engine():
    db = Database(":memory:")
    db.migrate()
    sent: list[str] = []
    engine = GameEngine(Ctx(db), send=lambda chat, text: sent.append(text))
    return engine, db, sent


def wallet_rows(db, key):
    rows = db.execute(
        "SELECT amount, reason FROM game_wallet WHERE player_key = ?",
        (key,)).fetchall()
    return [(r[0], r[1]) for r in rows]


def force_reels(room, want):
    """Re-point a live slots room's draw stream at a seed whose
    first three reels satisfy ``want`` (the wager suite's helper)."""
    for seed in range(20000):
        probe = random.Random(seed)
        reels = [probe.choices(SlotsGame.SYMBOLS,
                               weights=SlotsGame.WEIGHTS, k=1)[0]
                 for _ in range(3)]
        if want(reels):
            room.state["fair"] = None
            room.seed = seed
            room._rng_instance = random.Random(seed)
            return reels
    raise AssertionError("no seed found")


def dispatch_verb(text: str) -> str:
    """The any-chat verb building in PartnerRuntime._process."""
    parsed = parse_control(text)
    assert parsed is not None, f"{text} did not parse"
    return (parsed.tail or parsed.arg) if parsed.kind == "game" \
        else parsed.kind + ((" " + parsed.tail) if parsed.tail else "")


def call_game(engine, text, chat_key, player, kind, blocked=False):
    # a bare mixin instance: _control_game's sibling helpers resolve
    # on the class, the engine + gate are instance-level doubles.
    rt = RuntimeGamesMixin.__new__(RuntimeGamesMixin)
    rt.context = SimpleNamespace()
    rt._game_engine = lambda: engine
    rt._group_system_blocked = lambda key: blocked
    return rt._control_game(
        dispatch_verb(text), chat_key=chat_key, player=player,
        kind=kind)


# ── DM rooms: the games wallet is the purse ─────────────────────

class SlotsQuickstartDmTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.db, self.sent = make_engine()
        self.engine.store.add_coins(ADA, 500, reason="seed funding")

    def tearDown(self):
        self.engine.shutdown()

    def test_fresh_table_bets_and_spins_in_one_command(self):
        reply = call_game(self.engine, "/slots 200", "test:qs1", ADA, "dm")
        # the wager landed through the bet path, journaled as a bet
        self.assertIn((-200, "game:slots slots bet"),
                      wallet_rows(self.db, ADA.key))
        # the bet confirmation is the game's own line
        self.assertIn("bet set to 200 coins — taken from your balance "
                      "(300 left)", reply)
        # the spin happened and settled: reels + the game-over card
        self.assertIn("[ ", reply)
        self.assertIn("🏁", reply)
        # the provably-fair commitment precedes the draw in the reply
        self.assertLess(reply.index("🔒"), reply.index("[ "))
        # the table settled and closed, exactly as a manual spin does
        self.assertIsNone(self.engine.live("test:qs1"))
        # money conserved: final coins = seed - stake + payout leg
        total = sum(a for a, _ in wallet_rows(self.db, ADA.key))
        self.assertEqual(self.engine.store.get_for(ADA).coins, total)
        self.assertIn(total, (300, 500, 700, 1300, 2300, 20300))

    def test_open_table_rebets_and_spins(self):
        room, _ = self.engine.start("test:qs2", "slots", ADA, kind="dm")
        self.engine.move("test:qs2", "bet 100", ADA, kind="dm")
        self.assertEqual(self.engine.store.get_for(ADA).coins, 400)
        force_reels(room, lambda r: r[0] == r[1] == r[2] == "🍒")
        reply = call_game(self.engine, "/slots 500", "test:qs2", ADA, "dm")
        # old escrow refunded, new one taken, cherries pay 2x on 500:
        # 500 seed - 100 bet + 100 refund - 500 bet + 1000 payout
        self.assertIn("bet set to 500 coins", reply)
        self.assertIn("🪙 Ada: +500 coins (slots paid 2x on 500)", reply)
        self.assertEqual(self.engine.store.get_for(ADA).coins, 1000)

    def test_open_table_losing_spin_consumes_the_escrow(self):
        room, _ = self.engine.start("test:qs3", "slots", ADA, kind="dm")
        force_reels(room, lambda r: len(set(r)) == 3)
        reply = call_game(self.engine, "/slots 200", "test:qs3", ADA, "dm")
        self.assertIn("no match — you lose 200 coins.", reply)
        self.assertIn("🪙 Ada: -200 coins (slots wager lost)", reply)
        self.assertEqual(self.engine.store.get_for(ADA).coins, 300)

    def test_insufficient_balance_refuses_and_leaves_no_room(self):
        reply = call_game(self.engine, "/slots 2000", "test:qs4", ADA, "dm")
        self.assertIn("bet is 2000 coins — your balance is 500", reply)
        self.assertIn("not enough", reply)
        # no half-open table, no debit, no phantom game-over card sent
        self.assertIsNone(self.engine.live("test:qs4"))
        self.assertNotIn("test:qs4", self.engine._rooms)
        self.assertEqual(self.engine.store.get_for(ADA).coins, 500)
        self.assertEqual(wallet_rows(self.db, ADA.key),
                         [(500, "seed funding")])
        self.assertFalse(any("🏁" in m for m in self.sent))

    def test_junk_amounts_fall_back_to_the_plain_start(self):
        for i, text in enumerate(
                ("/slots abc", "/slots 20x", "/slots 0", "/slots -50")):
            with self.subTest(text=text):
                key = f"test:junk{i}"
                reply = call_game(self.engine, text, key, ADA, "dm")
                self.assertIn("slots — bet and spin:", reply)
                room = self.engine.live(key)
                self.assertIsNotNone(room)
                self.assertEqual(room.state["bet"], 10)
                self.assertEqual(room.state["escrow"], 0)
                self.assertEqual(self.engine.store.get_for(ADA).coins, 500)
                self.engine.quit(key)

    def test_no_amount_start_is_unchanged(self):
        reply = call_game(self.engine, "/slots", "test:qs5", ADA, "dm")
        self.assertIn("🎮", reply)
        self.assertIn("slots — bet and spin:", reply)
        self.assertIn("bet <amount> — set your wager", reply)
        room = self.engine.live("test:qs5")
        self.assertIsNotNone(room)
        self.assertEqual(room.state["escrow"], 0)
        self.assertEqual(self.engine.store.get_for(ADA).coins, 500)

    def test_other_live_table_keeps_todays_refusal(self):
        self.engine.start("test:qs6", "numberguess", ADA, kind="dm")
        reply = call_game(self.engine, "/slots 200", "test:qs6", ADA, "dm")
        self.assertIn("a numberguess is already live here", reply)
        self.assertEqual(self.engine.store.get_for(ADA).coins, 500)


class SlotsQuickstartGroupTests(unittest.TestCase):
    """Group rooms: the estate is the purse (the wager build's
    _GroupWagerPurse seam, wired as the wager suite wires it)."""

    def setUp(self):
        import tempfile
        from pathlib import Path

        from nomorals.agents.partner.runtime_group import _GroupWagerPurse

        tmp = Path(tempfile.mkdtemp(prefix="slotstart-"))
        store = GroupStore(tmp / "groups.json").load()
        self.surface = GroupSurface(store)
        self.engine, self.db, self.sent = make_engine()
        surface = self.surface
        self.engine.purse_source = (
            lambda room: _GroupWagerPurse(surface, room.chat_key)
            if room.kind == "group" else None)
        self.surface.game_wager_credit(KEY, "111", 1000, "seed funding")

    def tearDown(self):
        self.engine.shutdown()

    def econ(self):
        return self.surface.economy

    def test_group_quickstart_moves_estate_money_exactly(self):
        reply = call_game(self.engine, "/slots 200", KEY, ADA, "group")
        self.assertIn("bet set to 200 coins — taken from your balance "
                      "(800 left)", reply)
        self.assertIn("🏁", reply)
        entries = [e for e in self.econ().statement(KEY, "111")["entries"]
                   if e["kind"] == "wager"]
        # seed credit, the escrow debit, and the settle credit
        self.assertEqual(entries[0]["amount"], 1000)
        self.assertEqual(entries[1]["amount"], -200)
        self.assertIn("game:slots", entries[1]["reason"])
        credit = sum(e["amount"] for e in entries[2:])
        self.assertEqual(self.econ().balance(KEY, "111"), 800 + credit)
        self.assertIn(credit, (0, 200, 400, 1000, 2000, 20000))
        # the games wallet never moves for a group wager
        self.assertEqual(self.engine.store.get_for(ADA).coins, 0)
        self.assertIsNone(self.engine.live(KEY))

    def test_group_open_table_forced_win_pays_the_estate(self):
        room, _ = self.engine.start(KEY, "slots", ADA, kind="group")
        force_reels(room, lambda r: r[0] == r[1] == r[2] == "🍒")
        reply = call_game(self.engine, "/slots 200", KEY, ADA, "group")
        self.assertIn("🪙 Ada: +200 coins (slots paid 2x on 200)", reply)
        self.assertEqual(self.econ().balance(KEY, "111"), 1200)

    def test_second_caller_gets_the_table_full_truth(self):
        self.engine.start(KEY, "slots", ADA, kind="group")
        self.surface.game_wager_credit(KEY, "222", 1000, "seed funding")
        reply = call_game(self.engine, "/slots 200", KEY, BOB, "group")
        self.assertIn("the table is full.", reply)
        # Bob's money never moved and Ada's table is untouched
        self.assertEqual(self.econ().balance(KEY, "222"), 1000)
        room = self.engine.live(KEY)
        self.assertIsNotNone(room)
        self.assertEqual(room.state["escrow"], 0)

    def test_group_insufficient_balance_leaves_no_room(self):
        reply = call_game(self.engine, "/slots 2000", KEY, ADA, "group")
        self.assertIn("not enough", reply)
        self.assertIsNone(self.engine.live(KEY))
        self.assertEqual(self.econ().balance(KEY, "111"), 1000)
        entries = [e for e in self.econ().statement(KEY, "111")["entries"]
                   if e["kind"] == "wager"]
        self.assertEqual([e["amount"] for e in entries], [1000])

    def test_unclaimed_room_gate_refuses_before_anything(self):
        reply = call_game(self.engine, "/slots 200", KEY, ADA, "group",
                          blocked=True)
        self.assertIn("aren't running in this group", reply)
        self.assertIsNone(self.engine.live(KEY))
        self.assertNotIn(KEY, self.engine._rooms)
        self.assertEqual(self.econ().balance(KEY, "111"), 1000)


# ── blackjack: the same wager-then-deal shape ───────────────────

class BlackjackQuickstartTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.db, self.sent = make_engine()
        self.engine.store.add_coins(ADA, 100, reason="seed funding")

    def tearDown(self):
        self.engine.shutdown()

    def test_amount_deals_a_hand_at_that_stake(self):
        reply = call_game(self.engine, "/blackjack 50", "test:bj1",
                          ADA, "dm")
        room = self.engine.live("test:bj1")
        self.assertIsNotNone(room)
        self.assertEqual(room.state["bet"], 50)
        self.assertEqual(room.state["escrow"], 50)
        self.assertTrue(room.state["dealt"])
        self.assertEqual(len(room.state["player"]), 2)
        self.assertEqual(self.engine.store.get_for(ADA).coins, 50)
        self.assertIn("bet: 50 coins", reply)
        self.assertIn("staked from your balance", reply)

    def test_amount_the_purse_cant_cover_refuses_cleanly(self):
        reply = call_game(self.engine, "/blackjack 500", "test:bj2",
                          ADA, "dm")
        self.assertIn("blackjack stakes 500 coins", reply)
        self.assertIsNone(self.engine.live("test:bj2"))
        self.assertEqual(self.engine.store.get_for(ADA).coins, 100)

    def test_no_amount_keeps_the_house_stake(self):
        call_game(self.engine, "/blackjack", "test:bj3", ADA, "dm")
        room = self.engine.live("test:bj3")
        self.assertIsNotNone(room)
        self.assertEqual(room.state["bet"], 10)
        self.assertEqual(room.state["escrow"], 10)
        self.assertEqual(self.engine.store.get_for(ADA).coins, 90)

    def test_junk_amount_keeps_the_house_stake(self):
        call_game(self.engine, "/blackjack abc", "test:bj4", ADA, "dm")
        room = self.engine.live("test:bj4")
        self.assertIsNotNone(room)
        self.assertEqual(room.state["bet"], 10)

    def test_parse_allows_the_amount_tail(self):
        # the arity table used to reject "/blackjack 500" at parse
        parsed = parse_control("/blackjack 500")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.kind, "blackjack")
        self.assertEqual(parsed.tail, "500")


if __name__ == "__main__":
    unittest.main()
