"""Wager integrity: real money at the games table.

Pins the fix for the owner-reported defect (2026-11-03): bets used
to be fiction — placing one wrote a number into room state, no
purse was debited, a loss CREDITED +15 "participation" coins, and
the game-over card dressed the score up as points ("-20000 pts").

Now: a bet escrows real coins from the room's bound purse the
moment it is placed (DM rooms: the games wallet; group rooms: the
group economy estate), settles from that escrow at the game's
stated odds, abandoned escrows come back, the coin sink never
double-moves wager money, progression (points/XP) is untouched,
non-wager settlement is byte-identical, and the card renders the
score AS a score with the post-settlement purse balance.
"""
from __future__ import annotations

import random
import unittest
from pathlib import Path

import pytest

from nomorals.games.engine import GameEngine, render_game_over_card
from nomorals.games.games.base import Room
from nomorals.games.games.casino import (
    BlackjackGame,
    RouletteGame,
    SlotsGame,
)
from nomorals.games.games.wild import CrapsGame, PokerGame
from nomorals.games.players import Player, PlayerStore
from nomorals.games.purse import BoundPurse, StorePurseBackend
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
    """Re-point a live slots room's (unfair) draw stream at a seed
    whose first three reels satisfy ``want`` — outcome control for
    engine-level tests."""
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


def card_of(settle_msgs):
    cards = [m for m in settle_msgs if "🏁" in m]
    assert len(cards) == 1
    return cards[0]


def settle_text(out):
    """The full text a move produced: move lines + the settle."""
    return "\n".join(out)


def lose_numberguess(engine, chat_key, player, kind):
    """Drive a deterministic numberguess loss: pick 50, then always
    guess a number that is not the house's secret (read from state)
    while its perfect bisection hunts 50 down."""
    room = engine._rooms[chat_key]
    engine.move(chat_key, "50", player, kind=kind)
    secret = int(room.state["a_secret"])
    wrong = secret % 100 + 1
    out: list[str] = []
    for _ in range(14):
        out = engine.move(chat_key, str(wrong), player, kind=kind)
        if room.state.get("done"):
            break
    assert room.state.get("done") == "house", room.state
    return out


# ── DM rooms: the games wallet is the purse ─────────────────────────

class SlotsDmTests(unittest.TestCase):
    def test_bet_debits_journals_and_settles_from_escrow(self):
        engine, db, _sent = make_engine()
        engine.store.add_coins(ADA, 500, reason="seed funding")
        room, panel = engine.start("test:sl", "slots", ADA, kind="dm")
        # the panel shows the purse balance the bet will move
        self.assertTrue(any("your balance: 500 coins" in m
                            for m in panel))
        # insufficient: refused plainly, nothing set, nothing moved
        out = engine.move("test:sl", "bet 20000", ADA, kind="dm")
        self.assertIn("not enough", out[0])
        self.assertEqual(room.state["bet"], 10)
        self.assertEqual(engine.store.get_for(ADA).coins, 500)
        # a real bet: escrowed immediately, journaled with the tag
        out = engine.move("test:sl", "bet 200", ADA, kind="dm")
        self.assertIn("bet set to 200 coins — taken from your "
                      "balance", out[0])
        self.assertEqual(engine.store.get_for(ADA).coins, 300)
        self.assertIn((-200, "game:slots slots bet"),
                      wallet_rows(db, ADA.key))
        # force a losing spin: the escrow is kept, the line is true
        force_reels(room, lambda r: len(set(r)) == 3)
        points_before = engine.store.get_for(ADA).points
        out = engine.move("test:sl", "spin", ADA, kind="dm")
        settle = settle_text(out)
        self.assertIn("no match — you lose 200 coins.", settle)
        self.assertIn("🪙 Ada: -200 coins (slots wager lost)", settle)
        self.assertEqual(engine.store.get_for(ADA).coins, 300)
        # progression is NOT money: points still paid on the loss
        self.assertGreater(engine.store.get_for(ADA).points,
                           points_before)
        # the card: score rendered as score, balance post-settlement
        self.assertIn("-200 score", settle)
        self.assertIn("300c", settle)
        card = settle.split("🏁", 1)[1]
        self.assertNotIn(" pts", card)

    def test_two_match_returns_the_bet(self):
        engine, db, _sent = make_engine()
        engine.store.add_coins(ADA, 100, reason="seed funding")
        room, _ = engine.start("test:sl", "slots", ADA, kind="dm")
        engine.move("test:sl", "bet 40", ADA, kind="dm")
        self.assertEqual(engine.store.get_for(ADA).coins, 60)
        force_reels(room, lambda r: len(set(r)) == 2)
        out = engine.move("test:sl", "spin", ADA, kind="dm")
        settle = settle_text(out)
        self.assertIn("two matching — bet returned.", settle)
        self.assertIn("🪙 Ada: +0 coins (slots — bet returned)",
                      settle)
        self.assertEqual(engine.store.get_for(ADA).coins, 100)

    def test_spin_stakes_the_standing_bet(self):
        engine, db, _sent = make_engine()
        engine.store.add_coins(ADA, 100, reason="seed funding")
        room, _ = engine.start("test:sl", "slots", ADA, kind="dm")
        force_reels(room, lambda r: r[0] == r[1] == r[2] == "🍒")
        out = engine.move("test:sl", "spin", ADA, kind="dm")
        settle = settle_text(out)
        # default bet 10 staked at spin, cherries pay 2x: credit 20
        self.assertIn((-10, "game:slots slots bet"),
                      wallet_rows(db, ADA.key))
        self.assertIn((20, "game:slots slots payout"),
                      wallet_rows(db, ADA.key))
        self.assertEqual(engine.store.get_for(ADA).coins, 110)
        self.assertIn("🪙 Ada: +10 coins (slots paid 2x on 10)",
                      settle)

    def test_broke_spin_is_refused_and_moves_nothing(self):
        engine, db, _sent = make_engine()
        room, _ = engine.start("test:sl", "slots", ADA, kind="dm")
        out = engine.move("test:sl", "spin", ADA, kind="dm")
        self.assertIn("not enough", out[0])
        self.assertFalse(room.state["done"])
        self.assertEqual(engine.store.get_for(ADA).coins, 0)
        self.assertEqual(wallet_rows(db, ADA.key), [])

    def test_bet_then_leave_refunds_the_escrow(self):
        engine, db, _sent = make_engine()
        engine.store.add_coins(ADA, 100, reason="seed funding")
        engine.start("test:sl", "slots", ADA, kind="dm")
        engine.move("test:sl", "bet 70", ADA, kind="dm")
        self.assertEqual(engine.store.get_for(ADA).coins, 30)
        out = engine.leave("test:sl", ADA)
        self.assertTrue(any("70-coin slots bet was refunded" in m
                            for m in out))
        self.assertEqual(engine.store.get_for(ADA).coins, 100)


class CasinoDmTests(unittest.TestCase):
    def test_blackjack_start_requires_the_stake(self):
        engine, db, _sent = make_engine()
        with self.assertRaises(ValueError) as cm:
            engine.start("test:bj", "blackjack", BOB, kind="dm")
        self.assertIn("blackjack stakes 10 coins", str(cm.exception))
        self.assertNotIn("test:bj", engine._rooms)
        self.assertEqual(engine.store.get_for(BOB).coins, 0)

    def test_blackjack_quit_refunds_the_stake(self):
        engine, db, _sent = make_engine()
        engine.store.add_coins(ADA, 100, reason="seed funding")
        engine.start("test:bj", "blackjack", ADA, kind="dm")
        self.assertEqual(engine.store.get_for(ADA).coins, 90)
        out = engine.quit("test:bj")
        settle = card_of(out)
        self.assertIn("blackjack bet refunded — table closed", settle)
        self.assertEqual(engine.store.get_for(ADA).coins, 100)

    def test_roulette_bet_escrows_and_quit_refunds(self):
        engine, db, _sent = make_engine()
        engine.store.add_coins(ADA, 100, reason="seed funding")
        room, panel = engine.start("test:ro", "roulette", ADA,
                                   kind="dm")
        self.assertTrue(any("your balance: 100 coins" in m
                            for m in panel))
        out = engine.move("test:ro", "bet red", ADA, kind="dm")
        self.assertIn("taken from your balance", out[0])
        self.assertEqual(engine.store.get_for(ADA).coins, 90)
        out = engine.quit("test:ro")
        settle = card_of(out)
        self.assertIn("roulette bet refunded — table closed", settle)
        self.assertEqual(engine.store.get_for(ADA).coins, 100)

    def test_craps_buyin_and_quit_cashout(self):
        engine, db, _sent = make_engine()
        engine.store.add_coins(ADA, 500, reason="seed funding")
        engine.start("test:cr", "craps", ADA, kind="dm")
        self.assertEqual(engine.store.get_for(ADA).coins, 400)
        self.assertIn((-100, "game:craps craps buy-in"),
                      wallet_rows(db, ADA.key))
        engine.quit("test:cr")
        self.assertEqual(engine.store.get_for(ADA).coins, 500)

    def test_poker_buyin_and_quit_cashout(self):
        engine, db, _sent = make_engine()
        engine.store.add_coins(ADA, 500, reason="seed funding")
        engine.start("test:pk", "poker", ADA, kind="dm")
        self.assertEqual(engine.store.get_for(ADA).coins, 300)
        # mid-hand quit: stack 196 + the 4 big-blind committed = 200
        engine.quit("test:pk")
        self.assertEqual(engine.store.get_for(ADA).coins, 500)

    def test_nonwager_loss_settle_is_byte_identical(self):
        # numberguess: a clean forced loss (see lose_numberguess) —
        # the settle must read exactly as it did before this build.
        engine, db, _sent = make_engine()
        engine.start("test:ng", "numberguess", ADA, kind="dm")
        out = lose_numberguess(engine, "test:ng", ADA, "dm")
        settle = settle_text(out)
        # the participation crumb, exactly as before this build
        self.assertIn("🪙 Ada: +15 coins (participation)", settle)
        self.assertIn("📈 points and coins credited — /game "
                      "leaderboard", settle)
        self.assertEqual(engine.store.get_for(ADA).coins, 15)
        # non-wager coins are not purse movements — they arrive via
        # record_outcome, as before, and never touch the wallet journal
        self.assertEqual(wallet_rows(db, ADA.key), [])


# ── unit level: the stated odds, exactly ────────────────────────────

def bare_room(game_name, game, store, db, players=None, seed=1):
    state = game.new_state(random.Random(0))
    state["fair"] = None
    room = Room(id="u1", game=game_name, chat_key="u:1",
                platform="local", kind="dm",
                players=players or [ADA], state=state, seed=seed)
    room.purse = BoundPurse(StorePurseBackend(store, db), room)
    return room


class OddsUnitTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.db.migrate()
        self.store = PlayerStore(self.db)
        self.store.add_coins(ADA, 100, reason="seed funding")

    def bal(self):
        return self.store.get_for(ADA).coins

    def test_slots_pays_table(self):
        game = SlotsGame()
        cases = [
            ("🍒", 2),    # cherries 2x
            ("🔔", 5),    # bells 5x
            ("7️⃣", 10),  # sevens 10x
            ("💎", 100),  # jackpot 100x
            ("🍋", 3),    # any other triple 3x
        ]
        for sym, mult in cases:
            room = bare_room("slots", game, self.store, self.db)
            # stub the reel draw itself: exact triples (the diamond
            # at weight 1/100) are not seed-searchable in range.
            game._fair_reel = lambda room, i, s=sym: s
            game.on_move(room, ADA, "bet 10", None)
            self.assertEqual(self.bal(), 90)  # stake escrowed
            before = self.bal()
            game.on_move(room, ADA, "spin", None)
            net, _why = game.wager_settle(room, ADA)
            self.assertEqual(net, 10 * (mult - 1))
            self.assertEqual(self.bal(), before + 10 * mult)
            # reset for the next case
            self.store.spend_coins(ADA, self.bal() - 100,
                                   reason="reset")

    def test_slots_two_match_and_loss(self):
        game = SlotsGame()
        room = bare_room("slots", game, self.store, self.db)
        force_reels(room, lambda r: len(set(r)) == 2)
        game.on_move(room, ADA, "bet 25", None)
        game.on_move(room, ADA, "spin", None)
        net, _ = game.wager_settle(room, ADA)
        self.assertEqual(net, 0)
        self.assertEqual(self.bal(), 100)
        room = bare_room("slots", game, self.store, self.db)
        force_reels(room, lambda r: len(set(r)) == 3)
        game.on_move(room, ADA, "bet 25", None)
        game.on_move(room, ADA, "spin", None)
        net, _ = game.wager_settle(room, ADA)
        self.assertEqual(net, -25)
        self.assertEqual(self.bal(), 75)

    def _roulette_room(self, landed):
        game = RouletteGame()
        for seed in range(5000):
            if random.Random(seed).randint(0, 36) == landed:
                room = bare_room("roulette", game, self.store,
                                 self.db, seed=seed)
                return game, room
        raise AssertionError("no seed")

    def test_roulette_stated_odds(self):
        # number pays 35:1 -> total credit 36x the stake
        game, room = self._roulette_room(7)
        game.on_move(room, ADA, "bet number 7", None)
        self.assertEqual(self.bal(), 90)
        game.on_move(room, ADA, "spin", None)
        net, _ = game.wager_settle(room, ADA)
        self.assertEqual(net, 350)
        self.assertEqual(self.bal(), 450)
        self.store.spend_coins(ADA, 350, reason="reset")
        # color pays 1:1 -> credit 2x
        game, room = self._roulette_room(3)  # 3 is red
        game.on_move(room, ADA, "bet red", None)
        game.on_move(room, ADA, "spin", None)
        net, _ = game.wager_settle(room, ADA)
        self.assertEqual(net, 10)
        self.assertEqual(self.bal(), 110)
        self.store.spend_coins(ADA, 10, reason="reset")
        # dozen pays 2:1 -> credit 3x
        game, room = self._roulette_room(20)  # second dozen
        game.on_move(room, ADA, "bet dozen 2", None)
        game.on_move(room, ADA, "spin", None)
        net, _ = game.wager_settle(room, ADA)
        self.assertEqual(net, 20)
        self.assertEqual(self.bal(), 120)
        self.store.spend_coins(ADA, 20, reason="reset")
        # a miss keeps the stake
        game, room = self._roulette_room(0)
        game.on_move(room, ADA, "bet red", None)
        game.on_move(room, ADA, "spin", None)
        net, _ = game.wager_settle(room, ADA)
        self.assertEqual(net, -10)
        self.assertEqual(self.bal(), 90)

    def _blackjack_room(self, player_cards, dealer_cards, deck):
        game = BlackjackGame()
        room = bare_room("blackjack", game, self.store, self.db,
                         players=[ADA, Player(
                             key="ai:dealer", platform="ai",
                             name="The Dealer", is_ai=True)])
        s = room.state
        # stake the hand the way setup does
        assert room.purse.debit(ADA, 10, "blackjack bet")
        s["escrow"] = 10
        s["dealt"] = True
        s["player"] = list(player_cards)
        s["dealer"] = list(dealer_cards)
        s["dealer_start"] = list(dealer_cards)
        s["deck"] = list(deck)
        if game._hand_value(s["player"]) == 21 and len(player_cards) == 2:
            s["natural"] = True
        return game, room

    def test_blackjack_win_push_loss(self):
        # win 1:1 -> credit 2x stake
        game, room = self._blackjack_room([10, 13], [10, 8], [5] * 8)
        game.on_move(room, ADA, "stand", None)
        net, _ = game.wager_settle(room, ADA)
        self.assertEqual((net, self.bal()), (10, 110))
        self.store.spend_coins(ADA, 10, reason="reset")
        # push -> stake back
        game, room = self._blackjack_room([10, 13], [13, 10], [5] * 8)
        game.on_move(room, ADA, "stand", None)
        net, _ = game.wager_settle(room, ADA)
        self.assertEqual((net, self.bal()), (0, 100))
        # loss -> stake kept
        game, room = self._blackjack_room([10, 8], [10, 13], [5] * 8)
        game.on_move(room, ADA, "stand", None)
        net, _ = game.wager_settle(room, ADA)
        self.assertEqual((net, self.bal()), (-10, 90))
        self.store.add_coins(ADA, 10, reason="reset")

    def test_blackjack_natural_pays_three_to_two(self):
        game, room = self._blackjack_room([14, 13], [10, 6], [10] * 8)
        game.on_move(room, ADA, "stand", None)  # dealer 16 hits, busts
        net, why = game.wager_settle(room, ADA)
        self.assertEqual(net, 15)  # 3:2 on 10
        self.assertEqual(self.bal(), 115)

    def test_blackjack_double_stakes_and_settles_double(self):
        game, room = self._blackjack_room([5, 6], [10, 7], [10] * 8)
        game.on_move(room, ADA, "double", None)  # draws 10 -> 21
        self.assertEqual(self.bal(), 80)  # second stake escrowed
        net, _ = game.wager_settle(room, ADA)
        self.assertEqual((net, self.bal()), (20, 120))

    def test_blackjack_split_settles_both_hands(self):
        # pair of 8s; deck pops: hand1 +10 (18), hand2 +10 (18);
        # dealer stands on 17 — both hands win 1:1.
        game, room = self._blackjack_room([8, 8], [10, 7],
                                          [5, 5, 10, 10])
        game.on_move(room, ADA, "split", None)
        self.assertEqual(self.bal(), 80)  # second stake escrowed
        game.on_move(room, ADA, "stand", None)
        self.assertEqual(room.state["result2"], "win")
        net, _ = game.wager_settle(room, ADA)
        self.assertEqual((net, self.bal()), (20, 120))

    def test_craps_cashout_and_leave(self):
        game = CrapsGame()
        room = bare_room("craps", game, self.store, self.db)
        s = room.state
        assert room.purse.debit(ADA, 100, "craps buy-in")
        s["buyin"][ADA.key] = 100
        s["bank"][ADA.key] = 130
        s["done"] = True
        net, _ = game.wager_settle(room, ADA)
        self.assertEqual((net, self.bal()), (30, 130))
        # leaving mid-table cashes the live bank out
        room = bare_room("craps", game, self.store, self.db)
        s = room.state
        assert room.purse.debit(ADA, 100, "craps buy-in")
        s["buyin"][ADA.key] = 100
        s["bank"][ADA.key] = 80
        note = game.wager_leave(room, ADA)
        self.assertIn("cashes out 80 chips", note)
        self.assertEqual(self.bal(), 110)
        self.assertNotIn(ADA.key, s["bank"])

    def test_poker_cashout_midhand_returns_committed(self):
        game = PokerGame()
        room = bare_room("poker", game, self.store, self.db)
        s = room.state
        self.store.add_coins(ADA, 200, reason="top up")  # 300 total
        assert room.purse.debit(ADA, 200, "poker buy-in")
        self.assertEqual(self.bal(), 100)
        s["buyin"][ADA.key] = 200
        s["stacks"][ADA.key] = 150
        s["committed"][ADA.key] = 30
        net, _ = game.wager_settle(room, ADA)
        self.assertEqual(net, -20)  # 150 + 30 back on a 200 buy-in
        self.assertEqual(self.bal(), 280)


class CardTests(unittest.TestCase):
    def test_card_renders_score_and_purse_balance(self):
        db = Database(":memory:")
        db.migrate()
        store = PlayerStore(db)
        card = render_game_over_card(
            "slots", [ADA], {ADA.key: -20000}, store=store, db=db,
            balances={ADA.key: 300})
        self.assertIn("-20000 score", card)
        self.assertIn("300c", card)
        self.assertNotIn("pts", card)


# ── the estate organ: wager movements journal, caps untouched ──────

def _surface(tmp_path: Path) -> GroupSurface:
    store = GroupStore(tmp_path / "groups.json").load()
    return GroupSurface(store)


def test_estate_wager_debit_credit_and_caps(tmp_path):
    surface = _surface(tmp_path)
    econ = surface.economy
    assert surface.game_wager_available(KEY)
    # insufficient debit moves nothing
    assert not surface.game_wager_debit(KEY, "111", 50, "game:slots x")
    assert econ.balance(KEY, "111") == 0
    surface.game_wager_credit(KEY, "111", 500, "seed")
    assert surface.game_wager_balance(KEY, "111") == 500
    assert surface.game_wager_debit(KEY, "111", 200, "game:slots bet",
                                    ref="wager:slots:r1")
    assert econ.balance(KEY, "111") == 300
    entries = [e for e in econ.statement(KEY, "111")["entries"]
               if e["kind"] == "wager"]
    assert [e["amount"] for e in entries] == [500, -200]
    assert entries[1]["ref"] == "wager:slots:r1"
    # wager credits are not earns: the daily earn cap is untouched
    entry = econ.award(KEY, "111", 200, "still fine", now=1_800_000_000.0)
    assert entry is not None


# ── group rooms: the estate is the purse, the sink stays out ───────

def make_group_engine(tmp_path):
    from nomorals.agents.partner.runtime_group import _GroupWagerPurse

    surface = _surface(tmp_path)
    engine, db, _sent = make_engine()
    engine.purse_source = (
        lambda room: _GroupWagerPurse(surface, room.chat_key)
        if room.kind == "group" else None)
    sink_calls: list = []
    engine.coin_sink = lambda ck, p, c, g: sink_calls.append((ck, c, g))
    return engine, db, surface, sink_calls


def test_group_slots_lifecycle_against_the_estate(tmp_path):
    engine, db, surface, sink_calls = make_group_engine(tmp_path)
    econ = surface.economy
    surface.game_wager_credit(KEY, "111", 1000, "seed funding")
    room, panel = engine.start(KEY, "slots", ADA, kind="group")
    # the panel's balance is the estate balance
    assert any("your balance: 1000 coins" in m for m in panel)
    assert any("group purse" in m for m in panel)
    out = engine.move(KEY, "bet 200", ADA, kind="group")
    assert "taken from your balance" in out[0]
    assert econ.balance(KEY, "111") == 800
    # the games wallet never moves for a group wager
    assert engine.store.get_for(ADA).coins == 0
    force_reels(room, lambda r: r[0] == r[1] == r[2] == "🍒")
    out = engine.move(KEY, "spin", ADA, kind="group")
    settle = card_of(out)
    # cherries 2x on 200: credit 400 -> estate 1200, net +200
    assert econ.balance(KEY, "111") == 1200
    assert "🪙 Ada: +200 coins (slots paid 2x on 200)" in settle
    assert "1200c" in settle  # card balance is post-settlement estate
    # the estate journal holds exactly the two wager legs
    entries = [e for e in econ.statement(KEY, "111")["entries"]
               if e["kind"] == "wager"]
    assert [e["amount"] for e in entries] == [1000, -200, 400]
    assert all("game:slots" in e["reason"] for e in entries[1:])
    # the sink never double-moves wager money
    assert sink_calls == []
    assert engine.store.get_for(ADA).coins == 0


def test_group_blackjack_start_checks_the_estate(tmp_path):
    engine, db, surface, sink_calls = make_group_engine(tmp_path)
    surface.game_wager_credit(KEY, "111", 5, "seed funding")
    with pytest.raises(ValueError):
        engine.start(KEY, "blackjack", ADA, kind="group")
    assert surface.economy.balance(KEY, "111") == 5
    assert sink_calls == []


def test_group_nonwager_coins_still_flow_through_the_sink(tmp_path):
    engine, db, surface, sink_calls = make_group_engine(tmp_path)
    engine.start(KEY, "numberguess", ADA, kind="group")
    lose_numberguess(engine, KEY, ADA, "group")
    # numberguess is not a wager game: its +15 participation coins
    # still mirror into the group purse through the sink, once
    assert sink_calls == [(KEY, 15, "numberguess")]


if __name__ == "__main__":
    unittest.main()
