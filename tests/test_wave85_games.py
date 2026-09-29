"""Wave 85 — the multiplayer game engine: every game, the engine itself,
profiles, the economy, the scheduler, and cross-platform simultaneity.

Everything is hermetic: in-memory database, no model (``suggest=None``),
fixed player keys, scripted moves.  A game that can't be played by a
script can't be played by a human either, so this file is also the
acceptance proof that all 19 games actually run to a finish.
"""
from __future__ import annotations

import threading
import time
import unittest

from nomorals.storage.db import Database
from nomorals.games import (
    GameEngine,
    GameMind,
    Player,
    PlayerStore,
)
from nomorals.games.economy import GameEconomy
from nomorals.games.players import Leaderboard
from nomorals.games.games.easy import (
    AuctionGame,
    HangmanGame,
    NumberGuessGame,
    SpyGame,
    TriviaRoyaleGame,
    TwoTruthsGame,
    WordChainGame,
    WyrrGame,
)
from nomorals.games.games.medium import (
    InvestigationGame,
    KingOfHillGame,
    MafiaGame,
    QuizDuelGame,
    RpgAdventureGame,
    ShopGame,
    StoryChainGame,
)
from nomorals.games.games.ambitious import (
    BattleArenaGame,
    EscapeRoomGame,
    PoliticalGame,
    WorldGame,
)


class Ctx:
    """Bare context: a db and nothing else (the engine never needs more)."""

    def __init__(self, db: Database) -> None:
        self.db = db


def make_engine():
    db = Database(":memory:")
    db.migrate()
    sent: list[str] = []
    engine = GameEngine(Ctx(db), send=lambda chat, text: sent.append(text))
    return engine, sent


ADA = Player.from_sender("telegram", "456", "Ada")
BOB = Player.from_sender("telegram", "789", "Bob")
CYR = Player.from_sender("telegram", "321", "Cyr")


def finished(room) -> bool:
    """A finished room is popped from the live map — None is done too."""
    return room is None or room.status == "finished"


def drive(engine, chat: str, game: str, host, *, kind: str = "dm",
          responder, max_moves: int = 80):
    """Start a room and feed it human moves until it finishes.

    ``responder(room, state, phase_info) -> str | None`` produces the
    next human move for the current human seat (None = no move this
    pass). Returns (final_room, all_messages).
    """
    room, msgs = engine.start(chat, game, host, kind=kind)
    out = list(msgs)
    for _ in range(max_moves):
        live = engine.live(chat)
        if live is None or live.status != "active":
            return live, out
        cur = live.current
        if cur is None or cur.is_ai:
            # the pump runs inside move(); a stuck AI seat is a bug
            raise AssertionError(f"stuck on AI seat {cur}")
        text = responder(live, live.state)
        if text is None:
            raise AssertionError("responder produced no move")
        out.extend(engine.move(chat, text, cur))
    raise AssertionError(f"{game} did not finish in {max_moves} moves")


# ── engine & lifecycle ───────────────────────────────────────────────────────

class EngineLifecycleTests(unittest.TestCase):
    def test_migration_tables_exist(self):
        db = Database(":memory:")
        db.migrate()
        tables = {r["name"] for r in db.query(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        for t in ("game_rooms", "game_players", "game_wallet"):
            self.assertIn(t, tables)

    def test_no_thread_leak_on_shutdown(self):
        before = {t.name for t in threading.enumerate()}
        engine, _ = make_engine()
        self.assertTrue(any("game-turns" in n for n in
                            {t.name for t in threading.enumerate()}))
        engine.shutdown()
        time.sleep(0.2)
        after = {t.name for t in threading.enumerate()}
        self.assertFalse({"nm-game-turns"} & (after - before))

    def test_room_survives_restart(self):
        db = Database(":memory:")
        db.migrate()
        engine = GameEngine(Ctx(db), send=lambda c, t: None)
        engine.start("telegram:1", "trivia", ADA, kind="dm")
        room = engine.live("telegram:1")
        self.assertIsNotNone(room)
        out = engine.move("telegram:1", "2007", ADA)
        engine.shutdown()
        engine2 = GameEngine(Ctx(db), send=lambda c, t: None)
        restored = engine2.live("telegram:1")
        self.assertIsNotNone(restored, "room did not survive restart")
        self.assertEqual(restored.game, "trivia")
        self.assertEqual(restored.state["round"], room.state["round"])
        engine2.shutdown()

    def test_unknown_game_rejected(self):
        engine, _ = make_engine()
        with self.assertRaises(ValueError):
            engine.start("telegram:1", "chess", ADA)
        engine.shutdown()

    def test_group_only_game_refuses_dm(self):
        engine, _ = make_engine()
        with self.assertRaises(ValueError):
            engine.start("telegram:1", "mafia", ADA, kind="dm")
        engine.shutdown()

    def test_second_game_refused_while_live(self):
        engine, _ = make_engine()
        engine.start("telegram:1", "wordchain", ADA)
        with self.assertRaises(ValueError):
            engine.start("telegram:1", "trivia", ADA)
        engine.shutdown()

    def test_cross_platform_simultaneous_rooms(self):
        engine, _ = make_engine()
        engine.start("telegram:1", "wordchain", ADA, kind="dm")
        engine.start("discord:9", "duel",
                     Player.from_sender("discord", "456", "Ada"), kind="dm")
        self.assertEqual(len(engine.rooms()), 2)
        out = engine.move("telegram:1", "yellow", ADA)
        self.assertTrue(any("yellow" in m for m in out))
        self.assertEqual(len(engine.rooms()), 2,
                         "one platform's move touched the other's room")
        engine.shutdown()

    def test_quit_credits_ledger(self):
        engine, _ = make_engine()
        engine.start("telegram:1", "trivia", ADA)
        engine.move("telegram:1", "2007", ADA)  # correct: Q1 answer
        engine.quit("telegram:1")
        self.assertEqual(engine.rooms(), [])
        prof = engine.store.get(ADA.key)
        self.assertEqual(prof.games_played, 1)

    def test_command_status_and_help(self):
        engine, _ = make_engine()
        engine.start("telegram:1", "wordchain", ADA)
        status = engine.move("telegram:1", "/status", ADA)
        self.assertIn("wordchain", status[0])
        help_ = engine.move("telegram:1", "/help", ADA)
        self.assertIn("last letter", help_[0])


# ── easy games ───────────────────────────────────────────────────────────────

class EasyGamesTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.sent = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def test_wordchain_plays_to_a_finish(self):
        WORDS = ["yellow", "owl", "willow", "octopus", "spoon", "novel",
                 "lens", "snake", "eagle", "orchid", "dwarf", "fly",
                 "yolk", "kite", "tiger"]

        def responder(room, state):
            last = state["last"][-1]
            for w in WORDS:
                if w.startswith(last) and w not in state["used"]:
                    return w
            # exhausted the bag: miss on purpose and lose cleanly
            return "zzzz"

        room, msgs = drive(self.engine, "telegram:1", "wordchain", ADA,
                           responder=responder)
        self.assertTrue(finished(room))
        self.assertTrue(any("🏁" in m for m in msgs))
        self.assertEqual(self.engine.rooms(), [])

    def test_wordchain_rejects_bad_starts(self):
        self.engine.start("telegram:1", "wordchain", ADA)
        out = self.engine.move("telegram:1", "apple", ADA)
        self.assertTrue(any("last letter" in m or "start with" in m
                            for m in out))

    def test_hangman_cracked_by_the_table(self):
        def responder(r, state):
            for letter in set(state["word"]):
                if letter not in state["revealed"]:
                    return letter
            return "word " + state["word"]

        room, msgs = drive(self.engine, "telegram:1", "hangman", ADA,
                           responder=responder)
        self.assertTrue(finished(room))
        self.assertTrue(any("cracked" in m for m in msgs))

    def test_numberguess_bisection_ends_cleanly(self):
        engine, chat = self.engine, "telegram:1"
        engine.start(chat, "numberguess", ADA)
        engine.move(chat, "50", ADA)  # pick my secret
        for _ in range(40):
            live = engine.live(chat)
            if live is None or live.status != "active":
                break
            cur = live.current
            if cur is None or cur.is_ai:
                raise AssertionError(f"stuck on {cur}")
            engine.move(chat, str((live.state["a_lo"] +
                                   live.state["a_hi"]) // 2), cur)
        self.assertTrue(any("🏁" in m for m in self.sent))

    def test_two_truths_three_rounds(self):
        def responder(r, state):
            if state["phase"] == "posting":
                return "I like cats\nI never swim\nI can juggle"
            if state["phase"] == "voting":
                return "1"
            if state["phase"] == "reveal":
                return "3"
            return "1"

        room, msgs = drive(self.engine, "telegram:1", "two_truths", ADA,
                           responder=responder, max_moves=40)
        self.assertTrue(finished(room))
        self.assertTrue(any("reveal" in m.lower() for m in msgs))

    def test_wyrr_five_rounds(self):
        room, msgs = drive(self.engine, "telegram:1", "wyrr", ADA,
                           responder=lambda r, s: "1")
        self.assertTrue(finished(room))
        self.assertTrue(any("five rounds" in m.lower() for m in msgs))

    def test_spy_bluffs_with_the_associate(self):
        engine, chat = self.engine, "telegram:1"
        engine.start(chat, "spy", ADA)
        for _ in range(12):
            live = engine.live(chat)
            if live is None or live.status != "active":
                break
            cur = live.current
            if cur is None or cur.is_ai:
                raise AssertionError(f"stuck on {cur}")
            engine.move(chat, live.state["current_assoc"], cur)
        self.assertTrue(any("🏁" in m for m in self.sent))

    def test_auction_three_lots(self):
        def responder(r, state):
            if state["leader"] > 100:
                return "pass"
            return f"bid {state['leader'] + 20}"

        room, msgs = drive(self.engine, "telegram:1", "auction", ADA,
                           responder=responder, max_moves=60)
        self.assertTrue(finished(room))
        self.assertTrue(any("lot 3/3" in m or "sold to" in m for m in msgs))

    def test_trivia_royale_always_right(self):
        def responder(r, state):
            return state["a"]

        room, msgs = drive(self.engine, "telegram:1", "trivia", ADA,
                           responder=responder, max_moves=40)
        self.assertTrue(finished(room))
        prof = self.engine.store.get(ADA.key)
        self.assertGreaterEqual(prof.points, 50)

    def test_game_registry_has_all_nineteen(self):
        from nomorals.games.games.easy import EASY_GAMES
        from nomorals.games.games.medium import MEDIUM_GAMES
        from nomorals.games.games.ambitious import AMBITIOUS_GAMES
        names = {g.name for g in (*EASY_GAMES, *MEDIUM_GAMES, *AMBITIOUS_GAMES)}
        expected = {
            "wordchain", "hangman", "numberguess", "two_truths", "wyrr",
            "spy", "auction", "trivia",
            "mafia", "king", "story", "rpg", "shop", "duel", "case",
            "world", "arena", "escape", "political",
        }
        self.assertEqual(names, expected)


# ── medium games ─────────────────────────────────────────────────────────────

class MediumGamesTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.sent = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def test_king_of_the_hill(self):
        def responder(r, state):
            return state["challenges"][state["round"]][1]

        room, msgs = drive(self.engine, "telegram:1", "king", ADA,
                           responder=responder, max_moves=60)
        self.assertTrue(finished(room))
        self.assertTrue(any("hill goes to" in m for m in msgs))

    def test_story_chain_two_loops(self):
        room, msgs = drive(
            self.engine, "telegram:1", "story", ADA,
            responder=lambda r, s: "And the door, at last, remembered "
                                   "whose hand had closed it.")
        self.assertTrue(finished(room))
        self.assertTrue(any("story is done" in m.lower() for m in msgs))

    def test_rpg_sixteen_scenes(self):
        """The full 16-scene campaign, played to its finish.

        Safe play: the choice with the largest (least negative) HP
        delta, potion when low — reckless 'd' gambles used to end the
        run before the finale, and a dead hero stalls the campaign.
        """
        from nomorals.games.games.medium import RPG_SCENES

        def responder(r, s):
            sh = s["sheets"][ADA.key]
            if sh["hp"] < 12 and sh["potions"] > 0:
                return "potion"
            choices = RPG_SCENES[s["scene"] % len(RPG_SCENES)]["choices"]
            return "123"[max(range(3), key=lambda i: choices[i][4])]
        room, msgs = drive(self.engine, "telegram:1", "rpg", ADA,
                           responder=responder, max_moves=40)
        self.assertTrue(finished(room))
        self.assertTrue(any("campaign ends" in m for m in msgs))

    def test_rpg_potion_heals(self):
        self.engine.start("telegram:1", "rpg", ADA)
        room = self.engine.live("telegram:1")
        out = self.engine.move("telegram:1", "potion", ADA)
        self.assertTrue(any("potion" in m.lower() for m in out))

    def test_shop_ten_rounds_waiting(self):
        room, msgs = drive(self.engine, "telegram:1", "shop", ADA,
                           responder=lambda r, s: "wait", max_moves=40)
        self.assertTrue(finished(room))
        self.assertTrue(any("final ledgers" in m for m in msgs))

    def test_shop_buy_and_sell(self):
        self.engine.start("telegram:1", "shop", ADA)
        room = self.engine.live("telegram:1")
        # buy whatever is in the 100c starting wallet — a 1.4x spike on
        # silk thread (110c fair) can price slot 1 out of reach
        idx = next(i + 1 for i, (_, price)
                   in enumerate(room.state["market"]) if price <= 100)
        out = self.engine.move("telegram:1", f"buy {idx}", ADA)
        self.assertTrue(any("bought" in m for m in out))
        # sell it back in a later round when a buyer has it
        for _ in range(12):
            room = self.engine.live("telegram:1")
            if room is None:
                break
            for i, (name, price) in enumerate(room.state["buyers"], 1):
                if name in room.state["goods"][ADA.key]:
                    out = self.engine.move("telegram:1", f"sell {i}", ADA)
                    self.assertTrue(any("sold" in m for m in out))
        self.engine.quit("telegram:1")

    def test_quiz_duel_rapid_fire(self):
        def responder(r, state):
            return state["questions"][state["idx"]][1]

        room, msgs = drive(self.engine, "telegram:1", "duel", ADA,
                           responder=responder, max_moves=40)
        self.assertTrue(finished(room))

    def test_duel_timeout_passes_the_turn(self):
        self.engine.start("telegram:1", "duel", ADA)
        room = self.engine.live("telegram:1")
        room.turn_started = time.time() - 60  # the clock ran out
        self.engine._sweep_timeouts()
        self.assertIn("clock", self.sent[-1].lower() + " ".join(
            self.sent[-4:]).lower())

    def test_case_cracked_with_clues(self):
        # accuse as soon as the evidence is decent — the house AI gets
        # its own accusations, and leaving it more turns only lets it
        # steal the case with three wrong names (a legal outcome)
        def responder(r, state):
            if state["clues_shown"] < 3:
                return "clue"
            return "accuse " + state["case"]["culprit"]

        room, msgs = drive(self.engine, "telegram:1", "case", ADA,
                           responder=responder, max_moves=30)
        self.assertTrue(finished(room))
        self.assertTrue(any("closes the case" in m for m in msgs))

    def test_case_three_strikes_escapes(self):
        self.engine.start("telegram:1", "case", ADA)
        room = self.engine.live("telegram:1")
        wrong = [s for s in room.state["case"]["suspects"]
                 if s != room.state["case"]["culprit"]][0]
        msgs: list[str] = []
        for _ in range(3):
            msgs.extend(self.engine.move("telegram:1", f"accuse {wrong}", ADA))
        self.assertEqual(self.engine.live("telegram:1"), None)
        self.assertTrue(any("walks" in m for m in msgs))

    def test_mafia_group_plays_to_a_finish(self):
        engine = self.engine
        chat = "telegram:group1"
        engine.start(chat, "mafia", ADA, kind="group", platform="telegram")
        for p in (BOB, CYR):
            engine.join(chat, p)
        talk = iter(["the well dried up oddly", "i trust the table",
                     "who was last at the vault?", "interesting timing",
                     "i saw nothing", "the locks matter"])

        def responder(room, state):
            if state["phase"] == "talk":
                return next(talk, "hm")
            if state["phase"] == "vote":
                return "skip"
            return "skip"

        done = False
        for _ in range(80):
            live = engine.live(chat)
            if live is None or live.status != "active":
                done = True
                break
            cur = live.current
            if cur is None or cur.is_ai:
                raise AssertionError(f"stuck on {cur}")
            engine.move(chat, responder(live, live.state), cur)
        self.assertTrue(done, "mafia did not finish")
        self.assertTrue(any("🏁" in m for m in self.sent))
        self.assertTrue(any("🏁" in m for m in self.sent))

    def test_political_three_elections(self):
        engine = self.engine
        chat = "telegram:group2"
        engine.start(chat, "political", ADA, kind="group", platform="telegram")
        engine.join(chat, BOB)
        pledges = iter(["pledge 1", "pledge 2", "pledge 3", "pledge 4"])

        def responder(room, state):
            if state["phase"] == "campaign":
                return next(pledges, "pledge 5")
            return "Bob"

        done = False
        msgs: list[str] = []
        for _ in range(80):
            live = engine.live(chat)
            if live is None or live.status != "active":
                done = True
                break
            cur = live.current
            if cur is None or cur.is_ai:
                raise AssertionError(f"stuck on {cur}")
            msgs.extend(engine.move(chat, responder(live, live.state), cur))
        self.assertTrue(done, "political did not finish")
        self.assertTrue(any("mayor" in m for m in msgs))


# ── ambitious games ──────────────────────────────────────────────────────────

class AmbitiousGamesTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.sent = make_engine()

    def tearDown(self):
        self.engine.shutdown()

    def test_world_town_grows_and_persists(self):
        self.engine.start("telegram:1", "world", ADA)
        msgs: list[str] = []
        for action in ("farm", "mine", "build shed", "farm", "rest"):
            msgs.extend(self.engine.move("telegram:1", action, ADA))
        room = self.engine.live("telegram:1")
        self.assertIsNotNone(room, "the town never ends")
        self.assertGreaterEqual(room.state["day"], 6)
        self.assertGreaterEqual(room.state["buildings"].get("shed", 0), 1)
        # the report keeps the table informed
        self.assertTrue(any("day " in m for m in msgs))
        self.engine.quit("telegram:1")
        self.assertEqual(self.engine.rooms(), [])

    def test_world_famine_when_food_runs_out(self):
        self.engine.start("telegram:1", "world", ADA)
        for _ in range(30):
            self.engine.move("telegram:1", "mine", ADA)
        room = self.engine.live("telegram:1")
        self.assertLess(room.state["food"], 30)

    def test_arena_to_the_death(self):
        def responder(r, state):
            if state["you"]["hp"] < 15 and state["you"]["potions"]:
                return "potion"
            return "attack"

        room, msgs = drive(self.engine, "telegram:1", "arena", ADA,
                           responder=responder, max_moves=40)
        self.assertTrue(finished(room))
        self.assertTrue(any("🏁" in m for m in msgs))

    def test_arena_shop_item_is_consumed(self):
        self.engine.store.grant_item(ADA, "sword", 1)
        self.assertEqual(self.engine.economy.count(ADA, "sword"), 1)
        self.engine.start("telegram:1", "arena", ADA)
        room = self.engine.live("telegram:1")
        out = self.engine.move("telegram:1", "item sword", ADA)
        self.assertTrue(any("steel sword" in m for m in out))
        self.assertEqual(room.state["you"]["atk"], 20)
        # finish the fight; the engine reconciles the real inventory
        for _ in range(40):
            if self.engine.live("telegram:1") is None:
                break
            self.engine.move("telegram:1", "attack", ADA)
        self.assertEqual(self.engine.economy.count(ADA, "sword"), 0,
                         "equipped sword must be consumed from inventory")

    def test_arena_shield_absorbs_one_death(self):
        self.engine.store.grant_item(ADA, "shield", 1)
        self.engine.start("telegram:1", "arena", ADA)
        self.engine.move("telegram:1", "item shield", ADA)
        room = self.engine.live("telegram:1")
        self.assertTrue(room.state["you"]["shield"])

    def test_escape_all_four_locks(self):
        def responder(r, state):
            return "answer " + state["puzzles"][state["lock"]]["answer"]

        room, msgs = drive(self.engine, "telegram:1", "escape", ADA,
                           responder=responder, max_moves=30)
        self.assertTrue(finished(room))
        self.assertTrue(any("door opens" in m for m in msgs))

    def test_escape_keycard_opens_a_lock(self):
        self.engine.store.grant_item(ADA, "keycard", 1)
        self.engine.start("telegram:1", "escape", ADA)
        out = self.engine.move("telegram:1", "use keycard", ADA)
        self.assertTrue(any("keycard" in m for m in out))
        room = self.engine.live("telegram:1")
        self.assertGreaterEqual(room.state["lock"], 1)

    def test_escape_three_strikes_seals_the_room(self):
        # mechanic, tested directly on the game (no AI luck involved)
        import random as _rnd
        from types import SimpleNamespace
        game = EscapeRoomGame()
        fake = SimpleNamespace(state=game.new_state(_rnd.Random(7)))
        outs = game._attempt(fake, "Ada", "answer no")
        outs += game._attempt(fake, "Ada", "answer no")
        self.assertFalse(fake.state["done"], "two strikes must not seal")
        outs += game._attempt(fake, "Ada", "answer no")
        self.assertTrue(fake.state["done"], "three strikes must seal")
        self.assertTrue(any("seals" in m for m in outs[-1:]))
        # engine level: wrong answers keep coming until the room ends
        self.engine.start("telegram:1", "escape", ADA)
        for _ in range(60):
            if self.engine.live("telegram:1") is None:
                break
            self.engine.move("telegram:1", "answer no", ADA)
        self.assertTrue(finished(self.engine.live("telegram:1")))


# ── players & economy ────────────────────────────────────────────────────────

class PlayerEconomyTests(unittest.TestCase):
    def setUp(self):
        self.engine, _ = make_engine()
        self.store = self.engine.store
        self.econ = self.engine.economy

    def tearDown(self):
        self.engine.shutdown()

    def test_win_loss_streaks(self):
        self.store.record_outcome(ADA, won=True, game="duel", points=60,
                                  coins=60, score=5)
        self.store.record_outcome(ADA, won=False, game="duel", points=10,
                                  coins=15, score=0)
        p = self.store.get(ADA.key)
        self.assertEqual((p.wins, p.losses), (1, 1))
        self.assertEqual(p.streak, -1)
        self.store.record_outcome(ADA, won=True, game="duel", points=60,
                                  coins=60, score=5)
        p = self.store.get(ADA.key)
        self.assertEqual(p.streak, 1)
        self.assertEqual(p.per_game["duel"]["played"], 3)

    def test_leaderboard_renders(self):
        self.store.record_outcome(ADA, won=True, game="duel", points=120,
                                  coins=60, score=5)
        self.store.record_outcome(BOB, won=False, game="duel", points=10,
                                  coins=15, score=0)
        text = self.engine.board.render(5)
        self.assertIn("Ada", text)
        self.assertIn("Bob", text)
        self.assertTrue(text.index("Ada") < text.index("Bob"),
                        "the leader must rank first")

    def test_leaderboard_per_game(self):
        self.store.record_outcome(ADA, won=True, game="trivia", points=80,
                                  coins=60, score=3)
        rows = self.engine.board.top(5, game="trivia")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], "Ada")

    def test_shop_buy_and_balance(self):
        self.store.add_coins(ADA, 500, "test grant")
        ok, msg = self.econ.purchase(ADA, "shield")
        self.assertTrue(ok, msg)
        self.assertEqual(self.econ.balance(ADA), 500 - 200)
        self.assertEqual(self.econ.count(ADA, "shield"), 1)
        # wallet receipts recorded
        rows = self.engine.db.query(
            "SELECT reason FROM game_wallet WHERE player_key = ?",
            (ADA.key,))
        self.assertTrue(any(r["reason"] == "buy:shield" for r in rows))

    def test_shop_refuses_overdraft(self):
        self.store.add_coins(ADA, 5, "test grant")
        ok, msg = self.econ.purchase(ADA, "keycard")
        self.assertFalse(ok)
        self.assertIn("500c", msg)

    def test_catalog_scoped_per_game(self):
        arena_items = self.econ.catalog("battle_arena")
        slugs = {i.slug for i in arena_items}
        self.assertIn("gear_sword", slugs)
        self.assertIn("potion", slugs)
        self.assertNotIn("keycard", slugs)

    def test_consume_item(self):
        self.store.grant_item(ADA, "potion", 2)
        self.assertTrue(self.econ.consume(ADA, "potion"))
        self.assertEqual(self.econ.count(ADA, "potion"), 1)
        self.assertFalse(self.econ.consume(ADA, "keycard"))


# ── the AI mind ──────────────────────────────────────────────────────────────

class GameMindTests(unittest.TestCase):
    def test_bisect_number_strategy(self):
        mind = GameMind()
        self.assertEqual(mind.number(1, 100), 50)
        self.assertEqual(mind.number(51, 100), 75)
        self.assertEqual(mind.number(1, 1), 1)

    def test_word_starting_with_falls_back(self):
        mind = GameMind()
        w = mind.word_starting_with("k")
        self.assertTrue(w.startswith("k") and 2 <= len(w) <= 15)
        w2 = mind.word_starting_with("z", ("zebra",))
        self.assertEqual(w2, "zebra")

    def test_letter_guess_avoids_revealed(self):
        mind = GameMind()
        revealed = set("abcdefghijklmnopqrstuvwxyz")
        letter = mind.letter_guess(revealed, 5)
        self.assertEqual(letter, "z")  # everything else seen

    def test_choice_without_model_is_deterministic_by_seed(self):
        a = GameMind(seed=7)
        b = GameMind(seed=7)
        self.assertEqual(a.choice(["x", "y", "z"]),
                         b.choice(["x", "y", "z"]))

    def test_model_upgrade_used_when_connected(self):
        def suggest(prompt):
            return "B" if "Choose one option" in prompt else "cat"

        mind = GameMind(suggest=suggest)
        self.assertEqual(mind.choice(["alpha", "beta"], "ctx"), "beta")
        self.assertEqual(mind.word_starting_with("c"), "cat")

    def test_vote_uses_suspicion(self):
        mind = GameMind()
        pick = mind.vote(["Ada", "Bob", "Cyr"],
                         {"Ada": 0.9, "Bob": 0.1, "Cyr": 0.4})
        self.assertEqual(pick, "Ada")

    def test_combat_move_saves_the_potion(self):
        mind = GameMind()
        move = mind.combat_move({"hp": 10, "max_hp": 50, "potions": 1},
                                {"attack": 5})
        self.assertEqual(move["action"], "potion")


if __name__ == "__main__":
    unittest.main()
