"""Sweep tests: Glicko-2 ratings, MCTS/adaptive AI, live achievement rarity,
tournament formats, affix loot, economy ledger/gems, streaks, trivia ladder,
quest board, game-over card."""
from __future__ import annotations

import random
import time
import unittest

from nomorals.games import achievements as A
from nomorals.games import economy as E
from nomorals.games import gamemaster as GM
from nomorals.games import gear as G
from nomorals.games import matchmaking as M
from nomorals.games import trivia_forge as TF
from nomorals.games.ai import (AdaptiveBrain, MCTSEngine, _C4Adapter,
                               connect4_mcts_move)
from nomorals.games.daily import (bump_streak, get_streak, grant_freeze,
                                  hunt_date, repair_streak, streak_calendar,
                                  _yesterday)
from nomorals.games.engine import render_game_over_card
from nomorals.games.players import Player
from nomorals.games.tournaments import (Tournament, buchholz,
                                        elimination_bracket, new_tournament,
                                        round_robin_schedule,
                                        sonneborn_berger, swiss_pairings,
                                        tiebreak_standings)
from nomorals.storage.db import Database


def fresh_db():
    db = Database(":memory:")
    db.migrate()
    return db


# ── Glicko-2 ───────────────────────────────────────────────────────────────

class GlickoTests(unittest.TestCase):
    def test_new_player_conservative_ordinal(self):
        db = fresh_db()
        g = M.get_glicko(db, "newbie", "pvp")
        self.assertEqual(g["games"], 0)
        self.assertEqual(g["tier"], "Bronze")
        # ordinal = mu - 3*phi: maximal uncertainty → deep Bronze
        self.assertLess(g["ordinal"], 500)

    def test_win_moves_ordinal_up(self):
        db = fresh_db()
        rep = M.record_glicko2(db, "pvp", "a", "b", 1.0,
                               a_name="A", b_name="B")
        self.assertGreater(rep["a"]["ordinal"], rep["b"]["ordinal"])
        self.assertTrue(rep["a"]["provisional"])  # first 5 games
        self.assertGreaterEqual(rep["a"]["delta"], 0)

    def test_placement_floor(self):
        db = fresh_db()
        # newcomer loses 5 straight — never drops below start ordinal band
        start = M.get_glicko(db, "a", "pvp")["ordinal"]
        for _ in range(5):
            M.record_glicko2(db, "pvp", "a", "b", 0.0)
        g = M.get_glicko(db, "a", "pvp")
        self.assertGreaterEqual(g["ordinal"], start - 1)
        self.assertFalse(M.get_glicko(db, "a", "pvp")["games"] <= 5
                         and g["ordinal"] < start - 1)

    def test_upset_detection(self):
        db = fresh_db()
        # make a strong, then have a newcomer beat them
        for _ in range(8):
            M.record_glicko2(db, "pvp", "champ", "b", 1.0)
        rep = M.record_glicko2(db, "pvp", "newb", "champ", 1.0)
        self.assertLess(rep["prob_a"], 0.5)
        # upset flag fires when the underdog wins at long odds
        self.assertIn("upset", rep)

    def test_tiers(self):
        self.assertEqual(M.rank_tier(950), "Silver")
        self.assertEqual(M.rank_tier(1200), "Gold")
        self.assertEqual(M.rank_tier(1600), "Diamond")
        self.assertEqual(M.rank_tier(1800), "Mythic")

    def test_predict_probability_symmetric(self):
        db = fresh_db()
        p = M.predict_win_probability(db, "pvp", "x", "y")
        self.assertAlmostEqual(p, 0.5, places=2)

    def test_tiers_render(self):
        db = fresh_db()
        M.record_glicko2(db, "pvp", "a", "b", 1.0, a_name="A", b_name="B")
        text = M.render_tiers(db, "pvp")
        self.assertIn("A", text)
        self.assertIn("tiers", text)


# ── MCTS + adaptive ────────────────────────────────────────────────────────

class MCTSTests(unittest.TestCase):
    def test_mcts_takes_immediate_win(self):
        board = [[0] * 7 for _ in range(6)]
        # three in a row for player 1 at bottom, column 3 completes
        board[5][0] = board[5][1] = board[5][2] = 1
        col = connect4_mcts_move(board, 1, budget=0.2,
                                 rng=random.Random(0))
        self.assertEqual(col, 3)

    def test_mcts_blocks_immediate_loss(self):
        board = [[0] * 7 for _ in range(6)]
        board[5][0] = board[5][1] = board[5][2] = 2
        col = connect4_mcts_move(board, 1, budget=0.2,
                                 rng=random.Random(1))
        self.assertEqual(col, 3)

    def test_mcts_returns_legal(self):
        board = [[0] * 7 for _ in range(6)]
        col = connect4_mcts_move(board, 1, budget=0.15,
                                 rng=random.Random(2))
        self.assertIn(col, range(7))

    def test_mcts_engine_game_agnostic(self):
        # tiny take-away game: heap of 3, take 1-2, take-last wins
        class TakeAway:
            def legal_moves(self, s): return [1, 2][:s] if s else []
            def apply(self, s, m): return s - m
            def winner(self, s):
                if s == 0:
                    return 1  # last mover was P1 in this harness
                return None
            def player_to_move(self, s): return 1
        eng = MCTSEngine(TakeAway(), budget=0.1,
                         rng=random.Random(0))
        move = eng.search(3, 1)
        self.assertIn(move, (1, 2))

    def test_c4_adapter_winner(self):
        ad = _C4Adapter(1)
        state = tuple(tuple([0] * 7) for _ in range(6))
        self.assertIsNone(ad.winner(state))
        rows = [list(r) for r in state]
        rows[5][0] = rows[5][1] = rows[5][2] = rows[5][3] = 1
        self.assertEqual(ad.winner(tuple(tuple(r) for r in rows)), 1)


class AdaptiveTests(unittest.TestCase):
    def test_climbs_on_wins(self):
        b = AdaptiveBrain(seed=0)
        self.assertEqual(b.difficulty_for("p", "c4"), "normal")
        b.record("p", "c4", True)
        d = b.record("p", "c4", True)
        self.assertEqual(d, "hard")

    def test_drops_on_losses(self):
        b = AdaptiveBrain(seed=0)
        b.record("p", "c4", False)
        d = b.record("p", "c4", False)
        self.assertEqual(d, "easy")

    def test_draw_resets_streak(self):
        b = AdaptiveBrain(seed=0)
        b.record("p", "c4", True)
        b.record("p", "c4", None)
        self.assertEqual(b.difficulty_for("p", "c4"), "normal")

    def test_caps_at_ends(self):
        b = AdaptiveBrain(seed=0)
        for _ in range(10):
            b.record("p", "c4", True)
            b.record("p", "c4", True)
        self.assertEqual(b.difficulty_for("p", "c4"), "grandmaster")


# ── achievements: rarity / progress / showcase ─────────────────────────────

class AchievementSweepTests(unittest.TestCase):
    def test_track_progress_unlocks_at_goal(self):
        db = fresh_db()
        for _ in range(9):
            prog, goal, new = A.track_progress(db, "p1", "games_10")
            self.assertFalse(new)
        prog, goal, new = A.track_progress(db, "p1", "games_10")
        self.assertTrue(new)
        self.assertEqual((prog, goal), (10, 10))

    def test_track_unknown(self):
        db = fresh_db()
        self.assertEqual(A.track_progress(db, "p1", "nope"), (0, 0, False))

    def test_progress_text_bar(self):
        db = fresh_db()
        A.track_progress(db, "p1", "wins_25", delta=13)
        text = A.progress_text(db, "p1", "wins_25")
        self.assertIn("13/25", text)
        self.assertIn("█", text)

    def test_live_rarity_fallback_small_sample(self):
        db = fresh_db()
        A.unlock_achievement(db, "p1", "connect4_win")
        tier, frac = A.live_rarity(db, "connect4_win")
        self.assertEqual(frac, 0.0)  # <30 players → catalog fallback
        self.assertIn(tier, A.RARITY_ORDER)

    def test_live_rarity_computed(self):
        db = fresh_db()
        # 40 distinct players, only one holds the rare one
        for i in range(40):
            A.unlock_achievement(db, f"p{i}", "connect4_win")
        A.unlock_achievement(db, "p0", "slots_jackpot")
        tier, frac = A.live_rarity(db, "slots_jackpot")
        self.assertEqual(tier, "epic")  # 1/40 = 2.5%
        self.assertAlmostEqual(frac, 1 / 40)
        badge = A.rarity_badge(db, "slots_jackpot")
        self.assertIn("epic", badge)

    def test_repeatable_milestones(self):
        db = fresh_db()
        self.assertEqual(A.repeatable_check(db, "p1", "wins_100", 99), 0)
        self.assertEqual(A.repeatable_check(db, "p1", "wins_100", 100), 1)
        self.assertEqual(A.repeatable_check(db, "p1", "wins_100", 150), 0)
        self.assertEqual(A.repeatable_check(db, "p1", "wins_100", 200), 2)

    def test_showcase(self):
        db = fresh_db()
        A.unlock_achievement(db, "p1", "connect4_win")
        A.unlock_achievement(db, "p1", "arena_win")
        ok, msg = A.set_showcase(db, "p1",
                                 ["connect4_win", "arena_win"])
        self.assertTrue(ok)
        items = A.get_showcase(db, "p1")
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["id"], "connect4_win")
        # locked achievement rejected
        ok, msg = A.set_showcase(db, "p1", ["slots_jackpot"])
        self.assertFalse(ok)

    def test_anniversaries_empty_when_fresh(self):
        db = fresh_db()
        A.unlock_achievement(db, "p1", "connect4_win")
        self.assertEqual(A.achievement_anniversaries(db, "p1"), [])

    def test_anniversaries_hit(self):
        db = fresh_db()
        db.execute(
            "INSERT OR REPLACE INTO achievements "
            "(player_key, achievement_id, unlocked_at) VALUES (?, ?, ?)",
            ("p1", "connect4_win", time.time() - 365.25 * 86400))
        ann = A.achievement_anniversaries(db, "p1")
        self.assertEqual(len(ann), 1)
        self.assertEqual(ann[0]["years"], 1)


# ── tournaments: formats ───────────────────────────────────────────────────

class TournamentFormatTests(unittest.TestCase):
    def test_swiss_no_rematch(self):
        players = ["a", "b", "c", "d"]
        pts = {p: 0 for p in players}
        r1 = swiss_pairings(players, pts, [], rng=random.Random(0))
        self.assertEqual(len(r1), 2)
        seen = set()
        for x, y in r1:
            self.assertNotIn((x, y), seen)
            seen.update({x, y})
        hist = [(x, y) for x, y in r1]
        pts[r1[0][0]] = 3
        r2 = swiss_pairings(players, pts, hist, rng=random.Random(0))
        for x, y in r2:
            self.assertNotIn((x, y), hist)
            self.assertNotIn((y, x), hist)

    def test_swiss_bye_odd(self):
        players = ["a", "b", "c"]
        pairs = swiss_pairings(players, {p: 0 for p in players}, [],
                               rng=random.Random(0))
        byes = [x for x, y in pairs if y is None]
        self.assertEqual(len(byes), 1)
        # no second bye for the same player next round
        pts = {p: 0 for p in players}
        hist = [(x, y) for x, y in pairs]
        r2 = swiss_pairings(players, pts, hist, rng=random.Random(0))
        byes2 = [x for x, y in r2 if y is None]
        self.assertNotEqual(byes2[0], byes[0])

    def test_round_robin_all_pairs(self):
        players = ["a", "b", "c", "d"]
        sched = round_robin_schedule(players)
        self.assertEqual(len(sched), 3)
        pairs = set()
        for rnd in sched:
            self.assertEqual(len(rnd), 2)
            for x, y in rnd:
                pairs.add(tuple(sorted((x, y))))
        self.assertEqual(len(pairs), 6)

    def test_elimination_bracket(self):
        br = elimination_bracket(["a", "b", "c", "d", "e"],
                                 rng=random.Random(0))
        self.assertEqual(len(br["winners"]), 4)  # padded to 8
        br2 = elimination_bracket(["a", "b"], rng=random.Random(0),
                                  double=True)
        self.assertIn("losers", br2)

    def test_tiebreaks(self):
        pts = {"a": 6.0, "b": 6.0, "c": 0.0}
        # a beat c (weak), b beat a... construct: a drew b, both beat c
        hist = [("a", "b", 0.5), ("b", "a", 0.5),
                ("a", "c", 1.0), ("c", "a", 0.0),
                ("b", "c", 1.0), ("c", "b", 0.0)]
        rows = tiebreak_standings(pts, hist)
        # a and b tied on 6; Buchholz: a faced b(6)+c(0)=6, b faced a(6)+c(0)=6;
        # S-B: a: 3.0+1.0=4.0? a drew b (3.0) beat c (0) → 3.0; b drew a (3.0)
        # beat c → 3.0; tie → name order
        self.assertEqual(rows[0][0], "a")
        self.assertGreaterEqual(buchholz("a", pts, hist), 0)
        self.assertGreaterEqual(sonneborn_berger("b", pts, hist), 0)

    def test_tournament_format_field(self):
        t = Tournament(id="t1", name="Swiss Night", games=["pvp"],
                       players=["a", "b", "c", "d"], rounds=3,
                       format="swiss")
        pairs = t.pair_round(seed=0)
        self.assertEqual(len(pairs), 2)
        self.assertIn("tiebreaks", t.standings_text())

    def test_new_tournament_format(self):
        from nomorals.games.tournaments import tournament_dir
        t = new_tournament("X", ["pvp"], ["a", "b"], format="elimination")
        try:
            self.assertEqual(t.format, "elimination")
            pairs = t.pair_round(seed=1)
            self.assertEqual(len(pairs), 1)
        finally:
            # don't pollute the tournaments data dir
            try:
                (tournament_dir() / f"{t.id}.json").unlink(missing_ok=True)
            except Exception:
                pass


# ── gear: affix loot ───────────────────────────────────────────────────────

class AffixLootTests(unittest.TestCase):
    def test_forge_loot(self):
        rng = random.Random(42)
        loot = G.forge_loot("broadsword_common", 30, rng=rng)
        self.assertIsNotNone(loot)
        self.assertTrue(loot.slug.startswith("loot:"))
        self.assertGreaterEqual(len(loot.affixes), 1)
        # one affix per group max (D2 rule)
        groups = []
        for name, _mods in loot.affixes:
            grp = next(a.group for a in G.PREFIXES + G.SUFFIXES
                       if a.name == name)
            groups.append(grp)
        self.assertEqual(len(groups), len(set(groups)))

    def test_forge_loot_unknown_base(self):
        self.assertIsNone(G.forge_loot("nope", 10,
                                       rng=random.Random(0)))

    def test_grant_and_stats(self):
        from nomorals.games.gear import GearStore
        db = fresh_db()
        store = GearStore(db)
        loot = G.forge_loot("broadsword_common", 50,
                            rng=random.Random(7))
        inst = G.grant_loot(store, "p1", loot)
        self.assertIsNotNone(inst)
        self.assertTrue(inst.is_loot())
        stats = G.instance_stats(inst)
        base_atk, _ = G.effective_stats(
            G.GEAR_CATALOG["broadsword_common"])
        self.assertGreaterEqual(stats["atk"], base_atk)
        # reload from DB keeps affixes
        reloaded = store.get(inst.id)
        self.assertEqual(reloaded.affixes, inst.affixes)
        name = reloaded.display_name()
        self.assertIn("Broadsword", name)

    def test_describe_loot(self):
        loot = G.forge_loot("plate_common", 60, rng=random.Random(3))
        text = G.describe_loot(loot)
        self.assertIn(loot.rarity, text)
        self.assertIn("ilvl 60", text)


# ── economy: ledger / gems / health ────────────────────────────────────────

class EconomySweepTests(unittest.TestCase):
    def test_ledger_health_balanced(self):
        db = fresh_db()
        E.record_ledger(db, "p1", "source", 1000, "game:arena")
        E.record_ledger(db, "p1", "sink", 900, "shop:potion")
        h = E.economy_health(db)
        self.assertEqual(h["sources"], 1000)
        self.assertEqual(h["sinks"], 900)
        self.assertIn("balanced", h["verdict"])

    def test_ledger_health_inflation(self):
        db = fresh_db()
        E.record_ledger(db, "p1", "source", 10000, "game:arena")
        E.record_ledger(db, "p1", "sink", 100, "shop:potion")
        h = E.economy_health(db)
        self.assertIn("INFLATION", h["verdict"])

    def test_ledger_health_starved(self):
        db = fresh_db()
        E.record_ledger(db, "p1", "source", 100, "game:arena")
        E.record_ledger(db, "p1", "sink", 1000, "repair")
        h = E.economy_health(db)
        self.assertIn("STARVED", h["verdict"])

    def test_gems(self):
        db = fresh_db()
        self.assertEqual(E.gem_balance(db, "p1"), 0)
        self.assertEqual(E.grant_gems(db, "p1", 10, reason="achievement"), 10)
        self.assertTrue(E.spend_gems(db, "p1", 4, reason="freeze"))
        self.assertEqual(E.gem_balance(db, "p1"), 6)
        self.assertFalse(E.spend_gems(db, "p1", 99))

    def test_dynamic_price_baseline(self):
        db = fresh_db()
        # no ledger data → base price stands
        self.assertEqual(E.dynamic_price(350, db), 350)
        self.assertEqual(E.dynamic_price(350, None), 350)

    def test_dynamic_price_inflation(self):
        db = fresh_db()
        # heavy earn velocity → prices drift up
        E.record_ledger(db, "p1", "source", 100000, "game:arena")
        E.record_ledger(db, "p2", "source", 100000, "game:arena")
        price = E.dynamic_price(350, db)
        self.assertGreater(price, 350)
        self.assertLessEqual(price, 700)


# ── daily streaks ──────────────────────────────────────────────────────────

class StreakTests(unittest.TestCase):
    def test_bump_starts_streak(self):
        db = fresh_db()
        r = bump_streak(db, "p1", date="2026-10-01")
        self.assertEqual(r["streak"], 1)
        self.assertTrue(r["continued"])

    def test_bump_consecutive(self):
        db = fresh_db()
        bump_streak(db, "p1", date="2026-10-01")
        r = bump_streak(db, "p1", date="2026-10-02")
        self.assertEqual(r["streak"], 2)

    def test_gap_resets_without_freeze(self):
        db = fresh_db()
        bump_streak(db, "p1", date="2026-10-01")
        bump_streak(db, "p1", date="2026-10-02")
        r = bump_streak(db, "p1", date="2026-10-04")
        self.assertEqual(r["streak"], 1)
        self.assertFalse(r["continued"])

    def test_freeze_covers_one_missed_day(self):
        db = fresh_db()
        bump_streak(db, "p1", date="2026-10-01")
        grant_freeze(db, "p1", 1)
        r = bump_streak(db, "p1", date="2026-10-03")
        self.assertEqual(r["streak"], 2)
        self.assertTrue(r["continued"])
        self.assertEqual(get_streak(db, "p1")["freezes"], 0)

    def test_milestone_at_7(self):
        db = fresh_db()
        r = {}
        for d in range(1, 8):
            r = bump_streak(db, "p1", date=f"2026-10-{d:02d}")
        self.assertEqual(r["streak"], 7)
        self.assertIn("7-day", r["milestone"])

    def test_calendar_renders(self):
        db = fresh_db()
        text = streak_calendar(db, "p1")
        self.assertIn("streak", text)

    def test_repair(self):
        db = fresh_db()
        bump_streak(db, "p1", date="2026-09-01")
        bump_streak(db, "p1", date="2026-09-02")
        ok, msg = repair_streak(db, "p1")
        self.assertTrue(ok)
        info = get_streak(db, "p1")
        self.assertEqual(info["last_date"], _yesterday(hunt_date()))


# ── trivia ladder ──────────────────────────────────────────────────────────

class TriviaLadderTests(unittest.TestCase):
    def _ladder(self):
        qs = [
            ("2+2?", "4", ["4", "3", "5", "22"]),
            ("capital of France?", "Paris",
             ["Paris", "London", "Rome", "Berlin"]),
            ("largest planet?", "Jupiter",
             ["Jupiter", "Mars", "Saturn", "Venus"]),
        ]
        return TF.TriviaLadder(qs, seed=0)

    def test_ask_card(self):
        lad = self._ladder()
        card = lad.ask()
        self.assertIn("round 1/3", card)
        self.assertIn("2+2?", card)
        self.assertIn("lifelines", card)

    def test_correct_answer_banks(self):
        lad = self._ladder()
        over, msg = lad.answer("4")
        self.assertFalse(over)
        self.assertIn("correct", msg)
        self.assertGreater(lad.banked, 0)

    def test_streak_multiplier(self):
        lad = self._ladder()
        lad.answer("4")
        banked_after_1 = lad.banked
        lad.answer("Paris")
        self.assertGreater(lad.banked - banked_after_1, banked_after_1)

    def test_miss_falls_to_haven(self):
        lad = self._ladder()
        lad.answer("4")      # round 1: 10c
        lad.answer("Paris")  # round 2: 25c (haven)
        lad.answer("Mars")   # round 3: miss → haven floor
        over, msg = lad.answer("x")  # already over
        self.assertTrue(lad.over)
        self.assertEqual(lad.banked, TF.LADDER_PRIZES[2])

    def test_fifty_lifeline(self):
        lad = self._ladder()
        msg = lad.use_lifeline("fifty")
        self.assertIn("removed", msg)
        self.assertFalse(lad.lifelines["fifty"])
        # spent lifeline rejected
        self.assertIn("spent", lad.use_lifeline("fifty"))

    def test_skip_lifeline(self):
        lad = self._ladder()
        msg = lad.use_lifeline("skip")
        self.assertIn("skipped", msg)
        self.assertEqual(lad.round, 1)

    def test_audience_lifeline(self):
        lad = self._ladder()
        msg = lad.use_lifeline("audience")
        self.assertIn("audience votes", msg)

    def test_walk_away(self):
        lad = self._ladder()
        lad.answer("4")
        msg = lad.walk_away()
        self.assertIn("walk away", msg)
        self.assertTrue(lad.over)

    def test_build_options(self):
        opts = TF.build_options("Paris", ["London", "Rome"], rng=random.Random(0))
        self.assertEqual(len(opts), 4)
        self.assertIn("Paris", opts)


# ── quest board / scenes ───────────────────────────────────────────────────

class QuestBoardTests(unittest.TestCase):
    def test_post_advance_complete(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            qb = GM.QuestBoard(data_dir=d, seed=0)
            q = qb.post("g1", "Rats!", "Clear the cellar",
                        beats=["find the nest", "slay the Rat King"],
                        twist="the barkeep hired them")
            self.assertEqual(q["status"], "active")
            q2 = qb.advance("g1", q["id"], "found the nest")
            self.assertEqual(q2["beats_done"], 1)
            self.assertFalse(q2["twist_revealed"])
            q3 = qb.advance("g1", q["id"], "rat king slain")
            self.assertEqual(q3["status"], "complete")
            self.assertTrue(q3["twist_revealed"])
            self.assertEqual(qb.active("g1"), [])

    def test_render(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            qb = GM.QuestBoard(data_dir=d, seed=0)
            self.assertIn("empty", qb.render("g1"))
            qb.post("g1", "Rats!", "Clear the cellar", beats=["a", "b"])
            text = qb.render("g1")
            self.assertIn("Rats!", text)
            self.assertIn("0/2", text)

    def test_continuity(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            qb = GM.QuestBoard(data_dir=d, seed=0)
            qb.note("g1", "Ada owes the barkeep 40c")
            qb.note("g1", "Ada owes the barkeep 40c")  # dedup
            self.assertEqual(len(qb.continuity("g1")), 1)
            prompt = qb.continuity_prompt("g1")
            self.assertIn("barkeep", prompt)

    def test_forge_scene(self):
        sc = GM.forge_scene(seed=1)
        text = sc.render()
        self.assertIn("📍", text)
        self.assertIn("⚖️", text)
        sc2 = GM.forge_scene(seed=1)
        self.assertEqual(sc.location, sc2.location)  # seeded


# ── game-over card ─────────────────────────────────────────────────────────

class GameOverCardTests(unittest.TestCase):
    def test_card_renders(self):
        a = Player.from_sender("telegram", "1", "Ada")
        b = Player.from_sender("telegram", "2", "Bob")
        card = render_game_over_card(
            "pvp", [a, b], {"telegram:1": 30, "telegram:2": 10},
            winner_label="Ada")
        self.assertIn("Ada", card)
        self.assertIn("30 pts", card)
        self.assertIn("rematch", card)
        # sorted: winner first
        self.assertLess(card.index("Ada"), card.index("Bob"))

    def test_card_never_raises(self):
        card = render_game_over_card("pvp", [], {})
        self.assertIsInstance(card, str)


if __name__ == "__main__":
    unittest.main()
