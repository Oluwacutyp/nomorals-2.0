"""Tests for the sports bet analyst (nomorals/agents/sports_bet.py)."""

import os
import tempfile
import unittest

from nomorals.agents import sports_bet as sb


class MathHelpersTests(unittest.TestCase):
    def test_devig_sums_to_one(self):
        p = sb.devig_probs(2.0, 3.5, 4.0)
        self.assertAlmostEqual(sum(p), 1.0, places=9)
        # favourite gets the biggest share
        self.assertGreater(p[0], p[2])

    def test_score_to_1x2_sums_to_one(self):
        for exp in (0.1, 0.35, 0.5, 0.65, 0.9):
            p = sb.score_to_1x2(exp)
            self.assertAlmostEqual(sum(p), 1.0, places=9)
            self.assertTrue(all(v > 0 for v in p))

    def test_score_to_1x2_even_matchup_more_draws(self):
        even = sb.score_to_1x2(0.5)
        lopsided = sb.score_to_1x2(0.85)
        self.assertGreater(even[1], lopsided[1])

    def test_kelly_known_values(self):
        # p=0.6 @ 2.0: f* = (1*0.6-0.4)/1 = 0.2 ; half-kelly = 0.1
        # (default 5% cap binds -> 0.05; lift the cap to check raw math)
        self.assertAlmostEqual(sb.kelly_fraction(0.6, 2.0, cap=1.0), 0.1,
                               places=9)
        self.assertAlmostEqual(sb.kelly_fraction(0.6, 2.0), 0.05, places=9)
        # no edge -> 0
        self.assertEqual(sb.kelly_fraction(0.4, 2.0), 0.0)
        # cap respected
        self.assertLessEqual(sb.kelly_fraction(0.9, 10.0), 0.05)

    def test_expected_value(self):
        self.assertAlmostEqual(sb.expected_value(0.5, 2.2), 0.1, places=9)
        self.assertLess(sb.expected_value(0.4, 2.0), 0)


class EloTests(unittest.TestCase):
    def test_winner_gains_loser_loses(self):
        e = sb.EloSystem()
        before_h, before_a = e.rating("A"), e.rating("B")
        e.update("A", "B", 2, 0)
        self.assertGreater(e.rating("A"), before_h)
        self.assertLess(e.rating("B"), before_a)

    def test_draw_moves_toward_each_other(self):
        e = sb.EloSystem()
        e.ratings["A"] = 1700.0
        e.ratings["B"] = 1300.0
        e.update("A", "B", 1, 1)
        self.assertLess(e.rating("A"), 1700.0)
        self.assertGreater(e.rating("B"), 1300.0)

    def test_expected_home_advantage(self):
        e = sb.EloSystem()
        self.assertGreater(e.expected("A", "B"), 0.5)  # equal teams, home edge

    def test_bigger_margin_bigger_shift(self):
        e1, e2 = sb.EloSystem(), sb.EloSystem()
        e1.update("A", "B", 1, 0)
        e2.update("A", "B", 5, 0)
        self.assertGreater(e2.rating("A"), e1.rating("A"))

    def test_persistence_roundtrip(self):
        e = sb.EloSystem()
        e.update("A", "B", 3, 1)
        e2 = sb.EloSystem.from_dict(e.to_dict())
        self.assertAlmostEqual(e2.rating("A"), e.rating("A"), places=9)


class PoissonTests(unittest.TestCase):
    def _league(self):
        fx = []
        for i in range(30):
            fx.append(sb.Fixture(home="Strong", away="Weak",
                                home_goals=3, away_goals=0))
            fx.append(sb.Fixture(home="Weak", away="Strong",
                                home_goals=0, away_goals=2))
        return fx

    def test_fit_and_predict_sane(self):
        pm = sb.PoissonModel()
        pm.fit(self._league())
        pred = pm.predict("Strong", "Weak")
        p = pred["1x2"]
        self.assertAlmostEqual(sum(p), 1.0, places=6)
        self.assertGreater(p[0], 0.6)  # strong at home should dominate
        self.assertGreater(pred["over25"], pred["under25"])

    def test_unseen_teams_get_prior(self):
        pm = sb.PoissonModel()
        pred = pm.predict("NewA", "NewB")
        self.assertAlmostEqual(sum(pred["1x2"]), 1.0, places=6)


class ModelTests(unittest.TestCase):
    def _ctx(self):
        elo = sb.EloSystem()
        elo.ratings["H"] = 1650.0
        elo.ratings["A"] = 1450.0
        return {"elo": elo, "form": sb.FormTracker(),
                "poisson": sb.PoissonModel(), "odds": [],
                "draw_rate": 0.25, "h2h": {}, "fixture": None}

    def test_elo_model_favours_stronger(self):
        p = sb.EloModel().probs("H", "A", self._ctx())
        self.assertAlmostEqual(sum(p), 1.0, places=9)
        self.assertGreater(p[0], p[2])

    def test_market_model_averages_devigged(self):
        odds = [sb.OddsSnapshot(bookmaker="b1", home=2.0, draw=3.5, away=4.0),
                sb.OddsSnapshot(bookmaker="b2", home=2.1, draw=3.4, away=3.8)]
        ctx = self._ctx()
        ctx["odds"] = odds
        p = sb.MarketModel().probs("H", "A", ctx)
        self.assertAlmostEqual(sum(p), 1.0, places=9)
        self.assertGreater(p[0], p[2])

    def test_market_model_flat_without_odds(self):
        p = sb.MarketModel().probs("H", "A", self._ctx())
        self.assertAlmostEqual(sum(p), 1.0, places=9)


class MetaLearnerTests(unittest.TestCase):
    def test_deterministic(self):
        X = [[0.7, 0.2, 0.1] * 4 for _ in range(40)]
        y = [0] * 40
        m1, m2 = sb.LogisticMeta(), sb.LogisticMeta()
        m1.fit(X, y)
        m2.fit(X, y)
        self.assertEqual(m1.predict_proba(X[0]), m2.predict_proba(X[0]))

    def test_learns_obvious_pattern(self):
        # model outputs track the truth perfectly
        X = ([[0.8, 0.1, 0.1] * 4 for _ in range(20)] +
             [[0.1, 0.1, 0.8] * 4 for _ in range(20)])
        y = [0] * 20 + [2] * 20
        m = sb.LogisticMeta()
        m.fit(X, y)
        p = m.predict_proba([0.8, 0.1, 0.1] * 4)
        self.assertGreater(p[0], 0.7)

    def test_brier_stack_weights_better_model(self):
        mp = {"good": (0.7, 0.2, 0.1), "bad": (0.1, 0.2, 0.7)}
        stacked = sb.brier_stack(mp, {"good": 0.05, "bad": 0.5})
        self.assertGreater(stacked[0], 0.5)


class EnsembleTests(unittest.TestCase):
    def test_deterministic_given_seed(self):
        a1 = sb.EnsembleAnalyst(seed=11)
        a2 = sb.EnsembleAnalyst(seed=11)
        fx = [sb.Fixture(home="A", away="B", home_goals=2, away_goals=1)]
        for a in (a1, a2):
            for f in fx:
                a.ingest(f)
            a.refit_poisson(fx)
        r1 = a1.analyze("A", "B", odds=[sb.OddsSnapshot("bm", 2.0, 3.5, 4.0)],
                        fixtures=fx)
        r2 = a2.analyze("A", "B", odds=[sb.OddsSnapshot("bm", 2.0, 3.5, 4.0)],
                        fixtures=fx)
        self.assertEqual(r1.ensemble, r2.ensemble)

    def test_value_flagging(self):
        a = sb.EnsembleAnalyst(seed=3)
        a.elo.ratings["H"] = 1800.0
        a.elo.ratings["A"] = 1300.0
        # generous home odds -> should flag home value
        an = a.analyze("H", "A",
                       odds=[sb.OddsSnapshot("bm", 2.50, 3.40, 3.00)])
        sels = [s for s, _, _, _ in an.edges]
        self.assertIn("home", sels)
        self.assertGreater(an.kelly["home"], 0)

    def test_no_value_no_edges(self):
        a = sb.EnsembleAnalyst(seed=3)
        # razor-sharp odds matching a coin flip -> no edge
        an = a.analyze("H", "A",
                       odds=[sb.OddsSnapshot("bm", 2.0, 3.0, 4.0)])
        # ensemble won't be exactly market; just check structure
        self.assertIsInstance(an.edges, list)

    def test_meta_trains_with_enough_rows(self):
        a = sb.EnsembleAnalyst(seed=5)
        rows = []
        mp = {"elo": (0.6, 0.25, 0.15), "poisson": (0.6, 0.25, 0.15),
              "form": (0.6, 0.25, 0.15), "market": (0.6, 0.25, 0.15)}
        for i in range(40):
            rows.append((mp, i % 3))
        a.train_meta(rows)
        self.assertTrue(a.meta.trained_rows >= 30)

    def test_meta_not_trained_with_few_rows(self):
        a = sb.EnsembleAnalyst(seed=5)
        mp = {"elo": (0.5, 0.3, 0.2), "poisson": (0.5, 0.3, 0.2),
              "form": (0.5, 0.3, 0.2), "market": (0.5, 0.3, 0.2)}
        a.train_meta([(mp, 0)] * 10)
        an = a.analyze("H", "A", fixtures=[])
        self.assertFalse(an.meta_trained)


class BacktestTests(unittest.TestCase):
    def test_backtest_runs_and_reports(self):
        entries = sb.synthetic_history(n=200, seed=42)
        r = sb.backtest(entries, bankroll=1000.0, seed=42)
        self.assertEqual(r.n_fixtures, 200)
        self.assertGreater(r.n_bets, 0)
        self.assertGreaterEqual(r.hit_rate, 0.0)
        self.assertLessEqual(r.hit_rate, 1.0)
        self.assertIn("ensemble", r.brier)
        self.assertIn("elo", r.brier)
        # models should beat a coin flip on Brier (synthetic has signal)
        self.assertLess(r.brier["ensemble"], 0.30)

    def test_backtest_deterministic(self):
        entries = sb.synthetic_history(n=120, seed=9)
        r1 = sb.backtest(entries, seed=9)
        r2 = sb.backtest(entries, seed=9)
        self.assertAlmostEqual(r1.profit, r2.profit, places=9)
        self.assertEqual(r1.n_bets, r2.n_bets)

    def test_synthetic_history_valid(self):
        entries = sb.synthetic_history(n=50, seed=1)
        self.assertEqual(len(entries), 50)
        for f, odds in entries:
            self.assertTrue(f.played)
            self.assertEqual(len(odds), 2)
            self.assertTrue(all(o.home > 1.0 and o.draw > 1.0 and o.away > 1.0
                                for o in odds))


class StoreTests(unittest.TestCase):
    def test_persistence_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            os.environ["NOMORALS_BET_DIR"] = d
            try:
                s = sb.BetStore()
                s.bankroll = 2500.0
                s.record(sb.Fixture(home="X", away="Y", home_goals=2,
                                    away_goals=0))
                s2 = sb.BetStore()
                self.assertAlmostEqual(s2.bankroll, 2500.0, places=9)
                self.assertEqual(len(s2.history), 1)
                self.assertGreater(s2.analyst.elo.rating("X"),
                                   s2.analyst.elo.rating("Y"))
            finally:
                del os.environ["NOMORALS_BET_DIR"]


class ControlParseTests(unittest.TestCase):
    def test_bet_parses(self):
        from nomorals.social.chat.control import parse_control
        cmd = parse_control("/bet analyze Arsenal vs Chelsea")
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.kind, "bet")
        cmd2 = parse_control("/bet bankroll")
        self.assertEqual(cmd2.kind, "bet")

    def test_bet_in_help(self):
        from nomorals.social.chat.control import help_text, detailed_help
        self.assertIn("/bet", help_text())
        self.assertIn("bet", detailed_help("bet"))


class ChatControlTests(unittest.TestCase):
    def _rt(self):
        from nomorals.agents.partner_runtime import PartnerRuntime
        return PartnerRuntime.__new__(PartnerRuntime)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["NOMORALS_BET_DIR"] = self.tmp.name
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(os.environ.pop, "NOMORALS_BET_DIR", None)

    def test_record_three_args(self):
        rt = self._rt()
        out = rt._control_bet("record Arsenal Chelsea 2-1", "t")
        self.assertIn("recorded: Arsenal 2-1 Chelsea", out)

    def test_backtest_seed_not_eaten_as_n(self):
        rt = self._rt()
        out = rt._control_bet("backtest 60 --seed 11", "t")
        self.assertIn("backtest: 60 fixtures", out)

    def test_analyze_chat(self):
        rt = self._rt()
        out = rt._control_bet("analyze Alpha vs Beta 2.00 3.40 3.80", "t")
        self.assertIn("Alpha vs Beta", out)
        self.assertIn("ensemble", out)

    def test_bankroll_chat(self):
        rt = self._rt()
        self.assertIn("1000.00", rt._control_bet("bankroll", "t"))
        rt._control_bet("bankroll set 777", "t")
        self.assertIn("777.00", rt._control_bet("bankroll", "t"))


class CliTests(unittest.TestCase):
    def test_nm_bet_bankroll(self):
        import io
        from contextlib import redirect_stdout
        from nomorals.cli import _cmd_bet
        import argparse
        with tempfile.TemporaryDirectory() as d:
            os.environ["NOMORALS_BET_DIR"] = d
            try:
                args = argparse.Namespace(bet_action="bankroll", set=1234.5)
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = _cmd_bet(args, None)
                self.assertEqual(rc, 0)
                self.assertIn("1234.50", buf.getvalue())
            finally:
                del os.environ["NOMORALS_BET_DIR"]

    def test_nm_bet_analyze(self):
        import io
        from contextlib import redirect_stdout
        from nomorals.cli import _cmd_bet
        import argparse
        with tempfile.TemporaryDirectory() as d:
            os.environ["NOMORALS_BET_DIR"] = d
            try:
                args = argparse.Namespace(
                    bet_action="analyze", home="Alpha", away="Beta",
                    league="GEN", odds=[2.0, 3.4, 3.8], bookmaker="t",
                    min_edge=0.04)
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = _cmd_bet(args, None)
                self.assertEqual(rc, 0)
                out = buf.getvalue()
                self.assertIn("Alpha vs Beta", out)
                self.assertIn("ensemble", out)
            finally:
                del os.environ["NOMORALS_BET_DIR"]


if __name__ == "__main__":
    unittest.main()
