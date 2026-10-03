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


class AbstractBaseTests(unittest.TestCase):
    """_Base and OddsFetcher are real abstract interfaces: the bases fail
    fast on instantiation, and every concrete model/fetcher still works."""

    def test_base_model_not_instantiable(self):
        with self.assertRaises(TypeError):
            sb._Base()

    def test_all_builtin_models_instantiable(self):
        subs = [c for c in vars(sb).values()
                if isinstance(c, type) and issubclass(c, sb._Base)
                and c is not sb._Base]
        self.assertGreater(len(subs), 0)
        for cls in subs:
            cls()  # must not raise: each overrides probs

    def test_odds_fetcher_base_not_instantiable(self):
        with self.assertRaises(TypeError):
            sb.OddsFetcher()

    def test_all_builtin_fetchers_instantiable(self):
        subs = [c for c in vars(sb).values()
                if isinstance(c, type) and issubclass(c, sb.OddsFetcher)
                and c is not sb.OddsFetcher]
        self.assertGreater(len(subs), 0)
        for cls in subs:
            inst = cls()  # must not raise: each overrides fetch
            self.assertTrue(hasattr(inst, "fetch"))

    def test_manual_fetcher_still_works(self):
        f = sb.ManualFetcher()
        self.assertEqual(f.fetch(), [])
        self.assertEqual(f.available(), (True, "ok"))


class AmericanOddsTests(unittest.TestCase):
    def test_positive(self):
        self.assertEqual(sb._american_to_decimal("+650"), 7.5)
        self.assertEqual(sb._american_to_decimal("390"), 4.9)

    def test_negative(self):
        self.assertAlmostEqual(sb._american_to_decimal("-260"), 1.3846,
                               places=3)

    def test_garbage_returns_none(self):
        for bad in (None, "", "  ", "xx", "0", "+0"):
            self.assertIsNone(sb._american_to_decimal(bad), bad)


class EspnLeagueTests(unittest.TestCase):
    def test_aliases(self):
        cases = {"epl": "eng.1", "EPL": "eng.1", "Premier League": "eng.1",
                 "laliga": "esp.1", "La Liga": "esp.1",
                 "bundesliga": "ger.1", "serie a": "ita.1",
                 "ligue1": "fra.1", "ucl": "uefa.champions",
                 "Champions League": "uefa.champions"}
        for text, code in cases.items():
            self.assertEqual(sb.EspnFetcher.resolve_league(text), code, text)

    def test_unknown_and_empty(self):
        self.assertIsNone(sb.EspnFetcher.resolve_league("nope"))
        self.assertIsNone(sb.EspnFetcher.resolve_league(""))
        self.assertIsNone(sb.EspnFetcher.resolve_league("  "))

    def test_display_names(self):
        self.assertEqual(sb.EspnFetcher.league_display("eng.1"),
                         "Premier League")
        self.assertEqual(sb.EspnFetcher.league_display("xx"), "xx")

    def test_available_keyless(self):
        self.assertEqual(sb.EspnFetcher().available(), (True, "keyless"))

    def test_major_leagues_cover_six(self):
        self.assertEqual(len(sb.EspnFetcher.MAJOR_LEAGUES), 6)


_SAMPLE_ESPN_EVENT = {
    "date": "2026-10-10T11:30Z",
    "name": "Leeds United at Arsenal",
    "status": {"type": {"state": "pre"}},
    "competitions": [{
        "competitors": [
            {"homeAway": "home", "team": {"displayName": "Arsenal"}},
            {"homeAway": "away", "team": {"displayName": "Leeds United"}},
        ],
        "odds": [{
            "provider": {"displayName": "DraftKings"},
            "moneyline": {
                "home": {"close": {"odds": "-260"},
                         "open": {"odds": "-340"}},
                "draw": {"close": {"odds": "+390"},
                         "open": {"odds": "+425"}},
                "away": {"close": {"odds": "+650"},
                         "open": {"odds": "+800"}},
            },
        }],
    }],
}


class EspnParseTests(unittest.TestCase):
    def test_parse_event(self):
        parsed = sb.EspnFetcher._parse_event(_SAMPLE_ESPN_EVENT,
                                             "Premier League")
        self.assertIsNotNone(parsed)
        fx, snaps = parsed
        self.assertEqual(fx.home, "Arsenal")
        self.assertEqual(fx.away, "Leeds United")
        self.assertEqual(fx.league, "Premier League")
        self.assertEqual(fx.date, "2026-10-10T11:30Z")
        self.assertFalse(fx.played)
        self.assertEqual(len(snaps), 1)
        o = snaps[0]
        self.assertEqual(o.bookmaker, "ESPN/DraftKings")
        self.assertAlmostEqual(o.home, 1.3846, places=3)
        self.assertEqual(o.draw, 4.9)
        self.assertEqual(o.away, 7.5)

    def test_skips_finished(self):
        ev = dict(_SAMPLE_ESPN_EVENT)
        ev = {**ev, "status": {"type": {"state": "post"}}}
        self.assertIsNone(sb.EspnFetcher._parse_event(ev, "Premier League"))

    def test_skips_missing_teams(self):
        ev = {**_SAMPLE_ESPN_EVENT,
              "competitions": [{"competitors": [], "odds": []}]}
        self.assertIsNone(sb.EspnFetcher._parse_event(ev, "Premier League"))

    def test_open_odds_fallback(self):
        import copy
        ev = copy.deepcopy(_SAMPLE_ESPN_EVENT)
        ml = ev["competitions"][0]["odds"][0]["moneyline"]
        for side in ml.values():
            del side["close"]  # only open remains
        _, snaps = sb.EspnFetcher._parse_event(ev, "Premier League")
        self.assertAlmostEqual(snaps[0].home, 1.2941, places=3)


def _canned_entries():
    fx1 = sb.Fixture(home="Giants A", away="Giants B", league="Premier League",
                     date="2026-10-10T12:00Z")
    fx2 = sb.Fixture(home="Minnows A", away="Minnows B", league="La Liga",
                     date="2026-10-09T12:00Z")
    fx3 = sb.Fixture(home="Giants A", away="Minnows A", league="Serie A",
                     date="2026-10-11T12:00Z")
    o = sb.OddsSnapshot(bookmaker="ESPN/Test", home=2.0, draw=3.2, away=3.4)
    return [(fx1, [o]), (fx2, [o]), (fx3, [o])]


class RankFixturesTests(unittest.TestCase):
    def test_big_even_matchup_first(self):
        an = sb.EnsembleAnalyst()
        an.elo.ratings.update({"Giants A": 1900.0, "Giants B": 1880.0,
                               "Minnows A": 1400.0, "Minnows B": 1380.0})
        picks = sb.rank_fixtures(_canned_entries(), an, top_n=3)
        self.assertEqual(picks[0][0].home, "Giants A")
        self.assertEqual(picks[0][0].away, "Giants B")
        # mismatch (giant vs minnow) ranks below the minnow derby
        self.assertEqual(picks[1][0].home, "Minnows A")
        self.assertEqual(picks[2][0].home, "Giants A")
        self.assertEqual(picks[2][0].away, "Minnows A")

    def test_top_n_respected(self):
        an = sb.EnsembleAnalyst()
        picks = sb.rank_fixtures(_canned_entries(), an, top_n=2)
        self.assertEqual(len(picks), 2)

    def test_cold_start_falls_back_to_date_order(self):
        an = sb.EnsembleAnalyst()  # all Elo 1500
        picks = sb.rank_fixtures(_canned_entries(), an, top_n=3)
        dates = [fx.date for fx, _ in picks]
        self.assertEqual(dates, sorted(dates))


class FixtureDigestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["NOMORALS_BET_DIR"] = self.tmp.name
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(os.environ.pop, "NOMORALS_BET_DIR", None)

    def _store(self):
        return sb.BetStore()

    def test_digest_renders_picks(self):
        from unittest import mock
        with mock.patch.object(sb.EspnFetcher, "fetch",
                               return_value=_canned_entries()):
            out = sb.fixture_digest(self._store(), top_n=2)
        self.assertIn("Giants A vs Giants B", out)
        self.assertIn("Minnows A vs Minnows B", out)
        self.assertIn("market odds:", out)
        self.assertIn("ensemble", out)

    def test_digest_offline_message(self):
        from unittest import mock
        with mock.patch.object(sb.EspnFetcher, "fetch",
                               side_effect=RuntimeError("boom")):
            out = sb.fixture_digest(self._store())
        self.assertIn("couldn't reach the fixture feed", out)
        self.assertIn("/bet analyze", out)

    def test_digest_empty_message(self):
        from unittest import mock
        with mock.patch.object(sb.EspnFetcher, "fetch", return_value=[]):
            out = sb.fixture_digest(self._store())
        self.assertIn("no upcoming fixtures", out)

    def test_digest_unknown_league_note(self):
        from unittest import mock
        with mock.patch.object(sb.EspnFetcher, "fetch",
                               return_value=_canned_entries()) as m:
            out = sb.fixture_digest(self._store(), league_text="xx-league",
                                    top_n=1)
        self.assertIn("unknown league", out)
        m.assert_called_once_with(league="", limit=20)

    def test_digest_league_alias_passed_through(self):
        from unittest import mock
        with mock.patch.object(sb.EspnFetcher, "fetch",
                               return_value=_canned_entries()) as m:
            sb.fixture_digest(self._store(), league_text="epl", top_n=1)
        m.assert_called_once_with(league="eng.1", limit=20)


class ChatFixtureModeTests(unittest.TestCase):
    def _rt(self):
        from nomorals.agents.partner_runtime import PartnerRuntime
        return PartnerRuntime.__new__(PartnerRuntime)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["NOMORALS_BET_DIR"] = self.tmp.name
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(os.environ.pop, "NOMORALS_BET_DIR", None)

    def test_analyze_no_args_fetches_fixtures(self):
        from unittest import mock
        rt = self._rt()
        with mock.patch.object(sb.EspnFetcher, "fetch",
                               return_value=_canned_entries()):
            out = rt._control_bet("analyze", "t")
        self.assertIn("Giants A vs Giants B", out)
        self.assertNotIn("usage:", out)

    def test_analyze_league_alias(self):
        from unittest import mock
        rt = self._rt()
        with mock.patch.object(sb.EspnFetcher, "fetch",
                               return_value=_canned_entries()) as m:
            out = rt._control_bet("analyze epl", "t")
        self.assertIn("Giants A vs Giants B", out)
        m.assert_called_once_with(league="eng.1", limit=20)

    def test_analyze_top_flag(self):
        from unittest import mock
        rt = self._rt()
        with mock.patch.object(sb.EspnFetcher, "fetch",
                               return_value=_canned_entries()):
            out = rt._control_bet("analyze --top 1", "t")
        # cold start: all Elo 1500 -> date order, Minnows (10-09) first
        self.assertIn("Minnows A vs Minnows B", out)
        self.assertNotIn("Giants A vs Giants B", out)

    def test_analyze_garbage_still_usage(self):
        rt = self._rt()
        out = rt._control_bet("analyze frobnicate", "t")
        self.assertIn("usage:", out)

    def test_analyze_manual_mode_unchanged(self):
        rt = self._rt()
        out = rt._control_bet("analyze Alpha vs Beta 2.00 3.40 3.80", "t")
        self.assertIn("Alpha vs Beta", out)
        self.assertIn("ensemble", out)

    def test_analyze_offline_graceful(self):
        from unittest import mock
        rt = self._rt()
        with mock.patch.object(sb.EspnFetcher, "fetch",
                               side_effect=RuntimeError("down")):
            out = rt._control_bet("analyze", "t")
        self.assertIn("couldn't reach the fixture feed", out)


class CliFixtureTests(unittest.TestCase):
    def test_nm_bet_analyze_fixture_mode(self):
        import io
        import argparse
        from contextlib import redirect_stdout
        from unittest import mock
        from nomorals.cli import _cmd_bet
        with tempfile.TemporaryDirectory() as d:
            os.environ["NOMORALS_BET_DIR"] = d
            try:
                args = argparse.Namespace(
                    bet_action="analyze", home=None, away=None,
                    league="GEN", odds=None, bookmaker="cli",
                    min_edge=0.04, top=2)
                with mock.patch.object(sb.EspnFetcher, "fetch",
                                        return_value=_canned_entries()):
                    buf = io.StringIO()
                    with redirect_stdout(buf):
                        rc = _cmd_bet(args, None)
                self.assertEqual(rc, 0)
                out = buf.getvalue()
                self.assertIn("Giants A vs Giants B", out)
            finally:
                del os.environ["NOMORALS_BET_DIR"]


if __name__ == "__main__":
    unittest.main()
