"""Tests for the money-making opportunities hunter (no network)."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.agents import opportunities as O


def _opp(**kw):
    base = dict(kind="paid_task", title="Test gig", source="example.com",
                url="https://example.com/gig1", payout_text="",
                effort="medium", regions=["global"])
    base.update(kw)
    return O.Opportunity(**base)


class PayoutTests(unittest.TestCase):
    def test_dollar(self):
        self.assertEqual(25.0, O.extract_payout_usd("Get paid $25 per test"))

    def test_up_to(self):
        self.assertEqual(60.0, O.extract_payout_usd("earn up to $60 per study"))

    def test_hourly_scaled(self):
        self.assertEqual(120.0, O.extract_payout_usd("$30/hour remote work"))

    def test_naira_converted(self):
        v = O.extract_payout_usd("₦15000 payout")
        self.assertIsNotNone(v)
        self.assertAlmostEqual(10.0, v, places=1)

    def test_none(self):
        self.assertIsNone(O.extract_payout_usd("a great opportunity awaits"))

    def test_post_init_extracts(self):
        o = _opp(payout_text="earn $40 today")
        self.assertEqual(40.0, o.payout_usd)


class UrlTests(unittest.TestCase):
    def test_strips_tracking(self):
        a = O.normalize_url("https://Example.com/page?utm_source=x&ref=1")
        b = O.normalize_url("https://example.com/page")
        self.assertEqual(a, b)

    def test_bad_url_never_raises(self):
        self.assertIsInstance(O.normalize_url("not a url"), str)

    def test_title_similarity(self):
        self.assertTrue(O.titles_similar(
            "Get paid to test websites remotely",
            "Get paid to test websites remote"))
        self.assertFalse(O.titles_similar(
            "Bug bounty program", "Free cooking course"))


class ScoringTests(unittest.TestCase):
    def test_deterministic(self):
        o = _opp(payout_text="$50")
        self.assertEqual(O.score_opportunity(o), O.score_opportunity(o))

    def test_higher_payout_scores_higher(self):
        low = _opp(payout_text="$5")
        high = _opp(payout_text="$500")
        self.assertGreater(O.score_opportunity(high), O.score_opportunity(low))

    def test_low_effort_beats_high_effort(self):
        easy = _opp(effort="low", title="a")
        hard = _opp(effort="high", title="b")
        self.assertGreater(O.score_opportunity(easy), O.score_opportunity(hard))

    def test_trusted_domain_bonus(self):
        trusted = _opp(url="https://www.upwork.com/jobs/1", title="t1")
        other = _opp(url="https://random-xyz.example/jobs/1", title="t2")
        self.assertGreater(O.score_opportunity(trusted),
                           O.score_opportunity(other))


class FinderTests(unittest.TestCase):
    def test_query_pools_global(self):
        total = sum(len(f.queries) for f in
                    (O.PaidTaskFinder(), O.FreeCourseFinder(),
                     O.BountyFinder(), O.GigFinder()))
        self.assertGreaterEqual(total, 30)

    def test_parse_results(self):
        f = O.PaidTaskFinder()
        opps = f.parse_results([
            {"url": "https://example.com/a", "title": "Paid user testing",
             "snippet": "earn $25 per test, worldwide"},
            {"url": "", "title": "bad", "snippet": ""},
        ])
        self.assertEqual(1, len(opps))
        self.assertEqual(25.0, opps[0].payout_usd)
        self.assertIn("global", opps[0].regions)

    def test_curated_referrals(self):
        opps = O.ReferralFinder().curated_opportunities()
        self.assertGreaterEqual(len(opps), 5)
        self.assertTrue(all(o.kind == "referral" for o in opps))
        self.assertTrue(all(o.url.startswith("http") for o in opps))

    def test_all_finders_registered(self):
        self.assertEqual(11, len(O.FINDERS))
        self.assertEqual(set(O.KINDS),
                         {f.kind for f in (c() for c in O.FINDERS)})


class HunterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.hunter = O.OpportunityHunter()
        self.hunter._data_dir = Path(self.tmp)

    def test_scan_curated_only_offline(self):
        opps = self.hunter.scan(use_curated=True)
        self.assertTrue(opps)
        # curated entries now span multiple kinds (referral + board finders)
        kinds = {o.kind for o in opps}
        self.assertTrue(kinds <= set(O.KINDS))
        self.assertIn("referral", kinds)
        scores = [o.score for o in opps]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_dedupe_exact_and_fuzzy(self):
        def fake_search(q, n=6):
            return [
                {"url": "https://example.com/a?utm_source=z",
                 "title": "Paid testing gig", "snippet": "$20"},
                {"url": "https://example.com/a",
                 "title": "Paid testing gig", "snippet": "$20"},
                {"url": "https://example.com/b",
                 "title": "Paid testing gig worldwide", "snippet": "$20"},
            ]
        h = O.OpportunityHunter(search_fn=fake_search)
        h._data_dir = Path(self.tmp)
        opps = h.scan(kinds=["paid_task"], use_curated=False)
        urls = {O.normalize_url(o.url) for o in opps}
        self.assertEqual(len(opps), len(urls))
        self.assertLessEqual(len(opps), 2)  # exact + fuzzy dedupe

    def test_seen_persistence_and_new_flag(self):
        first = self.hunter.scan(use_curated=True)
        self.assertTrue(all(o.is_new for o in first))
        h2 = O.OpportunityHunter()
        h2._data_dir = Path(self.tmp)
        second = h2.scan(use_curated=True)
        self.assertTrue(all(not o.is_new for o in second))
        # jsonl on disk
        lines = (Path(self.tmp) / "seen.jsonl").read_text().splitlines()
        self.assertEqual(len(first), len(lines))
        json.loads(lines[0])  # valid JSON

    def test_profile_round_trip(self):
        p = O.OpportunityProfile(skills=["python"], regions=["global"],
                                 min_payout_usd=10.0)
        self.hunter.save_profile(p)
        loaded = self.hunter.load_profile()
        self.assertEqual(p.skills, loaded.skills)
        self.assertEqual(10.0, loaded.min_payout_usd)

    def test_profile_filtering(self):
        p = O.OpportunityProfile(kinds=["bounty"], min_payout_usd=100.0)
        rich = _opp(kind="bounty", payout_text="$500 prize")
        poor = _opp(kind="bounty", payout_text="$5 prize")
        task = _opp(kind="paid_task", payout_text="$500")
        self.assertTrue(p.accepts(rich))
        self.assertFalse(p.accepts(poor))
        self.assertFalse(p.accepts(task))

    def test_render(self):
        opps = self.hunter.scan(use_curated=True)
        text = self.hunter.render(opps[:3])
        self.assertIn("money opportunities", text)
        self.assertIn("http", text)


class ControlTests(unittest.TestCase):
    def test_parse_money(self):
        from nomorals.social.chat.control import parse_control
        c = parse_control("/money scan bounty")
        self.assertEqual("money", c.kind)
        c2 = parse_control("/money")
        self.assertEqual("money", c2.kind)

    def test_handle_money_verbs_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            import os
            os.environ["NOMORALS_DATA_DIR"] = tmp  # ignored if unsupported
            try:
                out = O.handle_money_command("profile", None)
                self.assertIn("money profile", out)
                out = O.handle_money_command("scan", None)
                self.assertIn("money opportunities", out)
                out = O.handle_money_command("bogusverb", None)
                self.assertIn("usage", out)
                out = O.handle_money_command("scan nosuchkind", None)
                self.assertIn("unknown kind", out)
            finally:
                os.environ.pop("NOMORALS_DATA_DIR", None)

    def test_register_tool(self):
        calls = {}

        class FakeRegistry:
            def register(self, name, **kw):
                calls[name] = kw
                def deco(fn):
                    calls[name]["fn"] = fn
                    return fn
                return deco

        O.register(FakeRegistry())
        self.assertIn("money_scan", calls)
        res = calls["money_scan"]["fn"]()
        self.assertIn("opportunities", res)


if __name__ == "__main__":
    unittest.main()
