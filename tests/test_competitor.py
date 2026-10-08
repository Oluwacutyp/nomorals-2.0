"""Build-map #102 — no-access competitor intel. All offline, mock scrapes."""

from __future__ import annotations

import tempfile
import time
import unittest

from nomorals.marketing.competitor import (
    CompetitorStore, Post, analyze_pillars, analyze_cadence,
    analyze_engagement, aeo_compare, control_competitor,
)


def _db():
    return tempfile.mktemp(suffix=".db")


def _posts():
    base = time.time()
    out = []
    for i in range(8):
        out.append(Post(
            platform="instagram", account="rival", post_id=f"p{i}",
            text=("lagos rent prices are rising fast, affordable flats in yaba "
                  if i < 5 else "new video tour of lekki apartments, watch now "),
            format="video" if i % 2 == 0 else "image",
            posted_at=base - (8 - i) * 86400,
            likes=100 + i * 20, comments=10 + i * 2, shares=5 + i))
    return out


class TestAnalysis(unittest.TestCase):
    def test_pillars_found(self):
        pillars = analyze_pillars(_posts())
        self.assertTrue(len(pillars) >= 1)
        names = [p.name for p in pillars]
        terms = names + [t for p in pillars for t in p.sample_terms]
        self.assertTrue(any(n in ("lagos", "rent", "video", "yaba", "flats", "lekki")
                            for n in terms))

    def test_pillars_empty(self):
        self.assertEqual(analyze_pillars([]), [])

    def test_pillars_never_raises(self):
        self.assertEqual(analyze_pillars(None), [])
        self.assertEqual(analyze_pillars([None, "garbage"]), [])

    def test_cadence(self):
        c = analyze_cadence(_posts())
        self.assertEqual(c.total_posts, 8)
        self.assertGreater(c.posts_per_week, 0)
        self.assertIn("video", c.by_format)

    def test_cadence_empty(self):
        c = analyze_cadence([])
        self.assertEqual(c.total_posts, 0)

    def test_engagement(self):
        e = analyze_engagement(_posts())
        self.assertGreater(e.avg_likes, 0)
        self.assertIn(e.trend, ("rising", "falling", "flat"))

    def test_engagement_rising(self):
        e = analyze_engagement(_posts())
        self.assertEqual(e.trend, "rising")

    def test_engagement_never_raises(self):
        self.assertIsNotNone(analyze_engagement(None))


class TestStore(unittest.TestCase):
    def test_track_untrack(self):
        s = CompetitorStore(db_path=_db())
        self.assertTrue(s.track("rival", "instagram"))
        self.assertEqual(len(s.list_tracked()), 1)
        self.assertTrue(s.untrack("rival"))
        self.assertEqual(len(s.list_tracked()), 0)

    def test_track_bad_input(self):
        s = CompetitorStore(db_path=_db())
        self.assertFalse(s.track("", "instagram"))
        self.assertFalse(s.track("rival", "nonsense-platform"))
        self.assertFalse(s.track(None))

    def test_analyze_no_source(self):
        s = CompetitorStore(db_path=_db())
        r = s.analyze("rival")
        self.assertEqual(r.post_count, 0)
        self.assertEqual(r.source, "no-source")
        self.assertIn("no public posts", r.format())

    def test_analyze_injected(self):
        s = CompetitorStore(db_path=_db())
        r = s.analyze("rival", scrape_fn=lambda a, p: _posts())
        self.assertEqual(r.post_count, 8)
        self.assertEqual(r.source, "injected")
        out = r.format()
        self.assertIn("cadence", out)
        self.assertIn("pillars", out)
        self.assertIn("public data only", out)

    def test_analyze_raising_scrape(self):
        s = CompetitorStore(db_path=_db())

        def boom(a, p):
            raise RuntimeError("nope")
        r = s.analyze("rival", scrape_fn=boom)
        self.assertEqual(r.post_count, 0)

    def test_digest_needs_history(self):
        s = CompetitorStore(db_path=_db())
        self.assertIn("not enough history", s.digest("rival"))

    def test_digest_two_windows(self):
        s = CompetitorStore(db_path=_db())
        s.analyze("rival", scrape_fn=lambda a, p: _posts())
        time.sleep(0.01)
        s.analyze("rival", scrape_fn=lambda a, p: _posts())
        d = s.digest("rival")
        self.assertIn("video", d)
        self.assertIn("engagement delta", d)


class FakeTracker:
    def __init__(self, shares):
        self._shares = shares

    def track_visibility(self, brand, prompts=None):
        class R:
            pass
        r = R()
        share = self._shares.get(brand, 0.0)
        r.share = share
        r.confirmed_pairs = int(share * 8)
        r.total_pairs = 8
        return r


class TestAEOCompare(unittest.TestCase):
    def test_compare(self):
        out = aeo_compare("mybrand", "rival",
                          tracker=FakeTracker({"mybrand": 0.75, "rival": 0.25}))
        self.assertIn("75%", out)
        self.assertIn("you're ahead", out)

    def test_compare_tie(self):
        out = aeo_compare("a", "b",
                          tracker=FakeTracker({"a": 0.5, "b": 0.5}))
        self.assertIn("neck and neck", out)

    def test_compare_bad_input(self):
        self.assertIn("both brands", aeo_compare("", "rival"))

    def test_compare_never_raises(self):
        self.assertIsNotNone(aeo_compare(None, None))


class TestChat(unittest.TestCase):
    def _store(self):
        return CompetitorStore(db_path=_db())

    def test_usage(self):
        out = control_competitor("", store=self._store())
        self.assertIn("/competitor track", out)

    def test_track(self):
        out = control_competitor("track rival", store=self._store())
        self.assertIn("tracking @rival", out)
        self.assertIn("public posts only", out)

    def test_track_missing(self):
        out = control_competitor("track", store=self._store())
        self.assertIn("track who", out)

    def test_list(self):
        s = self._store()
        self.assertIn("no competitors", control_competitor("list", store=s))
        control_competitor("track rival", store=s)
        self.assertIn("@rival", control_competitor("list", store=s))

    def test_untrack(self):
        s = self._store()
        control_competitor("track rival", store=s)
        self.assertIn("stopped tracking", control_competitor("untrack rival", store=s))

    def test_report(self):
        s = self._store()
        out = control_competitor("report rival", store=s,
                                 scrape_fn=lambda a, p: _posts())
        self.assertIn("competitor intel", out)
        self.assertIn("cadence", out)

    def test_report_missing(self):
        out = control_competitor("report", store=self._store())
        self.assertIn("report on who", out)

    def test_digest_chat(self):
        s = self._store()
        s.analyze("rival", scrape_fn=lambda a, p: _posts())
        time.sleep(0.01)
        s.analyze("rival", scrape_fn=lambda a, p: _posts())
        out = control_competitor("digest rival", store=s)
        self.assertIn("competitor digest", out)

    def test_vs(self):
        s = self._store()
        out = control_competitor("vs rival mybrand", store=s,
                                 aeo_tracker=FakeTracker(
                                     {"mybrand": 0.6, "rival": 0.2}))
        self.assertIn("face-off", out)

    def test_vs_missing(self):
        out = control_competitor("vs rival", store=self._store())
        self.assertIn("need both", out)

    def test_garbage_never_raises(self):
        s = self._store()
        out = control_competitor("frobnicate the moon", store=s)
        self.assertIn("/competitor track", out)
        out2 = control_competitor(None, store=s)
        self.assertIn("/competitor track", out2)


if __name__ == "__main__":
    unittest.main()
