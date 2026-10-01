"""God-tier content pools: music, news feeds, lead queries, book prose,
responder fallbacks, swarm perspectives.

Every pool that drives user-facing variety must be big enough to feel
fresh, seeded/deterministic where reproducibility matters, and
anti-repeat where the user would notice loops.
"""

import random
import unittest
from types import SimpleNamespace


class MusicPoolTests(unittest.TestCase):
    def setUp(self):
        from nomorals.media import music as m
        self.m = m

    def test_pool_sizes(self):
        m = self.m
        self.assertGreaterEqual(len(m._VERB_BANK), 40)
        self.assertGreaterEqual(len(m._IMAGERY), 40)
        self.assertGreaterEqual(len(m._EMOTION), 24)
        self.assertGreaterEqual(len(m._LINE_TEMPLATES), 20)
        self.assertGreaterEqual(len(m._HOOK_TEMPLATES), 10)
        self.assertGreaterEqual(len(m._RHYME_GROUPS), 30)

    def test_no_duplicates(self):
        m = self.m
        for name in ("_VERB_BANK", "_IMAGERY", "_EMOTION", "_LINE_TEMPLATES",
                     "_HOOK_TEMPLATES"):
            pool = getattr(m, name)
            self.assertEqual(len(pool), len(set(pool)), name)

    def test_templates_format(self):
        slots = dict(verb="run", ing="running", topic_short="midnight",
                     em="quiet", imagery="neon rain", noun="spark")
        for t in self.m._LINE_TEMPLATES + self.m._HOOK_TEMPLATES:
            self.assertTrue(t.format(**slots), t)

    def test_compose_seeded_and_varied(self):
        from nomorals.media.music import MusicCreator
        mc = MusicCreator(SimpleNamespace(router=None))
        a = mc.compose("midnight city", style="lofi", seed=42,
                       with_midi=False)
        b = mc.compose("midnight city", style="lofi", seed=42,
                       with_midi=False)
        c = mc.compose("midnight city", style="lofi", seed=99,
                       with_midi=False)

        def lines(s):
            return [ln for sec in s.sections for ln in sec.lyrics]

        self.assertEqual(lines(a), lines(b))
        self.assertNotEqual(lines(a), lines(c))
        self.assertTrue(lines(a))

    def test_section_avoids_imagery_repeats(self):
        # the engine tracks used imagery per section; with 48 images a
        # 16-line verse should not lean on the same image twice
        from nomorals.media.music import MusicCreator
        mc = MusicCreator(SimpleNamespace(router=None))
        song = mc.compose("ocean drive", style="hiphop", seed=7,
                          with_midi=False)
        verse = next(s for s in song.sections if s.name == "verse")
        hits = []
        for img in self.m._IMAGERY:
            if any(img in ln.lower() for ln in verse.lyrics):
                hits.append(img)
        self.assertEqual(len(hits), len(set(hits)))


class NewsFeedTests(unittest.TestCase):
    def test_categories(self):
        from nomorals.agents.news import (DEFAULT_FEEDS, FEED_CATEGORIES,
                                          feeds_for)
        self.assertGreaterEqual(len(FEED_CATEGORIES), 5)
        total = sum(len(v) for v in FEED_CATEGORIES.values())
        self.assertGreaterEqual(total, 24)
        self.assertIn("nigeria", FEED_CATEGORIES)
        for region in ("usa", "europe", "asia", "wires"):
            self.assertIn(region, FEED_CATEGORIES)
            self.assertTrue(FEED_CATEGORIES[region],
                            f"category {region!r} is empty")
        self.assertGreaterEqual(len(DEFAULT_FEEDS), 10)
        # default digest is a global mix, not Nigeria-limited
        default_urls = [u for _, u in DEFAULT_FEEDS]
        self.assertTrue(any("vanguardngr" in u or "punchng" in u
                            for u in default_urls))
        self.assertTrue(any("nytimes" in u or "cnn.com" in u
                            for u in default_urls))

    def test_feed_urls_syntactically_valid(self):
        from urllib.parse import urlparse

        from nomorals.agents.news import FEED_CATEGORIES, feeds_for
        for cat, feeds in FEED_CATEGORIES.items():
            self.assertTrue(feeds, f"category {cat!r} is empty")
            for name, url in feeds:
                self.assertTrue(name and name.strip(), f"blank name in {cat}")
                parts = urlparse(url)
                self.assertIn(parts.scheme, ("http", "https"),
                              f"bad scheme: {url}")
                self.assertTrue(parts.netloc, f"no host: {url}")
                self.assertNotIn(" ", url, f"space in URL: {url}")
        # no URL may appear twice anywhere in the table
        urls = [u for _, u in feeds_for()]
        self.assertEqual(len(urls), len(set(urls)))
        # DEFAULT_FEEDS is the world+tech+business+regional flattening
        from nomorals.agents.news import DEFAULT_FEEDS
        self.assertEqual(list(DEFAULT_FEEDS),
                         feeds_for("world", "tech", "business", "usa",
                                   "europe", "asia", "nigeria"))

    def test_feeds_for(self):
        from nomorals.agents.news import feeds_for
        sports = feeds_for("sports")
        self.assertTrue(sports)
        self.assertTrue(all(u.startswith("http") for _, u in sports))
        combo = feeds_for("sports", "science")
        self.assertGreaterEqual(len(combo), len(sports))
        # dedupe
        urls = [u for _, u in feeds_for()]
        self.assertEqual(len(urls), len(set(urls)))
        # unknown categories are ignored, not fatal
        self.assertEqual(feeds_for("bogus"), [])
        # case-insensitive
        self.assertEqual(feeds_for("SPORTS"), feeds_for("sports"))


class LeadQueryTests(unittest.TestCase):
    def test_pool_size(self):
        from nomorals.agents.search.engine import _LEAD_QUERIES
        self.assertGreaterEqual(len(_LEAD_QUERIES), 10)
        self.assertEqual(len(_LEAD_QUERIES), len(set(_LEAD_QUERIES)))


class BookPoolTests(unittest.TestCase):
    def test_pool_sizes(self):
        from nomorals.books.write import _BRIDGES, _TAKEAWAY_OPENERS
        self.assertGreaterEqual(len(_BRIDGES), 10)
        self.assertGreaterEqual(len(_TAKEAWAY_OPENERS), 6)
        self.assertEqual(len(_BRIDGES), len(set(_BRIDGES)))

    def test_template_chapter_varies_by_seed(self):
        import nomorals.books.write as w
        from nomorals.books.model import Book, Chapter
        b1 = Book(topic="sourdough", slug="a")
        c1 = Chapter(number=1, title="Starters", beats=["mix", "wait"])
        b2 = Book(topic="sourdough", slug="b")
        t1 = w.template_chapter(b1, c1)
        t2 = w.template_chapter(b2, c1)
        self.assertTrue(t1 and t2)
        # different chapter seeds pick different bridges/openers
        self.assertNotEqual(t1, t2)


class FallbackLineTests(unittest.TestCase):
    def test_six_per_mood(self):
        from nomorals.partner.responder import FALLBACK_LINES
        self.assertGreaterEqual(len(FALLBACK_LINES), 15)
        for mood, lines in FALLBACK_LINES.items():
            self.assertEqual(len(lines), 6, mood)
            self.assertEqual(len(set(lines)), 6, mood)

    def test_no_consecutive_repeats(self):
        from nomorals.partner.responder import PartnerResponder
        r = PartnerResponder.__new__(PartnerResponder)
        r.rng = random.Random(11)
        seq = [r._fallback_parts("anxious")[0] for _ in range(20)]
        for a, b in zip(seq, seq[1:]):
            self.assertNotEqual(a, b)

    def test_unknown_label_falls_back_to_calm(self):
        from nomorals.partner.responder import PartnerResponder
        r = PartnerResponder.__new__(PartnerResponder)
        r.rng = random.Random(3)
        self.assertTrue(r._fallback_parts("nope")[0])


class SwarmPerspectiveTests(unittest.TestCase):
    def test_pool_and_rotation(self):
        from nomorals.agents.swarm import (_PERSPECTIVES,
                                           _rotate_perspectives)
        self.assertGreaterEqual(len(_PERSPECTIVES), 8)
        a = _rotate_perspectives("build a trading bot", 3)
        self.assertEqual(a, _rotate_perspectives("build a trading bot", 3))
        b = _rotate_perspectives("plan a wedding", 3)
        self.assertEqual(len(a), 3)
        self.assertNotEqual(a, b)
        many = _rotate_perspectives("x", 20)
        self.assertEqual(len(many), len(_PERSPECTIVES))
        self.assertEqual(len(set(many)), len(many))


if __name__ == "__main__":
    unittest.main()
