"""Organic BookForge: no predetermined chapter count.

The book grows as it is written — seed arc, then continuation assessments
until the topic is genuinely covered.  Offline suite: heuristic paths only,
no network, no model.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class _FakeSettings:
    def __init__(self, d):
        self.workspace_dir = d


class _FakeCtx:
    router = None

    def __init__(self, d):
        self.settings = _FakeSettings(d)


def _forge(d):
    from nomorals.books.forge import BookForge
    return BookForge(_FakeCtx(d))


class TestOrganicCreate(unittest.TestCase):
    def test_no_count_decided_up_front(self):
        # rich and thin topics both start with just the opening arc —
        # nothing about the final size is fixed at create time
        with tempfile.TemporaryDirectory() as d:
            forge = _forge(d)
            rich = forge.create(
                "an epic novel about a voyage to mars with crew drama",
                research=False)
            thin = forge.create("my cat", research=False)
            for book in (rich, thin):
                self.assertTrue(book.organic)
                self.assertFalse(book.concluded)
                self.assertEqual(len(book.chapters), 3)
                self.assertEqual(book.chapters_written, 0)
                self.assertFalse(book.complete)
            self.assertTrue(rich.coverage)  # the checklist exists…

    def test_coverage_comes_from_content(self):
        from nomorals.books.outline import coverage_map
        cov = coverage_map("Write me a novel about a voyage to mars")
        joined = " ".join(cov)
        self.assertIn("mars", joined)
        self.assertIn("voyage", joined)
        # form/request words never become chapters
        for junk in ("novel", "write"):
            self.assertNotIn(junk, cov)


class TestOrganicGrowth(unittest.TestCase):
    def test_chapters_emerge_from_topic(self):
        # the final count is a structural function of the topic's own
        # coverage — seed (3, one covering the first item) + remaining
        # items + closing arc (2) — never a preset constant
        from nomorals.books.outline import coverage_map
        with tempfile.TemporaryDirectory() as d:
            forge = _forge(d)
            topic = "a practical guide to sourdough baking with starter care"
            book = forge.create(topic, research=False)
            expected = len(coverage_map(topic)) + 4
            result = forge.write_all(book.slug)
            self.assertTrue(result["complete"])
            self.assertTrue(result["concluded"])
            self.assertEqual(result["total_chapters"], expected)
            self.assertGreater(expected, 3)

    def test_richer_topic_grows_larger(self):
        with tempfile.TemporaryDirectory() as d:
            forge = _forge(d)
            big = forge.write_all(forge.create(
                "a guide to python programming with examples and projects",
                research=False).slug)
            small = forge.write_all(
                forge.create("my cat", research=False).slug)
            self.assertGreater(big["total_chapters"], small["total_chapters"])

    def test_written_prose_has_no_boilerplate(self):
        with tempfile.TemporaryDirectory() as d:
            forge = _forge(d)
            book = forge.create("a guide to sourdough baking", research=False)
            forge.write_all(book.slug)
            from nomorals.books.model import STATUS_WRITTEN
            texts = [c.text for c in book.chapters
                     if c.status == STATUS_WRITTEN]
            # reload from disk — the real persisted state
            book = forge.load(book.slug)
            for c in book.chapters:
                for banned in ("Start here:",
                               "In practice this is where most people",
                               "Two details make this click"):
                    self.assertNotIn(banned, c.text)
            self.assertTrue(all(len(t.split()) > 50 for t in texts))

    def test_conclusion_sticks(self):
        with tempfile.TemporaryDirectory() as d:
            forge = _forge(d)
            book = forge.create("a guide to sourdough baking", research=False)
            forge.write_all(book.slug)
            r = forge.write_next(book.slug)
            self.assertTrue(r["done"])
            self.assertTrue(r["concluded"])


class TestOrganicResumable(unittest.TestCase):
    def test_pause_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            forge = _forge(d)
            slug = forge.create("a guide to sourdough baking",
                                research=False).slug
            forge.write_next(slug)
            forge.write_next(slug)
            # a fresh forge on the same workspace picks up exactly
            # where the killed run stopped
            forge2 = _forge(d)
            mid = forge2.load(slug)
            self.assertEqual(mid.chapters_written, 2)
            self.assertTrue(mid.organic)
            self.assertFalse(mid.concluded)
            result = forge2.write_all(slug)
            self.assertTrue(result["complete"])
            # every chapter persisted to disk
            import pathlib
            ch_dir = pathlib.Path(d) / "books" / slug / "chapters"
            self.assertEqual(len(list(ch_dir.glob("ch*.md"))),
                             result["total_chapters"])

    def test_plan_extends_organic_book(self):
        with tempfile.TemporaryDirectory() as d:
            forge = _forge(d)
            book = forge.create("a guide to sourdough baking", research=False)
            before = len(book.chapters)
            forge.plan(book.slug)
            after = len(forge.load(book.slug).chapters)
            self.assertGreaterEqual(after, before)


class TestOrganicSafety(unittest.TestCase):
    def test_backstop_concludes_runaway(self):
        from nomorals.books.model import Book, Chapter
        from nomorals.books.outline import (assess_continuation,
                                            MAX_ORGANIC_CHAPTERS)
        book = Book(topic="x", slug="x", organic=True)
        book.chapters = [Chapter(number=i + 1, title=f"c{i}",
                                 status="written", text="w " * 200)
                         for i in range(MAX_ORGANIC_CHAPTERS)]
        decision = assess_continuation(book)
        self.assertTrue(decision["complete"])
        self.assertTrue(book.concluded)

    def test_write_all_cannot_spin_forever(self):
        # even if assessments kept appending, write_all terminates
        with tempfile.TemporaryDirectory() as d:
            forge = _forge(d)
            book = forge.create("a guide to sourdough baking", research=False)
            result = forge.write_all(book.slug)
            self.assertTrue(result["complete"])
            self.assertLessEqual(result["total_chapters"], 70)


class TestModelContinuation(unittest.TestCase):
    def test_model_drives_next_chapters(self):
        from nomorals.books.model import Book, Chapter, STATUS_WRITTEN
        from nomorals.books.outline import assess_continuation

        class _Resp:
            ok = True

            def __init__(self, text):
                self.text = text

        class _Router:
            def __init__(self, texts):
                self._texts = list(texts)

            def chat(self, messages, params):
                return _Resp(self._texts.pop(0))

        class _Ctx:
            def __init__(self, router):
                self.router = router

        book = Book(topic="sourdough baking", slug="sour", organic=True)
        book._context = _Ctx(_Router([
            '{"complete": false, "reason": "starter care is missing", '
            '"next": [{"title": "Keeping Your Starter Alive", '
            '"beats": ["feeding schedule", "troubleshooting"]}]}',
        ]))
        book.chapters = [Chapter(number=1, title="Intro", status=STATUS_WRITTEN,
                                 text="word " * 150, coverage="__intro__")]
        decision = assess_continuation(book)
        self.assertFalse(decision["complete"])
        self.assertEqual(len(book.chapters), 2)
        self.assertEqual(book.chapters[1].title, "Keeping Your Starter Alive")
        self.assertFalse(book.concluded)

    def test_model_conclusion(self):
        from nomorals.books.model import Book, Chapter, STATUS_WRITTEN
        from nomorals.books.outline import assess_continuation

        class _Resp:
            ok = True
            text = ('{"complete": true, '
                    '"reason": "the topic is fully covered", "next": []}')

        class _Router:
            def chat(self, messages, params):
                return _Resp()

        class _Ctx:
            router = _Router()

        book = Book(topic="sourdough baking", slug="sour", organic=True)
        book._context = _Ctx()
        book.chapters = [Chapter(number=1, title="Intro", status=STATUS_WRITTEN,
                                 text="word " * 150, coverage="__intro__")]
        decision = assess_continuation(book)
        self.assertTrue(decision["complete"])
        self.assertTrue(book.concluded)
        self.assertEqual(len(book.chapters), 1)  # nothing appended

    def test_garbled_model_falls_back_to_heuristic(self):
        from nomorals.books.model import Book, Chapter, STATUS_WRITTEN
        from nomorals.books.outline import assess_continuation

        class _Resp:
            ok = True
            text = "not json at all"

        class _Router:
            def chat(self, messages, params):
                return _Resp()

        class _Ctx:
            router = _Router()

        book = Book(topic="sourdough baking", slug="sour", organic=True,
                    coverage=["starter"])
        book._context = _Ctx()
        book.chapters = [Chapter(number=1, title="Intro", status=STATUS_WRITTEN,
                                 text="word " * 150, coverage="__intro__")]
        decision = assess_continuation(book)
        self.assertFalse(decision["complete"])
        self.assertEqual(len(book.chapters), 2)  # heuristic filled the gap


if __name__ == "__main__":
    unittest.main()
