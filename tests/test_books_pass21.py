"""Pass #21 (books + BookForge) — verified-bug regression tests.

Each test pins one bug found by reading the code end-to-end:

1. ``template_outline`` silently under-delivered: deep-dive material
   ran out and ``chapters[:n]`` returned FEWER chapters than requested
   (asked 20 → got 17) with no signal anywhere.
2. Shape caps silently rewrote decisions: ``plan_book`` told the model
   "never truncate" then sliced its outline to 24; ``infer_chapter_count``
   rejected a model answer above 24; ``BookForge.create`` clamped an
   explicit owner count to 24.
3. ``SerialManager.start_novel(chapters=N)`` accepted the parameter and
   silently ignored it (the fiction state planned decide_shape's count
   while the meta claimed the owner's number).
4. ``Library.read`` raised a raw IndexError out of range and silently
   served + recorded progress for the WRONG chapter on negative numbers,
   instead of the LibraryError every sibling method raises.
   ``_guess_title`` also mistook the first chapter heading ("# One")
   for the book title.
5. Coherence name capture: ``_AGE_RE`` / ``_DEAD_RE`` run under
   IGNORECASE, so the "name" captured was the first word in a long lazy
   window — "The captain was age 50" → age regression for "The";
   "The old man died" → "The died ch.1 but said in ch.2" (error severity).
6. Branches + serialized publishing lived in a hardcoded
   ~/workspace/devon/books_data universe no writer populates, so
   /novel branch and /novel publish failed "story not found" for every
   real FictionWriter story, and tree() claimed "no chapters yet".
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


def _ctx(tmp: str | None = None):
    tmp = tmp or tempfile.mkdtemp()
    return SimpleNamespace(db=None,
                           settings=SimpleNamespace(workspace_dir=tmp)), tmp


def _router_ctx(payload_text: str):
    """Context whose router answers with ``payload_text`` (plan/infer)."""
    def snapshot():
        return {"active": "hf_serverless", "chain": ["hf_serverless"],
                "health": {"hf_serverless": {"consecutive_failures": 0}}}

    class Resp:
        ok = True

        def __init__(self, t):
            self.text = t

    class Router:
        def chat(self, msgs, params):
            return Resp(payload_text)

    router = SimpleNamespace(stats_snapshot=snapshot)
    ctx = SimpleNamespace(router=router, settings=SimpleNamespace(offline=False))
    ctx.router = Router()
    return ctx


class TestTemplateOutlineExactCount(unittest.TestCase):
    def test_delivers_exactly_the_asked_count(self):
        from nomorals.books.model import Book
        from nomorals.books.outline import template_outline
        book = Book(topic="the history of paper clips", slug="x")
        for n in (12, 20, 24, 30):
            chapters = template_outline(book, n_chapters=n)
            self.assertEqual(len(chapters), n, f"asked {n}")
            titles = [c.title for c in chapters]
            self.assertEqual(len(set(titles)), n, "titles must be distinct")


class TestShapeCaps(unittest.TestCase):
    def test_create_honors_explicit_count_above_24(self):
        from nomorals.books.forge import BookForge
        ctx, _tmp = _ctx()
        book = BookForge(ctx).create("the history of paper clips",
                                     chapters=30, research=False)
        self.assertEqual(len(book.chapters), 30)

    def test_plan_book_does_not_truncate_model_shape(self):
        import json
        from nomorals.books.forge import plan_book
        chapters = [{"title": f"Part {i}", "beats": ["a"]}
                    for i in range(1, 31)]
        payload = json.dumps({"title": "The Long Work", "chapters": chapters})
        plan = plan_book("a sweeping reference work", "", _router_ctx(payload))
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan["chapters"]), 30)

    def test_infer_accepts_model_answer_above_24(self):
        from nomorals.books.forge import infer_chapter_count
        n = infer_chapter_count("an epic multi-volume history",
                                _router_ctx("40"))
        self.assertEqual(n, 40)


class TestStartNovelChapters(unittest.TestCase):
    def test_explicit_chapters_are_planned(self):
        from nomorals.books.fiction import FictionWriter
        from nomorals.books.serial import SerialManager
        ctx, _tmp = _ctx()
        mgr = SerialManager(ctx)
        started = mgr.start_novel(
            "A lighthouse keeper keeps a log of ships that never arrive.",
            genre="mystery", chapters=5)
        state = FictionWriter(ctx).load(started["slug"])
        self.assertEqual(state.total_planned, 5)
        self.assertIn("5", started["shape"])


class TestLibraryReadBounds(unittest.TestCase):
    def _library_with_book(self):
        from nomorals.books.library import Library
        ctx, tmp = _ctx()
        lib = Library(ctx)
        para = " ".join(
            ["The keeper walked the long gallery and counted the lamps."] * 12)
        src = Path(tmp) / "tale.txt"
        src.write_text("# One\n\n" + para + "\n\n# Two\n\n" +
                       para.replace("keeper", "sailor") + "\n\n# Three\n\n" +
                       para.replace("keeper", "widow") + "\n")
        result = lib.ingest(src)
        return lib, result.slug

    def test_out_of_range_raises_library_error(self):
        from nomorals.books.library import LibraryError
        lib, slug = self._library_with_book()
        with self.assertRaises(LibraryError):
            lib.read(slug, chapter=99)

    def test_negative_chapter_raises_and_records_nothing(self):
        from nomorals.books.library import LibraryError
        lib, slug = self._library_with_book()
        with self.assertRaises(LibraryError):
            lib.read(slug, chapter=-1)
        self.assertFalse(lib.get_progress(slug)["started"])

    def test_title_is_not_the_first_chapter_heading(self):
        lib, slug = self._library_with_book()
        self.assertNotEqual(lib.load(slug)["title"], "# One")
        self.assertFalse(lib.load(slug)["title"].startswith("#"))


class TestCoherenceNameCapture(unittest.TestCase):
    def test_sentence_opener_is_not_a_dead_character(self):
        from nomorals.books.coherence import contradiction_scan
        chapters = [
            (1, "The old man died at dawn. The village mourned him."),
            (2, "The man said nothing at the funeral, then walked away."),
        ]
        kinds = [f.kind for f in contradiction_scan(chapters)]
        self.assertNotIn("contradiction_dead_speaks", kinds)

    def test_sentence_opener_has_no_age(self):
        from nomorals.books.coherence import analyze_coherence
        chapters = [
            (1, "The captain was age 50 at the helm of the old ship."),
            (2, "The captain was age 40 in his youth, the sailors said."),
        ]
        kinds = [f.kind for f in analyze_coherence({}, chapters).timeline]
        self.assertNotIn("timeline_age_regression", kinds)

    def test_real_name_still_caught(self):
        from nomorals.books.coherence import analyze_coherence, contradiction_scan
        dead = contradiction_scan([
            (1, "Mara died in the fire."),
            (2, "Mara said hello to the crowd."),
        ])
        self.assertIn("contradiction_dead_speaks", [f.kind for f in dead])
        aged = analyze_coherence({}, [
            (1, "Mara aged 30 lifted the Sunblade."),
            (2, "Mara aged 29 walked on."),
        ])
        self.assertIn("timeline_age_regression",
                      [f.kind for f in aged.timeline])


class TestBranchesPublishRealStories(unittest.TestCase):
    def _fiction_story(self, chapters: int = 1):
        from nomorals.books.fiction import FictionWriter
        ctx, _tmp = _ctx()
        fw = FictionWriter(ctx)
        started = fw.start(
            "A lighthouse keeper logs ships that never arrive",
            genre="mystery", mode="novel")
        for _ in range(chapters - 1):
            fw.write_next(started["slug"])
        return ctx, started["slug"]

    def test_fork_tree_diff_see_the_real_story(self):
        from nomorals.books.branches import StoryBranch, tree
        ctx, slug = self._fiction_story(chapters=2)
        branch = StoryBranch(slug, "alt", context=ctx)
        self.assertTrue(branch.fork(1, "what if the ships arrive")["ok"])
        self.assertTrue(branch.add_chapter("A different sea.", "Alt One")["ok"])
        self.assertEqual(tree(slug, context=ctx)["info"], "2 chapters")
        rows = branch.diff()["chapters"]
        self.assertEqual(len(rows), 1)
        self.assertIsNotNone(rows[0]["canon_words"])

    def test_publish_releases_the_real_chapter_text(self):
        from nomorals.books.publish import SerialPublication
        ctx, slug = self._fiction_story()
        pub = SerialPublication(slug, context=ctx)
        self.assertTrue(pub.start("manual")["ok"])
        released = pub.release()
        self.assertTrue(released["ok"])
        self.assertIn("Chapter 1", released["text"])
        self.assertGreater(len(released["text"]), 100)


if __name__ == "__main__":
    unittest.main()
