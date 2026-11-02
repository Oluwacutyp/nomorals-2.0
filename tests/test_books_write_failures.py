"""Tail-100 pins: nomorals/books/write.py census sites.

BookForge's writing path is a documented fallback ladder: the model
writes when a real model is answering, the template composer writes
otherwise, and a chapter is always produced. All four sites are
rungs of that ladder and are held as exclusions:

36 — ``model_available``: the primary authority
(``llm.power.model_usable``) crash -> pass, and the function falls
through to its independent secondary check (the router stats
snapshot). The answer still comes from a real probe, never a guess.

44 — ``model_available``: the fallback snapshot call itself crashes
-> return False. Fail-closed on capability detection: "no usable
model" routes the caller to the template floor; it never fabricates
an availability claim.

586 — ``write_chapter``: ``model_chapter`` crash -> the template
composer writes the chapter (the docstring's contract: "falls back
to the template composer on any failure so a chapter is always
produced"). Degradation to the documented floor, not a false claim
about which writer produced the text — the chapter object is what
the caller reads, and it is genuinely written.

627 — ``_model_pass``: the directed model call crashes -> return
"" — the function's own documented contract ("Empty string when no
model is answering"); callers fall to the heuristic floor on empty.

No site records to a failure ledger: write.py imports no failure
store, so the pins prove behaviour only — there is no recording
surface at these sites to assert silence against.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.books import write as W
from nomorals.books.model import Book, Chapter


def _book_with_chapter() -> tuple[Book, Chapter]:
    book = Book(topic="Beekeeping", slug="beekeeping")
    chapter = Chapter(number=1, title="Hive Basics",
                      beats=["open the hive"])
    book.chapters = [chapter]
    return book, chapter


class TestModelAvailablePrimaryCrash(unittest.TestCase):
    """36: primary probe crash -> the snapshot rung still answers."""

    def test_primary_crash_falls_through_to_snapshot(self):
        router = SimpleNamespace(
            stats_snapshot=lambda: {"active": "llama-3.3-70b"})
        ctx = SimpleNamespace(router=router)
        with mock.patch("nomorals.llm.power.model_usable",
                        side_effect=RuntimeError("power probe boom")):
            self.assertTrue(W.model_available(ctx))

    def test_primary_crash_and_no_router_is_false(self):
        ctx = SimpleNamespace(router=None)
        with mock.patch("nomorals.llm.power.model_usable",
                        side_effect=RuntimeError("power probe boom")):
            self.assertFalse(W.model_available(ctx))

    def test_primary_answer_is_used_when_it_works(self):
        # Control: with the primary healthy, its verdict is the answer
        # and the snapshot rung is not consulted.
        router = SimpleNamespace(
            stats_snapshot=lambda: {"active": "llama-3.3-70b"})
        ctx = SimpleNamespace(router=router)
        with mock.patch("nomorals.llm.power.model_usable",
                        return_value=False):
            self.assertFalse(W.model_available(ctx))


class TestModelAvailableSnapshotCrash(unittest.TestCase):
    """44: snapshot crash -> False (fail-closed capability probe)."""

    def test_snapshot_crash_returns_false(self):
        def boom():
            raise RuntimeError("snapshot boom")

        router = SimpleNamespace(stats_snapshot=boom)
        ctx = SimpleNamespace(router=router)
        with mock.patch("nomorals.llm.power.model_usable",
                        side_effect=RuntimeError("primary down")):
            self.assertFalse(W.model_available(ctx))

    def test_snapshot_mock_active_is_false(self):
        router = SimpleNamespace(
            stats_snapshot=lambda: {"active": "mock"})
        ctx = SimpleNamespace(router=router)
        with mock.patch("nomorals.llm.power.model_usable",
                        side_effect=RuntimeError("primary down")):
            self.assertFalse(W.model_available(ctx))


class TestWriteChapterModelCrash(unittest.TestCase):
    """586: model crash -> template chapter, chapter marked written."""

    def test_model_crash_uses_template_and_marks_written(self):
        book, chapter = _book_with_chapter()
        with mock.patch.object(W, "model_available", return_value=True), \
                mock.patch.object(
                    W, "model_chapter",
                    side_effect=RuntimeError("model hiccup")), \
                mock.patch.object(
                    W, "template_chapter",
                    return_value="TEMPLATE PROSE") as template:
            text = W.write_chapter(book, chapter, context=object(),
                                   use_model=True)
        self.assertEqual(text, "TEMPLATE PROSE")
        self.assertEqual(chapter.text, "TEMPLATE PROSE")
        self.assertEqual(chapter.status, "written")
        template.assert_called_once()

    def test_model_success_skips_template(self):
        # Control: a healthy model writes; the floor is not touched.
        book, chapter = _book_with_chapter()
        with mock.patch.object(W, "model_available", return_value=True), \
                mock.patch.object(W, "model_chapter",
                                  return_value="MODEL PROSE"), \
                mock.patch.object(W, "template_chapter") as template:
            text = W.write_chapter(book, chapter, context=object(),
                                   use_model=True)
        self.assertEqual(text, "MODEL PROSE")
        template.assert_not_called()


class TestModelPassCrash(unittest.TestCase):
    """627: directed model call crash -> "" (documented contract)."""

    def test_brain_crash_returns_empty(self):
        with mock.patch.object(W, "model_available", return_value=True), \
                mock.patch.object(
                    W, "brain_for",
                    side_effect=RuntimeError("brain down")):
            self.assertEqual(W._model_pass("expand this", object()), "")

    def test_unavailable_model_returns_empty_without_calling(self):
        with mock.patch.object(W, "model_available", return_value=False), \
                mock.patch.object(W, "brain_for") as brain:
            self.assertEqual(W._model_pass("expand this", object()), "")
        brain.assert_not_called()


if __name__ == "__main__":
    unittest.main()
