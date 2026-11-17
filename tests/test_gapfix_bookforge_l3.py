"""Gapfix R2-43: BookForge L3 engines wired into the forge as a
post-draft report step.

The forge must run coherence + critique + voice over a finished draft
and REPORT — never rewrite.  These tests fail before the wiring
(``BookForge.review`` does not exist, ``build()`` carries no report)
and pass after it.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.books.forge import BookForge
from nomorals.books.model import STATUS_WRITTEN, Book, Chapter


class _FakeSettings:
    def __init__(self, d):
        self.workspace_dir = d


class _FakeCtx:
    router = None

    def __init__(self, d):
        self.settings = _FakeSettings(d)


def _forge(d):
    return BookForge(_FakeCtx(d))


def _prose(seed: str, words: int = 240) -> str:
    """Varied multi-paragraph prose: narration + dialogue, no years,
    seasons, ages or deaths — a clean draft the coherence checks should
    mostly leave alone."""
    sents = [
        f"{seed} walked the long road home as the light thinned over the fields.",
        f"\"You came back,\" {seed} said, and the words hung in the air between them.",
        "The market was loud with traders calling out prices for bread and cloth and salt.",
        "She counted the coins in her palm twice before she paid, then tucked the change away.",
        "A dog slept by the well, and the children chased each other around the square.",
        f"\"Tell me what happened,\" {seed} asked, \"from the beginning, and leave nothing out.\"",
        "The river had risen overnight, and the ferryman refused to cross until morning.",
        "They ate in silence, listening to the rain on the roof and the wind in the trees.",
        "By the third day the road was dry again, and the carts rolled on toward the town.",
        f"{seed} laughed, a short surprised sound, and shook her head at the whole of it.",
    ]
    paras = []
    i = 0
    total = 0
    while total < words:
        para = " ".join(sents[(i + j) % len(sents)] for j in range(3))
        paras.append(para)
        total += len(para.split())
        i += 3
    return "\n\n".join(paras)


def _make_book(forge: BookForge, slug: str, texts: list[str]) -> Book:
    book = Book(
        topic=f"A tale of {slug}", slug=slug, title=f"The Tale of {slug}",
        chapters=[Chapter(number=i + 1, title=f"Chapter {i + 1}",
                          text=t, status=STATUS_WRITTEN)
                  for i, t in enumerate(texts)],
    )
    forge.save(book)
    return book


class L3ReportTests(unittest.TestCase):
    def test_build_carries_coherence_report(self):
        with tempfile.TemporaryDirectory() as d:
            forge = _forge(d)
            _make_book(forge, "clean-tale",
                       [_prose("Mara"), _prose("Joren"), _prose("Mara")])
            result = forge.build("clean-tale")
            report = result["l3_report"]
            self.assertEqual(report["slug"], "clean-tale")
            self.assertEqual(report["coherence"]["chapters_checked"], 3)
            self.assertEqual(
                set(report["coherence"]["counts"]),
                {"foreshadowing", "timeline", "arcs", "chekhov",
                 "contradictions", "total"})
            self.assertIn("critique", report)
            self.assertIn("voice", report)
            path = Path(d) / "books" / "clean-tale" / "l3_report.json"
            self.assertTrue(path.exists())
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")),
                             report)

    def test_continuity_break_is_named_not_silently_kept(self):
        with tempfile.TemporaryDirectory() as d:
            forge = _forge(d)
            _make_book(forge, "dead-speaks", [
                "Mara died in the winter, and the village mourned her for a week. "
                "The house stood empty after that, and no one spoke her name aloud. "
                + _prose("Joren", 120),
                "\"I am not afraid,\" Mara said, and she walked into the room "
                "as if nothing had happened at all. "
                + _prose("Mara", 120),
            ])
            report = forge.review("dead-speaks")
            contra = [f for f in report["coherence"]["findings"]
                      if f["kind"] == "contradiction_dead_speaks"]
            self.assertTrue(contra, "dead-then-speaking must be reported")
            self.assertEqual(contra[0]["severity"], "error")
            self.assertEqual(contra[0]["chapter"], 2)
            self.assertEqual(contra[0]["entity"], "Mara")
            self.assertGreaterEqual(
                report["coherence"]["counts"]["contradictions"], 1)

    def test_seeded_bible_fires_foreshadowing_and_arcs(self):
        with tempfile.TemporaryDirectory() as d:
            forge = _forge(d)
            from nomorals.books.bible import BibleBuilder, BibleCharacter, PlotThread, StoryBible
            bible = StoryBible(
                story_slug="long-tale", title="The Long Tale",
                characters=[BibleCharacter(name="Wren", first_seen=1,
                                           mentions=0)],
                threads=[PlotThread(id="t1", summary="find the silver key",
                                    status="open", introduced=1,
                                    last_seen=1)],
            )
            BibleBuilder(_FakeCtx(d)).save(bible)
            _make_book(forge, "long-tale",
                       [_prose("Mara", 60) for _ in range(12)])
            report = forge.review("long-tale")
            self.assertEqual(report["bible"], "existing")
            fore = [f for f in report["coherence"]["foreshadowing"]
                    if f["kind"] == "unpaid_setup"]
            self.assertTrue(
                any("t1" in f["message"] for f in fore),
                "the open thread must be flagged unpaid after 10 chapters")
            arcs = [f for f in report["coherence"]["arcs"]
                    if f["kind"] == "abandoned_character"]
            self.assertTrue(
                any(f["entity"] == "Wren" for f in arcs),
                "a character never mentioned again must be flagged")

    def test_no_bible_builds_one_from_the_draft(self):
        with tempfile.TemporaryDirectory() as d:
            forge = _forge(d)
            _make_book(forge, "fresh-tale",
                       [_prose("Mara"), _prose("Joren"), _prose("Mara")])
            report = forge.review("fresh-tale")
            self.assertEqual(report["bible"], "built")
            self.assertTrue(
                (Path(d) / "books" / "bibles" / "fresh-tale.json").exists())

    def test_critique_comes_from_the_one_engine(self):
        with tempfile.TemporaryDirectory() as d:
            forge = _forge(d)
            _make_book(forge, "crit-tale",
                       [_prose("Mara"), _prose("Joren"), _prose("Mara")])
            crit = forge.review("crit-tale")["critique"]
            self.assertIn("overall", crit)
            self.assertNotIn("score", crit)  # collab's shape, not the engine's
            self.assertEqual(
                {dim["dimension"] for dim in crit["dimensions"]},
                {"prose", "pacing", "dialogue", "hook", "structure",
                 "continuity"})
            self.assertIn(crit["verdict"], ("ship", "revise", "rewrite"))
            self.assertEqual(crit["floor"], 0.6)

    def test_voice_table_covers_every_chapter(self):
        with tempfile.TemporaryDirectory() as d:
            forge = _forge(d)
            _make_book(forge, "voice-tale",
                       [_prose("Mara"), _prose("Joren"), _prose("Mara")])
            voice = forge.review("voice-tale")["voice"]
            self.assertEqual(len(voice["per_chapter"]), 3)
            self.assertEqual(voice["reference_chapters"], [1, 2, 3])
            self.assertIn("mean_similarity", voice)
            self.assertIn("drifted_chapters", voice)

    def test_review_never_rewrites_the_draft(self):
        with tempfile.TemporaryDirectory() as d:
            forge = _forge(d)
            texts = [_prose("Mara"), _prose("Joren"), _prose("Mara")]
            book = _make_book(forge, "untouched", texts)
            before_texts = [c.text for c in book.chapters]
            forge.build("untouched")
            after = forge.load("untouched")
            self.assertEqual([c.text for c in after.chapters], before_texts)
            manuscript = (Path(d) / "books" / "untouched"
                          / "manuscript.md").read_text(encoding="utf-8")
            self.assertEqual(manuscript, after.manuscript())
            for i, text in enumerate(before_texts, start=1):
                ch_path = (Path(d) / "books" / "untouched" / "chapters"
                           / f"ch{i:02d}.md")
                self.assertEqual(
                    ch_path.read_text(encoding="utf-8"),
                    f"# Chapter {i}\n\n{text}")

    def test_review_standalone_without_build(self):
        with tempfile.TemporaryDirectory() as d:
            forge = _forge(d)
            _make_book(forge, "standalone",
                       [_prose("Mara"), _prose("Joren")])
            report = forge.review("standalone")
            self.assertEqual(report["coherence"]["chapters_checked"], 2)
            self.assertTrue(
                (Path(d) / "books" / "standalone"
                 / "l3_report.json").exists())

    def test_nm_book_review_verb(self):
        with tempfile.TemporaryDirectory() as d:
            forge = _forge(d)
            _make_book(forge, "cli-tale",
                       [_prose("Mara"), _prose("Joren"), _prose("Mara")])
            from nomorals.cmdline.commands.book import _cmd_book
            args = SimpleNamespace(action="review", slug="cli-tale",
                                   topic="", json=True)
            self.assertEqual(_cmd_book(args, _FakeCtx(d)), 0)
            self.assertTrue(
                (Path(d) / "books" / "cli-tale"
                 / "l3_report.json").exists())


if __name__ == "__main__":
    unittest.main()
