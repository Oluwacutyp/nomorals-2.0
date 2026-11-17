"""Gapfix R3-70 — the reader continuity digest engine, wired.

Failing-first: on the unmodified tree there is no
``reader_catch_up_digest`` / ``reader_previously_on`` tool, no
``/books catchup`` / ``/books previously`` verb, and no ``series`` key
on ``story_following`` — so the registration assertions fail, the chat
verbs fall through to the "write a new book" path, and the following
output has no grouping.

Fixture is disk-only: a followed story laid down the way follow+fetch
would (story.json + cached chapter files), no network, no adapter.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.books.reader import FollowedStory, StoryReader

CH2 = ("Mara bought yams at the YAM-MARKET stalls before dawn. "
       "The guards took her toll at the gate. "
       "She swore she would return before the rains. "
       "The road home was long and dusty.")
CH3 = ("Mara crossed the RIVER-CROSSING on a borrowed raft. "
       "The ferryman demanded a song for the toll. "
       "She sang the old road song and he let her pass. "
       "The far bank smelled of rain and woodsmoke.")
CH4 = ("Mara entered the THRONE-ROOM with mud on her boots. "
       "The queen asked what the market had cost her. "
       "Mara laid the yams at the foot of the throne. "
       "The court went quiet and the queen laughed.")


def _ctx():
    tmp = tempfile.mkdtemp()
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(db=None, settings=settings, extras={}), tmp


class _FakeRegistry:
    def __init__(self, context):
        self.context = context
        self.tools = {}

    def register(self, name, **kwargs):
        def deco(fn):
            self.tools[name] = (fn, kwargs)
            return fn
        return deco


def _lay_down_story(reader: StoryReader, slug: str = "tale") -> None:
    """A followed story at chapter 1 of 4, chapters 1-4 cached."""
    story = FollowedStory(
        slug=slug, title="The Long Tale",
        url="https://fake/novel/the-long-tale", source="fake",
        author="A. Teller", status="Ongoing",
        total_chapters=4, current_chapter=1)
    d = reader.story_dir(slug)
    (d / "story.json").write_text(
        json.dumps(story.to_dict(), indent=2), encoding="utf-8")
    ch_dir = d / "chapters"
    (ch_dir / "00001.md").write_text(
        "Mara left home at first light. The market road was empty.",
        encoding="utf-8")
    (ch_dir / "00002.md").write_text(CH2, encoding="utf-8")
    (ch_dir / "00003.md").write_text(CH3, encoding="utf-8")
    (ch_dir / "00004.md").write_text(CH4, encoding="utf-8")


class _Case(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx, self.tmp = _ctx()
        self.reader = StoryReader(self.ctx)
        _lay_down_story(self.reader)
        from nomorals.books import tools as tools_mod
        self.reg = _FakeRegistry(self.ctx)
        tools_mod.register(self.reg)
        from nomorals.agents.partner.runtime_search import RuntimeSearchMixin
        m = RuntimeSearchMixin.__new__(RuntimeSearchMixin)
        m.context = self.ctx
        self.m = m


class TestContinuityTools(_Case):
    def test_catch_up_digest_tool_registered(self):
        self.assertIn("reader_catch_up_digest", self.reg.tools)
        fn, kw = self.reg.tools["reader_catch_up_digest"]
        self.assertTrue(kw.get("description"))
        self.assertIn("parameters", kw)

    def test_previously_on_tool_registered(self):
        self.assertIn("reader_previously_on", self.reg.tools)
        fn, kw = self.reg.tools["reader_previously_on"]
        self.assertTrue(kw.get("description"))
        self.assertIn("parameters", kw)

    def test_digest_tool_summarizes_unread_tail(self):
        fn, _kw = self.reg.tools["reader_catch_up_digest"]
        out = fn("tale")
        self.assertTrue(out["ok"])
        self.assertEqual(out["covered"], [2, 3, 4])
        self.assertEqual(out["gaps_not_cached"], [])
        self.assertEqual(out["current_chapter"], 1)
        self.assertIn("YAM-MARKET", out["combined"])
        self.assertIn("RIVER-CROSSING", out["combined"])
        self.assertIn("THRONE-ROOM", out["combined"])

    def test_digest_does_not_advance_progress(self):
        fn, _kw = self.reg.tools["reader_catch_up_digest"]
        fn("tale")
        self.assertEqual(self.reader._load("tale").current_chapter, 1)

    def test_digest_unknown_story_is_honest(self):
        fn, _kw = self.reg.tools["reader_catch_up_digest"]
        out = fn("no-such-story")
        self.assertFalse(out["ok"])
        self.assertIn("no followed story", out["reason"])

    def test_previously_on_tool_uses_last_read_chapters(self):
        fn, _kw = self.reg.tools["reader_previously_on"]
        out = fn("tale", last_n=2)
        self.assertTrue(out["ok"])
        self.assertEqual(out["chapters_covered"], [3, 4])
        self.assertIn("RIVER-CROSSING", out["combined"])
        self.assertFalse(out["bible"])
        self.assertEqual(out["open_threads"], [])
        self.assertEqual(out["cast"], [])

    def test_previously_on_reads_bible_threads(self):
        bible_dir = Path(self.tmp) / "books" / "bibles"
        bible_dir.mkdir(parents=True, exist_ok=True)
        (bible_dir / "tale.json").write_text(json.dumps({
            "threads": [
                {"summary": "Mara owes the ferryman a song", "status": "open"},
                {"summary": "The queen's debt", "status": "closed"}],
            "characters": [{"name": "Mara"}, {"name": "The Queen"}],
        }), encoding="utf-8")
        fn, _kw = self.reg.tools["reader_previously_on"]
        out = fn("tale", last_n=1)
        self.assertTrue(out["bible"])
        self.assertEqual(out["open_threads"],
                         ["Mara owes the ferryman a song"])
        self.assertEqual(out["cast"], ["Mara", "The Queen"])

    def test_following_groups_by_series(self):
        fn, _kw = self.reg.tools["story_following"]
        out = fn()
        self.assertIn("series", out)
        self.assertIn("The Long Tale", out["series"])
        titles = [s["title"] for s in out["series"]["The Long Tale"]]
        self.assertIn("The Long Tale", titles)

    def test_existing_catch_up_tool_still_works(self):
        # Control: the wired read-and-advance path is untouched.
        self.assertIn("reader_catch_up", self.reg.tools)


class TestContinuityChatVerbs(_Case):
    def test_books_catchup_verb(self):
        reply = self.m._control_book("catchup tale", "whatsapp:123")
        self.assertIn("The Long Tale", reply)
        self.assertIn("YAM-MARKET", reply)
        self.assertIn("RIVER-CROSSING", reply)
        self.assertIn("THRONE-ROOM", reply)
        # The digest adds a recap; it does not read the chapters.
        self.assertEqual(self.reader._load("tale").current_chapter, 1)

    def test_books_catchup_names_gaps(self):
        ch4 = Path(self.tmp) / "books" / "reader" / "tale" / "chapters" / "00004.md"
        ch4.unlink()
        reply = self.m._control_book("catchup tale", "whatsapp:123")
        self.assertIn("4", reply)
        self.assertIn("not held", reply)

    def test_books_previously_verb(self):
        reply = self.m._control_book("previously tale", "whatsapp:123")
        self.assertIn("The Long Tale", reply)
        self.assertIn("RIVER-CROSSING", reply)
        self.assertIn("THRONE-ROOM", reply)

    def test_books_catchup_unknown_slug_is_honest(self):
        reply = self.m._control_book("catchup no-such-story", "whatsapp:123")
        self.assertIn("no followed story", reply.lower())

    def test_books_catchup_needs_slug(self):
        reply = self.m._control_book("catchup", "whatsapp:123")
        self.assertIn("usage", reply.lower())


if __name__ == "__main__":
    unittest.main()
