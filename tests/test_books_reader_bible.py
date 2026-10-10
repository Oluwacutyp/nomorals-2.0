"""StoryReader (follows, progress, bookmarks, cache) and the heuristic
story-bible pipeline — no network, fake adapter injected.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.books import reader as reader_mod
from nomorals.books.bible import BibleBuilder
from nomorals.books.reader import FollowedStory, ReaderError, StoryReader
from nomorals.books.sources import Chapter, StoryMeta


def _ctx():
    tmp = tempfile.mkdtemp()
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(db=None, settings=settings), tmp


class _FakeAdapter:
    name = "fake"
    verified = True

    def novel(self, url: str) -> StoryMeta:
        return StoryMeta(title="The Ember Crown", url=url, source="fake",
                         author="Test Author", synopsis="A crown of fire.",
                         genres=["Fantasy"], status="Ongoing",
                         total_chapters=50)

    def chapter_list(self, novel_url: str, *, limit: int = 0):
        out = [(n, f"Chapter {n}", f"https://fake/ch/{n}") for n in range(1, 51)]
        return out[:limit] if limit else out

    def fetch_chapter(self, url: str) -> Chapter:
        n = int(url.rsplit("/", 1)[-1])
        paras = [f"Paragraph {i} of chapter {n}, with story content. " * 8
                 for i in range(1, 6)]
        prev_url = f"https://fake/ch/{n - 1}" if n > 1 else ""
        return Chapter(number=n, title=f"Chapter {n}: The Turn",
                       url=url, paragraphs=paras, prev_url=prev_url,
                       next_url=f"https://fake/ch/{n + 1}", source="fake")


class TestStoryReader(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx, self.tmp = _ctx()
        self._orig = reader_mod.adapter_for_url
        reader_mod.adapter_for_url = lambda url: _FakeAdapter()  # noqa: E731
        self.r = StoryReader(self.ctx)

    def tearDown(self) -> None:
        reader_mod.adapter_for_url = self._orig

    def test_follow_and_following(self) -> None:
        story = self.r.follow("https://fake/novel/ember-crown")
        self.assertEqual(story.title, "The Ember Crown")
        self.assertEqual(story.total_chapters, 50)
        following = self.r.following()
        self.assertEqual(len(following), 1)
        self.assertEqual(following[0]["slug"], story.slug)

    def test_follow_by_title_search_fails_without_hits(self) -> None:
        import nomorals.books.reader as rm
        orig_search = rm.search_all
        rm.search_all = lambda *a, **k: []
        try:
            with self.assertRaises(ReaderError):
                self.r.follow("a title that matches nothing")
        finally:
            rm.search_all = orig_search

    def test_read_caches_and_tracks_progress(self) -> None:
        story = self.r.follow("https://fake/novel/ember-crown")
        first = self.r.read(story.slug, chapter=3)
        self.assertEqual(first["chapter"], 3)
        self.assertEqual(first["source"], "fake")
        self.assertGreater(first["words"], 100)
        # second read comes from cache
        second = self.r.read(story.slug, chapter=3)
        self.assertEqual(second["source"], "cache")
        prog = self.r.get_progress(story.slug)
        self.assertEqual(prog["chapter"], 3)

    def test_next_prev(self) -> None:
        story = self.r.follow("https://fake/novel/ember-crown")
        self.r.read(story.slug, chapter=5)
        nxt = self.r.next(story.slug)
        self.assertEqual(nxt["chapter"], 6)
        prv = self.r.prev(story.slug)
        self.assertEqual(prv["chapter"], 5)

    def test_set_progress_and_resume(self) -> None:
        story = self.r.follow("https://fake/novel/ember-crown")
        self.r.set_progress(story.slug, 12, offset=400)
        resume = self.r.resume(story.slug)
        self.assertEqual(resume["resume_chapter"], 12)
        self.assertEqual(resume["offset"], 400)

    def test_bookmarks(self) -> None:
        story = self.r.follow("https://fake/novel/ember-crown")
        mark = self.r.add_bookmark(story.slug, 7, label="the betrayal")
        self.assertEqual(mark["bookmark"]["id"], 1)
        marks = self.r.list_bookmarks(story.slug)
        self.assertEqual(len(marks), 1)
        self.assertEqual(marks[0]["label"], "the betrayal")
        self.r.remove_bookmark(story.slug, 1)
        self.assertEqual(self.r.list_bookmarks(story.slug), [])
        with self.assertRaises(ReaderError):
            self.r.remove_bookmark(story.slug, 99)

    def test_sync(self) -> None:
        story = self.r.follow("https://fake/novel/ember-crown")
        report = self.r.sync(story.slug, chapters=2)
        self.assertEqual(report["synced"][0]["fetched"], [1, 2])

    def test_unfollow(self) -> None:
        story = self.r.follow("https://fake/novel/ember-crown")
        self.r.read(story.slug, chapter=1)
        res = self.r.unfollow(story.slug)
        self.assertTrue(res["unfollowed"])
        self.assertEqual(self.r.following(), [])
        # cache kept by default
        self.assertTrue((self.r.story_dir(story.slug) / "chapters").exists())


STORY_TEXT = [
    (1, "Awakening", """
Kael woke to the smell of smoke. "We have to move," Mara said, shaking him.
"I must find the Ember Crown before the Ashen Court does," Kael said.
Mara Voss had been his shadow for three years, ever since the fire.
"The Court cannot be allowed to take the spire," Mara whispered.
Dorian watched from the doorway, saying nothing. Dorian had his own plans.
The road south was watched, Kael knew, and the river road meant dealing with Mara's old crew.
"I will carry the map," Kael said, "and you carry the lies, Mara."
Mara laughed despite herself. "You always did talk too much, Kael," she said.
Behind them, Dorian's footsteps were deliberately loud — a courtesy, or a warning.
Kael counted the exits the way his father had taught him, and Kael did not like the math.
"Whatever comes," Kael said, "we face it together."
"""),
    (2, "The Spire", """
"The spire remembers every name carved into it," said Elder Sable.
Kael touched the stone. "I will not let the Ashen Court have it," he vowed.
Mara drew her blade. Dorian smiled the smile of a man holding a secret.
"We need to reach the crown before dawn," Mara said.
The old rule held: no fire may be kindled within the spire. It costs the bearer a memory.
Sable studied Kael for a long moment. "The boy has his mother's eyes," Sable murmured.
"I must know what the Court promised you, Dorian," Kael said quietly.
Dorian only shrugged, but Mara saw his hand tighten on the hilt.
Kael's hand found the hilt of his own blade, and did not let go.
"""),
]


class TestBibleHeuristics(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx, self.tmp = _ctx()
        self.builder = BibleBuilder(self.ctx)

    def test_build_from_chapters(self) -> None:
        bible = self.builder.build("ember-crown", "The Ember Crown",
                                   STORY_TEXT)
        names = {c.name for c in bible.characters}
        self.assertIn("Kael", names)
        self.assertIn("Mara", names)
        # protagonist: most mentioned, early
        protag = next(c for c in bible.characters if c.role == "protagonist")
        self.assertEqual(protag.name, "Kael")
        # threads mined from goal statements
        self.assertTrue(any("Ember Crown" in t.summary
                            for t in bible.threads))
        # world rules mined
        self.assertTrue(any("spire" in r.lower() or "fire" in r.lower()
                            for r in bible.world_rules))
        # style profiled
        self.assertIn(bible.pov, ("first", "third"))
        self.assertIn(bible.tense, ("past", "present"))
        self.assertTrue(bible.voice_notes)
        self.assertEqual(bible.chapters_digested, 2)

    def test_brief_is_steering_text(self) -> None:
        bible = self.builder.build("ember-crown", "The Ember Crown",
                                   STORY_TEXT)
        brief = bible.brief()
        self.assertIn("Kael", brief)
        self.assertIn("Open plot threads", brief)

    def test_update_folds_new_chapters(self) -> None:
        bible = self.builder.build("ember-crown", "The Ember Crown",
                                   STORY_TEXT[:1])
        self.assertEqual(bible.chapters_digested, 1)
        updated = self.builder.update("ember-crown", [STORY_TEXT[1]])
        self.assertEqual(updated.chapters_digested, 2)
        names = {c.name for c in updated.characters}
        self.assertIn("Sable", names)
        # persistence round-trip
        reloaded = self.builder.load("ember-crown")
        self.assertIsNotNone(reloaded)
        self.assertEqual(len(reloaded.characters),
                         len(updated.characters))

    def test_update_without_bible_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.builder.update("nope", STORY_TEXT)


if __name__ == "__main__":
    unittest.main()
