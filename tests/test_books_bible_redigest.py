"""Legacy bible re-digest — pass 1's flagged gap, pinned.

Pass 1 grounded continuations on the story bible's NEW fields
(character states, arc recaps, locations, current situation), but a
bible written by the OLD digestion already claims every chapter
digested — so prepare()'s incremental loop never touches it and the
new fields stay empty on exactly the books followed longest.  These
tests pin the one-time fix on the same synthetic novel as the
grounding suite (facts: ch.5 Ember Key taken, ch.6 Mara captured,
ch.7 Quinn's hand crushed, ch.8 stops mid-scene at the ember gate):

* a bible JSON in the OLD shape (no state on characters, no recaps,
  no locations, no current situation, no digestion stamp,
  chapters_digested == the full count) re-digests on prepare();
* a current bible NEVER re-digests, and an unstamped bible whose
  grounding already reaches its end is only stamped;
* the re-digest reads only the local ingest cache (zero fetches);
* when a local chapter cannot be read, the rebuild pauses and the
  continuation is still served, anchored at the true end — and the
  next prepare resumes and completes the rebuild.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.books import reader as reader_mod
from nomorals.books.bible import (DIGESTION_VERSION, BibleBuilder,
                                  bible_needs_redigest)
from nomorals.books.continuation import StoryContinuer
from nomorals.books.reader import ReaderError, StoryReader
from nomorals.books.sources import Chapter, SourceError, StoryMeta


def _ctx():
    tmp = tempfile.mkdtemp()
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(db=None, settings=settings), tmp


_CHAPTERS: dict[int, list[str]] = {
    1: ["Quinn woke with the taste of copper in his mouth. "
        "\"You died last night,\" Mara said, not looking up from her "
        "book. The old rule was simple: the thirst comes first, and "
        "it does not negotiate.",
        "\"I must find out what I am now,\" Quinn said. Mara "
        "shrugged. \"Then stop bleeding on my floor,\" Mara said."],
    2: ["\"I will not drink from the living,\" Quinn vowed. Mara "
        "laughed, a dry sound. \"We need to leave the city before "
        "the Vigil finds you,\" Mara said.",
        "The Vigil cannot be reasoned with; it can only be outrun. "
        "Quinn nodded, and Mara began to pack."],
    3: ["They descended into the Sunken Archive beneath the city, "
        "where the old maps slept under glass. \"The gate is real,\" "
        "Mara said. \"The maps do not lie.\"",
        "Quinn traced the route with one finger. \"Then we go "
        "through the archive,\" Quinn said."],
    4: ["Inside the Sunken Archive, dust lay thick on the reading "
        "desks. Something moved between the stacks. \"Quinn, "
        "behind you,\" Mara said, and Quinn turned with the knife "
        "already out.",
        "The thing in the stacks fled upward. Mara let out a slow "
        "breath. \"It knows your name,\" Mara said."],
    5: ["At the altar stone Quinn stopped. \"That is it,\" Mara "
        "said. Quinn took the Ember Key from the altar, and its "
        "teeth bit his palm.",
        "\"It chose you,\" Mara said. Quinn closed his fingers "
        "around the Ember Key and did not answer."],
    6: ["The trap was patient and the bridge was narrow. The Vigil "
        "captured Mara at the bridge and dragged her into the "
        "dark before Quinn could cross.",
        "\"Run,\" Mara shouted, but Quinn did not run. Quinn "
        "watched the dark take her and memorised every face."],
    7: ["The second trap was cruder and faster. The trap crushed "
        "Quinn's left hand; the bones ground together and would "
        "never set right.",
        "Quinn bound the hand without a sound. \"It still holds a "
        "key,\" Quinn said. \"That is enough.\""],
    8: ["Quinn stood before the ember gate with the Ember Key cold "
        "in his ruined left hand. The gate's seam glowed like a "
        "held breath.",
        "\"I will open the ember gate before the Vigil returns,\" "
        "Quinn said. Then the seam split wide and a bell began to "
        "toll beneath the world."],
}


class _FactAdapter:
    """Fake archive whose chapters carry the facts above."""

    name = "freewebnovel"
    verified = True

    def __init__(self, total: int = 8) -> None:
        self.total = total
        self.title = "My Vampire System"
        self.fetch_log: list[int] = []

    def novel(self, url: str) -> StoryMeta:
        return StoryMeta(title=self.title, url=url, source=self.name,
                         author="JKSManga", synopsis="A vampire system.",
                         status="Ongoing", total_chapters=self.total)

    def chapter_list(self, novel_url: str, *, limit: int = 0):
        out = [(n, f"Chapter {n}", f"https://fake/ch/{n}")
               for n in range(1, self.total + 1)]
        return out[:limit] if limit else out

    def chapter_url(self, novel_url: str, number: int) -> str:
        return f"https://fake/ch/{number}" \
            if 1 <= number <= self.total else ""

    def fetch_chapter(self, url: str) -> Chapter:
        n = int(url.rsplit("/", 1)[-1])
        self.fetch_log.append(n)
        if n not in _CHAPTERS or n > self.total:
            raise SourceError(f"no chapter {n}")
        return Chapter(
            number=n, title=f"Chapter {n}", url=url,
            paragraphs=list(_CHAPTERS[n]),
            prev_url=f"https://fake/ch/{n - 1}" if n > 1 else "",
            next_url=f"https://fake/ch/{n + 1}" if n < self.total else "",
            source=self.name)


_FAST_PROFILE = {"sync_chapters": 3, "list_limit": 3,
                 "pause_s": 0.0, "timeout": 5.0}

_TEXT9 = (
    "DEVON-CH9-MARKER. Quinn now stood with the Ember Key pressed "
    "to the seam, his ruined left hand shaking against the heat. "
    "The gate drank the light and gave back a low, pleased hum. "
    "\"I will burn the Vigil archive before dawn,\" Quinn said, "
    "and the seam widened another inch. Behind him the tolling "
    "bell climbed toward a note it had never reached, and Quinn "
    "felt the old thirst rise and wait, patient as debt. He thought "
    "of Mara in the dark below the bridge and set his jaw. "
    "\"One lock at a time,\" Quinn said. The Ember Key turned a "
    "fraction, and heat licked between his fingers.")


class _RedigestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx, self.tmp = _ctx()
        self.adapter = _FactAdapter()
        self._orig_adapter = reader_mod.adapter_for_url
        reader_mod.adapter_for_url = lambda url: self.adapter
        self._orig_profile = StoryReader._profile
        StoryReader._profile = lambda self: dict(_FAST_PROFILE)
        self.r = StoryReader(self.ctx)
        story = self.r.follow("https://fake/novel/my-vampire-system")
        self.slug = story.slug
        # ingest the whole library locally; everything below must
        # work from this cache alone
        report = self.r.ingest(self.slug)
        assert report["complete"], report
        self.adapter.fetch_log = []

    def tearDown(self) -> None:
        reader_mod.adapter_for_url = self._orig_adapter
        StoryReader._profile = self._orig_profile

    # ── fixture: a bible in the OLD shape ─────────────────────────
    def _chapter_tuples(self, upto: int = 8):
        out = []
        for n in range(1, upto + 1):
            text = self.r.chapter_text(self.slug, n)
            title = text.split("\n", 1)[0].lstrip("# ").strip()[:120]
            out.append((n, title, text))
        return out

    def _write_legacy_bible(self) -> None:
        """A bible as the OLD digestion left it: cast/threads/rules
        mined and the full count claimed, but no grounding fields
        and no digestion stamp anywhere in the file."""
        builder = BibleBuilder(self.ctx)
        builder.build(self.slug, "My Vampire System",
                      self._chapter_tuples())
        path = builder.bible_path(self.slug)
        d = json.loads(path.read_text(encoding="utf-8"))
        for key in ("digestion_version", "grounding_through",
                    "recaps", "locations", "current_situation"):
            d.pop(key, None)
        for c in d.get("characters", []):
            c.pop("state", None)
            c.pop("state_chapter", None)
        d["progressions"] = []
        d["voiceprint"] = {}
        d["chapters_digested"] = 8
        path.write_text(json.dumps(d, indent=2), encoding="utf-8")
        bible = builder.load(self.slug)
        assert bible is not None and bible_needs_redigest(bible), \
            "fixture must read as a stale legacy bible"
        assert all(not c.state for c in bible.characters)
        assert not bible.recaps and not bible.current_situation

    def _continuer_with_recording_model(self):
        cont = StoryContinuer(self.ctx)
        calls: list[dict] = []

        def fake_model(bible, chapter_no, tails, words, direction,
                       temperature=0.75, scene=None):
            system, user = StoryContinuer._assemble_prompt(
                cont, bible, chapter_no, tails, words, direction,
                scene=scene)
            calls.append({"chapter_no": chapter_no,
                          "tails": list(tails), "system": system,
                          "user": user, "scene": scene})
            return _TEXT9

        cont._model_continuation = fake_model
        return cont, calls

    def _break_chapter_text(self, bad: int):
        """Make chapter_text raise for one chapter; returns a
        restore callable (also registered as cleanup)."""
        orig = StoryReader.chapter_text

        def broken(reader, slug, number):
            if number == bad:
                raise ReaderError(f"chapter {bad} unavailable")
            return orig(reader, slug, number)

        def restore() -> None:
            StoryReader.chapter_text = orig

        StoryReader.chapter_text = broken
        self.addCleanup(restore)
        return restore


class TestStaleBibleRedigests(_RedigestCase):
    def test_stale_bible_redigests_on_prepare(self) -> None:
        self._write_legacy_bible()
        position_before = self.r._load(self.slug).current_chapter
        cont = StoryContinuer(self.ctx)
        with self.assertLogs("nomorals.books.continuation",
                             level="INFO") as logs:
            bible = cont.prepare(self.slug)
        # the rebuild is an event, not a silent fix-up
        self.assertTrue(any("rebuilt" in line and "1–8" in line
                            for line in logs.output), logs.output)
        # the SAME late-story facts pass 1 pins, now on the legacy book
        self.assertEqual(bible.chapters_digested, 8)
        by_name = {c.name: c for c in bible.characters}
        self.assertIn("crushed", by_name["Quinn"].state)
        self.assertEqual(by_name["Quinn"].state_chapter, 7)
        self.assertIn("captured Mara", by_name["Mara"].state)
        self.assertEqual(by_name["Mara"].state_chapter, 6)
        details = " ".join(p.detail for p in bible.progressions)
        self.assertIn("Ember Key", details)
        self.assertIn("Sunken Archive",
                      {x.name for x in bible.locations})
        self.assertTrue(bible.recaps)
        self.assertEqual(bible.recaps[-1].end, 8)
        self.assertIn("End of chapter 8", bible.current_situation)
        # the voiceprint was re-measured by the fold, and the bible
        # is stamped current through its claimed end
        self.assertIsNotNone(bible.voice)
        self.assertEqual(bible.digestion_version, DIGESTION_VERSION)
        self.assertEqual(bible.grounding_through, 8)
        # the re-digest moved nothing else: reader position, and the
        # claimed digested range, exactly as before
        self.assertEqual(self.r._load(self.slug).current_chapter,
                         position_before)


class TestCurrentBibleNeverRedigests(_RedigestCase):
    def test_current_bible_is_never_redigested(self) -> None:
        cont = StoryContinuer(self.ctx)
        bible = cont.prepare(self.slug)
        self.assertEqual(bible.digestion_version, DIGESTION_VERSION)
        builder = BibleBuilder(self.ctx)
        path = builder.bible_path(self.slug)
        before = path.read_bytes()
        calls: list = []
        orig_refold = BibleBuilder.refold

        def counting(builder_, slug, chapters):
            calls.append(list(chapters))
            return orig_refold(builder_, slug, chapters)

        BibleBuilder.refold = counting
        try:
            with self.assertNoLogs("nomorals.books.continuation",
                                   level="INFO"):
                again = cont.prepare(self.slug)
        finally:
            BibleBuilder.refold = orig_refold
        self.assertEqual(calls, [])
        self.assertEqual(again.chapters_digested, 8)
        self.assertEqual(path.read_bytes(), before)

    def test_unstamped_grounded_bible_is_only_stamped(self) -> None:
        # a bible written by the grounding fold BEFORE the stamp
        # existed: full grounding, no digestion_version key at all
        cont = StoryContinuer(self.ctx)
        cont.prepare(self.slug)
        builder = BibleBuilder(self.ctx)
        path = builder.bible_path(self.slug)
        d = json.loads(path.read_text(encoding="utf-8"))
        d.pop("digestion_version", None)
        d.pop("grounding_through", None)
        path.write_text(json.dumps(d, indent=2), encoding="utf-8")
        unstamped = builder.load(self.slug)
        self.assertEqual(unstamped.digestion_version, 0)
        self.assertFalse(bible_needs_redigest(unstamped))
        calls: list = []
        orig_refold = BibleBuilder.refold

        def counting(builder_, slug, chapters):
            calls.append(list(chapters))
            return orig_refold(builder_, slug, chapters)

        BibleBuilder.refold = counting
        try:
            with self.assertNoLogs("nomorals.books.continuation",
                                   level="INFO"):
                bible = cont.prepare(self.slug)
        finally:
            BibleBuilder.refold = orig_refold
        self.assertEqual(calls, [])          # no re-digest happened
        self.assertEqual(bible.digestion_version, DIGESTION_VERSION)
        self.assertEqual(bible.grounding_through, 8)
        self.assertIn("crushed",
                      {c.name: c for c in bible.characters}
                      ["Quinn"].state)


class TestRedigestIsLocalOnly(_RedigestCase):
    def test_redigest_reads_only_the_local_cache(self) -> None:
        self._write_legacy_bible()
        self.adapter.fetch_log = []
        cont = StoryContinuer(self.ctx)
        bible = cont.prepare(self.slug)
        # every chapter came from the ingest cache: the source was
        # never asked for a single chapter body
        self.assertEqual(self.adapter.fetch_log, [])
        self.assertEqual(bible.digestion_version, DIGESTION_VERSION)
        self.assertIn("crushed",
                      {c.name: c for c in bible.characters}
                      ["Quinn"].state)


class TestRedigestDegradesNeverBlocks(_RedigestCase):
    def test_continuation_served_while_redigest_is_paused(
            self) -> None:
        self._write_legacy_bible()
        self._break_chapter_text(5)
        cont, calls = self._continuer_with_recording_model()
        result = cont.continue_story(self.slug, n=1)
        # the continuation happened anyway, anchored at the true end
        self.assertEqual([c["number"] for c in result["chapters"]],
                         [9])
        self.assertEqual(result["continued_from_chapter"], 8)
        self.assertIn("WHERE THE STORY STOPPED", calls[0]["user"])
        self.assertIn("bell began to toll", calls[0]["user"])
        # and the rebuild was NOT half-marked: still pending, so the
        # next prepare resumes it instead of believing it done
        bible = BibleBuilder(self.ctx).load(self.slug)
        self.assertEqual(bible.digestion_version, 0)

    def test_paused_redigest_resumes_and_completes(self) -> None:
        self._write_legacy_bible()
        restore = self._break_chapter_text(5)
        cont = StoryContinuer(self.ctx)
        paused = cont.prepare(self.slug)
        self.assertEqual(paused.digestion_version, 0)
        self.assertEqual(paused.chapters_digested, 8)
        restore()
        bible = cont.prepare(self.slug)
        self.assertEqual(bible.digestion_version, DIGESTION_VERSION)
        by_name = {c.name: c for c in bible.characters}
        self.assertIn("crushed", by_name["Quinn"].state)

    def test_mid_redigest_resumes_without_refolding_covered_range(
            self) -> None:
        # a rebuild interrupted halfway: grounding folded through
        # chapter 4 (recap + situation on disk), the rest untouched,
        # no stamp — prepare must resume at 5, not restart at 1,
        # and must not duplicate the folded segment's recap
        self._write_legacy_bible()
        builder = BibleBuilder(self.ctx)
        path = builder.bible_path(self.slug)
        d = json.loads(path.read_text(encoding="utf-8"))
        d["grounding_through"] = 4
        d["recaps"] = [{"start": 1, "end": 4,
                        "text": "half-folded recap"}]
        d["current_situation"] = "End of chapter 4: earlier beat."
        path.write_text(json.dumps(d, indent=2), encoding="utf-8")
        cont = StoryContinuer(self.ctx)
        with self.assertLogs("nomorals.books.continuation",
                             level="INFO") as logs:
            bible = cont.prepare(self.slug)
        self.assertTrue(any("rebuilt" in line and "5–8" in line
                            for line in logs.output), logs.output)
        segments = [(r.start, r.end) for r in bible.recaps]
        self.assertEqual(segments.count((1, 4)), 1)
        self.assertIn((5, 8), segments)
        self.assertEqual(bible.grounding_through, 8)
        self.assertEqual(bible.digestion_version, DIGESTION_VERSION)
        self.assertIn("End of chapter 8", bible.current_situation)


if __name__ == "__main__":
    unittest.main()
