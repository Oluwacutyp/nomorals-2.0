"""Pass #57 — follow INGESTS the whole story; continue anchors at the
story's TRUE END; story names resolve.

Fake adapter only (NovelFull-shaped pagination) — no network.  The
owner's bug: followed 'My Vampire System', read chapter 1, and
``/novel continue my-vampire-system-freewebnovel`` wrote a "Chapter 2"
of recycled filler from a near-empty bible, while
``my_vampire_system`` (underscores) failed with "no followed story".
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.books import reader as reader_mod
from nomorals.books.bible import BibleBuilder
from nomorals.books.continuation import ContinuationError, StoryContinuer
from nomorals.books.reader import ReaderError, StoryReader
from nomorals.books.sources import Chapter, SourceError, StoryMeta


def _ctx():
    tmp = tempfile.mkdtemp()
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(db=None, settings=settings), tmp


class _PagedAdapter:
    """A paginated fake archive: ``chapter_list(limit=N)`` returns at
    most N entries (NovelFull semantics); ``limit=0`` walks all pages.
    """

    name = "freewebnovel"
    verified = True

    def __init__(self, total: int = 10, title: str = "My Vampire System",
                 fail_on=()) -> None:
        self.total = total
        self.title = title
        self.fail_on = set(fail_on)
        self.fetch_log: list[int] = []
        self.list_limits: list[int] = []

    def novel(self, url: str) -> StoryMeta:
        return StoryMeta(title=self.title, url=url, source=self.name,
                         author="JKSManga", synopsis="A vampire system.",
                         status="Ongoing", total_chapters=self.total)

    def chapter_list(self, novel_url: str, *, limit: int = 0):
        self.list_limits.append(limit)
        out = [(n, f"Chapter {n}", f"https://fake/ch/{n}")
               for n in range(1, self.total + 1)]
        return out[:limit] if limit else out

    def chapter_url(self, novel_url: str, number: int) -> str:
        return f"https://fake/ch/{number}" if 1 <= number <= self.total else ""

    def fetch_chapter(self, url: str) -> Chapter:
        n = int(url.rsplit("/", 1)[-1])
        self.fetch_log.append(n)
        if n in self.fail_on:
            raise SourceError(f"fake source died on chapter {n}")
        paras = [
            f"Quinn crossed the yard in chapter {n} and the mark {n} "
            f"burned cold on his wrist. EMBER-MARKER-{n}",
            '"We move at dusk," Mara said. "The Vigil cannot be '
            'reasoned with; it can only be outrun."',
            "The old rule held: the thirst comes first, and it does "
            "not negotiate.",
        ]
        if n == self.total:
            paras.append("Then, at the very end of all things, the "
                         "ember gate swung open. FINAL-LINE-TEN")
        return Chapter(
            number=n, title=f"Chapter {n}", url=url, paragraphs=paras,
            prev_url=f"https://fake/ch/{n - 1}" if n > 1 else "",
            next_url=f"https://fake/ch/{n + 1}" if n < self.total else "",
            source=self.name)


_FAST_PROFILE = {"sync_chapters": 3, "list_limit": 3,
                 "pause_s": 0.0, "timeout": 5.0}


class _ReaderCase(unittest.TestCase):
    """Reader wired to the fake adapter with a tiny profile list cap."""

    def setUp(self) -> None:
        self.ctx, self.tmp = _ctx()
        self.adapter = _PagedAdapter()
        self._orig_adapter = reader_mod.adapter_for_url
        reader_mod.adapter_for_url = lambda url: self.adapter
        self._orig_profile = StoryReader._profile
        StoryReader._profile = lambda self: dict(_FAST_PROFILE)
        self.r = StoryReader(self.ctx)

    def tearDown(self) -> None:
        reader_mod.adapter_for_url = self._orig_adapter
        StoryReader._profile = self._orig_profile

    def follow(self) -> str:
        story = self.r.follow("https://fake/novel/my-vampire-system")
        return story.slug


class TestIngest(_ReaderCase):
    def test_ingest_reaches_final_chapter_despite_list_cap(self) -> None:
        slug = self.follow()
        # the profile list cap really does truncate a normal listing
        self.assertEqual(
            len(self.adapter.chapter_list("https://fake/x", limit=3)), 3)
        report = self.r.ingest(slug)
        self.assertTrue(report["complete"])
        self.assertEqual(report["ingested"], 10)
        self.assertEqual(report["total"], 10)
        self.assertEqual(report["fetched"], list(range(1, 11)))
        # full enumeration was requested (limit=0), not the profile cap
        self.assertIn(0, self.adapter.list_limits)
        # progress persisted on the record, readable by a fresh reader
        fresh = StoryReader(self.ctx)._load(slug)
        self.assertEqual(fresh.ingested_chapters, 10)
        self.assertEqual(fresh.ingest_total, 10)
        chapters_dir = self.r.story_dir(slug) / "chapters"
        self.assertEqual(len(list(chapters_dir.glob("*.md"))), 10)

    def test_ingest_resumes_fetching_only_missing_chapters(self) -> None:
        slug = self.follow()
        for n in (1, 2, 3, 4):
            self.r.read(slug, chapter=n)
        self.adapter.fetch_log = []  # only count the ingest's fetches
        self.adapter.fail_on = {7}
        first = self.r.ingest(slug)
        self.assertFalse(first["complete"])
        self.assertEqual(first["ingested"], 6)  # watermark stops at the gap
        self.assertEqual([f["number"] for f in first["failed"]], [7])
        self.assertEqual(self.adapter.fetch_log, [5, 6, 7, 8, 9, 10])
        # source heals; the second run fetches ONLY the missing chapter
        self.adapter.fail_on = set()
        self.adapter.fetch_log = []
        second = self.r.ingest(slug)
        self.assertTrue(second["complete"])
        self.assertEqual(self.adapter.fetch_log, [7])
        self.assertEqual(second["ingested"], 10)

    def test_ingest_never_moves_reading_position(self) -> None:
        slug = self.follow()
        self.r.read(slug, chapter=1)
        self.r.ingest(slug)
        self.assertEqual(self.r.get_progress(slug)["chapter"], 1)
        self.assertEqual(StoryReader(self.ctx)._load(slug).current_chapter, 1)

    def test_ingest_state_reports_honestly(self) -> None:
        slug = self.follow()
        state = self.r.ingest_state(slug)
        self.assertEqual(state["ingested"], 0)
        self.assertFalse(state["complete"])
        self.r.ingest(slug)
        state = self.r.ingest_state(slug)
        self.assertEqual(state["ingested"], 10)
        self.assertTrue(state["complete"])


class TestContinueFromTrueEnd(_ReaderCase):
    def _continuer_with_recording_composer(self):
        cont = StoryContinuer(self.ctx)
        seen: list[dict] = []

        def fake_compose(bible, chapter_no, tails, words, direction,
                         seed_salt=0):
            seen.append({"chapter_no": chapter_no, "tails": list(tails),
                         "digested": bible.chapters_digested})
            return (f"DEVON-WROTE-{chapter_no} — Quinn stepped through "
                    "the ember gate and the night held its breath. "
                    '"We end this," Mara said.')

        cont._compose_continuation = fake_compose
        return cont, seen

    def test_continue_after_reading_chapter_1_writes_chapter_11(self) -> None:
        slug = self.follow()
        self.r.read(slug, chapter=1)  # the owner's exact position
        cont, seen = self._continuer_with_recording_composer()
        result = cont.continue_story(slug, n=1)
        self.assertEqual(result["continued_from_chapter"], 10)
        self.assertEqual(result["final_chapter"], 10)
        self.assertEqual([c["number"] for c in result["chapters"]], [11])
        # the bible was digested through the FINAL chapter before writing
        self.assertEqual(seen[0]["digested"], 10)
        # the writer got the actual tails: ch.9 and the story's last line
        joined = "\n".join(seen[0]["tails"])
        self.assertIn("EMBER-MARKER-9", joined)
        self.assertIn("FINAL-LINE-TEN", joined)
        # the written chapter is persisted in the story's library
        path = (self.r.story_dir(slug) / "continuations"
                / "chapter-00011.md")
        self.assertTrue(path.exists())
        # writing never moved the owner's reading position
        self.assertEqual(self.r.get_progress(slug)["chapter"], 1)
        # and the bible now digests through the written chapter too
        bible = BibleBuilder(self.ctx).load(slug)
        self.assertGreaterEqual(bible.chapters_digested, 11)

    def test_second_continue_builds_on_first(self) -> None:
        slug = self.follow()
        cont, seen = self._continuer_with_recording_composer()
        cont.continue_story(slug, n=1)
        result = cont.continue_story(slug, n=1)
        self.assertEqual([c["number"] for c in result["chapters"]], [12])
        self.assertEqual(result["continued_from_chapter"], 11)
        joined = "\n".join(seen[-1]["tails"])
        self.assertIn("DEVON-WROTE-11", joined)   # builds on chapter 11
        self.assertIn("FINAL-LINE-TEN", joined)   # and still on the end

    def test_continue_refuses_a_partial_library(self) -> None:
        slug = self.follow()
        self.adapter.fail_on = {7}
        cont, _seen = self._continuer_with_recording_composer()
        with self.assertRaises(ContinuationError) as cm:
            cont.continue_story(slug, n=1)
        message = str(cm.exception)
        self.assertIn("6 of 10", message)
        self.assertIn("chapter 7", message)
        # nothing was written from the partial library
        out_dir = self.r.story_dir(slug) / "continuations"
        self.assertFalse(list(out_dir.glob("chapter-*.md")))


class TestSlugResolution(_ReaderCase):
    def test_variants_resolve(self) -> None:
        slug = self.follow()
        self.assertEqual(slug, "my-vampire-system-freewebnovel")
        for variant in ("my_vampire_system", "my-vampire-system",
                        "My Vampire System",
                        "my-vampire-system-freewebnovel",
                        "MY-VAMPIRE-SYSTEM-FREEWEBNOVEL",
                        "my-vampire"):
            with self.subTest(variant=variant):
                self.assertEqual(self.r.resolve_slug(variant), slug)

    def test_owner_failing_input_now_reads(self) -> None:
        slug = self.follow()
        # the exact shape that failed live: underscores, no source suffix
        data = self.r.read("my_vampire_system", chapter=1)
        self.assertEqual(data["slug"], slug)
        self.assertEqual(data["chapter"], 1)

    def test_unknown_stays_honest_and_creates_nothing(self) -> None:
        self.follow()
        with self.assertRaises(ReaderError) as cm:
            self.r.read("no-such-story")
        self.assertIn("no followed story", str(cm.exception))
        # a failed lookup must not conjure a story directory
        self.assertFalse(
            (self.r.reader_dir() / "no-such-story").exists())

    def test_ambiguous_prefix_lists_candidates(self) -> None:
        self.follow()
        other = _PagedAdapter(title="My Vampire System Returns")
        reader_mod.adapter_for_url = lambda url: other
        try:
            self.r.follow("https://fake/novel/my-vampire-system-returns")
        finally:
            reader_mod.adapter_for_url = lambda url: self.adapter
        with self.assertRaises(ReaderError) as cm:
            self.r.resolve_slug("my-vampire")
        message = str(cm.exception)
        self.assertIn("ambiguous", message)
        self.assertIn("my-vampire-system-freewebnovel", message)
        self.assertIn("my-vampire-system-returns-freewebnovel", message)
        # ...but the exact base name still resolves to the first story
        self.assertEqual(self.r.resolve_slug("my-vampire-system"),
                         "my-vampire-system-freewebnovel")


class TestChatFollowReply(unittest.TestCase):
    """The /novel follow reply states ingestion honestly and the
    background ingest actually completes the library."""

    def setUp(self) -> None:
        from nomorals.agents import partner_runtime as facade
        from nomorals.agents.context import build_context
        from nomorals.core.config import load_settings

        self.tmp = tempfile.TemporaryDirectory(prefix="nm-novel-")
        settings = load_settings(overrides={
            "home": self.tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
        })
        self.context = build_context(settings, with_executor=False,
                                     with_tools=False)

        class _Router:
            def chat(self, *a, **k):
                raise AssertionError("novel follow must not call the model")

        self.context.router = _Router()

        class _StubGateway:
            adapters: dict = {}

            def status(self):
                return {}

            def set_rate_limit(self, *a, **k):
                pass

            def send(self, *a, **k):
                return {}

        self.runtime = facade.PartnerRuntime(self.context,
                                             gateway=_StubGateway())
        self.adapter = _PagedAdapter()
        self._orig_adapter = reader_mod.adapter_for_url
        reader_mod.adapter_for_url = lambda url: self.adapter

    def tearDown(self) -> None:
        reader_mod.adapter_for_url = self._orig_adapter
        try:
            self.runtime.stop()
        except Exception:
            pass
        self.context.close()
        self.tmp.cleanup()

    def test_follow_reply_reports_ingestion_and_it_completes(self) -> None:
        reply = self.runtime.handle_control(
            "/novel follow https://fake/novel/my-vampire-system",
            "local:console")
        # honest NOW-state: nothing ingested yet, background job running
        self.assertIn("Ingesting the whole story", reply)
        self.assertIn("0 of 10", reply)
        self.assertIn("my-vampire-system-freewebnovel", reply)
        reader = StoryReader(self.context)
        deadline = time.time() + 30
        state = reader.ingest_state("my-vampire-system-freewebnovel")
        while not state["complete"] and time.time() < deadline:
            time.sleep(0.25)
            state = reader.ingest_state("my-vampire-system-freewebnovel")
        self.assertTrue(state["complete"], f"ingest never finished: {state}")
        self.assertEqual(state["ingested"], 10)
        # a second follow tells the truth: the library is already whole
        reply2 = self.runtime.handle_control(
            "/novel follow https://fake/novel/my-vampire-system",
            "local:console")
        self.assertIn("already in its library", reply2)


if __name__ == "__main__":
    unittest.main()
