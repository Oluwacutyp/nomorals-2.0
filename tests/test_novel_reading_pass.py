"""Novel pass (2026-11-07) — follows become reading; reading pages.

The owner's defect: "the novel class doesn't seem to go reading
supposed novels I followed."  The trace: StoryReader.follow()
recorded the follow and stopped (no trigger, no consumer); the one
chat surface that wanted more hand-rolled a background INGEST
thread — library download only, and ingestion is explicitly not
reading; the brain-tool follow path triggered nothing at all; and
new chapters were never discovered (check_updates was a manual
tool nobody called).  Digestion into the story bible — her actually
reading — happened only inside an explicit /novel continue.

These pins lock the fixed shape:

- follow() publishes story.followed on the shared bus (the organ is
  the trigger, so EVERY follow path fires it);
- the reading engine consumes it immediately on its own worker:
  ingest + digest every chapter into the bible, her progress the
  bible's watermark, the owner's position untouched, a ping to the
  chat that followed;
- the discovery sweep (full enumeration, no interactive list cap)
  publishes story.chapters_discovered and the same consumer reads
  the new chapters on discovery — the sweep itself reads nothing;
- the story discovery job registers by the family pattern;
- chapter reading presents through the live pager (one message,
  Next/Prev, session "novel") on buttons+edit surfaces, and the
  pulse story list delivers as ONE paged message — no (i/n) labels,
  F6's full-delivery/confirmed-marking contract intact.

Fake adapter only — no network.
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
from nomorals.books.reader import StoryReader
from nomorals.books.sources import Chapter, SourceError, StoryMeta


def _ctx(db=None):
    tmp = tempfile.mkdtemp()
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(db=db, settings=settings, extras={}), tmp


class _PagedAdapter:
    """Fake archive (the test_books_ingest_continue shape)."""

    name = "freewebnovel"
    verified = True

    def __init__(self, total: int = 10, title: str = "My Vampire System",
                 fail_on=()) -> None:
        self.total = total
        self.title = title
        self.fail_on = set(fail_on)
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
        return Chapter(
            number=n, title=f"Chapter {n}", url=url, paragraphs=paras,
            prev_url=f"https://fake/ch/{n - 1}" if n > 1 else "",
            next_url=f"https://fake/ch/{n + 1}" if n < self.total else "",
            source=self.name)


_FAST_PROFILE = {"sync_chapters": 3, "list_limit": 3,
                 "pause_s": 0.0, "timeout": 5.0}
_URL = "https://fake/novel/my-vampire-system"


class _ReaderCase(unittest.TestCase):
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


def _wait_for(predicate, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


class FollowTriggerTests(_ReaderCase):
    def test_follow_publishes_story_followed(self):
        from nomorals.core.events import global_bus

        seen: list = []
        sub = global_bus.subscribe(
            "story.followed", lambda e: seen.append(e), sync=True)
        try:
            story = self.r.follow(_URL, chat_key="telegram:42")
        finally:
            global_bus.unsubscribe(sub)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].data["slug"], story.slug)
        self.assertEqual(seen[0].data["chat_key"], "telegram:42")
        self.assertEqual(seen[0].data["title"], "My Vampire System")


class EngineReadTests(_ReaderCase):
    def test_read_story_digests_everything_and_spares_owner_progress(self):
        from nomorals.books.reading_engine import StoryReadingEngine

        slug = self.r.follow(_URL).slug
        engine = StoryReadingEngine(self.ctx, chat_notify=lambda k, t: None)
        report = engine.read_story(slug, reason="follow")
        self.assertTrue(report["ok"], report)
        self.assertTrue(report["complete"])
        self.assertEqual(report["new_chapters"], list(range(1, 11)))
        self.assertEqual(report["digested_through"], 10)
        bible = BibleBuilder(self.ctx).load(slug)
        self.assertIsNotNone(bible)
        self.assertEqual(bible.chapters_digested, 10)
        # her reading never moves the OWNER's reading position
        story = StoryReader(self.ctx)._load(slug)
        self.assertEqual(story.current_chapter, 0)
        self.assertEqual(story.ingested_chapters, 10)
        # a second read is a no-op (nothing new to read)
        again = engine.read_story(slug, reason="sync")
        self.assertEqual(again["new_chapters"], [])
        self.assertEqual(again["digested_through"], 10)

    def test_follow_event_makes_the_engine_read_without_being_asked(self):
        from nomorals.books.reading_engine import StoryReadingEngine
        from nomorals.core.events import global_bus

        pings: list = []
        engine = StoryReadingEngine(
            self.ctx,
            chat_notify=lambda k, t: pings.append((k, t)),
            bus=global_bus)
        engine.start()
        try:
            story = self.r.follow(_URL, chat_key="telegram:42")
            slug = story.slug
            ok = _wait_for(
                lambda: (BibleBuilder(self.ctx).load(slug) is not None
                         and BibleBuilder(self.ctx).load(slug)
                         .chapters_digested == 10))
            self.assertTrue(ok, "engine never finished reading the follow")
        finally:
            engine.stop()
        self.assertEqual(len(pings), 1)
        chat_key, text = pings[0]
        self.assertEqual(chat_key, "telegram:42")
        self.assertIn("she's read", text)
        self.assertIn("all 10 chapters", text)

    def test_catch_up_reads_a_library_left_undigested(self):
        from nomorals.books.reading_engine import StoryReadingEngine

        slug = self.r.follow(_URL).slug
        self.r.ingest(slug)  # library complete, nobody read it
        self.assertIsNone(BibleBuilder(self.ctx).load(slug))
        engine = StoryReadingEngine(self.ctx, chat_notify=lambda k, t: None)
        queued = engine.catch_up_all()
        self.assertEqual(queued, [slug])
        report = engine.read_story(slug, reason="catchup")
        self.assertEqual(report["digested_through"], 10)
        # catch-up runs are silent by contract
        self.assertEqual(
            StoryReadingEngine._announce_text(report, reason="catchup"),
            "")


class _FakeRegistry:
    def __init__(self, context):
        self.context = context
        self.fns: dict = {}

    def register(self, name, **kw):
        def deco(fn):
            self.fns[name] = fn
            return fn
        return deco


class DiscoveryTests(_ReaderCase):
    def _tools(self):
        from nomorals.books import tools as book_tools

        reg = _FakeRegistry(self.ctx)
        book_tools.register(reg)
        return reg.fns

    def test_sweep_publishes_discovery_and_engine_reads_the_new(self):
        from nomorals.books.reading_engine import StoryReadingEngine

        slug = self.r.follow(_URL).slug
        engine = StoryReadingEngine(self.ctx, chat_notify=lambda k, t: None)
        first = engine.read_story(slug, reason="follow")
        self.assertEqual(first["digested_through"], 10)
        # three new chapters land at the source (past the profile's
        # interactive list cap of 3 — discovery must still see them)
        self.adapter.total = 13
        fns = self._tools()
        # the standing consumer is subscribed BEFORE the sweep runs —
        # the discovery event is what triggers the reading
        engine.start()
        try:
            result = fns["story_discovery_sweep"]()
            self.assertEqual(result["checked"], 1)
            self.assertEqual(len(result["discovered"]), 1)
            self.assertEqual(result["discovered"][0]["new_chapters"],
                             [11, 12, 13])
            ok = _wait_for(
                lambda: BibleBuilder(self.ctx).load(slug)
                .chapters_digested == 13)
            self.assertTrue(ok, "discovered chapters were never read")
        finally:
            engine.stop()
        # a second sweep finds nothing new and publishes nothing
        result2 = fns["story_discovery_sweep"]()
        self.assertEqual(result2["discovered"], [])


class DiscoveryJobTests(unittest.TestCase):
    def test_job_registers_once_with_the_sweep_tool_payload(self):
        from nomorals.books.jobs import STORY_DISCOVERY_JOB, ensure_story_jobs

        class FakeScheduler:
            def __init__(self):
                self.jobs: list = []

            def list_jobs(self):
                return list(self.jobs)

            def add(self, name, spec, kind, payload, routine=False):
                job = {"id": f"job-{len(self.jobs)}", "name": name,
                       "spec": spec, "kind": kind, "payload": payload,
                       "routine": routine}
                self.jobs.append(job)
                return job

        sched = FakeScheduler()
        out = ensure_story_jobs(sched)
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0]["scheduled"])
        job = sched.jobs[0]
        self.assertEqual(job["name"], STORY_DISCOVERY_JOB)
        self.assertEqual(job["kind"], "tool")
        self.assertEqual(job["payload"],
                         {"tool": "story_discovery_sweep", "args": {}})
        self.assertTrue(job["routine"])
        out2 = ensure_story_jobs(sched)
        self.assertTrue(out2[0]["already_scheduled"])
        self.assertEqual(len(sched.jobs), 1)


class PagerRenderTests(unittest.TestCase):
    def test_chapter_pages_partition_the_text_and_carry_the_grammar(self):
        from nomorals.agents.partner.runtime_search import (
            _chapter_actionable,
            _reading_pager_ok,
        )
        from nomorals.social.chat.chunking import chunk_message

        text = "\n\n".join(
            f"Paragraph {i} carries the story forward with enough "
            f"substance to matter, sentence after sentence, beat "
            f"after beat, until the chapter turns." for i in range(60))
        expected = chunk_message(text, max_text=3000)
        self.assertGreater(len(expected), 1)
        slug = "my-vampire-system-freewebnovel"
        # 31 chars: over the worst-case 47-byte budget, so the
        # buttons carry the resolvable 24-char prefix (see _page_ref)
        ref = slug[:24]
        first = _chapter_actionable(
            kind="novel", slug=slug, title="My Vampire System",
            chapter=3, chapter_title="Chapter 3", text=text, page=1,
            total_chapters=900, has_next=True, session="novel",
            edit=False)
        self.assertEqual(first.session, "novel")
        self.assertFalse(first.session_edit)
        datas = [d for row in first.buttons for _l, d in row]
        self.assertIn(f"novel page {ref} 3 2", datas)
        self.assertIn(f"novel read {ref} 4", datas)
        self.assertIn(f"novel read {ref} 2", datas)
        # page bodies partition the chapter text exactly (lossless)
        bodies = []
        for p in range(1, len(expected) + 1):
            render = _chapter_actionable(
                kind="novel", slug=slug, title="My Vampire System",
                chapter=3, chapter_title="Chapter 3", text=text, page=p,
                total_chapters=900, has_next=True, session="novel",
                edit=True)
            self.assertTrue(render.session_edit)
            body = render.text.split("\n\n", 1)[1].rsplit("\n\n— page", 1)[0]
            bodies.append(body)
        self.assertEqual(bodies, list(expected))
        # surface gate: buttons+edit surfaces page; WhatsApp/SMS don't
        self.assertTrue(_reading_pager_ok("telegram:1"))
        self.assertTrue(_reading_pager_ok("telegram-bot:1"))
        self.assertFalse(_reading_pager_ok("whatsapp:1"))
        self.assertFalse(_reading_pager_ok("sms:1"))

    def test_page_ref_shortens_long_slugs_to_a_resolvable_prefix(self):
        from nomorals.agents.partner.runtime_search import _page_ref

        long_slug = "a-very-long-story-title-that-goes-on-forever-and-ever"
        ref = _page_ref("novel", long_slug)
        self.assertEqual(ref, long_slug[:24])
        self.assertLessEqual(
            len(f"novel page {ref} 888 888".encode()), 47)
        self.assertEqual(_page_ref("novel", "short-slug"), "short-slug")


class NovelSurfaceTests(_ReaderCase):
    def _mixin(self):
        from nomorals.agents.partner.runtime_search import RuntimeSearchMixin

        m = RuntimeSearchMixin.__new__(RuntimeSearchMixin)
        m.context = self.ctx
        return m

    def test_read_renders_through_the_pager_on_telegram(self):
        from nomorals.agents.partner.runtime_search import _page_ref

        slug = self.r.follow(_URL).slug
        ref = _page_ref("novel", slug)
        # the button reference resolves back to the story it names
        self.assertEqual(self.r.resolve_slug(ref), slug)
        m = self._mixin()
        reply = m._control_novel(f"read {slug} 1", "telegram:123")
        self.assertTrue(hasattr(reply, "text") and hasattr(reply, "buttons"),
                        f"expected an actionable pager render, got {reply!r}")
        self.assertEqual(reply.session, "novel")
        self.assertIn("chapter 1", reply.text)
        datas = [d for row in reply.buttons for _l, d in row]
        self.assertIn(f"novel read {ref} 2", datas)
        # the tap grammar re-enters and renders page/another chapter
        reply2 = m._control_novel(f"read {ref} 2", "telegram:123")
        self.assertIn("chapter 2", reply2.text)

    def test_read_falls_back_to_the_chunked_send_without_a_pager(self):
        slug = self.r.follow(_URL).slug
        m = self._mixin()
        sent: list = []
        m._ref_from_key = lambda key: SimpleNamespace(platform="whatsapp")
        m._send_long_checked = (
            lambda platform, chat, text: sent.append(text) or "")
        reply = m._control_novel(f"read {slug} 1", "whatsapp:123")
        self.assertEqual(reply, "")
        self.assertEqual(len(sent), 1)
        self.assertIn("Quinn crossed the yard in chapter 1", sent[0])


class PulsePagerTests(unittest.TestCase):
    """The pulse story list: ONE paged publish, F6 marking intact."""

    def setUp(self) -> None:
        from nomorals.storage.db import Database

        tmp = tempfile.mkdtemp()
        self.db = Database(os.path.join(tmp, "test.db"))
        self.db.migrate()
        partner = SimpleNamespace(
            owner_chats="telegram:111", proactive_enabled=True,
            proactive_briefing=True, proactive_watchers=True,
            quiet_start=22, quiet_end=8, timezone="UTC")
        self.ctx = SimpleNamespace(
            db=self.db,
            settings=SimpleNamespace(workspace_dir=tmp, partner=partner,
                                     pulse=None),
            extras={})

    def _seed(self, n: int, summary_words: int = 2):
        from nomorals.agents.news import NewsAgent
        from nomorals.core.ids import new_id

        NewsAgent(self.ctx)
        urls = []
        for i in range(n):
            url = f"https://news.test/s{i}/full"
            summary = " ".join(
                f"Detail{i} sentence number {j} carries real substance."
                for j in range(summary_words))
            self.db.execute(
                "INSERT INTO news_items (id, source, publisher, title, "
                "url, summary, published, created_at, delivery_state) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (new_id(), "testfeed", "TestWire", f"Story {i} headline",
                 url, summary, 1000.0 + i, time.time(), "undelivered"))
            urls.append(url)
        return urls

    def _gateway(self, fail: bool = False):
        class _Result:
            ok = not fail
            message_id = "m1"
            error = "boom (fake)" if fail else ""

        class FakeGateway:
            def __init__(self):
                self.sent: list = []

            def status(self):
                return {"telegram-bot": {"running_in_session": True}}

            def send(self, platform, chat_ref, text, **kwargs):
                self.sent.append((platform, chat_ref.chat_id, text,
                                  kwargs.get("buttons")))
                return _Result()

            def platforms_for_chat_id(self, chat_id):
                return []

        gw = FakeGateway()
        self.ctx.gateway = gw
        return gw

    def test_forty_stories_one_paged_publish_all_marked(self):
        from nomorals.agents import morning_pulse as mp
        from nomorals.agents.news import NewsAgent

        urls = self._seed(40)
        gw = self._gateway()
        items = NewsAgent(self.ctx).undelivered()
        marked = mp._deliver_stories(self.ctx, "morning", items)
        self.assertEqual(sorted(marked), sorted(urls))
        self.assertEqual(NewsAgent(self.ctx).undelivered(), [])
        # exactly one story-list send, paged — no (i/n) dump labels
        story_sends = [s for s in gw.sent if mp.STORY_LIST_TITLE in s[2]]
        self.assertEqual(len(story_sends), 1)
        _plat, _chat, text, buttons = story_sends[0]
        self.assertNotIn("(1/", text)
        self.assertIn("— page 1 of", text)
        flat = [d for row in (buttons or []) for _l, d in row]
        self.assertTrue(any(d.startswith("pulse stories ") for d in flat),
                        f"no pager buttons on the delivery: {buttons!r}")
        # every story is reachable exactly once across the pages
        seen: list = []
        page = 1
        while True:
            render = mp.story_page_render(self.ctx, "latest", page)
            self.assertIsNotNone(render)
            for u in urls:
                if u in render["text"]:
                    seen.append(u)
            if page >= render["pages"]:
                break
            page += 1
        self.assertEqual(sorted(seen), sorted(urls))

    def test_failed_publish_marks_nothing_and_the_list_resumes(self):
        from nomorals.agents import morning_pulse as mp
        from nomorals.agents.news import NewsAgent

        urls = self._seed(40)
        self._gateway(fail=True)
        items = NewsAgent(self.ctx).undelivered()
        marked = mp._deliver_stories(self.ctx, "morning", items)
        self.assertEqual(marked, [])
        remaining = {str(r.get("url"))
                     for r in NewsAgent(self.ctx).undelivered()}
        self.assertEqual(remaining, set(urls))
        # healthy gateway next pulse: the whole list delivers
        self._gateway()
        marked2 = mp._deliver_stories(
            self.ctx, "morning", NewsAgent(self.ctx).undelivered())
        self.assertEqual(sorted(marked2), sorted(urls))


if __name__ == "__main__":
    unittest.main()
