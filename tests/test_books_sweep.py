"""Books sweep tests: mined-then-built upgrades across the books module.

Covers the new behavior added in the sweep:
- model: chapter takes, progress/reading-time, new metadata fields
- styles: themes, cards, bars, beat sheets, trees, retention
- bible: NovelAI-style lorebook (keys incl. regex/AND, always-on,
  chain activation, injection), mention index, progressions,
  consistency checks, export/import
- fiction: Save-the-Cat beat sheets, scene beats, voice hooks,
  character psychology, XianxiaEngine
- continuation: AI-Dungeon-ordered prompts, alternate takes + scoring,
  rolling summaries
- write: expand/describe/rewrite/suggest_hooks primitives
- outline: beat_sheet_outline, three_act_map
- forge: stdlib EPUB3 builder, single-file HTML, md→xhtml
- sources: normalize_url, fetcher rate limiting, download_story
- library: sessions/stats/streaks, annotation export, series
- reader: highlights, check_updates, catch_up, reading_stats
- branches: choices, tree, diff, rename, merge strategies
- publish: weekday cadences, backlog buffer, author notes,
  launch plan, retention curve
- collab: StyleAnalyzer, CharacterSheet, critic personas
- tools: registration integrity (73 tools, no duplicates)
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _ctx():
    tmp = tempfile.mkdtemp()
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(db=None, settings=settings), tmp


# ── model ────────────────────────────────────────────────────────────────────


class TestModelSweep(unittest.TestCase):
    def test_chapter_takes(self):
        from nomorals.books.model import Chapter
        c = Chapter(number=1, text="live text")
        take = c.add_take("alternate text", "alt-1")
        self.assertEqual(take["label"], "alt-1")
        self.assertEqual(len(c.takes), 1)
        self.assertTrue(c.use_take("alt-1"))
        self.assertEqual(c.text, "alternate text")
        self.assertFalse(c.use_take("nope"))
        d = Chapter.from_dict(c.to_dict())
        self.assertEqual(d.takes[0]["label"], "alt-1")

    def test_book_progress_and_meta(self):
        from nomorals.books.model import Book, Chapter, STATUS_WRITTEN
        b = Book(topic="t", slug="s",
                 chapters=[Chapter(number=1, status=STATUS_WRITTEN,
                                   text="w " * 400),
                           Chapter(number=2, word_target=800)])
        self.assertEqual(b.progress_pct, 50.0)
        self.assertEqual(b.reading_minutes, 2)
        self.assertEqual(b.words_remaining(), 800)
        b2 = Book(topic="t", slug="s", language="yo", series="Saga",
                  series_index=2, cover_image="c.png")
        d = Book.from_dict(b2.to_dict())
        self.assertEqual((d.language, d.series, d.series_index,
                          d.cover_image), ("yo", "Saga", 2.0, "c.png"))
        # old dicts without the new fields still load
        d3 = Book.from_dict({"topic": "t", "slug": "s"})
        self.assertEqual(d3.language, "en")


# ── styles ───────────────────────────────────────────────────────────────────


class TestStyles(unittest.TestCase):
    def test_progress_bar(self):
        from nomorals.books.styles import progress_bar
        bar = progress_bar(50, width=10, theme="rich")
        self.assertIn("50%", bar)
        self.assertEqual(bar.count("█"), 5)
        plain = progress_bar(25, width=8, theme="plain")
        self.assertIn("#", plain)
        self.assertIn("25%", plain)

    def test_book_card(self):
        from nomorals.books.styles import render_book_card
        card = render_book_card({"display_title": "T", "author": "A",
                                 "chapters_written": 3, "total_chapters": 6,
                                 "words": 9000, "status": "writing",
                                 "progress_pct": 50.0})
        self.assertIn("T", card)
        self.assertIn("50%", card)
        mini = render_book_card({"display_title": "T"}, theme="minimal")
        self.assertIn("T", mini)

    def test_beat_tree_retention(self):
        from nomorals.books.styles import (render_beat_sheet, render_tree,
                                            render_retention)
        beats = [{"chapter": 1, "beat": "Opening Image", "position_pct": 0.5,
                  "note": "before-world"}]
        self.assertIn("Opening Image", render_beat_sheet(beats))
        tree = {"name": "canon", "info": "2 ch",
                "children": [{"name": "b1", "info": "fork@1",
                              "children": []}]}
        out = render_tree(tree)
        self.assertIn("canon", out)
        self.assertIn("b1", out)
        curve = [{"chapter": 1, "retention_pct": 100.0},
                 {"chapter": 2, "retention_pct": 40.0, "drop": True}]
        rendered = render_retention(curve)
        self.assertIn("ch.2", rendered)


# ── bible lorebook ───────────────────────────────────────────────────────────


class TestLorebook(unittest.TestCase):
    def test_key_matching(self):
        from nomorals.books.bible import LorebookEntry
        e = LorebookEntry(name="Aiko", keys=["blade"], aliases=["Ai"])
        self.assertTrue(e.matches("Aiko drew her sword"))
        self.assertTrue(e.matches("the BLADE sang"))
        self.assertTrue(e.matches("Ai ran"))
        self.assertFalse(e.matches("nothing here"))
        # AND keys
        pact = LorebookEntry(name="pact", keys=["blood & oath"])
        self.assertTrue(pact.matches("the blood oath sealed"))
        self.assertFalse(pact.matches("just blood here"))
        # regex keys
        rx = LorebookEntry(name="rx", keys=["/Q[a-z]+/"])
        self.assertTrue(rx.matches("Qing arrived"))
        self.assertFalse(rx.matches("qing arrived"))

    def test_inject_order_and_budget(self):
        from nomorals.books.bible import StoryBible, LorebookEntry
        entries = [
            LorebookEntry(name="world", text="magic costs blood",
                          always_on=True, order=1),
            LorebookEntry(name="Aiko", kind="character",
                          text="a blade-dancer", keys=["blade"], order=50),
            LorebookEntry(name="off", text="never fires", enabled=False,
                          keys=["zzz"]),
        ]
        b = StoryBible(story_slug="s", title="T", lore=entries)
        inj = b.inject("Aiko raised the blade high")
        self.assertIn("magic costs blood", inj)
        self.assertIn("blade-dancer", inj)
        # always-on comes first
        self.assertLess(inj.index("magic"), inj.index("blade-dancer"))
        # disabled entry never fires
        self.assertNotIn("never fires", inj)
        # nothing relevant → only always-on
        inj2 = b.inject("a quiet morning with tea")
        self.assertIn("magic costs blood", inj2)
        self.assertNotIn("blade-dancer", inj2)

    def test_chain_activation(self):
        from nomorals.books.bible import StoryBible, LorebookEntry
        # 'elder' fires on the window; its text mentions 'sect', which
        # chain-activates the Iron Sect entry (NovelAI/1667 behavior)
        elder = LorebookEntry(name="Elder Sable", kind="character",
                              text="Elder Sable rules the sect with an iron fan",
                              keys=["elder"])
        sect = LorebookEntry(name="Iron Sect", kind="faction",
                             text="the Iron Sect taxes every village",
                             keys=["sect"])
        b = StoryBible(story_slug="s", title="T", lore=[elder, sect])
        inj = b.inject("the elder arrived at dawn")
        self.assertIn("Elder Sable", inj)
        self.assertIn("Iron Sect", inj)

    def test_mention_index_and_progressions(self):
        from nomorals.books.bible import StoryBible, BibleCharacter
        b = StoryBible(
            story_slug="s", title="T",
            characters=[BibleCharacter(name="Aiko", role="protagonist")])
        idx = b.mention_index([(1, "a", "Aiko walked in"),
                               (2, "b", "nothing here"),
                               (3, "c", "Aiko and Mara talked")])
        self.assertEqual(idx, {"Aiko": [1, 3]})
        p = b.add_progression("Aiko", "scarred across the left eye", chapter=3)
        self.assertEqual(p.chapter, 3)
        self.assertEqual(len(b.progressions_for("aiko")), 1)

    def test_consistency_check_and_export(self):
        from nomorals.books.bible import StoryBible, LorebookEntry, PlotThread
        entries = [LorebookEntry(name=f"e{i}", text="x", always_on=True)
                   for i in range(10)]
        entries.append(LorebookEntry(name="keyless", text="y"))
        b = StoryBible(story_slug="s", title="T", lore=entries,
                       threads=[PlotThread(id="t1", summary="cold", heat=0.0,
                                           last_seen=3)])
        flags = b.consistency_check()
        self.assertTrue(any("always-on bloat" in f for f in flags))
        self.assertTrue(any("keyless" in f for f in flags))
        self.assertTrue(any("cold" in f for f in flags))
        exported = b.export_lorebook()
        b2 = StoryBible(story_slug="s2", title="T2")
        added = b2.import_lorebook(exported)
        self.assertEqual(added, 11)
        self.assertEqual(b2.import_lorebook(exported), 0)  # dedup

    def test_builder_syncs_lorebook(self):
        from nomorals.books.bible import BibleBuilder
        ctx, tmp = _ctx()
        bb = BibleBuilder(ctx)
        chapters = [
            (1, "Ch1", 'Mara said, "We ride at dawn." Quinn nodded. '
                       'The old rule held: the thirst comes first. ' * 10),
            (2, "Ch2", 'Mara drew her blade. "Quinn, run," she whispered. '
                       'The thirst comes first, always. ' * 10),
        ]
        bible = bb.build("sweep-bible", "Sweep Tale", chapters)
        self.assertTrue(len(bible.lore) > 0)
        kinds = {e.kind for e in bible.lore}
        self.assertIn("character", kinds)
        # round-trips through disk
        loaded = bb.load("sweep-bible")
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded.lore), len(bible.lore))


# ── fiction ──────────────────────────────────────────────────────────────────


class TestFictionSweep(unittest.TestCase):
    def test_beat_sheet_positions(self):
        from nomorals.books.fiction import BeatSheet
        bs = BeatSheet(20)
        self.assertEqual(bs.beat_for(10)["beat"], "Midpoint")
        self.assertEqual(bs.beat_for(15)["beat"], "All Is Lost")
        self.assertEqual(bs.beat_for(20)["beat"], "Final Image")
        self.assertEqual(len(bs.full()), 15)
        nxt = bs.beat_for(1)
        self.assertIsNotNone(nxt["next_beat"])

    def test_scene_beats_and_voice(self):
        from nomorals.books.fiction import (engine_for, ChapterBrief,
                                             Arc, StoryState)
        e = engine_for("thriller", seed=7)
        voice = e.voice()
        self.assertIn("pov", voice)
        brief = ChapterBrief(chapter_no=3, arc_name="A",
                             must_happen=["a", "b", "c", "d"],
                             plant=["seed x"], tension_target=8.0)
        scenes = e.scene_beats(brief, n=3)
        self.assertGreaterEqual(len(scenes), 3)
        self.assertTrue(any("hook" in s.lower() for s in scenes))
        self.assertTrue(any("PLANT" in s for s in scenes))
        note = e.beat_note(10, 20)
        self.assertIn("Midpoint", note)
        self.assertIn("thriller", note)

    def test_cast_psychology(self):
        from nomorals.books.fiction import engine_for
        e = engine_for("mystery", seed=3)
        cast = e._cast("a murder", 3, ["detective", "suspect", "suspect"])
        c = cast[0]
        for field in ("want", "need", "wound", "lie", "ghost", "arc"):
            self.assertTrue(c[field], field)
        self.assertNotEqual(c["want"], c["need"])

    def test_xianxia_engine(self):
        from nomorals.books.fiction import engine_for, StoryState, ENGINES
        self.assertIn("xianxia", ENGINES)
        e = engine_for("xianxia", seed=11)
        st = StoryState(slug="s", title="T", premise="a boy cultivates",
                        genre="xianxia", mode="novel")
        arcs = e.plan_arcs("a boy cultivates", st)
        self.assertEqual(len(arcs), 3)
        self.assertEqual(st.ledger["cultivation"]["realm_idx"], 0)
        roles = {c["role"] for c in st.characters}
        self.assertIn("rival", roles)
        brief = e.chapter_brief(st, arcs[0], 4)
        self.assertTrue(brief.extra["breakthrough"])
        self.assertIn("Qi Condensation", brief.extra["realm"])
        # unearned realm in prose fails validation
        problems = e.validate("he ascended to Nascent Soul in a breath",
                              st, brief)
        self.assertTrue(any("unearned realm" in p for p in problems))
        # breakthrough chapter without the breakthrough fails
        problems2 = e.validate("a quiet day of tea", st, brief)
        self.assertTrue(any("breakthrough" in p for p in problems2))
        # advance moves the cultivation ladder
        e.advance(st, "he broke through the tribulation", brief)
        self.assertEqual(st.ledger["cultivation"]["stage_idx"], 1)


# ── continuation ─────────────────────────────────────────────────────────────


class TestContinuationSweep(unittest.TestCase):
    def _continuer(self):
        from nomorals.books.continuation import StoryContinuer
        ctx, tmp = _ctx()
        return StoryContinuer(ctx), tmp

    def _bible(self):
        from nomorals.books.bible import (StoryBible, BibleCharacter,
                                          PlotThread, LorebookEntry)
        return StoryBible(
            story_slug="s", title="T", pov="third", tense="past",
            tone=["dark"], voice_notes="lean and brutal",
            characters=[BibleCharacter(name="Aiko", role="protagonist",
                                       description="a blade-dancer")],
            threads=[PlotThread(id="t1", summary="the debt comes due",
                                heat=8.0)],
            lore=[LorebookEntry(name="Iron Law", kind="rule",
                                text="magic costs blood", always_on=True,
                                order=1),
                  LorebookEntry(name="Aiko", kind="character",
                                text="blade-dancer", keys=["blade"],
                                order=50)],
            arc_summary="[ch.1] Aiko fled the city.")

    def test_assemble_prompt_order(self):
        sc, _ = self._continuer()
        b = self._bible()
        system, user = sc._assemble_prompt(
            b, 2, ["Aiko drew the blade and ran."], 800, "")
        self.assertIn("ghostwriter", system)
        # canonical AI-Dungeon order
        sections = ["PLOT ESSENTIALS", "WORLD LORE", "STORY SO FAR",
                    "OPEN THREADS", "RECENT STORY", "Author's note",
                    "Write CHAPTER 2"]
        idx = [user.index(s) for s in sections]
        self.assertEqual(idx, sorted(idx))
        self.assertIn("blade-dancer", user)  # keyed lore fired on "blade"

    def test_score_take_ranking(self):
        sc, _ = self._continuer()
        b = self._bible()
        good = ('She was running through the rain. "Come back!" he shouted. '
                'Too late — the gate was already closing?')
        bad = ("Aiko drew the blade. Aiko drew the blade. Aiko drew the "
               "blade. Aiko drew the blade.")
        s_good = sc._score_take(good, b, ["Aiko drew the blade."])
        s_bad = sc._score_take(bad, b, ["Aiko drew the blade."])
        self.assertGreater(s_good, s_bad)

    def test_takes_and_summary(self):
        from nomorals.books.continuation import StoryContinuer
        sc, tmp = self._continuer()
        b = self._bible()
        # heuristic takes vary by seed salt
        t0 = sc._compose_continuation(b, 2, ["Aiko drew the blade."], 400,
                                      "", seed_salt=0)
        t1 = sc._compose_continuation(b, 2, ["Aiko drew the blade."], 400,
                                      "", seed_salt=1)
        self.assertNotEqual(t0, t1)
        self.assertGreater(len(t0.split()), 100)
        sc._update_summary(b, 2, t0)
        self.assertIn("[ch.2]", b.arc_summary)
        self.assertIn("[ch.1]", b.arc_summary)

    def test_write_one_takes(self):
        from nomorals.books.continuation import StoryContinuer
        sc, tmp = self._continuer()
        b = self._bible()
        sc.reader.chapter_text = lambda slug, n: "Aiko drew the blade."
        result = sc._write_one("s", "T", b, 2, words=300, direction="",
                               takes=3)
        self.assertEqual(result["number"], 2)
        self.assertEqual(len(result["takes"]), 2)
        self.assertEqual(len(result["take_scores"]), 3)
        self.assertTrue(
            (Path(tmp) / "books" / "reader" / "s" / "continuations" /
             "chapter-00002.md").exists())


# ── write primitives ─────────────────────────────────────────────────────────


class TestWritePrimitives(unittest.TestCase):
    def test_expand(self):
        from nomorals.books.write import expand
        ctx, _ = _ctx()
        out = expand("Mara entered the room.", ctx, target_words=120)
        self.assertIn("Mara entered the room.", out)
        self.assertGreater(len(out.split()), 20)

    def test_describe(self):
        from nomorals.books.write import describe
        ctx, _ = _ctx()
        out = describe("the old sword", ctx, mood="mournful")
        self.assertIn("the old sword", out)
        self.assertGreater(len(out.split()), 30)

    def test_rewrite(self):
        from nomorals.books.write import rewrite
        ctx, _ = _ctx()
        out = rewrite("She saw that he seemed to begin to run quickly.",
                      "tighter", ctx)
        self.assertNotIn("seemed to", out)
        self.assertNotIn("began to", out)
        self.assertTrue(out.strip())

    def test_suggest_hooks(self):
        from nomorals.books.write import suggest_hooks
        ctx, _ = _ctx()
        hooks = suggest_hooks("The night was dark and full of teeth.", ctx)
        self.assertEqual(len(hooks), 3)
        self.assertTrue(all(h.strip() for h in hooks))


# ── outline ──────────────────────────────────────────────────────────────────


class TestOutlineSweep(unittest.TestCase):
    def test_beat_sheet_outline(self):
        from nomorals.books.model import Book
        from nomorals.books.outline import beat_sheet_outline, three_act_map
        b = Book(topic="t", slug="s")
        chs = beat_sheet_outline(b, n_chapters=12, genre="mystery")
        self.assertEqual([c.number for c in chs], list(range(1, 13)))
        self.assertTrue(any("Midpoint" in c.coverage for c in chs))
        self.assertTrue(all(c.beats for c in chs))
        m = three_act_map(20)
        self.assertEqual(m, {"act_1": (1, 4), "act_2": (5, 16),
                             "act_3": (17, 20)})


# ── forge epub/html ──────────────────────────────────────────────────────────


class TestForgeSweep(unittest.TestCase):
    def _book(self):
        from nomorals.books.model import Book, Chapter, STATUS_WRITTEN
        b = Book(topic="t", slug="s", title="Test Book", author="Devon",
                 language="en", genre="essay", description="a test")
        b.chapters = [Chapter(number=1, title="First",
                              text="Hello **world**. And *more*.\n\n"
                                   "- one\n- two\n\n1. a\n2. b",
                              status=STATUS_WRITTEN)]
        return b

    def test_md_to_xhtml(self):
        from nomorals.books.forge import md_to_xhtml
        x = md_to_xhtml("# T\n\npara one\n\n- a\n- b\n\n> quote\n\n---\n\n"
                        "1. x\n2. y", "T")
        for tag in ("<h1>T</h1>", "<p>para one</p>", "<ul>", "<ol>",
                    "<blockquote>", "<hr/>"):
            self.assertIn(tag, x)

    def test_epub_is_valid_zip(self):
        from nomorals.books.forge import epub_bytes
        data = epub_bytes(self._book())
        z = zipfile.ZipFile(io.BytesIO(data))
        names = z.namelist()
        self.assertEqual(names[0], "mimetype")
        self.assertEqual(z.read("mimetype"), b"application/epub+zip")
        # mimetype must be STORED (uncompressed)
        info = z.getinfo("mimetype")
        self.assertEqual(info.compress_type, zipfile.ZIP_STORED)
        for required in ("OEBPS/content.opf", "OEBPS/toc.ncx",
                         "OEBPS/nav.xhtml", "OEBPS/title.xhtml",
                         "OEBPS/ch001.xhtml", "OEBPS/style.css",
                         "META-INF/container.xml"):
            self.assertIn(required, names)
        opf = z.read("OEBPS/content.opf").decode()
        self.assertIn("<dc:title>Test Book</dc:title>", opf)
        self.assertIn("<dc:creator>Devon</dc:creator>", opf)
        ch = z.read("OEBPS/ch001.xhtml").decode()
        self.assertIn("<strong>world</strong>", ch)

    def test_html_book(self):
        from nomorals.books.forge import html_book
        h = html_book(self._book(), theme="dark")
        self.assertIn("<!DOCTYPE html>", h)
        self.assertIn("Test Book", h)
        self.assertIn('id="ch1"', h)
        self.assertIn("#1a1a2e", h)


# ── sources ──────────────────────────────────────────────────────────────────


class TestSourcesSweep(unittest.TestCase):
    def test_normalize_url(self):
        from nomorals.books.sources import normalize_url
        self.assertEqual(
            normalize_url("https://FreeWebNovel.com/novel/foo/?utm_source=x#frag"),
            "https://freewebnovel.com/novel/foo")
        self.assertEqual(
            normalize_url("http://EXAMPLE.com/a/?b=1&utm_medium=y"),
            "http://example.com/a?b=1")
        self.assertEqual(normalize_url("https://x.com/a/"), "https://x.com/a")

    def test_fetcher_rate_limit(self):
        from nomorals.books.sources import Fetcher
        f = Fetcher(rate_limit=0.05)
        t = time.time()
        f.polite_wait("http://example.com/1")
        f.polite_wait("http://example.com/2")
        self.assertGreaterEqual(time.time() - t, 0.04)
        # different hosts don't block each other
        t2 = time.time()
        f.polite_wait("http://other.com/1")
        self.assertLess(time.time() - t2, 0.04)
        # UA rotation
        uas = {f._next_ua() for _ in range(8)}
        self.assertGreater(len(uas), 1)

    def test_adapter_for_url(self):
        from nomorals.books.sources import adapter_for_url
        a = adapter_for_url("https://www.royalroad.com/fiction/123/foo")
        self.assertEqual(a.name, "royalroad")
        g = adapter_for_url("https://unknown-aggregator.example/novel/x")
        self.assertTrue(g.name.startswith("generic:"))
        self.assertFalse(g.verified)

    def test_download_story(self):
        from nomorals.books.sources import (SourceAdapter, StoryMeta,
                                            Chapter, normalize_url)
        meta = StoryMeta(title="T", url="https://x.com/n", source="fake")

        class FakeAdapter(SourceAdapter):
            name = "fake"
            domains = ("x.com",)

            def search(self, query, *, limit=10):
                return []

            def novel(self, url):
                return meta

            def chapter_list(self, novel_url, *, limit=0):
                return [(1, "One", "https://x.com/n/1?utm_source=x"),
                        (2, "Two", "https://x.com/n/2")]

            def fetch_chapter(self, url):
                n = 1 if url.endswith("/1") else 2
                if n == 2:
                    from nomorals.books.sources import SourceError
                    raise SourceError("boom")
                return Chapter(number=0, title="", url=url,
                               paragraphs=["para one", "para two"])

        dl = FakeAdapter().download_story("https://x.com/n")
        self.assertEqual(dl["meta"]["title"], "T")
        self.assertEqual(dl["total"], 2)
        self.assertEqual(len(dl["chapters"]), 1)
        self.assertEqual(len(dl["failed"]), 1)
        self.assertEqual(dl["failed"][0]["number"], 2)
        # resume skips the fetched chapter
        seen = [dl["chapters"][0]["url"]]
        dl2 = FakeAdapter().download_story("https://x.com/n",
                                           skip_urls=seen)
        self.assertEqual(len(dl2["chapters"]), 0)


# ── library ──────────────────────────────────────────────────────────────────


class TestLibrarySweep(unittest.TestCase):
    def _lib(self):
        from nomorals.books.library import Library
        ctx, tmp = _ctx()
        lib = Library(ctx)
        p = Path(tmp) / "book.txt"
        p.write_text(("CHAPTER 1\n\nHello world. Dragons everywhere. " * 40) +
                     "\n\nCHAPTER 2\n\n" +
                     ("More dragons. The end. " * 40))
        r = lib.ingest(str(p), title="Dragons", author="Anon")
        return lib, r.slug

    def test_sessions_and_stats(self):
        lib, slug = self._lib()
        s = lib.start_session(slug)
        time.sleep(0.02)
        ended = lib.end_session(s["session_id"], chapters=2, words=60)
        self.assertGreaterEqual(ended["seconds"], 0)
        stats = lib.stats(slug)
        self.assertEqual(stats["sessions"], 1)
        self.assertEqual(stats["words_read"], 60)
        self.assertGreaterEqual(stats["streak_days"], 1)
        self.assertTrue(any(b["slug"] == slug for b in stats["per_book"]))

    def test_export_annotations(self):
        lib, slug = self._lib()
        lib.add_bookmark(slug, 1, label="good bit")
        lib.add_note(slug, 1, note="dragons!", quote="Hello world")
        md = lib.export_annotations(slug, format="markdown")
        self.assertIn("good bit", md["text"])
        self.assertIn("dragons!", md["text"])
        js = lib.export_annotations(slug, format="json")
        payload = json.loads(js["text"])
        self.assertEqual(len(payload["bookmarks"]), 1)
        self.assertEqual(len(payload["notes"]), 1)

    def test_series(self):
        lib, slug = self._lib()
        lib.set_series(slug, "Dragon Saga", 1)
        self.assertEqual(lib.get_series(slug)["series"], "Dragon Saga")
        series = lib.list_series()
        self.assertEqual(series[0]["series"], "Dragon Saga")
        self.assertEqual(series[0]["books"][0]["slug"], slug)

    def test_currently_reading(self):
        lib, slug = self._lib()
        lib.set_progress(slug, 1)
        current = lib.currently_reading()
        self.assertTrue(any(b["slug"] == slug for b in current))


# ── reader ───────────────────────────────────────────────────────────────────


class TestReaderSweep(unittest.TestCase):
    def _reader(self):
        from nomorals.books.reader import StoryReader, FollowedStory
        from nomorals.books.sources import Chapter
        ctx, tmp = _ctx()
        r = StoryReader(ctx)
        s = FollowedStory(slug="rs", title="RS",
                          url="https://example.com/novel/x",
                          source="generic", total_chapters=5)
        r._save(s)
        for n in (1, 2, 3):
            r._store_chapter("rs", Chapter(
                number=n, title=f"Ch {n}",
                url=f"https://example.com/c{n}",
                paragraphs=[f"text {n} here"]))
        return r

    def test_highlights(self):
        r = self._reader()
        h = r.add_highlight("rs", 1, "text 1 here", note="nice",
                            color="blue")
        self.assertEqual(h["highlight"]["color"], "blue")
        self.assertEqual(len(r.list_highlights("rs")), 1)
        x = r.export_highlights("rs")
        self.assertIn("text 1 here", x["text"])
        self.assertIn("nice", x["text"])
        r.remove_highlight("rs", h["highlight"]["id"])
        self.assertEqual(r.list_highlights("rs"), [])

    def test_catch_up(self):
        r = self._reader()
        c = r.catch_up("rs", limit=2)
        self.assertEqual([x["chapter"] for x in c["read"]], [1, 2])
        self.assertEqual(r._load("rs").current_chapter, 2)

    def test_reading_stats(self):
        r = self._reader()
        st = r.reading_stats("rs")
        row = st["stories"][0]
        self.assertEqual(row["cached"], 3)
        self.assertEqual(row["total_chapters"], 5)

    def test_check_updates(self):
        from nomorals.books import reader as reader_mod
        r = self._reader()

        class FakeAdapter:
            name = "fake"

            def chapter_list(self, url, *, limit=0):
                return [(n, f"Ch {n}", f"https://example.com/c{n}")
                        for n in range(1, 6)]

        with patch.object(reader_mod.StoryReader, "_adapter",
                          return_value=FakeAdapter()):
            with patch.object(reader_mod.StoryReader, "_profile",
                              return_value={"list_limit": 400}):
                rep = r.check_updates("rs")
        row = rep["updates"][0]
        self.assertTrue(row["ok"])
        self.assertEqual(row["new_chapters"], [4, 5])
        self.assertEqual(row["new_count"], 2)


# ── branches ─────────────────────────────────────────────────────────────────


class TestBranchesSweep(unittest.TestCase):
    def _branch(self):
        import nomorals.books.branches as br
        ctx, tmp = _ctx()
        with patch.object(br, "_books_root",
                          return_value=Path(tmp) / "books_data"):
            story = Path(tmp) / "books_data" / "s1"
            story.mkdir(parents=True)
            (story / "chapter_00001.md").write_text(
                "# Ch1\n\n" + "word " * 100)
            (story / "chapter_00002.md").write_text(
                "# Ch2\n\n" + "other " * 50)
            b = br.StoryBranch("s1", "what-if")
            return br, b, tmp

    def test_choices_and_decide(self):
        br, b, tmp = self._branch()
        with patch.object(br, "_books_root",
                          return_value=Path(tmp) / "books_data"):
            self.assertTrue(b.fork(1, "what if she fled")["ok"])
            c = b.add_choice(1, "Run or fight?", ["run", "fight"])
            self.assertTrue(c["ok"])
            bad = b.add_choice(1, "Solo?", ["only"])
            self.assertFalse(bad["ok"])
            d = b.decide(1, "fight")
            self.assertEqual(d["pick"], "fight")
            self.assertEqual(len(b.list_choices()), 1)

    def test_diff_and_tree(self):
        br, b, tmp = self._branch()
        with patch.object(br, "_books_root",
                          return_value=Path(tmp) / "books_data"):
            b.fork(1, "what if she fled")
            b.add_chapter("completely different text " * 30, "Flight")
            df = b.diff()
            self.assertTrue(df["ok"])
            self.assertEqual(df["first_divergence"], 1)
            self.assertIsNotNone(df["chapters"][0]["text_overlap"])
            t = br.tree("s1")
            self.assertEqual(t["children"][0]["name"], "⑂ what-if")

    def test_rename_and_merge_strategies(self):
        br, b, tmp = self._branch()
        with patch.object(br, "_books_root",
                          return_value=Path(tmp) / "books_data"):
            b.fork(1, "premise")
            b.add_chapter("text here " * 20, "C1")
            m = b.merge("replace-from")
            self.assertTrue(m["ok"])
            self.assertEqual(m["strategy"], "replace-from")
            bad = b.merge("nonsense")
            self.assertFalse(bad["ok"])
            self.assertTrue(b.rename("what-if-2")["ok"])
            self.assertEqual(b.branch_name, "what-if-2")


# ── publish ──────────────────────────────────────────────────────────────────


class TestPublishSweep(unittest.TestCase):
    def _pub(self):
        import nomorals.books.publish as pb
        ctx, tmp = _ctx()
        with patch.object(pb, "_books_root",
                          return_value=Path(tmp) / "books_data"):
            d = Path(tmp) / "books_data" / "s1"
            d.mkdir(parents=True)
            for n in range(1, 6):
                (d / f"chapter_{n:03d}.md").write_text(
                    f"# Ch{n}\n\ntext " * 20)
            p = pb.SerialPublication("s1")
            return pb, p, tmp

    def test_parse_cadence(self):
        import nomorals.books.publish as pb
        self.assertEqual(pb.parse_cadence("daily")["kind"], "daily")
        self.assertEqual(pb.parse_cadence("weekly:mon,wed,fri"),
                         {"kind": "weekday", "weekdays": [0, 2, 4]})
        with self.assertRaises(ValueError):
            pb.parse_cadence("weekly:funday")
        with self.assertRaises(ValueError):
            pb.parse_cadence("sometimes")

    def test_start_and_release(self):
        pb, p, tmp = self._pub()
        with patch.object(pb, "_books_root",
                          return_value=Path(tmp) / "books_data"):
            r = p.start("weekly:tue,thu", title="T")
            self.assertTrue(r["ok"])
            self.assertIn("next_due", r)
            pub = p._load()
            pub["next_due"] = 0  # force due
            p._save(pub)
            p.set_author_note(1, "thanks for reading!")
            rel = p.release()
            self.assertTrue(rel["ok"])
            self.assertEqual(rel["author_note"], "thanks for reading!")
            self.assertEqual(rel["chapter"], 1)

    def test_buffer_and_launch_plan(self):
        pb, p, tmp = self._pub()
        with patch.object(pb, "_books_root",
                          return_value=Path(tmp) / "books_data"):
            p.start("daily")
            buf = p.buffer_status()
            self.assertEqual(buf["buffer"], 5)
            self.assertTrue(buf["healthy"])
            plan = p.launch_plan(day_one=2)
            self.assertEqual(plan["day_one_chapters"], [1, 2])
            upcoming = p.next_releases(2)
            self.assertEqual(len(upcoming), 2)
            self.assertTrue(upcoming[0]["at"] > 0)

    def test_retention_curve(self):
        pb, p, tmp = self._pub()
        with patch.object(pb, "_books_root",
                          return_value=Path(tmp) / "books_data"):
            p.start("manual")
            for i in range(4):
                p.feedback(f"r{i}", 1, "love")
            p.feedback("rx", 2, "meh")
            pub = p._load()
            pub["released"] = [1, 2]
            p._save(pub)
            rc = p.retention_curve()
            self.assertTrue(rc["ok"])
            self.assertEqual(rc["curve"][0]["retention_pct"], 100.0)
            self.assertEqual(rc["curve"][1]["retention_pct"], 25.0)
            self.assertIn(2, rc["drop_chapters"])


# ── collab ───────────────────────────────────────────────────────────────────


class TestCollabSweep(unittest.TestCase):
    def test_style_analyzer(self):
        from nomorals.books.collab import StyleAnalyzer
        text = ("She saw the tapestry of fate. It was very, very intricate. "
                "She felt sad. Suddenly, suddenly, everything changed. "
                '"Hello," she whispered. "Hi," he murmured. '
                '"Hey," she breathed. ' * 6)
        rep = StyleAnalyzer().analyze(text)
        self.assertTrue(rep["findings"])
        self.assertTrue(any("filter-word" in f for f in rep["findings"]))
        self.assertTrue(any("AI-tell" in f for f in rep["findings"]))
        self.assertGreater(rep["score_penalty"], 0)
        clean = StyleAnalyzer().analyze(
            "The door opened. Rain hammered the street. She stepped out.")
        self.assertLessEqual(clean["score_penalty"], 0.1)

    def test_character_sheet(self):
        from nomorals.books.collab import CharacterSheet
        sheet = CharacterSheet(
            name="Mara", role="protagonist", want="to be seen as worthy",
            need="to trust another", wound="betrayed by her mentor",
            lie="strength means needing no one",
            ghost="the smell of smoke",
            speech_tics=["Listen."], rhythm="short bursts",
            taboo_topics=["her mentor"])
        block = sheet.persona_block()
        self.assertIn("Mara", block)
        self.assertIn("betrayed", block)
        good = sheet.voice_check("Listen. We move at dawn.")
        self.assertGreaterEqual(good["score"], 0.6)
        bad = sheet.voice_check("Her mentor waits inside, remember?")
        self.assertLess(bad["score"], good["score"])
        d = CharacterSheet.from_dict(sheet.to_dict())
        self.assertEqual(d.name, "Mara")

    def test_critic_personas(self):
        from nomorals.books.collab import CriticAgent
        cr = CriticAgent()
        self.assertEqual(set(cr.PERSONAS),
                         {"editor", "line", "beta", "brutal"})
        self.assertIn("developmental", cr.persona_brief("editor"))
        weak = "Short."
        c = cr.review(weak, {"summary": "t"}, persona="brutal")
        self.assertIn(c.verdict, ("revise", "rewrite"))
        self.assertLess(c.score, 0.8)
        # style findings flow into heuristic issues
        text = ("She saw the tapestry. It was very intricate. She felt sad. "
                "Suddenly suddenly changed. " * 8)
        c2 = cr.review(text, {"summary": "t"}, persona="line")
        self.assertTrue(c2.issues)

    def test_session_with_sheet(self):
        from nomorals.books.collab import (CollaborativeSession,
                                           CharacterSheet)
        sess = CollaborativeSession("s1")
        sheet = CharacterSheet(name="Mara", speech_tics=["Listen."],
                               rhythm="short bursts")
        sess.cast_character(sheet)
        self.assertIs(sess.character_sheet("Mara"), sheet)
        sc = sess.write_scene(
            "Mara", "she enters the tavern",
            suggest=lambda p, max_tokens=0: (
                '"Listen," she said. "We move at dawn." '
                "The room went quiet."))
        self.assertTrue(sc["ok"])
        self.assertGreaterEqual(sc["voice_report"]["avg_score"], 0.5)
        crit = sess.critique("A decent draft with real scenes. " * 40,
                             {"summary": "t"}, persona="beta")
        self.assertTrue(crit["ok"])
        self.assertEqual(crit["persona"], "beta")


# ── tools registration ───────────────────────────────────────────────────────


class TestToolsSweep(unittest.TestCase):
    def test_register_integrity(self):
        import re

        class FakeRegistry:
            def __init__(self):
                self.context = SimpleNamespace(
                    settings=SimpleNamespace(
                        workspace_dir=tempfile.mkdtemp()))
                self.tools = {}

            def register(self, name, **kwargs):
                def deco(fn):
                    self.tools[name] = (fn, kwargs)
                    return fn
                return deco

        from nomorals.books import tools as tools_mod
        reg = FakeRegistry()
        tools_mod.register(reg)
        names = list(reg.tools)
        self.assertEqual(len(names), len(set(names)))
        for expected in ("book_build_epub", "book_build_html", "book_card",
                         "story_lorebook", "fiction_beats", "branch_tree",
                         "branch_choice", "branch_diff", "publish_launch_plan",
                         "publish_retention", "publish_buffer",
                         "publish_author_note", "library_stats",
                         "library_export", "library_session",
                         "library_series", "reader_updates",
                         "reader_highlight", "reader_catch_up",
                         "reader_stats", "collab_stylecheck",
                         "character_sheet", "collab_critique_persona",
                         "write_expand", "write_describe", "write_rewrite",
                         "story_download"):
            self.assertIn(expected, reg.tools, expected)
        # every registration has a description; new sweep tools also
        # document their parameters
        new_tools = {"book_build_epub", "book_build_html", "book_card",
                     "story_lorebook", "fiction_beats", "branch_tree",
                     "branch_choice", "branch_diff", "publish_launch_plan",
                     "publish_retention", "publish_buffer",
                     "publish_author_note", "library_stats",
                     "library_export", "library_session",
                     "library_series", "reader_updates",
                     "reader_highlight", "reader_catch_up",
                     "reader_stats", "collab_stylecheck",
                     "character_sheet", "collab_critique_persona",
                     "write_expand", "write_describe", "write_rewrite",
                     "story_download"}
        for name, (fn, kw) in reg.tools.items():
            self.assertTrue(kw.get("description"), name)
            if name in new_tools:
                self.assertIn("parameters", kw, name)


if __name__ == "__main__":
    unittest.main()
