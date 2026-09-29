"""Wave 70 — BookForge: book model, outline, writers, forge, tools, PDF layout.

Offline by design: the template writer (no router) is the floor under test;
a scripted router exercises the model path.  PDF structure is verified from
the raw bytes (fonts, page breaks, TOC page numbers).
"""

from __future__ import annotations

import os
import re
import tempfile
import unittest
import zlib
from pathlib import Path
from typing import Any

from nomorals.agents.context import build_context
from nomorals.books import BookForge
from nomorals.books.model import (STATUS_WRITTEN, Book, BookError, Chapter,
                                  count_words, slugify)
from nomorals.books.outline import key_terms, make_outline, template_outline
from nomorals.books.write import model_available, template_chapter, write_chapter
from nomorals.core.config import load_settings
from nomorals.core.pdf import PdfError, pdf_read, read_pdf_text, render_pdf


def _settings(tmp: str) -> Any:
    return load_settings(overrides={"home": tmp, "partner.platforms": "local",
                                    "chat.local_enabled": "true"})


# ── a scripted router: real chat() contract, canned answers ─────────────────
class _ScriptedRouter:
    """stats_snapshot says a real model is active; chat() pops scripted replies."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list[Any]] = []

    def stats_snapshot(self) -> dict:
        return {"active": "scripted-model"}

    def chat(self, messages, params=None, **kw: Any) -> Any:
        from nomorals.llm.base import LLMResponse

        self.calls.append([m for m in messages])
        if not self.replies:
            return LLMResponse(text="", model="scripted-model")
        return LLMResponse(text=self.replies.pop(0), model="scripted-model")


# ═══════════════════════════════════════════════════════════════════════════
# PDF layout
# ═══════════════════════════════════════════════════════════════════════════


def _page_texts(data: bytes) -> list[str]:
    """Per-page text of a PDF we rendered ourselves (deterministic streams).

    Slices the stream by its declared /Length — a non-greedy 'endstream'
    regex mis-fires when the compressed bytes themselves end in a newline.
    """
    from nomorals.core import pdf

    objects = pdf._parse_objects(data)
    root = int(re.search(rb"/Root\s+(\d+)\s+0\s+R", data).group(1))
    pages_ref = int(re.search(rb"/Pages\s+(\d+)\s+0\s+R", objects[root]).group(1))
    kids = re.search(rb"/Kids\s*\[([^\]]+)\]", objects[pages_ref]).group(1)
    page_ids = [int(x) for x in re.findall(rb"(\d+)\s+0\s+R", kids)]
    out: list[str] = []
    for pid in page_ids:
        cbody = objects[int(re.search(rb"/Contents\s+(\d+)\s+0\s+R", objects[pid]).group(1))]
        length = int(re.search(rb"/Length (\d+)", cbody).group(1))
        start = cbody.find(b"stream\n") + len(b"stream\n")
        stream = zlib.decompress(cbody[start:start + length]).decode("latin-1")
        out.append(" ".join(stream.split()))
    return out


def _xref_valid(data: bytes) -> bool:
    off = data.find(b"xref")
    lines = data[off:].split(b"\n")
    seen = 0
    for e in lines[2:]:
        parts = e.split()
        if len(parts) == 3 and parts[2] == b"n":
            o = int(parts[0])
            if not re.match(rb"\d+ 0 obj", data[o:o + 16]):
                return False
            seen += 1
        elif len(parts) == 3 and parts[2] == b"f":
            continue
    return seen > 1


class PdfLayoutTests(unittest.TestCase):
    def test_legacy_path_unchanged(self) -> None:
        data = render_pdf("Hello world. " * 150, title="Legacy")
        self.assertNotIn(b"/F3", data)
        self.assertIn(b"Helvetica-Oblique", data)
        self.assertTrue(_xref_valid(data))
        self.assertIn("Hello world", read_pdf_text(data))

    def test_unknown_page_size(self) -> None:
        with self.assertRaises(PdfError):
            render_pdf("text", page_size="B5")

    def test_empty_refused(self) -> None:
        with self.assertRaises(PdfError):
            render_pdf("   ")
        with self.assertRaises(PdfError):
            render_pdf("", headings=True)

    def test_structured_without_headings_allocates_no_bold(self) -> None:
        data = render_pdf("just plain words on a page", title="P", headings=True)
        self.assertNotIn(b"/F3", data)

    def test_headings_render_bold_larger(self) -> None:
        data = render_pdf("# Big Title\n\nbody text here\n\n## Small Head\n\nmore body",
                          title="Doc", headings=True)
        self.assertIn(b"Helvetica-Bold", data)
        pages = _page_texts(data)
        self.assertRegex(pages[0], r"BT /F3 16 Tf 1 0 0 1 56 \d+ Tm \(Big Title\) Tj ET")
        self.assertRegex(pages[0], r"BT /F3 13 Tf 1 0 0 1 56 \d+ Tm \(Small Head\) Tj ET")

    def test_chapter_break_starts_fresh_page(self) -> None:
        filler = ("Body sentence with some substance. " * 30)
        doc = f"# Chapter One\n\n{filler}\n\n# Chapter Two\n\n{filler}"
        data = render_pdf(doc, title="T", headings=True, chapter_break=True)
        pages = _page_texts(data)
        # chapter one opens page 1 (after nothing), chapter two opens a NEW page
        ch1_pages = [i for i, p in enumerate(pages) if "Chapter One" in p]
        ch2_pages = [i for i, p in enumerate(pages) if "Chapter Two" in p]
        self.assertEqual(len(ch1_pages), 1)
        self.assertEqual(len(ch2_pages), 1)
        self.assertGreater(ch2_pages[0], ch1_pages[0])
        # the heading is the first text drawn on its page
        for i in ch2_pages:
            first_tj = re.search(r"\(([^)]*)\) Tj", pages[i])
            self.assertIn("Chapter Two", first_tj.group(1))

    def test_toc_page_numbers_are_true(self) -> None:
        body: list[str] = []
        for i in range(1, 6):
            body.append(f"# Chapter {i} — Topic {i}")
            body.append("")
            body.append(f"Chapter {i} content. " + ("Real words on the subject. " * 35))
        data = render_pdf("\n".join(body), title="Book", headings=True,
                          chapter_break=True, toc=True)
        pages = _page_texts(data)
        full = " ".join(" ".join(p.split()) for p in pages)
        # actual chapter pages (1-based)
        # a chapter "sits" on the page where its heading is DRAWN (F3 size 16),
        # not where its title also appears in the TOC
        actual: dict[int, int] = {}
        for n, p in enumerate(pages, 1):
            for i in range(1, 6):
                if re.search(rf"BT /F3 16 Tf .*\(Chapter {i} \x97 Topic {i}\) Tj", p) \
                        and i not in actual:
                    actual[i] = n
        # claimed by the TOC
        claimed: dict[int, int] = {}
        for m in re.finditer(r"(\d)\. Chapter (\d) \x97 Topic \d\s*\\\(p\. (\d+)\\\)", full):
            claimed[int(m.group(2))] = int(m.group(3))
        self.assertEqual(len(claimed), 5)
        self.assertEqual(claimed, actual)
        # the TOC itself is on page 1
        self.assertIn("Contents", pages[0])

    def test_front_matter_excluded_from_toc(self) -> None:
        doc = ("# The Great Book\n\nby the author\n\n"
               "# First Chapter\n\n" + ("Words. " * 60) + "\n\n"
               "# Second Chapter\n\n" + ("Words. " * 60))
        data = render_pdf(doc, title="The Great Book", headings=True,
                          chapter_break=True, toc=True)
        full = " ".join(" ".join(p.split()) for p in _page_texts(data))
        toc = full[full.find("Contents"):]
        self.assertNotIn("The Great Book (p.", toc)
        self.assertIn("First Chapter", toc)
        self.assertIn("Second Chapter", toc)

    def test_write_and_read_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "book.pdf"
            data = render_pdf("# One\n\ntext of the chapter here\n", title="RT",
                              headings=True, toc=True)
            path.write_bytes(data)
            text = pdf_read(str(path))
            self.assertIn("One", text)


# ═══════════════════════════════════════════════════════════════════════════
# Book model
# ═══════════════════════════════════════════════════════════════════════════


class BookModelTests(unittest.TestCase):
    def test_slugify(self) -> None:
        self.assertEqual(slugify("Hello, World!"), "hello-world")
        self.assertEqual(slugify("  "), "book")
        self.assertEqual(slugify("a" * 200), "a" * 60)

    def test_chapter_lifecycle(self) -> None:
        c = Chapter(number=1, title="T")
        self.assertEqual(c.status, "planned")
        c.text = "one two three four"
        c.mark_written()
        self.assertEqual(c.status, STATUS_WRITTEN)
        self.assertEqual(c.words, 4)

    def test_book_roundtrip_and_manuscript(self) -> None:
        book = Book(topic="t", slug="t", title="T")
        book.chapters = [
            Chapter(number=1, title="One", text="written text"),
            Chapter(number=2, title="Two"),  # unwritten
        ]
        book.chapters[0].mark_written()
        clone = Book.from_dict(book.to_dict())
        self.assertEqual(clone.chapters[0].status, STATUS_WRITTEN)
        self.assertEqual(clone.chapters[1].status, "planned")
        self.assertFalse(clone.complete)
        ms = clone.manuscript()
        self.assertIn("# One", ms)
        self.assertIn("written text", ms)
        self.assertNotIn("# Two", ms)  # unwritten chapters are skipped
        self.assertEqual(clone.total_words, 2)

    def test_next_unwritten(self) -> None:
        book = Book(topic="t", slug="t")
        book.chapters = [Chapter(number=1, title="A", text="x"),
                         Chapter(number=2, title="B")]
        book.chapters[0].mark_written()
        self.assertEqual(book.next_unwritten().number, 2)
        book.chapters[1].text = "y"
        book.chapters[1].mark_written()
        self.assertIsNone(book.next_unwritten())
        self.assertTrue(book.complete)


# ═══════════════════════════════════════════════════════════════════════════
# Outline
# ═══════════════════════════════════════════════════════════════════════════


class OutlineTests(unittest.TestCase):
    def test_key_terms_drops_stops(self) -> None:
        terms = key_terms("the eBPF for system security and the kernel")
        self.assertIn("ebpf", terms)
        self.assertIn("kernel", terms)
        self.assertNotIn("the", terms)
        self.assertNotIn("for", terms)

    def test_template_outline_shape(self) -> None:
        book = Book(topic="quantum error correction", slug="qec")
        for n in (3, 4, 8, 12, 16):
            chapters = template_outline(book, n_chapters=n)
            self.assertEqual(len(chapters), n, f"n={n}")
            self.assertTrue(all(c.title for c in chapters))
            self.assertTrue(all(len(c.beats) >= 3 for c in chapters))
            joined = " ".join(c.title for c in chapters)
            self.assertIn("Foundations", joined)
            if n >= 4:
                self.assertIn("Putting It All Together", joined)
            self.assertIn("Mastery", joined)
            # numbering is contiguous
            self.assertEqual([c.number for c in chapters], list(range(1, n + 1)))

    def test_template_outline_spins_topic_terms(self) -> None:
        book = Book(topic="crdt replication", slug="crdt")
        chapters = template_outline(book, n_chapters=8)
        deep = [c.title for c in chapters if c.title.startswith("Deep Dive")]
        self.assertTrue(any("Crdt" in t for t in deep), deep)

    def test_make_outline_falls_back_without_model(self) -> None:
        book = Book(topic="test topic xyz", slug="tt")
        chapters = make_outline(book, n_chapters=6, context=None)
        self.assertEqual(len(chapters), 6)
        self.assertEqual(book.status, "planned")

    def test_make_outline_uses_model_json(self) -> None:
        book = Book(topic="test topic xyz", slug="tt")
        outline_json = (
            '[{"title": "Alpha", "beats": ["b1", "b2", "b3"]},'
            ' {"title": "Beta", "beats": ["b1", "b2"]},'
            ' {"title": "Gamma", "beats": ["b1"]},'
            ' {"title": "Delta", "beats": ["b1"]}]'
        )
        router = _ScriptedRouter([outline_json])
        ctx = _CtxStub(router)
        chapters = make_outline(book, n_chapters=4, context=ctx)
        self.assertEqual([c.title for c in chapters], ["Alpha", "Beta", "Gamma", "Delta"])
        self.assertEqual(len(chapters[0].beats), 3)

    def test_make_outline_rejects_garbage_model(self) -> None:
        book = Book(topic="test topic xyz", slug="tt")
        ctx = _CtxStub(_ScriptedRouter(["no json here, just words"]))
        chapters = make_outline(book, n_chapters=5, context=ctx)
        self.assertEqual(len(chapters), 5)  # template floor
        self.assertTrue(all(c.title for c in chapters))


class _CtxStub:
    def __init__(self, router: Any) -> None:
        self.router = router


# ═══════════════════════════════════════════════════════════════════════════
# Writers
# ═══════════════════════════════════════════════════════════════════════════


class WriterTests(unittest.TestCase):
    def _book(self, notes: str = "") -> Book:
        book = Book(topic="crdt replication", slug="crdt", title="CRDTs")
        book.chapters = template_outline(book, n_chapters=5)
        book.notes = notes
        return book

    def test_template_chapter_covers_beats(self) -> None:
        book = self._book()
        ch = book.chapters[2]
        text = template_chapter(book, ch)
        for beat in ch.beats:
            self.assertIn(f"## {beat.strip()}", text)
        self.assertIn("## Key takeaways", text)
        self.assertGreaterEqual(count_words(text), 150)
        low = text.lower()
        for forbidden in ("lorem ipsum", "placeholder", "todo", "tbd"):
            self.assertNotIn(forbidden, low)

    def test_template_chapter_weaves_research(self) -> None:
        notes = ("The Diamond datatype provides grow-only set semantics. "
                 "Merkle trees enable delta state transfer between replicas. "
                 "COUNTER CRDTS support both increment and decrement operations. "
                 "The anti-entropy protocol converges without a coordinator.")
        book = self._book(notes)
        ch = book.chapters[1]  # Foundations
        text = template_chapter(book, ch)
        self.assertIn("Diamond datatype", text)  # a notes sentence matched a beat

    def test_write_chapter_uses_model_when_available(self) -> None:
        book = self._book()
        ch = book.chapters[0]
        model_text = ("MODEL MARKER — a real chapter paragraph about the topic. "
                      "It develops the idea with concrete detail and examples. " * 40)
        ctx = _CtxStub(_ScriptedRouter([model_text]))
        write_chapter(book, ch, context=ctx)
        self.assertEqual(ch.status, STATUS_WRITTEN)
        self.assertIn("MODEL MARKER", ch.text)
        self.assertGreater(ch.words, 50)

    def test_write_chapter_falls_back_on_model_failure(self) -> None:
        book = self._book()
        ch = book.chapters[0]

        class _BoomRouter:
            def stats_snapshot(self) -> dict:
                return {"active": "boom"}

            def chat(self, messages, params=None, **kw: Any) -> Any:
                raise RuntimeError("model down")

        ctx = _CtxStub(_BoomRouter())
        text = write_chapter(book, ch, context=ctx)
        self.assertEqual(ch.status, STATUS_WRITTEN)
        self.assertIn("## ", text)  # template sections

    def test_write_chapter_offline_floor(self) -> None:
        book = self._book()
        ch = book.chapters[0]
        text = write_chapter(book, ch, context=None)
        self.assertEqual(ch.status, STATUS_WRITTEN)
        self.assertGreaterEqual(count_words(text), 100)

    def test_model_available_probe(self) -> None:
        self.assertTrue(model_available(_CtxStub(_ScriptedRouter([]))))
        self.assertFalse(model_available(None))
        self.assertFalse(model_available(_CtxStub(None)))


# ═══════════════════════════════════════════════════════════════════════════
# Forge: the pipeline on disk
# ═══════════════════════════════════════════════════════════════════════════


class ForgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-forge-")
        self.context = build_context(_settings(self.tmp.name), with_executor=False,
                                     with_tools=False, with_router=False,
                                     with_memory=False)
        self.forge = BookForge(self.context)

    def tearDown(self) -> None:
        self.context.close()
        self.tmp.cleanup()

    def test_create_researches_outlines_saves(self) -> None:
        book = self.forge.create("crdt replication", chapters=5, research=False,
                                 author="Tester")
        self.assertEqual(book.slug, "crdt-replication")
        self.assertEqual(len(book.chapters), 5)
        self.assertTrue(self.forge._json_path(book.slug).exists())
        self.assertTrue((self.forge.book_dir(book.slug) / "book.json").exists())

    def test_create_rejects_duplicate_slug(self) -> None:
        self.forge.create("crdt replication", chapters=4, research=False)
        with self.assertRaises(BookError):
            self.forge.create("crdt replication", chapters=4, research=False)

    def test_create_requires_topic(self) -> None:
        with self.assertRaises(BookError):
            self.forge.create("   ", research=False)

    def test_notes_are_persisted(self) -> None:
        book = self.forge.create("crdt replication", chapters=4, research=False,
                                 notes="MY NOTE LINE — the diamond datatype.")
        reloaded = self.forge.load(book.slug)
        self.assertIn("MY NOTE LINE", reloaded.notes)
        self.assertTrue((self.forge.book_dir(book.slug) / "notes.md").exists())

    def test_write_is_resumable_from_disk(self) -> None:
        self.forge.create("crdt replication", chapters=4, research=False)
        for _ in range(2):
            self.forge.write_next("crdt-replication")
        # a FRESH forge (new process simulation) picks up where we stopped
        forge2 = BookForge(self.context)
        r = forge2.write_next("crdt-replication")
        self.assertEqual(r["chapter"], 3)  # not 1
        self.assertFalse(r["done"])
        r = forge2.write_next("crdt-replication")
        self.assertEqual(r["chapter"], 4)
        self.assertTrue(r["done"])

    def test_write_all_respects_limit(self) -> None:
        self.forge.create("crdt replication", chapters=6, research=False)
        r = self.forge.write_all("crdt-replication", limit=2)
        self.assertEqual(r["chapters_written"], 2)
        self.assertFalse(r["complete"])
        r = self.forge.write_all("crdt-replication")
        self.assertTrue(r["complete"])
        self.assertEqual(r["chapters_written"], 6)

    def test_chapter_files_written(self) -> None:
        self.forge.create("crdt replication", chapters=3, research=False)
        self.forge.write_all("crdt-replication")
        cdir = self.forge.book_dir("crdt-replication") / "chapters"
        self.assertEqual(sorted(p.name for p in cdir.iterdir()),
                         ["ch01.md", "ch02.md", "ch03.md"])

    def test_build_makes_real_pdf(self) -> None:
        self.forge.create("crdt replication", chapters=4, research=False)
        self.forge.write_all("crdt-replication")
        built = self.forge.build("crdt-replication")
        pdf_path = Path(built["pdf"])
        self.assertTrue(pdf_path.exists())
        self.assertGreater(built["pdf_bytes"], 1500)
        self.assertGreaterEqual(built["pages"], 2)
        data = pdf_path.read_bytes()
        self.assertTrue(data.startswith(b"%PDF-"))
        self.assertTrue(_xref_valid(data))
        text = read_pdf_text(data)
        self.assertIn("Contents", text)
        self.assertIn("crdt replication", text)  # display title = topic (verbatim)
        # chapter one's title is in the TOC
        book = self.forge.load("crdt-replication")
        self.assertIn(book.chapters[0].title[:20], text)
        # manuscript.md exists and matches
        ms = (self.forge.book_dir("crdt-replication") / "manuscript.md").read_text()
        self.assertIn("# crdt replication", ms)
        # status advanced
        self.assertEqual(self.forge.load("crdt-replication").status, "built")

    def test_build_without_chapters_refused(self) -> None:
        self.forge.create("crdt replication", chapters=3, research=False)
        with self.assertRaises(BookError):
            self.forge.build("crdt-replication")

    def test_send_without_gateway_fails_loudly(self) -> None:
        self.forge.create("crdt replication", chapters=3, research=False)
        self.forge.write_all("crdt-replication")
        with self.assertRaises(Exception) as cm:
            self.forge.send("crdt-replication", "telegram", "123")
        self.assertIn("gateway", str(cm.exception).lower())

    def test_list_and_load(self) -> None:
        self.forge.create("crdt replication", chapters=3, research=False)
        self.forge.create("kernel hardening", chapters=3, research=False)
        slugs = {b["slug"] for b in self.forge.list_books()}
        self.assertEqual(slugs, {"crdt-replication", "kernel-hardening"})
        with self.assertRaises(BookError):
            self.forge.load("nope")

    def test_run_pipeline_end_to_end(self) -> None:
        progress: list[dict] = []
        result = self.forge.run(
            "kernel hardening", title="Kernel Hardening", chapters=3,
            research=False, on_chapter=progress.append,
        )
        self.assertEqual(result["chapters_written"], 3)
        self.assertTrue(Path(result["pdf"]).exists())
        self.assertEqual(len(progress), 3)
        self.assertNotIn("sent", result, "no send was requested")


# ═══════════════════════════════════════════════════════════════════════════
# Registry tools
# ═══════════════════════════════════════════════════════════════════════════


class BookToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-booktool-")
        self.context = build_context(_settings(self.tmp.name), with_executor=False,
                                     with_tools=True, with_router=False,
                                     with_memory=False)

    def tearDown(self) -> None:
        self.context.close()
        self.tmp.cleanup()

    def test_all_book_tools_registered(self) -> None:
        names = self.context.tools.names()
        for expected in ("book_create", "book_write", "book_build", "book_send",
                         "book_run", "book_status", "book_list"):
            self.assertIn(expected, names)

    def test_tool_pipeline(self) -> None:
        tools = self.context.tools
        out = tools.call("book_create", topic="kernel hardening",
                         chapters=3, research=False)
        self.assertTrue(out.ok, out.error)
        slug = out.unwrap()["slug"]
        self.assertEqual(len(out.unwrap()["chapters"]), 3)

        out = tools.call("book_write", slug=slug)
        self.assertTrue(out.ok, out.error)
        self.assertEqual(out.unwrap()["chapter"], 1)

        tools.call("book_write", slug=slug, all=True)
        status = tools.call("book_status", slug=slug).unwrap()
        self.assertEqual(status["chapters_written"], 3)
        self.assertEqual(status["total_chapters"], 3)

        built = tools.call("book_build", slug=slug).unwrap()
        self.assertTrue(Path(built["pdf"]).exists())
        self.assertGreaterEqual(built["pages"], 1)

        listing = tools.call("book_list").unwrap()
        self.assertIn(slug, {b["slug"] for b in listing["books"]})

    def test_book_run_research_false_offline(self) -> None:
        out = self.context.tools.call("book_run", topic="crdt replication",
                                      chapters=3, research=False)
        self.assertTrue(out.ok, out.error)
        result = out.unwrap()
        self.assertTrue(Path(result["pdf"]).exists())


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


class BookCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-bookcli-")
        self._old_home = os.environ.get("NM_HOME")
        os.environ["NM_HOME"] = self.tmp.name

    def tearDown(self) -> None:
        if self._old_home is None:
            os.environ.pop("NM_HOME", None)
        else:
            os.environ["NM_HOME"] = self._old_home
        self.tmp.cleanup()

    def _books_on_disk(self) -> list[Path]:
        return sorted(Path(self.tmp.name).glob("**/book.json"))

    def test_run_pipeline_offline(self) -> None:
        from nomorals.cli import main

        rc = main(["book", "run", "kernel hardening", "--chapters", "3",
                   "--no-research", "--words", "400"])
        self.assertEqual(rc, 0)
        books = self._books_on_disk()
        self.assertEqual(len(books), 1)
        slug = books[0].parent.name
        pdfs = list(books[0].parent.glob("*.pdf"))
        self.assertEqual(len(pdfs), 1)
        self.assertGreater(pdfs[0].stat().st_size, 1500)

    def test_create_list_status_build(self) -> None:
        from nomorals.cli import main

        self.assertEqual(main(["book", "create", "crdt replication",
                               "--chapters", "3", "--no-research"]), 0)
        self.assertEqual(main(["book", "write", "crdt-replication"]), 0)
        self.assertEqual(main(["book", "write", "crdt-replication"]), 0)
        self.assertEqual(main(["book", "write", "crdt-replication"]), 0)
        self.assertEqual(main(["book", "build", "crdt-replication"]), 0)
        self.assertEqual(main(["book", "status", "crdt-replication"]), 0)
        self.assertEqual(main(["book", "list"]), 0)

    def test_usage_errors(self) -> None:
        from nomorals.cli import main

        self.assertEqual(main(["book", "run"]), 2)
        self.assertEqual(main(["book", "status"]), 2)
        self.assertEqual(main(["book", "build"]), 2)
        self.assertEqual(main(["book", "send", "only-slug"]), 2)


if __name__ == "__main__":
    unittest.main()
