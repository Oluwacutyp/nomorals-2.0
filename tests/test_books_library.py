"""Books library improvements: formats (epub/pdf via documents), reading
progress, bookmarks, notes, collections, tags, ratings, search provenance,
and the `nm books` CLI."""
from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.books.library import Library, LibraryError, split_chapters
from nomorals.cmdline.commands.books import _cmd_books
from nomorals.documents.parsers import parse_bytes
from nomorals.documents.errors import DocumentError
from nomorals.search.sources import BooksAdapter


def _ctx():
    tmp = tempfile.mkdtemp()
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(db=None, settings=settings), tmp


BOOK_TEXT = """# The Dragon Codex

by Ember Wright

# Chapter One — Awakening

Dragons fly over the valley at dawn. The wizard watched them from the tower,
counting seven bright wings against the rising sun.

# Chapter Two — The Bargain

The wizard made a bargain with the eldest dragon. Fire for wisdom, scale for
spell — an old trade, older than the mountains themselves.

# Chapter Three — Departure

At dusk the dragons left the valley. The wizard kept the ember they gave,
a small sun in a glass jar, and the valley grew quiet.
"""


def _write(tmp, name, text):
    p = Path(tmp) / name
    p.write_text(text, encoding="utf-8")
    return p


def _make_epub(path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("META-INF/container.xml",
                   '<?xml version="1.0"?><container version="1.0" '
                   'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                   '<rootfiles><rootfile full-path="OEBPS/content.opf" '
                   'media-type="application/oebps-package+xml"/>'
                   "</rootfiles></container>")
        z.writestr("OEBPS/content.opf",
                   '<?xml version="1.0"?><package '
                   'xmlns="http://www.idpf.org/2007/opf" version="3.0">'
                   '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
                   "<dc:title>The Dragon Codex</dc:title>"
                   "<dc:creator>Ember Wright</dc:creator></metadata>"
                   "<manifest>"
                   '<item id="c1" href="ch1.xhtml" media-type="application/xhtml+xml"/>'
                   '<item id="c2" href="ch2.xhtml" media-type="application/xhtml+xml"/>'
                   "</manifest><spine>"
                   '<itemref idref="c1"/><itemref idref="c2"/>'
                   "</spine></package>")
        z.writestr("OEBPS/ch1.xhtml",
                   "<html><body><h1>Chapter One</h1><p>"
                   + "Dragons fly over the valley at dawn. " * 20
                   + "</p></body></html>")
        z.writestr("OEBPS/ch2.xhtml",
                   "<html><body><h1>Chapter Two</h1><p>"
                   + "The wizard made a bargain with the eldest dragon. " * 20
                   + "</p></body></html>")
    Path(path).write_bytes(buf.getvalue())
    return Path(path)


def _make_pdf(path):
    import zlib
    stream = (b"BT /F1 12 Tf 72 720 Td "
              b"(" + b"Dragons fly over the valley at dawn. " * 10 + b") Tj ET")
    comp = zlib.compress(stream)
    pdf = b"%PDF-1.4\n"
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        (b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(comp)
         + comp + b"\nendstream"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(pdf))
        pdf += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(pdf)
    pdf += b"xref\n0 6\n0000000000 65535 f \n"
    for o in offsets:
        pdf += b"%010d 00000 n \n" % o
    pdf += (b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF"
            % xref)
    Path(path).write_bytes(pdf)
    return Path(path)


def _ingest(ctx, tmp, name="codex.md", text=BOOK_TEXT, **kw):
    lib = Library(ctx)
    return lib, lib.ingest(_write(tmp, name, text), **kw)


# ── ingest formats ────────────────────────────────────────────────────

class IngestFormatTests(unittest.TestCase):
    def test_ingest_txt(self):
        ctx, tmp = _ctx()
        lib, res = _ingest(ctx, tmp, "codex.txt")
        self.assertEqual(res.chapters, 4)
        self.assertGreater(res.words, 50)
        self.assertEqual(res.strategy, "markdown")

    def test_ingest_epub_uses_documents_parser(self):
        ctx, tmp = _ctx()
        lib = Library(ctx)
        res = lib.ingest(_make_epub(Path(tmp) / "codex.epub"))
        self.assertEqual(res.title, "The Dragon Codex")
        self.assertEqual(res.author, "Ember Wright")
        self.assertGreaterEqual(res.chapters, 1)
        hits = lib.search("dragons")
        self.assertTrue(hits)

    def test_ingest_pdf(self):
        ctx, tmp = _ctx()
        lib = Library(ctx)
        res = lib.ingest(_make_pdf(Path(tmp) / "codex.pdf"), title="PDF Codex")
        self.assertEqual(res.title, "PDF Codex")
        self.assertGreater(res.words, 0)
        hits = lib.search("dragons")
        self.assertTrue(hits)

    def test_ingest_html(self):
        ctx, tmp = _ctx()
        lib = Library(ctx)
        res = lib.ingest(_write(tmp, "codex.html",
                                "<html><head><title>HTML Codex</title></head><body>"
                                "<h1>Chapter One</h1><p>" + "Dragons fly. " * 60 +
                                "</p><h1>Chapter Two</h1><p>" + "Wizards walk. " * 60 +
                                "</p></body></html>"))
        self.assertEqual(res.title, "HTML Codex")
        self.assertGreaterEqual(res.chapters, 1)

    def test_unsupported_extension(self):
        ctx, tmp = _ctx()
        lib = Library(ctx)
        with self.assertRaises(LibraryError):
            lib.ingest(_write(tmp, "codex.xyz", "x" * 500))

    def test_too_small(self):
        ctx, tmp = _ctx()
        lib = Library(ctx)
        with self.assertRaises(LibraryError):
            lib.ingest(_write(tmp, "tiny.md", "hello"))

    def test_missing_file(self):
        ctx, _ = _ctx()
        with self.assertRaises(LibraryError):
            Library(ctx).ingest("/tmp/does-not-exist-xyz.md")

    def test_explicit_title_author_win(self):
        ctx, tmp = _ctx()
        lib, res = _ingest(ctx, tmp, title="My Title", author="Me")
        self.assertEqual(res.title, "My Title")
        self.assertEqual(res.author, "Me")


# ── search provenance ───────────────────────────────────────────────

class SearchProvenanceTests(unittest.TestCase):
    def test_hits_carry_full_provenance(self):
        ctx, tmp = _ctx()
        lib, res = _ingest(ctx, tmp, author="Ember Wright")
        hits = lib.search("dragons")
        self.assertTrue(hits)
        h = hits[0]
        self.assertEqual(h.book_slug, res.slug)
        self.assertEqual(h.author, "Ember Wright")
        self.assertGreaterEqual(h.chapter_number, 1)
        self.assertGreater(h.ingested_at, 0)
        d = h.to_dict()
        self.assertEqual(d["author"], "Ember Wright")
        self.assertIn("chapter_number", d)
        self.assertIn("ingested_at", d)

    def test_memory_bm25_path_has_provenance(self):
        ctx, tmp = _ctx()
        lib, res = _ingest(ctx, tmp)
        con, _ = lib._connect()
        try:
            hits = lib._search_memory(con, "dragons", 5, 260, "")
        finally:
            con.close()
        self.assertTrue(hits)
        self.assertEqual(hits[0].book_slug, res.slug)
        self.assertGreater(hits[0].ingested_at, 0)

    def test_federated_adapter_provenance(self):
        ctx, tmp = _ctx()
        lib, res = _ingest(ctx, tmp, author="Ember Wright")
        ad = BooksAdapter(ctx, library=lib)
        self.assertIsNone(ad.probe())
        hits = ad.search("dragons", limit=3)
        self.assertTrue(hits)
        h = hits[0]
        self.assertEqual(h.provenance["book_slug"], res.slug)
        self.assertEqual(h.provenance["author"], "Ember Wright")
        self.assertGreaterEqual(h.provenance["chapter_number"], 1)
        self.assertIsNotNone(h.timestamp)


# ── reading progress ────────────────────────────────────────────────

class ProgressTests(unittest.TestCase):
    def test_set_get_resume(self):
        ctx, tmp = _ctx()
        lib, res = _ingest(ctx, tmp)
        slug = res.slug
        prog = lib.get_progress(slug)
        self.assertFalse(prog["started"])
        self.assertEqual(prog["percent"], 0.0)

        out = lib.set_progress(slug, 2, 150)
        self.assertEqual(out["chapter"], 2)
        self.assertEqual(out["offset_chars"], 150)
        self.assertAlmostEqual(out["percent"], 50.0, places=1)
        self.assertEqual(out["chapter_title"], "Chapter One — Awakening")

        r = lib.resume(slug)
        self.assertEqual(r["next"]["number"], 3)
        self.assertIn("continue", r["hint"])

    def test_resume_unstarted_and_finished(self):
        ctx, tmp = _ctx()
        lib, res = _ingest(ctx, tmp)
        slug = res.slug
        r = lib.resume(slug)
        self.assertEqual(r["next"]["number"], 1)
        self.assertIn("not started", r["hint"])
        lib.set_progress(slug, 4)
        r = lib.resume(slug)
        self.assertIn("finished", r["hint"])

    def test_read_marks_progress(self):
        ctx, tmp = _ctx()
        lib, res = _ingest(ctx, tmp)
        lib.read(res.slug, chapter=2)
        prog = lib.get_progress(res.slug)
        self.assertTrue(prog["started"])
        self.assertEqual(prog["chapter"], 2)

    def test_chapter_clamped(self):
        ctx, tmp = _ctx()
        lib, res = _ingest(ctx, tmp)
        out = lib.set_progress(res.slug, 99)
        self.assertEqual(out["chapter"], 4)
        out = lib.set_progress(res.slug, -5)
        self.assertEqual(out["chapter"], 1)

    def test_unknown_slug_fails_fast(self):
        ctx, _ = _ctx()
        lib = Library(ctx)
        with self.assertRaises(LibraryError):
            lib.set_progress("nope", 1)
        with self.assertRaises(LibraryError):
            lib.get_progress("nope")


# ── bookmarks & notes ───────────────────────────────────────────────

class BookmarkNoteTests(unittest.TestCase):
    def test_bookmarks_crud(self):
        ctx, tmp = _ctx()
        lib, res = _ingest(ctx, tmp)
        slug = res.slug
        b1 = lib.add_bookmark(slug, 1, 42, label="great line")
        b2 = lib.add_bookmark(slug, 3)
        self.assertEqual(b1["label"], "great line")
        marks = lib.list_bookmarks(slug)
        self.assertEqual(len(marks), 2)
        self.assertEqual(marks[0]["chapter"], 1)  # ordered by chapter
        self.assertEqual(lib.list_bookmarks(), [m for m in lib.list_bookmarks()])
        out = lib.remove_bookmark(b1["id"])
        self.assertEqual(out["removed"], b1["id"])
        self.assertEqual(len(lib.list_bookmarks(slug)), 1)
        with self.assertRaises(LibraryError):
            lib.remove_bookmark(999999)

    def test_bookmark_bad_chapter(self):
        ctx, tmp = _ctx()
        lib, res = _ingest(ctx, tmp)
        with self.assertRaises(LibraryError):
            lib.add_bookmark(res.slug, 99)
        with self.assertRaises(LibraryError):
            lib.add_bookmark("nope", 1)

    def test_notes_crud(self):
        ctx, tmp = _ctx()
        lib, res = _ingest(ctx, tmp)
        slug = res.slug
        n = lib.add_note(slug, 2, 10, quote="Fire for wisdom",
                         note="the core bargain of the book")
        self.assertEqual(n["quote"], "Fire for wisdom")
        notes = lib.list_notes(slug)
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["note"], "the core bargain of the book")
        lib.remove_note(n["id"])
        self.assertEqual(lib.list_notes(slug), [])
        with self.assertRaises(LibraryError):
            lib.add_note(slug, 1, note="   ")  # empty note fails
        with self.assertRaises(LibraryError):
            lib.remove_note(999999)


# ── collections, tags, ratings ──────────────────────────────────────

class CurationTests(unittest.TestCase):
    def _two_books(self):
        ctx, tmp = _ctx()
        lib = Library(ctx)
        r1 = lib.ingest(_write(tmp, "a.md", BOOK_TEXT), title="Book A")
        r2 = lib.ingest(_write(tmp, "b.md", BOOK_TEXT), title="Book B")
        return lib, r1.slug, r2.slug

    def test_collections(self):
        lib, a, b = self._two_books()
        out = lib.create_collection("fantasy")
        self.assertTrue(out["created"])
        self.assertFalse(lib.create_collection("fantasy")["created"])
        lib.add_to_collection("fantasy", a)
        lib.add_to_collection("fantasy", b)
        cols = lib.list_collections()
        self.assertEqual(cols[0]["books"], 2)
        shelf = lib.shelf("fantasy")
        self.assertEqual(len(shelf["books"]), 2)
        self.assertIn("progress_percent", shelf["books"][0])
        lib.set_progress(a, 2)
        shelf = lib.shelf("fantasy")
        by_slug = {x["slug"]: x for x in shelf["books"]}
        self.assertGreater(by_slug[a]["progress_percent"], 0)
        lib.remove_from_collection("fantasy", b)
        self.assertEqual(len(lib.shelf("fantasy")["books"]), 1)
        with self.assertRaises(LibraryError):
            lib.remove_from_collection("fantasy", b)
        lib.delete_collection("fantasy")
        self.assertEqual(lib.list_collections(), [])
        with self.assertRaises(LibraryError):
            lib.delete_collection("fantasy")

    def test_collection_errors(self):
        lib, a, _ = self._two_books()
        with self.assertRaises(LibraryError):
            lib.create_collection("   ")
        with self.assertRaises(LibraryError):
            lib.add_to_collection("missing", a)
        with self.assertRaises(LibraryError):
            lib.add_to_collection("fantasy", "nope")
        with self.assertRaises(LibraryError):
            lib.shelf("missing")

    def test_tags(self):
        lib, a, b = self._two_books()
        lib.set_tags(a, "fantasy, dragons")
        lib.set_tags(b, ["fantasy", "short"])
        self.assertEqual(lib.get_tags(a), ["dragons", "fantasy"])
        tags = lib.list_tags()
        self.assertEqual(tags[0]["tag"], "fantasy")
        self.assertEqual(tags[0]["books"], 2)
        found = lib.books_with_tag("dragons")
        self.assertEqual([x["slug"] for x in found], [a])
        lib.set_tags(a, [])  # clear
        self.assertEqual(lib.get_tags(a), [])
        with self.assertRaises(LibraryError):
            lib.books_with_tag("")

    def test_ratings(self):
        lib, a, _ = self._two_books()
        self.assertEqual(lib.get_rating(a), 0)
        lib.rate(a, 5)
        self.assertEqual(lib.get_rating(a), 5)
        lib.rate(a, 3)  # re-rate replaces
        self.assertEqual(lib.get_rating(a), 3)
        for bad in (0, 6, -1):
            with self.assertRaises(LibraryError):
                lib.rate(a, bad)

    def test_list_books_enriched(self):
        lib, a, _ = self._two_books()
        lib.rate(a, 4)
        lib.set_tags(a, "fantasy")
        lib.set_progress(a, 4)
        books = {b["slug"]: b for b in lib.list_books()}
        self.assertEqual(books[a]["rating"], 4)
        self.assertEqual(books[a]["tags"], ["fantasy"])
        self.assertEqual(books[a]["progress_percent"], 100.0)

    def test_drop_cleans_curation(self):
        lib, a, _ = self._two_books()
        lib.set_progress(a, 2)
        lib.add_bookmark(a, 1, label="x")
        lib.add_note(a, 1, note="y")
        lib.create_collection("c")
        lib.add_to_collection("c", a)
        lib.set_tags(a, "t")
        lib.rate(a, 5)
        lib.drop(a)
        self.assertEqual(lib.list_bookmarks(), [])
        self.assertEqual(lib.list_notes(), [])
        self.assertEqual(lib.list_tags(), [])
        self.assertEqual(lib.list_collections()[0]["books"], 0)


# ── epub parser (documents) ─────────────────────────────────────────

class EpubParserTests(unittest.TestCase):
    def test_epub_parses(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("mimetype", "application/epub+zip")
            z.writestr("META-INF/container.xml",
                       '<?xml version="1.0"?><container version="1.0" '
                       'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                       '<rootfiles><rootfile full-path="content.opf"/>'
                       "</rootfiles></container>")
            z.writestr("content.opf",
                       '<?xml version="1.0"?><package xmlns='
                       '"http://www.idpf.org/2007/opf">'
                       '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
                       "<dc:title>EPUB Title</dc:title>"
                       "<dc:creator>EPUB Author</dc:creator></metadata>"
                       "<manifest>"
                       '<item id="p1" href="p1.xhtml" media-type='
                       '"application/xhtml+xml"/></manifest>'
                       '<spine><itemref idref="p1"/></spine></package>')
            z.writestr("p1.xhtml",
                       "<html><body><h1>Intro</h1>"
                       "<p>Some epub content here.</p></body></html>")
        doc = parse_bytes(buf.getvalue(), filename="book.epub")
        self.assertEqual(doc.format, "epub")
        self.assertEqual(doc.title, "EPUB Title")
        self.assertEqual(doc.author, "EPUB Author")
        self.assertTrue(any("epub content" in s.text for s in doc.sections))

    def test_epub_mime_hint(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("mimetype", "application/epub+zip")
            z.writestr("x.opf",
                       '<?xml version="1.0"?><package xmlns='
                       '"http://www.idpf.org/2007/opf"><metadata/>'
                       "<manifest>"
                       '<item id="p1" href="p1.xhtml" media-type='
                       '"application/xhtml+xml"/></manifest>'
                       '<spine><itemref idref="p1"/></spine></package>')
            z.writestr("p1.xhtml",
                       "<html><body><p>Fallback opf discovery.</p></body></html>")
        doc = parse_bytes(buf.getvalue(), mime="application/epub+zip")
        self.assertEqual(doc.format, "epub")
        self.assertTrue(any("Fallback" in s.text for s in doc.sections))

    def test_bad_epub_raises(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("random.txt", "not an epub")
        with self.assertRaises(DocumentError):
            parse_bytes(buf.getvalue(), filename="book.epub")


# ── split_chapters sanity (unchanged behavior) ──────────────────────

class SplitChaptersTests(unittest.TestCase):
    def test_markdown(self):
        chs, strat = split_chapters(BOOK_TEXT)
        self.assertEqual(strat, "markdown")
        self.assertEqual(len(chs), 4)

    def test_empty(self):
        self.assertEqual(split_chapters("   "), ([], "empty"))


# ── nm books CLI ────────────────────────────────────────────────────

def _ns(**kw):
    base = dict(action="list", target="", title="", author="", chapter=0,
                offset=0, limit=8, label="", quote="", note_text="",
                name="", create=False, delete=False, add="", remove="",
                set=None, find="", stars=0, id=0, list=False, json=False)
    base.update(kw)
    return argparse.Namespace(**base)


class BooksCLITests(unittest.TestCase):
    def _setup(self):
        ctx, tmp = _ctx()
        lib = Library(ctx)
        lib.ingest(_write(tmp, "codex.md", BOOK_TEXT), title="The Dragon Codex")
        slug = lib.list_books()[0]["slug"]
        return ctx, slug

    def test_list(self):
        ctx, slug = self._setup()
        self.assertEqual(_cmd_books(_ns(action="list"), ctx), 0)

    def test_ingest_search_read(self):
        ctx, tmp = _ctx()
        p = _write(tmp, "codex.md", BOOK_TEXT)
        self.assertEqual(
            _cmd_books(_ns(action="ingest", target=str(p)), ctx), 0)
        self.assertEqual(
            _cmd_books(_ns(action="search", target="dragons"), ctx), 0)
        slug = Library(ctx).list_books()[0]["slug"]
        self.assertEqual(
            _cmd_books(_ns(action="read", target=slug, chapter=1), ctx), 0)
        self.assertEqual(_cmd_books(_ns(action="read", target=slug), ctx), 0)

    def test_progress_resume(self):
        ctx, slug = self._setup()
        self.assertEqual(
            _cmd_books(_ns(action="progress", target=slug), ctx), 0)
        self.assertEqual(
            _cmd_books(_ns(action="progress", target=slug, chapter=2,
                            offset=10), ctx), 0)
        self.assertEqual(_cmd_books(_ns(action="resume", target=slug), ctx), 0)

    def test_bookmark_flow(self):
        ctx, slug = self._setup()
        self.assertEqual(
            _cmd_books(_ns(action="bookmark", target=slug, chapter=1,
                            label="x"), ctx), 0)
        self.assertEqual(
            _cmd_books(_ns(action="bookmark", target="", list=True), ctx), 0)
        bid = Library(ctx).list_bookmarks()[0]["id"]
        self.assertEqual(
            _cmd_books(_ns(action="bookmark", id=bid), ctx), 0)

    def test_note_flow(self):
        ctx, slug = self._setup()
        self.assertEqual(
            _cmd_books(_ns(action="note", target=slug, chapter=2,
                            note_text="a thought", quote="dragons"), ctx), 0)
        self.assertEqual(
            _cmd_books(_ns(action="note", target="", list=True), ctx), 0)
        nid = Library(ctx).list_notes()[0]["id"]
        self.assertEqual(_cmd_books(_ns(action="note", id=nid), ctx), 0)

    def test_shelf_flow(self):
        ctx, slug = self._setup()
        self.assertEqual(
            _cmd_books(_ns(action="shelf", target="fantasy",
                            create=True), ctx), 0)
        self.assertEqual(
            _cmd_books(_ns(action="shelf", target="fantasy",
                            add=slug), ctx), 0)
        self.assertEqual(
            _cmd_books(_ns(action="shelf", target="fantasy"), ctx), 0)
        self.assertEqual(
            _cmd_books(_ns(action="shelf", list=True), ctx), 0)
        self.assertEqual(
            _cmd_books(_ns(action="shelf", target="fantasy",
                            remove=slug), ctx), 0)
        self.assertEqual(
            _cmd_books(_ns(action="shelf", target="fantasy",
                            delete=True), ctx), 0)

    def test_tag_flow(self):
        ctx, slug = self._setup()
        self.assertEqual(
            _cmd_books(_ns(action="tag", target=slug,
                            set="fantasy,dragons"), ctx), 0)
        self.assertEqual(_cmd_books(_ns(action="tag", target=slug), ctx), 0)
        self.assertEqual(_cmd_books(_ns(action="tag", list=True), ctx), 0)
        self.assertEqual(
            _cmd_books(_ns(action="tag", find="fantasy"), ctx), 0)
        self.assertEqual(
            _cmd_books(_ns(action="tag", target=slug, set=""), ctx), 0)

    def test_rate_flow(self):
        ctx, slug = self._setup()
        self.assertEqual(
            _cmd_books(_ns(action="rate", target=slug, stars=5), ctx), 0)
        self.assertEqual(_cmd_books(_ns(action="rate", target=slug), ctx), 0)
        # invalid stars → LibraryError → exit 1
        self.assertEqual(
            _cmd_books(_ns(action="rate", target=slug, stars=9), ctx), 1)

    def test_drop_and_errors(self):
        ctx, slug = self._setup()
        self.assertEqual(_cmd_books(_ns(action="drop", target=slug), ctx), 0)
        # unknown slug → exit 1, not a traceback
        self.assertEqual(
            _cmd_books(_ns(action="resume", target="nope"), ctx), 1)
        # unknown action → exit 2
        self.assertEqual(_cmd_books(_ns(action="frobnicate"), ctx), 2)


if __name__ == "__main__":
    unittest.main()
