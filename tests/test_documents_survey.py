"""Document engine survey additions: RTF/EPUB/ODT/ODS parsers, OCR pipeline,
document comparison, extractive summarization, PDF table recovery, and
deeper metadata extraction.

RTF/EPUB/ODT/ODS are stdlib-only (no optional deps).  OCR tests pin the
fail-fast behaviour when the optional stack is missing.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from unittest import mock
from xml.etree import ElementTree as ET

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.documents import (
    Document,
    DocumentError,
    Section,
    Table,
    compare_documents,
    diff_documents,
    extract_text_tables,
    keywords,
    ocr_available,
    parse_bytes,
    parse_path,
    summarize,
    summarize_text,
)
from nomorals.documents.ocr import ocr_image, ocr_pdf
from nomorals.documents.parsers import _pdf_info, _rtf_to_text


# ── fixtures ────────────────────────────────────────────────────────────────

RTF_DOC = (
    b"{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0\\fnil Arial;}}\n"
    b"\\fs24 Hello \\b bold\\b0  world.\\par\n"
    b"Second \\u233? paragraph here.\\par\n"
    b"}"
)


def _epub_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?>'
            '<container version="1.0" '
            'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            '<rootfiles><rootfile full-path="OEBPS/content.opf" '
            'media-type="application/oebps-package+xml"/></rootfiles>'
            "</container>",
        )
        zf.writestr(
            "OEBPS/content.opf",
            '<?xml version="1.0"?>'
            '<package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
            "<metadata xmlns:dc=\"http://purl.org/dc/elements/1.1/\">"
            "<dc:title>Test Book</dc:title>"
            "<dc:creator>Jane Doe</dc:creator>"
            "</metadata>"
            "<manifest>"
            '<item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>'
            "</manifest>"
            '<spine><itemref idref="ch1"/></spine>'
            "</package>",
        )
        zf.writestr(
            "OEBPS/ch1.xhtml",
            '<?xml version="1.0"?>'
            '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>x</title></head>'
            "<body>"
            "<h1>Chapter One</h1>"
            "<p>First paragraph of the book.</p>"
            "<h2>Details</h2>"
            "<p>Some details follow.</p>"
            "<table><tr><th>Name</th><th>Age</th></tr>"
            "<tr><td>Ann</td><td>30</td></tr></table>"
            "</body></html>",
        )
    return buf.getvalue()


_ODT_NAMESPACES = (
    'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
    'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
    'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
    'xmlns:dc="http://purl.org/dc/elements/1.1/"'
)


def _odt_bytes() -> bytes:
    content = (
        '<?xml version="1.0"?>'
        f'<office:document-content {_ODT_NAMESPACES}>'
        "<office:body><office:text>"
        '<text:h text:outline-level="1">Chapter One</text:h>'
        "<text:p>Hello world from ODT.</text:p>"
        '<table:table table:name="People">'
        "<table:table-row>"
        "<table:table-cell><text:p>Name</text:p></table:table-cell>"
        "<table:table-cell><text:p>Age</text:p></table:table-cell>"
        "</table:table-row>"
        "<table:table-row>"
        "<table:table-cell><text:p>Ann</text:p></table:table-cell>"
        "<table:table-cell><text:p>30</text:p></table:table-cell>"
        "</table:table-row>"
        "</table:table>"
        "</office:text></office:body></office:document-content>"
    )
    meta = (
        '<?xml version="1.0"?>'
        '<office:document-meta xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/">'
        "<office:meta>"
        "<dc:title>ODT Title</dc:title><dc:creator>Doc Author</dc:creator>"
        "</office:meta></office:document-meta>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/vnd.oasis.opendocument.text")
        zf.writestr("content.xml", content)
        zf.writestr("meta.xml", meta)
    return buf.getvalue()


def _ods_bytes() -> bytes:
    content = (
        '<?xml version="1.0"?>'
        f'<office:document-content {_ODT_NAMESPACES}>'
        "<office:body><office:spreadsheet>"
        '<table:table table:name="Sheet1">'
        "<table:table-row>"
        "<table:table-cell><text:p>Item</text:p></table:table-cell>"
        "<table:table-cell><text:p>Qty</text:p></table:table-cell>"
        "</table:table-row>"
        "<table:table-row>"
        "<table:table-cell><text:p>Apples</text:p></table:table-cell>"
        "<table:table-cell><text:p>12</text:p></table:table-cell>"
        "</table:table-row>"
        "</table:table>"
        "</office:spreadsheet></office:body></office:document-content>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/vnd.oasis.opendocument.spreadsheet")
        zf.writestr("content.xml", content)
    return buf.getvalue()


def _pdf_bytes(*, info: bool = True, lines: tuple[str, ...] = ("Hello World",)) -> bytes:
    """Minimal valid one-page PDF; optional /Info dict, custom text lines."""
    stream_lines = []
    y = 720
    for line in lines:
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream_lines.append(f"BT /F1 12 Tf 72 {y} Td ({escaped}) Tj ET")
        y -= 20
    stream = ("\n".join(stream_lines) + "\n").encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    if info:
        objects.append(
            b"<< /Title (Test Title) /Author (Jane Doe) "
            b"/Subject (A subject) /CreationDate (D:20261003120000) >>")
    out = [b"%PDF-1.4"]
    offsets = []
    for i, body in enumerate(objects, 1):
        offsets.append(sum(len(p) + 1 for p in out))
        out.append(f"{i} 0 obj".encode() + b"\n" + body + b"\nendobj")
    trailer = (b"trailer\n<< /Size " + str(len(objects) + 1).encode()
               + b" /Root 1 0 R" + (b" /Info 6 0 R" if info else b"") + b" >>")
    out.append(trailer)
    return b"\n".join(out) + b"\n"


# ── RTF ─────────────────────────────────────────────────────────────────────


class TestRtfParser(unittest.TestCase):
    def test_rtf_basic(self) -> None:
        doc = parse_bytes(RTF_DOC, filename="note.rtf")
        self.assertEqual(doc.format, "rtf")
        text = "\n".join(s.text for s in doc.sections)
        self.assertIn("Hello bold world.", text)
        self.assertIn("Second \u00e9 paragraph here.", text)  # \u233? fallback skipped
        self.assertNotIn("fonttbl", text)
        self.assertNotIn("Arial", text)

    def test_rtf_magic_sniff_without_extension(self) -> None:
        doc = parse_bytes(RTF_DOC)
        self.assertEqual(doc.format, "rtf")

    def test_rtf_mime_hint(self) -> None:
        doc = parse_bytes(RTF_DOC, mime="application/rtf")
        self.assertEqual(doc.format, "rtf")

    def test_rtf_escapes(self) -> None:
        text = _rtf_to_text(b"{\\rtf1 a\\'e9\\u8364?x\\par\\tab end}")
        self.assertIn("\u00e9", text)     # \'e9
        self.assertIn("\u20ac", text)     # \u8364
        self.assertIn("\n", text)         # \par
        self.assertIn("\t", text)        # \tab

    def test_rtf_empty_raises(self) -> None:
        with self.assertRaises(DocumentError):
            parse_bytes(b"{\\rtf1\\ansi}", filename="empty.rtf")


# ── EPUB ────────────────────────────────────────────────────────────────────


class TestEpubParser(unittest.TestCase):
    def test_epub_structure(self) -> None:
        doc = parse_bytes(_epub_bytes(), filename="book.epub")
        self.assertEqual(doc.format, "epub")
        self.assertEqual(doc.title, "Test Book")
        self.assertEqual(doc.author, "Jane Doe")
        headings = [s.heading for s in doc.sections]
        self.assertIn("Chapter One", headings)
        self.assertIn("Details", headings)
        body = "\n".join(s.text for s in doc.sections)
        self.assertIn("First paragraph of the book.", body)
        self.assertEqual(len(doc.tables), 1)
        self.assertEqual(doc.tables[0].headers, ["Name", "Age"])
        self.assertEqual(doc.tables[0].rows, [["Ann", "30"]])

    def test_epub_zip_sniff_without_extension(self) -> None:
        doc = parse_bytes(_epub_bytes())
        self.assertEqual(doc.format, "epub")

    def test_epub_broken_container_raises(self) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("mimetype", "application/epub+zip")
        with self.assertRaises(DocumentError):
            parse_bytes(buf.getvalue(), filename="bad.epub")


# ── ODT / ODS ───────────────────────────────────────────────────────────────


class TestOdtParser(unittest.TestCase):
    def test_odt_structure(self) -> None:
        doc = parse_bytes(_odt_bytes(), filename="doc.odt")
        self.assertEqual(doc.format, "odt")
        self.assertEqual(doc.title, "ODT Title")
        self.assertEqual(doc.author, "Doc Author")
        headings = [s.heading for s in doc.sections]
        self.assertIn("Chapter One", headings)
        body = "\n".join(s.text for s in doc.sections)
        self.assertIn("Hello world from ODT.", body)
        self.assertEqual(len(doc.tables), 1)
        self.assertEqual(doc.tables[0].name, "People")
        self.assertEqual(doc.tables[0].headers, ["Name", "Age"])
        self.assertEqual(doc.tables[0].rows, [["Ann", "30"]])

    def test_odt_zip_sniff_without_extension(self) -> None:
        doc = parse_bytes(_odt_bytes())
        self.assertEqual(doc.format, "odt")


class TestOdsParser(unittest.TestCase):
    def test_ods_structure(self) -> None:
        doc = parse_bytes(_ods_bytes(), filename="sheet.ods")
        self.assertEqual(doc.format, "ods")
        self.assertEqual(len(doc.tables), 1)
        self.assertEqual(doc.tables[0].name, "Sheet1")
        self.assertEqual(doc.tables[0].headers, ["Item", "Qty"])
        self.assertEqual(doc.tables[0].rows, [["Apples", "12"]])
        self.assertEqual(doc.metadata["sheets"], ["Sheet1"])

    def test_ods_zip_sniff_without_extension(self) -> None:
        doc = parse_bytes(_ods_bytes())
        self.assertEqual(doc.format, "ods")


# ── PDF metadata + table recovery ───────────────────────────────────────────


class TestPdfInfoMetadata(unittest.TestCase):
    def test_info_dict_extracted(self) -> None:
        info = _pdf_info(_pdf_bytes())
        self.assertEqual(info.get("Title"), "Test Title")
        self.assertEqual(info.get("Author"), "Jane Doe")
        self.assertEqual(info.get("Subject"), "A subject")
        self.assertEqual(info.get("CreationDate"), "2026-10-03T12:00:00")

    def test_parse_pdf_uses_info(self) -> None:
        doc = parse_bytes(_pdf_bytes(), filename="x.pdf")
        self.assertEqual(doc.title, "Test Title")
        self.assertEqual(doc.author, "Jane Doe")
        self.assertEqual(doc.metadata.get("subject"), "A subject")
        self.assertEqual(doc.metadata.get("creationdate"), "2026-10-03T12:00:00")

    def test_no_info_is_not_fatal(self) -> None:
        info = _pdf_info(_pdf_bytes(info=False))
        self.assertEqual(info, {})
        doc = parse_bytes(_pdf_bytes(info=False), filename="x.pdf")
        self.assertEqual(doc.title, "x")  # filename stem fallback


class TestPdfTableRecovery(unittest.TestCase):
    TABLE_TEXT = (
        "Name    Age    City\n"
        "Ann     30     Lagos\n"
        "Bob     25     Abuja\n"
        "Cat     41     Kano\n"
    )

    def test_detect_columnar_block(self) -> None:
        tables = extract_text_tables(self.TABLE_TEXT)
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].headers, ["Name", "Age", "City"])
        self.assertEqual(tables[0].rows,
                         [["Ann", "30", "Lagos"], ["Bob", "25", "Abuja"],
                          ["Cat", "41", "Kano"]])

    def test_prose_is_not_a_table(self) -> None:
        prose = ("This is a normal paragraph of text that wraps naturally\n"
                 "across several lines without any columnar structure at all.\n"
                 "Nothing here should be detected as a table, ever.\n")
        self.assertEqual(extract_text_tables(prose), [])

    def test_inconsistent_width_rejected(self) -> None:
        text = ("A    B    C\n"
                "x    y\n"
                "p    q    r\n")
        self.assertEqual(extract_text_tables(text), [])

    def test_too_few_rows_rejected(self) -> None:
        self.assertEqual(extract_text_tables("A    B\nx    y\n"), [])

    def test_end_to_end_pdf_table(self) -> None:
        lines = self.TABLE_TEXT.strip().split("\n")
        doc = parse_bytes(_pdf_bytes(lines=tuple(lines)), filename="t.pdf")
        self.assertGreaterEqual(len(doc.tables), 1)
        headers = [t.headers for t in doc.tables]
        self.assertIn(["Name", "Age", "City"], headers)

    def test_bad_params_fail_fast(self) -> None:
        with self.assertRaises(ValueError):
            extract_text_tables("x", min_rows=1)
        with self.assertRaises(ValueError):
            extract_text_tables("x", min_cols=1)


# ── OCR ─────────────────────────────────────────────────────────────────────


class TestOcrFailFast(unittest.TestCase):
    def _without_deps(self):
        return mock.patch.dict(sys.modules,
                               {"pytesseract": None, "pdf2image": None,
                                "PIL": None, "PIL.Image": None})

    def test_ocr_available_never_raises(self) -> None:
        with self._without_deps():
            self.assertFalse(ocr_available())

    def test_ocr_pdf_missing_deps_hint(self) -> None:
        with self._without_deps():
            with self.assertRaises(DocumentError) as ctx:
                ocr_pdf(b"%PDF-1.4 fake")
        self.assertIn("install", str(ctx.exception).lower())

    def test_ocr_image_missing_deps_hint(self) -> None:
        with self._without_deps():
            with self.assertRaises(DocumentError) as ctx:
                ocr_image(b"\x89PNG fake")
        self.assertIn("install", str(ctx.exception).lower())

    def test_ocr_pdf_rejects_non_pdf_before_deps(self) -> None:
        with self.assertRaises(DocumentError):
            ocr_pdf(b"not a pdf")

    def test_ocr_pdf_rejects_empty(self) -> None:
        with self.assertRaises(DocumentError):
            ocr_pdf(b"")

    def test_ocr_pdf_page_window_validated(self) -> None:
        with self.assertRaises(DocumentError):
            ocr_pdf(b"%PDF-1.4 x", first_page=0)
        with self.assertRaises(DocumentError):
            ocr_pdf(b"%PDF-1.4 x", first_page=3, last_page=2)
        with self.assertRaises(DocumentError):
            ocr_pdf(b"%PDF-1.4 x", max_pages=0)


# ── compare ─────────────────────────────────────────────────────────────────


def _doc_with(title: str, sections: list[tuple[str, str]],
              tables: list[Table] | None = None) -> Document:
    doc = Document(id=title, title=title, format="txt")
    doc.sections = [Section(level=1, heading=h, text=t) for h, t in sections]
    doc.tables = tables or []
    return doc


class TestCompare(unittest.TestCase):
    def test_identical_documents(self) -> None:
        doc = _doc_with("a", [("Intro", "hello world")])
        result = compare_documents(doc, _doc_with("a", [("Intro", "hello world")]))
        self.assertEqual(result.summary, "no changes")
        self.assertFalse(result.text_changed)
        self.assertEqual(result.unified_diff, "")

    def test_section_changed(self) -> None:
        before = _doc_with("a", [("Intro", "hello world")])
        after = _doc_with("a", [("Intro", "hello brave world")])
        result = compare_documents(before, after)
        self.assertEqual(result.sections_changed, ["Intro"])
        self.assertTrue(result.text_changed)
        self.assertIn("hello brave world", result.unified_diff)

    def test_sections_added_removed(self) -> None:
        before = _doc_with("a", [("Intro", "x"), ("Old", "y")])
        after = _doc_with("a", [("Intro", "x"), ("New", "z")])
        result = compare_documents(before, after)
        self.assertEqual(result.sections_added, ["New"])
        self.assertEqual(result.sections_removed, ["Old"])
        self.assertEqual(result.sections_changed, [])

    def test_table_row_added(self) -> None:
        before = _doc_with("a", [], [Table(name="T", headers=["H"],
                                           rows=[["1"]])])
        after = _doc_with("a", [], [Table(name="T", headers=["H"],
                                          rows=[["1"], ["2"]])])
        result = compare_documents(before, after)
        self.assertEqual(result.tables_changed, ["T"])
        self.assertIn("1 row(s) added",
                      result.stats["table_details"]["T"])

    def test_table_added_removed(self) -> None:
        before = _doc_with("a", [], [Table(name="T1", headers=["H"], rows=[])])
        after = _doc_with("a", [], [Table(name="T2", headers=["H"], rows=[])])
        result = compare_documents(before, after)
        self.assertEqual(result.tables_added, ["T2"])
        self.assertEqual(result.tables_removed, ["T1"])

    def test_diff_documents_shortcut(self) -> None:
        before = _doc_with("a", [("", "line one")])
        after = _doc_with("b", [("", "line two")])
        diff = diff_documents(before, after)
        self.assertIn("-line one", diff)
        self.assertIn("+line two", diff)
        self.assertEqual(diff_documents(before, before), "")

    def test_compare_rejects_non_documents(self) -> None:
        with self.assertRaises(DocumentError):
            compare_documents("nope", _doc_with("a", []))
        with self.assertRaises(DocumentError):
            diff_documents(_doc_with("a", []), None)


# ── summarize ───────────────────────────────────────────────────────────────


class TestSummarize(unittest.TestCase):
    TEXT = (
        "The Mars rover discovered ancient riverbeds on the red planet. "
        "Scientists celebrated the Mars finding for weeks afterward. "
        "Meanwhile the stock market closed slightly higher on Tuesday. "
        "The Mars rover team published detailed maps of the riverbeds. "
        "Rain is expected tomorrow in the northern counties."
    )

    def test_summarize_text_picks_topical_sentences(self) -> None:
        out = summarize_text(self.TEXT, sentences=2)
        self.assertEqual(len(out), 2)
        # Both chosen sentences are about the rover; document order kept.
        self.assertTrue(out[0].startswith("The Mars rover discovered"))
        self.assertTrue(out[1].startswith("The Mars rover team"))

    def test_summarize_document(self) -> None:
        doc = _doc_with("a", [("News", self.TEXT)])
        out = summarize(doc, sentences=1)
        self.assertEqual(len(out), 1)
        self.assertIn("Mars", out[0])

    def test_short_text_returned_whole(self) -> None:
        self.assertEqual(summarize_text("One short sentence here.", sentences=5),
                         ["One short sentence here."])

    def test_keywords(self) -> None:
        words = keywords(self.TEXT, top_n=3)
        self.assertIn("mars", words)
        self.assertIn("rover", words)
        self.assertNotIn("the", words)

    def test_fail_fast(self) -> None:
        with self.assertRaises(DocumentError):
            summarize_text("", sentences=2)
        with self.assertRaises(DocumentError):
            summarize_text(self.TEXT, sentences=0)
        with self.assertRaises(DocumentError):
            keywords("", top_n=5)
        with self.assertRaises(DocumentError):
            summarize("not a doc")


# ── format registry sanity ──────────────────────────────────────────────────


class TestFormatRegistry(unittest.TestCase):
    def test_unknown_extension_still_raises(self) -> None:
        with self.assertRaises(DocumentError):
            parse_bytes(b"hello", filename="x.zzz")

    def test_new_mimes_recognized(self) -> None:
        self.assertEqual(parse_bytes(RTF_DOC, mime="text/rtf").format, "rtf")
        self.assertEqual(
            parse_bytes(_epub_bytes(), mime="application/epub+zip").format, "epub")
        self.assertEqual(
            parse_bytes(_odt_bytes(),
                        mime="application/vnd.oasis.opendocument.text").format, "odt")
        self.assertEqual(
            parse_bytes(_ods_bytes(),
                        mime="application/vnd.oasis.opendocument.spreadsheet").format,
            "ods")

    def test_roundtrip_conversions_still_work(self) -> None:
        from nomorals.documents import to_markdown, to_text
        doc = parse_bytes(_odt_bytes(), filename="x.odt")
        md = to_markdown(doc)
        self.assertIn("Chapter One", md)
        self.assertIn("Hello world from ODT.", to_text(doc))


# ── CLI verbs ───────────────────────────────────────────────────────────────

from nomorals.cmdline.commands.doc import _cmd_doc  # noqa: E402
from nomorals.cmdline.parser import _parser  # noqa: E402


def _args(argv):
    return _parser().parse_args(argv)


_CTX = SimpleNamespace(db=None, extras={})


def _run(argv):
    import io as _io
    from contextlib import redirect_stdout
    buf = _io.StringIO()
    with redirect_stdout(buf):
        rc = _cmd_doc(_args(argv), _CTX)
    return rc, buf.getvalue()


class TestDocCliNewVerbs(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.a = os.path.join(self.tmp.name, "a.md")
        self.b = os.path.join(self.tmp.name, "b.md")
        with open(self.a, "w", encoding="utf-8") as fh:
            fh.write("# Doc\n\nHello world.\n")
        with open(self.b, "w", encoding="utf-8") as fh:
            fh.write("# Doc\n\nHello brave world.\n")

    def test_diff_verb(self) -> None:
        rc, out = _run(["doc", "diff", self.a, self.b])
        self.assertEqual(rc, 0)
        self.assertIn("1 section(s) changed", out)

    def test_diff_verb_identical(self) -> None:
        rc, out = _run(["doc", "diff", self.a, self.a])
        self.assertEqual(rc, 0)
        self.assertIn("no changes", out)

    def test_summarize_verb(self) -> None:
        rc, out = _run(["doc", "summarize", self.a, "--sentences", "1"])
        self.assertEqual(rc, 0)
        self.assertIn("Hello world.", out)

    def test_summarize_verb_json(self) -> None:
        import json as _json
        rc, out = _run(["doc", "summarize", self.a, "--json"])
        self.assertEqual(rc, 0)
        payload = _json.loads(out)
        self.assertIn("sentences", payload)
        self.assertIn("keywords", payload)

    def test_ocr_verb_missing_deps(self) -> None:
        import io as _io
        from contextlib import redirect_stderr
        pdf = os.path.join(self.tmp.name, "scan.pdf")
        with open(pdf, "wb") as fh:
            fh.write(_pdf_bytes())
        buf = _io.StringIO()
        with mock.patch.dict(sys.modules, {"pytesseract": None, "pdf2image": None}):
            with redirect_stderr(buf):
                rc, _ = _run(["doc", "ocr", pdf])
        self.assertEqual(rc, 1)
        self.assertIn("install", buf.getvalue().lower())

    def test_search_indexes_new_formats(self) -> None:
        rtf = os.path.join(self.tmp.name, "note.rtf")
        with open(rtf, "wb") as fh:
            fh.write(RTF_DOC)
        rc, out = _run(["doc", "search", "bold", "--dir", self.tmp.name])
        self.assertEqual(rc, 0)
        self.assertIn("indexed", out)


if __name__ == "__main__":
    unittest.main()
