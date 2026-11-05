"""Wave 2 (Echelon documents port): error-code fold, xlsx/pptx
writers, embedded-font PDF.

Pins:
  * ``DocumentError`` is a NoMoralsError (code system, registry,
    ``to_dict``) AND still a ValueError, so every pre-existing catch
    keeps working; its machine-readable ``kind`` is separate from
    the human message and mirrored into ``details``.
  * raise sites classify: parse / unsupported_format /
    missing_dependency / empty / invalid_argument / io / write /
    ocr_unavailable.
  * the stdlib xlsx/pptx writers round-trip through the engine's
    own readers; xlsx cells are typed (number/date/boolean) while
    lossy-if-typed values stay strings.
  * ``render_pdf_embedded`` subsets+embeds a TrueType font so
    non-WinAnsi text extracts exactly through Devon's own reader,
    where the legacy WinAnsi writer loses it.
"""

from __future__ import annotations

import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

from nomorals.core.errors import NoMoralsError, error_codes
from nomorals.core.pdf import PdfError, read_pdf_text, render_pdf
from nomorals.documents import (DOCUMENT_ERROR_KINDS, Document,
                                DocumentError, Section, Table,
                                parse_bytes, parse_path, to_pptx, to_xlsx)
from nomorals.documents import expose

_TMP_ROOT = os.environ.get("NM_TEST_TMPDIR")


def _tmp() -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory(dir=_TMP_ROOT)


def _fixture() -> Document:
    return Document(
        format="test", title="Wave 2 Fixture",
        sections=[Section(level=1, heading="Intro",
                          text="Hello Αθήνα, café €50.")],
        tables=[Table(name="Data", headers=["N", "When", "On", "Code"],
                      rows=[["42", "2026-11-05", "TRUE", "007"],
                            ["4.50", "2026-11-06", "FALSE", "1,234"]])],
    )


# ── error system ────────────────────────────────────────────────────────────

class TestDocumentErrorFold(unittest.TestCase):
    def test_bases_preserve_existing_catches(self):
        err = DocumentError("boom")
        self.assertIsInstance(err, NoMoralsError)
        self.assertIsInstance(err, ValueError)
        self.assertIsInstance(err, Exception)
        with self.assertRaises(ValueError):
            raise DocumentError("caught as ValueError")
        with self.assertRaises(DocumentError):
            raise DocumentError("caught as itself")

    def test_code_system_membership(self):
        err = DocumentError("boom")
        self.assertEqual(err.code, "document.error")
        self.assertIn("document.error", error_codes())
        payload = err.to_dict()
        self.assertEqual(payload["code"], "document.error")
        self.assertEqual(payload["message"], "boom")
        self.assertEqual(payload["details"]["kind"], "error")

    def test_kind_is_separate_and_mirrored(self):
        err = DocumentError("the file is empty", kind="empty")
        self.assertEqual(err.kind, "empty")
        self.assertEqual(err.details["kind"], "empty")
        self.assertIn("empty", DOCUMENT_ERROR_KINDS)
        # unclassified default, not a fabricated kind
        self.assertEqual(DocumentError("x").kind, "error")

    def test_unknown_kind_rejected(self):
        with self.assertRaises(ValueError):
            DocumentError("x", kind="not-a-kind")

    def test_parse_failures_carry_parse_kind(self):
        with self.assertRaises(DocumentError) as caught:
            parse_bytes(b"PK\x03\x04 garbage", filename="bad.docx")
        self.assertEqual(caught.exception.kind, "parse")

    def test_unsupported_format_kind(self):
        with self.assertRaises(DocumentError) as caught:
            parse_bytes(b"hello", filename="notes.xyzq")
        self.assertEqual(caught.exception.kind, "unsupported_format")

    def test_empty_input_kind(self):
        with self.assertRaises(DocumentError) as caught:
            parse_bytes(b"")
        self.assertEqual(caught.exception.kind, "empty")

    def test_missing_file_kind_io(self):
        with self.assertRaises(DocumentError) as caught:
            parse_path("/definitely/not/here/missing.md")
        self.assertEqual(caught.exception.kind, "io")

    def test_invalid_argument_kind(self):
        from nomorals.documents import to_html
        with self.assertRaises(DocumentError) as caught:
            to_html(_fixture(), theme="neon")
        self.assertEqual(caught.exception.kind, "invalid_argument")

    def test_ocr_unavailable_code_and_kind(self):
        from nomorals.documents.ocr import ocr_image
        with _tmp() as td:
            img = Path(td) / "x.png"
            img.write_bytes(b"\x89PNG\r\n\x1a\n")
            # only asserts the coded failure when the stack is absent;
            # where Tesseract exists the call is environment-gated out
            from nomorals.documents.ocr import ocr_available
            if ocr_available():
                self.skipTest("OCR stack present on this machine")
            with self.assertRaises(DocumentError) as caught:
                ocr_image(img)
        self.assertEqual(caught.exception.kind, "ocr_unavailable")
        self.assertEqual(caught.exception.code,
                         "document.ocr_unavailable")

    def test_convert_validates_embedded_font_io(self):
        from nomorals.documents import to_pdf_embedded
        with self.assertRaises(DocumentError) as caught:
            to_pdf_embedded(_fixture(), "/no/such/font.ttf")
        self.assertEqual(caught.exception.kind, "io")


# ── xlsx writer ─────────────────────────────────────────────────────────────

class TestXlsxWriter(unittest.TestCase):
    def test_round_trip_values(self):
        back = parse_bytes(to_xlsx(_fixture()), filename="f.xlsx")
        table = next(t for t in back.tables if t.name == "Data")
        self.assertEqual(table.headers, ["N", "When", "On", "Code"])
        self.assertEqual(table.rows[0][0], "42")
        self.assertTrue(table.rows[0][1].startswith("2026-11-05"))
        self.assertEqual(table.rows[0][3], "007")
        self.assertEqual(table.rows[1][3], "1,234")
        prose = next(t for t in back.tables if t.name == "Document")
        flat = "\n".join(prose.headers + [c for r in prose.rows for c in r])
        self.assertIn("Αθήνα", flat)

    def test_typed_cells_via_independent_reader(self):
        try:
            import openpyxl
        except ImportError:
            self.skipTest("openpyxl not installed")
        import io
        workbook = openpyxl.load_workbook(
            io.BytesIO(to_xlsx(_fixture())), data_only=True)
        try:
            sheet = workbook["Data"]
            cells = {c.coordinate: c for row in sheet.iter_rows()
                     for c in row}
            self.assertEqual(cells["A2"].data_type, "n")
            self.assertEqual(cells["A2"].value, 42)
            self.assertEqual(cells["B2"].data_type, "d")
            self.assertEqual(cells["B2"].value.year, 2026)
            self.assertEqual(cells["C2"].data_type, "b")
            self.assertIs(cells["C2"].value, True)
            self.assertEqual(cells["D2"].data_type, "s")
            self.assertEqual(cells["D2"].value, "007")
        finally:
            workbook.close()

    def test_sheet_names_sanitized(self):
        import io
        import re
        doc = Document(format="t", tables=[
            Table(name="Bad/Name: [x]" + "y" * 40, headers=["a"], rows=[]),
            Table(name="Bad/Name: [x]" + "y" * 40, headers=["b"], rows=[]),
        ])
        with zipfile.ZipFile(io.BytesIO(to_xlsx(doc))) as zf:
            workbook = zf.read("xl/workbook.xml").decode()
        names = re.findall(r'<sheet name="([^"]+)"', workbook)
        self.assertEqual(len(names), 2)
        for name in names:
            self.assertLessEqual(len(name), 31)
            for illegal in "[]:*?/\\":
                self.assertNotIn(illegal, name)
        self.assertNotEqual(names[0], names[1])  # deduplicated

    def test_empty_document_refused_with_kind(self):
        with self.assertRaises(DocumentError) as caught:
            to_xlsx(Document(format="t"))
        self.assertEqual(caught.exception.kind, "empty")


# ── pptx writer ─────────────────────────────────────────────────────────────

class TestPptxWriter(unittest.TestCase):
    def test_round_trip_text(self):
        back = parse_bytes(to_pptx(_fixture()), filename="f.pptx")
        from nomorals.documents.model import full_text
        text = full_text(back)
        for needle in ("Wave 2 Fixture", "Intro", "Αθήνα", "café",
                       "42", "007", "1,234", "Data"):
            self.assertIn(needle, text)

    def test_all_xml_parts_well_formed(self):
        import io
        with zipfile.ZipFile(io.BytesIO(to_pptx(_fixture()))) as zf:
            names = zf.namelist()
            self.assertIn("ppt/slides/slide1.xml", names)
            self.assertIn("ppt/presentation.xml", names)
            for name in names:
                if name.endswith((".xml", ".rels")):
                    ET.fromstring(zf.read(name))  # raises if malformed

    def test_empty_document_refused_with_kind(self):
        with self.assertRaises(DocumentError) as caught:
            to_pptx(Document(format="t"))
        self.assertEqual(caught.exception.kind, "empty")


# ── surfaces ────────────────────────────────────────────────────────────────

class TestWriteSurfaces(unittest.TestCase):
    def test_tool_targets_include_office_writers(self):
        self.assertIn("xlsx", expose.CONVERT_TARGETS)
        self.assertIn("pptx", expose.CONVERT_TARGETS)
        blob = expose.convert_document(_fixture(), "xlsx")
        self.assertTrue(blob.startswith(b"PK"))
        blob = expose.convert_document(_fixture(), "pptx")
        self.assertTrue(blob.startswith(b"PK"))
        caps = expose.document_capabilities()
        self.assertIn("stdlib", caps["optional_deps"]["xlsx_output"])
        self.assertIn("stdlib", caps["optional_deps"]["pptx_output"])

    def test_cli_convert_writes_office_files(self):
        from nomorals.cmdline.commands.doc import _doc_convert
        from nomorals.documents import to_docx
        with _tmp() as td:
            src = Path(td) / "in.docx"
            src.write_bytes(to_docx(_fixture()))
            for target in ("xlsx", "pptx"):
                out = Path(td) / f"out.{target}"
                rc = _doc_convert(
                    SimpleNamespace(to=target, out=str(out)), [str(src)])
                self.assertEqual(rc, 0)
                self.assertTrue(out.read_bytes().startswith(b"PK"))
                back = parse_path(str(out))
                from nomorals.documents.model import full_text
                # xlsx reads back as tables; pptx as slide sections —
                # either way the content must be there
                self.assertIn("007", full_text(back))


# ── embedded-font PDF ───────────────────────────────────────────────────────

_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


@unittest.skipUnless(Path(_FONT).is_file(), "DejaVu font not present")
class TestEmbeddedPdf(unittest.TestCase):
    def _font(self) -> bytes:
        return Path(_FONT).read_bytes()

    def test_non_winansi_round_trip_exact(self):
        from nomorals.core.pdf import render_pdf_embedded
        text = "Report — Αθήνα και Москва, €50, arrow → done."
        blob = render_pdf_embedded(text, font_data=self._font(),
                                   title="R")
        self.assertIn(b"/FontFile2", blob)
        self.assertIn(b"/ToUnicode", blob)
        self.assertIn(b"/Identity-H", blob)
        extracted = read_pdf_text(blob)
        for needle in ("Αθήνα", "Москва", "€50", "→"):
            self.assertIn(needle, extracted)

    def test_legacy_writer_loses_what_embedded_keeps(self):
        text = "Αθήνα"
        legacy = read_pdf_text(render_pdf(text, title="R"))
        self.assertNotIn("Αθήνα", legacy)
        self.assertIn("?", legacy)

    def test_subset_font_is_valid_and_small(self):
        import zlib

        from nomorals.core.pdf import (_TtfFont, _parse_objects,
                                       _stream_bytes, render_pdf_embedded)
        blob = render_pdf_embedded("Tiny Α set.", font_data=self._font(),
                                   title="R")
        raw = None
        for body in _parse_objects(blob).values():
            if b"/Length1" in body:
                raw = zlib.decompress(_stream_bytes(body) or b"")
        self.assertIsNotNone(raw)
        subset = _TtfFont(raw)
        full = _TtfFont(self._font())
        self.assertLess(subset.num_glyphs, full.num_glyphs)
        self.assertIn(0x391, subset.cmap)  # Alpha survived subsetting

    def test_composite_glyphs_survive(self):
        from nomorals.core.pdf import render_pdf_embedded
        blob = render_pdf_embedded("Café résumé", font_data=self._font(),
                                   title="R")
        self.assertIn("Café résumé", read_pdf_text(blob))

    def test_bad_font_data_refused(self):
        from nomorals.core.pdf import render_pdf_embedded
        with self.assertRaises(PdfError):
            render_pdf_embedded("x", font_data=b"not a font at all")
        cff = b"OTTO" + b"\x00" * 64
        with self.assertRaises(PdfError):
            render_pdf_embedded("x", font_data=cff)

    def test_convert_entry_point(self):
        from nomorals.documents import to_pdf_embedded
        blob = to_pdf_embedded(_fixture(), _FONT)
        self.assertIn("Αθήνα", read_pdf_text(blob))


if __name__ == "__main__":
    unittest.main()
