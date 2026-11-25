"""Watermark onto every public artifact (audit N6).

The designed Peacethefirst mark already rides PDF (every page)
and HTML editions (the fixed layer) — this suite pins the
artifact writers that had nothing at the ship base:

* EPUB (documents.convert): a title page first in the spine
  carrying the brand line (the books EPUB's shape);
* DOCX: a closing brand paragraph;
* XLSX: one trailing brand row on the first sheet (no new sheet);
* PPTX: one rotated, translucent brand textbox per slide, drawn
  behind the content (same text, −28° and 0.15 alpha as the PDF);
* deliver_report HTML: the report-watermark fixed layer (the body
  renders chrome-free, so this function is the brand carrier).

Every writer resolves through the one home
(core.branding.resolve_brand_mark): an explicit brand_policy wins,
the manifest's policy is the fallback, "off" writes no mark and
no brand string anywhere, "custom" writes the caller's text.
Round-trips stay intact — the mark is furniture, never content.

Fake data only; no network.
"""

from __future__ import annotations

import io
import os
import re
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from nomorals.core.branding import BRAND_TEXT  # noqa: E402
from nomorals.documents import (  # noqa: E402
    Document,
    Section,
    Table,
    expose,  # noqa: E402
    parse_bytes,
    to_docx,
    to_epub,
    to_html,
    to_pptx,
    to_xlsx,
)

BRAND = BRAND_TEXT


def _doc() -> Document:
    return Document(
        id="wm-art", format="md", title="Prove Doc", author="Au",
        source="prove.md",
        sections=[Section(level=1, heading="Intro",
                          text="First paragraph of body text."),
                  Section(level=2, heading="More",
                          text="Second paragraph of body text.")],
        tables=[Table(name="Data", headers=["N", "Code"],
                      rows=[["42", "007"], ["7", "1,234"]])])


def _zip_texts(data: bytes) -> dict[str, str]:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        return {n: zf.read(n).decode("utf-8", "replace")
                for n in zf.namelist()}


def _slides(data: bytes) -> dict[str, str]:
    return {n: t for n, t in _zip_texts(data).items()
            if n.startswith("ppt/slides/slide") and n.endswith(".xml")}


class HtmlAlreadyCoveredTests(unittest.TestCase):
    def test_to_html_default_carries_the_layer(self) -> None:
        html = to_html(_doc())
        self.assertIn('class="watermark"', html)
        self.assertIn(f"<span>{BRAND}</span>", html)

    def test_to_html_off_is_clean(self) -> None:
        self.assertNotIn("peacethefirst",
                         to_html(_doc(), brand_policy="off"))


class EpubWatermarkTests(unittest.TestCase):
    def test_default_title_page_carries_brand_first_in_spine(self) -> None:
        data = to_epub(_doc())
        pages = {n: t for n, t in _zip_texts(data).items()
                 if n.endswith(".xhtml")}
        branded = [n for n, t in pages.items() if BRAND in t]
        self.assertEqual(branded, ["OEBPS/title.xhtml"])
        opf = _zip_texts(data)["OEBPS/content.opf"]
        self.assertLess(opf.find("title.xhtml"),
                        opf.find("chapter1.xhtml"))

    def test_off_writes_no_title_page_and_no_brand(self) -> None:
        data = to_epub(_doc(), brand_policy="off")
        parts = _zip_texts(data)
        self.assertNotIn("title.xhtml", parts)
        self.assertNotIn("peacethefirst", "".join(parts.values()))

    def test_custom_text_replaces_the_house_mark(self) -> None:
        data = to_epub(_doc(), brand_policy="custom",
                       brand_text="ACME HOUSE")
        text = "".join(_zip_texts(data).values())
        self.assertIn("ACME HOUSE", text)
        self.assertNotIn("peacethefirst", text)

    def test_round_trip_intact(self) -> None:
        back = parse_bytes(to_epub(_doc()), filename="x.epub")
        self.assertEqual(back.title, "Prove Doc")
        self.assertGreaterEqual(len(back.sections), 2)


class DocxWatermarkTests(unittest.TestCase):
    def test_default_closing_brand_paragraph(self) -> None:
        xml = _zip_texts(to_docx(_doc()))["word/document.xml"]
        self.assertIn(BRAND, xml)

    def test_off_is_clean(self) -> None:
        xml = _zip_texts(
            to_docx(_doc(), brand_policy="off"))["word/document.xml"]
        self.assertNotIn("peacethefirst", xml)

    def test_custom_text(self) -> None:
        xml = _zip_texts(to_docx(
            _doc(), brand_policy="custom",
            brand_text="ACME HOUSE"))["word/document.xml"]
        self.assertIn("ACME HOUSE", xml)
        self.assertNotIn("peacethefirst", xml)

    def test_round_trip_intact(self) -> None:
        back = parse_bytes(to_docx(_doc()), filename="x.docx")
        self.assertEqual(back.title, "Prove Doc")
        self.assertIn("Intro", [s.heading for s in back.sections])


class XlsxWatermarkTests(unittest.TestCase):
    def test_default_brand_cell(self) -> None:
        shared = _zip_texts(to_xlsx(_doc()))["xl/sharedStrings.xml"]
        self.assertIn(BRAND, shared)

    def test_off_is_clean(self) -> None:
        shared = _zip_texts(
            to_xlsx(_doc(), brand_policy="off"))["xl/sharedStrings.xml"]
        self.assertNotIn("peacethefirst", shared)

    def test_custom_text(self) -> None:
        shared = _zip_texts(to_xlsx(
            _doc(), brand_policy="custom",
            brand_text="ACME HOUSE"))["xl/sharedStrings.xml"]
        self.assertIn("ACME HOUSE", shared)
        self.assertNotIn("peacethefirst", shared)

    def test_no_new_sheet_and_data_intact(self) -> None:
        data = to_xlsx(_doc())
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            workbook = zf.read("xl/workbook.xml").decode()
        names = re.findall(r'<sheet name="([^"]+)"', workbook)
        self.assertEqual(names, ["Document", "Data"])
        back = parse_bytes(data, filename="f.xlsx")
        table = next(t for t in back.tables if t.name == "Data")
        self.assertEqual(table.headers, ["N", "Code"])
        self.assertEqual(table.rows[0][0], "42")

    def test_tables_only_doc_marks_first_sheet(self) -> None:
        doc = Document(format="t", tables=[
            Table(name="Data", headers=["a"], rows=[["1"]])])
        shared = _zip_texts(to_xlsx(doc))["xl/sharedStrings.xml"]
        self.assertIn(BRAND, shared)


class PptxWatermarkTests(unittest.TestCase):
    def test_every_slide_marked(self) -> None:
        slides = _slides(to_pptx(_doc()))
        self.assertGreaterEqual(len(slides), 2)
        for name, xml in slides.items():
            self.assertIn(BRAND, xml, name)

    def test_mark_is_rotated_translucent_furniture(self) -> None:
        slides = _slides(to_pptx(_doc()))
        for xml in slides.values():
            # −28° in DrawingML 60000ths of a degree, 15% alpha
            self.assertIn('rot="-1680000"', xml)
            self.assertIn('<a:alpha val="15000"/>', xml)
            # drawn first in the shape tree: content paints over it
            self.assertLess(xml.find('name="Brand"'),
                            xml.find('name="Title"'))

    def test_off_is_clean(self) -> None:
        for xml in _slides(to_pptx(_doc(), brand_policy="off")).values():
            self.assertNotIn("peacethefirst", xml)
            self.assertNotIn('name="Brand"', xml)

    def test_custom_text(self) -> None:
        slides = _slides(to_pptx(_doc(), brand_policy="custom",
                                  brand_text="ACME HOUSE"))
        for xml in slides.values():
            self.assertIn("ACME HOUSE", xml)
            self.assertNotIn("peacethefirst", xml)

    def test_all_xml_well_formed_and_round_trip_intact(self) -> None:
        import xml.etree.ElementTree as ET

        data = to_pptx(_doc())
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in zf.namelist():
                if name.endswith((".xml", ".rels")):
                    ET.fromstring(zf.read(name))
        back = parse_bytes(data, filename="f.pptx")
        from nomorals.documents.model import full_text

        text = full_text(back)
        for needle in ("Prove Doc", "Intro", "42", "007"):
            self.assertIn(needle, text)


class ExposePassthroughTests(unittest.TestCase):
    def test_convert_document_threads_brand_policy(self) -> None:
        for fmt in ("epub", "docx", "xlsx", "pptx"):
            off = expose.convert_document(_doc(), fmt,
                                          brand_policy="off")
            # unzipped: docx/xlsx/pptx deflate their XML, so a raw
            # byte scan would pass vacuously
            self.assertNotIn("peacethefirst",
                             "".join(_zip_texts(off).values()))
            marked = expose.convert_document(_doc(), fmt)
            self.assertIn("peacethefirst",
                          "".join(_zip_texts(marked).values()))


class ReportHtmlWatermarkTests(unittest.TestCase):
    _SECTIONS = [("Overview", "markets moved **up** today."),
                 ("Risks", "watch the curve.")]

    def test_default_watermark_layer_and_footer(self) -> None:
        from nomorals.tools import deliver_report as DR

        out = DR.render_report_html("Q3 markets", self._SECTIONS,
                                    generated="2026-10-01 12:00")
        self.assertIn('class="report-watermark"', out)
        self.assertIn(f"<span>{BRAND}</span>", out)
        self.assertIn(".report-watermark{position:fixed", out)
        self.assertIn(BRAND, out)

    def test_off_removes_layer_and_footer_brand(self) -> None:
        from nomorals.tools import deliver_report as DR

        out = DR.render_report_html("Q3 markets", self._SECTIONS,
                                    generated="2026-10-01 12:00",
                                    brand_policy="off")
        # the stylesheet still defines the class (as the edition's
        # does); the layer element itself must be gone
        self.assertNotIn('class="report-watermark"', out)
        self.assertNotIn("peacethefirst", out)

    def test_custom_text(self) -> None:
        from nomorals.tools import deliver_report as DR

        out = DR.render_report_html(
            "Q3 markets", self._SECTIONS, generated="2026-10-01 12:00",
            brand_policy="custom", brand_text="ACME HOUSE")
        self.assertIn("ACME HOUSE", out)
        self.assertNotIn("peacethefirst", out)

    def test_generate_report_marks_html_and_pdf(self) -> None:
        from nomorals.core.pdf import read_pdf_text
        from nomorals.tools import deliver_report as DR

        ctx = SimpleNamespace(
            settings=SimpleNamespace(workspace_dir=tempfile.mkdtemp()),
            extras={}, router=None)
        bundle = DR.generate_report(ctx, "Q3 markets", self._SECTIONS,
                                    generated="2026-10-01 12:00")
        html = Path(bundle.html_path).read_text(encoding="utf-8")
        self.assertIn('class="report-watermark"', html)
        self.assertIn(BRAND, html)
        # the PDF's streams are Flate-compressed: read the mark out
        # of the extracted page text, not the raw bytes
        pdf = Path(bundle.pdf_path).read_bytes()
        self.assertIn(b"/GSW", pdf)
        self.assertIn("peacethefirst", read_pdf_text(pdf))

    def test_generate_report_off_marks_neither_file(self) -> None:
        from nomorals.core.pdf import read_pdf_text
        from nomorals.tools import deliver_report as DR

        ctx = SimpleNamespace(
            settings=SimpleNamespace(workspace_dir=tempfile.mkdtemp()),
            extras={}, router=None)
        bundle = DR.generate_report(ctx, "Q3 markets", self._SECTIONS,
                                    generated="2026-10-01 12:00",
                                    brand_policy="off")
        html = Path(bundle.html_path).read_text(encoding="utf-8")
        self.assertNotIn("peacethefirst", html)
        pdf = Path(bundle.pdf_path).read_bytes()
        self.assertNotIn(b"/GSW", pdf)
        self.assertNotIn("peacethefirst", read_pdf_text(pdf))


if __name__ == "__main__":
    unittest.main()
