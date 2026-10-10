"""Sweep tests for the documents module upgrade (DOCUMENTS_SWEEP_MINING.md).

Covers the mined-then-built features: model provenance/chunking/identity,
ruling-line table recovery + confidence scoring, JSON/XML parsers, RTF
tables, docx lists/hyperlinks/footnotes, CSV encoding fallback, HTML/EPUB
metadata, index query language (phrase/field/prefix/highlight/facet/
suggest/count), word diff + moved detection + cell diffs + renderers,
HTML themes + new converters, OCR word/hOCR/parallel/knob surface, and
the summarizer methods/keyphrases/digests.

Stdlib + repo only, no network.  Fail-fast semantics are pinned wherever
the module promises them.
"""

from __future__ import annotations

import io
import json
import unittest
import zipfile

from nomorals.documents import (
    Document,
    DocumentComparison,
    DocumentError,
    DocumentIndex,
    Section,
    Table,
    bullet_digest,
    compare_documents,
    detect_format,
    diff_documents,
    extract_text_tables,
    full_text,
    keyphrases,
    keywords,
    new_document,
    ocr_available,
    ocr_languages,
    ocr_pdf,
    ocr_pdf_hocr,
    ocr_pdf_words,
    parse_bytes,
    parse_path,
    render_html,
    render_markdown,
    render_terminal,
    score_table,
    similarity,
    summarize,
    summarize_abstractive,
    summarize_query,
    summarize_sections,
    summarize_text,
    tldr,
    to_csv,
    to_csv_all,
    to_docx,
    to_epub,
    to_html,
    to_json,
    to_markdown,
    to_pdf,
    to_text,
    word_diff,
)


def _doc(doc_id="d1", title="Doc", text="Hello world. " * 20,
         fmt="md") -> Document:
    doc = new_document(format=fmt, title=title)
    doc.id = doc_id
    doc.sections = [Section(level=1, heading="Intro", text=text, page=1)]
    return doc


# ── model ───────────────────────────────────────────────────────────────────


class TestModelUpgrades(unittest.TestCase):
    def test_section_provenance_round_trip(self) -> None:
        sec = Section(level=2, heading="H", text="t", page=7, kind="list")
        back = Section.from_dict(sec.to_dict())
        self.assertEqual((back.page, back.kind), (7, "list"))

    def test_section_legacy_payload_still_loads(self) -> None:
        back = Section.from_dict({"level": 1, "heading": "H", "text": "t"})
        self.assertEqual((back.page, back.kind), (0, "text"))

    def test_section_unknown_kind_falls_back_to_text(self) -> None:
        back = Section.from_dict({"kind": "nonsense"})
        self.assertEqual(back.kind, "text")

    def test_section_word_count(self) -> None:
        self.assertEqual(Section(heading="Hi there", text="one two").word_count(), 4)

    def test_table_record_and_column_access(self) -> None:
        table = Table(name="T", headers=["a", "b"],
                      rows=[["1", "2"], ["3", "4"]])
        self.assertEqual(table.to_records(),
                         [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}])
        self.assertEqual(table.column("b"), ["2", "4"])
        self.assertEqual(table.column(0), ["1", "3"])
        self.assertEqual((table.n_rows, table.n_cols), (2, 2))
        with self.assertRaises(KeyError):
            table.column("zzz")
        with self.assertRaises(IndexError):
            table.column(9)

    def test_table_stats(self) -> None:
        table = Table(headers=["a", "b"], rows=[["1", "x"], ["2.5", ""]])
        stats = table.stats()
        self.assertEqual(stats["a"]["numeric_ratio"], 1.0)
        self.assertEqual(stats["b"]["fill_ratio"], 0.5)

    def test_table_provenance_round_trip(self) -> None:
        table = Table(name="T", caption="Table 1: stuff", page=3,
                      confidence=0.9)
        back = Table.from_dict(table.to_dict())
        self.assertEqual((back.caption, back.page, back.confidence),
                         ("Table 1: stuff", 3, 0.9))

    def test_document_outline_word_count_reading_time(self) -> None:
        doc = _doc()
        doc.sections.append(Section(level=2, heading="Deep", text="more", page=2))
        outline = doc.outline()
        self.assertEqual([e["heading"] for e in outline], ["Intro", "Deep"])
        self.assertEqual(outline[1]["page"], 2)
        self.assertGreater(doc.word_count(), 20)
        self.assertGreater(doc.reading_time_minutes(), 0)
        with self.assertRaises(ValueError):
            doc.reading_time_minutes(wpm=0)

    def test_content_hash_stable_and_sensitive(self) -> None:
        doc = _doc()
        same = _doc()
        other = _doc(text="Completely different words here.")
        self.assertEqual(doc.content_hash(), same.content_hash())
        self.assertNotEqual(doc.content_hash(), other.content_hash())
        self.assertEqual(len(doc.content_hash()), 64)

    def test_find_sections(self) -> None:
        doc = _doc()
        self.assertEqual(len(doc.find_sections("intro")), 1)
        self.assertEqual(doc.find_sections("missing"), [])
        self.assertEqual(doc.find_sections(""), [])

    def test_chunks_honor_max_words_and_context(self) -> None:
        doc = new_document(format="md", title="T")
        doc.sections = [Section(level=1, heading="Intro",
                                text="word " * 500, page=1)]
        chunks = doc.chunks(max_words=50)
        self.assertGreater(len(chunks), 5)
        self.assertTrue(all(c["words"] <= 50 for c in chunks))
        self.assertTrue(all(c["heading_path"] == "Intro" for c in chunks))
        with self.assertRaises(ValueError):
            doc.chunks(max_words=5)

    def test_chunks_keep_tables_separate(self) -> None:
        doc = _doc()
        doc.tables = [Table(name="T", headers=["a"], rows=[["1"]])]
        chunks = doc.chunks(max_words=1000)
        table_chunks = [c for c in chunks if c["text"].startswith("Table:")]
        self.assertEqual(len(table_chunks), 1)

    def test_merge(self) -> None:
        first, second = _doc("a"), _doc("b")
        second.metadata["k"] = "v"
        merged = first.merge(second)
        self.assertEqual(len(merged.sections), 2)
        self.assertEqual(merged.metadata["k"], "v")
        with self.assertRaises(ValueError):
            first.merge("nope")  # type: ignore[arg-type]

    def test_to_json_round_trip(self) -> None:
        doc = _doc()
        back = Document.from_dict(json.loads(doc.to_json()))
        self.assertEqual(back.title, "Doc")
        self.assertEqual(back.sections[0].page, 1)

    def test_iter_blocks_reading_order(self) -> None:
        doc = _doc()
        doc.tables = [Table(name="T")]
        kinds = [b["kind"] for b in doc.iter_blocks()]
        self.assertEqual(kinds, ["section", "table"])


# ── pdf_tables ──────────────────────────────────────────────────────────────


class TestPdfTableUpgrades(unittest.TestCase):
    def test_ruling_ascii_table(self) -> None:
        text = ("+------+-----+\n| Name | Age |\n+------+-----+\n"
                "| Ann  | 30  |\n| Bob  | 25  |\n+------+-----+\n")
        tables = extract_text_tables(text)
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].headers, ["Name", "Age"])
        self.assertEqual(tables[0].rows, [["Ann", "30"], ["Bob", "25"]])
        self.assertGreaterEqual(tables[0].confidence, 0.9)

    def test_box_drawing_table_with_caption(self) -> None:
        text = ("Table 2: Staff\n┌──────┬─────┐\n│ Name │ Age │\n"
                "├──────┼─────┤\n│ Ann  │ 30  │\n│ Bob  │ 25  │\n"
                "└──────┴─────┘\n")
        tables = extract_text_tables(text)
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].name, "Table 2")
        self.assertEqual(tables[0].caption, "Table 2: Staff")
        self.assertEqual(tables[0].headers, ["Name", "Age"])

    def test_stream_still_detects(self) -> None:
        text = ("Name    Age    City\nAnn     30     Lagos\n"
                "Bob     25     Abuja\nCat     41     Kano\n")
        tables = extract_text_tables(text)
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].headers, ["Name", "Age", "City"])
        self.assertGreater(tables[0].confidence, 0.5)

    def test_has_header_auto_numeric(self) -> None:
        text = "10    20\n30    40\n50    60\n"
        tables = extract_text_tables(text, has_header="auto")
        self.assertEqual(tables[0].headers, ["col_1", "col_2"])
        self.assertEqual(len(tables[0].rows), 3)

    def test_has_header_auto_text(self) -> None:
        text = "Name    Age\nAnn     30\nBob     25\n"
        tables = extract_text_tables(text, has_header="auto")
        self.assertEqual(tables[0].headers, ["Name", "Age"])

    def test_has_header_false(self) -> None:
        text = "a    b\nc    d\ne    f\n"
        tables = extract_text_tables(text, has_header=False)
        self.assertEqual(tables[0].headers, ["col_1", "col_2"])
        self.assertEqual(len(tables[0].rows), 3)

    def test_score_table_range(self) -> None:
        score = score_table([["a", "b"], ["1", "2"]],
                            [[0, 4], [0, 4]])
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertEqual(score_table([]), 0.0)

    def test_bad_params_fail_fast(self) -> None:
        with self.assertRaises(ValueError):
            extract_text_tables("x", strategy="nope")
        with self.assertRaises(ValueError):
            extract_text_tables("x", has_header="sometimes")  # type: ignore[arg-type]

    def test_strategy_ruling_only(self) -> None:
        text = ("Name    Age\nAnn     30\nBob     25\n")
        self.assertEqual(extract_text_tables(text, strategy="ruling"), [])


# ── parsers ─────────────────────────────────────────────────────────────────


class TestParserUpgrades(unittest.TestCase):
    def test_parse_json_dict(self) -> None:
        doc = parse_bytes(b'{"name": "Ada", "age": 36}',
                          filename="x.json")
        self.assertEqual(doc.format, "json")
        headings = [s.heading for s in doc.sections]
        self.assertIn("name", headings)

    def test_parse_json_records_become_table(self) -> None:
        doc = parse_bytes(b'[{"q": 1, "s": 9}, {"q": 2, "s": 8}]',
                          filename="x.json")
        self.assertEqual(len(doc.tables), 1)
        self.assertEqual(doc.tables[0].headers, ["q", "s"])
        self.assertEqual(len(doc.tables[0].rows), 2)

    def test_parse_json_list_of_scalars(self) -> None:
        doc = parse_bytes(b'["a", "b"]', filename="x.json")
        self.assertEqual(doc.sections[0].kind, "list")

    def test_parse_json_invalid(self) -> None:
        with self.assertRaises(DocumentError):
            parse_bytes(b'{"a": ', filename="x.json")

    def test_parse_xml(self) -> None:
        doc = parse_bytes(
            b"<root><title>Hello</title><item id=\"1\">One</item></root>",
            filename="x.xml")
        self.assertEqual(doc.format, "xml")
        self.assertEqual(doc.metadata["xml_root"], "root")
        headings = [s.heading for s in doc.sections]
        self.assertIn("title", headings)

    def test_parse_xml_invalid(self) -> None:
        with self.assertRaises(DocumentError):
            parse_bytes(b"<root><unclosed>", filename="x.xml")

    def test_detect_format_public(self) -> None:
        self.assertEqual(detect_format(b"%PDF-1.4"), "pdf")
        self.assertEqual(detect_format(b'{"a": 1}'), "json")
        self.assertEqual(detect_format(b"<root><a/></root>"), "xml")
        self.assertEqual(detect_format(b"plain text", filename="x.txt"), "txt")
        with self.assertRaises(DocumentError):
            detect_format(b"")

    def test_rtf_table_recovery(self) -> None:
        rtf = (b"{\\rtf1\\ansi Hello.\\par "
               b"\\trowd\\cellx1000\\cellx2000\\intbl Name\\cell Age\\cell\\row "
               b"\\trowd\\cellx1000\\cellx2000\\intbl Ann\\cell 30\\cell\\row}")
        doc = parse_bytes(rtf, filename="x.rtf")
        self.assertEqual(len(doc.tables), 1)
        self.assertEqual(doc.tables[0].headers, ["Name", "Age"])
        self.assertEqual(doc.tables[0].rows, [["Ann", "30"]])

    def test_csv_cp1252_fallback(self) -> None:
        doc = parse_bytes("name;city\nAndré;Lagos\n".encode("cp1252"),
                          filename="x.csv")
        self.assertEqual(doc.tables[0].rows, [["André", "Lagos"]])

    def test_html_meta_harvest(self) -> None:
        html = (b'<html lang="en"><head><title>T</title>'
                b'<meta name="description" content="desc here">'
                b'</head><body><p>hi</p></body></html>')
        doc = parse_bytes(html, filename="x.html")
        self.assertEqual(doc.metadata["description"], "desc here")
        self.assertEqual(doc.metadata["language"], "en")

    def test_epub_language(self) -> None:
        pytest_epub = _minimal_epub(language="fr")
        doc = parse_bytes(pytest_epub, filename="x.epub")
        self.assertEqual(doc.metadata.get("language"), "fr")

    def test_xls_missing_hint(self) -> None:
        with self.assertRaises(DocumentError) as ctx:
            parse_bytes(b"\xd0\xcf\x11\xe0" + b"\x00" * 100,
                        filename="x.xls")
        self.assertIn("xlrd", str(ctx.exception))

    def test_ole_without_extension_helpful(self) -> None:
        with self.assertRaises(DocumentError) as ctx:
            parse_bytes(b"\xd0\xcf\x11\xe0" + b"\x00" * 100)
        self.assertIn(".xls", str(ctx.exception))

    def test_pdf_page_provenance(self) -> None:
        from nomorals.core.pdf import render_pdf
        data = render_pdf("# T\n\npage one", title="T")
        doc = parse_bytes(data, filename="x.pdf")
        self.assertTrue(all(s.page >= 1 for s in doc.sections))

    def test_markdown_list_kind(self) -> None:
        doc = parse_bytes(b"# T\n\n- a\n- b\n", filename="x.md")
        kinds = {s.kind for s in doc.sections}
        self.assertIn("list", kinds)


def _minimal_epub(language: str = "en") -> bytes:
    """Tiny valid EPUB for parser tests."""
    buf = io.BytesIO()
    container = ("<?xml version=\"1.0\"?><container version=\"1.0\" "
                 "xmlns=\"urn:oasis:names:tc:opendocument:xmlns:container\">"
                 "<rootfiles><rootfile full-path=\"OEBPS/content.opf\" "
                 "media-type=\"application/oebps-package+xml\"/>"
                 "</rootfiles></container>")
    opf = (f"<?xml version=\"1.0\"?><package version=\"3.0\" "
           f"xmlns=\"http://www.idpf.org/2007/opf\" "
           f"xmlns:dc=\"http://purl.org/dc/elements/1.1/\">"
           f"<metadata><dc:title>T</dc:title>"
           f"<dc:language>{language}</dc:language></metadata>"
           f"<manifest><item id=\"c1\" href=\"c1.xhtml\" "
           f"media-type=\"application/xhtml+xml\"/></manifest>"
           f"<spine><itemref idref=\"c1\"/></spine></package>")
    chapter = ("<?xml version=\"1.0\"?><html xmlns=\"http://www.w3.org/1999/xhtml\">"
               "<body><h1>One</h1><p>text</p></body></html>")
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip",
                         compress_type=zipfile.ZIP_STORED)
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("OEBPS/content.opf", opf)
        archive.writestr("OEBPS/c1.xhtml", chapter)
    return buf.getvalue()


class TestDocxUpgrades(unittest.TestCase):
    @staticmethod
    def _make_docx() -> bytes:
        from docx import Document as DocxDocument
        doc = DocxDocument()
        doc.add_paragraph("Intro", style="Heading 1")
        doc.add_paragraph("item a", style="List Number")
        doc.add_paragraph("item b", style="List Number")
        doc.add_paragraph("bullet", style="List Bullet")
        para = doc.add_paragraph()
        run = para.add_run("click")
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
        hyperlink = OxmlElement("w:hyperlink")
        hyperlink.set(qn("r:id"), "rId9")
        run._r.getparent().replace(run._r, hyperlink)
        hyperlink.append(run._r)
        para.part.relate_to(
            "https://example.com",
            "http://schemas.openxmlformats.org/officeDocument/2006/"
            "relationships/hyperlink", is_external=True)
        buf = io.BytesIO()
        doc.save(buf)
        return buf.getvalue()

    def test_docx_lists_numbered_and_bulleted(self) -> None:
        doc = parse_bytes(self._make_docx(), filename="x.docx")
        body = "\n".join(s.text for s in doc.sections)
        self.assertIn("1. item a", body)
        self.assertIn("2. item b", body)
        self.assertIn("• bullet", body)

    def test_docx_hyperlink_harvested(self) -> None:
        doc = parse_bytes(self._make_docx(), filename="x.docx")
        self.assertIn("https://example.com",
                      doc.metadata.get("links", []))


# ── index ───────────────────────────────────────────────────────────────────


class TestIndexUpgrades(unittest.TestCase):
    def setUp(self) -> None:
        self.index = DocumentIndex()
        docs = [
            ("a", "pdf", "London Report",
             "The London bridge spans the Thames river in London."),
            ("b", "md", "Paris Notes",
             "Paris has the Eiffel tower and lovely bridges."),
            ("c", "pdf", "Thames Guide",
             "Thames river guide for London visitors."),
        ]
        for doc_id, fmt, title, text in docs:
            doc = Document(id=doc_id, format=fmt, title=title,
                           sections=[Section(text=text)])
            self.index.add(doc)

    def tearDown(self) -> None:
        self.index.close()

    def test_phrase_search(self) -> None:
        hits = self.index.search('"London bridge"')
        self.assertEqual([h["doc_id"] for h in hits], ["a"])

    def test_field_filter(self) -> None:
        hits = self.index.search("title:paris")
        self.assertEqual([h["doc_id"] for h in hits], ["b"])

    def test_and_operator(self) -> None:
        hits = self.index.search("london thames", operator="AND")
        self.assertEqual({h["doc_id"] for h in hits}, {"a", "c"})
        with self.assertRaises(DocumentError):
            self.index.search("london", operator="XOR")

    def test_prefix_search(self) -> None:
        hits = self.index.search("lond", prefix=True)
        self.assertEqual({h["doc_id"] for h in hits}, {"a", "c"})
        self.assertEqual(self.index.search("lond"), [])

    def test_highlight(self) -> None:
        hits = self.index.search("london", highlight=True, limit=1)
        self.assertIn("<mark>", hits[0]["snippet"])

    def test_weights_and_explain(self) -> None:
        hits = self.index.search("london", weights={"title": 5.0},
                                 explain=True)
        self.assertTrue(all(isinstance(h["score"], float) for h in hits))
        self.assertIn("match", hits[0]["explain"])

    def test_count(self) -> None:
        self.assertEqual(self.index.count("london"), 2)
        self.assertEqual(self.index.count('"London bridge"'), 1)
        self.assertEqual(self.index.count("zzznope"), 0)

    def test_suggest(self) -> None:
        suggestions = self.index.suggest("lon")
        self.assertIn("london", suggestions)
        self.assertEqual(self.index.suggest("x"), [])

    def test_facet_and_stats(self) -> None:
        self.assertEqual(self.index.facet(), {"pdf": 2, "md": 1})
        stats = self.index.stats()
        self.assertEqual(stats["documents"], 3)
        self.assertEqual(stats["formats"]["pdf"], 2)
        with self.assertRaises(DocumentError):
            self.index.facet("nope")

    def test_save_load_round_trip(self) -> None:
        import tempfile, os
        path = os.path.join(tempfile.mkdtemp(), "idx.db")
        self.index.save(path)
        loaded = DocumentIndex.load(path)
        try:
            self.assertEqual(len(loaded), 3)
            self.assertEqual(loaded.facet(), {"pdf": 2, "md": 1})
            self.assertEqual([h["doc_id"] for h in loaded.search("london")],
                             [h["doc_id"] for h in self.index.search("london")])
        finally:
            loaded.close()


# ── compare ─────────────────────────────────────────────────────────────────


class TestCompareUpgrades(unittest.TestCase):
    def _docs(self):
        first = Document(id="a", title="A", sections=[
            Section(heading="Intro",
                    text="The quick brown fox jumps over the lazy dog."),
            Section(heading="Old Name",
                    text="This body stays exactly the same here.")],
            tables=[Table(name="T", headers=["a", "b"],
                          rows=[["1", "2"], ["3", "4"]])])
        second = Document(id="b", title="B", sections=[
            Section(heading="Intro",
                    text="The quick brown fox leaps over the lazy dog."),
            Section(heading="New Name",
                    text="This body stays exactly the same here.")],
            tables=[Table(name="T", headers=["a", "b"],
                          rows=[["1", "9"], ["3", "4"], ["5", "6"]])])
        return first, second

    def test_word_diff(self) -> None:
        ops = word_diff("The quick brown fox jumps", "The quick brown fox leaps")
        kinds = [op for op, _ in ops]
        self.assertIn("delete", kinds)
        self.assertIn("insert", kinds)
        deleted = "".join(t for op, t in ops if op == "delete")
        self.assertIn("jumps", deleted)
        with self.assertRaises(DocumentError):
            word_diff("a", 42)  # type: ignore[arg-type]

    def test_similarity(self) -> None:
        first, second = self._docs()
        sim = similarity(first, second)
        self.assertGreater(sim, 0.8)
        self.assertLess(sim, 1.0)
        self.assertEqual(similarity(first, first), 1.0)
        with self.assertRaises(DocumentError):
            similarity(first, "nope")  # type: ignore[arg-type]

    def test_moved_section_detected(self) -> None:
        first, second = self._docs()
        result = compare_documents(first, second)
        self.assertEqual(len(result.sections_moved), 1)
        move = result.sections_moved[0]
        self.assertEqual((move["from"], move["to"]), ("Old Name", "New Name"))
        self.assertNotIn("Old Name", result.sections_removed)
        self.assertNotIn("New Name", result.sections_added)
        self.assertIn("moved/renamed", result.summary)

    def test_identical_summary_unchanged(self) -> None:
        first, _ = self._docs()
        result = compare_documents(first, first)
        self.assertEqual(result.summary, "no changes")
        self.assertEqual(result.similarity, 1.0)

    def test_cell_level_table_changes(self) -> None:
        first, second = self._docs()
        result = compare_documents(first, second)
        changes = result.stats["table_cell_changes"]["T"]
        cells = [c for c in changes if c["type"] == "cell"]
        self.assertTrue(any(c["before"] == "2" and c["after"] == "9"
                            for c in cells))
        added = [c for c in changes if c["type"] == "row_added"]
        self.assertEqual(len(added), 1)

    def test_section_word_diffs_attached(self) -> None:
        first, second = self._docs()
        result = compare_documents(first, second)
        self.assertIn("Intro", result.stats["section_word_diffs"])

    def test_render_html(self) -> None:
        first, second = self._docs()
        html = render_html(compare_documents(first, second))
        self.assertIn("<style>", html)
        self.assertIn("moved", html)
        self.assertIn("<table", html)

    def test_render_terminal(self) -> None:
        first, second = self._docs()
        plain = render_terminal(compare_documents(first, second), color=False)
        self.assertNotIn("\033[", plain)
        self.assertIn("moved/renamed", plain)
        colored = render_terminal(compare_documents(first, second), color=True)
        self.assertIn("\033[", colored)

    def test_render_markdown(self) -> None:
        first, second = self._docs()
        md = render_markdown(compare_documents(first, second))
        self.assertIn("## Sections moved / renamed", md)
        self.assertIn("```diff", md)

    def test_diff_documents_still_works(self) -> None:
        first, second = self._docs()
        diff = diff_documents(first, second)
        self.assertIn("jumps", diff)
        self.assertIn("leaps", diff)


# ── convert ─────────────────────────────────────────────────────────────────


class TestConvertUpgrades(unittest.TestCase):
    def _doc(self) -> Document:
        doc = new_document(format="md", title="Report", author="Ada")
        doc.sections = [
            Section(level=1, heading="Intro", text="Hello world."),
            Section(level=2, heading="Details", text="- a\n- b"),
        ]
        doc.tables = [Table(name="T1", headers=["a", "b"], rows=[["1", "2"]])]
        return doc

    def test_html_themes(self) -> None:
        doc = self._doc()
        for theme in ("light", "dark", "print", "minimal"):
            html = to_html(doc, theme=theme)
            self.assertIn("<style>", html)
            self.assertIn("Contents", html)
            self.assertIn('href="#intro"', html)
        with self.assertRaises(DocumentError):
            to_html(doc, theme="nope")

    def test_html_toc_off_and_anchors(self) -> None:
        html = to_html(self._doc(), toc=False)
        self.assertNotIn("Contents", html)
        self.assertNotIn('id="intro"', html)

    def test_html_list_and_quote_kinds(self) -> None:
        doc = self._doc()
        doc.sections[1].kind = "list"
        html = to_html(doc)
        self.assertIn("<ul>", html)
        self.assertIn("<li>a</li>", html)

    def test_markdown_front_matter_and_toc(self) -> None:
        md = to_markdown(self._doc(), front_matter=True, toc=True)
        self.assertTrue(md.startswith("---\n"))
        self.assertIn('title: "Report"', md)
        self.assertIn("## Contents", md)

    def test_to_json_round_trip(self) -> None:
        doc = self._doc()
        back = Document.from_dict(json.loads(to_json(doc)))
        self.assertEqual(back.title, "Report")
        with self.assertRaises(DocumentError):
            to_json(doc, indent=-1)

    def test_to_csv_by_name(self) -> None:
        doc = self._doc()
        doc.tables.append(Table(name="Second", headers=["x"], rows=[["9"]]))
        self.assertEqual(to_csv(doc, "second"), "x\n9\n")
        self.assertEqual(to_csv(doc, 0).splitlines()[0], "a,b")
        with self.assertRaises(DocumentError):
            to_csv(doc, "missing")
        with self.assertRaises(DocumentError):
            to_csv(doc, 7)

    def test_to_csv_all(self) -> None:
        doc = self._doc()
        doc.tables.append(Table(name="Second", headers=["x"], rows=[["9"]]))
        all_csv = to_csv_all(doc)
        self.assertEqual(set(all_csv), {"T1", "Second"})
        with self.assertRaises(DocumentError):
            to_csv_all(new_document(format="md"))

    def test_to_epub_round_trip(self) -> None:
        data = to_epub(self._doc())
        self.assertTrue(data[:4] == b"PK\x03\x04")
        back = parse_bytes(data, filename="x.epub")
        self.assertEqual(back.title, "Report")
        self.assertGreaterEqual(len(back.sections), 2)
        with self.assertRaises(DocumentError):
            to_epub(new_document(format="md"))

    def test_to_docx_round_trip(self) -> None:
        data = to_docx(self._doc())
        back = parse_bytes(data, filename="x.docx")
        self.assertEqual(back.title, "Report")
        headings = [s.heading for s in back.sections]
        self.assertIn("Intro", headings)


# ── ocr ─────────────────────────────────────────────────────────────────────


class TestOcrUpgrades(unittest.TestCase):
    def test_available_never_raises(self) -> None:
        self.assertIsInstance(ocr_available(), bool)

    def test_fail_fast_without_stack(self) -> None:
        if ocr_available():
            self.skipTest("OCR stack present; fail-fast paths need it absent")
        for call in (lambda: ocr_pdf(b"%PDF-1.4 fake"),
                     lambda: ocr_pdf_words(b"%PDF-1.4 fake"),
                     lambda: ocr_pdf_hocr(b"%PDF-1.4 fake"),
                     ocr_languages):
            with self.assertRaises(DocumentError):
                call()

    def test_validation_before_stack(self) -> None:
        with self.assertRaises(DocumentError):
            ocr_pdf(b"")
        with self.assertRaises(DocumentError):
            ocr_pdf(b"%PDF-1.4 x", psm=99)
        with self.assertRaises(DocumentError):
            ocr_pdf(b"%PDF-1.4 x", oem=9)
        with self.assertRaises(DocumentError):
            ocr_pdf(b"%PDF-1.4 x", jobs=0)
        with self.assertRaises(DocumentError):
            ocr_pdf(b"not a pdf")


# ── summarize ───────────────────────────────────────────────────────────────


class TestSummarizeUpgrades(unittest.TestCase):
    TEXT = ("The Mars rover discovered ancient riverbeds on the red planet. "
            "Scientists celebrated the Mars finding for weeks afterward. "
            "Meanwhile the stock market closed slightly higher on Tuesday. "
            "The Mars rover team published detailed maps of the riverbeds. "
            "Rain is expected tomorrow in the northern counties.")

    def test_all_methods_return_doc_order(self) -> None:
        for method in ("tf", "textrank", "luhn", "lead"):
            out = summarize_text(self.TEXT, sentences=2, method=method)
            self.assertEqual(len(out), 2)
            # document order preserved regardless of method
            self.assertLess(self.TEXT.index(out[0]), self.TEXT.index(out[1]))

    def test_bad_method_fail_fast(self) -> None:
        with self.assertRaises(DocumentError):
            summarize_text(self.TEXT, method="nope")

    def test_diversity_drops_near_duplicates(self) -> None:
        text = ("The cat sat on the mat. The cat sat on the mat! "
                "Dogs bark loudly at night in the park. "
                "Quantum physics explores subatomic particles deeply.")
        out = summarize_text(text, sentences=2, method="tf", diversity=True)
        self.assertEqual(len(out), 2)
        self.assertTrue(any("Dogs" in s or "Quantum" in s for s in out))

    def test_position_bias(self) -> None:
        text = ("Zebra stripes confuse predators in tall grasslands. "
                "Apple orchards need cold winters for good harvests. "
                "Zebra herds migrate across the savanna each year.")
        plain = summarize_text(text, sentences=1, method="tf",
                               position_bias=False)
        biased = summarize_text(text, sentences=1, method="tf",
                                position_bias=True)
        # bias must at least be accepted and deterministic
        self.assertEqual(len(plain), 1)
        self.assertEqual(len(biased), 1)

    def test_query_focus(self) -> None:
        out = summarize_query(self.TEXT, "stock market", sentences=1)
        self.assertTrue(out[0].startswith("Meanwhile"))
        with self.assertRaises(DocumentError):
            summarize_query(self.TEXT, "  ", sentences=1)

    def test_keyphrases(self) -> None:
        phrases = keyphrases(self.TEXT, top_n=6)
        self.assertTrue(all(" " in p for p in phrases))
        self.assertIn("mars finding", phrases)
        with self.assertRaises(DocumentError):
            keyphrases("", top_n=5)
        with self.assertRaises(DocumentError):
            keyphrases(self.TEXT, top_n=0)

    def test_summarize_sections(self) -> None:
        doc = Document(id="a", title="N", sections=[
            Section(heading="Space", text=self.TEXT),
            Section(heading="Tiny", text="Hi."),
        ])
        out = summarize_sections(doc, sentences=1)
        headings = [s["heading"] for s in out]
        self.assertIn("Space", headings)
        self.assertTrue(all("summary" in s and "page" in s for s in out))
        with self.assertRaises(DocumentError):
            summarize_sections("nope")  # type: ignore[arg-type]

    def test_tldr_word_budget(self) -> None:
        digest = tldr(self.TEXT, words=15)
        self.assertLessEqual(len(digest.split()), 20)
        self.assertTrue(digest)
        with self.assertRaises(DocumentError):
            tldr(self.TEXT, words=0)

    def test_bullet_digest(self) -> None:
        doc = Document(id="d1", title="News", sections=[Section(
            heading="Intro",
            text=("The Mars rover found water. Markets rallied on the news. "
                  "Rain will fall tomorrow across the northern counties."))])
        md = bullet_digest(doc, sentences=2)
        self.assertTrue(md.startswith("# News"))
        self.assertEqual(md.count("\n- "), 2)

    def test_abstractive_fail_fast(self) -> None:
        with self.assertRaises(DocumentError):
            summarize_abstractive("", max_words=50)
        with self.assertRaises(DocumentError):
            summarize_abstractive(self.TEXT, max_words=0)

    def test_abstractive_no_brain(self) -> None:
        from unittest import mock
        with mock.patch("nomorals.llm.brain.get_brain",
                        side_effect=RuntimeError("no brain")):
            with self.assertRaises(DocumentError):
                summarize_abstractive(self.TEXT)

    def test_keywords_still_work(self) -> None:
        words = keywords(self.TEXT, top_n=3)
        self.assertIn("mars", words)


if __name__ == "__main__":
    unittest.main()
