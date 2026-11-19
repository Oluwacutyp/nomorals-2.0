"""Lane 100 — WordCraft port: an editable word-processor document model
over real .docx bytes (OOXML, stdlib only), machine-checked parity TOML,
SHA-256 attribution DB, and the ``wordcraft`` brain tool.

Failing-first: every test here fails on the base (no word organ) and
passes only when the real engine exists. Fixtures are REAL .docx files
vendored under tests/fixtures/wordcraft/ — built by an independent
stdlib generator (never by the organ under test) plus one built by
python-docx as a real-world-producer cross-check. A missing fixture is
a failure, never a silent skip.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

from nomorals.documents import DocumentError
from nomorals.documents.word import (
    WordDocument,
    WordEditor,
    WordTable,
    load_attribution,
    load_parity,
    run_parity,
    verify_attribution,
)
from nomorals.tools.registry import ToolRegistry

FIXTURES = Path(__file__).parent / "fixtures" / "wordcraft"
REPO_ROOT = Path(__file__).parent.parent
PARITY_TOML = REPO_ROOT / "nomorals" / "documents" / "word-parity.toml"
ATTRIBUTION_TOML = REPO_ROOT / "nomorals" / "documents" \
    / "word-attribution.toml"

ALL_FIXTURES = (
    "simple.docx", "styled.docx", "table.docx", "lists.docx",
    "headers.docx", "image.docx", "empty.docx", "broken.docx",
    "no-document.docx", "corrupt-xml.docx", "python-docx.docx",
)


def _read(name: str) -> bytes:
    path = FIXTURES / name
    if not path.exists():
        raise AssertionError(f"fixture missing: {name} — never skip")
    return path.read_bytes()


class FixtureTests(unittest.TestCase):
    def test_all_fixtures_present(self) -> None:
        missing = [n for n in ALL_FIXTURES if not (FIXTURES / n).exists()]
        self.assertEqual(missing, [], f"missing fixtures: {missing}")

    def test_valid_fixtures_are_real_docx_packages(self) -> None:
        for name in ("simple.docx", "styled.docx", "table.docx",
                     "lists.docx", "headers.docx", "image.docx",
                     "empty.docx", "python-docx.docx"):
            with zipfile.ZipFile(io.BytesIO(_read(name))) as zf:
                self.assertIn("word/document.xml", zf.namelist(),
                              f"{name} has no word/document.xml")


class InspectTests(unittest.TestCase):
    def test_simple_document(self) -> None:
        doc = WordDocument.from_bytes(_read("simple.docx"))
        self.assertEqual(doc.core_properties.get("title"),
                         "Simple Fixture")
        self.assertEqual(doc.core_properties.get("author"), "fixture")
        self.assertEqual(len(doc.paragraphs), 4)
        self.assertEqual(doc.paragraphs[0].style_id, "Title")
        self.assertEqual(doc.paragraphs[0].text, "Simple Fixture")
        self.assertEqual(doc.paragraphs[1].text,
                         "First paragraph of the simple fixture.")
        self.assertEqual(doc.plain_text().splitlines()[0],
                         "Simple Fixture")
        self.assertEqual(doc.tables, [])

    def test_styled_runs(self) -> None:
        doc = WordDocument.from_bytes(_read("styled.docx"))
        self.assertEqual(doc.paragraphs[1].style_id, "Heading1")
        self.assertEqual(doc.paragraphs[1].heading_level, 1)
        runs = doc.paragraphs[2].runs
        self.assertEqual([r.text for r in runs],
                         ["Bold lead ", "and italic follow",
                          " and plain tail."])
        self.assertTrue(runs[0].bold)
        self.assertFalse(runs[0].italic)
        self.assertTrue(runs[1].italic)
        self.assertFalse(runs[1].bold)
        self.assertEqual(doc.paragraphs[3].runs[0].color, "C00000")
        self.assertEqual(doc.paragraphs[3].runs[0].size_half_points, 28)
        self.assertTrue(doc.paragraphs[4].runs[0].underline)
        self.assertEqual(doc.paragraphs[5].style_id, "Heading2")
        self.assertEqual(doc.paragraphs[5].heading_level, 2)
        outline = doc.outline()
        self.assertEqual(
            [(o["level"], o["text"]) for o in outline],
            [(1, "Chapter One"), (2, "A Subsection")])

    def test_tables(self) -> None:
        doc = WordDocument.from_bytes(_read("table.docx"))
        self.assertEqual(len(doc.tables), 2)
        first = doc.tables[0]
        self.assertIsInstance(first, WordTable)
        self.assertEqual(first.cell_text(0, 0), "Name")
        self.assertEqual(first.cell_text(1, 1), "92")
        self.assertEqual(first.rows_text(),
                         [["Name", "Score"], ["Ada", "92"],
                          ["Grace", "88"]])
        self.assertEqual(doc.tables[1].rows_text(),
                         [["City", "Country"], ["Ikere", "Nigeria"]])
        # tables sit between paragraphs in body order
        kinds = [b["kind"] for b in doc.body_blocks()]
        self.assertEqual(kinds,
                         ["paragraph", "paragraph", "table",
                          "paragraph", "table", "paragraph"])

    def test_lists(self) -> None:
        doc = WordDocument.from_bytes(_read("lists.docx"))
        lists = [p for p in doc.paragraphs if p.numbering is not None]
        self.assertEqual(len(lists), 7)
        self.assertEqual([p.numbering["kind"] for p in lists[:4]],
                         ["bullet", "bullet", "bullet", "bullet"])
        self.assertEqual(lists[2].numbering["level"], 1)
        self.assertEqual([p.numbering["kind"] for p in lists[4:]],
                         ["decimal", "decimal", "decimal"])
        self.assertEqual(lists[0].text, "Apples")
        self.assertEqual(lists[4].text, "Preheat the oven")

    def test_headers_and_footers(self) -> None:
        doc = WordDocument.from_bytes(_read("headers.docx"))
        self.assertEqual(len(doc.headers), 1)
        self.assertEqual(len(doc.footers), 1)
        self.assertIn("Fixture header", doc.headers[0].plain_text())
        self.assertIn("Page", doc.footers[0].plain_text())
        self.assertEqual(len(doc.sections), 1)

    def test_image_inventory(self) -> None:
        doc = WordDocument.from_bytes(_read("image.docx"))
        self.assertEqual(len(doc.images), 1)
        self.assertEqual(doc.images[0]["part"], "word/media/image1.png")
        self.assertEqual(doc.images[0]["content_type"], "image/png")
        self.assertGreater(doc.images[0]["size"], 0)
        self.assertIn("red pixel", doc.plain_text())

    def test_python_docx_producer(self) -> None:
        doc = WordDocument.from_bytes(_read("python-docx.docx"))
        self.assertEqual(doc.core_properties.get("title"),
                         "Python-docx Fixture")
        self.assertTrue(any(p.style_id == "Heading1"
                            for p in doc.paragraphs))
        self.assertEqual(len(doc.tables), 1)
        self.assertEqual(doc.tables[0].cell_text(1, 0), "cell one")
        self.assertIn("Bold and", doc.plain_text())
        self.assertIn("italic text.", doc.plain_text())

    def test_empty_document(self) -> None:
        doc = WordDocument.from_bytes(_read("empty.docx"))
        self.assertEqual(doc.paragraphs, [])
        self.assertEqual(doc.tables, [])
        self.assertEqual(doc.plain_text(), "")

    def test_to_dict_round_trip(self) -> None:
        doc = WordDocument.from_bytes(_read("styled.docx"))
        data = doc.to_dict()
        self.assertEqual(data["paragraphs"][1]["style_id"], "Heading1")
        self.assertEqual(data["paragraphs"][2]["runs"][0]["bold"], True)
        self.assertIn("core_properties", data)
        self.assertIn("part_names", data)


class ErrorTests(unittest.TestCase):
    def test_broken_zip_refused(self) -> None:
        with self.assertRaises(DocumentError) as ctx:
            WordDocument.from_bytes(_read("broken.docx"))
        self.assertEqual(ctx.exception.kind, "parse")

    def test_missing_document_xml_refused(self) -> None:
        with self.assertRaises(DocumentError) as ctx:
            WordDocument.from_bytes(_read("no-document.docx"))
        self.assertEqual(ctx.exception.kind, "parse")

    def test_corrupt_xml_refused(self) -> None:
        with self.assertRaises(DocumentError) as ctx:
            WordDocument.from_bytes(_read("corrupt-xml.docx"))
        self.assertEqual(ctx.exception.kind, "parse")

    def test_empty_bytes_refused(self) -> None:
        with self.assertRaises(DocumentError) as ctx:
            WordDocument.from_bytes(b"")
        self.assertEqual(ctx.exception.kind, "parse")

    def test_non_docx_zip_refused(self) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("hello.txt", "hello")
        with self.assertRaises(DocumentError) as ctx:
            WordDocument.from_bytes(buf.getvalue())
        self.assertEqual(ctx.exception.kind, "parse")


class FidelityTests(unittest.TestCase):
    def test_no_edit_round_trip_preserves_every_part(self) -> None:
        for name in ("simple.docx", "styled.docx", "table.docx",
                     "lists.docx", "headers.docx", "image.docx",
                     "python-docx.docx"):
            original = _read(name)
            doc = WordDocument.from_bytes(original)
            saved = doc.to_bytes()
            report = doc.fidelity_report(original)
            self.assertEqual(report["parts_total"],
                             report["parts_byte_identical"],
                             f"{name}: unedited parts must be "
                             "byte-identical")
            self.assertEqual(report["parts_changed"], [])
            # the re-read document is the same document
            again = WordDocument.from_bytes(saved)
            self.assertEqual(again.plain_text(), doc.plain_text())

    def test_edit_preserves_untouched_parts_and_paragraphs(self) -> None:
        original = _read("styled.docx")
        editor = WordEditor.from_bytes(original)
        editor.replace_text("Body under the subsection.",
                            "Body under the subsection — edited.")
        saved = editor.save()
        report = editor.document.fidelity_report(original)
        # only document.xml + core props may change; media/styles/rels stay
        self.assertNotIn("word/styles.xml", report["parts_changed"])
        self.assertNotIn("word/media/image1.png",
                         report["parts_changed"])
        self.assertIn("word/document.xml", report["parts_changed"])
        self.assertEqual(report["untouched_paragraphs_changed"], 0)
        again = WordDocument.from_bytes(saved)
        self.assertIn("edited.", again.plain_text())
        self.assertIn("Bold lead", again.plain_text())


class EditTests(unittest.TestCase):
    def test_replace_text_all_and_first_only(self) -> None:
        editor = WordEditor.from_bytes(_read("simple.docx"))
        count = editor.replace_text("paragraph", "passage")
        self.assertEqual(count, 3)
        saved = editor.save()
        doc = WordDocument.from_bytes(saved)
        self.assertIn("First passage of the simple fixture.",
                      doc.plain_text())
        self.assertNotIn("paragraph", doc.plain_text())
        editor2 = WordEditor.from_bytes(_read("simple.docx"))
        self.assertEqual(editor2.replace_text("paragraph", "passage",
                                              first_only=True), 1)
        doc2 = WordDocument.from_bytes(editor2.save())
        self.assertEqual(doc2.plain_text().count("passage"), 1)
        self.assertEqual(doc2.plain_text().count("paragraph"), 2)

    def test_replace_missing_text_is_an_error(self) -> None:
        editor = WordEditor.from_bytes(_read("simple.docx"))
        with self.assertRaises(DocumentError) as ctx:
            editor.replace_text("no such phrase anywhere", "x")
        self.assertEqual(ctx.exception.kind, "invalid_argument")

    def test_restyle_run(self) -> None:
        editor = WordEditor.from_bytes(_read("styled.docx"))
        editor.set_run_format(2, 2, bold=True, italic=True,
                              color="006100", size_half_points=24)
        doc = WordDocument.from_bytes(editor.save())
        run = doc.paragraphs[2].runs[2]
        self.assertTrue(run.bold)
        self.assertTrue(run.italic)
        self.assertEqual(run.color, "006100")
        self.assertEqual(run.size_half_points, 24)
        # the other runs of the same paragraph are untouched
        self.assertTrue(doc.paragraphs[2].runs[0].bold)
        self.assertFalse(doc.paragraphs[2].runs[0].italic)

    def test_set_paragraph_style(self) -> None:
        editor = WordEditor.from_bytes(_read("simple.docx"))
        editor.set_paragraph_style(1, "Heading1")
        doc = WordDocument.from_bytes(editor.save())
        self.assertEqual(doc.paragraphs[1].style_id, "Heading1")
        self.assertEqual(doc.paragraphs[1].heading_level, 1)
        with self.assertRaises(DocumentError) as ctx:
            editor.set_paragraph_style(1, "No Such Style")
        self.assertEqual(ctx.exception.kind, "invalid_argument")

    def test_add_heading_and_paragraph(self) -> None:
        editor = WordEditor.from_bytes(_read("simple.docx"))
        editor.add_heading("New Chapter", level=1)
        editor.add_paragraph("A brand new paragraph.")
        doc = WordDocument.from_bytes(editor.save())
        self.assertEqual(len(doc.paragraphs), 6)
        self.assertEqual(doc.paragraphs[4].style_id, "Heading1")
        self.assertEqual(doc.paragraphs[4].text, "New Chapter")
        self.assertEqual(doc.paragraphs[5].text, "A brand new paragraph.")
        self.assertEqual(doc.outline()[-1]["text"], "New Chapter")

    def test_add_list_items(self) -> None:
        editor = WordEditor.from_bytes(_read("lists.docx"))
        editor.add_list_item("Oranges", ordered=False)
        editor.add_list_item("Serve warm", ordered=True)
        doc = WordDocument.from_bytes(editor.save())
        added = doc.paragraphs[-2:]
        self.assertEqual(added[0].text, "Oranges")
        self.assertEqual(added[0].numbering["kind"], "bullet")
        self.assertEqual(added[1].text, "Serve warm")
        self.assertEqual(added[1].numbering["kind"], "decimal")
        # the numbering part survives and still defines both formats
        self.assertIn("word/numbering.xml", doc.part_names)

    def test_add_list_to_document_without_numbering(self) -> None:
        editor = WordEditor.from_bytes(_read("simple.docx"))
        editor.add_list_item("First item", ordered=True)
        doc = WordDocument.from_bytes(editor.save())
        self.assertEqual(doc.paragraphs[-1].numbering["kind"], "decimal")
        self.assertIn("word/numbering.xml", doc.part_names)

    def test_add_table(self) -> None:
        editor = WordEditor.from_bytes(_read("simple.docx"))
        editor.add_table(["Fruit", "Price"], [["Apple", "1.20"],
                                              ["Mango", "0.80"]])
        doc = WordDocument.from_bytes(editor.save())
        self.assertEqual(len(doc.tables), 1)
        self.assertEqual(doc.tables[0].rows_text(),
                         [["Fruit", "Price"], ["Apple", "1.20"],
                          ["Mango", "0.80"]])
        self.assertEqual(doc.tables[0].cell_text(2, 0), "Mango")

    def test_set_metadata(self) -> None:
        editor = WordEditor.from_bytes(_read("simple.docx"))
        editor.set_metadata(title="Renamed Fixture", author="devon",
                            keywords="wordcraft, test")
        doc = WordDocument.from_bytes(editor.save())
        self.assertEqual(doc.core_properties.get("title"),
                         "Renamed Fixture")
        self.assertEqual(doc.core_properties.get("author"), "devon")
        self.assertEqual(doc.core_properties.get("keywords"),
                         "wordcraft, test")

    def test_remove_paragraph(self) -> None:
        editor = WordEditor.from_bytes(_read("simple.docx"))
        editor.remove_paragraph(2)
        doc = WordDocument.from_bytes(editor.save())
        self.assertEqual(len(doc.paragraphs), 3)
        self.assertNotIn("Second paragraph", doc.plain_text())
        with self.assertRaises(DocumentError):
            editor.remove_paragraph(99)

    def test_editor_summary(self) -> None:
        editor = WordEditor.from_bytes(_read("simple.docx"))
        editor.replace_text("First", "1st")
        editor.add_heading("Tail", level=2)
        summary = editor.summary()
        self.assertEqual(summary["paragraphs"], 5)
        self.assertEqual(summary["edits"], 2)
        self.assertIn("title", summary)


class CreateTests(unittest.TestCase):
    def test_create_blank_document(self) -> None:
        editor = WordEditor.create(title="Fresh Doc", author="devon")
        editor.add_heading("Hello", level=1)
        editor.add_paragraph("Written from scratch.")
        data = editor.save()
        doc = WordDocument.from_bytes(data)
        self.assertEqual(doc.core_properties.get("title"), "Fresh Doc")
        self.assertEqual(doc.paragraphs[0].text, "Hello")
        self.assertEqual(doc.paragraphs[0].style_id, "Heading1")
        self.assertEqual(doc.paragraphs[1].text,
                         "Written from scratch.")
        # a created document is a real package
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            self.assertIn("word/document.xml", zf.namelist())
            self.assertIn("word/styles.xml", zf.namelist())


class ParityTests(unittest.TestCase):
    def test_parity_toml_loads(self) -> None:
        data = load_parity(PARITY_TOML)
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(len(data["cases"]), 10)

    def test_parity_runs_green(self) -> None:
        result = run_parity(PARITY_TOML)
        self.assertEqual(result["failed"], 0, json.dumps(
            [c for c in result["cases"] if c["status"] == "fail"],
            indent=2))
        self.assertEqual(result["passed"], 10)

    def test_parity_missing_fixture_is_a_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.toml"
            bad.write_text(
                'schema_version = 1\n'
                '[[cases]]\n'
                'id = "ghost"\n'
                'fixture = "no-such-file.docx"\n'
                'operation = "inspect"\n'
                'expect = { paragraphs = 1 }\n')
            result = run_parity(bad)
            self.assertEqual(result["failed"], 1)
            self.assertEqual(result["cases"][0]["status"], "fail")


class AttributionTests(unittest.TestCase):
    def test_attribution_toml_loads(self) -> None:
        data = load_attribution(ATTRIBUTION_TOML)
        self.assertGreaterEqual(len(data["assets"]), 10)
        self.assertIn("upstream_commit", data)

    def test_attribution_verifies_against_clone(self) -> None:
        result = verify_attribution(ATTRIBUTION_TOML)
        self.assertEqual(result["failed"], 0, json.dumps(
            result, indent=2))
        self.assertGreater(result["checked"], 0)


class ToolTests(unittest.TestCase):
    def _registry(self) -> ToolRegistry:
        registry = ToolRegistry(context=SimpleNamespace(
            settings=SimpleNamespace(), home=Path(tempfile.mkdtemp())))
        registry.register_builtins()
        return registry

    def _call(self, registry: ToolRegistry, **kwargs) -> dict:
        outcome = registry.call("wordcraft", **kwargs)
        if hasattr(outcome, "value"):
            return outcome.value
        return outcome

    def _call_outcome(self, registry: ToolRegistry, **kwargs):
        return registry.call("wordcraft", **kwargs)

    def test_tool_is_registered(self) -> None:
        registry = self._registry()
        spec = registry.get("wordcraft")
        self.assertIsNotNone(spec)
        self.assertIn("inspect", spec.description)

    def test_tool_inspect(self) -> None:
        registry = self._registry()
        out = self._call(registry, action="inspect",
                         sources=[str(FIXTURES / "simple.docx")])
        self.assertEqual(len(out["paragraphs"]), 4)
        self.assertEqual(out["core_properties"]["title"],
                         "Simple Fixture")
        self.assertIn("reply", out)

    def test_tool_edit_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.docx"
            src.write_bytes(_read("simple.docx"))
            out_path = Path(tmp) / "out.docx"
            registry = self._registry()
            out = self._call(
                registry, action="replace", sources=[str(src)],
                output=str(out_path),
                text="First paragraph of the simple fixture.",
                new_text="Replaced through the tool.")
            self.assertTrue(out["ok"])
            self.assertEqual(out["replacements"], 1)
            doc = WordDocument.from_bytes(out_path.read_bytes())
            self.assertIn("Replaced through the tool.", doc.plain_text())

    def test_tool_parity_and_attribution(self) -> None:
        registry = self._registry()
        parity = self._call(registry, action="parity")
        self.assertEqual(parity["failed"], 0)
        attrib = self._call(registry, action="attribution")
        self.assertEqual(attrib["failed"], 0)
        self.assertIn("reply", parity)
        self.assertIn("reply", attrib)

    def test_tool_unknown_action(self) -> None:
        registry = self._registry()
        outcome = self._call_outcome(registry, action="nope")
        self.assertFalse(outcome.ok)
        self.assertIn("unknown", str(outcome.error).lower())


if __name__ == "__main__":
    unittest.main()
