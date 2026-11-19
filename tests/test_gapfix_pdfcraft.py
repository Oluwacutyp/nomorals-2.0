"""Lane 99 — PdfCraft port: spec-decomposed PDF object model, incremental
editing, machine-checked parity TOML, SHA-256 attribution DB.

Failing-first: every test here fails on the base (no pdfcraft organ) and
passes only when the real engine exists. Fixtures are REAL PDFs: Devon's
own writers (core.pdf render_pdf / render_pdf_embedded), hand-built
nested-tree files via the COS serializer, and corrupted/encrypted
variants built from them — a missing fixture is a failure, never a
silent skip.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.core.config import Settings
from nomorals.core.pdf import render_pdf, render_pdf_embedded
from nomorals.documents import DocumentError
from nomorals.documents.pagewise import protect_pdf
from nomorals.documents.pdfcraft import (
    PdfDocument,
    PdfEditor,
    PdfName,
    PdfRef,
    PdfStream,
    PdfString,
    load_attribution,
    load_parity,
    parse_object,
    parse_xref,
    run_parity,
    serialize,
    verify_attribution,
)
from nomorals.documents.pdfcraft.cos import decode_stream, parse_indirect
from nomorals.tools.registry import ToolRegistry

FIXTURES = Path(__file__).parent / "fixtures" / "pdfcraft"
PARITY_TOML = Path(__file__).parent.parent / "nomorals" / "documents" \
    / "pdfcraft-parity.toml"
ATTRIBUTION_TOML = Path(__file__).parent.parent / "nomorals" / "documents" \
    / "pdfcraft-attribution.toml"


def simple_pdf(pages: int = 3) -> bytes:
    # chapter_break=True starts every h1 on a fresh page — the only
    # deterministic multi-page shape the plain writer offers.
    body = "\n\n".join(
        f"# Page {n}\n\nAlpha beta gamma {n}."
        for n in range(1, pages + 1))
    return render_pdf(body, title="Simple Fixture",
                      headings=True, chapter_break=True,
                      metadata={"title": "Simple Fixture"})


def embedded_pdf() -> bytes:
    return render_pdf_embedded(
        "Embedded one.\n\nEmbedded two with caf\xe9.",
        title="Embedded Fixture")


class CosLexerTests(unittest.TestCase):
    def test_literal_string_escapes_and_nesting(self):
        value, _ = parse_object(b"(a\\(b\\)c\\n\\053)", 0)
        self.assertIsInstance(value, PdfString)
        self.assertEqual(bytes(value), b"a(b)c\n+")

    def test_hex_string(self):
        value, _ = parse_object(b"<48656C6C6F>", 0)
        self.assertIsInstance(value, PdfString)
        self.assertEqual(bytes(value), b"Hello")

    def test_name_hex_escape(self):
        value, _ = parse_object(b"/A#42C", 0)
        self.assertIsInstance(value, PdfName)
        self.assertEqual(str(value), "A\x42C")

    def test_numbers_bool_null(self):
        for raw, want in ((b"42", 42), (b"-3.5", -3.5), (b"true", True),
                          (b"false", False), (b"null", None)):
            value, _ = parse_object(raw, 0)
            self.assertEqual(value, want, raw)

    def test_array_dict_and_ref(self):
        value, _ = parse_object(
            b"<< /Type /Page /Kids [3 0 R 4 0 R] /Count 2 >>", 0)
        self.assertEqual(value["/Type"], PdfName("Page"))
        self.assertEqual(value["/Kids"],
                         [PdfRef(3, 0), PdfRef(4, 0)])
        self.assertEqual(value["/Count"], 2)

    def test_stream_object_parses(self):
        raw = (b"5 0 obj\n<< /Length 4 /Filter /FlateDecode >>\n"
               b"stream\nxxxx\nendstream\nendobj")
        ref, obj, _ = parse_indirect(raw, 0)
        self.assertEqual(ref, PdfRef(5, 0))
        self.assertIsInstance(obj, PdfStream)
        self.assertEqual(obj.dict["/Length"], 4)
        self.assertEqual(obj.data, b"xxxx")

    def test_serialize_round_trip(self):
        obj = {"/Type": PdfName("Catalog"),
               "/Pages": PdfRef(2, 0),
               "/Kids": [PdfRef(3, 0), PdfRef(4, 0)],
               "/Title": PdfString(b"Hi (there)"),
               "/Count": 2, "/Ratio": 1.5, "/Flag": True,
               "/Nil": None}
        data = serialize(obj)
        back, _ = parse_object(data, 0)
        self.assertEqual(back, obj)

    def test_stream_serialize_round_trip(self):
        stream = PdfStream({"/Length": 5}, b"hello")
        data = serialize(stream)
        value, _ = parse_object(data, 0)
        self.assertIsInstance(value, PdfStream)
        self.assertEqual(value.data, b"hello")
        self.assertEqual(value.dict["/Length"], 5)

    def test_flate_stream_decodes(self):
        import zlib
        raw = zlib.compress(b"stream content here")
        stream = PdfStream({"/Length": len(raw),
                            "/Filter": PdfName("FlateDecode")}, raw)
        self.assertEqual(decode_stream(stream), b"stream content here")

    def test_unknown_filter_refused_by_name(self):
        stream = PdfStream({"/Length": 4,
                            "/Filter": PdfName("JBIG2Decode")}, b"xxxx")
        with self.assertRaises(DocumentError) as ctx:
            decode_stream(stream)
        self.assertEqual(ctx.exception.kind, "unsupported_format")
        self.assertIn("JBIG2Decode", str(ctx.exception))


class CosXrefTests(unittest.TestCase):
    def test_classic_xref_and_trailer(self):
        data = simple_pdf()
        xref = parse_xref(data)
        self.assertFalse(xref.repaired)
        self.assertGreater(xref.startxref, 0)
        self.assertIn("/Root", xref.trailer)
        self.assertIn("/Size", xref.trailer)
        entry = xref.entries.get(1)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.kind, "n")

    def test_prev_chain_after_incremental_updates(self):
        editor = PdfEditor.from_bytes(simple_pdf())
        editor.set_info(title="Gen one")
        gen1 = editor.save_incremental()
        editor2 = PdfEditor.from_bytes(gen1)
        editor2.set_info(author="Gen two")
        gen2 = editor2.save_incremental()
        xref = parse_xref(gen2)
        self.assertEqual(xref.generations, 3)
        doc = PdfDocument.from_bytes(gen2)
        self.assertEqual(doc.info.get("Title"), "Gen one")
        self.assertEqual(doc.info.get("Author"), "Gen two")

    def test_broken_xref_repairs_by_scanning(self):
        data = bytearray(simple_pdf())
        # Corrupt every xref offset digit so the table is unusable.
        start = data.find(b"xref\n")
        end = data.find(b"trailer\n", start)
        for i in range(start, end):
            if 48 <= data[i] <= 57:
                data[i] = ord("0")
        xref = parse_xref(bytes(data))
        self.assertTrue(xref.repaired)
        doc = PdfDocument.from_bytes(bytes(data))
        self.assertEqual(doc.page_count, 3)

    def test_garbage_prefix_still_opens(self):
        data = b"JUNKJUNK\n" + simple_pdf()
        doc = PdfDocument.from_bytes(data)
        self.assertEqual(doc.page_count, 3)


class ModelTests(unittest.TestCase):
    def test_inventory_of_simple_pdf(self):
        doc = PdfDocument.from_bytes(simple_pdf())
        self.assertEqual(doc.page_count, 3)
        self.assertEqual(doc.version, "1.4")
        self.assertFalse(doc.encrypted)
        self.assertFalse(doc.repaired)
        self.assertEqual(doc.info.get("Title"), "Simple Fixture")
        page = doc.pages[0]
        self.assertEqual(page.number, 1)
        self.assertEqual(page.mediabox, (0, 0, 595, 842))
        self.assertEqual(page.rotate, 0)
        self.assertEqual(len(page.contents), 1)

    def test_fonts_inventory(self):
        doc = PdfDocument.from_bytes(simple_pdf())
        fonts = doc.fonts()
        self.assertTrue(fonts)
        bases = {f["base_font"] for f in fonts}
        self.assertIn("/Helvetica", bases)
        self.assertTrue(all(f["page"] in (1, 2, 3) for f in fonts))

    def test_images_inventory_empty_and_present(self):
        doc = PdfDocument.from_bytes(simple_pdf())
        self.assertEqual(doc.images(), [])
        # A PDF with an inline image XObject is inventoried, not decoded.
        raw = _pdf_with_image()
        doc2 = PdfDocument.from_bytes(raw)
        images = doc2.images()
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0]["width"], 2)
        self.assertEqual(images[0]["height"], 2)
        self.assertEqual(images[0]["filter"], "/DCTDecode")

    def test_nested_page_tree_order(self):
        data = _nested_tree_pdf()
        doc = PdfDocument.from_bytes(data)
        self.assertEqual(doc.page_count, 3)
        self.assertEqual([p.number for p in doc.pages], [1, 2, 3])
        self.assertEqual(doc.text(1).strip(), "nested one")
        self.assertEqual(doc.text(3).strip(), "nested three")

    def test_per_page_and_whole_text(self):
        doc = PdfDocument.from_bytes(simple_pdf())
        self.assertIn("Alpha beta gamma 2", doc.text(2))
        self.assertNotIn("Alpha beta gamma 2", doc.text(1))
        self.assertIn("Alpha beta gamma 3", doc.text())

    def test_annotations_inventory(self):
        editor = PdfEditor.from_bytes(simple_pdf())
        editor.add_text_annotation(2, (72, 700, 200, 720),
                                   "Review this paragraph.")
        out = editor.save_incremental()
        doc = PdfDocument.from_bytes(out)
        annots = doc.annotations()
        self.assertEqual(len(annots), 1)
        self.assertEqual(annots[0]["page"], 2)
        self.assertEqual(annots[0]["subtype"], "/Text")
        self.assertEqual(annots[0]["contents"], "Review this paragraph.")

    def test_encrypted_pdf_refused_honestly(self):
        locked = protect_pdf(simple_pdf(), "secret")
        with self.assertRaises(DocumentError) as ctx:
            PdfDocument.from_bytes(locked)
        self.assertEqual(ctx.exception.kind, "encrypted")

    def test_not_a_pdf(self):
        with self.assertRaises(DocumentError) as ctx:
            PdfDocument.from_bytes(b"hello, not a pdf at all")
        self.assertEqual(ctx.exception.kind, "unsupported_format")

    def test_to_dict_inventory_shape(self):
        doc = PdfDocument.from_bytes(simple_pdf())
        summary = doc.to_dict()
        self.assertEqual(summary["page_count"], 3)
        self.assertEqual(summary["version"], "1.4")
        self.assertIn("fonts", summary)
        self.assertIn("images", summary)
        self.assertIn("annotations", summary)


class EditTests(unittest.TestCase):
    def test_incremental_metadata_keeps_original_prefix(self):
        original = simple_pdf()
        editor = PdfEditor.from_bytes(original)
        editor.set_info(title="Retitled", author="Lane 99")
        out = editor.save_incremental()
        self.assertEqual(out[:len(original)], original)
        self.assertGreater(len(out), len(original))
        doc = PdfDocument.from_bytes(out)
        self.assertEqual(doc.info.get("Title"), "Retitled")
        self.assertEqual(doc.info.get("Author"), "Lane 99")
        self.assertEqual(doc.page_count, 3)
        self.assertIn("Alpha beta gamma 1", doc.text(1))

    def test_incremental_rotate(self):
        editor = PdfEditor.from_bytes(simple_pdf())
        editor.rotate_page(2, 90)
        out = editor.save_incremental()
        doc = PdfDocument.from_bytes(out)
        self.assertEqual(doc.pages[1].rotate, 90)
        self.assertEqual(doc.pages[0].rotate, 0)
        self.assertIn("Alpha beta gamma 2", doc.text(2))

    def test_rotate_rejects_non_quarter_turn(self):
        editor = PdfEditor.from_bytes(simple_pdf())
        with self.assertRaises(DocumentError) as ctx:
            editor.rotate_page(1, 45)
        self.assertEqual(ctx.exception.kind, "invalid_argument")

    def test_incremental_stamp_text_is_extractable(self):
        editor = PdfEditor.from_bytes(simple_pdf())
        editor.stamp_text(1, "DRAFT COPY")
        out = editor.save_incremental()
        doc = PdfDocument.from_bytes(out)
        self.assertIn("DRAFT COPY", doc.text(1))
        self.assertIn("Alpha beta gamma 1", doc.text(1))

    def test_incremental_stamp_keeps_original_prefix(self):
        original = simple_pdf()
        editor = PdfEditor.from_bytes(original)
        editor.stamp_text(3, "PAGE THREE MARK")
        out = editor.save_incremental()
        self.assertEqual(out[:len(original)], original)
        self.assertIn("PAGE THREE MARK", PdfDocument.from_bytes(out).text(3))

    def test_second_generation_stacks(self):
        editor = PdfEditor.from_bytes(simple_pdf())
        editor.set_info(title="One")
        gen1 = editor.save_incremental()
        editor2 = PdfEditor.from_bytes(gen1)
        editor2.rotate_page(1, 180)
        gen2 = editor2.save_incremental()
        self.assertEqual(gen2[:len(gen1)], gen1)
        doc = PdfDocument.from_bytes(gen2)
        self.assertEqual(doc.info.get("Title"), "One")
        self.assertEqual(doc.pages[0].rotate, 180)

    def test_full_save_after_repair(self):
        data = bytearray(simple_pdf())
        start = data.find(b"xref\n")
        end = data.find(b"trailer\n", start)
        for i in range(start, end):
            if 48 <= data[i] <= 57:
                data[i] = ord("0")
        editor = PdfEditor.from_bytes(bytes(data))
        self.assertTrue(editor.document.repaired)
        editor.set_info(title="Repaired and retitled")
        out = editor.save_full()
        doc = PdfDocument.from_bytes(out)
        self.assertFalse(doc.repaired)
        self.assertEqual(doc.info.get("Title"), "Repaired and retitled")
        self.assertEqual(doc.page_count, 3)

    def test_dirty_tracking_names_changed_objects(self):
        editor = PdfEditor.from_bytes(simple_pdf())
        editor.set_info(title="Dirty check")
        editor.rotate_page(1, 90)
        self.assertTrue(editor.dirty)
        summary = editor.summary()
        self.assertEqual(summary["changed_objects"], len(editor.dirty))
        self.assertIn("save_incremental", summary["methods"])
        self.assertIn("save_full", summary["methods"])


class ParityTests(unittest.TestCase):
    def test_parity_toml_loads_and_runs_green(self):
        data = load_parity(PARITY_TOML)
        self.assertEqual(data["schema"]["version"], 1)
        self.assertTrue(data["case"])
        result = run_parity(PARITY_TOML, fixtures_dir=FIXTURES)
        self.assertEqual(result["failed"], 0, json.dumps(result, indent=2))
        self.assertGreaterEqual(result["passed"], 6)
        self.assertTrue(all(c["status"] == "pass" for c in result["cases"]))

    def test_parity_fails_on_tampered_expectation(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.toml"
            data = load_parity(PARITY_TOML)
            data["case"][0]["expect"]["page_count"] = 99
            bad.write_text(_dump_toml(data), encoding="utf-8")
            result = run_parity(bad, fixtures_dir=FIXTURES)
            self.assertGreater(result["failed"], 0)
            failed = [c for c in result["cases"]
                      if c["status"] == "fail"]
            self.assertTrue(failed)
            self.assertIn("page_count", json.dumps(failed[0]))

    def test_parity_fails_when_fixture_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.toml"
            missing.write_text(
                "[schema]\nversion = 1\n\n"
                "[[case]]\nid = \"ghost\"\nfixture = \"no-such-file.pdf\"\n"
                "ops = [\"inspect\"]\n[case.expect]\npage_count = 1\n",
                encoding="utf-8")
            result = run_parity(missing, fixtures_dir=FIXTURES)
            self.assertEqual(result["failed"], 1)
            self.assertEqual(result["cases"][0]["status"], "fail")


class AttributionTests(unittest.TestCase):
    def test_attribution_loads_and_verifies(self):
        data = load_attribution(ATTRIBUTION_TOML)
        self.assertTrue(data["asset"])
        result = verify_attribution(
            ATTRIBUTION_TOML,
            clone_root=Path(__file__).parent.parent.parent
            / "echelon-mining" / "repos" / "storytold" / "pdfcraft")
        self.assertEqual(result["failed"], 0, json.dumps(result, indent=2))
        self.assertGreater(result["checked"], 0)
        self.assertTrue(all(e["status"] == "match"
                            for e in result["entries"]
                            if e["kind"] == "upstream"))

    def test_attribution_fails_on_tampered_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.toml"
            data = load_attribution(ATTRIBUTION_TOML)
            data["asset"][0]["sha256"] = "0" * 64
            bad.write_text(_dump_toml(data), encoding="utf-8")
            result = verify_attribution(
                bad,
                clone_root=Path(__file__).parent.parent.parent
                / "echelon-mining" / "repos" / "storytold" / "pdfcraft")
            self.assertGreater(result["failed"], 0)

    def test_every_devon_module_names_upstream_units(self):
        data = load_attribution(ATTRIBUTION_TOML)
        devon = [a for a in data["asset"] if a.get("kind") == "devon"]
        self.assertTrue(devon)
        for entry in devon:
            self.assertTrue(entry.get("derives_from"),
                            entry.get("path"))


class ToolTests(unittest.TestCase):
    def _registry(self) -> ToolRegistry:
        registry = ToolRegistry(context=SimpleNamespace(
            settings=Settings()))
        registry.register_builtins()
        return registry

    def _call(self, registry: ToolRegistry, **kwargs) -> dict:
        outcome = registry.call("pagewise", **kwargs)
        self.assertTrue(outcome.ok, outcome)
        value = outcome.unwrap()
        self.assertIsInstance(value, dict)
        return value

    def test_pagewise_tool_has_pdfcraft_actions(self):
        registry = self._registry()
        spec = registry.get("pagewise")
        self.assertIsNotNone(spec)
        for action in ("inspect", "annotate", "metadata",
                       "parity", "attribution"):
            self.assertIn(action, spec.description)

    def test_tool_inspect(self):
        registry = self._registry()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.pdf"
            src.write_bytes(simple_pdf())
            out = self._call(registry, action="inspect",
                             sources=[str(src)])
        self.assertEqual(out["page_count"], 3)
        self.assertEqual(out["version"], "1.4")
        self.assertIn("reply", out)

    def test_tool_metadata_is_incremental(self):
        registry = self._registry()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.pdf"
            original = simple_pdf()
            src.write_bytes(original)
            dst = Path(tmp) / "out.pdf"
            out = self._call(registry, action="metadata",
                             sources=[str(src)], output=str(dst),
                             title="Tool Retitle")
            written = dst.read_bytes()
            self.assertEqual(written[:len(original)], original)
            self.assertEqual(out["title"], "Tool Retitle")

    def test_tool_annotate(self):
        registry = self._registry()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.pdf"
            src.write_bytes(simple_pdf())
            dst = Path(tmp) / "out.pdf"
            out = self._call(
                registry, action="annotate", sources=[str(src)],
                output=str(dst), text="Tool note here.", page=2)
            self.assertEqual(out["annotations"], 1)
            doc = PdfDocument.from_bytes(dst.read_bytes())
            self.assertEqual(doc.annotations()[0]["contents"],
                             "Tool note here.")

    def test_tool_parity_and_attribution(self):
        registry = self._registry()
        parity = self._call(registry, action="parity")
        self.assertEqual(parity["failed"], 0)
        self.assertGreater(parity["passed"], 0)
        attrib = self._call(registry, action="attribution")
        self.assertEqual(attrib["failed"], 0)


def _pdf_with_image() -> bytes:
    """A minimal PDF whose first page draws one DCTDecode XObject."""
    objs: dict[int, bytes] = {}
    objs[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objs[2] = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"
    objs[3] = (b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
               b"/Resources << /XObject << /Im1 5 0 R >> >> "
               b"/Contents 4 0 R >>")
    objs[4] = (b"<< /Length 12 >>\nstream\nq 200 0 0 200 0 0 cm /Im1 Do Q\n"
               b"endstream")
    objs[5] = (b"<< /Type /XObject /Subtype /Image /Width 2 /Height 2 "
               b"/ColorSpace /DeviceRGB /BitsPerComponent 8 "
               b"/Filter /DCTDecode /Length 4 >>\nstream\n\xff\xd8\xff\xd9\n"
               b"endstream")
    return _assemble(objs, root=1)


def _nested_tree_pdf() -> bytes:
    """Catalog -> Pages(2) -> [intermediate Pages(5) -> [6, 7], direct 8]."""
    objs: dict[int, bytes] = {}
    objs[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objs[2] = b"<< /Type /Pages /Kids [5 0 R 8 0 R] /Count 3 >>"
    objs[5] = b"<< /Type /Pages /Parent 2 0 R /Kids [6 0 R 7 0 R] /Count 2 >>"
    for num, word in ((6, "nested one"), (7, "nested two"),
                      (8, "nested three")):
        stream = (f"BT /F1 10 Tf 1 0 0 1 72 700 Tm ({word}) Tj ET"
                  ).encode("latin-1")
        objs[num] = (f"<< /Type /Page /Parent 5 0 R "
                     f"/MediaBox [0 0 595 842] "
                     f"/Resources << /Font << /F1 9 0 R >> >> "
                     f"/Contents {num + 10} 0 R >>").encode("latin-1")
        objs[num + 10] = (f"<< /Length {len(stream)} >>\nstream\n".encode()
                          + stream + b"\nendstream")
    objs[9] = (b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
               b"/Encoding /WinAnsiEncoding >>")
    return _assemble(objs, root=1)


def _assemble(objs: dict[int, bytes], *, root: int) -> bytes:
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: dict[int, int] = {}
    for obj_id in sorted(objs):
        offsets[obj_id] = len(out)
        out += f"{obj_id} 0 obj\n".encode()
        out += objs[obj_id]
        out += b"\nendobj\n"
    xref_pos = len(out)
    max_id = max(offsets)
    out += f"xref\n0 {max_id + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for obj_id in range(1, max_id + 1):
        if obj_id in offsets:
            out += f"{offsets[obj_id]:010d} 00000 n \n".encode()
        else:
            out += b"0000000000 65535 f \n"
    out += (f"trailer\n<< /Size {max_id + 1} /Root {root} 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n").encode()
    return bytes(out)


def _dump_toml(data: dict) -> str:
    """Minimal TOML writer for the test's mutated copies (stdlib has no
    TOML dumper; the files under test are small and regular)."""
    lines: list[str] = []

    def scalar(value) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, list):
            return "[" + ", ".join(
                f'"{v}"' if isinstance(v, str) else str(v)
                for v in value) + "]"
        return f'"{value}"'

    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"[{key}]")
            for k2, v2 in value.items():
                lines.append(f"{k2} = {scalar(v2)}")
        elif isinstance(value, list) and value \
                and isinstance(value[0], dict):
            for item in value:
                lines.append(f"[[{key}]]")
                for k2, v2 in item.items():
                    if isinstance(v2, dict):
                        lines.append(f"[{key}.{k2}]")
                        for k3, v3 in v2.items():
                            lines.append(f"{k3} = {scalar(v3)}")
                    else:
                        lines.append(f"{k2} = {scalar(v2)}")
        else:
            lines.append(f"{key} = {scalar(value)}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    unittest.main()
