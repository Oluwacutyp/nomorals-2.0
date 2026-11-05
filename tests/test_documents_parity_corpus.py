"""Wave 2 corpus gates: the measured-fidelity numbers in
``documents/PARITY.md`` are real, fresh, and never silently drop.

  * xlsx / pptx / embedded-PDF round-trips recover 100% of the
    fixture's values on this machine (DejaVu present).
  * the WinAnsi PDF measurably loses exactly the Greek/Cyrillic
    tokens the embedded writer keeps — the contrast is the point.
  * the committed PARITY.md is byte-identical to a fresh render.
  * OCR rows stay gated (named) wherever the Tesseract stack is
    absent — no fabricated coverage.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from nomorals.documents import corpus


class TestCorpusMeasurements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = corpus.measure_all()
        cls.by_format = {row["format"]: row for row in cls.m["rows"]}

    def test_office_writers_full_coverage(self):
        self.assertEqual(self.by_format["xlsx"]["coverage"], 100.0)
        self.assertEqual(self.by_format["pptx"]["coverage"], 100.0)

    def test_prose_formats_high_coverage(self):
        # named losses only: markdown table names, docx bullet marks
        self.assertGreaterEqual(self.by_format["markdown"]["coverage"],
                                95.0)
        if "coverage" in self.by_format["docx"]:
            self.assertGreaterEqual(self.by_format["docx"]["coverage"],
                                    95.0)
        self.assertEqual(self.by_format["epub"]["coverage"], 100.0)

    def test_embedded_pdf_full_coverage_where_font_exists(self):
        row = self.by_format["pdf (embedded font)"]
        if corpus.find_embedded_font() is None:
            self.skipTest("no subsettable font on this machine")
        self.assertEqual(row["coverage"], 100.0)
        self.assertLess(row["subset_glyphs"], row["font_glyphs"])

    def test_winansi_gap_is_measured_not_hidden(self):
        winansi = self.by_format["pdf (WinAnsi writer)"]
        embedded = self.by_format["pdf (embedded font)"]
        if "coverage" not in embedded:
            self.skipTest("no subsettable font on this machine")
        self.assertLess(winansi["coverage"], embedded["coverage"])
        # the characters the legacy encoder cannot represent are the
        # Greek/Cyrillic letters of the fixture — counted, not waved
        self.assertGreaterEqual(
            len(self.m["winansi_unrepresentable"]), 10)
        self.assertIn("Α", self.m["winansi_unrepresentable"])
        self.assertNotIn("é", self.m["winansi_unrepresentable"])

    def test_typed_cell_census(self):
        census = self.m["typed_cells"]
        if not census:
            self.skipTest("openpyxl not installed")
        self.assertGreater(census.get("n", 0), 0)
        self.assertGreater(census.get("d", 0), 0)
        self.assertGreater(census.get("b", 0), 0)
        self.assertGreater(census.get("s", 0), 0)

    def test_ocr_gate_is_named_never_faked(self):
        gate = self.m["ocr_gate"]
        self.assertIn("stack_available", gate)
        rendered = corpus.render_parity_md()
        if not gate["stack_available"]:
            self.assertIn("gated — Tesseract stack absent", rendered)
        else:  # pragma: no cover - depends on the machine
            self.assertIn("available here", rendered)


class TestParityDocFreshness(unittest.TestCase):
    def test_committed_parity_matches_fresh_render(self):
        committed = (Path(corpus.__file__).with_name("PARITY.md")
                     .read_text(encoding="utf-8"))
        fresh = corpus.render_parity_md()
        self.assertEqual(
            committed, fresh,
            "documents/PARITY.md is stale — regenerate with "
            "`python -m nomorals.documents.corpus`")

    def test_parity_states_present(self):
        text = (Path(corpus.__file__).with_name("PARITY.md")
                .read_text(encoding="utf-8"))
        for needle in ("shipped (Wave 2)", "gated", "open remainder",
                       "What does not convert",
                       "verified in tests"):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
