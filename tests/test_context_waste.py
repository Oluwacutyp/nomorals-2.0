"""Tests for nomorals.context.waste: WasteDetector reports."""

from __future__ import annotations

import unittest

from nomorals.context import (
    BuiltContext,
    ContextBudget,
    Section,
    WasteDetector,
)


def _built_with(sections: list[Section]) -> BuiltContext:
    text = "\n\n".join(s.render() for s in sections)
    return BuiltContext(text=text, sections=sections,
                        total_tokens=len(text.split()))


def _history(lines: list[str]) -> Section:
    return Section(name="history", content="\n".join(lines), priority=20.0)


class DuplicateOutputTest(unittest.TestCase):
    def test_flags_duplicated_tool_output(self) -> None:
        blob = "- [tool:web_search] " + "result payload alpha beta gamma " * 12
        section = _history([blob, "- [user] hello", blob])
        report = WasteDetector().analyze(_built_with([section]))
        dupes = [f for f in report.findings if f.kind == "duplicate_output"]
        self.assertTrue(dupes, "expected a duplicate_output finding")
        finding = dupes[0]
        self.assertEqual(finding.section, "history")
        self.assertGreater(finding.wasted_tokens, 0)
        # Sane estimate: reclaimable never exceeds wasted.
        self.assertLessEqual(finding.reclaimable_tokens, finding.wasted_tokens)
        self.assertGreater(finding.reclaimable_tokens, 0)

    def test_no_finding_without_duplicates(self) -> None:
        section = _history(["- [user] one thing", "- [assistant] another"])
        report = WasteDetector().analyze(_built_with([section]))
        self.assertFalse(
            [f for f in report.findings if f.kind == "duplicate_output"])


class StaleHistoryTest(unittest.TestCase):
    def test_flags_stale_history_beyond_window(self) -> None:
        lines = [
            f"- [user] old exchange number {i} with plenty of filler words here"
            for i in range(30)
        ]
        lines += ["- [user] recent one", "- [assistant] recent two"]
        section = _history(lines)
        detector = WasteDetector(stale_history_window=5, min_stale_tokens=10)
        report = detector.analyze(_built_with([section]))
        stale = [f for f in report.findings if f.kind == "stale_history"]
        self.assertTrue(stale, "expected a stale_history finding")
        finding = stale[0]
        self.assertIn("27 of 32", finding.detail)
        self.assertGreater(finding.reclaimable_tokens, 0)
        self.assertLessEqual(finding.reclaimable_tokens, finding.wasted_tokens)

    def test_no_finding_within_window(self) -> None:
        lines = [f"- [user] message {i}" for i in range(5)]
        report = WasteDetector(stale_history_window=10).analyze(
            _built_with([_history(lines)]))
        self.assertFalse(
            [f for f in report.findings if f.kind == "stale_history"])


class OversizedSectionTest(unittest.TestCase):
    def test_flags_oversized_low_signal_section(self) -> None:
        section = Section(
            name="history",
            content=" ".join(f"filler{i}" for i in range(4000)),
            priority=20.0,
        )
        budget = ContextBudget(total=8000)
        report = WasteDetector().analyze(_built_with([section]), budget)
        big = [f for f in report.findings if f.kind == "oversized_section"]
        self.assertTrue(big, "expected an oversized_section finding")
        finding = big[0]
        self.assertLessEqual(finding.reclaimable_tokens, finding.wasted_tokens)
        self.assertGreater(finding.reclaimable_tokens, 0)

    def test_load_bearing_never_flagged_as_waste(self) -> None:
        section = Section(
            name="mission",
            content=" ".join(f"critical{i}" for i in range(4000)),
            priority=90.0,
            load_bearing=True,
        )
        report = WasteDetector().analyze(
            _built_with([section]), ContextBudget(total=8000))
        self.assertFalse(
            [f for f in report.findings if f.kind == "oversized_section"])


class ReportTotalsTest(unittest.TestCase):
    def test_totals_and_summary(self) -> None:
        blob = "- [tool:x] " + "some duplicated output tokens here " * 10
        section = _history([blob, blob, "- [user] hi"])
        report = WasteDetector().analyze(_built_with([section]))
        self.assertEqual(report.total_wasted,
                         sum(f.wasted_tokens for f in report.findings))
        self.assertEqual(report.total_reclaimable,
                         sum(f.reclaimable_tokens for f in report.findings))
        self.assertLessEqual(report.total_reclaimable, report.total_wasted)
        self.assertIn("reclaimable", report.summary())

    def test_clean_context_reports_no_waste(self) -> None:
        section = _history(["- [user] hi", "- [assistant] hello"])
        report = WasteDetector().analyze(_built_with([section]))
        self.assertEqual(report.findings, [])
        self.assertEqual(report.total_wasted, 0)
        self.assertIn("No context waste", report.summary())


if __name__ == "__main__":
    unittest.main()
