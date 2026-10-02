"""Tests for nomorals.context.compress: summarize-then-truncate semantics."""

from __future__ import annotations

import unittest

from nomorals.context import (
    Section,
    compress_section,
    extractive_summary,
    summarize_then_truncate,
)


def _big_section(words: int = 2000, **kwargs) -> Section:
    kwargs.setdefault("name", "history")
    kwargs.setdefault("priority", 20.0)
    return Section(content=" ".join(f"word{i}" for i in range(words)), **kwargs)


class SummarizeThenTruncateTest(unittest.TestCase):
    def test_fits_untouched(self) -> None:
        text, truncated = summarize_then_truncate("short text", 1000)
        self.assertEqual(text, "short text")
        self.assertFalse(truncated)

    def test_truncation_is_explicit(self) -> None:
        text = " ".join(f"w{i}" for i in range(2000))
        out, truncated = summarize_then_truncate(text, 100, section_name="history")
        self.assertTrue(truncated)
        self.assertIn("truncated", out)
        self.assertIn("history", out)

    def test_keep_survives_extreme_compression(self) -> None:
        text = "ACCEPTANCE: the widget must frobnicate\n" + " ".join(
            f"filler{i}" for i in range(2000))
        out, truncated = summarize_then_truncate(
            text, 50, section_name="mission",
            keep=("ACCEPTANCE: the widget must frobnicate",))
        self.assertTrue(truncated)
        self.assertIn("ACCEPTANCE: the widget must frobnicate", out)
        self.assertIn("truncated", out)

    def test_keep_not_duplicated(self) -> None:
        keep = "PINNED LINE"
        text = f"{keep}\n" + " ".join(f"w{i}" for i in range(500))
        out, _ = summarize_then_truncate(text, 60, keep=(keep,))
        self.assertEqual(out.count(keep), 1)

    def test_no_marker_when_keep_fits(self) -> None:
        text = "PINNED LINE\nsome more words here"
        out, truncated = summarize_then_truncate(text, 1000, keep=("PINNED LINE",))
        self.assertFalse(truncated)
        self.assertNotIn("truncated", out)
        self.assertIn("PINNED LINE", out)

    def test_keep_tail_keeps_newest(self) -> None:
        lines = [f"- [user] message number {i}" for i in range(50)]
        text = "\n".join(lines)
        out, truncated = summarize_then_truncate(text, 40, keep_tail=True)
        self.assertTrue(truncated)
        self.assertIn("message number 49", out)
        self.assertNotIn("message number 0", out)

    def test_summarizer_called_on_huge_text(self) -> None:
        calls: list[int] = []

        def fake_summarizer(text: str, max_tokens: int) -> str:
            calls.append(max_tokens)
            return "SUMMARY"

        text = " ".join(f"w{i}" for i in range(5000))
        out, truncated = summarize_then_truncate(
            text, 100, summarizer=fake_summarizer)
        self.assertTrue(calls)
        self.assertTrue(truncated)
        self.assertIn("SUMMARY", out)


class ExtractiveSummaryTest(unittest.TestCase):
    def test_keeps_head_and_tail(self) -> None:
        paras = [f"paragraph {i} with some words in it" for i in range(10)]
        text = "\n\n".join(paras)
        out = extractive_summary(text, 30)
        self.assertIn("paragraph 0", out)
        self.assertIn("paragraph 9", out)
        self.assertIn("omitted", out)

    def test_short_text_unchanged(self) -> None:
        text = "one\n\ntwo"
        self.assertEqual(extractive_summary(text, 1000), text)


class CompressSectionTest(unittest.TestCase):
    def test_marks_truncated_explicitly(self) -> None:
        section = _big_section()
        compress_section(section, 100)
        self.assertTrue(section.truncated)
        self.assertIn("truncated", section.content)

    def test_load_bearing_keep_never_silently_dropped(self) -> None:
        criteria = "Acceptance criteria:\n- frobnicate on demand"
        section = Section(
            name="mission",
            content=criteria + "\n" + " ".join(f"w{i}" for i in range(3000)),
            priority=90.0,
            load_bearing=True,
            keep=(criteria,),
        )
        compress_section(section, 80)
        self.assertIn("frobnicate on demand", section.content)
        # Explicit, never silent:
        self.assertTrue(section.truncated)
        self.assertIn("truncated", section.content)

    def test_fitting_section_untouched(self) -> None:
        section = Section(name="system", content="be helpful", priority=100.0)
        compress_section(section, 1000)
        self.assertFalse(section.truncated)
        self.assertEqual(section.content, "be helpful")

    def test_custom_summarizer_used(self) -> None:
        section = _big_section()
        compress_section(section, 50,
                         summarizer=lambda t, m: "CONDENSED")
        self.assertIn("CONDENSED", section.content)


if __name__ == "__main__":
    unittest.main()
