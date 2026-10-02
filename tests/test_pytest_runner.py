"""Unit tests for the pytest-runner output parsing.

Tier: unit. No subprocesses, no network — pure parsing.
"""

from __future__ import annotations

import unittest

from nomorals.tools.pytest_runner import _parse_pytest_counts


class ParsePytestCountsTests(unittest.TestCase):
    def test_all_passed(self) -> None:
        counts = _parse_pytest_counts("..  [100%]\n2 passed in 0.01s\n")
        self.assertEqual(counts.get("passed"), 2)
        self.assertEqual(counts.get("failed", 0), 0)

    def test_mixed_summary(self) -> None:
        counts = _parse_pytest_counts("1 failed, 2 passed in 0.03s\n")
        self.assertEqual(counts.get("failed"), 1)
        self.assertEqual(counts.get("passed"), 2)

    def test_error_summary(self) -> None:
        counts = _parse_pytest_counts("1 error in 0.02s\n")
        self.assertEqual(counts.get("error"), 1)

    def test_empty_output(self) -> None:
        self.assertEqual(_parse_pytest_counts(""), {})

    def test_counts_are_keyed_by_word_not_number(self) -> None:
        # Regression: dict(findall(...)) keyed by the number, so every
        # lookup missed and passed/failed/errors were always 0.
        counts = _parse_pytest_counts("12 passed in 0.10s\n")
        self.assertEqual(counts.get("passed"), 12)
        self.assertNotIn("12", counts)


if __name__ == "__main__":
    unittest.main()
