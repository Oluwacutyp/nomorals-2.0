"""Docs-consistency guard for README.md's headline numbers.

Runs in <5s, no network. Fails loudly on drift so the README cannot go stale
silently: the claimed test/module/line counts must each be within 15% of the
values measured directly from the tree, and the README must not contain stale
absolute developer-machine paths.
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
PKG = ROOT / "nomorals"
TESTS_DIR = ROOT / "tests"

TOLERANCE = 0.15  # claimed vs measured


def _read_readme() -> str:
    assert README.is_file(), f"README.md missing at {README}"
    return README.read_text(encoding="utf-8")


def _measure_modules():
    return [p for p in PKG.rglob("*.py") if p.is_file()]


def _measure_lines(modules) -> int:
    total = 0
    for p in modules:
        total += len(p.read_text(encoding="utf-8", errors="ignore").splitlines())
    return total


_TEST_DEF = re.compile(r"^\s*def\s+test_", re.MULTILINE)


def _measure_tests() -> int:
    total = 0
    for p in sorted(TESTS_DIR.glob("test_*.py")):
        total += len(_TEST_DEF.findall(p.read_text(encoding="utf-8", errors="ignore")))
    return total


def _parse_int(text: str, pattern: str, label: str) -> int:
    m = re.search(pattern, text)
    assert m is not None, f"README.md does not contain a parseable {label} claim"
    return int(m.group(1).replace(",", ""))


class TestDocsConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.readme = _read_readme()
        cls.modules = _measure_modules()
        cls.lines = _measure_lines(cls.modules)
        cls.tests = _measure_tests()

    def _check_within(self, label: str, pattern: str, measured: int) -> None:
        claimed = _parse_int(self.readme, pattern, label)
        ratio = measured / claimed if claimed else float("inf")
        self.assertLessEqual(
            abs(1.0 - ratio),
            TOLERANCE,
            f"{label}: README claims ~{claimed:,} but measured {measured:,} "
            f"(drift {abs(1.0 - ratio) * 100:.1f}% > {TOLERANCE * 100:.0f}%)",
        )

    def test_claimed_test_count_matches_tree(self) -> None:
        self._check_within("test count", r"~([\d,]+)\s+tests", self.tests)

    def test_claimed_line_count_matches_tree(self) -> None:
        self._check_within("line count", r"([\d,]+)\s+lines of Python", self.lines)

    def test_claimed_module_count_matches_tree(self) -> None:
        self._check_within("module count", r"(\d+)\+\s+modules", len(self.modules))

    def test_no_stale_absolute_paths(self) -> None:
        for stale in ("/home/hatch",):
            self.assertNotIn(
                stale,
                self.readme,
                f"README.md contains stale absolute path {stale!r}",
            )


if __name__ == "__main__":
    unittest.main()
