"""Ruff ratchet (Echelon Wave 8) — the lint-debt gate.

Tier: unit. Pins the ratchet semantics on synthetic trees (record →
pass; new debt → fail; paid-down debt with an untightened record →
fail), the honest not-installed verdict, and — when the pinned ruff is
available — that the REAL tree's recorded ratchet matches its live
counts exactly.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nomorals.tools import ruff_ratchet

_RUFF = shutil.which("ruff")


def _tree(root: Path, files: dict[str, str]) -> None:
    for rel, body in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")


@unittest.skipUnless(_RUFF, "ruff not installed on PATH")
class TestRatchetSemantics(unittest.TestCase):
    CLEAN = {"pkg/__init__.py": "VALUE = 1\n"}
    DIRTY = {"pkg/__init__.py": "import os\n\nVALUE = 1\n"}  # F401

    def test_recorded_ratchet_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _tree(root, self.DIRTY)
            recorded = ruff_ratchet.update_ratchet(repo=root, paths=("pkg/",))
            self.assertTrue(recorded["ok"], recorded)
            self.assertEqual(recorded["by_code"].get("F401"), 1)
            verdict = ruff_ratchet.check_ratchet(repo=root, paths=("pkg/",))
            self.assertTrue(verdict["ok"], verdict)
            self.assertEqual(verdict["total"], verdict["recorded_total"])

    def test_new_debt_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _tree(root, self.CLEAN)
            ruff_ratchet.update_ratchet(repo=root, paths=("pkg/",))
            _tree(root, self.DIRTY)  # introduce an F401 after recording
            verdict = ruff_ratchet.check_ratchet(repo=root, paths=("pkg/",))
            self.assertFalse(verdict["ok"])
            flagged = {o["code"] for o in verdict["overruns"]} | {
                o["code"] for o in verdict["unrecorded"]}
            self.assertIn("F401", flagged)

    def test_paid_down_debt_requires_tightening(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _tree(root, self.DIRTY)
            ruff_ratchet.update_ratchet(repo=root, paths=("pkg/",))
            _tree(root, self.CLEAN)  # pay the debt down, record untouched
            verdict = ruff_ratchet.check_ratchet(repo=root, paths=("pkg/",))
            self.assertFalse(verdict["ok"])
            self.assertIn("F401",
                          {o["code"] for o in verdict["stale_ceilings"]})

    def test_missing_ratchet_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _tree(root, self.CLEAN)
            verdict = ruff_ratchet.check_ratchet(repo=root, paths=("pkg/",))
            self.assertFalse(verdict["ok"])
            self.assertIn("no ratchet recorded", verdict["detail"])


class TestRatchetHonesty(unittest.TestCase):
    def test_absent_ruff_is_never_a_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(ruff_ratchet, "ruff_available",
                                   return_value=False):
                verdict = ruff_ratchet.check_ratchet(repo=Path(tmp))
            self.assertFalse(verdict["ok"])
            self.assertFalse(verdict["ruff_installed"])


@unittest.skipUnless(_RUFF, "ruff not installed on PATH")
class TestRealTreeRatchet(unittest.TestCase):
    def test_real_tree_matches_its_recorded_ratchet(self) -> None:
        recorded = ruff_ratchet.load_ratchet()
        self.assertTrue(recorded, "ruff-ratchet.json must be recorded")
        if recorded.get("ruff_version") != ruff_ratchet.ruff_version():
            self.skipTest(
                f"ruff {ruff_ratchet.ruff_version()} != the ratchet's "
                f"recorded {recorded.get('ruff_version')} — counts are "
                "not comparable across ruff releases")
        verdict = ruff_ratchet.check_ratchet()
        self.assertTrue(verdict["ok"], verdict.get("detail"))


if __name__ == "__main__":
    unittest.main()
