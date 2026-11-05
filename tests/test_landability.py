"""Landability scoring (Echelon Wave 8).

Tier: unit. Demonstrates the score on known-good and known-bad
synthetic diffs: a tested change lands at 100; frozen-file and
persona-file touches and secret-shaped additions are hard blocks; a
testless diff drops to needs_work; silent-failure shapes deduct; and
annotations are echoed but NEVER change score or status.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from nomorals.tools import landability

_GIT = shutil.which("git")

# Full-format token shapes, assembled from parts (never literals).
_GH_TOKEN = "ghp_" + "A" * 36


def _diff(path: str, added: list[str], removed: list[str] | None = None) -> str:
    lines = [f"diff --git a/{path} b/{path}",
             "--- a/" + path, "+++ b/" + path,
             "@@ -1,1 +1,1 @@"]
    lines += ["-" + line for line in (removed or [])]
    lines += ["+" + line for line in added]
    return "\n".join(lines) + "\n"


GOOD_DIFF = (
    _diff("nomorals/tools/example.py", ["def helper() -> int:", "    return 1"])
    + _diff("tests/test_example.py", ["def test_helper() -> None:", "    pass"])
)

BAD_SECRET_DIFF = _diff("nomorals/tools/example.py",
                        [f'TOKEN = "{_GH_TOKEN}"'])


class TestLandabilityScores(unittest.TestCase):
    def test_known_good_diff_is_landable_at_100(self) -> None:
        result = landability.score_patch(GOOD_DIFF)
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "landable")
        self.assertEqual(result["score"], 100)
        self.assertEqual(result["hard_blocks"], [])
        self.assertTrue(result["has_tests"])

    def test_frozen_migrations_touch_is_blocked(self) -> None:
        diff = _diff("nomorals/storage/migrations.py", ["x = 1"])
        result = landability.score_patch(diff)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["hard_blocks"][0]["kind"], "frozen_file")

    def test_persona_touch_is_blocked(self) -> None:
        diff = _diff("nomorals/memory/persona.py", ["x = 1"])
        result = landability.score_patch(diff)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["hard_blocks"][0]["kind"], "frozen_file")

    def test_secret_in_diff_is_blocked_and_redacted(self) -> None:
        result = landability.score_patch(BAD_SECRET_DIFF)
        self.assertEqual(result["status"], "blocked")
        kinds = {b["kind"] for b in result["hard_blocks"]}
        self.assertIn("secret_in_diff", kinds)
        self.assertNotIn(_GH_TOKEN, json.dumps(result),
                         "reports carry the redacted form only")

    def test_no_tests_drops_to_needs_work(self) -> None:
        diff = _diff("nomorals/tools/example.py", ["x = 1"])
        result = landability.score_patch(diff)
        self.assertEqual(result["status"], "needs_work")
        self.assertEqual(result["score"], 75)
        self.assertIn("no_tests",
                      {d["kind"] for d in result["deductions"]})

    def test_silent_failure_shapes_deduct(self) -> None:
        diff = _diff("nomorals/tools/example.py",
                     ["try:", "    work()", "except:", "    pass"])
        result = landability.score_patch(diff)
        self.assertGreaterEqual(result["silent_failure_shapes"], 1)
        self.assertIn("silent_failure_shape",
                      {d["kind"] for d in result["deductions"]})

    def test_oversized_diff_deducts_but_does_not_block(self) -> None:
        big = _diff("tests/test_big.py", [f"# line {i}" for i in range(900)])
        result = landability.score_patch(big)
        self.assertIn("oversized_diff",
                      {d["kind"] for d in result["deductions"]})
        self.assertEqual(result["score"], 90)
        self.assertEqual(result["status"], "landable")

    def test_annotations_never_change_the_score(self) -> None:
        plain = landability.score_patch(GOOD_DIFF)
        noted = landability.score_patch(
            GOOD_DIFF, annotations=["reviewer: ship it", "lgtm"])
        self.assertEqual(plain["score"], noted["score"])
        self.assertEqual(plain["status"], noted["status"])
        self.assertEqual(noted["annotations"], ["reviewer: ship it", "lgtm"])
        blocked_noted = landability.score_patch(
            BAD_SECRET_DIFF, annotations=["trust me"])
        self.assertEqual(blocked_noted["status"], "blocked")


@unittest.skipUnless(_GIT, "git not installed")
class TestLandabilityRanges(unittest.TestCase):
    def test_scores_a_real_range_in_a_synthetic_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def git(*args: str) -> str:
                proc = subprocess.run(
                    ["git", "-c", "user.email=t@t", "-c", "user.name=t",
                     "-C", str(root), *args],
                    capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                return proc.stdout.strip()

            git("init", "-q")
            (root / "mod.py").write_text("x = 1\n", encoding="utf-8")
            git("add", "-A")
            git("commit", "-qm", "base")
            base = git("rev-parse", "HEAD")
            (root / "mod.py").write_text("x = 2\n", encoding="utf-8")
            (root / "tests").mkdir()
            (root / "tests" / "test_mod.py").write_text(
                "def test_x():\n    pass\n", encoding="utf-8")
            git("add", "-A")
            git("commit", "-qm", "change")
            result = landability.score_range(repo=root, base=base,
                                             head="HEAD")
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["status"], "landable")
            self.assertTrue(result["has_tests"])

    def test_bad_range_is_an_honest_error_not_a_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = landability.score_range(repo=Path(tmp),
                                              base="nope", head="alsono")
            self.assertFalse(result["ok"])


if __name__ == "__main__":
    unittest.main()
