"""Tests for the edit_loop.py surgical tooling.

Covers: atomic multi-hunk apply (surgical_replace_many), dry-run/preview,
post-edit Python syntax check with auto-revert, conflict detection
(overlap / invalidation), format-preservation verification, the pure-Python
unified-diff applier, and the apply_patch registry tool's `patch`-binary
fallback.

All tests are offline and hermetic (tmp dirs only).
"""

from __future__ import annotations

import asyncio
import random
import shutil
import subprocess
import tempfile
import types
import unittest
from difflib import unified_diff
from pathlib import Path
from unittest import mock

from nomorals.core.diff import apply_unified_diff as core_apply_unified_diff
from nomorals.tools import edit_loop
from nomorals.tools.edit_loop import (
    DiffApplyError,
    EditConflictError,
    EditLoop,
    EditPlan,
    EditSyntaxError,
    apply_unified_diff,
    verify_format_preserved,
    write_text_verified,
)
from nomorals.tools.registry import ToolRegistry


PY_SAMPLE = "x = 1\ny = 2\nz = 3\n"


def make_loop(tmp: Path, **kw) -> EditLoop:
    kw.setdefault("auto_backup", False)
    return EditLoop(agent=None, project_root=str(tmp), **kw)  # type: ignore[arg-type]


def write(tmp: Path, name: str, content: str) -> Path:
    p = tmp / name
    p.write_text(content, encoding="utf-8")
    return p


def make_registry(tmp: Path) -> ToolRegistry:
    ctx = types.SimpleNamespace(
        settings=types.SimpleNamespace(workspace_dir=str(tmp))
    )
    reg = ToolRegistry(context=ctx)
    edit_loop.register(reg)
    return reg


class TestSurgicalReplaceMany(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="editloop_"))
        self.loop = make_loop(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_atomic_success(self):
        write(self.tmp, "a.py", PY_SAMPLE)
        diff = self.loop.surgical_replace_many(
            "a.py", [("x = 1", "x = 10"), ("z = 3", "z = 30")]
        )
        self.assertEqual((self.tmp / "a.py").read_text(), "x = 10\ny = 2\nz = 30\n")
        self.assertIn("-x = 1", diff)
        self.assertIn("+x = 10", diff)
        self.assertIn("-z = 3", diff)
        self.assertIn("+z = 30", diff)

    def test_atomicity_second_hunk_missing(self):
        write(self.tmp, "a.py", PY_SAMPLE)
        with self.assertRaises(EditConflictError):
            self.loop.surgical_replace_many(
                "a.py", [("x = 1", "x = 10"), ("nope = 0", "nope = 1")]
            )
        # all-or-nothing: the good first hunk must NOT have been applied
        self.assertEqual((self.tmp / "a.py").read_text(), PY_SAMPLE)

    def test_atomicity_second_hunk_ambiguous(self):
        write(self.tmp, "a.txt", "k = 1\nk = 1\nother\n")
        with self.assertRaises(EditConflictError) as cm:
            self.loop.surgical_replace_many(
                "a.txt", [("other", "OTHER"), ("k = 1", "k = 2")]
            )
        self.assertIn("2 times", str(cm.exception))
        self.assertEqual((self.tmp / "a.txt").read_text(), "k = 1\nk = 1\nother\n")

    def test_overlap_rejected_before_writing(self):
        write(self.tmp, "a.txt", "abcdef\n")
        with self.assertRaises(EditConflictError) as cm:
            self.loop.surgical_replace_many(
                "a.txt", [("bcd", "X"), ("cde", "Y")]
            )
        msg = str(cm.exception)
        self.assertIn("overlap", msg)
        self.assertIn("0", msg)
        self.assertIn("1", msg)
        self.assertEqual((self.tmp / "a.txt").read_text(), "abcdef\n")

    def test_identical_old_texts_overlap(self):
        write(self.tmp, "a.txt", "hello\n")
        with self.assertRaises(EditConflictError) as cm:
            self.loop.surgical_replace_many(
                "a.txt", [("hello", "hi"), ("hello", "hey")]
            )
        self.assertIn("overlap", str(cm.exception))
        self.assertEqual((self.tmp / "a.txt").read_text(), "hello\n")

    def test_empty_edits_rejected(self):
        write(self.tmp, "a.txt", "hello\n")
        with self.assertRaises(EditConflictError):
            self.loop.surgical_replace_many("a.txt", [])

    def test_dry_run_writes_nothing(self):
        write(self.tmp, "a.py", PY_SAMPLE)
        diff = self.loop.surgical_replace_many(
            "a.py", [("x = 1", "x = 10")], dry_run=True
        )
        self.assertEqual((self.tmp / "a.py").read_text(), PY_SAMPLE)
        self.assertIn("+x = 10", diff)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            self.loop.surgical_replace_many("nope.py", [("a", "b")])


class TestConflictDetection(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="editloop_"))
        self.loop = make_loop(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_sequential_invalidation_detected(self):
        # edit 0's new_text introduces a second "bb", so edit 1's old_text
        # is unique in the original but ambiguous after edit 0 is applied.
        write(self.tmp, "a.txt", "aa\nbb\n")
        with self.assertRaises(EditConflictError) as cm:
            self.loop.surgical_replace_many(
                "a.txt", [("aa", "aa\nbb"), ("bb", "cc")]
            )
        msg = str(cm.exception)
        self.assertIn("edit 1", msg)
        self.assertIn("invalidated", msg)
        self.assertEqual((self.tmp / "a.txt").read_text(), "aa\nbb\n")

    def test_nested_spans_rejected_as_overlap(self):
        # edit 0 deletes the text edit 1 was looking for; spans nest.
        write(self.tmp, "a.txt", "alpha beta gamma\n")
        with self.assertRaises(EditConflictError) as cm:
            self.loop.surgical_replace_many(
                "a.txt", [("alpha beta", "ALPHA"), ("beta gamma", "BETA")]
            )
        # spans (0,10) and (6,16) overlap -> caught by the overlap check first
        self.assertIn("overlap", str(cm.exception))
        self.assertEqual((self.tmp / "a.txt").read_text(), "alpha beta gamma\n")

    def test_contained_span_rejected_as_overlap(self):
        # edit 1's old_text is contained in edit 0's span.
        write(self.tmp, "a.txt", "one two three\n")
        with self.assertRaises(EditConflictError) as cm:
            self.loop.surgical_replace_many(
                "a.txt", [("one two", "1+2"), ("two", "TWO")]
            )
        # "two" span (4,7) is inside "one two" span (0,7) -> overlap check fires
        self.assertIn("overlap", str(cm.exception))

    def test_consumption_reported_as_overlap(self):
        write(self.tmp, "a.txt", "start middle end\n")
        with self.assertRaises(EditConflictError) as cm:
            self.loop.surgical_replace_many(
                "a.txt", [("middle", ""), ("middle end", "M")]
            )
        self.assertIn("overlap", str(cm.exception))

    def test_surgical_replace_ambiguous_still_raises(self):
        write(self.tmp, "a.txt", "k = 1\nk = 1\n")
        with self.assertRaises(ValueError) as cm:
            self.loop.surgical_replace("a.txt", "k = 1", "k = 2")
        self.assertIn("2 times", str(cm.exception))
        self.assertEqual((self.tmp / "a.txt").read_text(), "k = 1\nk = 1\n")

    def test_surgical_replace_missing_still_raises(self):
        write(self.tmp, "a.txt", "hello\n")
        with self.assertRaises(ValueError):
            self.loop.surgical_replace("a.txt", "missing", "x")
        self.assertEqual((self.tmp / "a.txt").read_text(), "hello\n")


class TestDryRunAndPreview(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="editloop_"))
        self.loop = make_loop(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_surgical_replace_dry_run(self):
        write(self.tmp, "a.py", PY_SAMPLE)
        diff = self.loop.surgical_replace("a.py", "y = 2", "y = 22", dry_run=True)
        self.assertEqual((self.tmp / "a.py").read_text(), PY_SAMPLE)
        self.assertIn("+y = 22", diff)
        # and the real run still works afterwards
        self.loop.surgical_replace("a.py", "y = 2", "y = 22")
        self.assertIn("y = 22", (self.tmp / "a.py").read_text())

    def test_preview_replace(self):
        write(self.tmp, "a.py", PY_SAMPLE)
        result = self.loop.preview_replace(
            "a.py", [("x = 1", "x = 10"), ("z = 3", "z = 30")]
        )
        self.assertEqual((self.tmp / "a.py").read_text(), PY_SAMPLE)  # untouched
        self.assertTrue(result["would_change"])
        self.assertIn("+x = 10", result["diff"])
        self.assertEqual(
            result["matches"],
            [
                {"old_text": "x = 1", "count": 1, "line": 1},
                {"old_text": "z = 3", "count": 1, "line": 3},
            ],
        )

    def test_preview_replace_no_change(self):
        write(self.tmp, "a.py", PY_SAMPLE)
        result = self.loop.preview_replace("a.py", [("x = 1", "x = 1")])
        self.assertFalse(result["would_change"])
        self.assertEqual((self.tmp / "a.py").read_text(), PY_SAMPLE)

    def test_preview_replace_reports_missing_and_ambiguous(self):
        write(self.tmp, "a.txt", "dup\ndup\nline3\n")
        with self.assertRaises(EditConflictError):
            self.loop.preview_replace("a.txt", [("dup", "D"), ("zzz", "Z")])
        self.assertEqual((self.tmp / "a.txt").read_text(), "dup\ndup\nline3\n")

    def test_preview_replace_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            self.loop.preview_replace("nope.py", [("a", "b")])


class TestSyntaxAutoRevert(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="editloop_"))
        self.loop = make_loop(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_surgical_replace_syntax_break_reverts(self):
        write(self.tmp, "m.py", PY_SAMPLE)
        with self.assertRaises(EditSyntaxError) as cm:
            self.loop.surgical_replace("m.py", "y = 2", "def broken(:")
        msg = str(cm.exception)
        self.assertIn("line 2", msg)
        self.assertIn("invalid syntax", msg)
        self.assertEqual((self.tmp / "m.py").read_text(), PY_SAMPLE)

    def test_surgical_replace_many_syntax_break_reverts(self):
        write(self.tmp, "m.py", PY_SAMPLE)
        with self.assertRaises(EditSyntaxError):
            self.loop.surgical_replace_many(
                "m.py", [("x = 1", "x = 10"), ("z = 3", "def broken(:")]
            )
        self.assertEqual((self.tmp / "m.py").read_text(), PY_SAMPLE)

    def test_non_python_file_skipped(self):
        # "broken" content in a non-.py file must write fine — no guessing
        # at other grammars.
        write(self.tmp, "n.txt", "hello\n")
        diff = self.loop.surgical_replace("n.txt", "hello", "def broken(:")
        self.assertEqual((self.tmp / "n.txt").read_text(), "def broken(:\n")
        self.assertIn("+def broken(:", diff)

    def test_write_text_verified_direct(self):
        p = write(self.tmp, "m.py", PY_SAMPLE)
        with self.assertRaises(EditSyntaxError) as cm:
            write_text_verified(p, PY_SAMPLE, "x = \n")
        self.assertIn("line 1", str(cm.exception))
        self.assertEqual(p.read_text(), PY_SAMPLE)
        # a good write passes through
        write_text_verified(p, PY_SAMPLE, "x = 2\n")
        self.assertEqual(p.read_text(), "x = 2\n")

    def test_apply_edit_syntax_break_reverts(self):
        write(self.tmp, "m.py", PY_SAMPLE)
        plan = EditPlan(
            file_path="m.py",
            original=PY_SAMPLE,
            modified="x = 1\ndef broken(:\n",
            diff="broken",
        )
        with self.assertRaises(EditSyntaxError):
            asyncio.run(self.loop.apply_edit(plan))
        self.assertEqual((self.tmp / "m.py").read_text(), PY_SAMPLE)

    def test_apply_edit_good_write(self):
        write(self.tmp, "m.py", PY_SAMPLE)
        plan = EditPlan(
            file_path="m.py",
            original=PY_SAMPLE,
            modified="x = 100\ny = 2\nz = 3\n",
            diff="good",
        )
        result = asyncio.run(self.loop.apply_edit(plan))
        self.assertTrue(result.success)
        self.assertEqual((self.tmp / "m.py").read_text(), "x = 100\ny = 2\nz = 3\n")


class TestVerifyFormatPreserved(unittest.TestCase):
    def test_true_for_legit_edit(self):
        original = "aaa bbb ccc\n"
        modified = "aaa BBB ccc\n"
        self.assertTrue(verify_format_preserved(original, modified, [(4, 7)]))

    def test_false_for_extra_change_outside_spans(self):
        original = "aaa bbb ccc\n"
        modified = "aaa BBB CCC\n"  # ccc also changed, outside the span
        self.assertFalse(verify_format_preserved(original, modified, [(4, 7)]))

    def test_true_for_growing_replacement(self):
        original = "x=1\ny=2\n"
        modified = "x=100\ny=2\n"
        self.assertTrue(verify_format_preserved(original, modified, [(0, 3)]))

    def test_true_for_multiple_spans(self):
        original = "abcdef\n"
        modified = "XbcYef\n"
        self.assertTrue(verify_format_preserved(original, modified, [(0, 1), (3, 4)]))

    def test_false_for_untouched_region_rewrite(self):
        original = "aaa bbb\n"
        modified = "aaa bbb \n"  # trailing space added outside the span
        self.assertFalse(verify_format_preserved(original, modified, [(0, 3)]))

    def test_empty_spans_identical(self):
        self.assertTrue(verify_format_preserved("same\n", "same\n", []))
        self.assertFalse(verify_format_preserved("a\n", "b\n", []))


class TestPurePythonApplier(unittest.TestCase):
    DIFF_MULTI = (
        "--- a/alpha.py\n"
        "+++ b/alpha.py\n"
        "@@ -1,3 +1,3 @@\n"
        " line1\n"
        "-old2\n"
        "+new2\n"
        " line3\n"
        "@@ -8,3 +8,3 @@\n"
        " line8\n"
        "-old9\n"
        "+new9\n"
        " line10\n"
        "--- a/beta.txt\n"
        "+++ b/beta.txt\n"
        "@@ -1,2 +1,2 @@\n"
        "-hello\n"
        "+goodbye\n"
        " world\n"
    )

    def test_multi_file_multi_hunk(self):
        files = {
            "alpha.py": (
                "line1\nold2\nline3\nline4\nline5\nline6\nline7\n"
                "line8\nold9\nline10\n"
            ),
            "beta.txt": "hello\nworld\n",
        }
        out = apply_unified_diff(self.DIFF_MULTI, files)
        self.assertEqual(
            out["alpha.py"],
            "line1\nnew2\nline3\nline4\nline5\nline6\nline7\n"
            "line8\nnew9\nline10\n",
        )
        self.assertEqual(out["beta.txt"], "goodbye\nworld\n")
        # inputs not mutated
        self.assertEqual(files["beta.txt"], "hello\nworld\n")

    def test_new_file_creation(self):
        diff = (
            "--- /dev/null\n"
            "+++ b/new.txt\n"
            "@@ -0,0 +1,2 @@\n"
            "+one\n"
            "+two\n"
        )
        out = apply_unified_diff(diff, {})
        self.assertEqual(out["new.txt"], "one\ntwo\n")

    def test_file_deletion(self):
        diff = (
            "--- a/gone.txt\n"
            "+++ /dev/null\n"
            "@@ -1,2 +0,0 @@\n"
            "-del1\n"
            "-del2\n"
        )
        out = apply_unified_diff(diff, {"gone.txt": "del1\ndel2\n"})
        self.assertIsNone(out["gone.txt"])

    def test_context_mismatch_raises(self):
        diff = (
            "--- a/f.txt\n"
            "+++ b/f.txt\n"
            "@@ -1,3 +1,3 @@\n"
            " WRONG\n"
            "-old\n"
            "+new\n"
            " tail\n"
        )
        with self.assertRaises(DiffApplyError):
            apply_unified_diff(diff, {"f.txt": "line1\nold\ntail\n"})

    def test_create_existing_raises(self):
        diff = (
            "--- /dev/null\n"
            "+++ b/f.txt\n"
            "@@ -0,0 +1,1 @@\n"
            "+x\n"
        )
        with self.assertRaises(DiffApplyError):
            apply_unified_diff(diff, {"f.txt": "already here\n"})

    def test_delete_missing_raises(self):
        diff = (
            "--- a/f.txt\n"
            "+++ /dev/null\n"
            "@@ -1,1 +0,0 @@\n"
            "-x\n"
        )
        with self.assertRaises(DiffApplyError):
            apply_unified_diff(diff, {})

    def test_missing_target_raises(self):
        diff = (
            "--- a/f.txt\n"
            "+++ b/f.txt\n"
            "@@ -1,1 +1,1 @@\n"
            "-x\n"
            "+y\n"
        )
        with self.assertRaises(DiffApplyError):
            apply_unified_diff(diff, {})

    def test_malformed_diff_raises(self):
        with self.assertRaises(DiffApplyError):
            apply_unified_diff("this is not a diff\n", {"f.txt": "x\n"})
        with self.assertRaises(DiffApplyError):
            apply_unified_diff("--- a/f.txt\n", {"f.txt": "x\n"})

    def test_no_trailing_newline_preserved(self):
        diff = (
            "--- a/f.txt\n"
            "+++ b/f.txt\n"
            "@@ -1,2 +1,2 @@\n"
            " a\n"
            "-b\n"
            "\\ No newline at end of file\n"
            "+c\n"
            "\\ No newline at end of file\n"
        )
        out = apply_unified_diff(diff, {"f.txt": "a\nb"})
        self.assertEqual(out["f.txt"], "a\nc")

    def test_trailing_newline_added(self):
        diff = (
            "--- a/f.txt\n"
            "+++ b/f.txt\n"
            "@@ -1,1 +1,1 @@\n"
            "-b\n"
            "\\ No newline at end of file\n"
            "+b\n"
        )
        out = apply_unified_diff(diff, {"f.txt": "b"})
        self.assertEqual(out["f.txt"], "b\n")

    def test_offset_tolerance(self):
        # hunk header claims line 1, but the content actually starts at line 2
        diff = (
            "--- a/f.txt\n"
            "+++ b/f.txt\n"
            "@@ -1,3 +1,3 @@\n"
            " l1\n"
            "-l2old\n"
            "+l2new\n"
            " l3\n"
        )
        out = apply_unified_diff(diff, {"f.txt": "x0\nl1\nl2old\nl3\n"})
        self.assertEqual(out["f.txt"], "x0\nl1\nl2new\nl3\n")

    def test_pure_insertion_hunk(self):
        diff = (
            "--- a/f.txt\n"
            "+++ b/f.txt\n"
            "@@ -1,0 +2 @@\n"
            "+inserted\n"
        )
        out = apply_unified_diff(diff, {"f.txt": "first\nsecond\n"})
        self.assertEqual(out["f.txt"], "first\ninserted\nsecond\n")

    def test_content_lines_looking_like_diff_structure(self):
        # Removed/added lines starting with ---, +++, @@ must not confuse
        # the parser.
        diff = (
            "--- a/f.txt\n"
            "+++ b/f.txt\n"
            "@@ -1,3 +1,3 @@\n"
            " top\n"
            "---- header line\n"
            "-@@ not a hunk\n"
            "+++ b/real\n"
            "+@@ still not a hunk\n"
            " bottom\n"
        )
        out = apply_unified_diff(
            diff, {"f.txt": "top\n--- header line\n@@ not a hunk\nbottom\n"})
        self.assertEqual(
            out["f.txt"], "top\n++ b/real\n@@ still not a hunk\nbottom\n")

    def test_truncated_hunk_raises(self):
        # Header claims 2 old lines but only 1 is present.
        diff = (
            "--- a/f.txt\n"
            "+++ b/f.txt\n"
            "@@ -1,2 +1,1 @@\n"
            "-a\n"
            "+b\n"
        )
        with self.assertRaises(DiffApplyError) as cm:
            apply_unified_diff(diff, {"f.txt": "a\nz\n"})
        self.assertIn("truncated", str(cm.exception))

    def test_multi_hunk_no_trailing_newline(self):
        diff = (
            "--- a/f.txt\n"
            "+++ b/f.txt\n"
            "@@ -1,2 +1,2 @@\n"
            " a\n"
            "-b\n"
            "+B\n"
            "@@ -4,2 +4,2 @@\n"
            " d\n"
            "-e\n"
            "\\ No newline at end of file\n"
            "+E\n"
            "\\ No newline at end of file\n"
        )
        out = apply_unified_diff(diff, {"f.txt": "a\nb\nc\nd\ne"})
        self.assertEqual(out["f.txt"], "a\nB\nc\nd\nE")


@unittest.skipUnless(shutil.which("patch"), "GNU patch binary not available")
class TestDifferentialVsGnuPatch(unittest.TestCase):
    """The canonical engine must agree with GNU patch, including on the
    adjacent/overlapping hunks difflib emits (a cursor-model applier cannot
    express those)."""

    TRIALS = 120

    def _gnu_apply(self, old_text: str, diff: str) -> str | None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "f.txt").write_text(old_text)
            (root / "t.diff").write_text(diff)
            r = subprocess.run(
                ["patch", "-p1", "--no-backup-if-mismatch", "-i", "t.diff"],
                cwd=td, capture_output=True, text=True, timeout=30)
            if r.returncode != 0:
                return None  # GNU itself refuses this diff; not our problem
            return (root / "f.txt").read_text()

    def test_random_diffs_match_gnu(self):
        rng = random.Random(20261001)
        vocab = ["alpha", "beta", "gamma", "delta", "import os", "x = 1",
                 "return None", ""]
        compared = 0
        for _ in range(self.TRIALS):
            old = [rng.choice(vocab) for _ in range(rng.randint(1, 12))]
            new = list(old)
            for _ in range(rng.randint(1, 4)):
                op = rng.choice(["chg", "ins", "del"])
                pos = rng.randrange(len(new) + 1) if new else 0
                if op == "chg" and new:
                    new[pos % len(new)] = rng.choice(vocab)
                elif op == "ins":
                    new.insert(pos, rng.choice(vocab))
                elif new:
                    del new[pos % len(new)]
            old_text = "\n".join(old) + ("\n" if rng.random() < 0.8 else "")
            new_text = "\n".join(new) + ("\n" if rng.random() < 0.8 else "")
            diff = "".join(unified_diff(
                old_text.splitlines(keepends=True),
                new_text.splitlines(keepends=True), "a/f.txt", "b/f.txt"))
            if not diff.strip():
                continue
            gnu = self._gnu_apply(old_text, diff)
            if gnu is None:
                continue
            ours = core_apply_unified_diff(diff, {"f.txt": old_text})["f.txt"]
            self.assertEqual(ours, gnu, f"divergence on diff:\n{diff}")
            compared += 1
        self.assertGreater(compared, 50, "too few comparable trials ran")

    def test_adjacent_hunks_match_gnu(self):
        # difflib-style adjacent hunks: hunk 2's stated position overlaps
        # the region hunk 1 already rewrote.
        old_text = "p = 'a'\nq = 'b'\n"
        diff = (
            "--- a/f.txt\n+++ b/f.txt\n"
            "@@ -1,2 +1,2 @@\n-p = 'a'\n+p = 'A'\n q = 'b'\n"
            "@@ -2,1 +2,2 @@\n q = 'b'\n+r = 'new'\n"
        )
        gnu = self._gnu_apply(old_text, diff)
        self.assertIsNotNone(gnu)
        ours = core_apply_unified_diff(diff, {"f.txt": old_text})["f.txt"]
        self.assertEqual(ours, gnu)
        self.assertEqual(ours, "p = 'A'\nq = 'b'\nr = 'new'\n")


class TestApplyPatchTool(unittest.TestCase):
    DIFF = (
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1,3 +1,3 @@\n"
        " x = 1\n"
        "-y = 2\n"
        "+y = 20\n"
        " z = 3\n"
    )
    DIFF_SYNTAX_BREAK = (
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1,3 +1,3 @@\n"
        " x = 1\n"
        "-y = 2\n"
        "+def broken(:\n"
        " z = 3\n"
    )

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="editloop_"))
        self.reg = make_registry(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fallback_applies_when_patch_missing(self):
        write(self.tmp, "a.py", PY_SAMPLE)
        with mock.patch("shutil.which", return_value=None):
            outcome = self.reg.call("apply_patch", path="a.py", unified_diff=self.DIFF)
        self.assertTrue(outcome, outcome.error if hasattr(outcome, "error") else outcome)
        result = outcome.unwrap()
        self.assertTrue(result["applied"])
        self.assertEqual(result["fallback"], "pure-python")
        self.assertEqual((self.tmp / "a.py").read_text(), "x = 1\ny = 20\nz = 3\n")

    def test_fallback_multi_file(self):
        write(self.tmp, "a.py", PY_SAMPLE)
        diff = self.DIFF + (
            "--- a/b.txt\n"
            "+++ b/b.txt\n"
            "@@ -1,1 +1,1 @@\n"
            "-old\n"
            "+new\n"
        )
        write(self.tmp, "b.txt", "old\n")
        with mock.patch("shutil.which", return_value=None):
            outcome = self.reg.call("apply_patch", path="a.py", unified_diff=diff)
        result = outcome.unwrap()
        self.assertTrue(result["applied"])
        self.assertEqual((self.tmp / "a.py").read_text(), "x = 1\ny = 20\nz = 3\n")
        self.assertEqual((self.tmp / "b.txt").read_text(), "new\n")
        self.assertIn("a.py", result["files"])
        self.assertIn("b.txt", result["files"])

    def test_fallback_bad_diff_is_tool_error(self):
        write(self.tmp, "a.py", PY_SAMPLE)
        with mock.patch("shutil.which", return_value=None):
            outcome = self.reg.call("apply_patch", path="a.py", unified_diff="garbage")
        self.assertFalse(outcome)
        self.assertIn("no file sections", str(outcome.error))
        self.assertEqual((self.tmp / "a.py").read_text(), PY_SAMPLE)

    def test_fallback_context_mismatch_is_tool_error(self):
        write(self.tmp, "a.py", "totally different\n")
        with mock.patch("shutil.which", return_value=None):
            outcome = self.reg.call("apply_patch", path="a.py", unified_diff=self.DIFF)
        self.assertFalse(outcome)
        self.assertIn("does not match", str(outcome.error))
        self.assertEqual((self.tmp / "a.py").read_text(), "totally different\n")

    def test_fallback_syntax_break_reverts(self):
        write(self.tmp, "a.py", PY_SAMPLE)
        with mock.patch("shutil.which", return_value=None):
            outcome = self.reg.call(
                "apply_patch", path="a.py", unified_diff=self.DIFF_SYNTAX_BREAK
            )
        self.assertFalse(outcome)
        self.assertIn("syntax", str(outcome.error).lower())
        self.assertIn("revert", str(outcome.error).lower())
        self.assertEqual((self.tmp / "a.py").read_text(), PY_SAMPLE)

    def test_fallback_diff_not_touching_target(self):
        write(self.tmp, "a.py", PY_SAMPLE)
        write(self.tmp, "b.txt", "old\n")
        diff = (
            "--- a/b.txt\n"
            "+++ b/b.txt\n"
            "@@ -1,1 +1,1 @@\n"
            "-old\n"
            "+new\n"
        )
        with mock.patch("shutil.which", return_value=None):
            outcome = self.reg.call("apply_patch", path="a.py", unified_diff=diff)
        self.assertFalse(outcome)
        self.assertIn("did not touch", str(outcome.error))

    def test_fallback_hunk_failure_names_hunk_and_why(self):
        # wave D: a patch failure must say WHICH hunk and WHY, not "done".
        write(self.tmp, "a.py", PY_SAMPLE)
        diff = (
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "@@ -1,3 +1,3 @@\n"
            " x = 1\n"
            "-y = 2\n"
            "+y = 20\n"
            " z = 3\n"
            "@@ -1,3 +1,3 @@\n"
            " line one\n"
            "-WRONG\n"
            "+right\n"
            " line three\n"
        )
        with mock.patch("shutil.which", return_value=None):
            outcome = self.reg.call("apply_patch", path="a.py", unified_diff=diff)
        self.assertFalse(outcome)
        msg = str(outcome.error)
        self.assertIn("hunk 2", msg)  # which hunk
        self.assertIn("a.py", msg)    # which file
        # all-or-nothing: nothing was written
        self.assertEqual((self.tmp / "a.py").read_text(), PY_SAMPLE)

    @unittest.skipIf(shutil.which("patch") is None, "`patch` binary not available")
    def test_fast_path_with_patch_binary(self):
        write(self.tmp, "a.py", PY_SAMPLE)
        outcome = self.reg.call("apply_patch", path="a.py", unified_diff=self.DIFF)
        result = outcome.unwrap()
        self.assertTrue(result["applied"])
        self.assertNotIn("fallback", result)
        self.assertEqual((self.tmp / "a.py").read_text(), "x = 1\ny = 20\nz = 3\n")

    @unittest.skipIf(shutil.which("patch") is None, "`patch` binary not available")
    def test_fast_path_syntax_break_reverts(self):
        write(self.tmp, "a.py", PY_SAMPLE)
        outcome = self.reg.call(
            "apply_patch", path="a.py", unified_diff=self.DIFF_SYNTAX_BREAK
        )
        self.assertFalse(outcome)
        self.assertIn("syntax", str(outcome.error).lower())
        self.assertEqual((self.tmp / "a.py").read_text(), PY_SAMPLE)

    def test_edit_file_tool_syntax_break_reverts(self):
        write(self.tmp, "a.py", PY_SAMPLE)
        outcome = self.reg.call(
            "edit_file", path="a.py", old_text="y = 2", new_text="def broken(:"
        )
        self.assertFalse(outcome)
        self.assertIn("invalid syntax", str(outcome.error))
        self.assertEqual((self.tmp / "a.py").read_text(), PY_SAMPLE)

    def test_edit_file_tool_happy_path(self):
        write(self.tmp, "a.py", PY_SAMPLE)
        outcome = self.reg.call(
            "edit_file", path="a.py", old_text="y = 2", new_text="y = 22"
        )
        result = outcome.unwrap()
        self.assertIn("+y = 22", result["diff"])
        self.assertEqual((self.tmp / "a.py").read_text(), "x = 1\ny = 22\nz = 3\n")


class TestLLMPathHonesty(unittest.TestCase):
    """Wave D: the LLM-driven edit path must fail honestly when the model
    returns no code block — never write prose into the file, never claim
    'no changes needed' for a response it couldn't parse."""

    PROSE_ONLY = (
        "Sure! The best approach here is to refactor the module. "
        "You should consider renaming the variable and adding tests. "
        "Let me know if you want the full diff."
    )

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="editloop_llm_"))
        write(self.tmp, "a.py", PY_SAMPLE)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _prose_agent(self):
        return types.SimpleNamespace(
            chat=lambda prompt: types.SimpleNamespace(content=self.PROSE_ONLY))

    def _code_agent(self, code: str):
        return types.SimpleNamespace(
            chat=lambda prompt: types.SimpleNamespace(
                content=f"Here you go:\n```python\n{code}\n```"))

    def test_extract_code_returns_empty_without_fence(self):
        loop = EditLoop(agent=None, project_root=str(self.tmp))
        self.assertEqual(loop._extract_code(self.PROSE_ONLY), "")
        self.assertEqual(
            loop._extract_code("```python\nx = 1\n```"), "x = 1")

    def test_edit_file_no_code_block_fails_honestly(self):
        loop = EditLoop(agent=self._prose_agent(), project_root=str(self.tmp))
        result = asyncio.run(loop.edit_file("a.py", "make it better"))
        self.assertFalse(result.success)
        self.assertIn("no fenced code block", result.error)
        # the file is untouched — prose was NOT written into it
        self.assertEqual((self.tmp / "a.py").read_text(), PY_SAMPLE)

    def test_edit_file_code_block_still_applies(self):
        loop = EditLoop(agent=self._code_agent("x = 10\ny = 2\nz = 3\n"),
                        project_root=str(self.tmp))
        result = asyncio.run(loop.edit_file("a.py", "bump x"))
        self.assertTrue(result.success, result.error)
        self.assertEqual((self.tmp / "a.py").read_text(),
                         "x = 10\ny = 2\nz = 3")

    def test_plan_edit_no_code_block_raises(self):
        loop = EditLoop(agent=self._prose_agent(), project_root=str(self.tmp))
        with self.assertRaises(ValueError) as ctx:
            asyncio.run(loop.plan_edit("a.py", "make it better"))
        self.assertIn("no fenced code block", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
