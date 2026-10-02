"""Wave K: code workspace object (nomorals/codews) — repos, patches, tests, builds."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from nomorals.codews import (
    CodeWorkspace,
    WorkspaceError,
    apply_patch,
    preview_patch,
    record_patch,
    review_patch,
    run_build,
    run_tests,
)
from nomorals.storage.artifacts import ARTIFACT_URI_SCHEME, ArtifactStore
from nomorals.storage.blob import BlobStore
from nomorals.storage.db import Database

GIT = shutil.which("git")

MODIFY_DIFF = """\
--- a/hello.txt
+++ b/hello.txt
@@ -1,3 +1,3 @@
 line1
-line2
+line2-changed
 line3
"""

CREATE_DIFF = """\
--- /dev/null
+++ b/new.txt
@@ -0,0 +1,2 @@
+hello
+world
"""

DELETE_DIFF = """\
--- a/gone.txt
+++ /dev/null
@@ -1,2 +0,0 @@
-gone1
-gone2
"""

BAD_DIFF = """\
--- a/hello.txt
+++ b/hello.txt
@@ -1,3 +1,3 @@
 line1
-WRONG-CONTEXT
+changed
 line3
"""

ESCAPE_DIFF = """\
--- /dev/null
+++ b/../../evil.txt
@@ -0,0 +1 @@
+evil
"""


def _git(*args, cwd):
    proc = subprocess.run([GIT, *args], cwd=str(cwd), capture_output=True,
                          text=True, timeout=60)
    if proc.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {proc.stderr}")
    return proc


def make_store(test):
    tmp = tempfile.mkdtemp()
    test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    db = Database(":memory:")
    db.migrate()
    blobs = BlobStore(db, Path(tmp) / "blobs")
    test.addCleanup(db.close)
    return ArtifactStore(db, blobs)


class CodewsTestCase(unittest.TestCase):
    def setUp(self):
        if not GIT:
            self.skipTest("git binary not found on PATH")
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.repo = Path(self.tmp) / "repo"
        self.repo.mkdir()
        _git("init", "-q", cwd=self.repo)
        _git("config", "user.email", "test@example.com", cwd=self.repo)
        _git("config", "user.name", "Test", cwd=self.repo)
        (self.repo / "hello.txt").write_text("line1\nline2\nline3\n")
        (self.repo / "gone.txt").write_text("gone1\ngone2\n")
        _git("add", ".", cwd=self.repo)
        _git("commit", "-q", "-m", "initial commit", cwd=self.repo)
        self.ws = CodeWorkspace(self.repo)

    # ── status / diff / log ────────────────────────────────────────────
    def test_status_clean_then_dirty(self):
        st = self.ws.status()
        self.assertEqual(st["staged"], [])
        self.assertEqual(st["unstaged"], [])
        self.assertEqual(st["untracked"], [])
        (self.repo / "hello.txt").write_text("line1\nline2!\nline3\n")
        (self.repo / "scratch.txt").write_text("tmp\n")
        st = self.ws.status()
        self.assertIn("hello.txt", st["unstaged"])
        self.assertIn("scratch.txt", st["untracked"])
        self.assertIsInstance(st["ahead"], int)
        self.assertIsInstance(st["behind"], int)

    def test_diff_nonempty_after_edit(self):
        (self.repo / "hello.txt").write_text("line1\nline2!\nline3\n")
        d = self.ws.diff()
        self.assertIn("hello.txt", d)
        self.assertIn("-line2", d)
        self.assertIn("+line2!", d)

    def test_log_has_commit(self):
        entries = self.ws.log(5)
        self.assertGreaterEqual(len(entries), 1)
        self.assertEqual(entries[0]["message"], "initial commit")
        self.assertEqual(len(entries[0]["sha"]), 40)

    # ── branches ───────────────────────────────────────────────────────
    def test_branches_create_switch_current(self):
        names = [b["name"] for b in self.ws.branches()]
        self.assertTrue(any(b["current"] for b in self.ws.branches()))
        self.assertIn(self.ws.current_branch(), names)
        self.ws.create_branch("feature-x")
        self.assertIn("feature-x", [b["name"] for b in self.ws.branches()])
        self.ws.switch_branch("feature-x")
        self.assertEqual(self.ws.current_branch(), "feature-x")

    # ── worktrees ──────────────────────────────────────────────────────
    def test_worktree_add_list_remove(self):
        wt = Path(self.tmp) / "wt1"
        self.ws.worktree_add(str(wt), "wt-branch")
        entries = self.ws.worktree_list()
        hit = [e for e in entries if e["branch"] == "wt-branch"]
        self.assertEqual(len(hit), 1)
        self.assertTrue(hit[0]["sha"])
        self.assertTrue(Path(hit[0]["path"]).is_dir())
        self.ws.worktree_remove(str(wt), force=True)
        entries = self.ws.worktree_list()
        self.assertFalse([e for e in entries if e["branch"] == "wt-branch"])

    # ── non-git dir ────────────────────────────────────────────────────
    def test_non_git_dir_raises(self):
        plain = Path(self.tmp) / "plain"
        plain.mkdir()
        ws = CodeWorkspace(plain)
        with self.assertRaises(WorkspaceError):
            ws.status()

    # ── review_patch ───────────────────────────────────────────────────
    def test_review_patch_stats(self):
        stats = review_patch(MODIFY_DIFF + CREATE_DIFF)
        self.assertEqual(stats["totals"]["files"], 2)
        self.assertEqual(stats["totals"]["additions"], 3)
        self.assertEqual(stats["totals"]["deletions"], 1)
        by_path = {f["path"]: f for f in stats["files"]}
        self.assertEqual(by_path["hello.txt"]["status"], "modified")
        self.assertEqual(by_path["hello.txt"]["hunks"], 1)
        self.assertEqual(by_path["new.txt"]["status"], "added")

    # ── apply_patch ────────────────────────────────────────────────────
    def test_apply_patch_dry_run_leaves_files_untouched(self):
        before = (self.repo / "hello.txt").read_text()
        results = apply_patch(MODIFY_DIFF, dry_run=True, root=self.repo)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["ok"])
        self.assertEqual((self.repo / "hello.txt").read_text(), before)

    def test_apply_patch_dry_run_false_writes(self):
        results = apply_patch(MODIFY_DIFF + CREATE_DIFF, dry_run=False,
                              root=self.repo)
        self.assertTrue(all(r["ok"] for r in results))
        self.assertIn("line2-changed", (self.repo / "hello.txt").read_text())
        self.assertEqual((self.repo / "new.txt").read_text(), "hello\nworld\n")
        results = apply_patch(DELETE_DIFF, dry_run=False, root=self.repo)
        self.assertTrue(results[0]["ok"])
        self.assertFalse((self.repo / "gone.txt").exists())

    def test_apply_patch_bad_diff_reports_not_ok(self):
        before = (self.repo / "hello.txt").read_text()
        results = apply_patch(BAD_DIFF, dry_run=True, root=self.repo)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["ok"])
        self.assertTrue(results[0]["error"])
        self.assertEqual((self.repo / "hello.txt").read_text(), before)

    def test_apply_patch_path_escape_raises(self):
        with self.assertRaises(WorkspaceError):
            apply_patch(ESCAPE_DIFF, dry_run=True, root=self.repo)
        with self.assertRaises(WorkspaceError):
            apply_patch(ESCAPE_DIFF, dry_run=False, root=self.repo)
        self.assertFalse((Path(self.tmp) / "evil.txt").exists())

    # ── preview_patch ──────────────────────────────────────────────────
    def test_preview_patch(self):
        snippet = preview_patch(MODIFY_DIFF, "hello.txt", root=self.repo)
        self.assertIn("line2", snippet["before"])
        self.assertIn("line2-changed", snippet["after"])
        self.assertNotIn("line2-changed", snippet["before"])

    def test_preview_patch_unknown_path_raises(self):
        with self.assertRaises(WorkspaceError):
            preview_patch(MODIFY_DIFF, "nope.txt", root=self.repo)

    # ── record_patch ───────────────────────────────────────────────────
    def test_record_patch(self):
        store = make_store(self)
        uri = record_patch(store, MODIFY_DIFF, mission_id="m-k",
                           creator="codews")
        self.assertTrue(uri.startswith(ARTIFACT_URI_SCHEME))
        art = store.resolve(uri)
        self.assertIsNotNone(art)
        self.assertEqual(art.type, "patch")
        self.assertEqual(art.mission_id, "m-k")
        self.assertEqual(art.provenance.source_type, "codews")
        self.assertEqual(store.read_text(art.id), MODIFY_DIFF)

    # ── run_tests ──────────────────────────────────────────────────────
    def test_run_tests_passing_fixture(self):
        pkg = Path(self.tmp) / "pkg"
        pkg.mkdir()
        (pkg / "test_sample.py").write_text(
            "import unittest\n"
            "class T(unittest.TestCase):\n"
            "    def test_one(self):\n"
            "        self.assertEqual(1 + 1, 2)\n")
        result = run_tests(pkg)
        self.assertTrue(result["ok"], msg=result["output"][-2000:])
        self.assertGreaterEqual(result["passed"], 1)
        self.assertEqual(result["failed"], 0)

    def test_run_tests_failing_fixture(self):
        pkg = Path(self.tmp) / "badpkg"
        pkg.mkdir()
        (pkg / "test_sample.py").write_text(
            "import unittest\n"
            "class T(unittest.TestCase):\n"
            "    def test_bad(self):\n"
            "        self.assertEqual(1, 2)\n")
        result = run_tests(pkg)
        self.assertFalse(result["ok"])
        self.assertGreaterEqual(result["failed"], 1)

    # ── run_build ──────────────────────────────────────────────────────
    def test_run_build_makefile_fixture(self):
        proj = Path(self.tmp) / "proj"
        proj.mkdir()
        (proj / "Makefile").write_text(
            ".PHONY: build\nbuild:\n\t@echo built-ok\n")
        result = run_build(proj)
        self.assertTrue(result["ok"], msg=result["output"][-2000:])
        self.assertIn("built-ok", result["output"])

    def test_run_build_no_build_system_raises(self):
        proj = Path(self.tmp) / "empty"
        proj.mkdir()
        with self.assertRaises(WorkspaceError):
            run_build(proj)


if __name__ == "__main__":
    unittest.main()
