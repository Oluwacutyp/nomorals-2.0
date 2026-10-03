"""Worker B: codews hardening — renames, binaries, empty diffs, runner
fallbacks, commit/sync/stash, and CLI end-to-end coverage."""

from __future__ import annotations

import contextlib
import io
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nomorals.codews import (
    CodeWorkspace,
    WorkspaceError,
    apply_patch,
    preview_patch,
    review_patch,
    run_build,
    run_tests,
)
from nomorals.codews import run as run_mod

GIT = shutil.which("git")

RENAME_DIFF = """\
diff --git a/old.txt b/new.txt
similarity index 66%
rename from old.txt
rename to new.txt
--- a/old.txt
+++ b/new.txt
@@ -1,3 +1,3 @@
 line1
-line2
+line2-renamed
 line3
"""

PURE_RENAME_DIFF = """\
diff --git a/old.txt b/moved.txt
similarity index 100%
rename from old.txt
rename to moved.txt
--- a/old.txt
+++ b/moved.txt
"""

BINARY_DIFF = """\
diff --git a/img.png b/img.png
index 3b18e51..a7f3c9d 100644
Binary files a/img.png and b/img.png differ
"""

BINARY_LITERAL_DIFF = """\
diff --git a/blob.bin b/blob.bin
new file mode 100644
index 0000000..9daeafb
GIT binary patch
literal 3
KcmZQzU|?o3

literal 0
HcmV?d00001

"""

MODE_ONLY_DIFF = """\
diff --git a/run.sh b/run.sh
old mode 100644
new mode 100755
"""

MODIFY_DIFF = """\
--- a/hello.txt
+++ b/hello.txt
@@ -1,3 +1,3 @@
 line1
-line2
+line2-changed
 line3
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

RENAME_ESCAPE_DIFF = """\
diff --git a/old.txt b/evil.txt
similarity index 100%
rename from old.txt
rename to ../../evil.txt
--- a/old.txt
+++ b/../../evil.txt
"""


def _git(*args, cwd):
    proc = subprocess.run([GIT, *args], cwd=str(cwd), capture_output=True,
                          text=True, timeout=60)
    if proc.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {proc.stderr}")
    return proc


class CodewsBTestCase(unittest.TestCase):
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
        (self.repo / "old.txt").write_text("line1\nline2\nline3\n")
        _git("add", ".", cwd=self.repo)
        _git("commit", "-q", "-m", "initial commit", cwd=self.repo)
        self.ws = CodeWorkspace(self.repo)

    # ── renames ──────────────────────────────────────────────────────
    def test_rename_with_hunks_applies(self):
        results = apply_patch(RENAME_DIFF, dry_run=False, root=self.repo)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertTrue(r["ok"], msg=r)
        self.assertEqual(r["status"], "renamed")
        self.assertEqual(r["path"], "new.txt")
        self.assertFalse((self.repo / "old.txt").exists())
        self.assertEqual((self.repo / "new.txt").read_text(),
                         "line1\nline2-renamed\nline3\n")

    def test_rename_dry_run_leaves_disk_untouched(self):
        results = apply_patch(RENAME_DIFF, dry_run=True, root=self.repo)
        self.assertTrue(results[0]["ok"])
        self.assertTrue((self.repo / "old.txt").exists())
        self.assertFalse((self.repo / "new.txt").exists())

    def test_pure_rename_no_hunks(self):
        results = apply_patch(PURE_RENAME_DIFF, dry_run=False, root=self.repo)
        self.assertTrue(results[0]["ok"], msg=results[0])
        self.assertFalse((self.repo / "old.txt").exists())
        self.assertEqual((self.repo / "moved.txt").read_text(),
                         "line1\nline2\nline3\n")

    def test_rename_missing_source_reports_error(self):
        (self.repo / "old.txt").unlink()
        results = apply_patch(RENAME_DIFF, dry_run=False, root=self.repo)
        self.assertFalse(results[0]["ok"])
        self.assertIn("no such file", results[0]["error"])
        self.assertFalse((self.repo / "new.txt").exists())

    def test_rename_target_exists_reports_error(self):
        (self.repo / "new.txt").write_text("occupied\n")
        results = apply_patch(RENAME_DIFF, dry_run=False, root=self.repo)
        self.assertFalse(results[0]["ok"])
        self.assertIn("already exists", results[0]["error"])
        self.assertTrue((self.repo / "old.txt").exists())
        self.assertEqual((self.repo / "new.txt").read_text(), "occupied\n")

    def test_rename_escape_refused(self):
        with self.assertRaises(WorkspaceError):
            apply_patch(RENAME_ESCAPE_DIFF, dry_run=True, root=self.repo)
        with self.assertRaises(WorkspaceError):
            apply_patch(RENAME_ESCAPE_DIFF, dry_run=False, root=self.repo)
        self.assertFalse((Path(self.tmp) / "evil.txt").exists())
        self.assertTrue((self.repo / "old.txt").exists())

    def test_review_rename_status(self):
        stats = review_patch(RENAME_DIFF)
        self.assertEqual(stats["files"][0]["status"], "renamed")
        self.assertEqual(stats["files"][0]["rename_from"], "old.txt")
        self.assertEqual(stats["files"][0]["path"], "new.txt")

    def test_preview_rename(self):
        snip = preview_patch(RENAME_DIFF, "new.txt", root=self.repo)
        self.assertIn("line2", snip["before"])
        self.assertNotIn("line2-renamed", snip["before"])
        self.assertIn("line2-renamed", snip["after"])

    # ── binary sections ──────────────────────────────────────────────
    def test_review_binary_section(self):
        stats = review_patch(BINARY_DIFF)
        self.assertEqual(len(stats["files"]), 1)
        f = stats["files"][0]
        self.assertEqual(f["status"], "binary")
        self.assertEqual(f["path"], "img.png")
        self.assertEqual((f["additions"], f["deletions"]), (0, 0))

    def test_review_binary_literal_section(self):
        stats = review_patch(BINARY_LITERAL_DIFF)
        self.assertEqual(stats["files"][0]["status"], "binary")
        self.assertEqual(stats["files"][0]["path"], "blob.bin")

    def test_apply_binary_refused_with_clear_error(self):
        before = sorted(p.name for p in self.repo.iterdir())
        results = apply_patch(BINARY_DIFF, dry_run=False, root=self.repo)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["ok"])
        self.assertIn("binary", results[0]["error"].lower())
        self.assertEqual(sorted(p.name for p in self.repo.iterdir()), before)

    def test_preview_binary_raises(self):
        with self.assertRaises(WorkspaceError):
            preview_patch(BINARY_DIFF, "img.png", root=self.repo)

    # ── empty / junk / mode-only diffs ───────────────────────────────
    def test_empty_diff_raises(self):
        with self.assertRaises(WorkspaceError):
            apply_patch("", dry_run=True, root=self.repo)
        with self.assertRaises(WorkspaceError):
            apply_patch("", dry_run=False, root=self.repo)

    def test_junk_diff_raises(self):
        with self.assertRaises(WorkspaceError):
            apply_patch("this is not a diff at all\n", root=self.repo)

    def test_mode_only_section_reports_clear_error(self):
        results = apply_patch(MODE_ONLY_DIFF, dry_run=False, root=self.repo)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["ok"])
        self.assertTrue(results[0]["error"])

    # ── mixed / partial failure ──────────────────────────────────────
    def test_review_mixed_totals(self):
        stats = review_patch(MODIFY_DIFF + RENAME_DIFF + BINARY_DIFF)
        by_path = {f["path"]: f for f in stats["files"]}
        self.assertEqual(stats["totals"]["files"], 3)
        self.assertEqual(by_path["hello.txt"]["status"], "modified")
        self.assertEqual(by_path["new.txt"]["status"], "renamed")
        self.assertEqual(by_path["img.png"]["status"], "binary")

    def test_partial_failure_reports_per_file(self):
        results = apply_patch(MODIFY_DIFF + BAD_DIFF, dry_run=False,
                              root=self.repo)
        self.assertEqual(len(results), 2)
        ok = [r for r in results if r["ok"]]
        failed = [r for r in results if not r["ok"]]
        self.assertEqual(len(ok), 1)
        self.assertEqual(len(failed), 1)
        self.assertTrue(failed[0]["error"])
        # the good file was still applied
        self.assertIn("line2-changed", (self.repo / "hello.txt").read_text())

    # ── run_tests fallbacks ──────────────────────────────────────────
    def test_run_tests_make_test_fallback(self):
        proj = Path(self.tmp) / "mkproj"
        proj.mkdir()
        (proj / "Makefile").write_text(
            ".PHONY: test\ntest:\n\t@echo make-test-ran\n")
        with mock.patch.object(run_mod, "_pytest_available", return_value=False):
            result = run_tests(proj)
        self.assertEqual(result["runner"], "make test")
        self.assertTrue(result["ok"], msg=result["output"][-1000:])
        self.assertIn("make-test-ran", result["output"])

    def test_run_tests_unittest_fallback_when_no_pytest_no_makefile(self):
        pkg = Path(self.tmp) / "upkg"
        pkg.mkdir()
        (pkg / "test_s.py").write_text(
            "import unittest\n"
            "class T(unittest.TestCase):\n"
            "    def test_one(self):\n"
            "        self.assertEqual(2 + 2, 4)\n")
        with mock.patch.object(run_mod, "_pytest_available", return_value=False):
            result = run_tests(pkg)
        self.assertEqual(result["runner"], "unittest")
        self.assertTrue(result["ok"], msg=result["output"][-1000:])
        self.assertEqual(result["failed"], 0)
        self.assertGreaterEqual(result["passed"], 1)

    def test_run_tests_selector_passed_through(self):
        seen = {}

        def fake_run_cmd(args, cwd, timeout):
            seen["args"] = args
            return subprocess.CompletedProcess(args, 0, "1 passed\n", "")

        pkg = Path(self.tmp) / "selpkg"
        pkg.mkdir()
        with mock.patch.object(run_mod, "_run_cmd", fake_run_cmd):
            result = run_tests(pkg, "tests/test_x.py::Test::test_y")
        self.assertTrue(result["ok"])
        self.assertIn("tests/test_x.py::Test::test_y", seen["args"])

    def test_run_tests_bad_root_raises(self):
        with self.assertRaises(WorkspaceError):
            run_tests(Path(self.tmp) / "no-such-dir")

    def test_run_cmd_timeout_raises(self):
        with self.assertRaises(WorkspaceError) as ctx:
            run_mod._run_cmd(["sleep", "5"], Path(self.tmp), timeout=1)
        self.assertIn("timed out", str(ctx.exception))

    def test_output_truncation(self):
        capped = run_mod._cap("x" * 30_000)
        self.assertLess(len(capped), 30_000)
        self.assertIn("truncated", capped)

    # ── run_build ────────────────────────────────────────────────────
    def test_run_build_explicit_target(self):
        seen = {}

        def fake_run_cmd(args, cwd, timeout):
            seen["args"] = args
            return subprocess.CompletedProcess(args, 0, "ok\n", "")

        proj = Path(self.tmp) / "bproj"
        proj.mkdir()
        (proj / "Makefile").write_text("all:\n\t@true\nlint:\n\t@true\n")
        with mock.patch.object(run_mod, "_run_cmd", fake_run_cmd):
            result = run_build(proj, "lint")
        self.assertTrue(result["ok"])
        self.assertEqual(seen["args"], ["make", "lint"])

    def test_run_build_failure_reports_not_ok(self):
        proj = Path(self.tmp) / "failproj"
        proj.mkdir()
        (proj / "Makefile").write_text("all:\n\t@exit 1\n")
        result = run_build(proj)
        self.assertFalse(result["ok"])

    def test_run_build_makefile_without_targets_raises(self):
        proj = Path(self.tmp) / "notargets"
        proj.mkdir()
        (proj / "Makefile").write_text("# nothing here\nVAR = 1\n")
        with self.assertRaises(WorkspaceError):
            run_build(proj)

    def test_run_build_pyproject_fallback(self):
        seen = {}

        def fake_run_cmd(args, cwd, timeout):
            seen["args"] = args
            return subprocess.CompletedProcess(args, 0, "built\n", "")

        proj = Path(self.tmp) / "pyproj"
        proj.mkdir()
        (proj / "pyproject.toml").write_text("[project]\nname='x'\n")
        with mock.patch("importlib.util.find_spec", return_value=object()), \
                mock.patch.object(run_mod, "_run_cmd", fake_run_cmd):
            result = run_build(proj)
        self.assertTrue(result["ok"])
        self.assertEqual(seen["args"][-2:], ["-m", "build"])

    def test_run_build_pyproject_without_build_module_raises(self):
        proj = Path(self.tmp) / "pyproj2"
        proj.mkdir()
        (proj / "pyproject.toml").write_text("[project]\nname='x'\n")
        with mock.patch("importlib.util.find_spec", return_value=None):
            with self.assertRaises(WorkspaceError) as ctx:
                run_build(proj)
        self.assertIn("build", str(ctx.exception))

    # ── commit / sync / stash ────────────────────────────────────────
    def test_commit_roundtrip(self):
        (self.repo / "hello.txt").write_text("line1\nline2!\nline3\n")
        res = self.ws.commit("second commit")
        self.assertTrue(res["committed"])
        self.assertEqual(len(res["sha"]), 40)
        self.assertEqual(self.ws.log(1)[0]["message"], "second commit")
        self.assertEqual(self.ws.status()["unstaged"], [])

    def test_commit_paths_subset(self):
        (self.repo / "hello.txt").write_text("changed\n")
        (self.repo / "other.txt").write_text("untracked\n")
        self.ws.commit("partial", paths=["hello.txt"])
        st = self.ws.status()
        self.assertEqual(st["unstaged"], [])
        self.assertIn("other.txt", st["untracked"])

    def test_commit_empty_message_raises(self):
        with self.assertRaises(WorkspaceError):
            self.ws.commit("   ")

    def test_commit_nothing_to_commit_raises(self):
        with self.assertRaises(WorkspaceError):
            self.ws.commit("empty commit")

    def test_commit_non_repo_raises(self):
        plain = Path(self.tmp) / "plain"
        plain.mkdir()
        with self.assertRaises(WorkspaceError):
            CodeWorkspace(plain).commit("x")

    def test_stash_roundtrip(self):
        (self.repo / "hello.txt").write_text("stashed-change\n")
        res = self.ws.stash_push("wip")
        self.assertTrue(res["stashed"])
        self.assertEqual((self.repo / "hello.txt").read_text(),
                         "line1\nline2\nline3\n")
        self.assertEqual(len(self.ws.stash_list()), 1)
        self.ws.stash_pop()
        self.assertEqual((self.repo / "hello.txt").read_text(), "stashed-change\n")
        self.assertEqual(self.ws.stash_list(), [])

    def test_push_pull_bare_remote(self):
        bare = Path(self.tmp) / "origin.git"
        _git("init", "-q", "--bare", str(bare), cwd=self.tmp)
        _git("remote", "add", "origin", str(bare), cwd=self.repo)
        _git("branch", "-M", "main", cwd=self.repo)
        res = self.ws.push("origin", "main")
        self.assertTrue(res["pushed"])

        clone = Path(self.tmp) / "clone"
        _git("clone", "-q", "-b", "main", str(bare), str(clone), cwd=self.tmp)
        _git("config", "user.email", "t@e.com", cwd=clone)
        _git("config", "user.name", "T", cwd=clone)
        (clone / "from-remote.txt").write_text("remote change\n")
        _git("add", ".", cwd=clone)
        _git("commit", "-q", "-m", "remote commit", cwd=clone)
        _git("push", "-q", "origin", "HEAD:main", cwd=clone)

        res = self.ws.pull("origin", "main")
        self.assertTrue(res["pulled"])
        self.assertEqual((self.repo / "from-remote.txt").read_text(),
                         "remote change\n")

    def test_push_unknown_remote_raises(self):
        with self.assertRaises(WorkspaceError):
            self.ws.push("no-such-remote")

    def test_fetch_unknown_remote_raises(self):
        with self.assertRaises(WorkspaceError):
            self.ws.fetch("no-such-remote")

    def test_create_branch_duplicate_raises(self):
        with self.assertRaises(WorkspaceError):
            self.ws.create_branch(self.ws.current_branch())

    # ── CLI end-to-end ───────────────────────────────────────────────
    def _cli(self, *argv):
        from nomorals.cmdline.dispatch import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            rc = main(list(argv))
        return rc, buf.getvalue()

    def test_cli_patch_review_apply(self):
        diff_file = Path(self.tmp) / "change.diff"
        diff_file.write_text(MODIFY_DIFF)
        rc, out = self._cli("repo", "patch", "review", str(diff_file),
                            "--root", str(self.repo))
        self.assertEqual(rc, 0, msg=out)
        self.assertIn("+1 -1", out)
        # dry run by default: nothing written
        rc, out = self._cli("repo", "patch", "apply", str(diff_file),
                            "--root", str(self.repo))
        self.assertEqual(rc, 0, msg=out)
        self.assertIn("OK", out)
        self.assertIn("line2\n", (self.repo / "hello.txt").read_text())
        # --yes writes
        rc, out = self._cli("repo", "patch", "apply", str(diff_file),
                            "--root", str(self.repo), "--yes")
        self.assertEqual(rc, 0, msg=out)
        self.assertIn("line2-changed", (self.repo / "hello.txt").read_text())

    def test_cli_patch_apply_bad_diff_exits_1(self):
        diff_file = Path(self.tmp) / "bad.diff"
        diff_file.write_text(BAD_DIFF)
        rc, out = self._cli("repo", "patch", "apply", str(diff_file),
                            "--root", str(self.repo), "--yes")
        self.assertEqual(rc, 1, msg=out)
        self.assertIn("FAILED", out)

    def test_cli_commit_and_stash(self):
        (self.repo / "hello.txt").write_text("cli change\n")
        rc, out = self._cli("repo", "commit", "-m", "cli commit",
                            "--root", str(self.repo))
        self.assertEqual(rc, 0, msg=out)
        self.assertIn("committed", out)
        self.assertEqual(self.ws.log(1)[0]["message"], "cli commit")

        (self.repo / "hello.txt").write_text("stash me\n")
        rc, out = self._cli("repo", "stash", "push", "-m", "wip",
                            "--root", str(self.repo))
        self.assertEqual(rc, 0, msg=out)
        rc, out = self._cli("repo", "stash", "list", "--root", str(self.repo))
        self.assertEqual(rc, 0, msg=out)
        self.assertIn("wip", out)
        rc, out = self._cli("repo", "stash", "pop", "--root", str(self.repo))
        self.assertEqual(rc, 0, msg=out)
        self.assertEqual((self.repo / "hello.txt").read_text(), "stash me\n")

    def test_cli_test_and_build(self):
        pkg = Path(self.tmp) / "clipkg"
        pkg.mkdir()
        (pkg / "test_s.py").write_text(
            "import unittest\n"
            "class T(unittest.TestCase):\n"
            "    def test_one(self):\n"
            "        self.assertTrue(True)\n")
        (pkg / "Makefile").write_text(".PHONY: build\nbuild:\n\t@echo cli-built\n")
        rc, out = self._cli("repo", "test", "--root", str(pkg))
        self.assertEqual(rc, 0, msg=out)
        self.assertIn("OK", out)
        rc, out = self._cli("repo", "build", "--root", str(pkg))
        self.assertEqual(rc, 0, msg=out)
        self.assertIn("OK", out)

    def test_cli_unknown_verb_exits_2(self):
        rc, out = self._cli("repo", "frobnicate", "--root", str(self.repo))
        self.assertEqual(rc, 2)

    def test_cli_error_surfaces_workspace_error(self):
        rc, out = self._cli("repo", "status", "--root", str(Path(self.tmp) / "plain-x"))
        Path(self.tmp, "plain-x").mkdir(exist_ok=True)
        rc, out = self._cli("repo", "status", "--root", str(Path(self.tmp) / "plain-x"))
        self.assertEqual(rc, 1)
        self.assertIn("error:", out)


if __name__ == "__main__":
    unittest.main()
