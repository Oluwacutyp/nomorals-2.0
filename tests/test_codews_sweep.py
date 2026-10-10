"""Sweep tests: codews module upgrade (nomorals/codews).

Covers the mined-then-built additions:
- patch: review metadata (modes/similarity/hunk headers), split_patch,
  check_patch, GNU --fuzz application with visible fuzzy reporting,
  preview fuzz/context_lines, aider-style apply_edit_blocks ladder
- run: detect_stack marker table, parse_junit_xml, structured failures,
  durations, honest counts
- workspace: renames/conflicts in status, branch upstream/ahead/behind,
  remotes, tags, merge conflicts, cherry-pick/revert, blame/history/show,
  diff_stat, add/restore/clean, stash apply/drop, worktree prune
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from nomorals.codews import (
    CodeWorkspace,
    WorkspaceError,
    apply_edit_blocks,
    apply_patch,
    check_patch,
    detect_stack,
    parse_junit_xml,
    preview_patch,
    record_patch,
    review_patch,
    run_build,
    run_tests,
    split_patch,
)
from nomorals.storage.artifacts import ARTIFACT_URI_SCHEME, ArtifactStore
from nomorals.storage.blob import BlobStore
from nomorals.storage.db import Database

GIT = shutil.which("git")

RENAME_MODE_DIFF = """\
diff --git a/old.txt b/new.txt
similarity index 66%
rename from old.txt
rename to new.txt
old mode 100644
new mode 100755
--- a/old.txt
+++ b/new.txt
@@ -1,3 +1,3 @@ def foo():
 line1
-line2
+line2-renamed
 line3
"""

MODE_ONLY_DIFF = """\
diff --git a/run.sh b/run.sh
old mode 100644
new mode 100755
"""

MULTI_DIFF = """\
diff --git a/a.py b/a.py
new file mode 100755
--- /dev/null
+++ b/a.py
@@ -0,0 +1,2 @@
+x
+y
diff --git a/b.py b/b.py
--- a/b.py
+++ b/b.py
@@ -1 +1 @@
-a
+b
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


class PatchReviewSweepTestCase(unittest.TestCase):
    def test_review_reports_modes_similarity_and_hunk_headers(self):
        rev = review_patch(RENAME_MODE_DIFF)
        (f,) = rev["files"]
        self.assertEqual(f["status"], "renamed")
        self.assertEqual(f["rename_from"], "old.txt")
        self.assertEqual(f["old_mode"], "100644")
        self.assertEqual(f["new_mode"], "100755")
        self.assertEqual(f["similarity"], 66)
        self.assertFalse(f["is_added"])
        self.assertFalse(f["is_removed"])
        self.assertFalse(f["is_symlink"])
        self.assertFalse(f["is_binary"])
        (h,) = f["hunk_headers"]
        self.assertEqual(h["old_start"], 1)
        self.assertEqual(h["old_lines"], 3)
        self.assertEqual(h["new_start"], 1)
        self.assertEqual(h["new_lines"], 3)
        self.assertEqual(h["section"], "def foo():")
        self.assertEqual(f["hunks"], 1)

    def test_review_mode_only_section_has_no_hunks(self):
        rev = review_patch(MODE_ONLY_DIFF)
        (f,) = rev["files"]
        self.assertEqual(f["status"], "modified")
        self.assertEqual(f["old_mode"], "100644")
        self.assertEqual(f["new_mode"], "100755")
        self.assertEqual(f["hunks"], 0)
        self.assertEqual(f["additions"], 0)

    def test_review_symlink_flag(self):
        d = ("diff --git a/l b/l\nnew file mode 120000\n--- /dev/null\n"
             "+++ b/l\n@@ -0,0 +1 @@\n+target\n")
        (f,) = review_patch(d)["files"]
        self.assertTrue(f["is_symlink"])
        self.assertTrue(f["is_added"])

    def test_split_patch_round_trips_per_file(self):
        parts = split_patch(MULTI_DIFF)
        self.assertEqual(sorted(parts), ["a.py", "b.py"])
        # each part applies on its own
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        (tmp / "b.py").write_text("a\n")
        res = apply_patch(parts["a.py"], dry_run=False, root=tmp)
        self.assertTrue(res[0]["ok"])
        self.assertEqual((tmp / "a.py").read_text(), "x\ny\n")
        res = apply_patch(parts["b.py"], dry_run=False, root=tmp)
        self.assertTrue(res[0]["ok"])
        self.assertEqual((tmp / "b.py").read_text(), "b\n")

    def test_check_patch_summary(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        (tmp / "b.py").write_text("a\n")
        chk = check_patch(MULTI_DIFF, root=tmp)
        self.assertTrue(chk["ok"])
        self.assertEqual(chk["totals"], {"files": 2, "ok": 2, "failed": 0})
        self.assertFalse((tmp / "a.py").exists())  # dry run: untouched
        bad = "--- a/nope.py\n+++ b/nope.py\n@@ -1 +1 @@\n-x\n+y\n"
        chk = check_patch(bad, root=tmp)
        self.assertFalse(chk["ok"])
        self.assertEqual(chk["totals"]["failed"], 1)
        self.assertIn("nope.py", chk["files"][0]["error"])


class PatchFuzzSweepTestCase(unittest.TestCase):
    DIFF = """\
--- a/f.py
+++ b/f.py
@@ -1,5 +1,5 @@
 l1
-l2
+l2x
 l3
 l4
 l5
"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_exact_rejects_drifted_context(self):
        (self.tmp / "f.py").write_text("L1-DRIFT\nl2\nl3\nl4\nL5-DRIFT\n")
        (r,) = apply_patch(self.DIFF, dry_run=True, root=self.tmp)
        self.assertFalse(r["ok"])
        self.assertFalse(r["fuzzy"])

    def test_fuzz_applies_edge_drift_and_reports_it(self):
        (self.tmp / "f.py").write_text("L1-DRIFT\nl2\nl3\nl4\nL5-DRIFT\n")
        (r,) = apply_patch(self.DIFF, dry_run=True, root=self.tmp, fuzz=1)
        self.assertTrue(r["ok"])
        self.assertTrue(r["fuzzy"])  # never a silent fuzzy success
        (r,) = apply_patch(self.DIFF, dry_run=False, root=self.tmp, fuzz=1)
        self.assertTrue(r["ok"])
        self.assertEqual((self.tmp / "f.py").read_text(),
                         "L1-DRIFT\nl2x\nl3\nl4\nL5-DRIFT\n")

    def test_fuzz_does_not_save_middle_drift(self):
        # GNU semantics: fuzz only ignores leading/trailing context lines.
        (self.tmp / "f.py").write_text("l1\nl2\nMID-DRIFT\nl4\nl5\n")
        for level in (1, 2):
            (r,) = apply_patch(self.DIFF, dry_run=True, root=self.tmp,
                               fuzz=level)
            self.assertFalse(r["ok"], f"fuzz={level} should not apply")
            self.assertFalse(r["fuzzy"])
            self.assertIn("fuzz", r["error"])

    def test_negative_fuzz_rejected(self):
        with self.assertRaises(WorkspaceError):
            apply_patch(self.DIFF, root=self.tmp, fuzz=-1)

    def test_preview_reports_fuzzy_and_context_lines(self):
        (self.tmp / "f.py").write_text("L1-DRIFT\nl2\nl3\nl4\nL5-DRIFT\n")
        pv = preview_patch(self.DIFF, "f.py", root=self.tmp, fuzz=1,
                           context_lines=1)
        self.assertTrue(pv["fuzzy"])
        self.assertIn("l2x", pv["after"])
        self.assertIn("L1-DRIFT", pv["after"])
        pv2 = preview_patch(self.DIFF, "f.py", root=self.tmp, fuzz=1,
                            context_lines=0)
        self.assertLessEqual(len(pv2["after"].splitlines()),
                             len(pv["after"].splitlines()))


class EditBlocksSweepTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _block(self, path, search, replace):
        return (f"{path}\n```\n<<<<<<< SEARCH\n{search}\n=======\n"
                f"{replace}\n>>>>>>> REPLACE\n```\n")

    def test_exact_block(self):
        (self.tmp / "g.py").write_text("def g():\n    return 1\n")
        (r,) = apply_edit_blocks(
            self._block("g.py", "    return 1", "    return 2"),
            dry_run=False, root=self.tmp)
        self.assertTrue(r["ok"])
        self.assertEqual(r["strategy"], "exact")
        self.assertIn("return 2", (self.tmp / "g.py").read_text())

    def test_trailing_ws_block(self):
        (self.tmp / "g.py").write_text('x = "a"   \n')
        (r,) = apply_edit_blocks(
            self._block("g.py", 'x = "a"', 'x = "b"'),
            dry_run=False, root=self.tmp)
        self.assertTrue(r["ok"])
        self.assertEqual(r["strategy"], "trailing-ws")
        self.assertEqual((self.tmp / "g.py").read_text(), 'x = "b"\n')

    def test_indent_drift_block(self):
        (self.tmp / "g.py").write_text("def g():\n        return 1\n")
        (r,) = apply_edit_blocks(
            self._block("g.py", "    return 1", "    return 2"),
            dry_run=False, root=self.tmp)
        self.assertTrue(r["ok"])
        self.assertEqual(r["strategy"], "indent-drift")
        self.assertEqual((self.tmp / "g.py").read_text(),
                         "def g():\n        return 2\n")

    def test_fuzzy_block(self):
        (self.tmp / "g.py").write_text("def greet(name):\n    print(name)\n")
        (r,) = apply_edit_blocks(
            self._block("g.py", "def greet(naem):\n    print(name)",
                        "def greet(name):\n    print(name, '!')"),
            dry_run=False, root=self.tmp)
        self.assertTrue(r["ok"])
        self.assertEqual(r["strategy"], "fuzzy")

    def test_create_block(self):
        (r,) = apply_edit_blocks(
            self._block("new.py", "", "hello\n"),
            dry_run=False, root=self.tmp)
        self.assertTrue(r["ok"])
        self.assertEqual(r["strategy"], "create")
        self.assertEqual((self.tmp / "new.py").read_text(), "hello\n")

    def test_create_refused_when_file_exists(self):
        (self.tmp / "new.py").write_text("x\n")
        (r,) = apply_edit_blocks(self._block("new.py", "", "hello\n"),
                                 dry_run=True, root=self.tmp)
        self.assertFalse(r["ok"])
        self.assertIn("already exists", r["error"])

    def test_ambiguous_block_refused(self):
        (self.tmp / "g.py").write_text("same\nsame\n")
        (r,) = apply_edit_blocks(self._block("g.py", "same", "diff"),
                                 dry_run=True, root=self.tmp)
        self.assertFalse(r["ok"])
        self.assertIn("ambiguous", r["error"])
        self.assertEqual((self.tmp / "g.py").read_text(), "same\nsame\n")

    def test_no_match_block(self):
        (self.tmp / "g.py").write_text("alpha\nbeta\n")
        (r,) = apply_edit_blocks(self._block("g.py", "zzz", "q"),
                                 dry_run=True, root=self.tmp)
        self.assertFalse(r["ok"])
        self.assertIn("does not match", r["error"])

    def test_no_blocks_raises(self):
        with self.assertRaises(WorkspaceError):
            apply_edit_blocks("just some prose", root=self.tmp)

    def test_path_escape_refused(self):
        (r,) = apply_edit_blocks(
            self._block("../evil.py", "a", "b"), dry_run=True, root=self.tmp)
        self.assertFalse(r["ok"])
        self.assertIn("outside workspace", r["error"])

    def test_dry_run_writes_nothing(self):
        (self.tmp / "g.py").write_text("one\n")
        (r,) = apply_edit_blocks(self._block("g.py", "one", "two"),
                                 dry_run=True, root=self.tmp)
        self.assertTrue(r["ok"])
        self.assertEqual((self.tmp / "g.py").read_text(), "one\n")


class RecordPatchSweepTestCase(unittest.TestCase):
    def test_record_includes_sha256(self):
        store = make_store(self)
        uri = record_patch(store, MULTI_DIFF, mission_id="m1")
        self.assertTrue(uri.startswith(ARTIFACT_URI_SCHEME))
        art = store.resolve(uri)
        self.assertIsNotNone(art)
        self.assertEqual(len(art.metadata["sha256"]), 64)
        self.assertEqual(art.metadata["totals"]["files"], 2)
        self.assertEqual(art.metadata["files"], ["a.py", "b.py"])


class RunSweepTestCase(unittest.TestCase):
    def _proj(self, files):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        for name, content in files.items():
            p = tmp / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        return tmp

    def test_detect_python_pytest(self):
        proj = self._proj({"pyproject.toml": "[tool.pytest.ini_options]\n",
                           "tests/test_a.py": "def test_a(): pass\n"})
        stack = detect_stack(proj)
        self.assertEqual(stack["language"], "python")
        self.assertEqual(stack["test"]["runner"], "pytest")

    def test_detect_node(self):
        proj = self._proj({"package.json":
                           '{"scripts": {"test": "jest", "build": "tsc"},'
                           ' "devDependencies": {"jest": "^29"}}'})
        stack = detect_stack(proj)
        self.assertEqual(stack["language"], "node")
        self.assertEqual(stack["test"]["runner"], "jest")
        self.assertEqual(stack["build"]["runner"], "npm run build")

    def test_detect_go_and_rust(self):
        proj = self._proj({"go.mod": "module x\n"})
        stack = detect_stack(proj)
        self.assertEqual(stack["test"]["runner"], "go test")
        self.assertEqual(stack["build"]["runner"], "go build")
        proj = self._proj({"Cargo.toml": "[package]\nname='x'\n"})
        stack = detect_stack(proj)
        self.assertEqual(stack["test"]["runner"], "cargo test")

    def test_detect_make_test(self):
        proj = self._proj({"Makefile": "test:\n\techo hi\n"})
        stack = detect_stack(proj)
        self.assertEqual(stack["test"]["runner"], "make test")
        self.assertIsNotNone(stack["build"])

    def test_detect_nothing(self):
        proj = self._proj({"README.md": "hi\n"})
        stack = detect_stack(proj)
        self.assertIsNone(stack["test"])
        self.assertIsNone(stack["build"])
        # legacy contract: marker-less dirs still fall back to pytest when
        # importable (pre-existing test_run_tests_selector_passed_through)
        seen = {}

        def fake_run_cmd(args, cwd, timeout, env=None):
            seen["args"] = args

            class P:
                returncode = 0
                stdout = ""
                stderr = ""
            return P()

        from nomorals.codews import run as run_mod
        real = run_mod._run_cmd
        run_mod._run_cmd = fake_run_cmd
        try:
            r = run_tests(proj)
        finally:
            run_mod._run_cmd = real
        self.assertEqual(r["runner"], "pytest")
        with self.assertRaises(WorkspaceError):
            run_build(proj)

    def test_run_tests_structured_failures(self):
        proj = self._proj({"tests/test_x.py":
                           "def test_ok():\n    assert True\n\n"
                           "def test_bad():\n    assert 1 == 2, 'boom'\n"})
        r = run_tests(proj)
        self.assertEqual(r["runner"], "pytest")
        self.assertFalse(r["ok"])
        self.assertEqual(r["passed"], 1)
        self.assertEqual(r["failed"], 1)
        self.assertEqual(len(r["failed_tests"]), 1)
        self.assertIn("test_bad", r["failed_tests"][0])
        self.assertEqual(r["failures"][0]["message"], "AssertionError: boom")
        self.assertGreaterEqual(r["duration_s"], 0)
        self.assertTrue(r["command"][0].endswith("python3")
                        or "python" in r["command"][0])

    def test_run_tests_make_target(self):
        proj = self._proj({"Makefile": "test:\n\t@echo ok\n"})
        r = run_tests(proj)
        self.assertEqual(r["runner"], "make test")
        self.assertTrue(r["ok"])

    def test_run_build_make(self):
        proj = self._proj({"Makefile": "all:\n\t@echo built\n"})
        r = run_build(proj)
        self.assertTrue(r["ok"])
        self.assertIn("built", r["output"])
        self.assertEqual(r["runner"], "make all")

    def test_run_build_target_override(self):
        proj = self._proj({"Makefile": "all:\n\t@echo all\n\nlint:\n\t@echo lint\n"})
        r = run_build(proj, target="lint")
        self.assertTrue(r["ok"])
        self.assertIn("lint", r["output"])

    def test_run_tests_timeout_and_env(self):
        proj = self._proj({"tests/test_e.py":
                           "import os\ndef test_e():\n"
                           "    assert os.environ.get('FOO') == 'bar'\n",
                           "tests/test_slow.py":
                           "import time\ndef test_slow():\n"
                           "    time.sleep(30)\n"})
        r = run_tests(proj, selector="tests/test_e.py", env={"FOO": "bar"})
        self.assertTrue(r["ok"])
        with self.assertRaises(WorkspaceError) as ctx:
            run_tests(proj, selector="tests/test_slow.py", timeout=5)
        self.assertIn("timed out", str(ctx.exception))

    def test_parse_junit_xml(self):
        xml = """<?xml version="1.0"?>
<testsuites><testsuite name="s" tests="3" time="1.5">
<testcase classname="t" name="a" time="0.1"/>
<testcase classname="t" name="b" time="0.2">
<failure message="bad">AssertionError: bad
line2</failure></testcase>
<testcase classname="t" name="c" time="0.3"><skipped/></testcase>
</testsuite></testsuites>"""
        p = Path(tempfile.mkdtemp()) / "j.xml"
        self.addCleanup(shutil.rmtree, p.parent, ignore_errors=True)
        p.write_text(xml)
        res = parse_junit_xml(p)
        self.assertEqual(res["tests"], 3)
        self.assertEqual(res["passed"], 1)
        self.assertEqual(res["failed"], 1)
        self.assertEqual(res["skipped"], 1)
        by_name = {c["name"]: c for c in res["cases"]}
        self.assertEqual(by_name["b"]["outcome"], "failed")
        self.assertEqual(by_name["b"]["message"], "bad")
        self.assertEqual(by_name["b"]["nodeid"], "t::b")
        self.assertEqual(by_name["c"]["outcome"], "skipped")


class WorkspaceSweepTestCase(unittest.TestCase):
    def setUp(self):
        if not GIT:
            self.skipTest("git binary not found on PATH")
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.repo = Path(self.tmp) / "repo"
        self.repo.mkdir()
        _git("init", "-b", "main", cwd=self.repo)
        _git("config", "user.email", "t@t", cwd=self.repo)
        _git("config", "user.name", "t", cwd=self.repo)
        (self.repo / "a.txt").write_text("hi\n")
        _git("add", "-A", cwd=self.repo)
        _git("commit", "-qm", "init", cwd=self.repo)
        self.ws = CodeWorkspace(self.repo)

    def _conflict(self):
        self.ws.create_branch("feat")
        self.ws.switch_branch("feat")
        (self.repo / "c.txt").write_text("feat\n")
        self.ws.commit("feat")
        self.ws.switch_branch("main")
        (self.repo / "c.txt").write_text("main\n")
        self.ws.commit("main")

    def test_status_renames_and_conflicts(self):
        _git("mv", "a.txt", "b.txt", cwd=self.repo)
        st = self.ws.status()
        self.assertEqual(st["renames"], [{"from": "a.txt", "to": "b.txt"}])
        self.assertEqual(st["staged"], ["b.txt"])
        self.assertEqual(st["conflicts"], [])
        self.assertTrue(self.ws.is_dirty())
        self.ws.commit("rename")
        self.assertFalse(self.ws.is_dirty())

    def test_branches_carry_upstream_tracking(self):
        for b in self.ws.branches():
            self.assertIn("upstream", b)
            self.assertIn("ahead", b)
            self.assertIn("behind", b)
        self.assertTrue(any(b["current"] and b["name"] == "main"
                            for b in self.ws.branches()))

    def test_delete_and_rename_branch(self):
        self.ws.create_branch("tmp1")
        self.ws.rename_branch("tmp1", "tmp2")
        self.assertIn("tmp2", [b["name"] for b in self.ws.branches()])
        self.ws.delete_branch("tmp2")
        self.assertNotIn("tmp2", [b["name"] for b in self.ws.branches()])

    def test_add_restore_clean(self):
        (self.repo / "a.txt").write_text("changed\n")
        self.ws.add(["a.txt"])
        self.assertIn("a.txt", self.ws.status()["staged"])
        self.ws.restore(["a.txt"], staged=True)
        self.assertNotIn("a.txt", self.ws.status()["staged"])
        (self.repo / "junk.tmp").write_text("x")
        dry = self.ws.clean()
        self.assertTrue(dry["dry_run"])
        self.assertIn("junk.tmp", dry["removed"])
        self.assertTrue((self.repo / "junk.tmp").exists())
        self.ws.clean(force=True)
        self.assertFalse((self.repo / "junk.tmp").exists())

    def test_merge_conflict_reports_files(self):
        self._conflict()
        with self.assertRaises(WorkspaceError) as ctx:
            self.ws.merge_branch("feat")
        self.assertIn("c.txt", str(ctx.exception))
        self.assertEqual(self.ws.conflicts(), ["c.txt"])
        self.ws.abort_merge()
        self.assertEqual(self.ws.conflicts(), [])

    def test_merge_clean(self):
        self.ws.create_branch("feat")
        self.ws.switch_branch("feat")
        (self.repo / "d.txt").write_text("d\n")
        self.ws.commit("add d")
        self.ws.switch_branch("main")
        r = self.ws.merge_branch("feat")
        self.assertTrue(r["merged"])
        self.assertTrue((self.repo / "d.txt").exists())

    def test_cherry_pick_and_revert(self):
        self.ws.create_branch("x")
        self.ws.switch_branch("x")
        (self.repo / "e.txt").write_text("e\n")
        self.ws.commit("add e")
        sha = self.ws.log(1)[0]["sha"]
        self.ws.switch_branch("main")
        self.assertTrue(self.ws.cherry_pick(sha)["cherry_picked"])
        self.assertTrue((self.repo / "e.txt").exists())
        self.assertTrue(self.ws.revert("HEAD")["reverted"])
        self.assertFalse((self.repo / "e.txt").exists())

    def test_amend(self):
        (self.repo / "a.txt").write_text("v2\n")
        self.ws.commit("v2")
        before = self.ws.log(1)[0]["sha"]
        r = self.ws.amend("v2 amended")
        self.assertTrue(r["amended"])
        self.assertNotEqual(r["sha"], before)
        self.assertEqual(self.ws.log(1)[0]["message"], "v2 amended")

    def test_remotes(self):
        self.assertEqual(self.ws.remotes(), [])
        self.ws.remote_add("fork", "https://example.com/x.git")
        self.ws.remote_set_url("fork", "https://example.com/y.git")
        self.assertEqual(self.ws.remotes(),
                         [{"name": "fork", "url": "https://example.com/y.git"}])
        self.ws.remote_remove("fork")
        self.assertEqual(self.ws.remotes(), [])

    def test_tags(self):
        self.assertEqual(self.ws.tags(), [])
        self.ws.create_tag("v1")
        self.ws.create_tag("v2", message="second")
        names = [t["name"] for t in self.ws.tags()]
        self.assertEqual(names, ["v1", "v2"])
        self.assertTrue(all(t["sha"] for t in self.ws.tags()))
        self.ws.delete_tag("v1")
        self.assertEqual([t["name"] for t in self.ws.tags()], ["v2"])

    def test_stash_apply_keeps_entry(self):
        (self.repo / "a.txt").write_text("wip\n")
        self.ws.stash_push("wip")
        self.assertEqual(len(self.ws.stash_list()), 1)
        self.ws.stash_apply()
        self.assertEqual(len(self.ws.stash_list()), 1)  # kept
        self.assertEqual((self.repo / "a.txt").read_text(), "wip\n")
        self.ws.stash_drop()
        self.assertEqual(self.ws.stash_list(), [])

    def test_diff_two_refs_and_paths(self):
        (self.repo / "a.txt").write_text("v2\n")
        self.ws.commit("v2")
        d = self.ws.diff("HEAD~1", "HEAD")
        self.assertIn("v2", d)
        d = self.ws.diff(paths=["a.txt"])
        self.assertEqual(d, "")  # clean tree: no diff
        (self.repo / "a.txt").write_text("v3\n")
        d = self.ws.diff(paths=["a.txt"])
        self.assertIn("v3", d)

    def test_diff_stat(self):
        (self.repo / "a.txt").write_text("l1\nl2\n")
        stat = self.ws.diff_stat()
        self.assertEqual(stat["totals"]["files"], 1)
        self.assertEqual(stat["files"][0]["path"], "a.txt")
        self.assertGreater(stat["files"][0]["additions"], 0)

    def test_log_paths_and_stats(self):
        (self.repo / "a.txt").write_text("v2\n")
        self.ws.commit("touch a")
        (self.repo / "z.txt").write_text("z\n")
        self.ws.commit("add z")
        commits = self.ws.log(5, paths=["z.txt"], stats=True)
        self.assertEqual(len(commits), 1)
        self.assertEqual(commits[0]["message"], "add z")
        self.assertEqual(commits[0]["files"][0]["path"], "z.txt")

    def test_file_history_follows_rename(self):
        _git("mv", "a.txt", "b.txt", cwd=self.repo)
        _git("commit", "-qm", "rename", cwd=self.repo)
        hist = self.ws.file_history("b.txt")
        self.assertEqual(len(hist), 2)
        self.assertEqual(hist[0]["message"], "rename")
        self.assertEqual(hist[1]["message"], "init")

    def test_blame(self):
        (self.repo / "a.txt").write_text("one\ntwo\n")
        self.ws.commit("two lines")
        blame = self.ws.blame("a.txt")
        self.assertEqual(len(blame), 2)
        self.assertEqual(blame[0]["line"], 1)
        self.assertEqual(blame[0]["content"], "one")
        self.assertEqual(blame[0]["author"], "t")
        self.assertEqual(len(blame[0]["sha"]), 40)

    def test_show(self):
        self.assertEqual(self.ws.show("HEAD", "a.txt"), "hi\n")

    def test_worktree_list_keys_and_prune(self):
        for wt in self.ws.worktree_list():
            self.assertIn("locked", wt)
            self.assertIn("prunable", wt)
            self.assertIn("branch", wt)
        self.assertTrue(self.ws.worktree_prune()["pruned"])

    def test_push_set_upstream_flag(self):
        # no remote: fails fast with a clear error either way
        with self.assertRaises(WorkspaceError):
            self.ws.push("nonexistent-remote")


if __name__ == "__main__":
    unittest.main()
