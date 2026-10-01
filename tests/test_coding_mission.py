"""Acceptance tests for the coding-agent mission upgrades (wave A stream 1):

- git snapshot()/rollback(): a failed mission restores the workdir, both
  from a clean repo and from a dirty (stashed) repo; safe when not a repo
- bisect_regression(): finds the culprit commit in a synthetic repo
- apply_unified_diff(): multi-file/multi-hunk apply, all-or-nothing per
  file on bad hunks, new/deleted files, path-escape rejection, and the
  ```diff protocol wired into CodingAgent
- repo map: bounded, informative, and never breaks _plan_files
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.bisect import BisectError, bisect_regression
from nomorals.agents.coding import (
    CodingAgent, _parse_patch_block, build_repo_map,
)
from nomorals.agents.patch import apply_unified_diff, parse_unified_diff


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True,
                   capture_output=True, timeout=60)


def _git_out(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True,
                          capture_output=True, text=True,
                          timeout=60).stdout.strip()


class ScriptedRouter:
    """Deterministic stand-in for the LLM: canned responses in call order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.messages = []

    def chat(self, messages, params=None):
        self.calls += 1
        self.messages.append(messages)
        text = self._responses.pop(0) if self._responses else ""
        return SimpleNamespace(ok=bool(text), text=text,
                               error="" if text else "empty")


class StubDB:
    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute(
            "CREATE TABLE coding_log (id TEXT, task TEXT, filename TEXT,"
            " attempt INT, exit_code INT, timed_out INT, stdout TEXT,"
            " stderr TEXT, code TEXT, created_at REAL)")

    def execute(self, sql, params=()):
        self.conn.execute(sql, params)
        self.conn.commit()

    def query(self, sql, params=()):
        cur = self.conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


class StubContext:
    def __init__(self, router):
        self.router = router
        self.db = StubDB()
        self.settings = SimpleNamespace(reasoning_mode="off")


def _json_block(payload):
    return "```json\n" + json.dumps(payload) + "\n```"


class FixtureRepo:
    def __init__(self, with_git=False):
        self.root = Path(tempfile.mkdtemp(prefix="mission_"))
        if with_git:
            _git("init", "-q", cwd=self.root)
            _git("config", "user.email", "t@example.com", cwd=self.root)
            _git("config", "user.name", "T", cwd=self.root)

    def write(self, rel, content):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return p

    def commit_all(self, msg="init"):
        _git("add", ".", cwd=self.root)
        _git("commit", "-qm", msg, cwd=self.root)

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)


FAIL_ACCEPT = (f'"{sys.executable}" -c "import sys; sys.exit(1)"')
OK_ACCEPT = (f'"{sys.executable}" -c "import sys; sys.exit(0)"')


def _plan_response(*paths):
    return _json_block({"files": [
        {"path": p, "why": "test", "new_file": False} for p in paths]})


# ── rollback ──────────────────────────────────────────────────────────────

class RollbackTests(unittest.TestCase):
    def setUp(self):
        self.fx = FixtureRepo(with_git=True)
        self.addCleanup(self.fx.cleanup)
        self.fx.write("a.py", "X = 1\n")
        self.fx.commit_all()

    def _failing_agent(self):
        router = ScriptedRouter([
            _plan_response("a.py"),
            _json_block({"edits": [
                {"old_text": "X = 1", "new_text": "X = 2"}]}),
        ])
        return CodingAgent(StubContext(router), root=str(self.fx.root))

    def test_failed_mission_restores_clean_repo(self):
        agent = self._failing_agent()
        result = agent.run("bump X", accept=FAIL_ACCEPT, max_iterations=1)
        self.assertFalse(result.ok)
        self.assertEqual((self.fx.root / "a.py").read_text(), "X = 1\n")
        self.assertEqual(_git_out("status", "--porcelain", cwd=self.fx.root),
                         "")
        self.assertEqual(_git_out("stash", "list", cwd=self.fx.root), "")

    def test_failed_mission_restores_stashed_dirty_state(self):
        # Pre-mission dirty state: must survive the mission via the stash.
        self.fx.write("a.py", "X = 9\n")
        agent = self._failing_agent()
        result = agent.run("bump X", accept=FAIL_ACCEPT, max_iterations=1)
        self.assertFalse(result.ok)
        self.assertEqual((self.fx.root / "a.py").read_text(), "X = 9\n")
        self.assertEqual(_git_out("stash", "list", cwd=self.fx.root), "")
        self.assertIn("M a.py",
                      _git_out("status", "--porcelain", cwd=self.fx.root))

    def test_failed_mission_only_reverts_touched_files(self):
        self.fx.write("b.py", "Y = 1\n")
        self.fx.commit_all("add b")
        self.fx.write("b.py", "Y = 99\n")  # dirty, but the mission won't touch it
        agent = self._failing_agent()
        result = agent.run("bump X", accept=FAIL_ACCEPT, max_iterations=1)
        self.assertFalse(result.ok)
        # b.py's dirty state is restored from the stash untouched by rollback
        self.assertEqual((self.fx.root / "b.py").read_text(), "Y = 99\n")
        self.assertEqual((self.fx.root / "a.py").read_text(), "X = 1\n")

    def test_failed_mission_removes_mission_created_files(self):
        router = ScriptedRouter([
            _json_block({"files": [
                {"path": "brand_new.py", "why": "new", "new_file": True}]}),
            "```python\nprint('hello')\n```\n",
        ])
        agent = CodingAgent(StubContext(router), root=str(self.fx.root))
        result = agent.run("create a file", accept=FAIL_ACCEPT,
                           max_iterations=1, filename="brand_new.py")
        self.assertFalse(result.ok)
        self.assertFalse((self.fx.root / "brand_new.py").exists())

    def test_non_repo_mission_fails_without_crash(self):
        fx = FixtureRepo(with_git=False)
        self.addCleanup(fx.cleanup)
        fx.write("a.py", "X = 1\n")
        router = ScriptedRouter([
            _plan_response("a.py"),
            _json_block({"edits": [
                {"old_text": "X = 1", "new_text": "X = 2"}]}),
        ])
        agent = CodingAgent(StubContext(router), root=str(fx.root))
        result = agent.run("bump X", accept=FAIL_ACCEPT, max_iterations=1)
        self.assertFalse(result.ok)  # failed, but no exception escaped
        self.assertIsNone(agent.snapshot())  # not a repo: explicit None
        self.assertTrue(agent.rollback())    # no-op, still True

    def test_snapshot_rollback_on_demand(self):
        self.fx.write("a.py", "X = 9\n")  # dirty
        agent = CodingAgent(StubContext(ScriptedRouter([])),
                            root=str(self.fx.root))
        snap = agent.snapshot()
        self.assertTrue(snap["stashed"])
        self.fx.write("a.py", "X = 2\n")  # the "mission" change
        agent._mission_touched = ["a.py"]
        self.assertTrue(agent.rollback())
        self.assertEqual((self.fx.root / "a.py").read_text(), "X = 9\n")
        self.assertEqual(_git_out("stash", "list", cwd=self.fx.root), "")

    def test_successful_mission_is_not_rolled_back(self):
        agent = self._failing_agent()
        result = agent.run("bump X", accept=OK_ACCEPT, max_iterations=1)
        self.assertTrue(result.ok)
        self.assertEqual((self.fx.root / "a.py").read_text(), "X = 2\n")


# ── bisect ────────────────────────────────────────────────────────────────

class BisectTests(unittest.TestCase):
    def setUp(self):
        self.fx = FixtureRepo(with_git=True)
        self.addCleanup(self.fx.cleanup)
        self.fx.write("flag.txt", "ok\n")
        self.fx.commit_all("c1 good")
        self.good = _git_out("rev-parse", "HEAD", cwd=self.fx.root)
        self.fx.write("other.txt", "v2\n")
        self.fx.commit_all("c2 good")
        self.fx.write("other.txt", "v3\n")
        self.fx.commit_all("c3 good")
        self.fx.write("flag.txt", "bad\n")  # ← the regression
        self.fx.commit_all("c4 BREAKS the flag")
        self.culprit = _git_out("rev-parse", "HEAD", cwd=self.fx.root)
        self.fx.write("other.txt", "v5\n")
        self.fx.commit_all("c5 bad")
        self.test_cmd = [
            sys.executable, "-c",
            "import sys; sys.exit(0 if open('flag.txt').read().strip()"
            "=='ok' else 1)",
        ]

    def test_finds_culprit_commit(self):
        result = bisect_regression(
            self.fx.root, self.good, "HEAD", self.test_cmd, timeout=30)
        self.assertIsNotNone(result)
        self.assertEqual(result["hash"], self.culprit)
        self.assertIn("BREAKS", result["subject"])
        self.assertTrue(result["author"])
        self.assertTrue(result["date"])
        # the repo is left exactly as found: no bisect state left behind
        self.assertFalse((self.fx.root / ".git" / "BISECT_LOG").exists())

    def test_head_restored_after_bisect(self):
        before = _git_out("rev-parse", "HEAD", cwd=self.fx.root)
        bisect_regression(self.fx.root, self.good, "HEAD", self.test_cmd,
                          timeout=30)
        self.assertEqual(_git_out("rev-parse", "HEAD", cwd=self.fx.root),
                         before)

    def test_not_a_repo_raises(self):
        fx = FixtureRepo(with_git=False)
        self.addCleanup(fx.cleanup)
        with self.assertRaises(BisectError):
            bisect_regression(fx.root, "a", "b", self.test_cmd, timeout=30)

    def test_bad_ref_already_good_returns_none(self):
        result = bisect_regression(
            self.fx.root, self.good, self.good, self.test_cmd, timeout=30)
        self.assertIsNone(result)

    def test_dirty_tree_raises(self):
        self.fx.write("flag.txt", "dirty\n")
        with self.assertRaises(BisectError):
            bisect_regression(self.fx.root, self.good, "HEAD",
                              self.test_cmd, timeout=30)


# ── unified-diff applier ──────────────────────────────────────────────────

class ApplyDiffTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="patch_"))
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))

    def test_multi_file_multi_hunk(self):
        (self.root / "a.py").write_text("x = 1\ny = 2\nz = 3\n")
        (self.root / "b.py").write_text("p = 'a'\nq = 'b'\n")
        patch = (
            "--- a/a.py\n+++ b/a.py\n"
            "@@ -1,3 +1,3 @@\n-x = 1\n+x = 10\n y = 2\n-z = 3\n+z = 30\n"
            "--- a/b.py\n+++ b/b.py\n"
            "@@ -1,2 +1,2 @@\n-p = 'a'\n+p = 'A'\n q = 'b'\n"
            "@@ -2,1 +2,2 @@\n q = 'b'\n+r = 'new'\n"
        )
        result = apply_unified_diff(patch, self.root)
        self.assertEqual(result["failed_hunks"], [])
        self.assertCountEqual(result["applied_files"], ["a.py", "b.py"])
        self.assertEqual((self.root / "a.py").read_text(),
                         "x = 10\ny = 2\nz = 30\n")
        self.assertEqual((self.root / "b.py").read_text(),
                         "p = 'A'\nq = 'b'\nr = 'new'\n")

    def test_bad_hunk_leaves_file_untouched(self):
        (self.root / "a.py").write_text("x = 1\n")
        (self.root / "b.py").write_text("y = 2\n")
        before_b = (self.root / "b.py").read_bytes()
        patch = (
            "--- a/a.py\n+++ b/a.py\n"
            "@@ -1 +1 @@\n-x = 1\n+x = 10\n"
            "--- a/b.py\n+++ b/b.py\n"
            "@@ -1 +1 @@\n-y = WRONG\n+y = 20\n"
        )
        result = apply_unified_diff(patch, self.root)
        # a.py applied; b.py all-or-nothing untouched
        self.assertEqual(result["applied_files"], ["a.py"])
        self.assertEqual(len(result["failed_hunks"]), 1)
        self.assertEqual(result["failed_hunks"][0]["file"], "b.py")
        self.assertEqual(result["failed_hunks"][0]["hunk"], 1)
        self.assertEqual((self.root / "b.py").read_bytes(), before_b)

    def test_second_hunk_failure_is_all_or_nothing(self):
        (self.root / "a.py").write_text("l1\nl2\nl3\nl4\n")
        before = (self.root / "a.py").read_bytes()
        patch = (
            "--- a/a.py\n+++ b/a.py\n"
            "@@ -1,2 +1,2 @@\n-l1\n+L1\n l2\n"
            "@@ -3,2 +3,2 @@\n-l3-WRONG\n+L3\n l4\n"
        )
        result = apply_unified_diff(patch, self.root)
        self.assertEqual(result["applied_files"], [])
        self.assertEqual(len(result["failed_hunks"]), 1)
        self.assertEqual(result["failed_hunks"][0]["hunk"], 2)
        self.assertEqual((self.root / "a.py").read_bytes(), before)

    def test_new_and_deleted_files(self):
        (self.root / "gone.py").write_text("bye = 1\n")
        patch = (
            "--- /dev/null\n+++ b/fresh.py\n"
            "@@ -0,0 +1,2 @@\n+hello = 1\n+world = 2\n"
            "--- a/gone.py\n+++ /dev/null\n"
            "@@ -1 +0,0 @@\n-bye = 1\n"
        )
        result = apply_unified_diff(patch, self.root)
        self.assertEqual(result["failed_hunks"], [])
        self.assertCountEqual(result["applied_files"],
                              ["fresh.py", "gone.py"])
        self.assertEqual((self.root / "fresh.py").read_text(),
                         "hello = 1\nworld = 2\n")
        self.assertFalse((self.root / "gone.py").exists())

    def test_path_escape_rejected(self):
        (self.root / "a.py").write_text("x = 1\n")
        patch = ("--- a/a.py\n+++ b/../../evil.py\n"
                 "@@ -1 +1 @@\n-x = 1\n+x = 2\n")
        result = apply_unified_diff(patch, self.root)
        self.assertEqual(result["applied_files"], [])
        self.assertTrue(result["failed_hunks"])
        self.assertFalse((self.root.parent / "evil.py").exists())

    def test_empty_patch_is_empty_result(self):
        result = apply_unified_diff("no diff here", self.root)
        self.assertEqual(result, {"applied_files": [], "failed_hunks": []})

    def test_parse_patch_block(self):
        text = ("some prose\n```diff\n--- a/x.py\n+++ b/x.py\n"
                "@@ -1 +1 @@\n-a\n+b\n```\ntrailing")
        block = _parse_patch_block(text)
        self.assertIsNotNone(block)
        self.assertEqual(len(parse_unified_diff(block)), 1)
        self.assertIsNone(_parse_patch_block("no blocks at all"))
        self.assertIsNone(_parse_patch_block("```json\n{}\n```"))


class PatchProtocolTests(unittest.TestCase):
    """The ```diff protocol wired into CodingAgent._draft_change."""

    def test_diff_response_applied_then_rolled_back(self):
        fx = FixtureRepo(with_git=True)
        self.addCleanup(fx.cleanup)
        fx.write("a.py", "X = 1\n")
        fx.commit_all()
        diff = ("```diff\n--- a/a.py\n+++ b/a.py\n"
                "@@ -1 +1 @@\n-X = 1\n+X = 2\n```\n")
        router = ScriptedRouter([_plan_response("a.py"), diff])
        agent = CodingAgent(StubContext(router), root=str(fx.root))
        result = agent.run("bump X", accept=FAIL_ACCEPT, max_iterations=1)
        self.assertFalse(result.ok)
        # The diff WAS applied mid-mission (journal proves it)…
        rows = agent.db.query(
            "SELECT code FROM coding_log WHERE filename='a.py'")
        self.assertTrue(rows)
        self.assertIn("+X = 2", rows[0]["code"])
        # …and the failed mission rolled it back.
        self.assertEqual((fx.root / "a.py").read_text(), "X = 1\n")
        self.assertNotIn("WARNING", result.error)


# ── repo map ──────────────────────────────────────────────────────────────

class RepoMapTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="repomap_"))
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))
        pkg = self.root / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text('"""The pkg package."""\n')
        (pkg / "mod.py").write_text(
            '"""Does important things."""\n\n'
            "def alpha():\n    pass\n\n\n"
            "def beta():\n    pass\n\n\n"
            "class Gamma:\n    def method(self):\n        pass\n")
        (self.root / "notes.txt").write_text("not python\n")

    def test_map_bounded_and_informative(self):
        repo_map = build_repo_map(self.root)
        self.assertTrue(repo_map)
        self.assertLessEqual(len(repo_map), 2000)
        self.assertIn("pkg", repo_map)  # top-level dir
        self.assertIn("Does important things.", repo_map)  # docstring purpose
        self.assertIn("symbols", repo_map)  # symbol counts
        self.assertIn("mod.py", repo_map)

    def test_map_survives_broken_tree(self):
        self.assertEqual(build_repo_map(self.root / "nope"), "")

    def test_plan_files_survives_indexer_failure(self):
        router = ScriptedRouter([_plan_response("x.py")])
        agent = CodingAgent(StubContext(router), root=str(self.root))
        with mock.patch("nomorals.agents.coding.build_repo_map",
                        side_effect=ImportError("nope")):
            plan = agent._plan_files("do things", "x.py", self.root)
        self.assertEqual(plan[0]["path"], "x.py")

    def test_plan_files_includes_map(self):
        router = ScriptedRouter([_plan_response("x.py")])
        agent = CodingAgent(StubContext(router), root=str(self.root))
        with mock.patch("nomorals.agents.coding.build_repo_map",
                        return_value="SENTINEL-MAP"):
            agent._plan_files("do things", "x.py", self.root)
        user_texts = [m.content for m in router.messages[0]
                      if m.role == "user"]
        self.assertTrue(any("SENTINEL-MAP" in t for t in user_texts))


if __name__ == "__main__":
    unittest.main()
