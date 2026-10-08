"""Item #17 — checkpoints + rewind for the coding agent.

All offline: temp git repos via ``git init`` in tmp_path, scripted routers,
no network, no model calls.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from nomorals.agents.checkpoints import (
    CHECKPOINT_KEEP,
    Checkpoint,
    CheckpointStore,
)
from nomorals.agents.coding import CodingAgent


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd),
                          capture_output=True, text=True, timeout=60)


class StubDB:
    def __init__(self):
        self.conn = sqlite3.connect(":memory:")

    def execute(self, sql, params=()):
        self.conn.execute(sql, params)
        self.conn.commit()

    def query(self, sql, params=()):
        cur = self.conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


class StubRouter:
    def chat(self, *a, **k):
        raise AssertionError("no model calls in checkpoint tests")


def _ctx(home: Path) -> SimpleNamespace:
    ctx = SimpleNamespace()
    ctx.router = StubRouter()
    ctx.db = StubDB()
    ctx.settings = SimpleNamespace(reasoning_mode="off",
                                   home_path=str(home))
    return ctx


class FixtureRepo:
    """Temp dir, optionally a git repo with user config."""

    def __init__(self, with_git: bool = False):
        self.root = Path(tempfile.mkdtemp(prefix="ckpt_"))
        if with_git:
            _git("init", "-q", cwd=self.root)
            _git("config", "user.email", "t@example.com", cwd=self.root)
            _git("config", "user.name", "T", cwd=self.root)

    def write(self, rel: str, content: str) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return p

    def commit_all(self, msg: str = "init") -> None:
        _git("add", ".", cwd=self.root)
        _git("commit", "-qm", msg, cwd=self.root)

    def stash_count(self) -> int:
        proc = _git("stash", "list", cwd=self.root)
        return len([l for l in proc.stdout.splitlines() if l.strip()])

    def cleanup(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)


def _agent(fx: FixtureRepo, home: Path) -> CodingAgent:
    return CodingAgent(_ctx(home), root=str(fx.root))


class CheckpointCaptureTests(unittest.TestCase):
    def setUp(self):
        self.fx = FixtureRepo(with_git=True)
        self.home = Path(tempfile.mkdtemp(prefix="ckpt_home_"))
        self.addCleanup(self.fx.cleanup)
        self.addCleanup(lambda: __import__("shutil").rmtree(
            self.home, ignore_errors=True))
        self.fx.write("a.py", "X = 1\n")
        self.fx.commit_all()

    def test_checkpoint_captures_dirty_tracked_files(self):
        agent = _agent(self.fx, self.home)
        self.fx.write("a.py", "X = 2\n")
        ckpt = agent.checkpoint(label="before")
        self.assertTrue(ckpt.code_captured)
        self.assertEqual(ckpt.label, "before")
        self.assertIn("a.py", ckpt.scope_files)
        self.assertTrue(ckpt.stash_hash, "dirty tree must yield a stash commit")
        self.assertTrue(ckpt.head_hash)
        # Working tree untouched by the capture.
        self.assertEqual((self.fx.root / "a.py").read_text(), "X = 2\n")

    def test_checkpoint_never_touches_user_stash_list(self):
        agent = _agent(self.fx, self.home)
        before = self.fx.stash_count()
        self.fx.write("a.py", "X = 2\n")
        agent.checkpoint()
        self.assertEqual(self.fx.stash_count(), before,
                         "git stash create must not add to the stash list")

    def test_clean_tree_checkpoint_has_no_stash(self):
        agent = _agent(self.fx, self.home)
        ckpt = agent.checkpoint()
        self.assertTrue(ckpt.code_captured)
        self.assertIsNone(ckpt.stash_hash)
        self.assertEqual(ckpt.scope_files, [])

    def test_untracked_files_captured(self):
        agent = _agent(self.fx, self.home)
        self.fx.write("new.txt", "hello\n")
        ckpt = agent.checkpoint()
        self.assertIn("new.txt", ckpt.untracked_files)
        stored = (self.home / ".nomorals" / "checkpoints"
                  / ckpt.id / "untracked" / "new.txt")
        self.assertTrue(stored.exists())
        self.assertEqual(stored.read_text(), "hello\n")

    def test_non_repo_dir_stores_convo_only(self):
        plain = FixtureRepo(with_git=False)
        self.addCleanup(plain.cleanup)
        agent = _agent(plain, self.home)
        ckpt = agent.checkpoint(label="x")
        self.assertFalse(ckpt.code_captured)
        self.assertIsNone(ckpt.stash_hash)

    def test_checkpoint_ids_unique(self):
        agent = _agent(self.fx, self.home)
        ids = {agent.checkpoint().id for _ in range(3)}
        self.assertEqual(len(ids), 3)


class RewindTests(unittest.TestCase):
    def setUp(self):
        self.fx = FixtureRepo(with_git=True)
        self.home = Path(tempfile.mkdtemp(prefix="ckpt_home_"))
        self.addCleanup(self.fx.cleanup)
        self.addCleanup(lambda: __import__("shutil").rmtree(
            self.home, ignore_errors=True))
        self.fx.write("a.py", "X = 1\n")
        self.fx.write("b.py", "Y = 1\n")
        self.fx.commit_all()

    def test_rewind_restores_tracked_files(self):
        agent = _agent(self.fx, self.home)
        self.fx.write("a.py", "X = 2\n")
        agent.checkpoint(label="v2")
        self.fx.write("a.py", "X = 3\n")   # bad edit after the checkpoint
        summary = agent.rewind(1)
        self.assertEqual((self.fx.root / "a.py").read_text(), "X = 2\n")
        self.assertIn("restored 1 tracked", summary)

    def test_rewind_restores_deleted_tracked_file(self):
        agent = _agent(self.fx, self.home)
        self.fx.write("a.py", "X = 2\n")
        agent.checkpoint()
        (self.fx.root / "a.py").unlink()
        agent.rewind(1)
        self.assertEqual((self.fx.root / "a.py").read_text(), "X = 2\n")

    def test_rewind_does_not_touch_out_of_scope_files(self):
        agent = _agent(self.fx, self.home)
        self.fx.write("a.py", "X = 2\n")
        agent.checkpoint()
        self.fx.write("b.py", "Y = 99\n")  # edited AFTER the checkpoint
        agent.rewind(1)
        self.assertEqual((self.fx.root / "b.py").read_text(), "Y = 99\n")

    def test_rewind_restores_untracked_files(self):
        agent = _agent(self.fx, self.home)
        self.fx.write("notes.txt", "original\n")
        agent.checkpoint()
        self.fx.write("notes.txt", "mangled\n")
        agent.rewind(1)
        self.assertEqual((self.fx.root / "notes.txt").read_text(),
                         "original\n")

    def test_rewind_recreates_deleted_untracked_file(self):
        agent = _agent(self.fx, self.home)
        self.fx.write("notes.txt", "original\n")
        agent.checkpoint()
        (self.fx.root / "notes.txt").unlink()
        agent.rewind(1)
        self.assertEqual((self.fx.root / "notes.txt").read_text(),
                         "original\n")

    def test_rewind_never_moves_head(self):
        agent = _agent(self.fx, self.home)
        head_before = _git("rev-parse", "HEAD",
                           cwd=self.fx.root).stdout.strip()
        branch_before = _git("branch", "--show-current",
                             cwd=self.fx.root).stdout.strip()
        self.fx.write("a.py", "X = 2\n")
        agent.checkpoint()
        self.fx.write("a.py", "X = 3\n")
        agent.rewind(1)
        head_after = _git("rev-parse", "HEAD",
                          cwd=self.fx.root).stdout.strip()
        branch_after = _git("branch", "--show-current",
                            cwd=self.fx.root).stdout.strip()
        self.assertEqual(head_before, head_after)
        self.assertEqual(branch_before, branch_after)

    def test_pre_rewind_safety_checkpoint_created(self):
        agent = _agent(self.fx, self.home)
        self.fx.write("a.py", "X = 2\n")
        agent.checkpoint(label="good")
        self.fx.write("a.py", "X = 3\n")
        summary = agent.rewind(1)
        self.assertIn("pre-rewind", summary)
        ckpts = agent.list_checkpoints()
        self.assertEqual(ckpts[0].label, "pre-rewind")
        self.assertEqual((self.fx.root / "a.py").read_text(), "X = 2\n")
        # Rewinding the rewind targets the safety checkpoint → X = 3 back.
        agent.rewind(1)
        self.assertEqual((self.fx.root / "a.py").read_text(), "X = 3\n")

    def test_rewind_restores_conversation_state(self):
        agent = _agent(self.fx, self.home)
        agent._last_task = "build widgets"
        agent._last_plan_id = "plan123"
        agent._last_scope = ["w.py"]
        agent._last_iterations = 4
        agent.checkpoint(label="s")
        agent._last_task = "something else"
        agent._last_iterations = 9
        agent.rewind(1)
        self.assertEqual(agent._last_task, "build widgets")
        self.assertEqual(agent._last_plan_id, "plan123")
        self.assertEqual(agent._last_scope, ["w.py"])
        self.assertEqual(agent._last_iterations, 4)

    def test_rewind_no_checkpoints_clean_error(self):
        agent = _agent(self.fx, self.home)
        msg = agent.rewind(1)
        self.assertIn("no checkpoints", msg.lower())

    def test_rewind_bad_number(self):
        agent = _agent(self.fx, self.home)
        self.assertIn("bad checkpoint number", agent.rewind("zzz").lower())

    def test_rewind_nth_checkpoint(self):
        agent = _agent(self.fx, self.home)
        self.fx.write("a.py", "X = 2\n")
        agent.checkpoint(label="first")
        self.fx.write("a.py", "X = 5\n")
        agent.checkpoint(label="second")
        self.fx.write("a.py", "X = 9\n")
        agent.rewind(2)  # second-latest = "first"
        self.assertEqual((self.fx.root / "a.py").read_text(), "X = 2\n")

    def test_new_untracked_files_left_in_place(self):
        agent = _agent(self.fx, self.home)
        agent.checkpoint()
        self.fx.write("later.txt", "made after\n")
        summary = agent.rewind(1)
        self.assertTrue((self.fx.root / "later.txt").exists())
        self.assertIn("left in place", summary)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.fx = FixtureRepo(with_git=True)
        self.home = Path(tempfile.mkdtemp(prefix="ckpt_home_"))
        self.addCleanup(self.fx.cleanup)
        self.addCleanup(lambda: __import__("shutil").rmtree(
            self.home, ignore_errors=True))
        self.fx.write("a.py", "X = 1\n")
        self.fx.commit_all()

    def _store(self) -> CheckpointStore:
        return CheckpointStore(settings=SimpleNamespace(
            home_path=str(self.home)))

    def test_prune_keeps_ten(self):
        store = self._store()
        for i in range(CHECKPOINT_KEEP + 3):
            ckpt = store.capture(self.fx.root, label=f"c{i}")
            store.save(ckpt)
        items = store.list()
        self.assertEqual(len(items), CHECKPOINT_KEEP)
        labels = [c.label for c in items]
        self.assertNotIn("c0", labels)
        self.assertIn(f"c{CHECKPOINT_KEEP + 2}", labels)

    def test_list_newest_first(self):
        store = self._store()
        first = store.save(store.capture(self.fx.root, label="a"))
        second = store.save(store.capture(self.fx.root, label="b"))
        items = store.list()
        self.assertEqual(items[0].id, second.id)
        self.assertEqual(items[1].id, first.id)

    def test_checkpoint_round_trip(self):
        store = self._store()
        ckpt = store.save(store.capture(
            self.fx.root, label="rt",
            convo={"task": "t", "plan_id": "p",
                   "scope": ["a.py"], "iterations": 2}))
        loaded = store.load(ckpt.id)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.id, ckpt.id)
        self.assertEqual(loaded.convo["task"], "t")
        self.assertEqual(loaded.short.split(" · ")[0], ckpt.id)

    def test_checkpoint_to_dict_json_safe(self):
        ckpt = Checkpoint(id="x", ts=1.0, label="l", head_hash="h",
                          stash_hash=None, workdir="/tmp",
                          convo={"weird": object()})
        json.dumps(ckpt.to_dict())  # must not raise


class ExecutePlanCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.fx = FixtureRepo(with_git=True)
        self.home = Path(tempfile.mkdtemp(prefix="ckpt_home_"))
        self.addCleanup(self.fx.cleanup)
        self.addCleanup(lambda: __import__("shutil").rmtree(
            self.home, ignore_errors=True))
        self.fx.write("a.py", "X = 1\n")
        self.fx.commit_all()
        self.plan_file = Path(tempfile.mkdtemp(prefix="plans_")) \
            / "plans.json"
        self._old_env = os.environ.get("NM_PLAN_STORE")
        os.environ["NM_PLAN_STORE"] = str(self.plan_file)
        self.addCleanup(self._restore_env)
        self.addCleanup(lambda: __import__("shutil").rmtree(
            self.plan_file.parent, ignore_errors=True))

    def _restore_env(self):
        if self._old_env is None:
            os.environ.pop("NM_PLAN_STORE", None)
        else:
            os.environ["NM_PLAN_STORE"] = self._old_env

    def test_execute_plan_auto_checkpoints(self):
        from nomorals.agents.plan_mode import PlanStore
        from nomorals.agents.coding import CodingResult

        plan = PlanStore.new(
            "bump X",
            [{"path": "a.py", "why": "test", "new_file": False}],
            approach="", risks=[])
        PlanStore.approve(plan.id)
        agent = _agent(self.fx, self.home)
        self.fx.write("a.py", "X = 2\n")  # dirty pre-execution state
        with patch.object(
                CodingAgent, "run",
                return_value=CodingResult(ok=True, iterations=1)) as m:
            result = agent.execute_plan(plan.id)
        self.assertTrue(result.ok)
        m.assert_called_once()
        ckpts = agent.list_checkpoints()
        self.assertTrue(ckpts, "execute_plan must auto-checkpoint")
        auto = ckpts[0]
        self.assertEqual(auto.label, f"plan:{plan.id}")
        self.assertEqual(auto.convo.get("plan_id"), plan.id)
        self.assertEqual(auto.convo.get("scope"), ["a.py"])
        self.assertIn("a.py", auto.scope_files)

    def test_execute_plan_unknown_plan_no_checkpoint(self):
        agent = _agent(self.fx, self.home)
        result = agent.execute_plan("nope-not-real")
        self.assertFalse(result.ok)
        self.assertEqual(agent.list_checkpoints(), [])


if __name__ == "__main__":
    unittest.main()
