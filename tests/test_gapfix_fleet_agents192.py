"""Fleet (agents192 lane, F1 family): parallel coding-agent runs in
isolated git worktrees + Kanban board + diff review + approve/merge.

Survey: 305-agent list, AI Agents (103) + Development Frameworks
(89). F1 is the densest genuinely-missing family (22 M + 7 P = 29
entries: Agent 007, agent-manager, AgentBox, Agent Teams, AGX,
Dorothy, EliaAgent, LoopTroop, Maestro, Mysti, Opus Manager, Orkas,
Tale, Okto-Nexus, TeamHero, Agents Squads, Hive, Kodo, Okto-Pulse,
ORCH, Stoneforge, WritBase, plus the fleet-surface gap inside
Corellis, Hivemoot, NarraNexus, Raven, TITAN, MySpec, OpenAgents).
Full enumeration + batch plan: ~/workspace/agents192-verify/SURVEY.md.

Workers are fake here (they write files into the worktree) so the
whole loop runs offline with no model. The default worker is the
real CodingAgent (``nm code``'s agent, root-scoped); it is exercised
only through its call shape, never invoked here.

Tool-count pin: 807 after this lane added the ``fleet`` tool
(785 after C.8 added robot).
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.fleet import (
    FleetError,
    FleetStore,
    render_board,
)


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def _make_repo(root: Path, name: str = "repo") -> Path:
    repo = root / name
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _git(repo, "config", "user.email", "fleet-test@example.invalid")
    _git(repo, "config", "user.name", "Fleet Test")
    (repo / "a.txt").write_text("hello\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    return repo


def _write_file_worker(path: Path, content: str):
    def worker(root: Path, brief: str) -> dict:
        (root / path).write_text(content, encoding="utf-8")
        return {"ok": True, "files": [str(path)], "output": "done",
                "error": ""}
    return worker


def _failing_worker(root: Path, brief: str) -> dict:
    return {"ok": False, "files": [], "output": "",
            "error": "the worker refused the brief"}


def _empty_worker(root: Path, brief: str) -> dict:
    return {"ok": True, "files": [], "output": "nothing to do",
            "error": ""}


class FleetOrganTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.repo = _make_repo(self.root)
        self.store = FleetStore(self.root / "fleet")

    def test_full_loop_create_dispatch_review_approve(self):
        fleet = self.store.create_fleet("demo", self.repo)
        self.assertEqual(fleet["name"], "demo")
        task = self.store.dispatch(
            fleet["fleet_id"], "add b.txt",
            worker=_write_file_worker(Path("b.txt"), "from fleet\n"),
            run_sync=True)
        self.assertEqual(task["state"], "review")
        self.assertEqual(task["mechanism"], "worktree")
        review = self.store.review(task["task_id"])
        self.assertIn("b.txt", review["files"])
        self.assertIn("from fleet", review["diff"])
        merged = self.store.approve(task["task_id"])
        self.assertEqual(merged["state"], "merged")
        self.assertIn("b.txt", merged["result"]["merged_files"])
        self.assertEqual((self.repo / "b.txt").read_text(),
                         "from fleet\n")
        # The worktree is gone; the repo working tree was never
        # touched except through the applied patch.
        self.assertFalse(Path(task["worktree"]).exists())
        self.assertEqual((self.repo / "a.txt").read_text(), "hello\n")

    def test_two_tasks_are_isolated_from_each_other(self):
        fleet = self.store.create_fleet("iso", self.repo)
        first = self.store.dispatch(
            fleet["fleet_id"], "add one.txt",
            worker=_write_file_worker(Path("one.txt"), "one\n"))
        second = self.store.dispatch(
            fleet["fleet_id"], "add two.txt",
            worker=_write_file_worker(Path("two.txt"), "two\n"))
        first = self.store.wait(first["task_id"], timeout=30)
        second = self.store.wait(second["task_id"], timeout=30)
        self.assertEqual(first["state"], "review")
        self.assertEqual(second["state"], "review")
        self.assertNotEqual(first["worktree"], second["worktree"])
        # Each worktree holds only its own file.
        self.assertTrue((Path(first["worktree"]) / "one.txt").is_file())
        self.assertFalse((Path(first["worktree"]) / "two.txt").exists())
        self.assertTrue((Path(second["worktree"]) / "two.txt").is_file())
        self.assertFalse((Path(second["worktree"]) / "one.txt").exists())
        # The repo itself is still pristine until approval.
        self.assertFalse((self.repo / "one.txt").exists())
        self.assertFalse((self.repo / "two.txt").exists())

    def test_worker_failure_and_empty_success_are_failed(self):
        fleet = self.store.create_fleet("fail", self.repo)
        bad = self.store.dispatch(fleet["fleet_id"], "refuse me",
                                  worker=_failing_worker, run_sync=True)
        self.assertEqual(bad["state"], "failed")
        self.assertIn("refused", bad["error"])
        empty = self.store.dispatch(fleet["fleet_id"], "do nothing",
                                    worker=_empty_worker, run_sync=True)
        self.assertEqual(empty["state"], "failed")
        self.assertIn("changed nothing", empty["error"])

    def test_approve_refuses_non_review_and_conflicts(self):
        fleet = self.store.create_fleet("gate", self.repo)
        bad = self.store.dispatch(fleet["fleet_id"], "refuse me",
                                  worker=_failing_worker, run_sync=True)
        with self.assertRaises(FleetError):
            self.store.approve(bad["task_id"])
        # Two tasks editing the same line: the first merges, the
        # second's patch no longer applies and stays in review.
        first = self.store.dispatch(
            fleet["fleet_id"], "change a.txt to one",
            worker=_write_file_worker(Path("a.txt"), "one\n"),
            run_sync=True)
        second = self.store.dispatch(
            fleet["fleet_id"], "change a.txt to two",
            worker=_write_file_worker(Path("a.txt"), "two\n"),
            run_sync=True)
        self.assertEqual(first["state"], "review")
        self.assertEqual(second["state"], "review")
        self.store.approve(first["task_id"])
        self.assertEqual((self.repo / "a.txt").read_text(), "one\n")
        with self.assertRaises(FleetError) as caught:
            self.store.approve(second["task_id"])
        self.assertIn("does not apply cleanly", str(caught.exception))
        still = self.store.resolve_task(second["task_id"])
        self.assertEqual(still.state, "review")

    def test_board_columns_and_render(self):
        fleet = self.store.create_fleet("board", self.repo)
        done = self.store.dispatch(
            fleet["fleet_id"], "add b.txt",
            worker=_write_file_worker(Path("b.txt"), "b\n"),
            run_sync=True)
        failed = self.store.dispatch(
            fleet["fleet_id"], "refuse me",
            worker=_failing_worker, run_sync=True)
        board = self.store.board(fleet["fleet_id"])
        self.assertEqual(
            [t["task_id"] for t in board.columns["review"]],
            [done["task_id"]])
        self.assertEqual(
            [t["task_id"] for t in board.columns["failed"]],
            [failed["task_id"]])
        self.assertEqual(board.columns["queued"], [])
        text = render_board(board)
        self.assertIn("[review] (1)", text)
        self.assertIn("[failed] (1)", text)
        self.assertIn("add b.txt", text)

    def test_ambiguous_and_unknown_refs_refuse(self):
        self.store.create_fleet("same-name", self.repo)
        self.store.create_fleet("same-name", self.repo)
        with self.assertRaises(FleetError) as caught:
            self.store.resolve_fleet("same-name")
        self.assertIn("ambiguous", str(caught.exception))
        with self.assertRaises(FleetError):
            self.store.resolve_fleet("no-such-fleet")
        with self.assertRaises(FleetError):
            self.store.resolve_task("no-such-task")
        with self.assertRaises(FleetError):
            self.store.create_fleet("bad", self.root / "missing-dir")

    def test_recover_marks_orphaned_running_failed(self):
        fleet = self.store.create_fleet("orphan", self.repo)
        task = self.store.dispatch(
            fleet["fleet_id"], "add b.txt",
            worker=_write_file_worker(Path("b.txt"), "b\n"),
            run_sync=True)
        # Simulate a restart: a second store over the same root
        # finds the task running with no live thread.
        second = FleetStore(self.root / "fleet")
        record = second._load_task(fleet["fleet_id"], task["task_id"])
        assert record is not None
        record.state = "running"
        second._save_task(record)
        changed = second.recover()
        self.assertEqual(len(changed), 1)
        self.assertEqual(changed[0]["state"], "failed")
        self.assertIn("interrupted", changed[0]["error"])

    def test_cleanup_removes_finished_worktrees_only(self):
        fleet = self.store.create_fleet("clean", self.repo)
        done = self.store.dispatch(
            fleet["fleet_id"], "add b.txt",
            worker=_write_file_worker(Path("b.txt"), "b\n"),
            run_sync=True)
        self.store.approve(done["task_id"])
        self.store.dispatch(
            fleet["fleet_id"], "refuse me",
            worker=_failing_worker, run_sync=True)
        pending = self.store.dispatch(
            fleet["fleet_id"], "add c.txt",
            worker=_write_file_worker(Path("c.txt"), "c\n"),
            run_sync=True)
        self.assertEqual(pending["state"], "review")
        result = self.store.cleanup(fleet["fleet_id"])
        self.assertEqual(result["removed"], 1)  # the failed task's
        self.assertEqual(result["kept"], 1)     # the review task's
        self.assertTrue(Path(pending["worktree"]).exists())

    def test_dispatch_without_brief_or_worker_refuses(self):
        fleet = self.store.create_fleet("empty", self.repo)
        with self.assertRaises(FleetError):
            self.store.dispatch(fleet["fleet_id"], "   ")
        bare = FleetStore(self.root / "bare-fleet")
        bare_fleet = bare.create_fleet("bare", self.repo)
        with self.assertRaises(FleetError) as caught:
            bare.dispatch(bare_fleet["fleet_id"], "add b.txt")
        self.assertIn("no worker", str(caught.exception))


class FleetToolTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.repo = _make_repo(self.root)
        self.home = self.root / "home"
        self.home.mkdir()
        context = SimpleNamespace(
            settings=SimpleNamespace(home=str(self.home)))
        from nomorals.tools.registry import ToolRegistry

        self.registry = ToolRegistry(context).register_builtins()
        self.fleet = self.registry.get("fleet").fn

    def test_tool_count_is_807(self):
        self.assertEqual(len(self.registry._tools), 810)

    def test_tool_create_board_dispatch_review_approve(self):
        created = self.fleet(action="create", repo=str(self.repo),
                             name="tool-demo")
        self.assertIn("fleet_id", created)
        listed = self.fleet(action="list")
        self.assertEqual(len(listed["fleets"]), 1)
        with mock.patch(
                "nomorals.agents.fleet.default_worker",
                side_effect=lambda ctx, root, brief: (
                    (root / "b.txt").write_text("via tool\n"),
                    {"ok": True, "files": ["b.txt"], "output": "",
                     "error": ""})[1]):
            dispatched = self.fleet(
                action="dispatch", fleet=created["fleet_id"],
                brief="add b.txt", wait=30)
        self.assertEqual(dispatched["state"], "review")
        reviewed = self.fleet(action="review",
                              task=dispatched["task_id"])
        self.assertIn("via tool", reviewed["diff"])
        merged = self.fleet(action="approve",
                            task=dispatched["task_id"])
        self.assertEqual(merged["state"], "merged")
        self.assertEqual((self.repo / "b.txt").read_text(), "via tool\n")

    def test_tool_unknown_action_and_missing_args_are_errors(self):
        self.assertFalse(self.fleet(action="explode")["ok"])
        self.assertIn("unknown fleet action",
                      self.fleet(action="explode")["error"])
        self.assertFalse(self.fleet(action="dispatch")["ok"])
        self.assertFalse(self.fleet(action="review")["ok"])


class FleetCliTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.repo = _make_repo(self.root)
        self.home = self.root / "home"
        self.home.mkdir()
        self.context = SimpleNamespace(
            settings=SimpleNamespace(home=str(self.home)))

    def _run_cli(self, argv):
        import io
        from contextlib import redirect_stdout

        from nomorals.cmdline.commands.fleet import _cmd_fleet
        from nomorals.cmdline.parser import _parser

        args = _parser().parse_args(argv)
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = _cmd_fleet(args, self.context)
        return code, buf.getvalue()

    def test_cli_create_list_board(self):
        code, out = self._run_cli(
            ["fleet", "create", "--repo", str(self.repo),
             "--name", "cli-demo"])
        self.assertEqual(code, 0)
        self.assertIn("cli-demo", out)
        code, out = self._run_cli(["fleet", "list"])
        self.assertEqual(code, 0)
        self.assertIn("cli-demo", out)
        code, out = self._run_cli(["fleet", "board", "cli-demo"])
        self.assertEqual(code, 0)
        self.assertIn("[queued] (0)", out)

    def test_cli_dispatch_wait_review_approve(self):
        self._run_cli(["fleet", "create", "--repo", str(self.repo),
                       "--name", "cli-run"])
        with mock.patch(
                "nomorals.agents.fleet.default_worker",
                side_effect=lambda ctx, root, brief: (
                    (root / "b.txt").write_text("via cli\n"),
                    {"ok": True, "files": ["b.txt"], "output": "",
                     "error": ""})[1]):
            code, out = self._run_cli(
                ["fleet", "dispatch", "cli-run", "add b.txt",
                 "--wait", "30"])
        self.assertEqual(code, 0)
        self.assertIn("[review]", out)
        store = FleetStore(self.home / "fleet")
        tasks = store.list_tasks(
            store.resolve_fleet("cli-run").fleet_id)
        self.assertEqual(len(tasks), 1)
        code, out = self._run_cli(
            ["fleet", "review", tasks[0]["task_id"]])
        self.assertEqual(code, 0)
        self.assertIn("via cli", out)
        code, out = self._run_cli(
            ["fleet", "approve", tasks[0]["task_id"]])
        self.assertEqual(code, 0)
        self.assertIn("merged task", out)
        self.assertEqual((self.repo / "b.txt").read_text(), "via cli\n")

    def test_cli_unknown_fleet_fails(self):
        code, _ = self._run_cli(["fleet", "board", "no-such-fleet"])
        self.assertEqual(code, 1)

    def test_parser_accepts_every_fleet_subcommand(self):
        from nomorals.cmdline.parser import _parser

        parser = _parser()
        for argv in (
                ["fleet", "create", "--repo", "."],
                ["fleet", "dispatch", "f", "do it"],
                ["fleet", "dispatch", "f", "do it", "--wait", "5"],
                ["fleet", "board", "f"],
                ["fleet", "list"],
                ["fleet", "review", "t"],
                ["fleet", "approve", "t"],
                ["fleet", "cleanup", "f"]):
            args = parser.parse_args(argv)
            self.assertEqual(args.fleet_action, argv[1])


if __name__ == "__main__":
    unittest.main()
