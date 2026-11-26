"""Build swarm (audit N4) — failing-first tests.

A multi-worker BUILD swarm: a job splits into single-job workers (each
with a role + a task + a language rung), a VERIFIER agent runs every
worker's tests and reviews the diff, results merge into one report.
Sibling of research_swarm (research-only) and swarm.py (investigation
legs, prose digests, model-leg verifier) — neither builds code.

Covered here:
- language rungs (python / rust / go): probe + test + check commands
- heuristic decomposition (conjunction / numbered / role fallback)
- the NL request parser used by the agent loop
- the verifier: accepts green tests, rejects red tests with reasons,
  rejects an empty diff, rejects a missing toolchain honestly, falls
  back to the rung's compile/syntax check when no test files exist
- the run: parallel per-job workers with per-swarm dirs (no shared
  queue), exactly one retry for a rejected leg, merged report
- the real test runner: a passing pytest file is accepted, a failing
  one is rejected (subprocess, real pytest)
- the tool registration hook (build_swarm in the ToolRegistry)
- the render: job, per-leg verdicts, verifier reasons
- the agent loop routes NL "swarm build ..." to the build swarm
"""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.build_swarm import (
    LANGUAGE_RUNGS,
    BuildSwarm,
    BuildTask,
    parse_build_swarm_request,
    render_build_swarm,
)


def _ctx(home=None):
    return SimpleNamespace(
        db=None, tools=None, router=None,
        settings=SimpleNamespace(home=str(home) if home else None),
        home=str(home) if home else None, extras={})


def _worker_ok(task, work_dir, deadline, attempt=0):
    """A worker that writes one source file + one passing test."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    if task.language == "python":
        (work_dir / "thing.py").write_text(
            "def double(n):\n    return n * 2\n", encoding="utf-8")
        (work_dir / "test_thing.py").write_text(
            "from thing import double\n\n\n"
            "def test_double():\n    assert double(3) == 6\n",
            encoding="utf-8")
    elif task.language == "rust":
        (work_dir / "Cargo.toml").write_text(
            "[package]\nname = \"thing\"\nversion = \"0.1.0\"\n"
            "edition = \"2021\"\n",
            encoding="utf-8")
        src = work_dir / "src"
        src.mkdir(exist_ok=True)
        (src / "main.rs").write_text(
            "fn main() { println!(\"{}\", 3 * 2); }\n", encoding="utf-8")
    else:
        (work_dir / "go.mod").write_text(
            "module thing\n\ngo 1.21\n", encoding="utf-8")
        (work_dir / "main.go").write_text(
            "package main\n\nimport \"fmt\"\n\n"
            "func main() { fmt.Println(3 * 2) }\n",
            encoding="utf-8")
    return {"ok": True, "files": ["thing.py"], "output": "built it",
            "error": ""}


def _worker_empty(task, work_dir, deadline, attempt=0):
    """A worker that produces nothing — the verifier must reject it."""
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    return {"ok": True, "files": [], "output": "done (nothing written)",
            "error": ""}


def _worker_crashes(task, work_dir, deadline, attempt=0):
    raise RuntimeError("worker exploded")


class LanguageRungTests(unittest.TestCase):
    def test_three_rungs_exist(self):
        self.assertEqual(set(LANGUAGE_RUNGS), {"python", "rust", "go"})

    def test_python_rung(self):
        rung = LANGUAGE_RUNGS["python"]
        self.assertEqual(rung["probe"], "python3")
        self.assertIn("pytest", rung["test_cmd"])
        self.assertIn("py_compile", rung["check_cmd"])
        self.assertIn(".py", rung["extensions"])

    def test_rust_rung(self):
        rung = LANGUAGE_RUNGS["rust"]
        self.assertEqual(rung["probe"], "cargo")
        self.assertEqual(rung["test_cmd"][:2], ["cargo", "test"])
        self.assertEqual(rung["check_cmd"][:2], ["cargo", "check"])

    def test_go_rung(self):
        rung = LANGUAGE_RUNGS["go"]
        self.assertEqual(rung["probe"], "go")
        self.assertEqual(rung["test_cmd"][:2], ["go", "test"])
        self.assertEqual(rung["check_cmd"][:2], ["go", "vet"])

    def test_unknown_language_rejected(self):
        swarm = BuildSwarm(_ctx(), work_root="unused")
        with self.assertRaises(ValueError):
            swarm.run("build a thing", language="cobol")


class DecomposeTests(unittest.TestCase):
    def test_conjunction_split(self):
        swarm = BuildSwarm(_ctx(), work_root="unused")
        tasks, planner = swarm.decompose(
            "write the parser and add tests and update the docs", 3)
        self.assertEqual(planner, "heuristic")
        self.assertEqual(len(tasks), 3)
        self.assertTrue(all(isinstance(t, BuildTask) for t in tasks))
        self.assertEqual({t.role for t in tasks},
                         {"builder", "tester", "documenter"})

    def test_numbered_split(self):
        swarm = BuildSwarm(_ctx(), work_root="unused")
        tasks, _ = swarm.decompose(
            "do these:\n1. parse the file\n2. validate the rows", 2)
        self.assertEqual(len(tasks), 2)

    def test_role_fallback_single_job(self):
        swarm = BuildSwarm(_ctx(), work_root="unused")
        tasks, _ = swarm.decompose("write a tiny http server", 3)
        self.assertEqual(len(tasks), 3)
        self.assertEqual([t.role for t in tasks],
                         ["builder", "tester", "documenter"])
        self.assertIn("http server", tasks[0].task)

    def test_workers_cap(self):
        swarm = BuildSwarm(_ctx(), work_root="unused")
        tasks, _ = swarm.decompose("a and b and c and d and e and f", 99)
        self.assertLessEqual(len(tasks), 5)

    def test_empty_job_rejected(self):
        swarm = BuildSwarm(_ctx(), work_root="unused")
        with self.assertRaises(ValueError):
            swarm.run("   ")


class ParseRequestTests(unittest.TestCase):
    def test_swarm_build(self):
        req = parse_build_swarm_request("swarm build a tiny http server")
        self.assertEqual(req["job"], "a tiny http server")
        self.assertEqual(req["language"], "python")
        self.assertEqual(req["workers"], 3)

    def test_build_swarm_in_rust(self):
        req = parse_build_swarm_request(
            "build swarm a fast json parser in rust")
        self.assertEqual(req["job"], "a fast json parser")
        self.assertEqual(req["language"], "rust")

    def test_workers_and_go(self):
        req = parse_build_swarm_request(
            "swarm build a cli tool in go with 2 workers")
        self.assertEqual(req["job"], "a cli tool")
        self.assertEqual(req["language"], "go")
        self.assertEqual(req["workers"], 2)

    def test_trailing_number_is_workers(self):
        req = parse_build_swarm_request("swarm build a thing 4")
        self.assertEqual(req["job"], "a thing")
        self.assertEqual(req["workers"], 4)

    def test_run_a_prefix(self):
        req = parse_build_swarm_request("run a build swarm for a cache")
        self.assertIsNotNone(req)
        self.assertEqual(req["job"], "a cache")

    def test_non_match(self):
        self.assertIsNone(parse_build_swarm_request("swarm the bees"))
        self.assertIsNone(parse_build_swarm_request("build me a house"))
        self.assertIsNone(parse_build_swarm_request("swarm build"))
        self.assertIsNone(parse_build_swarm_request("hello"))


class VerifierTests(unittest.TestCase):
    def _swarm(self, tmp):
        return BuildSwarm(_ctx(home=tmp), work_root=str(Path(tmp) / "w"))

    def _leg(self, tmp, language="python", files=("thing.py",)):
        from nomorals.agents.build_swarm import BuildLeg

        d = Path(tmp) / "leg"
        d.mkdir(parents=True, exist_ok=True)
        return BuildLeg(task="build a thing", role="builder",
                        language=language, name="build-a-thing",
                        work_dir=d, ok=True, files=list(files))

    def test_accepts_green_tests(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            swarm = self._swarm(tmp)
            leg = self._leg(tmp)
            _worker_ok(BuildTask(role="builder", task="x",
                                 language="python", name="n"),
                       leg.work_dir, 60.0)
            leg.files = ["thing.py", "test_thing.py"]
            verified = swarm._verify_leg(leg)
            self.assertTrue(verified.accepted)
            self.assertEqual(verified.verdict, "accepted")
            self.assertEqual(verified.test_rc, 0)

    def test_rejects_red_tests_with_reason(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            swarm = self._swarm(tmp)
            leg = self._leg(tmp)
            (leg.work_dir / "thing.py").write_text(
                "def double(n):\n    return n * 2\n", encoding="utf-8")
            (leg.work_dir / "test_thing.py").write_text(
                "from thing import double\n\n\n"
                "def test_double():\n    assert double(3) == 7\n",
                encoding="utf-8")
            leg.files = ["thing.py", "test_thing.py"]
            verified = swarm._verify_leg(leg)
            self.assertFalse(verified.accepted)
            self.assertEqual(verified.verdict, "rejected")
            self.assertNotEqual(verified.test_rc, 0)
            self.assertTrue(any("test" in r.lower()
                                for r in verified.reasons))

    def test_rejects_empty_diff(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            swarm = self._swarm(tmp)
            leg = self._leg(tmp, files=[])
            verified = swarm._verify_leg(leg)
            self.assertFalse(verified.accepted)
            self.assertTrue(any("no files" in r.lower()
                                for r in verified.reasons))

    def test_missing_toolchain_is_honest_rejection(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            swarm = self._swarm(tmp)
            leg = self._leg(tmp, language="rust")
            (leg.work_dir / "Cargo.toml").write_text(
                "[package]\nname = \"thing\"\nversion = \"0.1.0\"\n"
                "edition = \"2021\"\n",
                encoding="utf-8")
            leg.files = ["Cargo.toml"]
            # hide cargo from PATH for this probe
            import os

            env_path = os.environ.get("PATH", "")
            kept = [p for p in env_path.split(os.pathsep)
                    if "cargo" not in p and ".cargo" not in p]
            with (mock.patch.dict(
                    os.environ, {"PATH": os.pathsep.join(kept)}),
                    mock.patch(
                        "nomorals.agents.build_swarm.shutil.which",
                        return_value=None)):
                verified = swarm._verify_leg(leg)
            self.assertFalse(verified.accepted)
            self.assertTrue(any("cargo" in r and "not found" in r
                                for r in verified.reasons))

    def test_no_tests_falls_back_to_check_command(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            swarm = self._swarm(tmp)
            leg = self._leg(tmp)
            (leg.work_dir / "thing.py").write_text(
                "def double(n):\n    return n * 2\n", encoding="utf-8")
            leg.files = ["thing.py"]
            verified = swarm._verify_leg(leg)
            self.assertTrue(verified.accepted)
            self.assertTrue(any("no tests" in r.lower()
                                for r in verified.reasons))

    def test_real_pytest_run_rejects_failing_file(self):
        """The verifier really executes pytest (subprocess), it does not
        trust the worker's word."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "thing.py").write_text("x = 1\n", encoding="utf-8")
            (d / "test_thing.py").write_text(
                "def test_nope():\n    assert 1 == 2\n", encoding="utf-8")
            rc, out = BuildSwarm(_ctx(), work_root=tmp)._run_command(
                ["python3", "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                cwd=d, timeout=60.0)
            self.assertNotEqual(rc, 0)
            self.assertIn("FAILED", out)


class RunTests(unittest.TestCase):
    def test_parallel_workers_distinct_dirs_no_shared_queue(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            swarm = BuildSwarm(_ctx(home=tmp), work_root=str(Path(tmp) / "w"),
                               worker_fn=_worker_ok)
            result = swarm.run("write the parser and add tests", workers=2)
            self.assertTrue(result.ok)
            self.assertEqual(len(result.legs), 2)
            dirs = [str(leg.work_dir) for leg in result.legs]
            self.assertEqual(len(set(dirs)), 2)
            self.assertTrue(all(d.startswith(str(Path(tmp) / "w"))
                                for d in dirs))
            self.assertTrue(all(leg.accepted for leg in result.legs))

    def test_empty_worker_rejected_after_one_retry(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            calls: list[int] = []

            def counting(task, work_dir, deadline, attempt=0):
                calls.append(attempt)
                return _worker_empty(task, work_dir, deadline, attempt)

            swarm = BuildSwarm(_ctx(home=tmp), work_root=str(Path(tmp) / "w"),
                               worker_fn=counting)
            result = swarm.run("build a thing", workers=1)
            self.assertFalse(result.ok)
            self.assertEqual(len(result.legs), 1)
            self.assertFalse(result.legs[0].accepted)
            # first attempt + exactly one retry
            self.assertEqual(calls, [0, 1])
            self.assertTrue(any("no files" in r.lower()
                                for r in result.legs[0].reasons))

    def test_crashing_worker_is_a_failed_leg_not_a_dead_swarm(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            def mixed(task, work_dir, deadline, attempt=0):
                if task.role == "builder":
                    return _worker_ok(task, work_dir, deadline, attempt)
                return _worker_crashes(task, work_dir, deadline, attempt)

            swarm = BuildSwarm(_ctx(home=tmp), work_root=str(Path(tmp) / "w"),
                               worker_fn=mixed)
            result = swarm.run("build a thing", workers=2)
            self.assertTrue(result.ok)
            by_role = {leg.role: leg for leg in result.legs}
            self.assertTrue(by_role["builder"].accepted)
            self.assertFalse(by_role["tester"].accepted)
            self.assertTrue(any("exploded" in r
                                for r in by_role["tester"].reasons))

    def test_two_runs_get_distinct_run_dirs(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            swarm = BuildSwarm(_ctx(home=tmp), work_root=str(Path(tmp) / "w"),
                               worker_fn=_worker_ok)
            r1 = swarm.run("build a thing", workers=1)
            r2 = swarm.run("build a thing", workers=1)
            self.assertNotEqual(r1.run_id, r2.run_id)
            self.assertNotEqual(str(r1.legs[0].work_dir),
                                str(r2.legs[0].work_dir))

    def test_report_dict_and_render(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            swarm = BuildSwarm(_ctx(home=tmp), work_root=str(Path(tmp) / "w"),
                               worker_fn=_worker_ok)
            result = swarm.run("write the parser and add tests", workers=2)
            payload = result.to_dict()
            self.assertEqual(payload["job"], "write the parser and add tests")
            self.assertTrue(payload["ok"])
            self.assertEqual(len(payload["legs"]), 2)
            self.assertIn("verdict", payload["legs"][0])
            text = render_build_swarm(result)
            self.assertIn("parser", text)
            self.assertIn("accepted", text)

    def test_no_model_worker_fails_honestly(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            # default worker with no router/model available
            swarm = BuildSwarm(_ctx(home=tmp), work_root=str(Path(tmp) / "w"))
            result = swarm.run("build a thing", workers=1)
            self.assertFalse(result.ok)
            self.assertTrue(any("no model" in r.lower()
                                for r in result.legs[0].reasons))


class RegisterTests(unittest.TestCase):
    def test_build_swarm_tool_registers(self):
        from nomorals.tools.registry import ToolRegistry

        registry = ToolRegistry(SimpleNamespace())
        registry.register_builtins()
        self.assertIn("build_swarm", registry._tools)
        self.assertEqual(len(registry._tools), 788)  # goal-class adds goal_class


class AgentLoopTests(unittest.TestCase):
    def test_run_loop_routes_swarm_build(self):
        import tempfile

        from nomorals.agents.agent_loop import run_loop

        def _msg(text):
            chat = SimpleNamespace(
                platform="telegram", chat_id="123", kind="dm", thread_id="",
                key="telegram:123")
            return SimpleNamespace(chat=chat, text=text, media=[],
                                   incoming=True, sender="owner", meta={})

        routes: list[dict] = []

        class FakeMind:
            def _get_pending(self, key):
                return None

            def _live_game(self, key):
                return None

            def _record_route(self, intent):
                routes.append({"kind": intent.kind,
                               "route": intent.route})

            def _dispatch(self, intent, chat_key, message):
                raise AssertionError("dispatch must not run for swarm build")

        with tempfile.TemporaryDirectory() as tmp:
            import nomorals.agents.build_swarm as bs

            real = bs.BuildSwarm
            calls: list[str] = []

            class FakeSwarm:
                def __init__(self, *a, **k):
                    pass

                def run(self, job, workers=3, language="python",
                        on_progress=None):
                    calls.append(job)
                    res = real(_ctx(home=tmp),
                               work_root=str(Path(tmp) / "w"),
                               worker_fn=_worker_ok).run(
                        job, workers=workers, language=language)
                    return res

            bs.BuildSwarm = FakeSwarm  # type: ignore[assignment]
            try:
                reply, _ctx2 = run_loop(
                    FakeMind(), "swarm build a tiny http server in python",
                    message=_msg("swarm build a tiny http server in python"),
                    chat_key="telegram:123")
            finally:
                bs.BuildSwarm = real
        self.assertEqual(calls, ["a tiny http server"])
        self.assertIn("accepted", reply or "")
        self.assertTrue(any(r["route"] == "build_swarm" for r in routes))


class CliTests(unittest.TestCase):
    def test_nm_swarm_build_parses(self):
        from nomorals.cmdline.parser import _parser

        args = _parser().parse_args(
            ["swarm", "build", "a tiny http server",
             "--workers", "2", "--lang", "rust"])
        self.assertEqual(args.swarm_action, "build")
        self.assertEqual(args.job, "a tiny http server")
        self.assertEqual(args.workers, 2)
        self.assertEqual(args.lang, "rust")

    def test_nm_swarm_build_runs(self):
        import argparse
        import tempfile

        from nomorals.cmdline.commands.swarm import _cmd_swarm

        with tempfile.TemporaryDirectory() as tmp:
            import nomorals.agents.build_swarm as bs

            real = bs.BuildSwarm
            calls: list[str] = []

            class FakeSwarm:
                def __init__(self, *a, **k):
                    pass

                def run(self, job, workers=3, language="python",
                        on_progress=None):
                    calls.append(job)
                    return real(_ctx(home=tmp),
                                work_root=str(Path(tmp) / "w"),
                                worker_fn=_worker_ok).run(
                        job, workers=workers, language=language)

            bs.BuildSwarm = FakeSwarm  # type: ignore[assignment]
            try:
                args = argparse.Namespace(
                    swarm_action="build", job="build a thing",
                    workers=1, lang="python", json=True)
                rc = _cmd_swarm(args, _ctx(home=tmp))
            finally:
                bs.BuildSwarm = real
        self.assertEqual(rc, 0)
        self.assertEqual(calls, ["build a thing"])


if __name__ == "__main__":
    unittest.main()
