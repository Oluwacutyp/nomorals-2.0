"""Pass #66: scoped registry governs the coding loop + background tests on.

The two survey items (2026-10-20 coding-agent survey, item 8 + W10):

- the roles layer runs agents under a locked tool scope (a bound
  ``RoleSpec`` enforced by ``check_spec_call`` in the roles layer; the
  orchestrator's ``CodingRoleAgent`` builds a ``RoleScopedRegistry``) —
  but BOTH bridges constructed the real loop as ``CodingAgent(context)``
  and the loop read ``context.tools`` (the FULL registry), so a role's
  allowlist never governed the loop it delegated to.  The
  ``CodingRoleAgent``'s scoped registry was built and dangled, never
  passed anywhere.
- ``CodingAgent.run(background_tests=True)`` existed and worked
  (agents/bg_tests.py) but NO production caller ever enabled it — and
  its cap was a flat ``test_cap_seconds=900.0`` default that could
  silently bind below a run's budget-derived verify timeout (the
  ceiling class the owner closed for the foreground path).

The contract pinned here:

- ``CodingAgent(..., tools=<scoped registry>)`` (default None =
  unscoped, byte-identical behavior) gates every governed surface —
  explore ``fs_read``, edit application (``edit_file`` /
  ``apply_patch``), verify (``run_tests``), lint (``lint``), accept
  command (``shell_run``).  A refusal is recorded on
  ``CodingResult.scope_denials`` (and in ``to_dict``), names the tool
  and the scope's own reason, and settles the run NOT green — it never
  falls back to the full registry or a direct read (no silent widening);
- both bridges forward the scope they were themselves scoped with:
  ``MissionCodingAgent`` builds the same ``RoleEnforcingRegistry`` the
  swarm layer builds, from its bound ``role_spec`` (unbound = no
  ``tools`` kwarg at all, the pre-#66 construction exactly), and
  ``CodingRoleAgent`` passes its ``RoleScopedRegistry``;
- background tests are enabled on the autonomous bridges; the
  background cap derives from the run budget the way the foreground
  verify does (``max(timeout * 5, 60)``) when ``test_cap_seconds`` is
  None (the new default), and an explicit cap is honored verbatim;
- a background run that hits its cap reports capped-with-partials
  (the bg note rides the run's error), never green, never a mystery.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nomorals.agents.bg_tests as bg_tests_module
import nomorals.agents.coding as coding_module
from nomorals.agents.coding import CodingAgent, CodingResult
from nomorals.agents.orchestrator import PlanStep
from nomorals.agents.role_specs import (
    BUILTIN_ROLES,
    RoleEnforcingRegistry,
    RoleSpec,
)
from nomorals.agents.roles import MissionCodingAgent
from nomorals.core.policy import Capability
from nomorals.core.result import Err, Ok
from nomorals.storage.db import Database
from nomorals.tools.agents import (
    CODING_CAPABILITIES,
    CODING_TOOLS,
    CodingRoleAgent,
    RoleScopedRegistry,
)


# ── fakes (pass #61/#65 shapes) ──────────────────────────────────────


def _usage(prompt, completion):
    return SimpleNamespace(prompt_tokens=prompt,
                           completion_tokens=completion,
                           total_tokens=prompt + completion)


class UsageRouter:
    """Content-routed fake LLM whose responses carry usage."""

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[tuple[str, str]] = []

    def chat(self, messages, params=None):
        texts = [getattr(m, "content", "") or "" for m in messages]
        system = texts[0] if texts else ""
        user = texts[-1] if texts else ""
        self.calls.append((system, user))
        text = self.handler(system, user)
        resp = SimpleNamespace(ok=bool(text), text=text,
                               error="" if text else "empty")
        resp.usage = _usage(10, 5)
        return resp


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
    def __init__(self, router, tools=None):
        self.router = router
        self.db = StubDB()
        self.tools = tools
        self.settings = SimpleNamespace(reasoning_mode="off")
        self.extras = {}


_TOOL_CAPS = {
    "fs_read": Capability.FS_READ,
    "edit_file": Capability.FS_WRITE,
    "apply_patch": Capability.FS_WRITE,
    "shell_run": Capability.EXEC_SHELL,
    "run_tests": Capability.EXEC_SHELL,
    "lint": Capability.EXEC_SHELL,
    "git_status": Capability.FS_READ,
    "git_diff": Capability.FS_READ,
    "search_code": Capability.MEM_READ,
    "index_repo": Capability.MEM_READ,
    "repo_map": Capability.FS_READ,
    "symbol_search": Capability.FS_READ,
    "browser": Capability.NET_BROWSER,
}


class FakeInnerRegistry:
    """The context's FULL registry stand-in: real get/names/schemas
    surface for the scope probes; call/call_many fail the way a real
    registry's fs_read fails when it cannot resolve the agent root —
    so the loop's explore fallback reads from disk, exactly as today.
    """

    def __init__(self):
        self.context = None
        self.calls: list = []

    def get(self, name):
        if name in _TOOL_CAPS:
            return SimpleNamespace(name=name, capability=_TOOL_CAPS[name])
        return None

    def names(self):
        return sorted(_TOOL_CAPS)

    def schemas(self, capabilities=None):
        return [{"name": n} for n in self.names()]

    def call(self, name, *args, actor="system", capabilities=None,
             **kwargs):
        self.calls.append((name, kwargs))
        return Err(ValueError(f"cannot resolve {kwargs.get('path', '')}"))

    def call_many(self, calls, max_workers=1, actor="system",
                  capabilities=None, **common):
        return [self.call(name, **kw) for name, kw in calls]


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True,
                   capture_output=True, timeout=60)


class FixtureRepo:
    def __init__(self):
        self.root = Path(tempfile.mkdtemp(prefix="pass66_"))
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


def _steps_json(steps):
    return "```json\n" + json.dumps({"steps": steps}) + "\n```"


def _code(code):
    return "```python\n" + code + "\n```"


_CALC_STEPS = [
    {"path": "calc.py", "why": "the math", "new_file": True,
     "instructions": "implement add(a, b) returning the sum",
     "acceptance": "test_calc passes"},
]


def _calc_handler(system, user):
    if "build planner" in system:
        return _steps_json(_CALC_STEPS)
    if "Write complete, runnable Python" in system:
        if "for the file 'calc.py'" in system:
            return _code("def add(a, b):\n    return a + b\n")
    return ""


def _calc_repo(fx):
    fx.write("tests/__init__.py", "")
    fx.write("tests/test_calc.py",
             "import calc\n\n\ndef test_add():\n"
             "    assert calc.add(2, 3) == 5\n")
    fx.commit_all()
    return fx


class CleanReviewer:
    """review_text double on the post-#63/#65 contract: no flaws,
    status channel filled (available, no usage reported)."""

    def __init__(self):
        self.calls = []

    def __call__(self, context, text, *, focus="", max_calls=None,
                 status=None):
        self.calls.append({"focus": focus})
        if status is not None:
            status.update(available=True, error="")
        return []


class BgSpy:
    """Wraps the REAL background_run_tests, recording each call."""

    def __init__(self):
        self.real = bg_tests_module.background_run_tests
        self.calls: list[dict] = []

    def __call__(self, *args, **kwargs):
        self.calls.append(dict(kwargs))
        return self.real(*args, **kwargs)


def _run_calc_loop(test, fx, **agent_kwargs):
    """One real loop run over the calc fixture (reviewer stubbed)."""
    router = UsageRouter(_calc_handler)
    ctx = StubContext(router, tools=agent_kwargs.pop("inner", None))
    agent = CodingAgent(ctx, root=str(fx.root), **agent_kwargs)
    with mock.patch("nomorals.agents.reasoning.review_text",
                    CleanReviewer()):
        return agent.run("Build a tiny calculator",
                         max_iterations=3, timeout=30, plan_mode="never")


# ── (a) a scope denial refuses the call and settles honestly ─────────


@unittest.skipUnless(shutil.which("git"), "git not available")
class ScopeDenialTests(unittest.TestCase):

    def test_a_run_tests_out_of_scope_settles_not_green(self):
        fx = _calc_repo(FixtureRepo())
        self.addCleanup(fx.cleanup)
        allowlist = tuple(t for t in BUILTIN_ROLES["coder"].tool_allowlist
                          if t != "run_tests")
        spec = RoleSpec(name="coder", tool_allowlist=allowlist)
        inner = FakeInnerRegistry()
        ctx_tools = inner
        router = UsageRouter(_calc_handler)
        ctx = StubContext(router, tools=ctx_tools)
        scoped = RoleEnforcingRegistry(inner, spec, ctx)
        agent = CodingAgent(ctx, root=str(fx.root), tools=scoped)
        with mock.patch("nomorals.agents.reasoning.review_text",
                        CleanReviewer()):
            result = agent.run("Build a tiny calculator",
                               max_iterations=3, timeout=30,
                               plan_mode="never")
        self.assertFalse(result.ok)
        # the drafted file proves the run got as far as verify
        self.assertTrue((fx.root / "calc.py").exists() or
                        result.scope_denials)
        self.assertEqual(len(result.scope_denials), 1)
        denial = result.scope_denials[0]
        self.assertEqual(denial["tool"], "run_tests")
        self.assertIn("run_tests", denial["reason"])
        self.assertIn("allowlist", denial["reason"])
        self.assertIn("run_tests", result.error)
        self.assertEqual(result.to_dict()["scope_denials"],
                         result.scope_denials)
        for step in result.steps:
            self.assertNotEqual(step["status"], "done")

    def test_a_fs_read_out_of_scope_aborts_before_drafting(self):
        fx = _calc_repo(FixtureRepo())
        self.addCleanup(fx.cleanup)
        allowlist = tuple(t for t in BUILTIN_ROLES["coder"].tool_allowlist
                          if t != "fs_read")
        spec = RoleSpec(name="coder", tool_allowlist=allowlist)
        inner = FakeInnerRegistry()
        router = UsageRouter(_calc_handler)
        ctx = StubContext(router, tools=inner)
        scoped = RoleEnforcingRegistry(inner, spec, ctx)
        agent = CodingAgent(ctx, root=str(fx.root), tools=scoped)
        with mock.patch("nomorals.agents.reasoning.review_text",
                        CleanReviewer()):
            result = agent.run("Build a tiny calculator",
                               max_iterations=3, timeout=30,
                               plan_mode="never")
        self.assertFalse(result.ok)
        self.assertIn("fs_read", result.error)
        self.assertTrue(result.scope_denials)
        self.assertEqual(result.scope_denials[0]["tool"], "fs_read")
        # nothing was drafted: the refusal landed at the explore phase
        self.assertFalse((fx.root / "calc.py").exists())
        for step in result.steps:
            self.assertEqual(step["status"], "skipped")
            self.assertIn("fs_read", step["reason"])


# ── (b) an in-scope registry runs exactly as today ───────────────────


@unittest.skipUnless(shutil.which("git"), "git not available")
class ScopedGreenTests(unittest.TestCase):

    def test_b_role_scoped_registry_full_allowlist_goes_green(self):
        fx = _calc_repo(FixtureRepo())
        self.addCleanup(fx.cleanup)
        inner = FakeInnerRegistry()
        scoped = RoleScopedRegistry(inner, "coding", CODING_TOOLS,
                                    CODING_CAPABILITIES)
        result = _run_calc_loop(self, fx, inner=inner, tools=scoped)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.scope_denials, [])
        self.assertIn("def add", (fx.root / "calc.py").read_text())


# ── (c) no scope: byte-identical behavior ────────────────────────────


@unittest.skipUnless(shutil.which("git"), "git not available")
class UnscopedTests(unittest.TestCase):

    def test_c_unscoped_run_is_green_and_denial_free(self):
        fx = _calc_repo(FixtureRepo())
        self.addCleanup(fx.cleanup)
        result = _run_calc_loop(self, fx)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.scope_denials, [])
        self.assertEqual(result.to_dict()["scope_denials"], [])


# ── (d) both bridges forward their scope (spies) ─────────────────────


class SpyLoop:
    """CodingAgent stand-in recording constructor + run kwargs."""

    scripted: CodingResult | None = None
    instances: list["SpyLoop"] = []

    def __init__(self, context, **kwargs):
        self.context = context
        self.init_kwargs = kwargs
        self.run_calls: list = []
        SpyLoop.instances.append(self)

    def run(self, task, **kwargs):
        self.run_calls.append((task, kwargs))
        assert SpyLoop.scripted is not None
        return SpyLoop.scripted


class BridgeForwardingTests(unittest.TestCase):

    def setUp(self):
        SpyLoop.scripted = CodingResult(
            ok=True, iterations=1, files=["a.py"], output="ok",
            verification="tests")
        SpyLoop.instances = []
        patcher = mock.patch.object(coding_module, "CodingAgent", SpyLoop)
        patcher.start()
        self.addCleanup(patcher.stop)

    @property
    def loop(self):
        self.assertEqual(len(SpyLoop.instances), 1)
        return SpyLoop.instances[0]

    def test_d_mission_bridge_forwards_its_role_spec_scope(self):
        inner = FakeInnerRegistry()
        ctx = SimpleNamespace(db=None, tools=inner, memory=None,
                              router=None, settings=SimpleNamespace(),
                              extras={})
        inner.context = ctx
        agent = MissionCodingAgent(context=ctx,
                                   role_spec=BUILTIN_ROLES["coder"])
        agent.work({"goal": "build it"})
        tools = self.loop.init_kwargs.get("tools")
        self.assertIsInstance(tools, RoleEnforcingRegistry)
        self.assertIs(tools.spec, BUILTIN_ROLES["coder"])
        _, run_kwargs = self.loop.run_calls[0]
        self.assertIs(run_kwargs["background_tests"], True)

    def test_d_mission_bridge_unbound_passes_no_tools_kwarg(self):
        ctx = SimpleNamespace(db=None, tools=FakeInnerRegistry(),
                              memory=None, router=None,
                              settings=SimpleNamespace(), extras={})
        agent = MissionCodingAgent(context=ctx)
        agent.work({"goal": "build it"})
        # the pre-#66 construction exactly: no tools kwarg at all, so
        # root-only loop doubles keep working on the unscoped path
        self.assertNotIn("tools", self.loop.init_kwargs)
        _, run_kwargs = self.loop.run_calls[0]
        self.assertIs(run_kwargs["background_tests"], True)

    def test_d_coding_role_agent_forwards_its_scoped_registry(self):
        inner = FakeInnerRegistry()
        ctx = SimpleNamespace(db=None, tools=inner, memory=None,
                              router=None, settings=SimpleNamespace(),
                              extras={})
        inner.context = ctx
        agent = CodingRoleAgent(ctx, inner)
        out = agent.run({"task": "build it"})
        self.assertIs(self.loop.init_kwargs.get("tools"), agent.tools)
        self.assertIsInstance(agent.tools, RoleScopedRegistry)
        _, run_kwargs = self.loop.run_calls[0]
        self.assertIs(run_kwargs["background_tests"], True)
        self.assertEqual(out.output["scope_denials"], [])


# ── (e)+(f) background tests: bridge path + cap derivation ───────────


def _bridge_ctx(router):
    db = Database(":memory:")
    db.migrate()
    return SimpleNamespace(
        db=db, tools=None, memory=None, router=router,
        settings=SimpleNamespace(reasoning_mode="off"), extras={})


@unittest.skipUnless(shutil.which("git"), "git not available")
class MissionBridgeBackgroundTests(unittest.TestCase):

    def test_e_mission_step_runs_background_and_settles_green(self):
        fx = _calc_repo(FixtureRepo())
        self.addCleanup(fx.cleanup)
        spy = BgSpy()
        ctx = _bridge_ctx(UsageRouter(_calc_handler))
        agent = MissionCodingAgent(context=ctx)
        agent.bind_step(PlanStep(
            name="build", goal="Build a tiny calculator", role="coding",
            payload={"root": str(fx.root)}))
        with mock.patch.object(bg_tests_module, "background_run_tests",
                               spy), \
                mock.patch("nomorals.agents.reasoning.review_text",
                           CleanReviewer()):
            out = agent.work("Build a tiny calculator")
        self.assertTrue(out["ok"], out.get("error"))
        self.assertNotIn("error", out)
        # the background path really ran, at the budget-derived cap:
        # bridge timeout is 300 -> max(300 * 5, 60) = 1500, not 900
        self.assertTrue(spy.calls)
        self.assertEqual(spy.calls[0]["cap_seconds"], 1500)
        self.assertIn("def add", (fx.root / "calc.py").read_text())
        self.assertEqual(out["scope_denials"], [])


@unittest.skipUnless(shutil.which("git"), "git not available")
class CapDerivationTests(unittest.TestCase):

    def _run_with_spy(self, fx, **run_kwargs):
        spy = BgSpy()
        router = UsageRouter(_calc_handler)
        agent = CodingAgent(StubContext(router), root=str(fx.root))
        with mock.patch.object(bg_tests_module, "background_run_tests",
                               spy), \
                mock.patch("nomorals.agents.reasoning.review_text",
                           CleanReviewer()):
            result = agent.run("Build a tiny calculator",
                               plan_mode="never", **run_kwargs)
        self.assertTrue(result.ok, result.error)
        self.assertTrue(spy.calls)
        return spy.calls[0]["cap_seconds"]

    def test_f_cap_derives_from_the_run_budget(self):
        fx = _calc_repo(FixtureRepo())
        self.addCleanup(fx.cleanup)
        cap = self._run_with_spy(fx, background_tests=True,
                                 max_iterations=3, timeout=300)
        self.assertEqual(cap, 1500)

    def test_f_cap_derives_at_the_loop_default_budget(self):
        fx = _calc_repo(FixtureRepo())
        self.addCleanup(fx.cleanup)
        cap = self._run_with_spy(fx, background_tests=True,
                                 max_iterations=3, timeout=60)
        self.assertEqual(cap, 300)

    def test_f_explicit_cap_is_honored_verbatim(self):
        fx = _calc_repo(FixtureRepo())
        self.addCleanup(fx.cleanup)
        cap = self._run_with_spy(fx, background_tests=True,
                                 max_iterations=3, timeout=300,
                                 test_cap_seconds=42)
        self.assertEqual(cap, 42)


# ── (g) a capped background run reports capped-with-partials ─────────


@unittest.skipUnless(shutil.which("git"), "git not available")
class CappedBackgroundTests(unittest.TestCase):

    def test_g_capped_run_is_not_green_and_says_so(self):
        fx = FixtureRepo()
        self.addCleanup(fx.cleanup)
        fx.write("tests/__init__.py", "")
        fx.write("tests/test_calc.py",
                 "import calc\n\n\ndef test_add():\n"
                 "    assert calc.add(2, 3) == 5\n")
        fx.write("tests/test_slow.py",
                 "import time\n\n\ndef test_slow():\n"
                 "    time.sleep(30)\n")
        fx.commit_all()
        router = UsageRouter(_calc_handler)
        agent = CodingAgent(StubContext(router), root=str(fx.root))
        with mock.patch("nomorals.agents.reasoning.review_text",
                        CleanReviewer()):
            result = agent.run("Build a tiny calculator",
                               max_iterations=1, timeout=30,
                               plan_mode="never",
                               background_tests=True, test_cap_seconds=2)
        self.assertFalse(result.ok)
        # the capped label rides the run's error — partial results,
        # named as a cap kill, never a silent or mysterious failure
        self.assertIn("cap", result.error.lower())
        self.assertIn("partial", result.error.lower())
        # and it really was cut short, not left to sleep it out
        self.assertLess(result.seconds, 25)


if __name__ == "__main__":
    unittest.main()
