"""Prompt 02 acceptance tests: agent swarm (roles, debate, fan-out, telemetry).

Covers the prompt's acceptance criteria:
 1. `nm swarm roles` lists built-in roles with allowlists and budgets.
 2. A critic-role agent attempting a file write gets a structured
    ToolDenied; the denial is logged + categorized as `tool_denied`.
 3. A debate over buggy code ends in `approve` only after the fix.
 4. A non-converging debate returns `unresolved` with a full transcript.
 5. fan_out (3 angles) + fan_in dedupes findings, keeps sources.
 6. A tied vote triggers arbitration; outcome + rationale recorded.
 7. Old-style string roles with no RoleRegistry behave exactly as before.
 8. stats() shows per-role cost/latency/success.
 9. (full suite green — verified by the suite run itself)
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

from nomorals.agents.blackboard import Blackboard
from nomorals.agents.debate import (
    Debate, Critique, Issue, WorkArtifact, arbitrate,
)
from nomorals.agents.fanout import (
    fan_in, fan_out, map_reduce,
    fan_out_compare, synthesize_comparison, compare, fanout_cap,
    render_comparison_table, FANOUT_WORKERS_BY_PROFILE,
)
from nomorals.agents.orchestrator import MasterOrchestrator, Plan, PlanStep
from nomorals.agents.role_specs import (
    RoleEnforcingRegistry, RoleRegistry, SwarmAgent, default_registry,
)
from nomorals.agents.runtime import HybridExecutor
from nomorals.core.errors import ToolDenied
from nomorals.storage.db import Database
from nomorals.tools.registry import ToolRegistry


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register_builtins()
    return reg


def _context(db: Database | None = None, tools: ToolRegistry | None = None):
    events: list[tuple] = []
    return SimpleNamespace(
        db=db, tools=tools or _registry(), router=None,
        blackboard=Blackboard(),
        emit=lambda topic, **kw: events.append((topic, kw)),
        _events=events,
    )


class RoleSpecTests(unittest.TestCase):
    def test_builtin_roles_present(self):
        reg = default_registry()
        for name in ("researcher", "coder", "critic", "architect",
                     "tester", "writer"):
            spec = reg.resolve(name)
            self.assertEqual(spec.name, name)
            self.assertTrue(spec.tool_allowlist, name)

    def test_unknown_role_falls_back_to_safe_execution(self):
        reg = RoleRegistry()
        spec = reg.resolve("definitely_not_a_role")
        self.assertEqual(spec.name, "execution")
        self.assertTrue(spec.read_only)
        self.assertNotIn("fs_write", spec.tool_allowlist)
        self.assertNotIn("shell_run", spec.tool_allowlist)

    def test_role_aliases(self):
        reg = RoleRegistry()
        self.assertEqual(reg.resolve("research").name, "researcher")
        self.assertEqual(reg.resolve("coding").name, "coder")

    def test_critic_is_read_only(self):
        spec = default_registry().resolve("critic")
        self.assertTrue(spec.read_only)
        for tool in spec.tool_allowlist:
            self.assertNotIn(tool, ("fs_write", "edit_file", "apply_patch",
                                    "shell_run", "python_run", "run_tests"))

    def test_yaml_roles(self):
        import textwrap

        reg = RoleRegistry()
        with tempfile.NamedTemporaryFile("w", suffix=".yaml",
                                         delete=False) as fh:
            fh.write(textwrap.dedent("""\
                roles:
                  - name: analyst
                    description: test role
                    tool_allowlist: [web_search, fs_read]
                    read_only: true
                    output_contract: [findings]
                    budget: {wall_seconds: 120, tokens: 50000}
                """))
            path = fh.name
        try:
            count = reg.load_yaml(path)
        finally:
            os.unlink(path)
        self.assertEqual(count, 1)
        spec = reg.resolve("analyst")
        self.assertTrue(spec.read_only)
        self.assertEqual(spec.budget.tokens, 50000)


class ToolDeniedTests(unittest.TestCase):
    """Acceptance: critic attempting a file write gets structured ToolDenied."""

    def _critic(self, db=None):
        ctx = _context(db=db)
        return SwarmAgent(ctx, default_registry().resolve("critic"),
                          registry=ctx.tools), ctx

    def test_critic_write_denied_structured(self):
        agent, _ = self._critic()
        outcome = agent.call_tool("fs_write", path="/tmp/x.txt",
                                  content="hello")
        self.assertFalse(outcome.ok)
        self.assertIsInstance(outcome.error, ToolDenied)
        self.assertEqual(outcome.error.role, "critic")
        self.assertEqual(outcome.error.tool, "fs_write")

    def test_critic_cannot_escalate_via_injection(self):
        """Even an allowlisted-by-mistake mutating tool is denied for
        read-only roles — enforcement is in code, not in the prompt."""
        agent, _ = self._critic()
        # shell_run is NOT on the critic allowlist at all:
        outcome = agent.call_tool("shell_run", command="echo pwned")
        self.assertFalse(outcome.ok)
        self.assertIsInstance(outcome.error, ToolDenied)

    def test_denial_logged_and_categorized(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        db = Database(os.path.join(tmp, "t.db"))
        db.migrate()
        agent, _ = self._critic(db=db)
        agent.call_tool("fs_write", path="/tmp/x.txt", content="hello")
        self.assertEqual(agent.tools.denial_count(), 1)
        rows = db.execute(
            "SELECT family, source FROM failures WHERE family='tool_denied'"
        ).fetchall()
        self.assertTrue(rows, "denial must land in the failure ledger")
        self.assertEqual(rows[0][0], "tool_denied")

    def test_tester_path_guard(self):
        ctx = _context()
        agent = SwarmAgent(ctx, default_registry().resolve("tester"),
                           registry=ctx.tools)
        # outside test paths → denied even though fs_write is allowlisted
        bad = agent.call_tool("fs_write", path="src/main.py",
                              content="x")
        self.assertFalse(bad.ok)
        self.assertIsInstance(bad.error, ToolDenied)
        self.assertEqual(bad.error.reason, "path_guard")

    def test_writer_markdown_only(self):
        ctx = _context()
        agent = SwarmAgent(ctx, default_registry().resolve("writer"),
                           registry=ctx.tools)
        bad = agent.call_tool("fs_write", path="src/main.py", content="x")
        self.assertFalse(bad.ok)
        self.assertIsInstance(bad.error, ToolDenied)


class DebateTests(unittest.TestCase):
    """Acceptance: debate converges only after the fix; stalemates surface."""

    def _buggy_debate(self, max_rounds=3):
        state = {"fixed": False}

        def coder_fn(brief, feedback, artifact):
            if feedback and not state["fixed"]:
                state["fixed"] = True
                return WorkArtifact(
                    content="def add(a, b): return a + b  # fixed",
                    summary="fixed the off-by-one",
                    addresses=[i.id for i in feedback])
            if state["fixed"]:
                return WorkArtifact(
                    content="def add(a, b): return a + b  # fixed",
                    summary="already fixed", addresses=[])
            return WorkArtifact(content="def add(a, b): return a + b + 1  # BUG",
                                summary="first attempt")

        def critic_fn(artifact, rubric):
            if "BUG" in str(artifact.content):
                return Critique(
                    verdict="request_changes", score=40.0,
                    issues=[Issue(id="ISSUE-1", severity="critical",
                                  location="add()",
                                  detail="off-by-one: adds 1 too many")])
            return Critique(verdict="approve", score=95.0, issues=[])

        board = Blackboard()
        debate = Debate(coder_fn=coder_fn, critic_fn=critic_fn,
                        blackboard=board, max_rounds=max_rounds)
        return debate, board

    def test_debate_approves_only_after_fix(self):
        debate, board = self._buggy_debate()
        result = debate.run("write add()")
        self.assertEqual(result.verdict, "approved")
        self.assertEqual(result.rounds, 2)
        self.assertNotIn("BUG", str(result.final_artifact.content))
        # transcript recorded on the blackboard under the debate topic
        entries = board.topic(debate.debate_id)
        self.assertTrue(any("round.1.coder" in k for k in entries))
        self.assertTrue(any("round.2.critic" in k for k in entries))
        self.assertIn(f"{debate.debate_id}.result", entries)

    def test_debate_unresolved_when_no_convergence(self):
        def coder_fn(brief, feedback, artifact):
            return WorkArtifact(content="same broken work",
                                summary="no change", addresses=[])

        def critic_fn(artifact, rubric):
            # fresh issue id every round → no stalemate trigger, just rounds
            n = len(artifact.addresses)  # unused; use a counter instead
            return Critique(verdict="request_changes", score=30.0,
                            issues=[Issue(id=f"ISSUE-{id(artifact) % 997}",
                                          severity="major", location="x",
                                          detail="still broken")])

        board = Blackboard()
        debate = Debate(coder_fn=coder_fn, critic_fn=critic_fn,
                        blackboard=board, max_rounds=2)
        result = debate.run("impossible task")
        self.assertEqual(result.verdict, "unresolved")
        self.assertEqual(result.rounds, 2)
        self.assertEqual(len(result.transcript), 2)

    def test_stalemate_escalates_to_arbitration(self):
        def coder_fn(brief, feedback, artifact):
            # stubborn: never addresses the issue
            return WorkArtifact(content="unchanged", summary="wontfix",
                                addresses=[])

        def critic_fn(artifact, rubric):
            return Critique(
                verdict="request_changes", score=35.0,
                issues=[Issue(id="ISSUE-9", severity="critical",
                              location="core", detail="never fixed")])

        debate = Debate(coder_fn=coder_fn, critic_fn=critic_fn,
                        max_rounds=5)
        result = debate.run("stubborn work")
        self.assertEqual(result.verdict, "needs_arbitration")
        self.assertIn("ISSUE-9", result.reason)


class FanOutTests(unittest.TestCase):
    """Acceptance: 3-angle fan-out + fan-in dedupes, keeps sources."""

    def test_fanout_fanin_concat_dedupe(self):
        def worker(angle, agent):
            return {
                "angle": angle,
                "findings": [
                    "Python 3.12 released",          # duplicate across angles
                    f"unique insight from {angle}",
                ],
                "sources": [f"https://example.com/{angle}"],
            }

        board = Blackboard()
        ctx = _context()
        res = fan_out("what's new", ["docs", "community", "edge cases"],
                      worker_fn=worker, blackboard=board, context=ctx)
        self.assertEqual(len(res.angles), 3)
        self.assertEqual(len(res.results), 3)
        self.assertFalse(res.errors)

        merged = fan_in(res.run_id, "concat_dedupe", blackboard=board)
        findings = merged.merged["findings"]
        texts = [f["text"] for f in findings]
        # near-duplicate collapsed, uniques kept, sources preserved
        self.assertEqual(texts.count("Python 3.12 released"), 1)
        self.assertEqual(len(findings), 4)
        self.assertEqual(len(merged.merged["sources"]), 3)

    def test_vote_strategy(self):
        board = Blackboard()
        board.post("v1.a", {"label": "yes", "confidence": 0.9}, topic="v1")
        board.post("v1.b", {"label": "no", "confidence": 0.6}, topic="v1")
        board.post("v1.c", {"label": "yes", "confidence": 0.7}, topic="v1")
        res = fan_in("v1", "vote", blackboard=board)
        self.assertEqual(res.merged["winner"], "yes")
        self.assertFalse(res.merged["tied"])

    def test_map_reduce(self):
        res = map_reduce(
            list(range(10)),
            worker_fn=lambda chunk, i: sum(chunk),
            reduce_fn=lambda parts: sum(parts),
            k=3)
        self.assertEqual(res.reduced, 45)
        self.assertFalse(res.errors)


class CompareFanOutTests(unittest.TestCase):
    """Extension #6: Hark-style N-way fan-out — one worker per source,
    profile-gated parallelism (workstation 36 / laptop 12 / termux 4),
    synthesized into a comparison table."""

    def test_profile_caps(self):
        self.assertEqual(fanout_cap("workstation"), 36)
        self.assertEqual(fanout_cap("laptop"), 12)
        self.assertEqual(fanout_cap("termux"), 4)
        self.assertEqual(fanout_cap("bogus-profile"), 12)
        self.assertEqual(FANOUT_WORKERS_BY_PROFILE["workstation"], 36)

    def test_results_in_source_order(self):
        def worker(source, i):
            return {"source": source, "price": 10 * (i + 1)}

        res = fan_out_compare("compare widgets", ["c.com", "a.com", "b.com"],
                              worker_fn=worker, profile="termux")
        self.assertEqual(len(res.results), 3)
        self.assertEqual([r["source"] for r in res.results],
                         ["c.com", "a.com", "b.com"])
        self.assertEqual(res.results[2]["price"], 30)
        self.assertFalse(res.errors)
        self.assertLessEqual(res.workers_used, 4)

    def test_one_bad_source_does_not_kill_run(self):
        def worker(source, i):
            if source == "bad.com":
                raise RuntimeError("fetch exploded")
            return {"price": 99}

        res = fan_out_compare("compare", ["good.com", "bad.com"],
                              worker_fn=worker)
        self.assertEqual(len(res.results), 2)
        self.assertEqual(res.results[0]["price"], 99)
        self.assertIn("error", res.results[1])
        self.assertEqual(len(res.errors), 1)
        self.assertIn("bad.com", res.errors[0])

    def test_max_workers_override_caps_pool(self):
        seen_max = [0]
        active = [0]
        import threading
        lock = threading.Lock()

        def worker(source, i):
            with lock:
                active[0] += 1
                seen_max[0] = max(seen_max[0], active[0])
            import time as _t
            _t.sleep(0.05)
            with lock:
                active[0] -= 1
            return {"source": source}

        res = fan_out_compare("compare", [f"s{i}.com" for i in range(8)],
                              worker_fn=worker, max_workers=2)
        self.assertEqual(res.workers_used, 2)
        self.assertLessEqual(seen_max[0], 2)

    def test_synthesize_unions_aspects(self):
        results = [
            {"price": 100, "rating": 4.5},
            {"price": 120, "warranty": "2y"},
        ]
        synth = synthesize_comparison(results, ["a.com", "b.com"])
        aspects = [r["aspect"] for r in synth["table"]]
        self.assertEqual(sorted(aspects), ["price", "rating", "warranty"])
        self.assertEqual(synth["coverage"]["price"], 2)
        self.assertEqual(synth["coverage"]["warranty"], 1)
        md = synth["markdown"]
        self.assertIn("a.com", md)
        self.assertIn("—", md)  # missing value placeholder

    def test_render_table_empty(self):
        self.assertEqual(render_comparison_table([]), "")

    def test_compare_end_to_end(self):
        def worker(source, i):
            if i == 2:
                raise ValueError("timeout")
            return {"price": 50 + i, "stock": "yes"}

        out = compare("compare phones", ["x.com", "y.com", "z.com"],
                      worker_fn=worker)
        self.assertEqual(out["n_sources"], 3)
        self.assertEqual(out["n_ok"], 2)
        self.assertIn("z.com", out["failed_sources"])
        self.assertIn("failed: z.com", out["summary"])
        self.assertIn("| aspect |", out["markdown"])

    def test_never_raises_on_garbage(self):
        res = fan_out_compare(None, None)
        self.assertIsNotNone(res)
        res = fan_out_compare("g", ["ok"], worker_fn=lambda s, i: 1 / 0)
        self.assertEqual(len(res.errors), 1)
        synth = synthesize_comparison(None)
        self.assertEqual(synth["n_sources"], 0)
        out = compare(None, None)
        self.assertIn("summary", out)

    def test_default_worker_when_none_supplied(self):
        res = fan_out_compare("g", ["a.com"])
        self.assertEqual(len(res.results), 1)
        self.assertFalse(res.errors)


class ArbitrationTests(unittest.TestCase):
    """Acceptance: tied votes trigger arbitration; outcome recorded."""

    def test_tied_vote_triggers_arbitration(self):
        board = Blackboard()
        board.post("t.a", {"label": "buy", "confidence": 0.8}, topic="t")
        board.post("t.b", {"label": "sell", "confidence": 0.8}, topic="t")
        voted = fan_in("t", "vote", blackboard=board)
        self.assertTrue(voted.merged["tied"])

        positions = [
            {"label": "buy", "decision": "buy",
             "evidence": "momentum strong"},
            {"label": "sell", "decision": "sell",
             "evidence": "overbought RSI"},
        ]

        def judge_fn(positions, rubric):
            return {"decision": "buy",
                    "rationale": "momentum outweighs RSI on this timeframe",
                    "notes": "judged on correctness"}

        result = arbitrate(positions, judge_fn=judge_fn,
                           blackboard=board, topic="t")
        self.assertEqual(result.decision, "buy")
        self.assertEqual(result.level, 1)
        self.assertIn("momentum", result.rationale)
        # recorded on the blackboard
        keys = board.keys("arb-*")
        self.assertTrue(any(k.endswith(".result") for k in keys))


class OrchestratorSwarmTests(unittest.TestCase):
    """Acceptance: backward compat + role telemetry."""

    def _orch(self, roles=None):
        ctx = _context()
        return MasterOrchestrator(
            ctx, executor=HybridExecutor(threads=2, use_processes=False),
            roles=roles), ctx

    def test_backward_compat_no_role_registry(self):
        """Old-style string roles, no RoleSpecs: behaves exactly as before."""
        orch, _ = self._orch(roles=None)
        plan = Plan(goal="g", steps=[
            PlanStep(name="research", goal="find things", role="research"),
            PlanStep(name="mystery", goal="unknown role",
                     role="frobnicate"),
        ])
        handlers = {
            "research": lambda task: {"findings": ["a"]},
        }
        result = orch.run("g", plan=plan, handlers=handlers, reflect=False)
        self.assertTrue(result.ok)
        # unknown role with no registry → old "no handler" fallthrough
        self.assertEqual(result.report.results["mystery"]["status"],
                         "no handler")

    def test_roles_wired_unknown_gets_safe_spec(self):
        orch, _ = self._orch(roles=RoleRegistry())
        plan = Plan(goal="g", steps=[
            PlanStep(name="mystery", goal="unknown role",
                     role="frobnicate"),
        ])
        result = orch.run("g", plan=plan, handlers={}, reflect=False)
        # safe execution spec: honest stub, never full tool access
        out = result.report.results["mystery"]
        self.assertIn("status", out)

    def test_per_role_telemetry(self):
        orch, ctx = self._orch(roles=RoleRegistry())
        plan = Plan(goal="g", steps=[
            PlanStep(name="r1", goal="a", role="research"),
            PlanStep(name="r2", goal="b", role="research"),
            PlanStep(name="c1", goal="c", role="critic"),
        ])
        handlers = {
            "research": lambda task: {"findings": ["x"]},
            "critic": lambda task: {"verdict": "approve"},
        }
        orch.run("g", plan=plan, handlers=handlers, reflect=False)
        stats = orch.stats()
        roles = stats["roles"]
        self.assertEqual(roles["research"]["tasks"], 2)
        self.assertEqual(roles["critic"]["tasks"], 1)
        self.assertEqual(roles["research"]["success_rate"], 1.0)
        self.assertIn("avg_seconds", roles["research"])
        self.assertIn("denials", roles["research"])
        self.assertIn("tool_denials", stats)
        # task lifecycle events emitted on the bus
        topics = [t for t, _ in ctx._events]
        self.assertIn("swarm.task_done", topics)


class SwarmCliTests(unittest.TestCase):
    """Acceptance: `nm swarm roles` lists roles, allowlists, budgets."""

    def test_swarm_roles_lists_everything(self):
        from nomorals.cli import _cmd_swarm

        args = argparse.Namespace(swarm_action="roles", json=True)
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_swarm(args, _context())
        self.assertEqual(rc, 0)
        import json as _json

        payload = _json.loads(buf.getvalue())
        roles = payload["roles"]
        for name in ("researcher", "coder", "critic", "architect",
                     "tester", "writer"):
            self.assertIn(name, roles)
            self.assertTrue(roles[name]["tools"])
            self.assertIn("budget", roles[name])


class SpecBoundLegacyAgentTests(unittest.TestCase):
    """The pre-existing role agents enforce a bound RoleSpec.

    One rule on every path: whether a call goes through SwarmAgent or
    straight through a legacy agent, the locked contract holds.
    """

    def _bound(self, legacy_role: str, spec_name: str, db=None):
        from nomorals.agents.roles import build_agent
        ctx = _context(db=db)
        return (build_agent(legacy_role, context=ctx,
                            role_spec=default_registry().resolve(spec_name)),
                ctx)

    def test_researcher_spec_denies_write_despite_legacy_caps(self):
        # Legacy ResearchAgent declares FS_WRITE in required_capabilities;
        # the researcher spec forbids all writes. Spec wins.
        agent, _ = self._bound("research", "researcher")
        with self.assertRaises(ToolDenied) as cm:
            agent._call_tool("fs_write", path="/tmp/x.txt", content="hi")
        self.assertEqual(cm.exception.reason, "not_allowlisted")
        self.assertEqual(cm.exception.role, "researcher")

    def test_critic_spec_denies_db_tool_despite_legacy_caps(self):
        # Legacy CriticAgent has DB_READ; the critic spec has no db tools.
        agent, _ = self._bound("critic", "critic")
        with self.assertRaises(ToolDenied):
            agent._call_tool("db_query", sql="SELECT 1")

    def test_read_only_blocks_allowlisted_mutating_tool(self):
        # Even a mistakenly-allowlisted mutating tool is denied for
        # read-only roles — enforcement is in code, not in the prompt.
        from nomorals.agents.role_specs import RoleSpec
        from nomorals.agents.roles import build_agent
        spec = RoleSpec(name="auditor", description="read-only test",
                        tool_allowlist=("fs_write", "fs_read"),
                        read_only=True, output_contract=())
        agent = build_agent("execution", context=_context(), role_spec=spec)
        with self.assertRaises(ToolDenied) as cm:
            agent._call_tool("fs_write", path="/tmp/x.txt", content="hi")
        self.assertEqual(cm.exception.reason, "read_only")

    def test_architect_cannot_write_via_execution_agent(self):
        # architect spec is read-only; it delegates to legacy
        # ExecutionAgent which declares FS_WRITE. The spec must hold.
        agent, _ = self._bound("execution", "architect")
        with self.assertRaises(ToolDenied):
            agent._call_tool("fs_write", path="/tmp/x.txt", content="hi")

    def test_unbound_legacy_agent_unchanged(self):
        from nomorals.agents.roles import build_agent
        agent = build_agent("research", context=_context())
        self.assertIsNone(agent.role_spec)

    def test_denial_recorded_in_ledger_from_legacy_path(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        db = Database(os.path.join(tmp, "t.db"))
        db.migrate()
        agent, _ = self._bound("research", "researcher", db=db)
        with self.assertRaises(ToolDenied):
            agent._call_tool("fs_write", path="/tmp/x.txt", content="hi")
        rows = db.execute(
            "SELECT family, source FROM failures WHERE family='tool_denied'"
        ).fetchall()
        self.assertTrue(rows)
        self.assertEqual(rows[0][1], "role")

    def test_roles_package_reexports_spec_api(self):
        from nomorals.agents import roles as roles_pkg
        import nomorals.agents.role_specs as rs
        for name in ("RoleSpec", "RoleRegistry", "SwarmAgent",
                     "default_registry"):
            self.assertIs(getattr(roles_pkg, name), getattr(rs, name), name)

    def test_grant_covers_allowlist_no_silent_denials(self):
        # Every allowlisted tool's capability must be grantable: a locked
        # allowlist must never silently deny one of its own tools.
        reg = _registry()
        registry = default_registry()
        for name in registry.names():
            spec = registry.resolve(name)
            grant = spec.capabilities
            for tool in spec.tool_allowlist:
                ts = reg.get(tool)
                self.assertIsNotNone(ts, f"{name}: {tool} not registered")
                cap = ts.capability or ""
                if grant is not None and cap:
                    self.assertTrue(
                        grant.grants(cap),
                        f"{name}: allowlisted {tool} (cap {cap}) not granted")


if __name__ == "__main__":
    unittest.main()
