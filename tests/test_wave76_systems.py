"""Wave 76 systems — hermetic verification.

Phase 2 (the eight next-wave systems):
 1. Closed-Loop Self-Improvement — measure → propose → test → verify,
    with a persisted metrics time-series and per-cycle outcome
    accounting (evolution.measure / record_outcome / _outcome_context).
 2. Skill Library — auto-capture of recovered patterns from the
    tool-call ledger (SkillLibrary.capture_from_ledger) + the unified
    operator context (prior art + traps + failure prevention).
 3. Long-term Goal / Mission robustness — transient-error retry with
    backoff, failed-step failure-ledger recording, and the mission
    health report (stuck detection).
 4. Knowledge Graph memory upgrades — duplicate consolidation, time
    decay, label-propagation communities, hub ranking.
 5. Tool Creator improvements — smoke-invoke probe, import-boundary
    validation, and the failure-ledger suggest miner.
 6. Failure Analysis Agent — record() (always-on, deterministic
    learning), auto-recording from the central tool registry, ledger
    stats.
 7. Always-on mid-task reasoning — ReasoningAgent.mid_task_check with
    a persisted journal, wired into evolution.apply / tool install.
 8. Prompt / Mission Structuring sub-agent — deterministic brief
    extraction (intent, subgoals, inputs, constraints, acceptance,
    tools, risks), wired into mission planning + goal creation.

Phase 3 (high-capability upgrades):
 9.  Universal Decoder upgrades — bz2, xz, quoted-printable, single-
     byte XOR (with compressed-data guard), Caesar shift recovery
     (with false-positive guard), embedded-strings extraction.
10. Cookie analysis & handling — the CookieLab (parse, classify,
     service fingerprint, value decoding, security posture, KG
     ingest), the cookie_analyze tool, and decoder integration.
"""
from __future__ import annotations

import base64
import bz2
import gzip
import json
import lzma
import re
import tempfile
import time
import unittest
import zlib
from pathlib import Path
from unittest import mock

from nomorals.agents.context import build_context
from nomorals.cli import _cmd_cookies, _cmd_kg, _cmd_mission, _cmd_structure
from nomorals.core.config import Settings


def _ctx(tmp: str):
    return build_context(Settings(home=tmp), with_executor=False,
                         with_tools=True, with_router=False,
                         with_memory=False)


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _jwt(payload: dict) -> str:
    header = _b64u(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    return ".".join([header, _b64u(json.dumps(payload).encode()), "sig"])


class EvolutionClosedLoopTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="nm-w76-")
        self.ctx = _ctx(self.tmp)
        self.db = self.ctx.db

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_measure_snapshot_and_series(self):
        from nomorals.agents.evolution import EvolutionAgent
        agent = EvolutionAgent(self.ctx, repo_root=Path("."))
        m1 = agent.measure()
        for key in ("tests", "prod_lines", "prod_files", "tools", "skills",
                    "kg_nodes", "kg_edges"):
            self.assertIn(key, m1)
        self.assertGreater(m1["prod_lines"], 50000)
        self.assertGreater(m1["tests"], 1000)
        agent.measure()
        history = agent.metrics_history(10)
        self.assertEqual(len(history), 2)
        self.assertLess(history[0]["ts"], history[1]["ts"])

    def test_record_outcome_applied_and_reverted(self):
        from nomorals.agents.evolution import EvolutionAgent
        agent = EvolutionAgent(self.ctx, repo_root=Path("."))
        agent._last_instruction = "fix the thing"
        out = agent.record_outcome("evo-1", applied=True, commit="abc",
                                   tag="evo/evo-1",
                                   before={"tests": 10, "prod_lines": 100},
                                   after={"tests": 11, "prod_lines": 110})
        self.assertTrue(out["applied"])
        row = self.db.query_one(
            "SELECT * FROM evolution_outcomes WHERE proposal_id=?",
            ("evo-1",))
        self.assertIsNotNone(row)
        self.assertEqual(row["applied"], 1)
        self.assertEqual(row["tests_after"], 11)
        self.assertEqual(row["commit_id"], "abc")
        # a reverted cycle lands in the failure ledger as a lesson
        agent.record_outcome("evo-2", applied=False,
                             reason="verification failed: boom")
        fail = self.db.query_one(
            "SELECT * FROM failures WHERE source='evolution'")
        self.assertIsNotNone(fail)
        self.assertIn("evo-2", fail["summary"])

    def test_outcome_context_carries_signals(self):
        from nomorals.agents.evolution import EvolutionAgent
        agent = EvolutionAgent(self.ctx, repo_root=Path("."))
        agent.measure()
        agent.measure()
        # a recent failure + a lesson matching the instruction
        from nomorals.agents.failure import FailureAnalyzer
        FailureAnalyzer(self.ctx).record("tool", "evolve",
                                         "TimeoutError: connection timed out")
        ctx = agent._outcome_context("fix the timeout bug in evolve")
        self.assertIn("System metrics", ctx)
        self.assertTrue(len(ctx) > 30)


class SkillLibraryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="nm-w76-")
        self.ctx = _ctx(self.tmp)
        self.db = self.ctx.db

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_capture_from_ledger_retry_and_pivot(self):
        from nomorals.agents.skills import SkillLibrary
        lib = SkillLibrary(self.db)
        now = time.time()
        calls = [
            ("s1", "net_fetch", "error", "Connection refused", now - 40),
            ("s2", "net_fetch", "ok", "", now - 35),
            ("s3", "web_search", "error", "TimeoutError: timed out",
             now - 20),
            ("s4", "fetch_page", "ok", "", now - 15),
        ]
        for cid, tool, status, err, ts in calls:
            self.db.execute(
                "INSERT INTO tool_calls (id, actor, tool, capability, "
                "decision, status, error, created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (cid, "op", tool, "", "allow", status, err, ts))
        captured = lib.capture_from_ledger()
        self.assertIn("retry net_fetch after network error", captured)
        self.assertIn("recover web_search via fetch_page (network)",
                      captured)
        # idempotent: re-running upserts, does not duplicate
        before = self.db.query_one("SELECT COUNT(*) AS n FROM skills")["n"]
        lib.capture_from_ledger()
        after = self.db.query_one("SELECT COUNT(*) AS n FROM skills")["n"]
        self.assertEqual(before, after)

    def test_operator_context_combines_sources(self):
        from nomorals.agents.skills import SkillLibrary
        lib = SkillLibrary(self.db)
        lib.save("decode a base64 cookie", kind="strategy",
                 body="base64 then JWT", description="cookie work",
                 tags=["cookie", "base64"])
        block = lib.operator_context("decode a base64 cookie")
        self.assertIn("Relevant proven skills", block)
        self.assertIn("decode a base64 cookie", block)

    def test_skill_tool_context_and_capture(self):
        outcome = self.ctx.tools.call("skill", action="context",
                                      query="anything")
        self.assertTrue(outcome.ok)
        outcome = self.ctx.tools.call("skill", action="capture")
        self.assertTrue(outcome.ok)
        self.assertIn("captured", outcome.value)


class MissionRobustnessTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="nm-w76-")
        self.ctx = _ctx(self.tmp)
        self.db = self.ctx.db

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_health_flags_stuck_running_mission(self):
        from nomorals.missions.mission import MissionStore
        from nomorals.missions.runner import MissionRunner
        store = MissionStore(self.db)
        mission = store.create_new("a long research goal", name="stuck-m")
        store.set_status(mission.id, "running", "started")
        # an old checkpoint: the worker died 1 hour ago
        self.db.execute(
            "INSERT INTO mission_checkpoints (id, mission_id, label, state, "
            "created_at) VALUES (?,?,?,?,?)",
            ("cp-stuck", mission.id, "after:step1", json.dumps({}),
             time.time() - 3600))
        health = MissionRunner(self.ctx).health()
        self.assertEqual(health["stuck"], [mission.id])
        entry = health["active"][0]
        self.assertTrue(entry["stuck"])
        self.assertGreater(entry["checkpoint_age_seconds"], 3000)

    def test_retry_transient_then_success(self):
        import nomorals.agents.roles as roles
        from nomorals.missions.mission import MissionStore
        from nomorals.missions.runner import MissionRunner

        class _FakeResult:
            def __init__(self, ok, output, error="", tokens=0):
                self.ok = ok
                self.output = output
                self.error = error
                self.tokens = tokens

        class _FakeAgent:
            def __init__(self):
                self.runs = 0

            def run(self, prompt):
                self.runs += 1
                if self.runs == 1:
                    raise RuntimeError("Connection refused")
                return _FakeResult(True, {"done": 1})

        store = MissionStore(self.db)
        mission = store.create_new("do the thing", name="retry-m")
        runner = MissionRunner(self.ctx, store=store)
        sleeps = []
        fake = _FakeAgent()
        import nomorals.missions.runner as runner_mod
        orig_sleep = runner_mod.time.sleep
        runner_mod.time.sleep = lambda s: sleeps.append(s)
        try:
            with mock.patch.object(
                    roles, "build_agent",
                    side_effect=lambda role, **kw: fake):
                outcome = runner._execute_step(mission, _Step("s1"))
        finally:
            runner_mod.time.sleep = orig_sleep
        self.assertTrue(outcome.ok)
        self.assertIn("2 attempt(s)", outcome.detail)
        self.assertEqual(fake.runs, 2)
        self.assertTrue(sleeps and sleeps[0] >= 0.5)

    def test_failed_step_records_failure(self):
        import nomorals.agents.roles as roles
        from nomorals.missions.mission import MissionStore
        from nomorals.missions.runner import MissionRunner

        class _FakeAgent:
            def run(self, prompt):
                raise RuntimeError("NameError: x is not defined")

        store = MissionStore(self.db)
        mission = store.create_new("do the thing", name="fail-m")
        runner = MissionRunner(self.ctx, store=store)
        with mock.patch.object(
                roles, "build_agent",
                side_effect=lambda role, **kw: _FakeAgent()):
            outcome = runner._execute_step(mission, _Step("s1"))
        self.assertFalse(outcome.ok)
        fail = self.db.query_one(
            "SELECT * FROM failures WHERE source='mission'")
        self.assertIsNotNone(fail)
        self.assertEqual(fail["family"], "undefined_name")


class _Step:
    def __init__(self, name):
        self.name = name
        self.goal = f"step {name}"
        self.role = "execution"


class KgUpgradeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="nm-w76-")
        self.ctx = _ctx(self.tmp)
        from nomorals.agents.kg import KnowledgeGraph
        self.g = KnowledgeGraph(self.ctx.db)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_consolidate_merges_case_duplicates(self):
        a = self.g.upsert_node("Ada Lovelace", type="person",
                               properties={"note": "first"})
        b = self.g.upsert_node("ada  lovelace", type="person",
                               properties={"note2": "dup"})
        c = self.g.upsert_node("Babbage Engine", type="entity")
        self.g.link(a.id, c.id, "designed")
        self.g.link(b.id, c.id, "used")
        out = self.g.consolidate()
        self.assertEqual(out["merged"], 1)
        nodes = self.ctx.db.query("SELECT * FROM kg_nodes WHERE type='person'")
        self.assertEqual(len(nodes), 1)
        # the edge from the duplicate was rewired onto the survivor —
        # both relations now live on (survivor → engine)
        edges = self.ctx.db.query(
            "SELECT * FROM kg_edges WHERE (src=? OR dst=?) "
            "AND (src=? OR dst=?)",
            (a.id, c.id, a.id, c.id))
        rels = {e["relation"] for e in edges}
        self.assertIn("designed", rels)
        self.assertIn("used", rels)

    def test_communities_finds_clusters(self):
        # two disconnected triples
        a1 = self.g.upsert_node("a1")
        a2 = self.g.upsert_node("a2")
        a3 = self.g.upsert_node("a3")
        self.g.link(a1.id, a2.id, "related_to")
        self.g.link(a2.id, a3.id, "related_to")
        b1 = self.g.upsert_node("b1")
        b2 = self.g.upsert_node("b2")
        self.g.link(b1.id, b2.id, "related_to")
        comms = self.g.communities(limit=5)
        sizes = sorted((c["size"] for c in comms), reverse=True)
        self.assertEqual(sizes[:2], [3, 2])

    def test_top_ranks_by_degree(self):
        hub = self.g.upsert_node("hub")
        for i in range(5):
            n = self.g.upsert_node(f"leaf{i}")
            self.g.link(n.id, hub.id, "points_to")
        self.g.upsert_node("loner")
        top = self.g.top(n=3)
        self.assertEqual(top[0]["label"], "hub")
        self.assertEqual(top[0]["degree"], 5)

    def test_decay_lowers_confidence_and_marks_stale(self):
        n = self.g.upsert_node("fading", properties={"confidence": 1.0})
        self.ctx.db.execute(
            "UPDATE kg_nodes SET last_access=? WHERE id=?",
            (time.time() - 120 * 86400, n.id))
        out = self.g.decay()
        self.assertGreaterEqual(out["updated"], 1)
        node = self.g.get_node(n.id)
        self.assertLess(node.properties["confidence"], 0.5)
        self.assertTrue(node.properties.get("stale"))


class ToolMakerUpgradeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="nm-w76-")
        self.ctx = _ctx(self.tmp)
        self.db = self.ctx.db
        from nomorals.agents.toolmaker import ToolMaker
        self.tm = ToolMaker(self.ctx)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_smoke_invoke_accepts_working_and_needs_args(self):
        code = '''
def register(registry):
    @registry.register("t_ok", description="works", capability="io")
    def t_ok(*, task: str = "") -> dict:
        return {"ok": True}
    @registry.register("t_arg", description="needs arg", capability="io")
    def t_arg(task: str) -> dict:
        return {"task": task}
'''
        out = self.tm.test(code)
        self.assertTrue(out["ok"])
        self.assertEqual(out["smoke"]["t_ok"], "ok")
        self.assertEqual(out["smoke"]["t_arg"], "needs-args")

    def test_smoke_invoke_rejects_broken_tool(self):
        code = '''
def register(registry):
    @registry.register("t_bad", description="broken", capability="io")
    def t_bad(*, task: str = "") -> dict:
        return 1 / 0
'''
        out = self.tm.test(code)
        self.assertFalse(out["ok"])
        self.assertIn("smoke", out.get("error", ""))

    def test_install_refuses_forbidden_imports(self):
        out = self.tm.install("evil_tool", "import subprocess\n"
                                          "def register(r):\n    pass\n",
                              tested=True)
        self.assertFalse(out["ok"])
        self.assertIn("forbidden", out["error"])

    def test_suggest_mines_failures(self):
        now = time.time()
        for i in range(3):
            self.db.execute(
                "INSERT INTO failures (id, source, summary, error, family, "
                "lesson, ts) VALUES (?,?,?,?,?,?,?)",
                (f"f{i}", "tool", "net_fetch attempt",
                 "Connection refused", "network", "", now - i))
        # a flaky tool: 6 calls, 4 errors
        for i in range(6):
            status = "error" if i < 4 else "ok"
            self.db.execute(
                "INSERT INTO tool_calls (id, actor, tool, capability, "
                "decision, status, error, created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (f"c{i}", "op", "flaky_tool", "", "allow", status,
                 "boom", now - i))
        sug = self.tm.suggest(limit=5)
        names = [s["name"] for s in sug]
        self.assertIn("guard_network", names)
        self.assertIn("harden_flaky_tool", names)


class FailureAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="nm-w76-")
        self.ctx = _ctx(self.tmp)
        self.db = self.ctx.db

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_record_learns_known_family_and_dedupes(self):
        from nomorals.agents.failure import FailureAnalyzer
        fa = FailureAnalyzer(self.ctx)
        r1 = fa.record("tool", "decode x", "KeyError: 'foo'")
        self.assertEqual(r1["family"], "missing_key")
        self.assertTrue(r1["learned"])
        r2 = fa.record("tool", "decode x", "KeyError: 'foo'")
        self.assertEqual(r2["times_seen"], 2)
        # only ONE lesson for the deduped failure
        lessons = self.db.query(
            "SELECT * FROM lessons WHERE source='tool'")
        self.assertEqual(len(lessons), 1)
        # the ledger keeps every hit
        hits = self.db.query("SELECT * FROM failures")
        self.assertEqual(len(hits), 2)

    def test_registry_auto_records_tool_failures(self):
        # a tool that genuinely fails → the central registry records it
        def boom(*, x: str = "") -> dict:
            raise ValueError("Connection timed out")

        self.ctx.tools.register("w76_boom", boom, description="always fails",
                                capability="io")
        self.ctx.tools.call("w76_boom")
        row = self.db.query_one(
            "SELECT * FROM failures WHERE source='tool' AND "
            "summary LIKE 'w76_boom%'")
        self.assertIsNotNone(row)
        self.assertEqual(row["family"], "network")

    def test_stats_include_ledger(self):
        from nomorals.agents.failure import FailureAnalyzer
        FailureAnalyzer(self.ctx).record("tool", "a", "KeyError: k")
        FailureAnalyzer(self.ctx).record("mission", "b",
                                         "IndexError: index out of range")
        stats = FailureAnalyzer(self.ctx).stats()
        self.assertEqual(stats["failures"], 2)
        self.assertEqual(stats["by_family"].get("missing_key"), 1)
        self.assertEqual(stats["by_source"].get("mission"), 1)


class ReasoningMidTaskTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="nm-w76-")
        self.ctx = _ctx(self.tmp)
        self.db = self.ctx.db

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_mid_task_check_flags_irreversible_and_journals(self):
        from nomorals.agents.reasoning import ReasoningAgent
        ra = ReasoningAgent(self.ctx)
        out = ra.mid_task_check("apply evolution proposal evo-1 "
                                "(2 edits) to the repo")
        self.assertTrue(out["proceed"])
        self.assertTrue(any("hard to undo" in r for r in out["risks"]))
        journal = ra.mid_task_journal(5)
        self.assertEqual(len(journal), 1)
        self.assertIn("apply evolution", journal[0]["action"])

    def test_mid_task_check_respects_hard_risks(self):
        from nomorals.agents.reasoning import ReasoningAgent
        ra = ReasoningAgent(self.ctx)
        out = ra.mid_task_check("install custom tool",
                                hard_risks=["hard risk: untested code"])
        self.assertFalse(out["proceed"])
        self.assertIn("hard risk: untested code", out["risks"])

    def test_mid_task_check_sees_recent_failures(self):
        from nomorals.agents.failure import FailureAnalyzer
        from nomorals.agents.reasoning import ReasoningAgent
        for i in range(2):
            FailureAnalyzer(self.ctx).record(
                "tool", "publish release v2", "TimeoutError: timed out")
        ra = ReasoningAgent(self.ctx)
        out = ra.mid_task_check("publish release v2", log=False)
        self.assertTrue(any("failed 2x" in r for r in out["risks"]))


class StructuringTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="nm-w76-")
        self.ctx = _ctx(self.tmp)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_deterministic_brief(self):
        from nomorals.agents.structuring import structure_text
        brief = structure_text(
            self.ctx,
            "Build a decoder for the file at /data/x.b64 and send the "
            "result to the owner. It must finish before Friday and never "
            "call the API. Verify the output is valid JSON.",
            for_="mission", polish=False)
        self.assertTrue(brief["intent"])
        self.assertGreaterEqual(len(brief["subgoals"]), 2)
        self.assertIn("/data/x.b64", brief["inputs"])
        self.assertTrue(any("before Friday" in c or "never" in c
                            for c in brief["constraints"]))
        self.assertTrue(any("valid JSON" in a for a in brief["acceptance"]))
        self.assertIn("send", brief["risks"])
        self.assertIn("decoder_agent", brief["tools"])
        self.assertIn("Subgoals:", brief["brief"])

    def test_numbered_bullets_win_as_subgoals(self):
        from nomorals.agents.structuring import structure_text
        brief = structure_text(
            self.ctx,
            "Ship the upgrade.\n1. Update the decoder\n2. Add the tests\n"
            "3. Run the suite",
            for_="mission", polish=False)
        self.assertEqual(len(brief["subgoals"]), 3)
        self.assertEqual(brief["subgoals"][0], "Update the decoder")

    def test_goal_uses_structured_plan(self):
        from nomorals.agents.goals import GoalSystem
        goals = GoalSystem(self.ctx)
        goal = goals.create(
            "ship the upgrade",
            "1. Update the decoder\n2. Add the tests\n3. Run the suite")
        steps = [s.description for s in goal.steps]
        self.assertIn("Update the decoder", steps)
        self.assertIn("Add the tests", steps)
        self.assertIn("Run the suite", steps)

    def test_cli_structure(self):
        import io
        import contextlib
        from nomorals import cli
        args = cli._parser().parse_args(
            ["structure", "Build the tool and verify it works"])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = _cmd_structure(args, self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("Task:", buf.getvalue())


class DecoderUpgradeTest(unittest.TestCase):
    def setUp(self):
        from nomorals.core import decoder
        self.d = decoder
        self.msg = b"the secret is inside the box and the key is under the mat"

    def test_bz2_chain(self):
        rep = self.d.analyze(bz2.compress(self.msg))
        self.assertEqual(rep.best["chain"][0], "bz2")

    def test_xz_chain(self):
        rep = self.d.analyze(lzma.compress(self.msg))
        self.assertEqual(rep.best["chain"][0], "xz")

    def test_zlib_still_gzip(self):
        rep = self.d.analyze(zlib.compress(self.msg))
        self.assertEqual(rep.best["chain"][0], "gzip")

    def test_xor_recovers_key(self):
        key = 0x42
        blob = bytes(b ^ key for b in self.msg)
        rep = self.d.analyze(blob)
        self.assertEqual(rep.best["chain"][0], "xor")
        self.assertIn(f"0x{key:02x}", rep.best["note"])
        self.assertIn("the secret", str(rep.best["output"]))

    def test_xor_skips_compressed_data(self):
        blob = lzma.compress(self.msg)
        hits = self.d.analyze(blob)
        self.assertNotIn("xor", [h.decoder for h in hits.hits])

    def test_xor_needs_real_english(self):
        # high-entropy bytes with no recoverable english
        blob = bytes(range(1, 49))
        hits = self.d.analyze(blob)
        self.assertNotIn("xor", [h.decoder for h in hits.hits])

    def test_caesar_recovers_shift(self):
        plain = "the secret is inside the room"
        cipher = "".join(
            chr((ord(c) - 97 + 7) % 26 + 97) if c.isalpha() else c
            for c in plain)
        rep = self.d.analyze(cipher)
        caesar = [h for h in rep.hits if h.decoder == "caesar"]
        self.assertTrue(caesar)
        self.assertEqual(caesar[0].output, plain)

    def test_caesar_rejects_non_ciphertext(self):
        # already-plain text must not be 'decoded' into a shift
        rep = self.d.analyze("pbh ghuvgr vf ynivgg uvk ohb")
        self.assertNotIn("caesar", [h.decoder for h in rep.hits])

    def test_quoted_printable(self):
        rep = self.d.analyze("The=20quick=20brown=20fox=20jumps=0A")
        qp = [h for h in rep.hits if h.decoder == "quoted-printable"]
        self.assertTrue(qp)
        self.assertIn("The quick brown fox jumps", qp[0].output)

    def test_strings_extraction(self):
        blob = (b"\x00\x01\x02some embedded string here\x00\x00"
                b"http://example.com/abc\x00\xff\xfe")
        rep = self.d.analyze(blob)
        strings = [h for h in rep.hits if h.decoder == "strings"]
        self.assertTrue(strings)
        joined = " ".join(strings[0].output)
        self.assertIn("some embedded string here", joined)
        self.assertIn("http://example.com/abc", joined)


class CookieLabTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="nm-w76-")
        self.ctx = _ctx(self.tmp)
        self.db = self.ctx.db
        from nomorals.core.cookies import CookieLab
        self.lab = CookieLab()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    HDR = ("PHPSESSID=abc123; path=/; HttpOnly; Secure\n"
           "JSESSIONID=xyz; Domain=shop.example.com\n"
           "XSRF-TOKEN=token123\n"
           "_ga=GA1.2.123456789\n"
           "session_blob=" +
           base64.urlsafe_b64encode(json.dumps({"uid": 42}).encode())
           .rstrip(b"=").decode())

    def test_parse_classify_fingerprint(self):
        rep = self.lab.report(self.HDR)
        self.assertEqual(rep["count"], 5)
        self.assertIn("php", rep["services"])
        self.assertIn("java-servlet", rep["services"])
        self.assertIn("google-analytics", rep["services"])
        kinds = {c["name"]: c["kind"] for c in rep["cookies"]}
        self.assertEqual(kinds["PHPSESSID"], "session")
        self.assertEqual(kinds["XSRF-TOKEN"], "csrf")
        self.assertEqual(kinds["_ga"], "tracking")

    def test_flag_parsing(self):
        cookies = self.lab.parse(self.HDR)
        php = [c for c in cookies if c.name == "PHPSESSID"][0]
        self.assertIn("httponly", php.flags)
        self.assertIn("secure", php.flags)
        self.assertEqual(php.flags["path"], "/")

    def test_decode_value_base64_json(self):
        rep = self.lab.report(self.HDR)
        sb = [c for c in rep["cookies"] if c["name"] == "session_blob"][0]
        self.assertEqual(sb["decoded_value"], {"uid": 42})
        self.assertEqual(sb["decode_via"], "base64-json")

    def test_decode_value_jwt(self):
        token = _jwt({"sub": "user42", "email": "a@b.co"})
        rep = self.lab.report("authjwt=" + token)
        c = rep["cookies"][0]
        self.assertEqual(c["kind"], "jwt")
        self.assertEqual(c["decoded_value"]["payload"]["sub"], "user42")

    def test_decode_value_url(self):
        rep = self.lab.report("pref=name%3Dada%26theme%3Ddark")
        c = rep["cookies"][0]
        self.assertEqual(c["decode_via"], "url")
        self.assertEqual(c["decoded_value"], "name=ada&theme=dark")

    def test_security_posture(self):
        rep = self.lab.report(self.HDR)
        # PHPSESSID has HttpOnly+Secure → not plaintext
        self.assertNotIn("PHPSESSID", rep["security"]["plaintext_auth"])
        # JSESSIONID (session, no flags) IS plaintext
        self.assertIn("JSESSIONID", rep["security"]["plaintext_auth"])
        self.assertGreaterEqual(rep["security"]["with_httponly"], 1)

    def test_gzip_blob_value(self):
        blob = base64.urlsafe_b64encode(
            gzip.compress(json.dumps({"n": 7}).encode())).decode()
        rep = self.lab.report("blob=" + blob)
        c = rep["cookies"][0]
        self.assertEqual(c["decoded_value"], {"n": 7})
        self.assertEqual(c["decode_via"], "gzip-json")

    def test_ingest_feeds_graph(self):
        out = self.lab.ingest(self.ctx, self.HDR, source="test-cookies")
        self.assertGreaterEqual(out["nodes"], 3)
        from nomorals.agents.kg import KnowledgeGraph
        stats = KnowledgeGraph(self.db).stats()
        self.assertGreater(stats["nodes"], 0)
        # the service node exists and is linked from the source
        node = KnowledgeGraph(self.db).find_node("php", type="entity")
        self.assertIsNotNone(node)

    def test_ingest_jwt_claims_become_persons(self):
        token = _jwt({"sub": "ada@example.com", "email": "ada@example.com"})
        self.lab.ingest(self.ctx, "authjwt=" + token, source="jwt-test")
        from nomorals.agents.kg import KnowledgeGraph
        node = KnowledgeGraph(self.db).find_node("ada@example.com",
                                                 type="person")
        self.assertIsNotNone(node)


class CookieIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="nm-w76-")
        self.ctx = _ctx(self.tmp)
        self.db = self.ctx.db

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_cookie_analyze_tool(self):
        out = self.ctx.tools.call(
            "cookie_analyze",
            text="PHPSESSID=abc; path=/; HttpOnly\nXSRF-TOKEN=t1")
        self.assertTrue(out.ok)
        rep = out.value["report"]
        self.assertEqual(rep["count"], 2)
        self.assertIn("php", rep["services"])

    def test_cookie_tool_ingest(self):
        out = self.ctx.tools.call(
            "cookie_analyze", action="ingest", source="tool-test",
            text="PHPSESSID=abc; path=/; HttpOnly\nJSESSIONID=xyz")
        self.assertTrue(out.ok)
        self.assertGreaterEqual(out.value["ingested"]["nodes"], 2)

    def test_universal_decoder_includes_cookie_class(self):
        from nomorals.core import decoder
        rep = decoder.analyze(
            "PHPSESSID=abc123; path=/; HttpOnly\n_ga=GA1.2.123")
        cookie_hits = [h for h in rep.hits if h.decoder == "cookies"]
        self.assertTrue(cookie_hits)
        entries = [e for e in cookie_hits[0].output
                   if isinstance(e, dict) and "name" in e]
        by_name = {e["name"]: e for e in entries}
        self.assertEqual(by_name["PHPSESSID"]["class"], "session")
        self.assertEqual(by_name["PHPSESSID"]["service"], "php")
        self.assertEqual(by_name["_ga"]["class"], "tracking")
        # backward-compatible report cookies field
        self.assertGreaterEqual(len(rep.cookies), 2)


class Wave76CliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="nm-w76-")
        self.ctx = _ctx(self.tmp)
        self.db = self.ctx.db

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_cli_cookies(self):
        import io
        import contextlib
        from nomorals import cli
        args = cli._parser().parse_args(
            ["cookies", "PHPSESSID=abc; HttpOnly\nJSESSIONID=xyz"])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = _cmd_cookies(args, self.ctx)
        self.assertEqual(rc, 0)
        text = buf.getvalue()
        self.assertIn("2 parsed", text)
        self.assertIn("PHPSESSID", text)
        self.assertIn("security:", text)

    def test_cli_cookies_ingest(self):
        import io
        import contextlib
        from nomorals import cli
        args = cli._parser().parse_args(
            ["cookies", "ingest PHPSESSID=abc; HttpOnly"])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = _cmd_cookies(args, self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("ingested:", buf.getvalue())

    def test_cli_mission_health(self):
        import io
        import contextlib
        from nomorals import cli
        args = cli._parser().parse_args(["mission", "health"])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = _cmd_mission(args, self.ctx)
        self.assertEqual(rc, 0)
        self.assertIn("mission health", buf.getvalue())

    def test_cli_kg_actions(self):
        import io
        import contextlib
        from nomorals import cli
        # seed the graph
        from nomorals.agents.kg import KnowledgeGraph
        g = KnowledgeGraph(self.db)
        g.upsert_node("Alpha One")
        g.upsert_node("alpha one")
        g.upsert_node("Beta", type="person")
        for action in ("consolidate", "communities", "top", "decay"):
            args = cli._parser().parse_args(["kg", action])
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = _cmd_kg(args, self.ctx)
            self.assertEqual(rc, 0, f"kg {action}")
        # consolidate actually merged the duplicate
        args = cli._parser().parse_args(["kg", "stats"])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _cmd_kg(args, self.ctx)
        self.assertIn("2 nodes", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
