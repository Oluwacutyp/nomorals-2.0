"""Wave 78 systems: barcode/gift-card recovery, permanent reasoning
course-correction + self-challenge, orchestrator routing / arbitration /
supervision.  All hermetic — no model, no network."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from unittest import mock

from nomorals.agents.context import build_context
from nomorals.agents.orchestrator import MasterOrchestrator, Plan, PlanStep
from nomorals.agents.reasoning import ReasoningAgent
from nomorals.agents.runtime import HybridExecutor
from nomorals.agents.tasks import Task, TaskGraph, TaskKind
from nomorals.agents.base import AgentResult
from nomorals.core import barcode as bc
from nomorals.core.config import Settings
from nomorals.missions import MissionRunner, MissionStatus, MissionStore


# ── Code 128 tables & round trips ───────────────────────────────────────────

def _runs_of(bit: str) -> list[int]:
    widths, cur = [], bit[0]
    count = 1
    for ch in bit[1:]:
        if ch == cur:
            count += 1
        else:
            widths.append(count)
            cur, count = ch, 1
    widths.append(count)
    return widths


class Code128TableTests(unittest.TestCase):
    def test_table_structure_invariants(self):
        # 103 data + 3 start; all 11 modules; bar-sum even, space-sum odd;
        # unique — these invariants define the Code 128 pattern set.
        self.assertEqual(len(bc.C128_PATTERNS), 106)
        self.assertEqual(len(set(bc.C128_PATTERNS)), 106)
        for pat in bc.C128_PATTERNS:
            self.assertEqual(len(pat), 11)
            self.assertTrue(pat.startswith("1"))
            self.assertTrue(pat.endswith("0"))
            widths = _runs_of(pat)
            self.assertEqual(len(widths), 6)
            bars = sum(widths[0::2])
            spaces = sum(widths[1::2])
            self.assertIn(bars, (4, 6, 8))
            self.assertIn(spaces, (3, 5, 7))
            for w in widths:
                self.assertTrue(1 <= w <= 4)

    def test_start_and_stop_codes(self):
        self.assertEqual(bc.C128_PATTERNS[103], "11010000100")  # START A
        self.assertEqual(bc.C128_PATTERNS[104], "11010010000")  # START B
        self.assertEqual(bc.C128_PATTERNS[105], "11010011100")  # START C
        # stop symbol (value 106) + mandatory 2-module terminal bar = 2331112
        self.assertEqual(bc.C128_STOP, "11000111010")
        self.assertEqual(bc.C128_STOP_FULL, "1100011101011")
        self.assertEqual(_runs_of(bc.C128_STOP_FULL), [2, 3, 3, 1, 1, 1, 2])

    def test_check_symbol_math(self):
        # START B (104) + "A" (0) + "B" (1): 104 + 1*0 + 2*1 = 106 ≡ 3 (mod 103)
        self.assertEqual(bc.c128_check_symbol([104, 0, 1]), 3)


class Code128RoundTripTests(unittest.TestCase):
    def _rt(self, text: str) -> None:
        enc = bc.encode_code128(text)
        rep = bc.decode_code128("0000" + enc["bits"] + "0000")
        self.assertTrue(rep["ok"], (text, rep))
        self.assertEqual(rep["text"], text)
        self.assertTrue(rep["check_ok"])

    def test_round_trips(self):
        for text in ["GIFT-CARD-51934567890", "hello world",
                     "4111 1111 1111 1111", "0123456789", "A", "TEST",
                     "abc123def4567", "UPPER lower MiXed_9"]:
            self._rt(text)

    def test_damaged_symbol_fails_check(self):
        enc = bc.encode_code128("CARD4111111111111")
        bits = enc["bits"]
        start = bits.find(bc.C128_PATTERNS[104])
        # corrupt one data symbol (flip a bit)
        pos = start + 11
        bad = bits[:pos] + ("0" if bits[pos] == "1" else "1") + bits[pos + 1:]
        rep = bc.decode_code128(bad)
        self.assertFalse(rep["check_ok"])


# ── EAN-13 / UPC-A ──────────────────────────────────────────────────────────

class EanTests(unittest.TestCase):
    def test_check_digit_vectors(self):
        self.assertEqual(bc.ean_check_digit("000000000000"), 0)
        self.assertEqual(bc.ean_check_digit("000000000001"), 7)
        # the encoder's check digit matches the standalone computation
        for body in ("519345678901", "890123456789", "036000291450"):
            d = bc.ean_check_digit(body)
            self.assertEqual(bc.encode_ean13(body)["digits"][12], str(d))

    def test_ean13_round_trips(self):
        for body in ("519345678901", "890123456789", "123456789012",
                     "345678912345", "901234567890"):
            enc = bc.encode_ean13(body)
            rep = bc.decode_ean13("0000" + enc["bits"] + "0000")
            self.assertTrue(rep["ok"], (body, rep))
            self.assertEqual(rep["digits"][:12], body)
            self.assertEqual(rep["system"], bc.EAN_SYSTEMS[int(body[0])])

    def test_upca_round_trip(self):
        enc = bc.encode_upca("03600029145")
        rep = bc.decode_upca(enc["bits"])
        self.assertTrue(rep["ok"], rep)
        self.assertEqual(rep["digits"], "036000291452")
        self.assertIn("UPC", rep["system"])

    def test_wrong_check_digit_flagged(self):
        enc = bc.encode_ean13("519345678901")
        bits = list(enc["bits"])
        # flip the last data module to corrupt the check digit's pattern
        bits[-15] = "1" if bits[-15] == "0" else "0"
        rep = bc.decode_ean13("".join(bits))
        self.assertFalse(rep.get("check_ok", True) and rep["ok"])


# ── Luhn ────────────────────────────────────────────────────────────────────

class LuhnTests(unittest.TestCase):
    def test_valid_and_invalid(self):
        self.assertTrue(bc.luhn_valid("4111111111111111"))
        self.assertTrue(bc.luhn_valid("5500005555555559"))
        self.assertFalse(bc.luhn_valid("4111111111111112"))
        self.assertFalse(bc.luhn_valid(""))

    def test_check_digit(self):
        prefix = "41111111111111"
        d = bc.luhn_check_digit(prefix)
        self.assertTrue(bc.luhn_valid(prefix + str(d)))


# ── recovery ────────────────────────────────────────────────────────────────

class RecoveryTests(unittest.TestCase):
    def test_recover_ean_one_hole(self):
        rep = bc.recover_ean("5193?45678901")
        self.assertLessEqual(len(rep["candidates"]), 10)
        self.assertTrue(rep["candidates"])
        for c in rep["candidates"]:
            self.assertTrue(c["check_ok"])
            self.assertEqual(len(c["value"]), 13)
            self.assertTrue(c["value"].startswith("519"))
            self.assertTrue(c["value"].endswith("45678901"))

    def test_recover_ean_missing_check_digit_unique(self):
        rep = bc.recover_ean("519345678901?")
        self.assertEqual(len(rep["candidates"]), 1)
        self.assertEqual(rep["recovered"],
                         "519345678901" + str(bc.ean_check_digit("519345678901")))

    def test_recover_ean_two_holes_bounded(self):
        rep = bc.recover_ean("51?3?45678901")
        self.assertLessEqual(len(rep["candidates"]), 100)

    def test_recover_ean_too_many_holes_rejected(self):
        with self.assertRaises(ValueError):
            bc.recover_ean("?????????????")

    def test_recover_code128_payload(self):
        rep = bc.recover_code128("GIFT?519")
        self.assertTrue(rep["candidates"])
        for c in rep["candidates"]:
            self.assertTrue(c["value"].startswith("GIFT"))
            self.assertTrue(c["value"].endswith("519"))
            self.assertTrue(c["check_ok"])

    def test_recover_scanline_unique_with_known_number(self):
        true_text = "CARD4111111111111"
        enc = bc.encode_code128(true_text)
        bits = enc["bits"]
        start = bits.find(bc.C128_PATTERNS[104])
        damaged = bits[:start + 33] + "1" * 11 + bits[start + 44:]
        pattern = "CARD" + "?" + "1" * (len(true_text) - 5)
        rep = bc.recover_scanline(damaged, known_number=pattern)
        self.assertEqual(rep["recovered"], true_text)
        self.assertEqual(len(rep["candidates"]), 1)

    def test_recover_scanline_two_symbols_true_value_present(self):
        true_text = "5193456789012345"
        enc = bc.encode_code128(true_text)
        bits = enc["bits"]
        pos = bits.find(bc.C128_PATTERNS[105]) + 22
        damaged = bits[:pos] + "0" * 22 + bits[pos + 22:]
        rep = bc.recover_scanline(
            damaged, known_number="51????6789012345", max_symbols=2)
        values = [c["value"] for c in rep["candidates"]]
        self.assertIn(true_text, values)

    def test_recover_scanline_bad_span_reported(self):
        rep = bc.recover_scanline("000111000111")
        self.assertFalse(rep["recovered"])
        self.assertTrue(rep["note"])


# ── analyze & inputs ────────────────────────────────────────────────────────

class AnalyzeTests(unittest.TestCase):
    def test_analyze_detects_ean13(self):
        bits = bc.encode_ean13("519345678901")["bits"]
        rep = bc.analyze(bits)
        self.assertTrue(rep["ok"])
        top = rep["candidates"][0]
        self.assertEqual(top["symbology"], "ean13")
        self.assertEqual(top["value"], "519345678901"
                         + str(bc.ean_check_digit("519345678901")))

    def test_analyze_detects_code128(self):
        bits = bc.encode_code128("GIFT-CARD-99")["bits"]
        rep = bc.analyze(bits)
        self.assertTrue(rep["ok"])
        self.assertEqual(rep["candidates"][0]["symbology"], "code128")

    def test_analyze_garbage(self):
        rep = bc.analyze("no barcode here at all")
        self.assertFalse(rep["ok"])

    def test_scanline_inputs(self):
        bits = bc.encode_ean13("519345678901")["bits"]
        run_list = []
        cur, count = bits[0], 1
        for ch in bits[1:]:
            if ch == cur:
                count += 1
            else:
                run_list.append([1 if cur == "1" else 0, count])
                cur, count = ch, 1
        run_list.append([1 if cur == "1" else 0, count])
        self.assertEqual(bc.scanline_from_any(run_list), bits)
        self.assertEqual(bc.scanline_from_any(json.dumps({"runs": run_list})),
                         bits)
        self.assertEqual(bc.scanline_from_any(
            json.dumps({"bits": bits})), bits)
        self.assertEqual(bc.scanline_from_any(" ".join(bits)), bits)
        self.assertIsNone(bc.scanline_from_any("hello world"))

    def test_card_number_normalization(self):
        self.assertEqual(bc.card_number("Card no:  5193 4567 8901"),
                         "519345678901")
        self.assertEqual(bc.card_number(4111111111111111), "4111111111111111")
        self.assertEqual(bc.card_number(None), "")


# ── gift card tool ──────────────────────────────────────────────────────────

class GiftcardToolTests(unittest.TestCase):
    def _call(self, **kw):
        from nomorals.tools.giftcard import giftcard
        return giftcard(**kw)

    def test_analyze_action(self):
        bits = bc.encode_ean13("519345678901")["bits"]
        rep = self._call(action="analyze", data=bits)
        self.assertTrue(rep["ok"])

    def test_recover_action(self):
        rep = self._call(action="recover", number="519345678901?")
        self.assertEqual(len(rep["candidates"]), 1)

    def test_verify_action(self):
        rep = self._call(action="verify", number="4111111111111111")
        self.assertTrue(rep["luhn_ok"])
        self.assertTrue(rep["valid"])
        rep = self._call(action="verify", number="4111111111111112")
        self.assertFalse(rep["luhn_ok"])

    def test_encode_action(self):
        rep = self._call(action="encode", number="519345678901")
        self.assertEqual(len(rep["bits"]), 95)
        rep = self._call(action="encode", number="GIFT-99")
        self.assertIn("bits", rep)

    def test_registered_in_toolset(self):
        home = tempfile.mkdtemp(prefix="nm-w78-reg-")
        try:
            ctx = build_context(Settings(home=home))
            ctx.__enter__()
            self.assertIn("giftcard", ctx.tools.names())
        finally:
            ctx.__exit__(None, None, None)


# ── reasoning: course correction & self-challenge ───────────────────────────

class _ReasoningTestBase(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="nm-w78-reason-")
        self.context = build_context(Settings(home=self.home,
                                              reasoning_mode="off"))
        self.context.__enter__()

    def tearDown(self):
        self.context.__exit__(None, None, None)


class CourseCorrectTests(_ReasoningTestBase):
    def test_no_signals_no_pivot(self):
        agent = ReasoningAgent(self.context)
        out = agent.course_correct("step fetch failed: timeout",
                                   scope="orchestrator", attempts=1)
        self.assertFalse(out["should_pivot"])
        self.assertEqual(out["pivot"], "")

    def test_repeated_failures_trigger_pivot(self):
        db = self.context.db
        for _ in range(3):
            db.execute(
                "INSERT INTO failures (id, source, summary, error, family, "
                "lesson, ts) VALUES (?,?,?,?,?,?,?)",
                (f"w78-{time.time_ns()}", "orchestrator",
                 "step fetch failed timeout", "timeout", "net", "",
                 time.time()))
        agent = ReasoningAgent(self.context)
        out = agent.course_correct("step fetch failed: timeout",
                                   scope="orchestrator", attempts=3)
        self.assertTrue(out["should_pivot"])
        self.assertTrue(out["pivot"])
        self.assertTrue(any("ledger" in s or "approach" in s
                            for s in out["signals"]))

    def test_second_attempt_triggers_pivot_without_ledger(self):
        agent = ReasoningAgent(self.context)
        out = agent.course_correct("unique failure xyz123", attempts=2)
        self.assertTrue(out["should_pivot"])
        self.assertTrue(out["pivot"])

    def test_course_correct_journaled(self):
        agent = ReasoningAgent(self.context)
        agent.course_correct("journaled failure", attempts=2)
        rows = agent.challenges(limit=5)
        self.assertTrue(any(r.get("kind") == "course_correct" for r in rows))


class SelfChallengeTests(_ReasoningTestBase):
    def test_overconfident_without_evidence_is_contested(self):
        agent = ReasoningAgent(self.context)
        out = agent.self_challenge("This is definitely the cause")
        self.assertTrue(out["contested"])
        self.assertTrue(out["attacks"])

    def test_grounded_conclusion_not_contested(self):
        agent = ReasoningAgent(self.context)
        out = agent.self_challenge(
            "The cache TTL is 300s", evidence="config.yaml line 12: ttl=300")
        self.assertFalse(out["contested"])

    def test_challenges_journaled(self):
        agent = ReasoningAgent(self.context)
        agent.self_challenge("always works")
        rows = agent.challenges(limit=5)
        self.assertTrue(any(r.get("kind") == "self_challenge" for r in rows))


class TraceJournalTests(_ReasoningTestBase):
    def test_trace_persisted(self):
        agent = ReasoningAgent(self.context)

        class _R:
            strategy = "cot"
            answer = "42"
            confidence = 0.9
            trace = [1, 2, 3]

        agent.record_think_trace("why is x slow?", _R())
        rows = agent.traces(limit=5)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["answer"], "42")


# ── orchestrator: routing, arbitration, supervision ─────────────────────────

class _OrchestratorTestBase(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="nm-w78-orch-")
        self.context = build_context(Settings(home=self.home,
                                              reasoning_mode="off"))
        self.context.__enter__()
        self.orch = MasterOrchestrator(self.context, max_steps=4)

    def tearDown(self):
        self.context.__exit__(None, None, None)


class ParallelismTests(_OrchestratorTestBase):
    def test_diamond_plan_parallelism(self):
        plan = Plan(goal="g", steps=[
            PlanStep(name="a", goal="x"),
            PlanStep(name="b", goal="x", depends_on=["a"]),
            PlanStep(name="c", goal="x", depends_on=["a"]),
            PlanStep(name="d", goal="x", depends_on=["b", "c"]),
        ])
        p = plan.parallelism()
        self.assertEqual(p["width"], 2)
        self.assertEqual(p["critical_path"], 3)
        self.assertEqual(p["steps"], 4)

    def test_serial_plan_parallelism(self):
        plan = Plan(goal="g", steps=[
            PlanStep(name="a", goal="x"),
            PlanStep(name="b", goal="x", depends_on=["a"]),
            PlanStep(name="c", goal="x", depends_on=["b"]),
        ])
        p = plan.parallelism()
        self.assertEqual(p["width"], 1)
        self.assertEqual(p["critical_path"], 3)


class RoutingTests(_OrchestratorTestBase):
    def test_route_prefers_exact_role(self):
        graph = TaskGraph(name="t")
        task = graph.add(Task(name="s1", role="research", payload={}))
        record: list[dict] = []

        def h_research(t):
            return "research"

        def h_coding(t):
            return "coding"

        chosen, reason = self.orch.route(
            task, {"research": h_research, "coding": h_coding},
            lambda t: "default", record=record)
        self.assertIs(chosen, h_research)
        self.assertEqual(record[0]["chosen"], "research")

    def test_route_falls_back_when_handler_missing(self):
        graph = TaskGraph(name="t")
        task = graph.add(Task(name="s1", role="vision", payload={}))
        default = lambda t: "default"
        chosen, reason = self.orch.route(
            task, {"research": lambda t: "r"}, default)
        # an adjacent-role handler scores below the default, so the
        # generic handler is used and the decision is explained
        self.assertIs(chosen, default)
        self.assertIn("below default", reason)

    def test_route_learns_from_ledger(self):
        db = self.context.db
        for _ in range(3):
            db.execute(
                "INSERT INTO failures (id, source, summary, error, family, "
                "lesson, ts) VALUES (?,?,?,?,?,?,?)",
                (f"route-{time.time_ns()}", "orchestrator",
                 "badrole failed", "x", "badrole", "", time.time()))
        graph = TaskGraph(name="t")
        task = graph.add(Task(name="s1", role="badrole", payload={}))

        def h_badrole(t):
            return "bad"

        def h_research(t):
            return "good"

        # the ledger penalizes only the failing role itself; with no
        # handler for it, the adjacent candidate still beats the default
        chosen, reason = self.orch.route(
            task, {"research": h_research}, lambda t: "default")
        self.assertIs(chosen, h_research)

    def test_run_records_routing(self):
        handlers = {
            "research": lambda t: {"found": "3 files"},
            "execution": lambda t: {"done": "counted"},
            "critic": lambda t: {"verdict": "ok"},
        }
        result = self.orch.run("count the files", reflect=False,
                               handlers=handlers)
        self.assertTrue(all(r["task"] for r in result.routing))
        self.assertIn("pivots", result.to_dict())
        self.assertIn("conflicts", result.to_dict())


class ArbitrationTests(_OrchestratorTestBase):
    def _conflict_graph(self, critic_says="no", worker_says="yes"):
        graph = TaskGraph(name="t")
        graph.add(Task(name="worker", role="execution",
                       fn=lambda: {"verdict": worker_says}, kind=TaskKind.IO))
        graph.add(Task(name="critic", role="critic",
                       fn=lambda: {"verdict": critic_says}, kind=TaskKind.IO))
        self.context.executor.run(graph)
        return graph

    def test_critic_wins_conflict(self):
        graph = self._conflict_graph()
        conflicts = self.orch.arbitrate("g", graph)
        self.assertEqual(len(conflicts), 1)
        c = conflicts[0]
        self.assertEqual(c["winner"], "critic")
        self.assertEqual(c["positions"], {"worker": "yes", "critic": "no"})

    def test_agreement_is_not_a_conflict(self):
        graph = self._conflict_graph(critic_says="yes", worker_says="yes")
        self.assertEqual(self.orch.arbitrate("g", graph), [])

    def test_no_conflict_when_single_step(self):
        graph = TaskGraph(name="t")
        graph.add(Task(name="solo", role="execution",
                       fn=lambda: {"verdict": "yes"}, kind=TaskKind.IO))
        self.context.executor.run(graph)
        self.assertEqual(self.orch.arbitrate("g", graph), [])


class SupervisionTests(_OrchestratorTestBase):
    def test_pivot_on_repeated_failure(self):
        # ledger says this approach keeps failing -> deterministic pivot
        db = self.context.db
        # the runtime classifies exceptions before the orchestrator sees
        # them, so the ledger summary uses the classified wording
        for _ in range(2):
            db.execute(
                "INSERT INTO failures (id, source, summary, error, family, "
                "lesson, ts) VALUES (?,?,?,?,?,?,?)",
                (f"sup-{time.time_ns()}", "orchestrator",
                 "dothing failed unhandled runtimeerror boom",
                 "boom", "dothing", "", time.time()))
        attempts = {"n": 0}

        def flaky(task):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("boom")
            return {"ok": True, "goal": task.payload.get("goal", "")}

        plan = Plan(goal="g", steps=[PlanStep(name="dothing",
                                              goal="do it")])
        result = self.orch.run("g", plan=plan, reflect=False,
                               handlers={"execution": flaky})
        self.assertTrue(any(p["task"] == "dothing" for p in result.pivots))
        self.assertTrue(all(p["pivot"] for p in result.pivots))
        self.assertEqual(attempts["n"], 2, "one failure, one pivoted retry")
        self.assertTrue(result.ok)

    def test_hard_risk_aborts_step_before_it_runs(self):
        ran = {"n": 0}

        def handler(task):
            ran["n"] += 1
            return "ran"

        plan = Plan(goal="g", steps=[PlanStep(
            name="danger", goal="overwrite the production database",
            role="execution")])
        plan.steps[0].payload["hard_risks"] = ["hard risk: untested code"]
        result = self.orch.run("g", plan=plan, reflect=False,
                               handlers={"execution": handler})
        self.assertEqual(ran["n"], 0, "hard-risk step must never run")
        self.assertGreaterEqual(result.report.failed, 1)

    def test_step_timeout_fails_stalled_task(self):
        # the handler outlives its budget and then dies too: the step
        # must end FAILED (timed out), not silently "succeed" late
        def stall(task):
            time.sleep(0.4)
            raise RuntimeError("still hanging")

        plan = Plan(goal="g", steps=[PlanStep(name="stall", goal="hang")])
        result = self.orch.run("g", plan=plan, reflect=False,
                               handlers={"execution": stall},
                               step_timeout=0.3)
        self.assertGreaterEqual(result.report.failed, 1)

    def test_run_with_supervision_off_still_works(self):
        handlers = {
            "research": lambda t: {"ok": True},
            "execution": lambda t: {"ok": True},
            "critic": lambda t: {"ok": True},
        }
        result = self.orch.run("say hello", reflect=False, supervise=False,
                               handlers=handlers)
        self.assertEqual(
            result.report.done + result.report.failed, result.report.total)
        self.assertTrue(result.ok)


class MissionPivotTests(unittest.TestCase):
    """The mission runner course-corrects a failed step once, with the
    reasoning agent's pivot, before giving up on it."""

    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="nm-w78-mission-")
        self.context = build_context(Settings(home=self.home,
                                              reasoning_mode="off"))
        self.context.__enter__()
        self.store = MissionStore(self.context.db)
        self.runner = MissionRunner(self.context, store=self.store)
        # the agent fails until the prompt carries the injected pivot
        self._patcher = mock.patch(
            "nomorals.agents.roles.build_agent", self._fake_factory)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self.context.__exit__(None, None, None)

    def _fake_factory(self, role, **kwargs):
        class _Fake:
            def run(self, prompt):
                if "[PIVOT]" in str(prompt):
                    return AgentResult(agent_id="fake", role=role,
                                       output={"text": "recovered"}, ok=True,
                                       tokens=1)
                return AgentResult(agent_id="fake", role=role, output="",
                                   ok=False, error="ValueError: wrong approach")
        return _Fake()

    def test_failed_step_pivots_and_recovers(self):
        mission = self.store.create_new("recover the scratched card")
        mission.state["plan"] = [{
            "name": "gather", "goal": "find the card number",
            "role": "research", "kind": "io", "depends_on": [],
        }]
        self.store.save(mission)
        result = self.runner.run(mission, max_iterations=2, reflect=False)
        self.assertEqual(result.status, MissionStatus.DONE)
        reloaded = self.store.get(result.mission_id)
        pivots = reloaded.state.get("pivots") or []
        self.assertEqual(len(pivots), 1)
        self.assertEqual(pivots[0]["step"], "gather")
        self.assertTrue(pivots[0]["pivot"])
        self.assertGreaterEqual(reloaded.state.get("fail_counts", {}).get("gather", 0), 1)

    def test_permanently_failing_step_fails_the_mission(self):
        # no pivot marker is honoured here: the agent always fails
        def always_fail(role, **kwargs):
            class _Fake:
                def run(self, prompt):
                    return AgentResult(agent_id="fake", role=role,
                                       output="", ok=False,
                                       error="ValueError: dead end")
            return _Fake()
        self._patcher.stop()
        self._patcher = mock.patch(
            "nomorals.agents.roles.build_agent", always_fail)
        self._patcher.start()
        mission = self.store.create_new("impossible task")
        mission.state["plan"] = [{
            "name": "gather", "goal": "find the card number",
            "role": "research", "kind": "io", "depends_on": [],
        }]
        self.store.save(mission)
        result = self.runner.run(mission, max_iterations=2, reflect=False)
        self.assertEqual(result.status, MissionStatus.FAILED)
        # the pivot was still tried (and recorded) before the mission gave up
        reloaded = self.store.get(result.mission_id)
        self.assertTrue(reloaded.state.get("pivots"))


if __name__ == "__main__":
    unittest.main()
