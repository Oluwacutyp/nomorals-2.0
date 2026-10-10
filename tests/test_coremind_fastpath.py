"""Fast-path discipline for the Core Mind (wave D stream A).

- simple interactive requests never call the model router and return fast
- a stalled router can never block the chat thread: _model_check is bounded
  by MODEL_CHECK_TIMEOUT_S and the deterministic intent stands on timeout —
  loudly (logged + counted), never silently swallowed
- heuristic planning always carries an explicit plan_error (planner.py +
  devon.py)
- timeout budgets are named constants, interactive short / background long
- mission execution runs off the chat thread
"""
from __future__ import annotations

import asyncio
import os
import time
import types
import unittest
import unittest.mock

from nomorals.agents.coremind import (
    CoreMind,
    fast_path,
    CODING_JOB_TIMEOUT_S,
    MODEL_CHECK_TIMEOUT_S,
)
from nomorals.agents.planner import AgentPlanner


# ── fakes ──────────────────────────────────────────────────────────────────

class FakeSettings:
    def resolve(self, key):
        raise RuntimeError("no settings in tests")


class FakeContext:
    def __init__(self, router=None):
        self.settings = FakeSettings()
        self.extras = {}
        self.memory = None
        self.router = router


class SpyRouter:
    """Explodes if touched — the fast path must never need the model."""

    def __init__(self):
        self.calls = 0

    def complete(self, prompt, params=None, **kw):
        self.calls += 1
        raise AssertionError("fast path must not call the router")


class SlowRouter:
    """Simulates a stalled provider chain."""

    def __init__(self, hang=30.0):
        self.hang = hang

    def chat(self, messages, params=None, **kw):
        time.sleep(self.hang)
        return types.SimpleNamespace(ok=True, text='{"kind": "chat"}')

    def complete(self, prompt, params=None, **kw):
        time.sleep(self.hang)
        return types.SimpleNamespace(ok=True, text='{"kind": "chat"}')


class ChattyRouter:
    """Returns a research verdict with high confidence."""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, params=None, **kw):
        self.calls += 1
        return types.SimpleNamespace(
            ok=True,
            text='{"kind": "research", "target": "fusion reactors", '
                 '"confidence": 0.95, "why": "user asked to research"}')

    def complete(self, prompt, params=None, **kw):
        self.calls += 1
        return types.SimpleNamespace(
            ok=True,
            text='{"kind": "research", "target": "fusion reactors", '
                 '"confidence": 0.95, "why": "user asked to research"}')


class FailingRouter:
    def chat(self, messages, params=None, **kw):
        raise RuntimeError("provider down")

    def complete(self, prompt, params=None, **kw):
        raise RuntimeError("provider down")


def owner_dm():
    chat = types.SimpleNamespace(key="x:console", kind="dm")
    return types.SimpleNamespace(chat=chat)


class _FailingLLM:
    async def generate(self, prompt, max_tokens=2000):
        raise RuntimeError("hub is down")


class _GarbageLLM:
    async def generate(self, prompt, max_tokens=2000):
        return types.SimpleNamespace(content="not json at all")


class _GoodLLM:
    async def generate(self, prompt, max_tokens=2000):
        return types.SimpleNamespace(content=(
            '{"steps": [{"step_id": "s1", "name": "Wait a beat", '
            '"description": "pause", "action": "wait", '
            '"parameters": {"seconds": 0}, "depends_on": []}]}'))


# ── the fast path ───────────────────────────────────────────────────────────

class FastPathTest(unittest.TestCase):
    def test_simple_inputs_never_call_router_and_stay_fast(self):
        router = SpyRouter()
        mind = CoreMind(FakeContext(router=router), runtime=None)
        for text in ["hello", "what time is it", "research fusion reactors",
                     "build me a todo app", "status"]:
            t0 = time.perf_counter()
            mind.decide(text)
            dt = time.perf_counter() - t0
            self.assertLess(dt, 1.0, f"{text!r} took {dt:.2f}s on the fast path")
        self.assertEqual(router.calls, 0)
        self.assertEqual(mind._router_calls, 0)

    def test_chat_falls_through_fast(self):
        # wave F1 stream 2: trivial chat no longer falls through to the
        # brain — the fast path answers it deterministically, with zero
        # model calls and zero heavy-path activations.
        router = SpyRouter()
        mind = CoreMind(FakeContext(router=router), runtime=None)
        t0 = time.perf_counter()
        reply = mind.handle("hello", message=owner_dm(), chat_key="x:console")
        dt = time.perf_counter() - t0
        self.assertIsNotNone(reply)  # fast path answers trivial chat
        self.assertEqual(router.calls, 0)
        self.assertEqual(mind._router_calls, 0)
        self.assertLess(dt, 1.0)


class ModelCheckBoundTest(unittest.TestCase):
    def test_timeout_stays_deterministic(self):
        mind = CoreMind(FakeContext(router=SlowRouter(hang=30.0)), runtime=None)
        t0 = time.perf_counter()
        intent = mind.decide("build something")  # conf 0.6 -> model band
        dt = time.perf_counter() - t0
        self.assertEqual(intent.kind, "build",
                         "timeout must keep the deterministic intent")
        self.assertAlmostEqual(intent.confidence, 0.6)
        # Two bounded model calls run in this path (intent interpret ≤6s,
        # model check ≤8s+2s backstop); a fully stalled chain must still
        # resolve inside their combined budget, never hang.
        self.assertLess(dt, 2 * MODEL_CHECK_TIMEOUT_S + 3.0,
                        f"chat thread stalled {dt:.1f}s")
        self.assertEqual(mind._router_timeouts, 1)
        self.assertTrue(mind._model_check_note,
                        "timeout must be surfaced, not swallowed")

    def test_success_still_routes(self):
        mind = CoreMind(FakeContext(router=ChattyRouter()), runtime=None)
        intent = mind.decide("build something")
        self.assertEqual(intent.kind, "research")
        self.assertEqual(intent.route, "research_swarm")

    def test_failure_returns_deterministic(self):
        mind = CoreMind(FakeContext(router=FailingRouter()), runtime=None)
        t0 = time.perf_counter()
        intent = mind.decide("build something")
        dt = time.perf_counter() - t0
        self.assertEqual(intent.kind, "build")
        self.assertEqual(mind._router_timeouts, 0)  # failure, not timeout
        self.assertLess(dt, 5.0)

    def test_status_reports_model_timeouts(self):
        mind = CoreMind(FakeContext(router=SlowRouter(hang=30.0)), runtime=None)
        mind.decide("build something")
        text = mind.status()
        self.assertIn("timed out 1×", text)
        self.assertIn("last model check", text)


class TimeoutBudgetsTest(unittest.TestCase):
    def test_constants_sane(self):
        # interactive = fail fast (seconds); background jobs get a long budget
        self.assertGreater(MODEL_CHECK_TIMEOUT_S, 0)
        self.assertLessEqual(MODEL_CHECK_TIMEOUT_S, 15,
                             f"interactive budget too big: {MODEL_CHECK_TIMEOUT_S}s")
        self.assertGreaterEqual(CODING_JOB_TIMEOUT_S, 60,
                                f"background budget too small: {CODING_JOB_TIMEOUT_S}s")
        self.assertGreater(CODING_JOB_TIMEOUT_S, MODEL_CHECK_TIMEOUT_S * 4)


class MissionDispatchTest(unittest.TestCase):
    def test_mission_run_does_not_block_chat_thread(self):
        mind = CoreMind(FakeContext(router=None), runtime=None)

        class SlowDirectives:
            def run(self, did):
                time.sleep(5.0)
                return {"ok": True, "id": "abc12345", "result": "done"}

        mind._directives = lambda: SlowDirectives()
        t0 = time.perf_counter()
        reply = mind.handle("run my mission", message=owner_dm(),
                            chat_key="x:console")
        dt = time.perf_counter() - t0
        self.assertLess(dt, 2.0,
                        f"mission run blocked the chat thread {dt:.1f}s")
        self.assertIn("running the mission", reply or "")


# ── plan_error ──────────────────────────────────────────────────────────────

class PlannerPlanErrorTest(unittest.TestCase):
    def test_template_plan_sets_plan_error_no_llm(self):
        planner = AgentPlanner(llm_router=None)
        result = asyncio.run(planner.execute(
            goal="plan a dinner party", account="owner"))
        self.assertTrue(result.plan_error,
                        "template fallback must set plan_error")
        self.assertIn("template", result.summary.lower())
        self.assertEqual(result.to_dict()["plan_error"], result.plan_error)

    def test_template_plan_sets_plan_error_when_llm_fails(self):
        planner = AgentPlanner(llm_router=_FailingLLM())
        result = asyncio.run(planner.execute(
            goal="plan a dinner party", account="owner"))
        self.assertTrue(result.plan_error,
                        "LLM failure must surface as plan_error")
        self.assertIn("template", result.plan_error.lower())

    def test_template_plan_sets_plan_error_when_llm_garbles(self):
        planner = AgentPlanner(llm_router=_GarbageLLM())
        result = asyncio.run(planner.execute(
            goal="plan a dinner party", account="owner"))
        self.assertIn("no JSON", result.plan_error)

    def test_model_plan_has_no_plan_error(self):
        planner = AgentPlanner(llm_router=_GoodLLM())
        result = asyncio.run(planner.execute(goal="do it", account="owner"))
        self.assertEqual(result.plan_error, "")
        self.assertNotIn("template", result.summary.lower())


class DevonPlanErrorTest(unittest.TestCase):
    def test_empty_task_carries_plan_error(self):
        from nomorals.agents.devon import DevonAgent

        agent = DevonAgent(FakeContext())
        result = agent.run("")
        self.assertEqual(result.planned_by, "heuristic")
        self.assertTrue(result.plan_error,
                        "heuristic plan must explain itself")

    def test_heuristic_fallback_never_empty_error(self):
        from nomorals.agents.devon import DevonAgent

        agent = DevonAgent(FakeContext())  # no router -> heuristic path
        result = agent.run("check the repo state")
        self.assertIn(result.planned_by, ("heuristic", "reasoning", "llm"))
        if result.planned_by == "heuristic":
            self.assertTrue(result.plan_error,
                            "silent heuristic success is forbidden")


# ── wave F1 stream 2: the zero-model fast path ─────────────────────────────

TRIVIAL = [
    "hello", "hi", "hey", "yo", "good morning", "good evening", "sup",
    "what's up", "hello!", "hi",
    "what time is it", "what's the time", "current time", "tell me the time",
    "what time is it?", "what day is it", "today's date", "what's the date",
    "current date", "how are you", "how're you", "how do you feel",
    "thanks", "thank you", "thx", "bye", "good night", "see you later",
    "ok", "cool", "lol", "\U0001f44d",
]

# Ambiguous or organ-flavored: must NEVER fast-path (fail-open toward
# capability — these fall through to understand()/the brain).
MUST_FALL_THROUGH = [
    "what time is the meeting tomorrow",   # time words, but a real question
    "hello, research volcanoes",           # greeting + organ invocation
    "research what time the market opens", # organ verb wins over time words
    "build me a clock app",                # organ verb wins
    "hi, download the report",             # greeting + download verb
    "what's the weather like",
    "game",                                # the word alone never launches
    "this game is boring",
    "how's it going",                      # _RE_STATUS owns this (status organ)
    "how's things",
    "hello there",                         # not the whole message
]


def _group_msg():
    chat = types.SimpleNamespace(key="group:123", kind="group")
    return types.SimpleNamespace(chat=chat)


class FastPathResponderTest(unittest.TestCase):
    """Trivial chat gets a deterministic reply: zero model calls, zero
    heavy-path activations (no router, no decide, no swarms, no organs)."""

    def _mind(self, router=None):
        mind = CoreMind(FakeContext(router=router or SpyRouter()), runtime=None)
        def _boom(*a, **k):
            raise AssertionError("fast path must not reach decide()")
        mind.decide = _boom  # type: ignore[method-assign]
        return mind

    def test_trivial_inputs_answered_with_zero_heavy_activations(self):
        router = SpyRouter()
        mind = self._mind(router)
        for text in TRIVIAL:
            t0 = time.perf_counter()
            reply = mind.handle(text, message=owner_dm(), chat_key="x:console")
            dt = time.perf_counter() - t0
            self.assertIsNotNone(reply, f"{text!r} should fast-path")
            self.assertTrue(str(reply).strip(), f"{text!r} reply is empty")
            self.assertLess(dt, 1.0, f"{text!r} took {dt:.2f}s")
        self.assertEqual(router.calls, 0,
                         "fast path must never call the router model")
        self.assertEqual(mind._router_calls, 0)

    def test_time_reply_is_real_clock_time(self):
        mind = self._mind()
        reply = mind.handle("what time is it", message=owner_dm(),
                            chat_key="x:console")
        self.assertRegex(str(reply), r"^it's \d{1,2}:\d{2} [AP]M \S+\.$")

    def test_date_reply_is_real_calendar_date(self):
        mind = self._mind()
        reply = mind.handle("what day is it", message=owner_dm(),
                            chat_key="x:console")
        self.assertRegex(str(reply),
                         r"^today is \w+, \w+ \d{1,2}, \d{4}\.$")

    def test_fast_path_pure_function_never_swallows_ambiguity(self):
        for text in TRIVIAL:
            got = fast_path(text)
            self.assertIsNotNone(got, f"fast_path missed {text!r}")
            self.assertEqual(len(got), 2)  # (reply, why)
        for text in MUST_FALL_THROUGH:
            self.assertIsNone(fast_path(text),
                              f"fast_path swallowed {text!r}")

    def test_fast_path_structural_gate_intact(self):
        # not the owner's DM → no fast-path reply, no launch path
        mind = self._mind()
        self.assertIsNone(mind.handle("hello", message=_group_msg(),
                                      chat_key="group:123"))

    def test_fast_path_records_route_telemetry(self):
        import sqlite3

        class SqliteDB:
            def __init__(self):
                self.con = sqlite3.connect(":memory:")
                self.con.execute(
                    "CREATE TABLE coremind_telemetry "
                    "(key TEXT PRIMARY KEY, value TEXT, updated_at REAL)")

            def execute(self, sql, params=()):
                return self.con.execute(sql, params)

        class Ctx(FakeContext):
            def __init__(self):
                super().__init__()
                self.db = SqliteDB()

        mind = CoreMind(Ctx(), runtime=None)
        mind.handle("hello", message=owner_dm(), chat_key="x:console")
        row = mind.context.db.con.execute(
            "SELECT value FROM coremind_telemetry WHERE key='route:fastchat'"
        ).fetchone()
        self.assertIsNotNone(row, "fastchat route must be counted")
        self.assertEqual(row[0], "1")


class ExplicitDemandTest(unittest.TestCase):
    """Every organ stays reachable by direct invocation — the fast path
    must never eat an explicit demand."""

    def test_decide_routes_every_organ(self):
        mind = CoreMind(FakeContext(router=SpyRouter()), runtime=None)
        cases = {
            "research fusion reactors": ("research", "research_swarm"),
            "build me a todo app": ("build", "coding"),
            "open https://example.com": ("browse", "browser"),
            "download the file at https://example.com/x.pdf": ("download", "media"),
            "mission: water the plants": ("mission", "directives"),
            "let's play hangman": ("game", "games"),
            "status": ("status", "mind"),
            "research fusion and build a dashboard": ("multi", "orchestrator"),
        }
        for text, (kind, route) in cases.items():
            intent = mind.decide(text)
            self.assertEqual(intent.kind, kind, text)
            self.assertEqual(intent.route, route, text)

    def test_handle_dispatches_every_organ(self):
        mind = CoreMind(FakeContext(router=SpyRouter()), runtime=None)
        cases = {
            "research fusion reactors": "on it — researching",
            "build me a todo app": "building —",
            "open https://example.com": "opening https://example.com",
            "download the file at https://example.com/x.pdf": "downloading",
            "mission: water the plants": "mission queued",
            "let's play hangman": "games live in the chat",
            "status": "core mind",
            "research fusion and build a dashboard": "multi-part goal",
        }
        for text, marker in cases.items():
            reply = mind.handle(text, message=owner_dm(), chat_key="x:console")
            self.assertIsNotNone(reply, f"{text!r} produced no reply")
            self.assertIn(marker, str(reply), text)

    def test_explicit_demand_needs_no_model(self):
        # high-confidence deterministic intents skip the router model
        router = SpyRouter()
        mind = CoreMind(FakeContext(router=router), runtime=None)
        reply = mind.handle("research fusion reactors", message=owner_dm(),
                            chat_key="x:console")
        self.assertIn("researching", str(reply))
        self.assertEqual(router.calls, 0)


class AmbiguousFallthroughTest(unittest.TestCase):
    """Ambiguous input fails OPEN toward the heavy path: handle() returns
    None so the message reaches the normal conversation flow (the brain),
    and decide() still sees the organ signals."""

    def test_ambiguous_returns_none_from_handle(self):
        mind = CoreMind(FakeContext(router=SpyRouter()), runtime=None)
        for text in ["what time is the meeting tomorrow",
                     "hi, download the report",
                     "what's the weather like",
                     "game",
                     "this game is boring",
                     "hello there"]:
            reply = mind.handle(text, message=owner_dm(), chat_key="x:console")
            self.assertIsNone(reply, f"{text!r} should fall through")

    def test_organ_verbs_beat_time_words(self):
        mind = CoreMind(FakeContext(router=SpyRouter()), runtime=None)
        intent = mind.decide("research what time the market opens")
        self.assertEqual(intent.kind, "research")
        self.assertEqual(intent.route, "research_swarm")
        intent = mind.decide("build me a clock app")
        self.assertEqual(intent.kind, "build")
        self.assertEqual(intent.route, "coding")

    def test_greeting_plus_intent_routes_the_intent(self):
        mind = CoreMind(FakeContext(router=SpyRouter()), runtime=None)
        intent = mind.decide("hello, research volcanoes")
        self.assertEqual(intent.kind, "research")

    def test_hows_it_going_stays_status_not_chitchat(self):
        # _RE_STATUS owns "how's it going" — the fast path must not
        # preempt an existing organ intent.
        mind = CoreMind(FakeContext(router=SpyRouter()), runtime=None)
        self.assertIsNone(fast_path("how's it going"))
        intent = mind.decide("how's it going")
        self.assertEqual(intent.kind, "status")


class FastPathTimezoneTest(unittest.TestCase):
    def test_nm_timezone_overrides_server_clock(self):
        with unittest.mock.patch.dict(os.environ, {"NM_TIMEZONE": "America/Denver"}):
            reply, _ = fast_path("what time is it")
        self.assertRegex(reply, r"MDT|MST")

    def test_bad_nm_timezone_falls_back_to_server_local(self):
        with unittest.mock.patch.dict(os.environ, {"NM_TIMEZONE": "Not/AZone"}):
            reply, _ = fast_path("what time is it")
        self.assertRegex(reply, r"it's \d{1,2}:\d{2} [AP]M \S+\.")


if __name__ == "__main__":
    unittest.main()
