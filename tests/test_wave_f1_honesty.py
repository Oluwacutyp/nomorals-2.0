"""Wave F1 Stream 3: honesty audit — fake completion fixes, plan_error and
provider-degradation visibility.

(a) plan_error is SET (never swallowed) on every template/heuristic fallback:
    MasterOrchestrator.plan, MissionRunner._plan, DevonAgent._plan.
(b) plan_error is persisted end-to-end: a forced heuristic fallback ->
    CoreMind.record_plan_error -> router telemetry (migration 65) ->
    `nm mind` shows it (text and JSON).
(c) Provider degradation is explicit: LLMRouter failover annotates the
    response (degraded / failed_providers / fallback_note); the task
    router's chosen-provider failure is attributed instead of swallowed;
    PartnerResponder carries the degradation onto the ReplyBundle.
(d) Chat send paths are honest: _send_long logs failed chunks instead of
    dropping them silently; _send_long_checked reports a total delivery
    failure instead of returning "".
"""

from __future__ import annotations

import argparse
import io
import time
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.coremind import CoreMind
from nomorals.agents.devon import DevonAgent
from nomorals.agents.orchestrator import MasterOrchestrator
from nomorals.agents.router_select import ModelProfile, TaskRouter
from nomorals.llm.base import LLMProvider, LLMResponse
from nomorals.llm.router import LLMRouter
from nomorals.missions.runner import (
    MissionRunner,
    _rehydrate_plan,
    _serialize_plan,
)
from nomorals.partner.mood import MoodEngine
from nomorals.partner.persona import default_persona
from nomorals.partner.relationship import Relationship
from nomorals.partner.responder import PartnerResponder
from nomorals.social.chat.base import ChatKind, SendResult
from nomorals.storage.db import Database
from nomorals.storage.router_telemetry import record_plan_error, snapshot


def _db() -> Database:
    db = Database(":memory:")
    db.migrate()
    return db


class _FakeSettings:
    def resolve(self, key):
        raise RuntimeError("no settings in tests")

    def __getattr__(self, name):
        raise AttributeError(name)


class _Ctx:
    """Minimal context: no router -> every planner must degrade honestly."""

    def __init__(self, db=None, router=None):
        self.settings = _FakeSettings()
        self.extras = {}
        self.memory = None
        self.router = router
        self.db = db
        self.executor = None
        self.blackboard = None
        self.tools = None


# ── fake LLM providers ───────────────────────────────────────────────────────


class _OkProvider(LLMProvider):
    def __init__(self, name="fallback"):
        super().__init__()
        self._name = name

    @property
    def model_id(self):
        return f"{self._name}-model"

    def chat(self, messages, params=None, **kw):
        return LLMResponse(text="served", model=self.model_id,
                           provider=self._name)


class _ErrProvider(LLMProvider):
    """Returns an error response instead of raising."""

    def __init__(self, name="primary", error="upstream 503"):
        super().__init__()
        self._name = name
        self._error = error

    @property
    def model_id(self):
        return f"{self._name}-model"

    def chat(self, messages, params=None, **kw):
        return LLMResponse(text="", model=self.model_id,
                           provider=self._name, error=self._error)


class _RaiseProvider(LLMProvider):
    def __init__(self, name="primary"):
        super().__init__()
        self._name = name

    @property
    def model_id(self):
        return f"{self._name}-model"

    def chat(self, messages, params=None, **kw):
        raise RuntimeError("connection refused")


class _GarbageRouter:
    """chat() succeeds but the text is not a plan."""

    def chat(self, messages, params=None, **kw):
        return LLMResponse(text="I am not JSON at all", model="garbage",
                           provider="garbage")


class _GoodRouter:
    def chat(self, messages, params=None, **kw):
        return LLMResponse(
            text=('{"rationale": "t", "steps": [{"name": "s1", "goal": "g", '
                  '"role": "execution", "kind": "io", "depends_on": []}]}'),
            model="good", provider="good")


# ── (a) plan_error is set on every fallback ──────────────────────────────────


class OrchestratorFallbackTest(unittest.TestCase):
    def test_no_router_fallback_carries_plan_error(self):
        orch = MasterOrchestrator(_Ctx(), max_steps=4)
        plan = orch.plan("do the thing")
        self.assertTrue(plan.steps, "fallback must still produce steps")
        self.assertTrue(plan.plan_error, "template plan must carry plan_error")
        self.assertIn("no LLM router", plan.plan_error)

    def test_model_failure_fallback_carries_plan_error(self):
        ctx = _Ctx(router=_ErrProvider())
        # wrap the provider-shaped object in the router interface the
        # orchestrator expects: it only calls router.chat(...)
        orch = MasterOrchestrator(ctx, max_steps=4)
        plan = orch.plan("do the thing")
        self.assertTrue(plan.steps)
        self.assertTrue(plan.plan_error)
        self.assertIn("model call failed", plan.plan_error)

    def test_unparseable_fallback_carries_plan_error(self):
        orch = MasterOrchestrator(_Ctx(router=_GarbageRouter()), max_steps=4)
        plan = orch.plan("do the thing")
        self.assertTrue(plan.steps)
        self.assertTrue(plan.plan_error)
        self.assertIn("no usable plan", plan.plan_error)

    def test_clean_model_plan_has_empty_plan_error(self):
        orch = MasterOrchestrator(_Ctx(router=_GoodRouter()), max_steps=4)
        plan = orch.plan("do the thing")
        self.assertEqual(plan.plan_error, "",
                         "a real model plan must not look degraded")
        self.assertTrue(plan.steps)


class MissionPlanErrorTest(unittest.TestCase):
    def test_mission_plan_records_telemetry(self):
        db = _db()
        ctx = _Ctx(db=db)
        runner = MissionRunner(ctx)
        mission = runner.store.create_new("migrate the docs")
        steps = runner._plan(mission)
        self.assertTrue(steps)
        self.assertTrue(mission.state.get("plan_error"),
                        "mission state must keep the degradation marker")
        err = snapshot(db)["last_plan_error"]
        self.assertIsNotNone(err, "plan_error must reach the telemetry table")
        self.assertIn("template plan", err["error"])
        self.assertEqual(err["route"], "mission")
        self.assertGreater(err["at"], time.time() - 60)

    def test_serialize_rehydrate_keeps_marker_out_of_steps(self):
        db = _db()
        ctx = _Ctx(db=db)
        runner = MissionRunner(ctx)
        mission = runner.store.create_new("migrate the docs")
        runner._plan(mission)
        raw = mission.state["plan"]
        self.assertTrue(any("__plan_error__" in e for e in raw))
        steps = _rehydrate_plan(mission.goal, raw)
        self.assertTrue(steps)
        self.assertFalse(any("__plan_error__" in str(s) for s in steps))
        for s in steps:
            self.assertTrue(s.name and not s.name.startswith("__"))

    def test_serialize_clean_plan_has_no_marker(self):
        plan = SimpleNamespace(steps=[])
        raw = _serialize_plan(plan, plan_error="")
        self.assertEqual(raw, [])


# ── (b) plan_error end-to-end: heuristic -> telemetry -> nm mind ─────────────


class PlanErrorEndToEndTest(unittest.TestCase):
    def test_heuristic_fallback_persisted_and_shown_by_nm_mind(self):
        from nomorals.cli import _cmd_mind

        db = _db()
        ctx = _Ctx(db=db)  # router=None forces the heuristic ladder
        agent = DevonAgent(ctx)
        result = agent.run("check git status of the repo")
        self.assertEqual(result.planned_by, "heuristic",
                         "no router must force the heuristic fallback")
        self.assertTrue(result.plan_error,
                        "heuristic plan must carry a non-empty plan_error")

        # the exact production path: CoreMind.record_plan_error, as called
        # by PartnerRuntime._control_devon for every degraded devon plan
        mind = CoreMind(ctx)
        mind.record_plan_error(result.plan_error, route="devon")

        err = snapshot(db)["last_plan_error"]
        self.assertIsNotNone(err)
        self.assertIn("no-router", err["error"])
        self.assertEqual(err["route"], "devon")

        # ... and `nm mind` surfaces it, both as text and JSON
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_mind(argparse.Namespace(json=True), ctx)
        self.assertEqual(rc, 0)
        payload = __import__("json").loads(buf.getvalue())
        self.assertIn("no-router", payload["last_plan_error"]["error"])
        self.assertEqual(payload["last_plan_error"]["route"], "devon")

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_mind(argparse.Namespace(json=False), ctx)
        self.assertEqual(rc, 0)
        self.assertIn("no-router", buf.getvalue())
        self.assertIn("last plan_error", buf.getvalue())

    def test_record_plan_error_without_db_never_raises(self):
        mind = CoreMind(_Ctx(db=None))
        mind.record_plan_error("x", route="devon")  # must not raise


# ── (c) provider degradation visibility ──────────────────────────────────────


def _router_with(*providers) -> LLMRouter:
    r = LLMRouter(cooldown_seconds=0.0)
    for i, p in enumerate(providers):
        r.add(p, primary=(i == 0), name=p._name)
    return r


class RouterDegradationTest(unittest.TestCase):
    def test_failover_annotates_response(self):
        r = _router_with(_RaiseProvider("primary"), _OkProvider("fallback"))
        resp = r.chat([])
        self.assertTrue(resp.ok)
        self.assertEqual(resp.provider, "fallback")
        self.assertTrue(resp.degraded, "failover must be flagged on the response")
        self.assertEqual(resp.failed_providers, ["primary"])
        self.assertIn("primary failed", resp.fallback_note)
        self.assertIn("served by fallback", resp.fallback_note)
        self.assertEqual(r.stats["failovers"], 1)

    def test_failover_on_error_response_annotates(self):
        r = _router_with(_ErrProvider("primary", "upstream 503"),
                         _OkProvider("fallback"))
        resp = r.chat([])
        self.assertTrue(resp.ok)
        self.assertTrue(resp.degraded)
        self.assertEqual(resp.failed_providers, ["primary"])
        self.assertIn("upstream 503", resp.fallback_note)

    def test_total_failure_lists_all_attempts(self):
        r = _router_with(_RaiseProvider("p1"), _ErrProvider("p2", "bad"))
        resp = r.chat([])
        self.assertFalse(resp.ok)
        self.assertEqual(resp.failed_providers, ["p1", "p2"])
        self.assertIn("p1 failed", resp.fallback_note)
        self.assertIn("p2 failed", resp.fallback_note)
        self.assertEqual(r.stats["failures"], 1)

    def test_no_failover_means_not_degraded(self):
        r = _router_with(_OkProvider("solo"))
        resp = r.chat([])
        self.assertTrue(resp.ok)
        self.assertFalse(resp.degraded)
        self.assertEqual(resp.failed_providers, [])
        self.assertEqual(resp.fallback_note, "")
        self.assertEqual(r.stats["failovers"], 0)

    def test_degradation_survives_to_dict(self):
        r = _router_with(_RaiseProvider("primary"), _OkProvider("fallback"))
        d = r.chat([]).to_dict()
        self.assertTrue(d["degraded"])
        self.assertEqual(d["failed_providers"], ["primary"])
        self.assertIn("served by fallback", d["fallback_note"])


class TaskRouterDegradationTest(unittest.TestCase):
    def test_chosen_provider_failure_is_attributed(self):
        r = _router_with(_RaiseProvider("chosen"), _OkProvider("chain-fb"))
        ctx = _Ctx(router=r)
        ctx.settings = SimpleNamespace(router_intelligent="on")
        task_router = TaskRouter(ctx)
        with mock.patch.object(
            TaskRouter, "select",
            return_value=ModelProfile(name="chosen"),
        ):
            resp = task_router.route("hello", task_type="chat", objective="balanced")
        self.assertTrue(resp.ok)
        self.assertTrue(resp.degraded)
        self.assertEqual(resp.failed_providers[0], "chosen",
                         "the failed chosen provider must lead the chain")
        self.assertIn("chosen failed", resp.fallback_note)
        self.assertIn("served by chain-fb", resp.fallback_note)

    def test_chosen_provider_error_response_is_attributed(self):
        r = _router_with(_ErrProvider("chosen", "quota exhausted"),
                         _OkProvider("chain-fb"))
        ctx = _Ctx(router=r)
        ctx.settings = SimpleNamespace(router_intelligent="on")
        task_router = TaskRouter(ctx)
        with mock.patch.object(
            TaskRouter, "select",
            return_value=ModelProfile(name="chosen"),
        ):
            resp = task_router.route("hello", task_type="chat", objective="balanced")
        self.assertTrue(resp.ok)
        self.assertTrue(resp.degraded)
        self.assertIn("quota exhausted", resp.fallback_note)


class _DegradedRouter:
    def chat(self, messages, params=None, **kw):
        return LLMResponse(text="here is the answer", model="fb-7b",
                           provider="chain-fb", degraded=True,
                           failed_providers=["primary"],
                           fallback_note="primary failed (boom); served by chain-fb")


class _CleanRouter:
    def chat(self, messages, params=None, **kw):
        return LLMResponse(text="here is the answer", model="ok-7b",
                           provider="primary")


def _responder(router) -> PartnerResponder:
    persona = default_persona()
    return PartnerResponder(
        router, persona, MoodEngine(persona.baselines),
        Relationship(id="default", stage="in_love", trust=90), None, None)


class ResponderDegradationTest(unittest.TestCase):
    def test_bundle_carries_degradation(self):
        bundle = _responder(_DegradedRouter()).respond(
            chat_platform="local", user_text="hey")
        self.assertFalse(bundle.fallback)
        self.assertTrue(bundle.degraded)
        self.assertEqual(bundle.degraded_note,
                         "primary failed (boom); served by chain-fb")
        d = bundle.to_dict()
        self.assertTrue(d["degraded"])
        self.assertIn("primary failed", d["degraded_note"])

    def test_bundle_clean_when_no_degradation(self):
        bundle = _responder(_CleanRouter()).respond(
            chat_platform="local", user_text="hey")
        self.assertFalse(bundle.degraded)
        self.assertEqual(bundle.degraded_note, "")


# ── (d) chat send honesty ────────────────────────────────────────────────────


class _FailGateway:
    def send(self, platform, chat, text, **kw):
        raise RuntimeError("gateway down")

    def typing(self, *a, **k):
        pass


class _OkGateway:
    def __init__(self):
        self.sent = []

    def send(self, platform, chat, text, **kw):
        self.sent.append(text)
        return SendResult(ok=True, platform=platform, message_id="m1")

    def typing(self, *a, **k):
        pass


def _runtime(gateway):
    from nomorals.agents.partner_runtime import PartnerRuntime

    rt = PartnerRuntime.__new__(PartnerRuntime)
    rt.settings = SimpleNamespace(
        partner=SimpleNamespace(typing_in_groups=False,
                                typing_seconds=1.0, typing_cap_seconds=5.0))
    rt.brain = None  # typing disabled for GROUP chats in tests
    rt.gateway = gateway
    return rt


def _group_chat():
    return SimpleNamespace(kind=ChatKind.GROUP, key="telegram:1")


class SendHonestyTest(unittest.TestCase):
    def test_send_long_returns_zero_and_does_not_raise_on_failure(self):
        rt = _runtime(_FailGateway())
        with self.assertLogs("nomorals.agents.partner_runtime",
                             level="WARNING") as logs:
            sent = rt._send_long("telegram", _group_chat(), "hello " * 1000)
        self.assertEqual(sent, 0)
        self.assertTrue(any("send_long" in m for m in logs.output),
                        "failed chunks must be logged, never silent")

    def test_send_long_checked_reports_total_failure(self):
        rt = _runtime(_FailGateway())
        note = rt._send_long_checked("telegram", _group_chat(), "hello")
        self.assertIn("delivery failed", note)
        self.assertIn("Nothing was delivered", note)

    def test_send_long_checked_ok_on_delivery(self):
        gw = _OkGateway()
        rt = _runtime(gw)
        note = rt._send_long_checked("telegram", _group_chat(), "hello " * 2000)
        self.assertEqual(note, "")
        self.assertGreater(len(gw.sent), 1, "long text must go out in chunks")


if __name__ == "__main__":
    unittest.main()
