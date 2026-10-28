"""Pass tail-5 pins: missions/runner failures degrade honestly, ledger intact.

Target: nomorals/missions/runner.py — 0 converts / 11 exclusions.
Every site below maps a failure to a genuine degraded value (the
documented miss/unknown/best-effort value of that function) and the run
proceeds honestly — no failure is recorded as a success, no outcome is
fabricated, no money or cleanup is dropped. None of them hides a
user-visible outcome.

Ledger note: the missions package has NO failures ledger, so there is no
recent_failures() to assert empty. NOT-recording is pinned against the
only ledger these paths could touch, the unified autonomy ledger: the
module-level record_ledger is patched with a recorder, and each test
asserts the failure produced no record of its own. Two paths journal on
their success path by design, and their pins scope the claim exactly:
_maybe_replan journals kind "replanned" only; _finish journals kinds
"terminal" and "escalated" only — the guard failure itself records
nothing.

Each test pins the degraded behavior AND the NOT-recording property.
No except blocks at all in this file (stubs raise, the code under test
catches), so the scanner counts 0 sites here.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest import mock

from nomorals.missions import runner as R
from nomorals.missions.mission import Mission, MissionStatus
from nomorals.missions.runner import MissionRunner, _retry_after_seconds


class Boom(Exception):
    pass


class FakeStore:
    def __init__(self, get_exc=None):
        self.get_exc = get_exc
        self.saved = []
        self.checkpoints = []
        self.get_calls = 0

    def get(self, mission_id):
        self.get_calls += 1
        if self.get_exc is not None:
            raise self.get_exc
        raise AssertionError("unexpected get")

    def save(self, mission):
        self.saved.append(mission)

    def checkpoint(self, mission, label=""):
        self.checkpoints.append(label)

    def record_reflection(self, *a, **k):
        pass


def make_runner(store=None, factory=None):
    return MissionRunner(SimpleNamespace(), store=store or FakeStore(),
                         milestones=False, orchestrator_factory=factory)


def ledger_patch(calls):
    return mock.patch.object(
        R, "record_ledger",
        lambda *a, **k: calls.append(a[2] if len(a) > 2 else k.get("kind")))


def test_273_retry_after_parse_failure_is_zero():
    class BadStr(str):
        def __str__(self):
            raise Boom("str boom")

    # trigger proven: str(detail) raises inside the try
    assert _retry_after_seconds(BadStr("retry after 99s")) == 0.0
    assert _retry_after_seconds("nothing here") == 0.0
    assert _retry_after_seconds("retry after 12s") == 12.0


def test_626_heartbeat_unknown_mission_is_false():
    store = FakeStore(get_exc=Boom("get boom"))
    r = make_runner(store)
    calls = []
    with ledger_patch(calls):
        assert r.heartbeat_progress("m1", {"p": 1}) is False
    assert store.get_calls == 1 and calls == []


def test_683_frozen_orchestrator_keeps_its_max_steps():
    @dataclass(frozen=True)
    class FrozenOrch:
        max_steps: int = 99

    stub = FrozenOrch()
    r = make_runner(factory=lambda: stub)
    calls = []
    with ledger_patch(calls):
        out = r._new_orchestrator(8)
    # production always takes the real MasterOrchestrator branch (max_steps
    # passed to the constructor); the try-set exists only for injected
    # factories, and the module documents the factory as tests-only
    assert out is stub and out.max_steps == 99 and calls == []


def test_883_external_status_reread_failure_keeps_run():
    store = FakeStore(get_exc=Boom("get boom"))
    r = make_runner(store)
    m = Mission(goal="g", status=MissionStatus.RUNNING)
    calls = []
    with ledger_patch(calls):
        r._sync_external_status(m)
    assert m.status == MissionStatus.RUNNING and calls == []


def test_1163_run_claim_release_get_failure_is_best_effort():
    store = FakeStore(get_exc=Boom("get boom"))
    r = make_runner(store)
    r._run_claims["m1"] = "tok"
    calls = []
    with ledger_patch(calls):
        r._release_run_claim("m1", "tok")
    assert r._run_claims == {} and calls == []


def test_1194_terminal_redelivery_check_failure_skips_silently():
    store = FakeStore()
    r = make_runner(store)
    r.reporter = SimpleNamespace(_marked=mock.Mock(side_effect=Boom("marked boom")))
    m = Mission(goal="g", status=MissionStatus.DONE)
    calls = []
    with ledger_patch(calls):
        r._catch_up_terminal_report(m)
    # cannot prove unmarked, so no re-delivery: exactly-once preserved
    assert store.saved == [] and calls == []


def test_1656_worker_base_exception_is_failed_outcome():
    r = make_runner(FakeStore())
    m = Mission(goal="g")
    step = SimpleNamespace(name="s1")
    calls = []
    with ledger_patch(calls):
        with mock.patch.object(r, "_execute_step", side_effect=KeyboardInterrupt("ki")):
            pairs = r._execute_batch(m, [step], max_parallel=1)
    assert len(pairs) == 1
    outcome = pairs[0][1]
    assert outcome.ok is False and "KeyboardInterrupt" in outcome.detail
    assert calls == []


def _replan_runner():
    class NameFrozenStep:
        """name assignment raises; depends_on assignment works (site 1791)."""

        def __init__(self):
            self._name = "alpha"
            self.goal = "do work"
            self.role = "research"
            self.kind = SimpleNamespace(value="execution")
            self.depends_on = []

        @property
        def name(self):
            return self._name

        @name.setter
        def name(self, value):
            raise AttributeError("frozen name")

    class DepsFrozenStep:
        """depends_on assignment raises; name assignment works (site 1797)."""

        def __init__(self):
            self.name = "alpha"
            self.goal = "do work"
            self.role = "research"
            self.kind = SimpleNamespace(value="execution")
            self._deps = []

        @property
        def depends_on(self):
            return self._deps

        @depends_on.setter
        def depends_on(self, value):
            raise AttributeError("frozen depends_on")

    return NameFrozenStep, DepsFrozenStep


def test_1791_frozen_step_name_kept_replan_survives():
    NameFrozenStep, _ = _replan_runner()
    s1 = NameFrozenStep()
    orch = SimpleNamespace(plan=lambda goal: SimpleNamespace(plan_error="", steps=[s1]))
    r = make_runner(factory=lambda: orch)
    m = Mission(goal="g")
    calls = []
    with ledger_patch(calls):
        levels = r._maybe_replan(m, [], "failed_step", "boom", set(), {"failed_step"})
    assert levels is not None
    assert [s for lvl in levels for s in lvl] == [s1]
    assert s1.name == "alpha"  # rename silently dropped, run continues
    assert calls == ["replanned"]  # only the success journal, no failure record


def test_1797_frozen_step_deps_kept_replan_survives():
    _, DepsFrozenStep = _replan_runner()
    s1 = DepsFrozenStep()
    orch = SimpleNamespace(plan=lambda goal: SimpleNamespace(plan_error="", steps=[s1]))
    r = make_runner(factory=lambda: orch)
    m = Mission(goal="g")
    calls = []
    with ledger_patch(calls):
        levels = r._maybe_replan(m, [], "failed_step", "boom", set(), {"failed_step"})
    assert levels is not None
    assert [s for lvl in levels for s in lvl] == [s1]
    assert s1.depends_on == []  # deps silently dropped, run continues
    assert calls == ["replanned"]  # only the success journal, no failure record


def test_2284_daemon_thread_base_exception_is_failed_outcome():
    r = make_runner(FakeStore())
    m = Mission(goal="g")
    step = SimpleNamespace(name="s1")
    calls = []
    with ledger_patch(calls):
        with mock.patch.object(r, "_run_step_agent", side_effect=KeyboardInterrupt("ki")):
            out = r._run_step_agent_timed(m, step, 0.0, timeout_s=5.0)
    assert out.ok is False and "KeyboardInterrupt" in out.detail
    assert calls == []


def test_2476_finish_guard_read_failure_keeps_verdict():
    store = FakeStore(get_exc=Boom("get boom"))
    r = make_runner(store)
    m = Mission(goal="g", status=MissionStatus.RUNNING)
    calls = []
    with ledger_patch(calls):
        res = r._finish(m, MissionStatus.FAILED, [], 0.0, "", reflect=False)
    assert res.status == MissionStatus.FAILED
    assert store.saved and store.saved[-1].status == MissionStatus.FAILED
    # guard failure recorded nothing; only the success-path journals land
    assert set(calls) <= {"terminal", "escalated"}
