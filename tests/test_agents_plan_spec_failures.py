"""Tail-22 pins: plan_spec.py failure values + NOT-recording.

Every handler in nomorals/agents/plan_spec.py is documented never-raises;
each pin drives the handler into its except path and pins the exact
fallback value plus the fact that the agents failures ledger is
unchanged (snapshot before/after, tail-8 method). plan_spec.py calls no
record function anywhere (grep-proven), so the ledger must not move.

Verdict: 0 converts / 9 exclusions. Weakest is 188 (empty unstored spec
on a caller type error): excluded because the value is never stored,
never persisted, never rendered, its only in-repo consumer
(_last_spec_id) is never read, and a later regenerate on the bogus key
would fail honestly as "unknown spec" — but a future caller could
misread non-None as success, so the caveat is recorded in SURVEY.md.
"""

import os
import tempfile

import pytest

from nomorals.agents import failures as ag_fail
from nomorals.agents import plan_spec as ps
from nomorals.agents.plan_spec import PlanSpec, SpecStore


@pytest.fixture
def _isolated_store():
    old_specs = dict(SpecStore._specs)
    old_loaded = SpecStore._loaded
    old_env = os.environ.get("NM_SPEC_STORE")
    os.environ["NM_SPEC_STORE"] = os.path.join(tempfile.mkdtemp(), "specs.json")
    SpecStore._specs = {}
    SpecStore._loaded = True
    yield
    SpecStore._specs = old_specs
    SpecStore._loaded = old_loaded
    if old_env is None:
        os.environ.pop("NM_SPEC_STORE", None)
    else:
        os.environ["NM_SPEC_STORE"] = old_env


def _ledger():
    return list(ag_fail.recent_failures())


class _FlakyStr:
    def __init__(self):
        self.n = 0

    def __str__(self):
        self.n += 1
        if self.n == 1:
            raise RuntimeError("boom-once")
        return "hello"


class _RaisingStr:
    def __str__(self):
        raise RuntimeError("boom-str")


def test_inject_failure_returns_input_unchanged():
    before = _ledger()
    assert ps.inject_backend_included(_FlakyStr()) == "hello"
    assert "BACKEND INCLUDED" not in ps.inject_backend_included(_FlakyStr())
    assert _ledger() == before


def test_should_include_failure_is_safe_false():
    before = _ledger()
    assert ps.should_backend_include(_RaisingStr()) is False
    assert _ledger() == before


def test_new_failure_is_empty_unstored_spec(_isolated_store):
    before = _ledger()
    empty = SpecStore.new("g", tasks=5)
    assert empty.spec_id == "" and empty.goals == "" and empty.tasks == []
    assert SpecStore.get("@v1") is None
    assert _ledger() == before


def test_get_failure_is_none(_isolated_store):
    before = _ledger()
    orig = SpecStore.__dict__["_ensure_loaded"]
    SpecStore._ensure_loaded = classmethod(
        lambda cls: (_ for _ in ()).throw(RuntimeError("boom")))
    try:
        assert SpecStore.get("anything") is None
    finally:
        SpecStore._ensure_loaded = orig
    assert _ledger() == before


def test_bump_failure_is_none(_isolated_store):
    before = _ledger()
    s = SpecStore.new("g")
    orig = SpecStore.__dict__["get"]
    SpecStore.get = classmethod(
        lambda cls, k: (_ for _ in ()).throw(RuntimeError("boom")))
    try:
        assert SpecStore.bump_version(s.key) is None
    finally:
        SpecStore.get = orig
    assert _ledger() == before


def test_history_failure_is_empty(_isolated_store):
    before = _ledger()
    s = SpecStore.new("g")
    orig = SpecStore.__dict__["_ensure_loaded"]
    SpecStore._ensure_loaded = classmethod(
        lambda cls: (_ for _ in ()).throw(RuntimeError("boom")))
    try:
        assert SpecStore.history(s.spec_id) == []
    finally:
        SpecStore._ensure_loaded = orig
    assert _ledger() == before


def test_diff_failure_is_partial_dict(_isolated_store):
    before = _ledger()

    class _RaisingGoals:
        key = "x@v1"
        tasks = []

        @property
        def goals(self):
            raise RuntimeError("boom-goals")

    assert ps.diff_specs(_RaisingGoals(), PlanSpec(spec_id="x")) == {}
    assert _ledger() == before


def test_render_failure_is_bare_header(_isolated_store):
    before = _ledger()

    class _RaisingTasks:
        key = "y@v1"
        parent_spec_id = ""
        change_note = ""
        goals = "gg"
        architecture = ""
        data_model = ""
        api = ""

        @property
        def tasks(self):
            raise RuntimeError("boom-tasks")

    assert ps.render_spec(_RaisingTasks()) == "SPEC y@v1"
    assert _ledger() == before


def test_plan_to_spec_failure_is_none(_isolated_store):
    before = _ledger()

    class _RaisingFiles:
        task = "t"

        @property
        def files(self):
            raise RuntimeError("boom-files")

    assert ps.plan_to_spec(_RaisingFiles()) is None
    assert _ledger() == before
