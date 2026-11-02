"""Tail-101 pins: nomorals/agents/mission.py census sites.

All three sites are estimate/display rungs inside MissionControl's
portfolio math, and all three are held as exclusions:

149 — plan(): the CognitiveLoop read of the effective cadence
crashes -> cadence["effective_hours"] keeps its pre-initialised
None. None is the field's documented "unknown" value (the dict is
built with it), and a tree-wide grep shows no consumer of
effective_hours outside this dict — nothing acts on it.
344 — _step_cost(): the projects-table read crashes -> the
per-step cost stays at the narration default (1 call). The site's
own contract is "estimate, never fatal": the number only feeds
_est_calls' planning arithmetic, and the fallback is the cheaper,
documented default rather than an invented one.
378 — _budget_fit(): ModelBudget cannot be constructed -> the
fit dict reports the unlimited-budget shape {fits_today: True,
steps_today: pending, eta_days: 0}. WEAKEST EXCLUSION of this
file: the direction is optimistic on a budget-adjacent surface.
Held because the trace is display-only — the sole caller is
_score_goal (mission.py:414), whose dict feeds the portfolio
ranking in plan(); nothing in the loop gates steps on fits_today,
and a ModelBudget that cannot be constructed means there is no
cap machinery in force to violate. The unlimited-cap early return
two lines below returns the identical dict by design.

NOT-recording is pinned with the snapshot method: mission.py
calls no failures-ledger record function at any of these sites,
so both ledgers must be unchanged across each trigger.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import nomorals.agency.failures as ag_fail
import nomorals.agents.cognition as cognition
import nomorals.agents.failures as agents_fail
from nomorals.agents import mission as M


def _snap():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


def _boom(*args, **kwargs):
    raise RuntimeError("boom")


def _control(context=None):
    mc = M.MissionControl.__new__(M.MissionControl)
    mc.context = context
    return mc


# ── 149: cadence read crash -> effective_hours stays None ──────

class _LoopBoom:
    def __init__(self, *a, **k):
        raise RuntimeError("loop boom")


class _LoopOk:
    def __init__(self, ctx):
        self.ctx = ctx

    def effective_interval(self):
        return 1.25


def _plan_control():
    ctx = SimpleNamespace(
        db=SimpleNamespace(query=lambda *a, **k: []),
        settings=SimpleNamespace(
            autonomy=SimpleNamespace(interval_hours=2.5,
                                     adaptive_cadence=False)),
    )
    mc = _control(ctx)
    mc.settings = ctx.settings
    mc.goals = SimpleNamespace(next_goal=lambda: None)
    return mc


def test_149_cadence_crash_keeps_effective_hours_none():
    before = _snap()
    mc = _plan_control()
    with mock.patch.object(cognition, "CognitiveLoop", _LoopBoom):
        out = mc.plan()
    assert out["cadence"]["base_hours"] == 2.5
    assert out["cadence"]["effective_hours"] is None
    assert _snap() == before


def test_149_cadence_control_reads_effective_interval():
    mc = _plan_control()
    with mock.patch.object(cognition, "CognitiveLoop", _LoopOk):
        out = mc.plan()
    assert out["cadence"]["effective_hours"] == 1.25


# ── 344: step-cost read crash -> narration default ─────────────

def test_344_step_cost_db_crash_defaults_to_narration():
    before = _snap()
    mc = _control(SimpleNamespace(
        db=SimpleNamespace(query_one=_boom)))
    goal = SimpleNamespace(project_id="proj-1")
    assert mc._step_cost(goal) == M._CALLS_PER_NARRATION_STEP
    assert _snap() == before


def test_344_step_cost_controls():
    mc = _control(SimpleNamespace(db=SimpleNamespace(
        query_one=lambda *a, **k: {"task_kind": "build"})))
    assert mc._step_cost(SimpleNamespace(project_id="p")) == \
        M._CALLS_PER_BUILD_STEP
    # projectless goals cost nothing and never touch the db
    mc2 = _control(SimpleNamespace(
        db=SimpleNamespace(query_one=_boom)))
    assert mc2._step_cost(SimpleNamespace(project_id="")) == 0


# ── 378: budget-fit crash -> documented display default ────────

class _BudgetBoom:
    def __init__(self, *a, **k):
        raise RuntimeError("budget boom")


class _Budget:
    def __init__(self, cap, remaining):
        self.cap = cap
        self._remaining = remaining

    def remaining_available(self):
        return self._remaining


def test_378_budget_fit_crash_returns_display_default():
    before = _snap()
    mc = _control(SimpleNamespace())
    with mock.patch.object(cognition, "ModelBudget", _BudgetBoom):
        out = mc._budget_fit(est_calls=10, pending=4, per_step=3)
    assert out == {"fits_today": True, "steps_today": 4, "eta_days": 0}
    assert _snap() == before


def test_378_budget_fit_controls():
    mc = _control(SimpleNamespace())
    # unlimited budget (cap 0) -> same fits shape, no failure involved
    with mock.patch.object(cognition, "ModelBudget",
                           lambda ctx: _Budget(cap=0, remaining=0)):
        assert mc._budget_fit(10, 4, 3) == {
            "fits_today": True, "steps_today": 4, "eta_days": 0}
    # capped budget: floor-divided steps, honest eta
    with mock.patch.object(cognition, "ModelBudget",
                           lambda ctx: _Budget(cap=100, remaining=5)):
        assert mc._budget_fit(10, 4, 3) == {
            "fits_today": False, "steps_today": 1, "eta_days": 1}
