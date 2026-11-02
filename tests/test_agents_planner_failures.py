"""Tail-102 pins: nomorals/agents/planner.py census sites —
the plan diff/render surface, all held as exclusions.

``diff_plans`` and the two renderers carry a "Never raises"
contract (diff_plans and render_plan_diff say it in their
docstrings; render_plan's guard comment says rendering never
breaks callers — render_plan feeds agents/coding.py's plan
display). Every fallback NAMES its failure in-band — an
``"error": "diff failed"`` key, a "(render failed)" suffix —
so no surface presents a degraded result as a good one: each
is refusal-shaped, the opposite of a false success. All three
triggers were probed live before pinning.

956 — ``diff_plans``: a step whose attribute access raises ->
{"from", "to", "error": "diff failed"}. The caller gets the
plan ids and an explicit error marker instead of a fabricated
empty diff (an empty diff would claim "nothing changed").
988 — ``render_plan_diff``: malformed deltas (a 1-tuple where
(old, new) is unpacked) -> "plan diff (render failed)".
1025 — ``render_plan``: a plan whose ``steps`` access raises
-> "plan <id> (render failed)".

A bind + ``_log.debug`` sketch for the three sites (the module
HAS a logger; all three guards are bare) is shelved as a
follow-up — observability only; the user-visible fallback
already names the failure.

NOT-recording is pinned with the snapshot method: the module
calls no failures-ledger record function anywhere, so both
general ledgers must be unchanged across each trigger.
"""
from __future__ import annotations

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
from nomorals.agents import planner as pl


def _snapshot():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


def _plan(plan_id, name):
    step = pl.PlanStep(step_id="s1", name=name, description="",
                       action="a", status="pending", depends_on=[])
    return pl.Plan(plan_id=plan_id, goal="g",
                   status=pl.PlanStatus.PENDING, steps=[step])


def test_956_diff_failure_is_named_never_fabricated():
    before = _snapshot()

    class ExplodingStep:
        step_id = "s1"

        def __getattr__(self, _name):
            raise RuntimeError("step boom")

    old = pl.Plan(plan_id="p1", goal="g", status=pl.PlanStatus.PENDING,
                  steps=[ExplodingStep()])
    new = pl.Plan(plan_id="p2", goal="g", status=pl.PlanStatus.PENDING,
                  steps=[ExplodingStep()])
    assert pl.diff_plans(old, new) == {"from": "p1", "to": "p2",
                                       "error": "diff failed"}
    # control: a real change diffs for real
    diff = pl.diff_plans(_plan("p1", "one"), _plan("p2", "uno"))
    assert diff["changed"] == {"s1": {"name": ("one", "uno")}}
    assert _snapshot() == before


def test_988_render_diff_failure_is_named():
    before = _snapshot()
    bad = {"from": "a", "to": "b", "changed": {"s1": {"name": ("x",)}}}
    assert pl.render_plan_diff(bad) == "plan diff (render failed)"
    # control: a well-formed diff renders its changes
    rendered = pl.render_plan_diff(pl.diff_plans(_plan("p1", "one"),
                                                 _plan("p2", "uno")))
    assert "uno" in rendered
    assert _snapshot() == before


def test_1025_render_plan_failure_is_named():
    before = _snapshot()

    class BadPlan:
        plan_id = "p9"
        goal = "g"
        status = "open"
        plan_error = None
        error = None

        @property
        def steps(self):
            raise RuntimeError("steps boom")

    assert pl.render_plan(BadPlan()) == "plan p9 (render failed)"
    # control: a real plan renders with its id
    assert "p1" in pl.render_plan(_plan("p1", "one"))
    assert _snapshot() == before
