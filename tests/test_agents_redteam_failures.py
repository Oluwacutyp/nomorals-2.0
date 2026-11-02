"""Tail-97 pins: nomorals/agents/redteam.py census sites.

All four sites are held as exclusions — this module's whole design
posture is fail-closed (its own report renders an explicit
"fail-closed — re-run, don't assume pass" section), and every
failure value below points that same direction:

587 — the weekly run's fire-time profile recheck crash -> the run
is SKIPPED (``return``). The gate grants nothing when it cannot be
read: no suite runs, no report is produced. A crashed probe can
never let a non-workstation machine run the job.
706 — crescendo scorer crash -> progress = 0.0, the floor ("a dead
scorer scores 0"). A dead scorer can only under-score an attack
line; the line stalls, gets pruned, and the result stays
succeeded=False. It can never fabricate progress toward success.
812 — ``render_report`` rendering crash -> falls back to the
report's own ``summary()`` (failed/passed/inconclusive counts and
the failure lines). Degraded formatting, real content — the stated
contract is "rendering never breaks callers".
822 — ``_scenario_category`` lookup crash -> "" ("best-effort,
never raises"). The category only picks a recommendation text;
"" falls through to the generic recommendation.

NOT-recording is pinned with the snapshot method: redteam.py calls
no failures-ledger record function anywhere, so both the agency and
agents ledgers must be unchanged across each trigger.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest import mock

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
from nomorals.agents import redteam as R


def _snap():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


def _raise(*args, **kwargs):
    raise RuntimeError("boom")


# ── 587: profile recheck crash -> weekly run skipped ────────────

class _Sched:
    db = None

    def __init__(self):
        self.actions: dict = {}

    async def list_cron_jobs(self):
        return []

    def register_action(self, action_id, fn):
        self.actions[action_id] = fn

    async def schedule_cron(self, *args, **kwargs):
        return None


def test_587_profile_recheck_crash_skips_weekly_run():
    before = _snap()
    sched = _Sched()
    with mock.patch("nomorals.core.profiles.get_profile_kind",
                    lambda: "workstation"):
        assert asyncio.run(R.ensure_redteam_weekly_job(sched)) is True
    weekly = sched.actions[R.REDTEAM_WEEKLY_ACTION]

    ran: list = []

    class _FakeTeam:
        def __init__(self, failure_kb=None):
            pass

        def run_suite(self):
            ran.append(True)
            return SimpleNamespace(summary=lambda: "redteam: ok")

    # failure half: the fire-time profile recheck crashes -> the
    # closed gate stays closed; no suite runs, no exception escapes.
    with mock.patch("nomorals.core.profiles.get_profile_kind", _raise), \
            mock.patch.object(R, "RedTeam", _FakeTeam):
        assert asyncio.run(weekly()) is None
    assert ran == []
    # control: workstation confirmed at fire time -> the suite runs.
    with mock.patch("nomorals.core.profiles.get_profile_kind",
                    lambda: "workstation"), \
            mock.patch.object(R, "RedTeam", _FakeTeam):
        asyncio.run(weekly())
    assert ran == [True]
    assert _snap() == before


# ── 706: dead scorer scores the floor, never progress ───────────

def test_706_dead_scorer_scores_zero():
    before = _snap()
    strat = R.CrescendoStrategy(max_turns=5, success_threshold=0.9,
                                stall_turns=2)
    res = strat.run(
        "scn-1", "objective",
        attacker_fn=lambda h, o, e: "prompt",
        target_fn=lambda p, h: "response",
        scorer_fn=_raise)
    assert res.succeeded is False
    assert res.turns and all(t.progress == 0.0 for t in res.turns)
    assert res.final_progress == 0.0
    # control: a live scorer at threshold succeeds on turn 1 with
    # its real progress recorded.
    res2 = R.CrescendoStrategy().run(
        "scn-1", "objective",
        attacker_fn=lambda h, o, e: "prompt",
        target_fn=lambda p, h: "response",
        scorer_fn=lambda o, h: 0.95)
    assert res2.succeeded is True
    assert res2.turns[0].progress == 0.95
    assert _snap() == before


# ── 812: render crash -> the report's own summary ───────────────

def test_812_render_failure_falls_back_to_summary():
    before = _snap()
    rep = R.RedTeamReport(findings=[], duration_s=1.0)
    with mock.patch("nomorals.agents.render.kv", _raise):
        out = R.render_report(rep)
    assert out == rep.summary()
    # control: the healthy render is the markdown report, not the
    # one-line summary.
    out2 = R.render_report(rep)
    assert "Red-team report" in out2
    assert out2 != rep.summary()
    assert _snap() == before


# ── 822: category lookup crash -> "" (generic recommendation) ───

def test_822_category_lookup_failure_returns_empty():
    before = _snap()
    with mock.patch.object(R, "build_scenarios", _raise):
        assert R._scenario_category("anything") == ""
    # control halves: a real scenario resolves its own category,
    # and an unknown id is "" by the same best-effort contract.
    sc = R.build_scenarios()[0]
    assert R._scenario_category(sc.id) == str(sc.category or "")
    assert R._scenario_category("no-such-scenario-xyz") == ""
    assert _snap() == before
