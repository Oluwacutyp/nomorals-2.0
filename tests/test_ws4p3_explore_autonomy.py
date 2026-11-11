"""WS4 Pass 3 — exploration autonomy + the ``/explore`` board.

Pins, end to end:

(a) ``ensure_explore_jobs`` registers BOTH durable sweeps at the
    census cadence (morning + evening, owner-local), tool payload
    ``video_explore sweep``, routine, fire_now / skip, and is
    idempotent across boots (social/jobs.py + dayrecord shape).
(b) A job fire — through the tool the stored payload names, and
    through the scheduler's own tick with the STORED payload —
    runs sweep → evaluation: the store gains observations and,
    after corroboration, a trend claim.
(c) The cycle's discipline: quiet hours (the notifier's window,
    mirrored) defer a scheduled sweep; the per-day cap (2
    scheduled sweeps, census §7/§9) defers the third; a manual
    cycle is never gated and never consumes the cap.
(d) The board renders last-sweep truth — failed observations
    counted, a failed sweep named FAILED — current claims with
    kind, candidate standings, the next sweep due, and the
    proxy-dependency line in the census's own terms.  No internal
    record id renders anywhere.
(e) ``/explore run`` triggers the same cycle entry point the job
    calls, with manual=True.
(f) The morning-briefing provider renders with a fresh trend
    (basis included) or a failed sweep, and is ABSENT when
    nothing changed — silence is the default.
(g) Catalog pins: ``explore`` owner-only, not public, in the
    details / list / help pins exactly once each, parses.
"""

from __future__ import annotations

import json
import time
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest import mock
from zoneinfo import ZoneInfo

import pytest

from nomorals.agents.scheduler import Scheduler
from nomorals.media.explore import (
    EXPLORE_SWEEP_JOB_EVENING,
    EXPLORE_SWEEP_JOB_MORNING,
    EXPLORE_SWEEP_JOBS,
    ExploreStore,
)
from nomorals.media.explore import board as board_mod
from nomorals.media.explore import engine as engine_mod
from nomorals.media.explore import jobs as jobs_mod
from nomorals.media.explore import tool as tool_mod
from nomorals.media.explore.beats import EXPLORE_SEED_BEATS
from nomorals.media.explore.board import (
    PROXY_DEPENDENCY_LINE,
    render_explore_board,
    render_explore_candidates,
    render_explore_trends,
)
from nomorals.media.explore.jobs import (
    ensure_explore_jobs,
    run_sweep_cycle,
)
from nomorals.media.explore.trends import current_claims, evaluate_trends
from nomorals.storage.db import Database
from nomorals.tools.registry import ToolRegistry


def _text(reply):
    """A board render's text — boards return actionable replies
    since button lane 5; read the text half here."""
    return reply.text if hasattr(reply, "text") else reply


LAGOS = ZoneInfo("Africa/Lagos")


@pytest.fixture(autouse=True)
def _owner_tz(monkeypatch):
    """Owner-local is deterministic in this suite (dayrecord's own
    pin): the ensure install and the day-boundary math read
    Africa/Lagos."""
    monkeypatch.setenv("NM_OWNER_TZ", "Africa/Lagos")


def _lagos_ts(day: int, hour: int, minute: int = 0) -> float:
    return datetime(2026, 11, day, hour, minute,
                    tzinfo=LAGOS).timestamp()


class FakeInstrument:
    """The P1/P2 engine-test instruments, verbatim shape."""

    def __init__(self, name: str, rows: list[dict] | None = None):
        self.name = name
        self.rows = rows if rows is not None else []
        self.calls: list[str] = []

    def search(self, query, *, limit=10):
        self.calls.append(query)
        return [dict(r) for r in self.rows[:limit]]


def _row(title: str, channel: str = "Chan", views: int = 5000,
         platform: str = "youtube") -> dict:
    return {"title": title, "channel": channel, "url": "",
            "views": views, "platform": platform, "snippet": "",
            "published": ""}


def _beats(topic: str = "gold trading") -> list[dict]:
    return [{"topic": topic, "queries": ["gold trading tutorial"],
             "origins": ["test"]}]


def _settings(partner=None) -> SimpleNamespace:
    return SimpleNamespace(partner=partner)


def _partner(quiet_start: int, quiet_end: int) -> SimpleNamespace:
    return SimpleNamespace(quiet_start=quiet_start, quiet_end=quiet_end)


def _ctx(db=None, partner=None, tmp_path=None) -> SimpleNamespace:
    return SimpleNamespace(
        db=db, settings=_settings(partner), extras={}, tools=None,
        home_path=str(tmp_path or ""), workspace_dir=str(tmp_path or ""))


def _registry(ctx, monkeypatch, store, instruments) -> ToolRegistry:
    monkeypatch.setattr(engine_mod, "default_instruments",
                        lambda context: instruments)
    monkeypatch.setattr(engine_mod, "_DEFAULT_STORE", store)
    registry = ToolRegistry(ctx)
    tool_mod.register(registry)
    ctx.tools = registry
    return registry


# ----------------------------------------------------------------
# (a) the ensure install
# ----------------------------------------------------------------


class EnsureJobTests(unittest.TestCase):

    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db = Database(":memory:")
        self.db.migrate()
        self.context = _ctx(db=self.db, tmp_path=self._tmp.name)
        self.scheduler = Scheduler(self.context)

    def _job(self, name: str) -> dict:
        for row in self.scheduler.list_jobs():
            if row["name"] == name:
                return row
        raise AssertionError(f"job not registered: {name}")

    def test_registers_both_sweeps_at_the_census_cadence(self) -> None:
        out = ensure_explore_jobs(self.scheduler)
        self.assertEqual([o["scheduled"] for o in out], [True, True])
        morning = self._job(EXPLORE_SWEEP_JOB_MORNING)
        evening = self._job(EXPLORE_SWEEP_JOB_EVENING)
        self.assertIn("08:30", morning["spec"])
        self.assertIn("18:30", evening["spec"])
        self.assertEqual(morning["timezone"], "Africa/Lagos")
        self.assertEqual(evening["timezone"], "Africa/Lagos")
        self.assertEqual(morning["payload_kind"], "tool")
        self.assertEqual(morning["missed_fire_policy"], "fire_now")
        self.assertEqual(morning["overlap_policy"], "skip")
        self.assertTrue(morning["enabled"])
        self.assertTrue(morning["next_run"])

    def test_payload_is_the_sweep_action_and_the_job_is_routine(self) -> None:
        ensure_explore_jobs(self.scheduler)
        row = self.db.query_one(
            "SELECT payload, routine FROM schedule_jobs WHERE name = ?",
            (EXPLORE_SWEEP_JOB_MORNING,))
        self.assertEqual(json.loads(row["payload"]),
                         {"tool": "video_explore",
                          "args": {"action": "sweep"}})
        # Census §7: routine — findings reach him via the briefing
        # + the board, not job alerts (scheduler maps this to
        # routine=1).
        self.assertTrue(row["routine"])

    def test_idempotent_across_boots(self) -> None:
        ensure_explore_jobs(self.scheduler)
        again = ensure_explore_jobs(self.scheduler)
        self.assertEqual(
            [bool(o.get("already_scheduled")) for o in again],
            [True, True])
        names = [j["name"] for j in self.scheduler.list_jobs()
                 if j["name"] in EXPLORE_SWEEP_JOBS]
        self.assertEqual(sorted(names), sorted(EXPLORE_SWEEP_JOBS))


# ----------------------------------------------------------------
# (b) a job fire runs sweep → evaluation
# ----------------------------------------------------------------


def test_job_fire_runs_sweep_then_evaluation(tmp_path, monkeypatch):
    db = Database(":memory:")
    db.migrate()
    ctx = _ctx(db=db, tmp_path=tmp_path)
    store = ExploreStore()
    fake = FakeInstrument("fake_web", [_row("Gold Trading Masterclass")])
    registry = _registry(ctx, monkeypatch, store, [fake])

    first = registry.call("video_explore", action="sweep")
    assert first.ok, first.error
    assert first.value["sweep"]["trigger"] == "scheduled"
    assert store.observations()
    kinds = {c["kind"] for c in current_claims(store)}
    assert "trend" not in kinds  # one sweep can never be a trend

    second = registry.call("video_explore", action="sweep")
    assert second.ok, second.error
    assert second.value["evaluation"]["claims_current"] >= 1
    trends = [c for c in current_claims(store) if c["kind"] == "trend"]
    assert trends and trends[0]["confidence_basis"]

    # The cap is real through the same door: the day's two
    # scheduled sweeps are spent.
    third = registry.call("video_explore", action="sweep")
    assert third.ok, third.error
    assert third.value["deferred"] == "daily_cap"
    assert len(store.sweeps()) == 2


def test_scheduler_tick_fires_the_stored_payload(tmp_path, monkeypatch):
    db = Database(":memory:")
    db.migrate()
    ctx = _ctx(db=db, tmp_path=tmp_path)
    store = ExploreStore()
    fake = FakeInstrument("fake_web", [_row("Gold Trading Masterclass")])
    _registry(ctx, monkeypatch, store, [fake])
    scheduler = Scheduler(ctx)
    ensure_explore_jobs(scheduler)
    # Make both jobs due, then let the scheduler's own tick fire.
    db.execute("UPDATE schedule_jobs SET next_run = 0 "
               "WHERE name IN (?, ?)",
               (EXPLORE_SWEEP_JOB_MORNING, EXPLORE_SWEEP_JOB_EVENING))
    results = scheduler.tick()
    sweep_runs = [r for r in results
                  if r.get("name") in EXPLORE_SWEEP_JOBS]
    assert sweep_runs, results
    assert all(r["ok"] for r in sweep_runs)
    # Two fires in one tick: the first sweeps, the second is the
    # day's last allowed slot — or defers if the first already
    # consumed… both may run; either way at least one real sweep
    # landed and every claim on record came from the evaluation.
    assert store.sweeps()
    assert store.observations()


def test_failed_sweep_surfaces_as_a_failed_job_run(tmp_path, monkeypatch):
    """A sweep that dies inside the cycle raises through the tool,
    so the scheduler records the run FAILED — never a silent
    success (the P1 engine records the failed sweep row first)."""
    db = Database(":memory:")
    db.migrate()
    ctx = _ctx(db=db, tmp_path=tmp_path)
    store = ExploreStore()
    registry = _registry(ctx, monkeypatch, store,
                         [FakeInstrument("fake_web", [_row("x")])])
    monkeypatch.setattr(
        engine_mod.SweepEngine, "run_sweep",
        lambda self, *a, **k: (_ for _ in ()).throw(
            RuntimeError("boom")))
    outcome = registry.call("video_explore", action="sweep")
    assert outcome.ok is False


# ----------------------------------------------------------------
# (c) quiet hours + the daily cap
# ----------------------------------------------------------------


def test_quiet_hours_defer_a_scheduled_sweep(monkeypatch):
    store = ExploreStore()
    ctx = _ctx(partner=_partner(22, 8))
    fake = FakeInstrument("fake_web", [_row("Gold Trading Masterclass")])
    cycle = run_sweep_cycle(ctx, store=store, instruments=[fake],
                            beats=_beats(), now=_lagos_ts(9, 23, 30))
    assert cycle["deferred"] == "quiet_hours"
    assert cycle["ok"] is True
    assert store.sweeps() == []
    assert fake.calls == []


def test_outside_quiet_hours_the_sweep_runs(monkeypatch):
    store = ExploreStore()
    ctx = _ctx(partner=_partner(22, 8))
    fake = FakeInstrument("fake_web", [_row("Gold Trading Masterclass")])
    cycle = run_sweep_cycle(ctx, store=store, instruments=[fake],
                            beats=_beats(), now=_lagos_ts(9, 12, 0))
    assert cycle["deferred"] == ""
    assert cycle["sweep"]["trigger"] == "scheduled"
    assert store.sweeps()


def test_manual_cycle_is_never_gated(monkeypatch):
    store = ExploreStore()
    ctx = _ctx(partner=_partner(22, 8))
    fake = FakeInstrument("fake_web", [_row("Gold Trading Masterclass")])
    cycle = run_sweep_cycle(ctx, manual=True, store=store,
                            instruments=[fake], beats=_beats(),
                            now=_lagos_ts(9, 23, 30))
    assert cycle["deferred"] == ""
    assert cycle["sweep"]["trigger"] == "manual"


def test_daily_cap_defers_the_third_scheduled_sweep(monkeypatch):
    store = ExploreStore()
    ctx = _ctx()  # no partner settings: no quiet window
    fake = FakeInstrument("fake_web", [_row("Gold Trading Masterclass")])
    for hour in (9, 12):
        cycle = run_sweep_cycle(ctx, store=store, instruments=[fake],
                                beats=_beats(), now=_lagos_ts(9, hour))
        assert cycle["deferred"] == "", cycle
    third = run_sweep_cycle(ctx, store=store, instruments=[fake],
                            beats=_beats(), now=_lagos_ts(9, 15))
    assert third["deferred"] == "daily_cap"
    assert len(store.sweeps()) == 2
    # A manual sweep still runs — and does not consume the cap.
    manual = run_sweep_cycle(ctx, manual=True, store=store,
                             instruments=[fake], beats=_beats(),
                             now=_lagos_ts(9, 16))
    assert manual["deferred"] == ""
    assert len(store.sweeps()) == 3
    fourth = run_sweep_cycle(ctx, store=store, instruments=[fake],
                             beats=_beats(), now=_lagos_ts(9, 17))
    assert fourth["deferred"] == "daily_cap"


def test_next_sweep_due_reads_the_registered_jobs(tmp_path):
    db = Database(":memory:")
    db.migrate()
    ctx = _ctx(db=db, tmp_path=tmp_path)
    scheduler = Scheduler(ctx)
    assert jobs_mod.next_sweep_due(scheduler) is None
    ensure_explore_jobs(scheduler)
    due = jobs_mod.next_sweep_due(scheduler)
    assert due is not None
    assert due["name"] in EXPLORE_SWEEP_JOBS
    assert due["at"] > 0


# ----------------------------------------------------------------
# (d) the board
# ----------------------------------------------------------------


def _swept_store() -> tuple[ExploreStore, float]:
    store = ExploreStore()
    engine = engine_mod.SweepEngine(store, [
        FakeInstrument("fake_web", [_row("Gold Trading Masterclass")]),
        FakeInstrument("fake_dead", []),
    ])
    dead = engine.instruments[1]

    def _boom(query, *, limit=10):
        raise RuntimeError("provider down")

    dead.search = _boom  # type: ignore[method-assign]
    t1 = time.time() - 3600
    engine.run_sweep(_beats(), trigger="scheduled", now=t1)
    engine.run_sweep(_beats(), trigger="scheduled", now=t1 + 600)
    evaluate_trends(store, now=t1 + 600)
    return store, t1


class BoardTests(unittest.TestCase):

    def test_empty_board_states_the_truth(self) -> None:
        # doc 17 P1: the board's proxy line is state-dependent now
        # (it renders the host's activation record) — pin the
        # not-yet-activated form by fixing the state seam to None.
        with mock.patch.object(board_mod, "_activation_state",
                               return_value=None):
            text = render_explore_board(store=ExploreStore())
        self.assertIn("none yet", text)
        self.assertIn("not scheduled", text)
        self.assertIn(PROXY_DEPENDENCY_LINE, text)

    def test_board_renders_last_sweep_truth_claims_and_candidates(
            self) -> None:
        store, _t = _swept_store()
        scheduler = Scheduler(_ctx(db=_fresh_db()))
        ensure_explore_jobs(scheduler)
        # not-yet-activated form (doc 17 P1) — see above.
        with mock.patch.object(board_mod, "_activation_state",
                               return_value=None):
            text = render_explore_board(store=store,
                                        scheduler=scheduler)
        # Last sweep: beats, observations, and the failed provider
        # counted as a failure — never smoothed over.
        self.assertIn("last sweep:", text)
        self.assertIn("1 failed", text)
        # Next sweep due, from the registered jobs.
        self.assertIn("next sweep:", text)
        self.assertNotIn("not scheduled", text)
        # Current claim with its kind, candidate standings, pool.
        self.assertIn("[trend]", text)
        self.assertIn("candidates:", text)
        self.assertIn("watching", text)
        self.assertIn(PROXY_DEPENDENCY_LINE, text)
        # No internal record id renders, anywhere.
        self.assertNotIn("sweep-", text)
        self.assertNotIn("obs-", text)
        self.assertNotIn("claim:", text)

    def test_board_names_a_failed_sweep_failed(self) -> None:
        store = ExploreStore()
        at = time.time() - 60
        sid = store.begin_sweep(trigger="scheduled",
                                beats=["gold trading"], at=at)
        store.finish_sweep(sid, status="failed", at=at + 5)
        text = render_explore_board(store=store)
        self.assertIn("FAILED", text)
        self.assertIn("did not finish", text)

    def test_trends_render_carries_the_basis(self) -> None:
        store, _t = _swept_store()
        text = render_explore_trends(store=store)
        self.assertIn("trend", text)
        # P2's render seam: the basis in plain words — sweep count,
        # source classes — and no record id anywhere in it.
        self.assertIn("sweep(s)", text)
        self.assertIn("classes:", text)
        self.assertNotIn("sweep-", text)

    def test_candidates_render_shows_standings(self) -> None:
        store, _t = _swept_store()
        text = render_explore_candidates(store=store)
        # The surfacing channel is noted as a watching candidate.
        self.assertIn("Chan", text)
        self.assertIn("seen", text)


def _fresh_db() -> Database:
    db = Database(":memory:")
    db.migrate()
    return db


# ----------------------------------------------------------------
# (e) /explore run — the same entry point the job calls
# ----------------------------------------------------------------


def test_control_explore_run_calls_the_cycle_manually(monkeypatch):
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    calls: list[dict] = []

    def _spy(context, **kwargs):
        calls.append({"context": context, **kwargs})
        return {"ok": True, "manual": True, "deferred": "",
                "sweep": {"beats": [], "beats_deferred": [],
                          "observations": 0, "results": 0, "empty": 0,
                          "failed": 0, "new_candidates": 0,
                          "candidates_deferred": 0},
                "evaluation": {"claims_current": 0, "claims_written": 0,
                               "promoted": 0, "shed": 0, "declined": 0}}

    monkeypatch.setattr(jobs_mod, "run_sweep_cycle", _spy)
    surface = RuntimeMediaMixin()
    surface.context = _ctx()
    surface._scheduler = None
    text = surface._control_explore("run")
    assert calls and calls[0]["manual"] is True
    assert calls[0]["context"] is surface.context
    assert "Exploration sweep complete" in text


def test_control_explore_board_and_usage(monkeypatch):
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    store = ExploreStore()
    monkeypatch.setattr(engine_mod, "_DEFAULT_STORE", store)
    # doc 17 P1: the proxy line renders the host's activation
    # record — pin the not-yet-activated form hermetically.
    monkeypatch.setattr(board_mod, "_activation_state", lambda ctx: None)
    surface = RuntimeMediaMixin()
    surface.context = _ctx()
    surface._scheduler = None
    board_text = _text(surface._control_explore(""))
    assert "Exploration" in board_text
    assert PROXY_DEPENDENCY_LINE in board_text
    trends_text = surface._control_explore("trends")
    assert "Exploration trends" in trends_text
    candidates_text = surface._control_explore("candidates")
    assert "candidates" in candidates_text
    usage = surface._control_explore("bogus")
    assert usage.startswith("usage: /explore")


# ----------------------------------------------------------------
# (f) the briefing provider
# ----------------------------------------------------------------


def _swept_store_at(at: float) -> ExploreStore:
    store = ExploreStore()
    engine = engine_mod.SweepEngine(store, [
        FakeInstrument("fake_web", [_row("Gold Trading Masterclass")])])
    engine.run_sweep(_beats(), trigger="scheduled", now=at)
    engine.run_sweep(_beats(), trigger="scheduled", now=at + 300)
    evaluate_trends(store, now=at + 300)
    return store


def _seeded_trend_store(at: float) -> ExploreStore:
    """A corroborated beat trend whose EVIDENCE sits at ``at`` —
    seeded the way P2's suite seeds: observations carry their own
    timestamps (the engine stamps real time by design, so
    time-travel pins go through the store's record seam)."""
    store = ExploreStore()
    rows = [{"rank": 1, "title": "Gold Trading Masterclass",
             "url": "https://youtu.be/gold1", "source": "youtube",
             "channel": "Chan", "view_count": 5000,
             "duration_s": None}]
    for start in (at, at + 300):
        sid = store.begin_sweep(trigger="scheduled",
                                beats=["gold trading"], at=start)
        store.record_observation({
            "sweep_id": sid, "beat": "gold trading",
            "instrument": "web_search", "query": "gold trading",
            "at": start + 5, "status": "ok", "error": "",
            "result_count": len(rows), "results": rows})
        store.finish_sweep(sid, at=start + 10)
    evaluate_trends(store, now=at + 300)
    return store


class BriefingProviderTests(unittest.TestCase):

    def setUp(self) -> None:
        from unittest import mock

        self._patch = mock.patch.object(engine_mod, "_DEFAULT_STORE",
                                        ExploreStore())
        self._patch.start()
        self.addCleanup(self._patch.stop)
        from nomorals.agents.morning_briefing import ExploreTrendsProvider

        self.provider = ExploreTrendsProvider()
        self.ctx = _ctx()

    def test_priority_and_identity(self) -> None:
        self.assertEqual(self.provider.name, "explore_trends")
        self.assertEqual(self.provider.priority, 26)

    def test_absent_when_the_store_is_empty(self) -> None:
        self.assertIsNone(self.provider.collect(self.ctx, 0.0))

    def test_renders_a_fresh_trend_with_its_basis(self) -> None:
        store = _swept_store_at(time.time() - 600)
        engine_mod._DEFAULT_STORE = store
        section = self.provider.collect(self.ctx, time.time() - 3600)
        self.assertIsNotNone(section)
        self.assertEqual(section.title, "🔭 Moving in your field")
        self.assertTrue(any("basis:" in ln for ln in section.lines))
        self.assertTrue(any("/explore" in ln for ln in section.lines))
        for ln in section.lines:
            # No internal record id prints — the authored basis
            # prose enumerates sweep ids; the provider renders the
            # structured basis instead.
            self.assertNotIn("claim:", ln)
            self.assertNotIn("sweep-", ln)

    def test_silent_when_the_trend_predates_the_window(self) -> None:
        store = _seeded_trend_store(time.time() - 7200)
        engine_mod._DEFAULT_STORE = store
        self.assertIsNone(
            self.provider.collect(self.ctx, time.time() - 60))

    def test_silent_when_the_trend_is_stale(self) -> None:
        store = _seeded_trend_store(time.time() - 10 * 86400)
        engine_mod._DEFAULT_STORE = store
        # Window wide open (since = the evidence era): only the
        # freshness leg can silence it — a 10-day-old trend claim
        # is stale (3-day freshness) and never briefs.
        self.assertIsNone(
            self.provider.collect(self.ctx, time.time() - 11 * 86400))

    def test_a_failed_sweep_is_news(self) -> None:
        store = ExploreStore()
        at = time.time() - 120
        sid = store.begin_sweep(trigger="scheduled",
                                beats=["gold trading"], at=at)
        store.finish_sweep(sid, status="failed", at=at + 5)
        engine_mod._DEFAULT_STORE = store
        section = self.provider.collect(self.ctx, time.time() - 3600)
        self.assertIsNotNone(section)
        self.assertTrue(any("failed" in ln for ln in section.lines))


# ----------------------------------------------------------------
# (g) catalog pins
# ----------------------------------------------------------------


class CatalogTests(unittest.TestCase):

    def test_kind_registered_owner_only(self) -> None:
        from nomorals.social.chat.control import (
            CONTROL_COMMANDS,
            PUBLIC_COMMANDS,
            caller_may_run,
        )

        self.assertIn("explore", CONTROL_COMMANDS)
        self.assertNotIn("explore", PUBLIC_COMMANDS)
        self.assertFalse(caller_may_run("explore", is_owner=False))
        self.assertTrue(caller_may_run("explore", is_owner=True))

    def test_details_list_and_help_pins(self) -> None:
        from nomorals.social.chat import control as ctl

        self.assertIn("explore", ctl.COMMAND_DETAILS)
        self.assertIn("/explore", ctl.detailed_help("explore"))
        self.assertIn("explore", ctl.LIST_ONELINERS)
        grouped = [k for _n, keys in ctl.LIST_GROUPS for k in keys]
        self.assertEqual(grouped.count("explore"), 1)
        helped = [k for _n, keys in ctl._HELP_GROUPS for k in keys]
        self.assertEqual(helped.count("explore"), 1)

    def test_parse(self) -> None:
        from nomorals.social.chat.control import parse_control

        self.assertEqual(parse_control("/explore").kind, "explore")
        self.assertEqual(parse_control("/explore trends").kind, "explore")
        self.assertEqual(parse_control("/explore trends").tail, "trends")

    def test_seed_beats_still_ship(self) -> None:
        # The autonomy watches the beats P1 seeded — the loop has
        # input from the first boot.
        self.assertTrue(EXPLORE_SEED_BEATS)


if __name__ == "__main__":
    unittest.main()
