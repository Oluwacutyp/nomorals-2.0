"""Gapfix lane 10 — the calibration -> abstention -> learning-targets
loop (audit gap R1-9) actually runs.

Before the fix: no CalibrationStore is ever constructed in production,
threshold_from_calibration and learning_targets have zero callers, and
trajectory rows never feed calibration — so the loop cannot learn from
overconfidence. These tests drive the PRODUCTION entry points (the
scheduler job + the __calibration_loop__ tool + the loop module), never
the dormant organs in isolation.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest


def _scheduler_ctx(tmp_path):
    from nomorals.storage.db import Database

    db = Database(":memory:")
    db.migrate()
    settings = SimpleNamespace(
        partner=SimpleNamespace(owner_chats="telegram:111"),
        resolve=lambda rel: tmp_path / rel,
        workspace_dir=str(tmp_path),
    )
    return SimpleNamespace(db=db, settings=settings, tools=None, extras={})


def _capture_tools(ctx):
    """Register the autonomy tools into a capturing registry."""
    from nomorals.tools import autonomy as autonomy_tools

    captured: dict = {}

    class _Registry:
        def __init__(self):
            self.context = ctx

        def register(self, name, fn=None, **kw):
            if fn is None:
                def _deco(f):
                    captured[name] = f
                    return f
                return _deco
            captured[name] = fn
            return fn

    autonomy_tools.register(_Registry())
    return captured


def _seed_drift(db, *, task="chat", model="model-a",
                good=5, bad=5):
    """A slice that starts perfect then collapses: the track record
    says ~0.9 while the empirical rate ends at 0.5 — overconfident."""
    from nomorals.cognition.trajectories import TrajectoryStore

    store = TrajectoryStore(db)
    for _ in range(good):
        store.record(task_kind=task, capability="chat", model_id=model,
                     success=True)
    for _ in range(bad):
        store.record(task_kind=task, capability="chat", model_id=model,
                     success=False, error="TimeoutError: provider timed out")


def _seed_good(db, *, task="chat", model="model-good", n=6):
    from nomorals.cognition.trajectories import TrajectoryStore

    store = TrajectoryStore(db)
    for _ in range(n):
        store.record(task_kind=task, capability="chat", model_id=model,
                     success=True)


# ── replay ────────────────────────────────────────────────────────────


def test_replay_moves_calibration_state(tmp_path):
    """Trajectory rows replayed into a store carry real calibration
    signal: the drift slice ends empirical 0.5 with mean predicted
    well above it (the track record was saying ~0.9)."""
    from nomorals.cognition.calibration_loop import build_calibration_store

    ctx = _scheduler_ctx(tmp_path)
    _seed_drift(ctx.db)
    store, stats = build_calibration_store(ctx.db)
    assert stats["replayed"] == 10
    assert stats["skipped"] == 0
    assert store.empirical_rate("chat", "model-a") == pytest.approx(0.5)
    mean_pred = store.mean_predicted("chat", "model-a")
    assert mean_pred is not None and mean_pred > 0.7
    gap = store.calibration_gap("chat", "model-a")
    assert gap is not None and gap > 0.2


def test_learning_targets_ranked_from_real_rows(tmp_path):
    """The worst slice (failing + overconfident) ranks first; a
    clean slice never outranks it. The loop rebuilds the store; the
    caller ranks it — here the test plays the caller's part."""
    from nomorals.autonomy.model_weakness import learning_targets
    from nomorals.cognition.calibration_loop import run_calibration_loop_once

    ctx = _scheduler_ctx(tmp_path)
    _seed_drift(ctx.db)
    _seed_good(ctx.db)
    result = run_calibration_loop_once(ctx.db)
    assert result["ok"] is True
    targets = learning_targets(result["store"])
    assert targets, "the drift slice must be a learning target"
    top = targets[0]
    assert top["task"] == "chat" and top["model"] == "model-a"
    assert top["uncertainty"] > 0.5
    assert top["n_samples"] == 10


def test_threshold_moves_when_overconfident_not_on_empty_data(tmp_path):
    """Overconfidence lifts the abstention bar above the 0.55 base;
    no data leaves the base alone (honest default, not a stricter
    one)."""
    from nomorals.cognition.calibration_loop import (
        run_calibration_loop_once,
        threshold_for,
    )

    ctx = _scheduler_ctx(tmp_path)
    result = run_calibration_loop_once(ctx.db)
    assert result["ok"] is True
    assert result["thresholds"] == {}
    assert threshold_for(ctx.db, "chat", "model-a") == pytest.approx(0.55)

    _seed_drift(ctx.db)
    result = run_calibration_loop_once(ctx.db)
    assert result["thresholds"].get("chat::model-a", 0.55) > 0.55
    assert threshold_for(ctx.db, "chat", "model-a") > 0.55


def test_second_replay_does_not_double_count(tmp_path):
    """A restart / second tick rebuilds the same state — replaying
    twice never doubles the records, and the threshold table keeps
    one row per slice."""
    from nomorals.cognition.calibration_loop import (
        build_calibration_store,
        run_calibration_loop_once,
    )

    ctx = _scheduler_ctx(tmp_path)
    _seed_drift(ctx.db)
    first = run_calibration_loop_once(ctx.db)
    second = run_calibration_loop_once(ctx.db)
    assert first["replayed"] == second["replayed"] == 10
    _, stats1 = build_calibration_store(ctx.db)
    _, stats2 = build_calibration_store(ctx.db)
    assert stats1["replayed"] == stats2["replayed"] == 10
    rows = ctx.db.query(
        "SELECT task, model FROM cog_calibration_thresholds")
    assert len(rows) == 1
    assert (rows[0]["task"], rows[0]["model"]) == ("chat", "model-a")


def test_bad_row_never_kills_tick(tmp_path):
    """A hand-corrupted trajectory row (impossible success value,
    empty slice) is skipped and counted — the tick still reports ok
    and still learns from the good rows."""
    from nomorals.cognition.calibration_loop import run_calibration_loop_once

    ctx = _scheduler_ctx(tmp_path)
    _seed_drift(ctx.db)
    ctx.db.execute(
        "INSERT INTO cog_trajectories (id, task_kind, capability,"
        " model_id, skill_id, tool, success, latency_s, cost,"
        " error, failure_class, created_at)"
        " VALUES ('traj_bad1', 'chat', 'chat', 'model-a', '', '',"
        " 7, 0.0, 0.0, '', '', ?)",
        (time.time(),))
    ctx.db.execute(
        "INSERT INTO cog_trajectories (id, task_kind, capability,"
        " model_id, skill_id, tool, success, latency_s, cost,"
        " error, failure_class, created_at)"
        " VALUES ('traj_bad2', '', '', '', '', '',"
        " 1, 0.0, 0.0, '', '', ?)",
        (time.time(),))
    result = run_calibration_loop_once(ctx.db)
    assert result["ok"] is True
    assert result["replayed"] == 10
    assert result["skipped"] == 2
    from nomorals.autonomy.model_weakness import learning_targets

    assert learning_targets(result["store"]), (
        "good rows must still produce a target")


def test_weaknesses_filed_in_shared_ledger(tmp_path):
    """Top learning targets land as model weaknesses in the shared
    ledger — the same ledger the autonomy tools and coordinator read.
    Filing happens on the tick (the tool), not in the L3 loop module."""
    from nomorals.autonomy.weakness import open_weaknesses

    ctx = _scheduler_ctx(tmp_path)
    _seed_drift(ctx.db)
    captured = _capture_tools(ctx)
    result = captured["__calibration_loop__"]()
    assert result["ok"] is True
    assert result["weaknesses_filed"], "the target must be filed"
    filed = open_weaknesses(ctx.db)
    assert any(
        w["kind"] == "model_weakness"
        and w["subject"] == "model:chat:model-a"
        for w in filed)


# ── scheduler job + tool ──────────────────────────────────────────────


def test_calibration_loop_job_registration_idempotent(tmp_path):
    from nomorals.agents.scheduler import Scheduler
    from nomorals.cognition.calibration_loop import (
        CALIBRATION_LOOP_JOB,
        CALIBRATION_LOOP_TOOL,
        ensure_calibration_loop_job,
    )

    ctx = _scheduler_ctx(tmp_path)
    sched = Scheduler(ctx)
    job = ensure_calibration_loop_job(sched)
    assert job is not None and job["name"] == CALIBRATION_LOOP_JOB
    assert ensure_calibration_loop_job(sched)["id"] == job["id"]
    jobs = [j for j in sched.list_jobs() if j["name"] == CALIBRATION_LOOP_JOB]
    assert len(jobs) == 1
    row = ctx.db.query_one(
        "SELECT payload FROM schedule_jobs WHERE id = ?", (job["id"],))
    assert json.loads(row["payload"])["tool"] == CALIBRATION_LOOP_TOOL


def test_calibration_loop_tool_registered_and_runs(tmp_path):
    ctx = _scheduler_ctx(tmp_path)
    _seed_drift(ctx.db)
    captured = _capture_tools(ctx)
    assert "__calibration_loop__" in captured
    result = captured["__calibration_loop__"]()
    assert result["ok"] is True
    assert result["replayed"] == 10
    assert result["targets"], "the tool run must compute targets"
    assert result["thresholds"].get("chat::model-a", 0.55) > 0.55
    from nomorals.autonomy.weakness import open_weaknesses

    assert any(
        w["kind"] == "model_weakness" for w in open_weaknesses(ctx.db))


def test_scheduler_fires_calibration_loop_end_to_end(tmp_path):
    """The real Scheduler fires the real __calibration_loop__ tool job
    and the thresholds + weakness land — the full production path."""
    from nomorals.agents.scheduler import Scheduler
    from nomorals.cognition.calibration_loop import (
        CALIBRATION_LOOP_JOB,
        ensure_calibration_loop_job,
    )
    from nomorals.storage.db import Database

    ctx = _scheduler_ctx(tmp_path)
    # The scheduler executes on its own thread; a :memory: Database
    # is per-connection, so the fired tool must share a FILE store
    # with the assertions (as in production).
    ctx.db = Database(str(tmp_path / "fired.db"))
    ctx.db.migrate()
    _seed_drift(ctx.db)
    captured = _capture_tools(ctx)

    def _call(tool, capabilities=None, **kw):
        return SimpleNamespace(ok=True, value=captured[tool](), error=None)

    ctx.tools = SimpleNamespace(call=_call)
    sched = Scheduler(ctx)
    job = ensure_calibration_loop_job(sched)
    ctx.db.execute("UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
                   (time.time() - 1, job["id"]))
    outcomes = sched.tick()
    assert any(o.get("id") == job["id"] for o in outcomes)
    rows = ctx.db.query("SELECT task, model, threshold"
                        " FROM cog_calibration_thresholds")
    assert len(rows) == 1 and rows[0]["threshold"] > 0.55
    jobs = [j for j in sched.list_jobs() if j["name"] == CALIBRATION_LOOP_JOB]
    assert len(jobs) == 1
