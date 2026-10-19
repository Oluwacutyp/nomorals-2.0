"""Failing-first tests for nomorals/core/jobs.py — heartbeat + background jobs.

Semantics under test:
- a slow job that heartbeats is never flagged (no blunt timeout);
- a job whose heartbeat goes silent is presumed dead — honest UNKNOWN;
- a late result after presumed-dead is recorded, never a fresh success;
- the caller never blocks on submit; collection is explicit;
- every job has visible state (no silent drops).
"""

import threading
import time

import pytest

from nomorals.core.jobs import (
    HEARTBEAT_INTERVAL_S,
    MAX_MISSED_HEARTBEATS,
    JobRegistry,
    report_progress,
)


def _fresh() -> JobRegistry:
    return JobRegistry()


def _wait_done(reg: JobRegistry, job_id: str, timeout: float = 10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = reg.poll(job_id)
        if job is not None and job.done():
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} not done within {timeout}s")


# ── 1. background execution: submit never blocks ──────────────────────

def test_submit_returns_immediately_and_job_completes():
    reg = _fresh()
    started = threading.Event()

    def slow():
        started.set()
        time.sleep(0.5)
        return "result-value"

    t0 = time.time()
    job = reg.submit(slow, tool="slow_tool", args={"x": 1})
    dt = time.time() - t0
    assert dt < 0.5, f"submit blocked {dt:.2f}s on a slow tool"
    assert job.status == "running"
    assert started.wait(2), "worker never started"
    done = _wait_done(reg, job.id)
    assert done.status == "done"
    assert done.result == "result-value"
    assert done.error == ""


def test_failed_job_records_error_honestly():
    reg = _fresh()

    def boom():
        raise RuntimeError("kaput")

    job = reg.submit(boom, tool="boom")
    done = _wait_done(reg, job.id)
    assert done.status == "failed"
    assert "RuntimeError" in done.error
    assert "kaput" in done.error


# ── 2. heartbeats: slow but alive is never flagged ────────────────────

def test_heartbeat_events_emitted_while_running():
    from nomorals.core.events import global_bus

    reg = _fresh()
    seen: list = []
    evt = threading.Event()
    sub = global_bus.subscribe(
        "tool.heartbeat", lambda e: (seen.append(e), evt.set()),
        predicate=lambda e: e.data.get("job_id") == job.id,
    )
    try:
        job = reg.submit(lambda: time.sleep(2.5), tool="sleeper",
                         heartbeat_interval_s=1.0)
        assert evt.wait(5), "no heartbeat event arrived for a live job"
        assert seen[0].data["alive"] is True
        assert seen[0].data["job_id"] == job.id
    finally:
        global_bus.unsubscribe(sub)
    _wait_done(reg, job.id)


def test_slow_heartbeating_job_not_presumed_dead():
    reg = _fresh()
    job = reg.submit(lambda: time.sleep(2.0), tool="slow",
                     heartbeat_interval_s=1.0)
    time.sleep(1.5)  # several heartbeat intervals pass
    flagged = reg.reap()
    assert flagged == 0
    job = reg.poll(job.id)
    assert job.status == "running"
    assert job.heartbeat_count >= 1
    _wait_done(reg, job.id)


def test_report_progress_attaches_note():
    reg = _fresh()

    def with_note():
        report_progress("halfway there")
        time.sleep(1.5)
        return "ok"

    job = reg.submit(with_note, tool="noted", heartbeat_interval_s=1.0)
    done = _wait_done(reg, job.id)
    assert done.last_progress == "halfway there"


# ── 3. dead-man's switch: silence → presumed dead, honest UNKNOWN ─────

def test_silent_job_presumed_dead_with_unknown_outcome():
    reg = _fresh()
    gate = threading.Event()

    def stuck():
        gate.wait(30)
        return "eventually"

    job = reg.submit(stuck, tool="stuck", heartbeat_interval_s=1.0)
    # Simulate lost visibility: heartbeat stopped long ago.
    with job.lock:
        job.last_heartbeat_at -= (
            HEARTBEAT_INTERVAL_S * MAX_MISSED_HEARTBEATS + 5)
    flagged = reg.reap()
    assert flagged == 1
    job = reg.poll(job.id)
    assert job.status == "presumed-dead"
    assert job.outcome_unknown is True
    assert "no heartbeat" in job.error
    assert "UNKNOWN" in job.error
    assert "never re-issue blindly" in job.error
    # A late result is still collected and recorded — never a fresh success.
    gate.set()
    worker = reg._threads[job.id][0]
    worker.join(35)
    assert not worker.is_alive(), "worker thread did not finish"
    done = reg.poll(job.id)
    assert done.status == "presumed-dead"
    assert done.late_result is True
    assert done.result == "eventually"


def test_collect_returns_presumed_dead_job():
    reg = _fresh()
    gate = threading.Event()

    def stuck():
        gate.wait(30)
        return "late"

    job = reg.submit(stuck, tool="stuck2", heartbeat_interval_s=1.0)
    with job.lock:
        job.last_heartbeat_at -= (
            HEARTBEAT_INTERVAL_S * MAX_MISSED_HEARTBEATS + 5)
    got = reg.collect(job.id, timeout=1.0)
    assert got.status == "presumed-dead"
    gate.set()
    reg._threads[job.id][0].join(35)


# ── 4. collection semantics ───────────────────────────────────────────

def test_collect_ready_harvests_once():
    reg = _fresh()
    j1 = reg.submit(lambda: "a", tool="t1")
    j2 = reg.submit(lambda: "b", tool="t2")
    _wait_done(reg, j1.id)
    _wait_done(reg, j2.id)
    ready = reg.collect_ready()
    assert {j.id for j in ready} == {j1.id, j2.id}
    assert reg.collect_ready() == [], "harvested jobs returned twice"


def test_collect_ready_label_filter():
    reg = _fresh()
    j1 = reg.submit(lambda: "a", tool="t1", labels={"owner": "devon"})
    j2 = reg.submit(lambda: "b", tool="t2", labels={"owner": "loop"})
    _wait_done(reg, j1.id)
    _wait_done(reg, j2.id)
    mine = reg.collect_ready(labels={"owner": "devon"})
    assert [j.id for j in mine] == [j1.id]
    rest = reg.collect_ready()
    assert [j.id for j in rest] == [j2.id]


def test_collect_with_timeout_returns_unfinished_job():
    reg = _fresh()
    job = reg.submit(lambda: time.sleep(3), tool="slow3")
    got = reg.collect(job.id, timeout=0.2)
    assert got is not None
    assert not got.done(), "collect(timeout) must not block on a slow job"
    assert got.status == "running"
    _wait_done(reg, job.id, timeout=10)


def test_poll_missing_job_returns_none():
    assert _fresh().poll("job-does-not-exist") is None


# ── 5. visibility: no silent drops ────────────────────────────────────

def test_list_shows_every_state():
    reg = _fresh()
    running = reg.submit(lambda: time.sleep(2), tool="r",
                         heartbeat_interval_s=1.0)
    done_job = reg.submit(lambda: "x", tool="d")
    _wait_done(reg, done_job.id)
    states = {s["id"]: s["status"] for s in reg.list()}
    assert states[running.id] == "running"
    assert states[done_job.id] == "done"
    assert reg.list(status="running") != []
    assert all(s["status"] == "running" for s in reg.list(status="running"))
    snap = reg.list()[0]
    for key in ("id", "tool", "status", "elapsed_s", "heartbeat_count",
                "last_progress", "late_result", "collected", "labels"):
        assert key in snap, f"snapshot missing {key}"
    _wait_done(reg, running.id)


def test_sweep_drops_only_old_terminal_jobs():
    reg = _fresh()
    old = reg.submit(lambda: "x", tool="old")
    _wait_done(reg, old.id)
    with old.lock:
        old.ended_at -= 4000  # older than TERMINAL_JOB_TTL_S
    fresh = reg.submit(lambda: "y", tool="fresh")
    _wait_done(reg, fresh.id)
    dropped = reg.sweep()
    assert dropped == 1
    ids = {s["id"] for s in reg.list()}
    assert old.id not in ids
    assert fresh.id in ids
