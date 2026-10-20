"""Immune wiring build — the adaptive gate, wired end-to-end.

Before this build (pass #28 state, HEAD 33810579):

* ``adaptive_consult_then_enforce`` had NO production caller — nothing
  recorded live samples at runtime, so the baseline stayed at 0
  samples and the gate could never arm, while ``/immune status``
  printed "cancer gate: adaptive p95 baseline" as if it were live.
* ``AdaptiveBaseline`` was in-memory only — a restart reset it to
  0 samples (report-only forever).
* Nothing stopped incident samples from being learned: 600 recorded
  runaway samples pushed p95 to the runaway's own level, after which a
  runaway read "below thresholds" (demonstrated in pass #28).
* The sample list grew unbounded.

These tests pin the wired behaviour: persistence, the sampler, the
incident quarantine, the recency window, the gate as the live decision
path, and honest status.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from nomorals.immune.baseline import (
    AdaptiveBaseline,
    collect_process_sample,
    sample_over_thresholds,
)
from nomorals.immune.chat import control_immune, reset_immune_state
from nomorals.immune.wiring import (
    BASELINE_WINDOW,
    IMMUNE_SAMPLER_JOB,
    IMMUNE_SAMPLER_TOOL,
    BaselineStore,
    ImmuneMonitor,
    default_store_path,
    ensure_immune_sampler_job,
    reset_shared_monitors,
    shared_monitor,
)

NORMAL = {"processes_spawned": 100.0, "process_growth": 1.0}
RUNAWAY = {"processes_spawned": 5000.0, "process_growth": 50.0}


@pytest.fixture(autouse=True)
def _clean_monitors():
    reset_shared_monitors()
    yield
    reset_shared_monitors()


def _ready_monitor(tmp_path, *, samples=40, **kw) -> ImmuneMonitor:
    """A monitor whose baseline learned ``samples`` normal samples."""
    mon = ImmuneMonitor(tmp_path / "baseline.json", **kw)
    for _ in range(samples):
        out = mon.tick(sample=dict(NORMAL))
        assert out["learned"] is True
    th = mon.baseline.thresholds()
    assert th.status == "ready"
    assert th.spawned_p95 == pytest.approx(100.0)
    assert th.growth_p95 == pytest.approx(1.0)
    return mon


# ── baseline window + sample access (baseline.py) ─────────────────

def test_baseline_window_keeps_most_recent():
    base = AdaptiveBaseline(window=10)
    for i in range(25):
        base.record({"processes_spawned": float(i), "process_growth": 1.0})
    assert len(base) == 10
    kept = [s["processes_spawned"] for s in base.samples()]
    assert kept == [float(i) for i in range(15, 25)]


def test_baseline_default_window_is_unbounded():
    base = AdaptiveBaseline()
    for _ in range(50):
        base.record({"processes_spawned": 3.0, "process_growth": 1.0})
    assert len(base) == 50


def test_sample_over_thresholds_helper_matches_gate_semantics():
    base = AdaptiveBaseline()
    for _ in range(40):
        base.record(dict(NORMAL))
    th = base.thresholds()
    assert sample_over_thresholds(dict(NORMAL), th) is False  # at p95: not over
    assert sample_over_thresholds(
        {"processes_spawned": 101.0, "process_growth": 1.0}, th) is False
    assert sample_over_thresholds(dict(RUNAWAY), th) is True


# ── 1. persistence ────────────────────────────────────────────────

def test_persistence_round_trip_thresholds_exact(tmp_path):
    path = tmp_path / "baseline.json"
    mon = ImmuneMonitor(path)
    for _ in range(40):
        mon.tick(sample=dict(NORMAL))
    before = mon.baseline.thresholds()
    assert before.status == "ready"

    reloaded = ImmuneMonitor(path)
    after = reloaded.baseline.thresholds()
    assert len(reloaded.baseline) == 40
    assert after.status == "ready"
    assert after.spawned_p95 == before.spawned_p95
    assert after.growth_p95 == before.growth_p95


def test_store_round_trip_via_baseline_store(tmp_path):
    store = BaselineStore(tmp_path / "s.json")
    store.save(samples=[dict(NORMAL)], quarantined=7,
               last_tick_at=111.0, last_sample_at=110.0)
    data = store.load()
    assert data["samples"] == [NORMAL]
    assert data["quarantined"] == 7
    assert data["last_tick_at"] == 111.0


def test_corrupt_store_starts_fresh_report_only(tmp_path):
    path = tmp_path / "baseline.json"
    path.write_bytes(b"{not json at all")
    mon = ImmuneMonitor(path)  # must not raise
    assert len(mon.baseline) == 0
    assert mon.baseline.thresholds().status == "insufficient"
    # and the monitor still works: a tick rewrites a valid store
    out = mon.tick(sample=dict(NORMAL))
    assert out["learned"] is True
    data = json.loads(path.read_text())
    assert len(data["samples"]) == 1


@pytest.mark.parametrize("blob", ['[1,2,3]', '"text"', '{"samples": 42}'])
def test_wrong_shape_store_starts_fresh(tmp_path, blob):
    path = tmp_path / "baseline.json"
    path.write_text(blob)
    mon = ImmuneMonitor(path)
    assert len(mon.baseline) == 0
    assert mon.baseline.thresholds().status == "insufficient"


def test_missing_store_not_created_until_first_write(tmp_path):
    path = tmp_path / "sub" / "baseline.json"
    mon = ImmuneMonitor(path)
    assert len(mon.baseline) == 0
    assert not path.exists()  # load alone never writes
    mon.tick(sample=dict(NORMAL))
    assert path.exists()


# ── 2. sampler tick ───────────────────────────────────────────────

def test_sampler_tick_records_into_persisted_store(tmp_path):
    path = tmp_path / "baseline.json"
    mon = ImmuneMonitor(path)
    out = mon.tick(sample=dict(NORMAL))
    assert out["ok"] is True and out["sampled"] is True
    assert out["learned"] is True
    on_disk = json.loads(path.read_text())
    assert len(on_disk["samples"]) == 1
    assert on_disk["samples"][0]["processes_spawned"] == 100.0
    fresh = ImmuneMonitor(path)
    assert len(fresh.baseline) == 1


def test_sampler_tick_without_live_sample_is_honest(tmp_path):
    mon = ImmuneMonitor(tmp_path / "b.json", collect=lambda baseline: None)
    out = mon.tick()
    assert out["sampled"] is False
    assert out["learned"] is False
    assert len(mon.baseline) == 0
    # the sampler DID tick (status can tell "running, no sample here"
    # from "never ran")
    assert mon.status()["last_tick_at"] is not None


# ── 3. incident quarantine (anti-poisoning) ───────────────────────

def test_fever_quarantines_samples_p95_unmoved(tmp_path):
    mon = _ready_monitor(tmp_path)
    mon.fever.start("test fever", 600, now=1000.0)
    for _ in range(10):
        out = mon.tick(sample=dict(RUNAWAY), now=1100.0)
        assert out["learned"] is False
        assert out["incident"] is True
    assert mon.status()["quarantined"] == 10
    th = mon.baseline.thresholds()
    assert th.spawned_p95 == pytest.approx(100.0)  # unmoved
    assert th.growth_p95 == pytest.approx(1.0)
    assert len(mon.baseline) == 40  # nothing learned during the fever
    mon.fever.end(now=1200.0)
    out = mon.tick(sample=dict(NORMAL), now=1300.0)
    assert out["learned"] is True
    assert len(mon.baseline) == 41


def test_gate_enforcement_opens_incident_and_blocks_learning(tmp_path):
    enforced: list = []
    mon = _ready_monitor(tmp_path, enforce_fn=enforced.append)
    # The first runaway tick goes through the gate: it enforces, opens
    # an incident (fever), and the sample itself is quarantined.
    out = mon.tick(sample=dict(RUNAWAY))
    assert out["enforced"] is True
    assert out["learned"] is False
    assert len(enforced) == 1
    assert mon.status()["gate_incident"] is True
    assert mon.status()["fever_active"] is True
    # Pass #28's poison demo: hundreds of runaway samples recorded
    # during the incident must NOT teach the baseline runaway=normal.
    for _ in range(50):
        out = mon.tick(sample=dict(RUNAWAY))
        assert out["learned"] is False
    th = mon.baseline.thresholds()
    assert th.spawned_p95 == pytest.approx(100.0)
    assert th.growth_p95 == pytest.approx(1.0)
    assert len(enforced) == 1  # consulted once; the rest quarantined
    assert mon.status()["quarantined"] == 51
    # The runaway stops: a below-threshold tick closes the incident
    # (that sample is still quarantined — it was collected mid-incident)
    out = mon.tick(sample=dict(NORMAL))
    assert out["learned"] is False
    assert mon.status()["gate_incident"] is False
    assert mon.status()["fever_active"] is False
    # …and the next normal tick learns again.
    out = mon.tick(sample=dict(NORMAL))
    assert out["learned"] is True
    assert len(mon.baseline) == 41


def test_quarantine_count_survives_restart(tmp_path):
    path = tmp_path / "baseline.json"
    mon = _ready_monitor(tmp_path)
    mon.fever.start("test fever", 600, now=1000.0)
    mon.tick(sample=dict(RUNAWAY), now=1100.0)
    mon.tick(sample=dict(RUNAWAY), now=1100.0)
    reloaded = ImmuneMonitor(path)
    assert reloaded.status()["quarantined"] == 2


# ── 4. bounded recency window through the monitor ─────────────────

def test_monitor_window_bound_enforced_and_persisted(tmp_path):
    path = tmp_path / "baseline.json"
    mon = ImmuneMonitor(path, window=10)
    for i in range(25):
        mon.baseline.record(
            {"processes_spawned": float(i), "process_growth": 1.0})
        mon.persist()
    assert len(mon.baseline) == 10
    on_disk = json.loads(path.read_text())
    assert len(on_disk["samples"]) == 10
    reloaded = ImmuneMonitor(path, window=10)
    assert len(reloaded.baseline) == 10
    kept = [s["processes_spawned"] for s in reloaded.baseline.samples()]
    assert kept == [float(i) for i in range(15, 25)]


def test_default_window_is_the_documented_bound():
    assert BASELINE_WINDOW == 5000


# ── 5. gate through the wired path ────────────────────────────────

def test_consult_report_only_below_min_samples(tmp_path):
    enforced: list = []
    mon = ImmuneMonitor(tmp_path / "b.json", enforce_fn=enforced.append)
    for _ in range(29):
        mon.tick(sample=dict(NORMAL))
    out = mon.consult(dict(RUNAWAY), is_system_process=True)
    assert out["action"] == "report_only"
    assert out["details"]["baseline_samples"] == 29
    assert enforced == []


def test_consult_enforces_over_threshold_with_real_enforcer(tmp_path):
    enforced: list = []
    mon = _ready_monitor(tmp_path, enforce_fn=enforced.append)
    out = mon.consult(dict(RUNAWAY), is_system_process=True)
    assert out["action"] == "enforce"
    assert len(enforced) == 1
    assert mon.status()["gate_incident"] is True


def test_consult_owner_directed_never_enforces(tmp_path):
    enforced: list = []
    mon = _ready_monitor(tmp_path, enforce_fn=enforced.append)
    out = mon.consult(dict(RUNAWAY), is_system_process=True,
                      owner_directed=True)
    assert out["action"] == "report_only"
    assert enforced == []


def test_shared_monitor_singleton_per_path(tmp_path):
    a = shared_monitor(tmp_path / "x.json")
    b = shared_monitor(tmp_path / "x.json")
    c = shared_monitor(tmp_path / "y.json")
    assert a is b
    assert a is not c


def test_default_store_path_env_override(tmp_path, monkeypatch):
    target = tmp_path / "custom.json"
    monkeypatch.setenv("NM_IMMUNE_BASELINE_STORE", str(target))
    assert default_store_path() == target
    assert default_store_path(SimpleNamespace(settings=None)) == target


# ── scheduler registration ────────────────────────────────────────

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


def test_ensure_sampler_job_idempotent(tmp_path):
    import json as _json

    from nomorals.agents.scheduler import Scheduler

    ctx = _scheduler_ctx(tmp_path)
    sched = Scheduler(ctx)
    first = ensure_immune_sampler_job(sched)
    second = ensure_immune_sampler_job(sched)
    assert first is not None and second is not None
    jobs = [j for j in sched.list_jobs() if j.get("name") == IMMUNE_SAMPLER_JOB]
    assert len(jobs) == 1
    row = ctx.db.query_one(
        "SELECT payload FROM schedule_jobs WHERE id = ?", (jobs[0]["id"],))
    assert _json.loads(row["payload"]).get("tool") == IMMUNE_SAMPLER_TOOL


def test_ensure_sampler_job_replaces_stale_spec(tmp_path):
    from nomorals.agents.scheduler import Scheduler

    sched = Scheduler(_scheduler_ctx(tmp_path))
    sched.add(IMMUNE_SAMPLER_JOB, "every 6h", "tool",
              {"tool": IMMUNE_SAMPLER_TOOL, "args": {}})
    ensure_immune_sampler_job(sched)
    jobs = [j for j in sched.list_jobs() if j.get("name") == IMMUNE_SAMPLER_JOB]
    assert len(jobs) == 1
    # the stale 6h job was replaced; a further ensure at the correct
    # spec is a no-op (same job id — no remove/re-add churn)
    kept_id = jobs[0]["id"]
    ensure_immune_sampler_job(sched)
    jobs = [j for j in sched.list_jobs() if j.get("name") == IMMUNE_SAMPLER_JOB]
    assert len(jobs) == 1
    assert jobs[0]["id"] == kept_id


def test_scheduler_fires_sampler_tool_end_to_end(tmp_path):
    """The real scheduler fires the real ``__immune_sampler__`` tool;
    a real /proc sample lands in the persisted baseline store."""
    if collect_process_sample() is None:
        pytest.skip("no /proc on this platform — no live sample exists")
    from nomorals.agents.scheduler import Scheduler
    from nomorals.tools import autonomy as autonomy_tools

    ctx = _scheduler_ctx(tmp_path)
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
    assert IMMUNE_SAMPLER_TOOL in captured

    def _call(tool, capabilities=None, **kw):
        return SimpleNamespace(ok=True, value=captured[tool](), error=None)

    ctx.tools = SimpleNamespace(call=_call)
    sched = Scheduler(ctx)
    ensure_immune_sampler_job(sched)
    job = [j for j in sched.list_jobs()
           if j.get("name") == IMMUNE_SAMPLER_JOB][0]
    ctx.db.execute(
        "UPDATE schedule_jobs SET next_run = ? WHERE id = ?",
        (time.time() - 1, job["id"]))
    outcomes = sched.tick()
    assert any(o.get("id") == job["id"] for o in outcomes)
    store_file = tmp_path / "data" / "immune-baseline.json"
    assert store_file.exists()
    data = json.loads(store_file.read_text())
    assert len(data["samples"]) == 1
    assert data["samples"][0]["processes_spawned"] > 0


# ── 6. honest status + the chat gate verb ─────────────────────────

def _chat_ctx(tmp_path):
    return SimpleNamespace(
        settings=SimpleNamespace(resolve=lambda rel: tmp_path / rel))


def test_status_reflects_live_counts(tmp_path):
    reset_immune_state("wiring-status")
    try:
        ctx = _chat_ctx(tmp_path)
        mon = shared_monitor(default_store_path(ctx))
        for _ in range(35):
            mon.tick(sample=dict(NORMAL))
        out = control_immune("status", context=ctx, chat_key="wiring-status")
        assert "35" in out
        assert "READY" in out
        assert "spawned p95 100" in out
        assert "quarantined: 0" in out
        # quarantine three samples under a fever; status must show it
        mon.fever.start("status fever", 600, now=1000.0)
        for _ in range(3):
            mon.tick(sample=dict(RUNAWAY), now=1100.0)
        out = control_immune("status", context=ctx, chat_key="wiring-status")
        assert "quarantined: 3" in out
    finally:
        reset_immune_state("wiring-status")


def test_status_report_only_before_min_samples(tmp_path):
    reset_immune_state("wiring-thin")
    try:
        ctx = _chat_ctx(tmp_path)
        out = control_immune("status", context=ctx, chat_key="wiring-thin")
        assert "0/30 samples" in out
        assert "report-only" in out
    finally:
        reset_immune_state("wiring-thin")


def test_gate_verb_probe_and_live(tmp_path):
    reset_immune_state("wiring-gate")
    try:
        ctx = _chat_ctx(tmp_path)
        mon = shared_monitor(default_store_path(ctx))
        for _ in range(40):
            mon.tick(sample=dict(NORMAL))
        # probe with numbers: over both p95s -> enforce, NOT learned
        out = control_immune("gate 5000 50", context=ctx,
                             chat_key="wiring-gate")
        assert "ENFORCE" in out
        assert "not learned" in out
        assert len(mon.baseline) == 40
        # probe below thresholds -> report-only
        out = control_immune("gate 5 1", context=ctx, chat_key="wiring-gate")
        assert "report-only" in out
        # usage on garbage
        out = control_immune("gate banana", context=ctx,
                             chat_key="wiring-gate")
        assert "usage:" in out
    finally:
        reset_immune_state("wiring-gate")
