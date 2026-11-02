"""Tail-89 pins: nomorals/immune/wiring.py census sites.

The module wires the adaptive cancer gate into production;
its posture is fail-safe degradation — a broken settings
object or a broken scheduler must not take the sampler (or
the surfaces resolving the store path) down with it. Every
site is held as an exclusion:

179 — ``_settings_home``: ``Path(settings.home_path)`` crash
-> pass, fall through to the ``~/.nomorals`` default. The
settings object is duck-typed by design (mirrors
agents/modes); the failure value is the designed fallback
home, a real path — never a fabricated one.
204 — ``default_store_path``: ``settings.resolve(...)``
crash -> pass, fall through to ``<home>/data/...``. Same
designed fallback chain; both live consumers (immune chat,
autonomy tool) resolve a valid store path either way.
861 — ``ensure_immune_sampler_job``: legacy-job removal
crash -> pass; the new job still registers. WEAKEST
exclusion of the tail-89 batch, tension stated plainly: the
remedy the docstring promises ("retired first") silently
does not happen, so the pre-Pass-3 alerting job can keep
running alongside the silent one — resurrecting the per-run
success alerts Pass 3 removed by design. Held: no surface
claims the legacy job is gone (the scheduler's own job list
still shows it, so the state is visible, not falsified),
and the remedy — retry/reconcile semantics for durable
scheduler rows — is a scheduler design change for a
deliberate pass, not an error-handling fix.
869 — stale-job removal crash -> pass, then add: the stale
sampler job and the fresh one coexist (probe-pinned below).
Same family as 861: visible in the job list, consequence is
duplicate sampler ticks into one shared monitor, remedy is
the same scheduler-reconciliation design pass.
875 — the whole registration body crash -> None. None is
the function's documented best-effort value ("never raises";
also the scheduler-is-None value), the sole production
caller discards the return, and the resulting absence of
scheduled sampling surfaces honestly through the monitor's
own status (sampler_running / stale tick age).

NOT-recording is pinned with the snapshot method: wiring.py
calls no failures-ledger record function anywhere, so both
the agency and agents ledgers must be unchanged across each
trigger.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
from nomorals.immune import wiring as W


def _snap():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


def _raise(*args, **kwargs):
    raise RuntimeError("boom")


# ── 179: duck-typed settings -> home default ────────────────────

class _BadFs:
    def __fspath__(self):
        raise RuntimeError("no fs")


def test_179_bad_home_path_falls_back_to_home():
    before = _snap()
    assert W._settings_home(SimpleNamespace(home_path=_BadFs())) == \
        Path.home() / ".nomorals"
    # control: a healthy home_path is honoured verbatim
    assert W._settings_home(SimpleNamespace(home_path="/tmp/xyz")) == \
        Path("/tmp/xyz")
    assert _snap() == before


# ── 204: settings.resolve crash -> home-default store ───────────

def test_204_resolve_crash_falls_back_to_home_store(monkeypatch):
    monkeypatch.delenv(W.STORE_ENV_VAR, raising=False)
    before = _snap()
    ctx = SimpleNamespace(settings=SimpleNamespace(resolve=_raise))
    assert W.default_store_path(ctx) == (
        Path.home() / ".nomorals" / "data" / "immune-baseline.json")
    # control: a healthy resolve() decides the path
    good = SimpleNamespace(settings=SimpleNamespace(
        resolve=lambda p: f"/tmp/s/{p}"))
    assert W.default_store_path(good) == \
        Path("/tmp/s/data/immune-baseline.json")
    assert _snap() == before


# ── fake durable scheduler ──────────────────────────────────────

class _FakeSched:
    """list_jobs() serves queued pages first, then the live list."""

    def __init__(self, pages, remove_raises=False):
        self._pages = list(pages)
        self._remove_raises = remove_raises
        self.jobs = [j for page in pages for j in page]
        self.added = []

    def list_jobs(self):
        return self._pages.pop(0) if self._pages else list(self.jobs)

    def remove(self, job_id):
        if self._remove_raises:
            raise RuntimeError("remove boom")
        self.jobs = [j for j in self.jobs if j["id"] != job_id]

    def add(self, name, spec, kind, payload, routine=False):
        job = {"id": 99, "name": name, "spec": spec}
        self.jobs.append(job)
        self.added.append(job)
        return job


# ── 861: legacy removal crash -> new job still registers ────────

def test_861_legacy_remove_crash_still_registers():
    before = _snap()
    legacy = {"id": 1, "name": W.LEGACY_IMMUNE_SAMPLER_JOB,
              "spec": "every 1m"}
    sched = _FakeSched([[legacy], []], remove_raises=True)
    job = W.ensure_immune_sampler_job(sched)
    assert job is not None and job["name"] == W.IMMUNE_SAMPLER_JOB
    assert len(sched.added) == 1
    # the pinned consequence: the legacy job remains registered,
    # visible in the scheduler's own list — not claimed gone
    assert any(j["name"] == W.LEGACY_IMMUNE_SAMPLER_JOB
               for j in sched.jobs)
    assert _snap() == before


# ── 869: stale removal crash -> duplicate coexists ──────────────

def test_869_stale_remove_crash_adds_alongside():
    before = _snap()
    stale = {"id": 2, "name": W.IMMUNE_SAMPLER_JOB, "spec": "every 5m"}
    sched = _FakeSched([[], [stale]], remove_raises=True)
    job = W.ensure_immune_sampler_job(sched)
    assert job is not None and len(sched.added) == 1
    # the pinned consequence: two sampler jobs, one at the stale
    # cadence — visible in the job list, not hidden
    assert sum(1 for j in sched.jobs
               if j["name"] == W.IMMUNE_SAMPLER_JOB) == 2
    # control: a removable stale job is replaced, not duplicated
    sched2 = _FakeSched([[], [dict(stale)]])
    job2 = W.ensure_immune_sampler_job(sched2)
    assert job2 is not None
    assert sum(1 for j in sched2.jobs
               if j["name"] == W.IMMUNE_SAMPLER_JOB) == 1
    assert _snap() == before


# ── 875: scheduler failure -> None, never raises ────────────────

def test_875_scheduler_crash_returns_none():
    before = _snap()
    sched = _FakeSched([])
    sched.list_jobs = _raise
    assert W.ensure_immune_sampler_job(sched) is None
    # controls: no scheduler at all -> None; healthy -> the job
    assert W.ensure_immune_sampler_job(None) is None
    job = W.ensure_immune_sampler_job(_FakeSched([[], []]))
    assert job is not None and job["name"] == W.IMMUNE_SAMPLER_JOB
    assert _snap() == before
