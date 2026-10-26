"""Pass #209 pins: research_loop convert records, exclusions stay silent.

``nomorals/agents/research_loop.py`` converts 1 of its 12 true sites
(the stale-job remove in `ensure_research_job`) to the REUSED agents
ledger, and excludes 11 (best-effort KV/default fallbacks, per-tick
retries, documented no-data outcomes, display-only status fields,
bookkeeping snapshots, and the already-source-traced scheduler list
read). The 17 durability tests below mirror the #141+ shape exactly
(same as #208's, against the REUSED agents ledger); the 1 behavior
pin proves the convert records while keeping its fallback; the 11
NOT-recording pins prove each exclusion stays silent with its
designed output. No pin hardcodes scan totals or research-loop
counts. Every test redirects stores with the app's own ``NM_HOME``
override (never the real home, never the repo tree). Restart tests
use real subprocesses. Every trigger was probed live before its pin
was written.

Pins rule (#169): no bare-swallow handler anywhere in this file — the
scanner counts test files too. Where a pin must swallow, it binds
``as exc`` + ``assert isinstance(exc, Exception)``. Docstrings avoid
the literal bare-swallow phrase (#173 lesson).
"""
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs
from nomorals.agents import failures as af
from nomorals.agents import research_loop as rl


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    af.clear_failures()
    yield
    af.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


def _assert_silent():
    assert af.failure_count() == 0
    assert af.recent_failures() == []


# ---- Store-level (durability mirror of #141+) ----------------------------

def test_store_round_trip():
    store = fs.Store("agents")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("agents").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("agents")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("agents").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("agents")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("agents").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("agents").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("agents").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("agents")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("agents").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("agents").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("agents").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("agents").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("agents")
    errors = []

    def worker(k):
        try:
            store.save(_entries(50, f"w{k}-"), k)
        except Exception as exc:  # noqa: BLE001 — test must surface it
            assert isinstance(exc, Exception)
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    entries, _ = fs.Store("agents").load()
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        af.record_failure("no-home", ValueError("boom"))
        assert af.failure_count() == 1
        assert [f.site for f in af.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_ledger_api_record_recent_count_clear_format():
    assert af.failure_count() == 0
    assert af.recent_failures() == []
    assert af.format_failures() == "no agents failures recorded."
    af.record_failure("agents.site-a", ValueError("boom"))
    af.record_failure("agents.site-b", RuntimeError("bang"))
    assert af.failure_count() == 2
    assert [f.site for f in af.recent_failures()] == ["agents.site-b", "agents.site-a"]
    assert [f.site for f in af.recent_failures(1)] == ["agents.site-b"]
    text = af.format_failures()
    assert "2 total" in text and "agents.site-a" in text and "agents.site-b" in text
    af.clear_failures()
    assert af.failure_count() == 0
    assert af.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    af.record_failure("agents.disk", ValueError("boom"))
    entries, total = fs.Store("agents").load()
    assert total == 1 and entries[0]["site"] == "agents.disk"
    af.clear_failures()
    assert fs.Store("agents").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    af.record_failure("agents.mem", ValueError("boom"))
    assert af.failure_count() == 1
    assert [f.site for f in af.recent_failures()] == ["agents.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                af.record_failure(f"agents.w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
            assert isinstance(exc, Exception)
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert af.failure_count() == 400
    assert len(af.recent_failures(1000)) == 200
    entries, total = af._store.load()
    assert len(entries) == 200
    assert total == 400


# ---- Restart (subprocess) ------------------------------------------------

def _run_subprocess(home: Path, code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["NM_HOME"] = str(home)
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, "-c", code], cwd=str(ROOT), env=env,
        capture_output=True, text=True, timeout=120)


def _record_code(n: int, clear_first: bool = True) -> str:
    clear = "f.clear_failures()\n" if clear_first else ""
    return (
        "from nomorals.agents import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.agents import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.agents import failures as f\n"
               "f.clear_failures()\nprint('ok')\n")


def test_restart_record_survives(tmp_path):
    home = tmp_path / "home"
    proc = _run_subprocess(home, _record_code(3))
    assert proc.returncode == 0, proc.stderr
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 3
    assert data["sites"] == ["site2", "site1", "site0"]


def test_restart_clear_is_gone(tmp_path):
    home = tmp_path / "home"
    assert _run_subprocess(home, _record_code(2)).returncode == 0
    assert _run_subprocess(home, _CLEAR_CODE).returncode == 0
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 0
    assert data["sites"] == []


def test_restart_total_preserved_and_cap_holds(tmp_path):
    home = tmp_path / "home"
    assert _run_subprocess(home, _record_code(250)).returncode == 0
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 250
    assert len(data["sites"]) == 200
    assert data["sites"][0] == "site249"


def test_restart_corrupt_store_recovers(tmp_path):
    home = tmp_path / "home"
    assert _run_subprocess(home, _record_code(2)).returncode == 0
    store_path = fs.Store("agents").path()
    store_path = home / "failures" / store_path.name
    assert store_path.exists()
    store_path.write_text("{not json", encoding="utf-8")
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 0
    assert data["sites"] == []


# ---- Behavior pin (the one convert) --------------------------------------

class _FakeSched:
    """Scheduler double: list/add succeed or fail per test."""

    def __init__(self, context, *, jobs=None, list_raises=False,
                 remove_raises=False):
        self.jobs = jobs if jobs is not None else []
        self.list_raises = list_raises
        self.remove_raises = remove_raises
        self.removed = []
        self.added = []

    def list_jobs(self):
        if self.list_raises:
            raise RuntimeError("scheduler table gone")
        return list(self.jobs)

    def remove(self, ref):
        if self.remove_raises:
            raise RuntimeError("db locked during remove")
        self.removed.append(ref)
        return True

    def add(self, *a, **k):
        self.added.append(a)
        return {"id": "new-1"}


def _ctx(db=None):
    return SimpleNamespace(db=db, settings=None)


def test_behavior_742_stale_remove_failure_records_and_still_schedules():
    sched = _FakeSched(None, jobs=[{"id": "old-1",
                                    "name": rl.RESEARCH_LOOP_JOB,
                                    "spec": "every 99999s"}],
                       remove_raises=True)
    with patch("nomorals.agents.scheduler.Scheduler",
               lambda context: sched):
        out = rl.ensure_research_job(_ctx())
    assert out["scheduled"] is True
    assert out["job_id"] == "new-1"
    assert len(sched.added) == 1  # fallback unchanged: new job added
    assert [f.site for f in af.recent_failures()] == \
        ["research_loop.ensure_research_job"]
    assert af.recent_failures()[0].exc_type == "RuntimeError"


# ---- NOT-recording pins (one per exclusion) ------------------------------

class _BoomDB:
    def query(self, *a, **k):
        raise RuntimeError("db down")

    def query_one(self, *a, **k):
        raise RuntimeError("db down")

    def execute(self, *a, **k):
        raise RuntimeError("db down")


def test_not_recording_207_kv_failure_falls_back_to_defaults():
    assert rl.owner_topics(_ctx(_BoomDB())) == list(rl._DEFAULT_TOPICS)
    _assert_silent()


def test_not_recording_253_kg_query_failure_returns_empty():
    assert rl._contradicted_topics(_ctx(_BoomDB())) == []
    _assert_silent()


def test_not_recording_280_recent_query_failure_returns_empty_set():
    assert rl._recent_topics(_ctx(_BoomDB())) == set()
    _assert_silent()


def test_not_recording_360_proposals_query_failure_returns_empty():
    assert rl._recent_research_proposals(_ctx(_BoomDB()), 7) == []
    _assert_silent()


def test_not_recording_420_signals_query_failure_returns_zeros():
    sig = rl.usefulness_signals(_ctx(_BoomDB()))
    assert sig == {"proposed": 0, "approved": 0, "denied": 0, "ignored": 0,
                   "pending": 0, "applied": 0, "tests_passed": 0,
                   "tests_failed": 0}
    _assert_silent()


def test_not_recording_509_last_run_query_failure_returns_none():
    assert rl.last_run(_ctx(_BoomDB())) is None
    _assert_silent()


def test_not_recording_734_list_failure_still_schedules():
    sched = _FakeSched(None, list_raises=True)
    with patch("nomorals.agents.scheduler.Scheduler",
               lambda context: sched):
        out = rl.ensure_research_job(_ctx())
    assert out["scheduled"] is True
    assert len(sched.added) == 1
    _assert_silent()


def test_not_recording_773_quiet_hours_failure_keeps_status_false():
    from nomorals.agents.scheduler import Scheduler
    from nomorals.agents.upgrade_queue import UpgradeQueue

    with patch.object(rl, "_in_quiet_hours_now",
                      side_effect=RuntimeError("clock down")), \
            patch.object(Scheduler, "list_jobs", return_value=[]), \
            patch.object(UpgradeQueue, "list", return_value=[]):
        out = rl.status(_ctx())
    assert out["gates"]["quiet_hours"] is False
    _assert_silent()


def test_not_recording_787_job_list_failure_keeps_status_unscheduled():
    from nomorals.agents.scheduler import Scheduler
    from nomorals.agents.upgrade_queue import UpgradeQueue

    with patch.object(Scheduler, "list_jobs",
                      side_effect=RuntimeError("scheduler down")), \
            patch.object(UpgradeQueue, "list", return_value=[]):
        out = rl.status(_ctx())
    assert out["job"]["scheduled"] is False
    _assert_silent()


def test_not_recording_793_pending_list_failure_keeps_zero():
    from nomorals.agents.scheduler import Scheduler
    from nomorals.agents.upgrade_queue import UpgradeQueue

    with patch.object(Scheduler, "list_jobs", return_value=[]), \
            patch.object(UpgradeQueue, "list",
                         side_effect=RuntimeError("queue down")):
        out = rl.status(_ctx())
    assert out["pending_proposals"] == 0
    _assert_silent()


def test_not_recording_833_signals_snapshot_failure_keeps_empty():
    record = {"id": "rlrun-test", "started_at": 0.0,
              "ok": True, "skipped_reason": "", "topics": ["q"],
              "findings_count": 0, "claims_count": 0,
              "proposals_created": 0, "proposal_ids": [],
              "proposals_deduped": 0, "proposals_capped": 0,
              "signals": {}, "notified": False, "error": ""}
    with patch.object(rl, "loop_gates", return_value=(True, "")), \
            patch.object(rl.ResearchLoop, "_run_cycle",
                         return_value=dict(record)), \
            patch.object(rl, "usefulness_signals",
                         side_effect=RuntimeError("signals down")):
        out = rl.run_topic(_ctx(), "q")
    assert out["ok"] is True
    assert out["signals"] == {}
    _assert_silent()
