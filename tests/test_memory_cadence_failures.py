"""Pass #206 pins: cadence exclusions stay silent (0 converts).

``nomorals/memory/cadence.py`` is a "Never raises" cadence helper: every
one of its 13 true sites degrades to a documented default, a genuine
fresh-DB value, or a safe-direction fallback. This pass converts
nothing, so there are no behavior pins — the 17 durability tests below
mirror the #141+ shape exactly (same as #182's, against the REUSED
memory ledger), and the 13 NOT-recording pins prove each exclusion
stays silent with its designed output. No pin hardcodes scan totals or
cadence counts. Every test redirects stores with the app's own
``NM_HOME`` override (never the real home, never the repo tree).
Restart tests use real subprocesses. Every trigger was probed live
before its pin was written.

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
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs
from nomorals.memory import cadence
from nomorals.memory import failures as mf


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    mf.clear_failures()
    yield
    mf.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


def _assert_silent():
    assert mf.failure_count() == 0
    assert mf.recent_failures() == []


# ---- Store-level (durability mirror of #141+) ----------------------------

def test_store_round_trip():
    store = fs.Store("memory")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("memory").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("memory")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("memory").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("memory")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("memory").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("memory").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("memory").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("memory")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("memory").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("memory").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("memory").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("memory").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("memory")
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
    entries, _ = fs.Store("memory").load()
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        mf.record_failure("no-home", ValueError("boom"))
        assert mf.failure_count() == 1
        assert [f.site for f in mf.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_ledger_api_record_recent_count_clear_format():
    assert mf.failure_count() == 0
    assert mf.recent_failures() == []
    assert mf.format_failures() == "no memory failures recorded."
    mf.record_failure("memory.site-a", ValueError("boom"))
    mf.record_failure("memory.site-b", RuntimeError("bang"))
    assert mf.failure_count() == 2
    assert [f.site for f in mf.recent_failures()] == ["memory.site-b", "memory.site-a"]
    assert [f.site for f in mf.recent_failures(1)] == ["memory.site-b"]
    text = mf.format_failures()
    assert "2 total" in text and "memory.site-a" in text and "memory.site-b" in text
    mf.clear_failures()
    assert mf.failure_count() == 0
    assert mf.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    mf.record_failure("memory.disk", ValueError("boom"))
    entries, total = fs.Store("memory").load()
    assert total == 1 and entries[0]["site"] == "memory.disk"
    mf.clear_failures()
    assert fs.Store("memory").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    mf.record_failure("memory.mem", ValueError("boom"))
    assert mf.failure_count() == 1
    assert [f.site for f in mf.recent_failures()] == ["memory.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                mf.record_failure(f"memory.w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
            assert isinstance(exc, Exception)
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert mf.failure_count() == 400
    assert len(mf.recent_failures(1000)) == 200
    entries, total = mf._store.load()
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
        "from nomorals.memory import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.memory import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.memory import failures as f\n"
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
    store_path = fs.Store("memory").path()
    store_path = home / "failures" / store_path.name
    assert store_path.exists()
    store_path.write_text("{not json", encoding="utf-8")
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 0
    assert data["sites"] == []
    proc = _run_subprocess(home, _record_code(1, clear_first=False))
    assert proc.returncode == 0, proc.stderr
    proc = _run_subprocess(home, _READ_CODE)
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 1


# ---- NOT-recording pins (one per exclusion) ------------------------------

class _RaisingDB:
    """Every DB op raises: the meta layer's whole-layer-broken mode."""

    def execute(self, *a, **k):
        raise RuntimeError("db down")

    def scalar(self, *a, **k):
        raise RuntimeError("db down")

    def query(self, *a, **k):
        raise RuntimeError("db down")


class _ScriptedDB:
    """DB scripted per statement kind: scalar answers by key, execute
    succeeds except on INSERT when told to fail."""

    def __init__(self, scalar_by_key=None, fail_insert=False):
        self.scalar_by_key = scalar_by_key or {}
        self.fail_insert = fail_insert
        self.executed = []

    def execute(self, sql, *a, **k):
        self.executed.append(sql)
        if self.fail_insert and "INSERT" in sql:
            raise RuntimeError("insert down")

    def scalar(self, sql, params=(), default=""):
        if params:
            return self.scalar_by_key.get(params[0], default)
        return default

    def query(self, *a, **k):
        return []


class _Settings:
    def __init__(self, interval):
        self.memory = _MemorySettings(interval)


class _MemorySettings:
    def __init__(self, interval):
        self.consolidation_interval_seconds = interval


class _RaisingIntervalSettings:
    @property
    def consolidation_interval_seconds(self):
        raise RuntimeError("settings unreadable")


class _Manager:
    def __init__(self, db=None, interval=3600.0, report=None):
        self.db = db if db is not None else _ScriptedDB()
        self.context = type("Ctx", (), {})()
        self.context.settings = _Settings(interval)
        self._report = report if report is not None else {"ok": False}

    def consolidate_additive(self, min_episodes=5):
        return self._report


class _FakeScheduler:
    def __init__(self, jobs=None, remove_raises=False):
        self.jobs = list(jobs or [])
        self.remove_raises = remove_raises
        self.added = []
        self.removed = []

    def list_jobs(self):
        raise RuntimeError("scheduler table gone")

    def add(self, name, spec, kind, payload):
        job = {"id": "new1", "name": name, "spec": spec}
        self.added.append(job)
        return job

    def remove(self, ref):
        self.removed.append(ref)
        if self.remove_raises:
            raise RuntimeError("delete down")
        return True


class _ListingScheduler(_FakeScheduler):
    def list_jobs(self):
        return list(self.jobs)


def test_not_recording_058_ensure_meta_table_create_failure():
    cadence._ensure_meta_table(_RaisingDB())
    _assert_silent()


def test_not_recording_069_meta_get_read_failure_returns_default():
    assert cadence.meta_get(_RaisingDB(), "consolidation.runs", "0") == "0"
    _assert_silent()


def test_not_recording_081_meta_set_insert_failure_passes():
    db = _ScriptedDB(fail_insert=True)
    cadence.meta_set(db, cadence._KEY_LAST_RUN, "1234.5")
    assert any("INSERT" in sql for sql in db.executed)
    _assert_silent()


def test_not_recording_094_interval_unreadable_returns_default():
    manager = _Manager()
    manager.context.settings = type("S", (), {})()
    manager.context.settings.memory = _RaisingIntervalSettings()
    assert cadence.interval_seconds(manager) == 3600.0
    _assert_silent()


def test_not_recording_112_unparseable_metadata_counts_pending():
    class _RowsDB(_ScriptedDB):
        def query(self, *a, **k):
            return [{"metadata": "{not json"},
                    {"metadata": '{"distilled_into": "fact1"}'}]

    manager = _Manager(db=_RowsDB())
    assert cadence.undistilled_episodes(manager) == 1
    _assert_silent()


def test_not_recording_141_last_report_dumps_failure_passes():
    report = {}
    report["self"] = report  # circular: dumps raises even with default=str
    manager = _Manager(report=report)
    out = cadence.consolidate_now(manager)
    assert out["ok"] is True
    assert out["run_number"] == 1
    _assert_silent()


def test_not_recording_145_runs_value_corrupt_resets_to_zero():
    db = _ScriptedDB(scalar_by_key={cadence._KEY_RUNS: "abc"})
    manager = _Manager(db=db)
    out = cadence.consolidate_now(manager)
    assert out["run_number"] == 1
    _assert_silent()


def test_not_recording_168_last_run_corrupt_runs_now():
    db = _ScriptedDB(scalar_by_key={cadence._KEY_LAST_RUN: "garbage"})
    manager = _Manager(db=db)
    out = cadence.maybe_run(manager)
    assert out["ran"] is True
    _assert_silent()


def test_not_recording_190_status_last_run_corrupt_renders_never_ran():
    db = _ScriptedDB(scalar_by_key={cadence._KEY_LAST_RUN: "garbage"})
    manager = _Manager(db=db)
    out = cadence.status(manager)
    assert out["last_run_at"] is None
    assert out["last_run_ago_s"] is None
    assert out["next_due_in_s"] == 0.0
    _assert_silent()


def test_not_recording_199_status_runs_corrupt_renders_zero():
    db = _ScriptedDB(scalar_by_key={cadence._KEY_RUNS: "garbage"})
    manager = _Manager(db=db)
    out = cadence.status(manager)
    assert out["runs"] == 0
    _assert_silent()


def test_not_recording_204_status_report_corrupt_renders_empty():
    db = _ScriptedDB(scalar_by_key={cadence._KEY_LAST_REPORT: '{"broken'})
    manager = _Manager(db=db)
    out = cadence.status(manager)
    assert out["last_report"] == {}
    _assert_silent()


def test_not_recording_247_list_jobs_failure_adds_job():
    context = type("Ctx", (), {})()
    context.memory = _Manager()
    sched = _FakeScheduler()
    out = cadence.ensure_consolidation_job(
        context, scheduler_factory=lambda ctx: sched)
    assert out["scheduled"] is True
    assert out["job_id"] == "new1"
    assert len(sched.added) == 1
    _assert_silent()


def test_not_recording_256_remove_failure_keeps_stale_adds_new():
    context = type("Ctx", (), {})()
    context.memory = _Manager()
    stale = {"name": cadence.CONSOLIDATION_JOB_NAME,
             "spec": "every 60s", "id": "stale1"}
    sched = _ListingScheduler(jobs=[stale], remove_raises=True)
    out = cadence.ensure_consolidation_job(
        context, scheduler_factory=lambda ctx: sched)
    assert out["scheduled"] is True
    assert out["job_id"] == "new1"
    assert sched.removed == ["stale1"]
    assert len(sched.added) == 1
    _assert_silent()
