"""Pass #168 pins: the scheduler failure ledger (NEW) + scheduler conversions.

``nomorals/scheduler/failures.py`` (namespace ``scheduler``) is new in
this pass: no scheduler-area ledger existed. The 17 durability tests
below mirror the #141+ shape exactly; the 19 site pins prove each
conversion records without changing flow (fallback value unchanged AND
a ledger entry under its site name); the 1 exclusion pin proves the
``_resolve_cron_expr`` dispatch records nothing. No pin hardcodes scan
totals or scheduler counts. Every test redirects stores with the app's
own ``NM_HOME`` override (never the real home, never the repo tree).
Restart tests use real subprocesses. Every trigger was probed live
before its pin was written.
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
from nomorals.scheduler import failures as sf


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    sf.clear_failures()
    yield
    sf.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


def _sites():
    return [f.site for f in sf.recent_failures()]


# ---- Store-level (durability mirror of #141+) ----------------------------

def test_store_round_trip():
    store = fs.Store("scheduler")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("scheduler").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("scheduler")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("scheduler").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("scheduler")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("scheduler").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("scheduler").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("scheduler").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("scheduler")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("scheduler").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("scheduler").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("scheduler").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("scheduler").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("scheduler")
    errors = []

    def worker(k):
        try:
            store.save(_entries(50, f"w{k}-"), k)
        except Exception as exc:  # noqa: BLE001 — test must surface it
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    entries, _ = fs.Store("scheduler").load()
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        sf.record_failure("no-home", ValueError("boom"))
        assert sf.failure_count() == 1
        assert [f.site for f in sf.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_ledger_api_record_recent_count_clear_format():
    assert sf.failure_count() == 0
    assert sf.recent_failures() == []
    assert sf.format_failures() == "no scheduler failures recorded."
    sf.record_failure("scheduler.site-a", ValueError("boom"))
    sf.record_failure("scheduler.site-b", RuntimeError("bang"))
    assert sf.failure_count() == 2
    assert [f.site for f in sf.recent_failures()] == ["scheduler.site-b", "scheduler.site-a"]
    assert [f.site for f in sf.recent_failures(1)] == ["scheduler.site-b"]
    text = sf.format_failures()
    assert "2 total" in text and "scheduler.site-a" in text and "scheduler.site-b" in text
    sf.clear_failures()
    assert sf.failure_count() == 0
    assert sf.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    sf.record_failure("scheduler.disk", ValueError("boom"))
    entries, total = fs.Store("scheduler").load()
    assert total == 1 and entries[0]["site"] == "scheduler.disk"
    sf.clear_failures()
    assert fs.Store("scheduler").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    sf.record_failure("scheduler.mem", ValueError("boom"))
    assert sf.failure_count() == 1
    assert [f.site for f in sf.recent_failures()] == ["scheduler.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                sf.record_failure(f"scheduler.w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert sf.failure_count() == 400
    assert len(sf.recent_failures(1000)) == 200
    entries, total = sf._store.load()
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
        "from nomorals.scheduler import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.scheduler import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.scheduler import failures as f\n"
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
    store_path = fs.Store("scheduler").path()
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


# ---- Site pins ------------------------------------------------------------

from nomorals.scheduler.scheduler import (
    Scheduler, _fmt_next, _quiet_shift, _ts_date,
)


class _Boom(Exception):
    pass


class FakeDB:
    """query/query_one raise unless the SQL contains `fail_on` (None = always)."""

    def __init__(self, fail_on=None):
        self.fail_on = fail_on

    def _maybe_raise(self, sql):
        if self.fail_on is None or self.fail_on in sql:
            raise _Boom("db down")

    def query(self, sql, params=()):
        self._maybe_raise(sql)
        return []

    def query_one(self, sql, params=()):
        self._maybe_raise(sql)
        return None

    def execute(self, sql, params=()):
        raise _Boom("db down")

    def transaction(self):
        import contextlib
        return contextlib.nullcontext()


def _sched(db):
    s = Scheduler.__new__(Scheduler)
    s.db = db
    s._resources = None
    s._running = False
    s._worker_thread = None
    s._started_at = None
    s._tick_errors = 0
    s._in_flight = set()
    s._flight_lock = threading.Lock()
    s._deferrals = {}
    s._listeners = {}
    s._listeners_lock = threading.Lock()
    s._action_handlers = {}
    s.run_history_limit = 100
    return s


def test_site_quiet_shift_zone_records_and_falls_back():
    import time
    result = _quiet_shift(time.time(), ("23:00", "07:00"), "Not/AZone")
    assert result is None or isinstance(result, float)
    assert "scheduler.quiet_shift_zone" in _sites()


def test_site_ts_date_zone_records_and_falls_back():
    import datetime
    result = _ts_date(1760000000, "Not/AZone")
    assert isinstance(result, datetime.date)
    assert "scheduler.ts_date_zone" in _sites()


def test_site_fmt_next_zone_records_and_falls_back():
    result = _fmt_next(1760000000, "Not/AZone")
    assert isinstance(result, str) and result
    assert "scheduler.fmt_next_zone" in _sites()


def test_site_catchup_cron_rows_records_and_skips():
    s = _sched(FakeDB())
    assert s.catch_up_on_startup() == []
    assert "scheduler.catchup_cron_rows" in _sites()


def test_site_catchup_reminder_rows_records_and_skips():
    s = _sched(FakeDB())
    assert s.catch_up_on_startup() == []
    assert "scheduler.catchup_reminder_rows" in _sites()


def test_site_catchup_task_rows_records_and_skips():
    s = _sched(FakeDB())
    assert s.catch_up_on_startup() == []
    assert "scheduler.catchup_task_rows" in _sites()


def test_site_run_count_records_and_returns_zero():
    s = _sched(FakeDB())
    assert s._run_count("task-x") == 0
    assert "scheduler.run_count" in _sites()


def test_site_health_tasks_records_and_keeps_default():
    s = _sched(FakeDB("GROUP BY status"))
    out = s.health()
    assert out["tasks"] == {}
    assert "scheduler.health_tasks" in _sites()


def test_site_health_due_now_records_and_keeps_default():
    import time

    class _RowDB(FakeDB):
        def query_one(self, sql, params=()):
            # next_task must return a row: the due_now query runs only
            # inside `if row:` in the same try block.
            if "LEFT JOIN cron_jobs" in sql:
                return {"task_id": "t1", "at_ts": time.time()}
            return super().query_one(sql, params)

    s = _sched(_RowDB("FROM cron_jobs WHERE next_run"))
    out = s.health()
    assert out["due_now"] == 0
    assert "scheduler.health_due_now" in _sites()


def test_site_health_upcoming_records_and_keeps_default():
    s = _sched(FakeDB("ORDER BY r.due_at ASC"))
    out = s.health()
    assert out["upcoming"] == []
    assert "scheduler.health_upcoming" in _sites()


def test_site_health_stale_records_and_keeps_default():
    s = _sched(FakeDB("c.last_run IS NOT"))
    out = s.health()
    assert out["stale"] == []
    assert "scheduler.health_stale" in _sites()


def test_site_health_dead_records_and_keeps_default():
    s = _sched(FakeDB("status = 'dead'"))
    out = s.health()
    assert out["dead"] == []
    assert "scheduler.health_dead" in _sites()


def test_site_health_recent_runs_records_and_keeps_default():
    s = _sched(FakeDB("FROM task_runs ORDER BY started_at DESC"))
    out = s.health()
    assert out["recent_runs"] == []
    assert "scheduler.health_recent_runs" in _sites()


def test_site_tick_cron_rows_records_and_skips():
    s = _sched(FakeDB())
    s._tick()  # must not raise
    assert "scheduler.tick_cron_rows" in _sites()


def test_site_tick_reminder_rows_records_and_skips():
    s = _sched(FakeDB())
    s._tick()
    assert "scheduler.tick_reminder_rows" in _sites()


def test_site_tick_task_rows_records_and_skips():
    s = _sched(FakeDB())
    s._tick()
    assert "scheduler.tick_task_rows" in _sites()


def test_site_recent_runs_records_and_returns_empty():
    s = _sched(FakeDB())
    assert s.recent_runs("task-x") == []
    assert "scheduler.recent_runs" in _sites()


def test_site_eta_estimate_records_and_returns_unavailable(monkeypatch):
    s = _sched(FakeDB())
    with patch.dict(sys.modules, {"nomorals.planning.estimates": None}):
        assert s.eta_for("cron") == "⏱️ estimate unavailable"
    assert "scheduler.eta_estimate" in _sites()


def test_site_record_outcome_records_and_returns_false(monkeypatch):
    s = _sched(FakeDB())
    with patch.dict(sys.modules, {"nomorals.planning.estimates": None}):
        assert s.record_outcome("cron", 5.0, 6.0) is False
    assert "scheduler.record_outcome" in _sites()


# ---- Exclusion pin ---------------------------------------------------------

def test_excluded_resolve_cron_expr_dispatch_records_nothing():
    assert Scheduler._resolve_cron_expr("every day at 9am") == "0 9 * * *"
    assert Scheduler._resolve_cron_expr("0 9 * * *") == "0 9 * * *"
    assert _sites() == []
    with pytest.raises(ValueError):
        Scheduler._resolve_cron_expr("blargh zzz")
