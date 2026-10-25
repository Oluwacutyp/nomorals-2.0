"""Pass #197 pins: the core failure ledger (NEW in this pass) + error_system.py
conversions.

The core ledger did not exist on this base, so this pass creates
``nomorals/core/failures.py`` (namespace ``core``, #141+ shape via the
stdlib-only failure store). The 17 durability tests below mirror the
#141+ shape for this ledger; the 2 site pins prove each conversion
records without changing flow (fallback behavior unchanged AND a ledger
entry under its site name); the 12 NOT-recording pins prove each
exclusion stays silent on its failure path. No pin hardcodes scan
totals or error_system.py counts. Every test redirects stores with the
app's own ``NM_HOME`` override (never the real home, never the repo
tree). Restart tests use real subprocesses. Every trigger was probed
live before its pin was written.
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

from nomorals.core import error_system as es
from nomorals.core import failures as cf
from nomorals.core import failure_store as fs


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    cf.clear_failures()
    yield
    cf.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


def _sites():
    return [f.site for f in cf.recent_failures()]


def _boom(*args, **kwargs):
    raise RuntimeError("boom")


# ---- Store-level (durability mirror of #141+) ----------------------------

def test_store_round_trip():
    store = fs.Store("core")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("core").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("core")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("core").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("core")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("core").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("core").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("core").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("core")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("core").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("core").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("core").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("core").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("core")
    errors = []

    def worker(k):
        try:
            store.save(_entries(50, f"w{k}-"), k)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    entries, _ = fs.Store("core").load()
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) -------------------------------------------

def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        cf.record_failure("no-home", ValueError("boom"))
        assert cf.failure_count() == 1
        assert [f.site for f in cf.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_ledger_api_record_recent_count_clear_format():
    assert cf.failure_count() == 0
    assert cf.recent_failures() == []
    assert cf.format_failures() == "no core failures recorded."
    cf.record_failure("core.site-a", ValueError("boom"))
    cf.record_failure("core.site-b", RuntimeError("bang"))
    assert cf.failure_count() == 2
    assert [f.site for f in cf.recent_failures()] == ["core.site-b", "core.site-a"]
    assert [f.site for f in cf.recent_failures(1)] == ["core.site-b"]
    text = cf.format_failures()
    assert "2 total" in text and "core.site-a" in text and "core.site-b" in text
    cf.clear_failures()
    assert cf.failure_count() == 0
    assert cf.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    cf.record_failure("core.disk", ValueError("boom"))
    entries, total = fs.Store("core").load()
    assert total == 1 and entries[0]["site"] == "core.disk"
    cf.clear_failures()
    assert fs.Store("core").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    cf.record_failure("core.mem", ValueError("boom"))
    assert cf.failure_count() == 1
    assert [f.site for f in cf.recent_failures()] == ["core.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                cf.record_failure(f"core.w{k}-{i}", ValueError("boom"))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert cf.failure_count() == 400
    assert len(cf.recent_failures(1000)) == 200
    entries, total = cf._store.load()
    assert len(entries) == 200
    assert total == 400


def test_ledger_ring_cap_drops_oldest_silently():
    for i in range(250):
        cf.record_failure(f"core.ring{i}", ValueError("boom"))
    assert cf.failure_count() == 250
    ring = cf.recent_failures(1000)
    assert len(ring) == 200
    assert ring[0].site == "core.ring249"
    assert ring[-1].site == "core.ring50"


# ---- Restart (subprocess) -------------------------------------------------

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
        "from nomorals.core import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.core import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.core import failures as f\n"
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


def test_restart_corrupt_store_recovers(tmp_path):
    home = tmp_path / "home"
    assert _run_subprocess(home, _record_code(2)).returncode == 0
    corrupt = (
        "from pathlib import Path\n"
        "from nomorals.core import failure_store as fs\n"
        "fs.Store('core').path().write_text('{not json', encoding='utf-8')\n"
        "print('corrupted')\n")
    proc = _run_subprocess(home, corrupt)
    assert proc.returncode == 0, proc.stderr
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 0 and data["sites"] == []
    proc = _run_subprocess(home, _record_code(1))
    assert proc.returncode == 0, proc.stderr
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 1 and data["sites"] == ["site0"]


# ---- Site pins (converts record, fallback behavior unchanged) -------------

class _RaisingJournal:
    def record_heartbeat(self, *args, **kwargs):
        raise RuntimeError("journal down")

    def record_incident(self, *args, **kwargs):
        raise RuntimeError("journal down")


def test_pin_heartbeat():
    sys_obj = es.ErrorSystem(journal=_RaisingJournal(), budgets=object(),
                             ladders=object())
    with patch.object(es, "get_error_system", return_value=sys_obj):
        es.heartbeat("sched", True)  # must not raise: telemetry never breaks callers
    assert _sites() == ["error_system.heartbeat"]


def test_pin_record_error():
    sys_obj = es.ErrorSystem(journal=_RaisingJournal(), budgets=object(),
                             ladders=object())
    with patch.object(es, "get_error_system", return_value=sys_obj):
        es.record_error("sched", RuntimeError("orig"))  # must not raise
    assert _sites() == ["error_system.record_error"]


# ---- NOT-recording pins (exclusions stay silent on failure) --------------

class _RaisingBudgets:
    def check_all(self):
        raise RuntimeError("budgets down")

    def status_all(self):
        raise RuntimeError("budgets down")


class _RaisingLadders:
    def status_all(self):
        raise RuntimeError("ladders down")

    def degraded_subsystems(self):
        raise RuntimeError("ladders down")


class _RaisingJournalStats:
    def top_failing(self, **kwargs):
        raise RuntimeError("journal down")

    def top_subsystems(self, **kwargs):
        raise RuntimeError("journal down")

    def mttr(self, **kwargs):
        raise RuntimeError("journal down")


def _health_system():
    return es.ErrorSystem(journal=_RaisingJournalStats(),
                          budgets=_RaisingBudgets(),
                          ladders=_RaisingLadders())


def test_pin_no_record_health_alerts():
    snap = _health_system().health()
    assert snap["alerts"] == [] and snap["ok"] is True
    assert _sites() == []


def test_pin_no_record_health_budget_status():
    snap = _health_system().health()
    assert snap["budgets"] == {}
    assert _sites() == []


def test_pin_no_record_health_ladders():
    snap = _health_system().health()
    assert snap["ladders"] == {} and snap["degraded_subsystems"] == []
    assert _sites() == []


def test_pin_no_record_health_top_failing():
    snap = _health_system().health()
    assert snap["top_failing_24h"] == []
    assert _sites() == []


def test_pin_no_record_health_top_subsystems():
    snap = _health_system().health()
    assert snap["top_subsystems_24h"] == []
    assert _sites() == []


def test_pin_no_record_health_mttr():
    snap = _health_system().health()
    assert snap["mttr_24h"] == {}
    assert _sites() == []


class _RaisingSupervisor:
    def stop_all(self):
        raise RuntimeError("stop down")


class _RaisingJournalClose:
    def close(self):
        raise RuntimeError("close down")


def test_pin_no_record_close_supervisor():
    sys_obj = es.ErrorSystem(journal=_RaisingJournalClose(),
                             budgets=object(), ladders=object())
    sys_obj.supervisors = {"s": _RaisingSupervisor()}
    sys_obj.close()  # must not raise
    assert _sites() == []


def test_pin_no_record_close_journal():
    sys_obj = es.ErrorSystem(journal=_RaisingJournalClose(),
                             budgets=object(), ladders=object())
    sys_obj.close()  # must not raise
    assert _sites() == []


class _BoomAttr:
    def __getattr__(self, name):
        raise RuntimeError("boom-" + name)


def test_pin_no_record_llm_probe():
    assert es._llm_probe(_BoomAttr())() is False
    assert _sites() == []


class _ExtrasBoom:
    gateway = None

    @property
    def extras(self):
        raise RuntimeError("extras down")


def test_pin_no_record_resolve_gateway():
    assert es._resolve_gateway(_ExtrasBoom()) is None
    assert _sites() == []


class _BadKV:
    def keys(self, prefix):
        return ["tg_retry:1"]

    def get(self, key):
        raise RuntimeError("row down")

    def delete(self, key):
        raise RuntimeError("delete down")


class _NoOutboxGW:
    outbox = None


def _ctx_with_db():
    return type("C", (), {"db": object()})()


def test_pin_no_record_drain_row():
    moved = es.drain_tg_retry_queue(
        _ctx_with_db(), _NoOutboxGW(), kv_factory=lambda db: _BadKV())
    assert moved == 0
    assert _sites() == []


def _boom_factory(db):
    raise RuntimeError("factory down")


def test_pin_no_record_drain_outer():
    moved = es.drain_tg_retry_queue(
        _ctx_with_db(), _NoOutboxGW(), kv_factory=_boom_factory)
    assert moved == 0
    assert _sites() == []
