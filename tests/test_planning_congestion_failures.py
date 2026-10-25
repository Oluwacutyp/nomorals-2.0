"""Pass #164 pins: the planning failure ledger (REUSED from #156/#159) + congestion conversions.

``nomorals/planning/failures.py`` (namespace ``planning``) already exists
from #156 and was reused by #159; this pass reuses it again with site
names ``congestion.*`` — no new ledger file. The 17 durability tests
mirror the #141+ shape exactly; the 21 site pins prove each conversion
records without changing flow (fallback value unchanged AND a ledger
entry under its site name); the 1 exclusion pin proves the ALTER
idempotency path runs AND records nothing. No pin hardcodes scan
totals or congestion counts. Every test redirects stores with the app's
own ``NM_HOME`` override (never the real home, never the repo tree).
Restart tests use real subprocesses. Every trigger was probed live
before its pin was written.
"""
import json
import os
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.planning import congestion as cg
from nomorals.planning import failures as pf
from nomorals.core import failure_store as fs


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    pf.clear_failures()
    yield
    pf.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


def _sites():
    return [f.site for f in pf.recent_failures()]


def _boom(*args, **kwargs):
    raise RuntimeError("boom")


# ---- Store-level (durability mirror of #141+) ----------------------------

def test_store_round_trip():
    store = fs.Store("planning")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("planning").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("planning")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("planning").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("planning")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("planning").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("planning").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("planning").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("planning")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("planning").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("planning").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("planning").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("os.path.expanduser", lambda p: (_ for _ in ()).throw(OSError("no home")))
        before = fs.write_failures()
        assert fs.Store("planning").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("planning")
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
    entries, _ = fs.Store("planning").load()
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("os.path.expanduser", lambda p: (_ for _ in ()).throw(OSError("no home")))
        before = fs.write_failures()
        pf.record_failure("no-home", ValueError("boom"))
        assert pf.failure_count() == 1
        assert [f.site for f in pf.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_ledger_api_record_recent_count_clear_format():
    assert pf.failure_count() == 0
    assert pf.recent_failures() == []
    assert pf.format_failures() == "no planning failures recorded."
    pf.record_failure("planning.site-a", ValueError("boom"))
    pf.record_failure("planning.site-b", RuntimeError("bang"))
    assert pf.failure_count() == 2
    assert [f.site for f in pf.recent_failures()] == ["planning.site-b", "planning.site-a"]
    assert [f.site for f in pf.recent_failures(1)] == ["planning.site-b"]
    text = pf.format_failures()
    assert "2 total" in text and "planning.site-a" in text and "planning.site-b" in text
    pf.clear_failures()
    assert pf.failure_count() == 0
    assert pf.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    pf.record_failure("planning.disk", ValueError("boom"))
    entries, total = fs.Store("planning").load()
    assert total == 1 and entries[0]["site"] == "planning.disk"
    pf.clear_failures()
    assert fs.Store("planning").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    pf.record_failure("planning.mem", ValueError("boom"))
    assert pf.failure_count() == 1
    assert [f.site for f in pf.recent_failures()] == ["planning.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                pf.record_failure(f"planning.w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert pf.failure_count() == 400
    assert len(pf.recent_failures(1000)) == 200
    entries, total = pf._store.load()
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
        "from nomorals.planning import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.planning import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.planning import failures as f\n"
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
    store_path = fs.Store("planning").path()
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

class _RaisingDB:
    def execute(self, *args, **kwargs):
        raise sqlite3.DatabaseError("boom")


def _mon_stub():
    mon = cg.ContentionMonitor(in_memory=True)
    mon._db = _RaisingDB()
    return mon


def _res():
    return cg.Resource(resource_id="res_1", name="r", kind="endpoint",
                       capacity=4, alternates=[], created_at=0.0)


def _mon_res():
    mon = _mon_stub()
    mon._resolve = lambda name: _res()
    return mon


def test_pin_profile_kind(monkeypatch):
    monkeypatch.delenv("NM_PROFILE", raising=False)
    assert cg._profile_kind() == "workstation"
    assert _sites() == ["congestion.profile_kind"]


def test_pin_resolve():
    assert _mon_stub()._resolve("x") is None
    assert _sites() == ["congestion.resolve"]


def test_pin_list_resources():
    assert _mon_stub().list_resources() == []
    assert _sites() == ["congestion.list_resources"]


def test_pin_register_alternate(monkeypatch):
    mon = _mon_stub()
    monkeypatch.setattr(mon, "_resolve", lambda name: _res())
    assert mon.register_alternate("a", "b") is False
    assert _sites() == ["congestion.register_alternate"]


def test_pin_release(monkeypatch):
    mon = _mon_stub()
    monkeypatch.setattr(mon, "_resolve", lambda name: _res())
    assert mon.release("a", "b") is False
    assert _sites() == ["congestion.release"]


def test_pin_queue_depth(monkeypatch):
    mon = _mon_stub()
    monkeypatch.setattr(mon, "_resolve", lambda name: _res())
    assert mon.queue_depth("a") == 0
    assert _sites() == ["congestion.queue_depth"]


def test_pin_oldest_hold_age(monkeypatch):
    mon = _mon_stub()
    monkeypatch.setattr(mon, "_resolve", lambda name: _res())
    assert mon.oldest_hold_age("a") is None
    assert _sites() == ["congestion.oldest_hold_age"]


def test_pin_wait_estimate(monkeypatch):
    mon = _mon_stub()
    monkeypatch.setattr(mon, "_resolve", lambda name: _res())
    monkeypatch.setattr(mon, "queue_depth", lambda *a, **k: 10)
    monkeypatch.setattr(mon, "_rates", _boom)
    assert mon.wait_estimate("a") is None
    assert _sites() == ["congestion.wait_estimate"]


def test_pin_status(monkeypatch):
    mon = _mon_stub()
    monkeypatch.setattr(mon, "list_resources", _boom)
    assert mon.status() == []
    assert _sites() == ["congestion.status"]


def test_pin_breaker_row():
    assert _mon_stub()._breaker_row("res_1") == {
        "state": "closed", "window": [], "consec_fail": 0,
        "probes_ok": 0, "opened_at": 0.0}
    assert _sites() == ["congestion.breaker_row"]


def test_pin_breaker_save():
    row = {"state": "closed", "window": [], "consec_fail": 0,
           "probes_ok": 0, "opened_at": 0.0}
    assert _mon_stub()._breaker_save("res_1", row) is None
    assert _sites() == ["congestion.breaker_save"]


def test_pin_breaker_state(monkeypatch):
    mon = _mon_stub()
    monkeypatch.setattr(mon, "_resolve", lambda name: _res())
    monkeypatch.setattr(mon, "_breaker_row", _boom)
    assert mon.breaker_state("a") == "closed"
    assert _sites() == ["congestion.breaker_state"]


def test_pin_breaker_observe(monkeypatch):
    mon = _mon_stub()
    monkeypatch.setattr(mon, "_breaker_row", _boom)
    assert mon._breaker_observe("res_1", True) == "closed"
    assert _sites() == ["congestion.breaker_observe"]


def test_pin_rates():
    assert _mon_stub()._rates("res_1") == (0.0, 0.0)
    assert _sites() == ["congestion.rates"]


class _RaisingAcquireMonitor:
    def acquire(self, *args, **kwargs):
        raise RuntimeError("boom")

    def release(self, *args, **kwargs):
        pass


def test_pin_hold_acquire():
    with cg.hold(_RaisingAcquireMonitor(), "r", "a") as h:
        assert h is None
    assert _sites() == ["congestion.hold_acquire"]


class _RaisingReleaseMonitor:
    def acquire(self, *args, **kwargs):
        return cg.Hold(hold_id="hold_1", resource_id="res_1",
                       agent="a", acquired_at=0.0)

    def release(self, *args, **kwargs):
        raise RuntimeError("boom")


def test_pin_hold_release():
    with cg.hold(_RaisingReleaseMonitor(), "r", "a") as h:
        assert h is not None
    assert _sites() == ["congestion.hold_release"]


class _RaisingAdviseMonitor:
    def advise(self, *args, **kwargs):
        raise RuntimeError("boom")


def test_pin_route_around_resource():
    out = cg.route_around(_RaisingAdviseMonitor(), ["r"])
    assert out["r"]["action"] == "proceed"
    assert _sites() == ["congestion.route_around_resource"]


def test_pin_route_around():
    assert cg.route_around(object(), 42) == {}
    assert _sites() == ["congestion.route_around"]


def test_pin_pre_fanout_check():
    assert cg.pre_fanout_check(object(), 42) == {}
    assert _sites() == ["congestion.pre_fanout_check"]


class _RaisingStatusMonitor:
    def status(self, *args, **kwargs):
        raise RuntimeError("boom")


def test_pin_format_status():
    assert cg.format_status(_RaisingStatusMonitor()) == "🚦 monitor unavailable."
    assert _sites() == ["congestion.format_status"]


class _EmptyAdviseMonitor:
    def advise(self, *args, **kwargs):
        return {}


def test_pin_control_congestion():
    out = cg.control_congestion("advise foo", monitor=_EmptyAdviseMonitor())
    assert out == "🚦 congestion check failed safely — proceed with caution."
    assert _sites() == ["congestion.control_congestion"]


# ---- Exclusion pin (NOT-recording) ---------------------------------------

def test_excl_init_alter_idempotent(tmp_path):
    db = tmp_path / "congestion.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE congestion_holds (hold_id TEXT PRIMARY KEY, "
        "resource_id TEXT, agent TEXT, acquired_at REAL)")
    conn.commit()
    conn.close()
    mon = cg.ContentionMonitor(db_path=str(db))
    assert mon._db is not None
    cols = [r[1] for r in sqlite3.connect(str(db)).execute(
        "PRAGMA table_info(congestion_holds)").fetchall()]
    assert "expires_at" in cols
    assert pf.failure_count() == 0
