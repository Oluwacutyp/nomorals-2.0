"""Pass #175 pins: the autonomy failure ledger (NEW) + weakness conversions.

``nomorals/autonomy/failures.py`` (namespace ``autonomy``) is new: no
autonomy-area ledger existed in the track, and weakness.py is an
autonomy-subsystem module, so a new ledger is correct (unlike #161's
engine/trivia split, and unlike reusing another subsystem's ledger).
The 17 durability tests below mirror the #141+ shape exactly; the 18
site pins prove each conversion records without changing flow (fallback
value unchanged AND a ledger entry under its site name). No exclusions
exist in this pass, so there are no NOT-recording pins. No pin hardcodes
scan totals or weakness counts. Every test redirects stores with the
app's own ``NM_HOME`` override (never the real home, never the repo
tree). Restart tests use real subprocesses. Every trigger was probed
live before its pin was written (probe.py).

Pins rule (#169): no bare-swallow handler anywhere in this file — the
scanner counts test files too. Where a pin must swallow, it binds
``as exc`` + ``assert isinstance(exc, Exception)``. Docstrings avoid
the literal bare-swallow phrase (#173 lesson).
"""
import contextlib
import json
import os
import subprocess
import sys
import threading
import types
from pathlib import Path

import pytest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs
from nomorals.autonomy import failures as wf
from nomorals.autonomy import weakness as wk


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    wf.clear_failures()
    yield
    wf.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


def _sites():
    return [f.site for f in wf.recent_failures()]


def _boom(*args, **kwargs):
    raise RuntimeError("boom")


# ---- Store-level (durability mirror of #141+) ----------------------------

def test_store_round_trip():
    store = fs.Store("autonomy")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("autonomy").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("autonomy")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("autonomy").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("autonomy")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("autonomy").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("autonomy").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("autonomy").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("autonomy")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("autonomy").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("autonomy").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("autonomy").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("autonomy").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("autonomy")
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
    entries, _ = fs.Store("autonomy").load()
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        wf.record_failure("no-home", ValueError("boom"))
        assert wf.failure_count() == 1
        assert [f.site for f in wf.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_ledger_api_record_recent_count_clear_format():
    assert wf.failure_count() == 0
    assert wf.recent_failures() == []
    assert wf.format_failures() == "no autonomy failures recorded."
    wf.record_failure("autonomy.site-a", ValueError("boom"))
    wf.record_failure("autonomy.site-b", RuntimeError("bang"))
    assert wf.failure_count() == 2
    assert [f.site for f in wf.recent_failures()] == ["autonomy.site-b", "autonomy.site-a"]
    assert [f.site for f in wf.recent_failures(1)] == ["autonomy.site-b"]
    text = wf.format_failures()
    assert "2 total" in text and "autonomy.site-a" in text and "autonomy.site-b" in text
    wf.clear_failures()
    assert wf.failure_count() == 0
    assert wf.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    wf.record_failure("autonomy.disk", ValueError("boom"))
    entries, total = fs.Store("autonomy").load()
    assert total == 1 and entries[0]["site"] == "autonomy.disk"
    wf.clear_failures()
    assert fs.Store("autonomy").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    wf.record_failure("autonomy.mem", ValueError("boom"))
    assert wf.failure_count() == 1
    assert [f.site for f in wf.recent_failures()] == ["autonomy.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                wf.record_failure(f"autonomy.w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
            assert isinstance(exc, Exception)
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert wf.failure_count() == 400
    assert len(wf.recent_failures(1000)) == 200
    entries, total = wf._store.load()
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
        "from nomorals.autonomy import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.autonomy import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.autonomy import failures as f\n"
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
    store_path = fs.Store("autonomy").path()
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

class FakeCursor:
    def __init__(self, row=None, rows=None, rowcount=0):
        self._row = row
        self._rows = rows or []
        self.rowcount = rowcount

    def fetchone(self):
        return self._row

    def fetchall(self):
        return self._rows


class FakeDB:
    """Dispatches fetch results by SQL substring; commit/SELECT can fail."""

    def __init__(self, fail_commit=False, fail_select=False):
        self.fail_commit = fail_commit
        self.fail_select = fail_select
        self.status_row = ("wid1", 2, "open")
        self.occurrences_row = (3,)
        self.weakness_row = ("tool_failure", "toolx", "{}", 3)
        self.rows = []
        self.rowcount = 0

    def execute(self, sql, params=()):
        if self.fail_select and str(sql).lstrip().upper().startswith("SELECT"):
            raise RuntimeError("select boom")
        sql = str(sql)
        if "SELECT id, occurrences, status FROM weaknesses" in sql:
            return FakeCursor(row=self.status_row)
        if "COUNT(*) FROM weakness_sightings" in sql:
            return FakeCursor(row=(5,))
        if "kind, subject, evidence, occurrences FROM weaknesses" in sql:
            return FakeCursor(row=self.weakness_row)
        if "occurrences FROM weaknesses WHERE id" in sql:
            return FakeCursor(row=self.occurrences_row)
        if "kind, subject FROM weaknesses WHERE id" in sql:
            return FakeCursor(row=("tool_failure", "toolx"))
        if "id, status FROM weaknesses" in sql:
            return FakeCursor(rows=self.rows)
        if "SELECT id FROM weaknesses WHERE kind" in sql:
            return FakeCursor(row=None)
        if "fix, verified_ts, subject FROM weakness_fixes" in sql:
            return FakeCursor(row=None)
        if "ok_streak FROM tool_health" in sql:
            return FakeCursor(row=(5,))
        return FakeCursor(row=None, rows=[], rowcount=self.rowcount)

    def commit(self):
        if self.fail_commit:
            raise RuntimeError("commit boom")


def _approved_db(fail_commit=False):
    """FakeDB whose proposed-row SELECT returns a real proposed row."""
    db = FakeDB(fail_commit=fail_commit)
    real_exec = db.execute
    db.execute = lambda sql, params=(): (
        FakeCursor(row=("tool_failure", "toolx", "fix it"))
        if "status = 'proposed'" in str(sql) else real_exec(sql, params))
    return db


def test_pin_record_fix_commit():
    db = FakeDB(fail_commit=True)
    assert wk.record_fix(db, "wid1", "do the thing") is True
    assert "weakness.record_fix_commit" in _sites()


def test_pin_report_weakness_commit(monkeypatch):
    db = FakeDB(fail_commit=True)
    db.status_row = None
    wid = wk.report_weakness(db, "tool_failure", "toolx")
    assert bool(wid)
    assert "weakness.report_weakness_commit" in _sites()


def test_pin_report_weakness_status_commit(monkeypatch):
    monkeypatch.setattr(wk, "recall_fix", lambda *a, **k: None)
    monkeypatch.setattr(wk, "investigate_weakness", lambda *a, **k: True)
    db = FakeDB(fail_commit=True)
    wid = wk.report_weakness(db, "tool_failure", "toolx")
    assert bool(wid)
    assert "weakness.report_weakness_status_commit" in _sites()


def test_pin_degradation_sample_commit():
    db = FakeDB(fail_commit=True)
    out = wk.record_degradation_sample(db, "subj", "metric", 1.0)
    assert out["subject"] == "subj"
    assert "weakness.degradation_sample_commit" in _sites()


def test_pin_severity_query():
    db = FakeDB(fail_select=True)
    sev = wk.severity(db, "wid1")
    assert sev["level"] == "watch"
    assert "weakness.severity_query" in _sites()


def test_pin_acceleration_query():
    db = FakeDB(fail_select=True)
    acc = wk.acceleration(db, "tool_failure", "toolx")
    assert acc["accelerating"] is False and acc["ratio"] == 0.0
    assert "weakness.acceleration_query" in _sites()


def test_pin_investigate_publish(monkeypatch):
    monkeypatch.setattr(wk._organs, "emit", lambda *a, **k: None)
    monkeypatch.setattr(wk.global_bus, "publish", _boom)
    assert wk.investigate_weakness(FakeDB(), "wid1") is True
    assert "weakness.investigate_publish" in _sites()


def test_pin_idle_tick_status_commit(monkeypatch):
    monkeypatch.setattr(wk, "investigate_weakness", lambda *a, **k: True)
    db = FakeDB(fail_commit=True)
    db.rows = [("wid1", "open")]
    rep = wk.idle_tick(db)
    assert rep["routed"] == 1
    assert "weakness.idle_tick_status_commit" in _sites()


def test_pin_expire_stale_commit():
    db = FakeDB(fail_commit=True)
    db.rowcount = 2
    assert wk.expire_stale(db) == 2
    assert "weakness.expire_stale_commit" in _sites()


def test_pin_record_proposal_commit(monkeypatch):
    monkeypatch.setattr(wk.global_bus, "publish", lambda *a, **k: None)
    wk.record_proposal(FakeDB(fail_commit=True), "wid1", "fix it")
    assert "weakness.record_proposal_commit" in _sites()


def test_pin_record_proposal_publish(monkeypatch):
    monkeypatch.setattr(wk.global_bus, "publish", _boom)
    wk.record_proposal(FakeDB(), "wid1", "fix it")
    assert "weakness.record_proposal_publish" in _sites()


def test_pin_approve_commit(monkeypatch):
    monkeypatch.setattr(wk._organs, "emit", lambda *a, **k: None)
    monkeypatch.setattr(wk.global_bus, "publish", lambda *a, **k: None)
    assert wk.approve_weakness(_approved_db(fail_commit=True), "wid1") is True
    assert "weakness.approve_commit" in _sites()


def test_pin_approve_publish(monkeypatch):
    monkeypatch.setattr(wk._organs, "emit", lambda *a, **k: None)
    monkeypatch.setattr(wk.global_bus, "publish", _boom)
    assert wk.approve_weakness(_approved_db(), "wid1") is True
    assert "weakness.approve_publish" in _sites()


def test_pin_resolve_commit():
    db = FakeDB(fail_commit=True)
    db.rowcount = 1
    assert wk.resolve_weakness(db, "wid1", "fixed") is True
    assert "weakness.resolve_commit" in _sites()


def test_pin_resolve_publish(monkeypatch):
    monkeypatch.setattr(wk.global_bus, "publish", _boom)
    db = FakeDB()
    db.rowcount = 1
    assert wk.resolve_weakness(db, "wid1", "fixed") is True
    assert "weakness.resolve_publish" in _sites()


def test_pin_dismiss_commit():
    db = FakeDB(fail_commit=True)
    db.rowcount = 1
    assert wk.dismiss_weakness(db, "wid1") is True
    assert "weakness.dismiss_commit" in _sites()


def test_pin_tool_result_commit():
    db = FakeDB(fail_commit=True)
    out = wk.record_tool_result(db, "toolx", True)
    assert out["ok_streak"] == 5
    assert "weakness.tool_result_commit" in _sites()


def test_pin_open_severity(monkeypatch):
    def _raise(*a, **k):
        raise RuntimeError("sev boom")
    monkeypatch.setattr(wk, "severity", _raise)
    db = FakeDB()
    row = ("wid1", "tool_failure", "toolx", "{}", 0.0, 0.0, 3, "open", "")
    real_exec = db.execute
    db.execute = lambda sql, params=(): (
        FakeCursor(rows=[row])
        if "ORDER BY occurrences DESC" in str(sql) else real_exec(sql, params))
    out = wk.open_weaknesses(db)
    assert len(out) == 1 and out[0]["severity"] == "watch"
    assert "weakness.open_severity" in _sites()
