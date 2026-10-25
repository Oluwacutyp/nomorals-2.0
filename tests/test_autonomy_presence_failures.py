"""Pass #180 pins: the REUSED autonomy failure ledger + presence conversions.

``nomorals/autonomy/failures.py`` (namespace ``autonomy``, from #175)
is reused — presence.py is an autonomy-subsystem module, so no new
ledger file. The 17 durability tests below mirror the #141+ shape
exactly (same as #175's, against the same ledger); the 13 site pins
prove each conversion records without changing flow (fallback value
unchanged AND a ledger entry under its site name); the 4
NOT-recording pins prove each exclusion stays silent with the
designed output. No pin hardcodes scan totals or presence counts.
Every test redirects stores with the app's own ``NM_HOME`` override
(never the real home, never the repo tree). Restart tests use real
subprocesses. Every trigger was probed live before its pin was
written (probe.py).

Pins rule (#169): no bare-swallow handler anywhere in this file — the
scanner counts test files too. Where a pin must swallow, it binds
``as exc`` + ``assert isinstance(exc, Exception)``. Docstrings avoid
the literal bare-swallow phrase (#173 lesson).
"""
import datetime as _dt
import json
import os
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs
from nomorals.autonomy import failures as pf
from nomorals.autonomy import presence as pr
from nomorals.autonomy import patterns as _patterns
from nomorals.autonomy import idle as _idle
from nomorals.core.events import global_bus


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
        pf.record_failure("no-home", ValueError("boom"))
        assert pf.failure_count() == 1
        assert [f.site for f in pf.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_ledger_api_record_recent_count_clear_format():
    assert pf.failure_count() == 0
    assert pf.recent_failures() == []
    assert pf.format_failures() == "no autonomy failures recorded."
    pf.record_failure("autonomy.site-a", ValueError("boom"))
    pf.record_failure("autonomy.site-b", RuntimeError("bang"))
    assert pf.failure_count() == 2
    assert [f.site for f in pf.recent_failures()] == ["autonomy.site-b", "autonomy.site-a"]
    assert [f.site for f in pf.recent_failures(1)] == ["autonomy.site-b"]
    text = pf.format_failures()
    assert "2 total" in text and "autonomy.site-a" in text and "autonomy.site-b" in text
    pf.clear_failures()
    assert pf.failure_count() == 0
    assert pf.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    pf.record_failure("autonomy.disk", ValueError("boom"))
    entries, total = fs.Store("autonomy").load()
    assert total == 1 and entries[0]["site"] == "autonomy.disk"
    pf.clear_failures()
    assert fs.Store("autonomy").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    pf.record_failure("autonomy.mem", ValueError("boom"))
    assert pf.failure_count() == 1
    assert [f.site for f in pf.recent_failures()] == ["autonomy.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                pf.record_failure(f"autonomy.w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
            assert isinstance(exc, Exception)
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

class _ProxyDB:
    """Delegates to a real sqlite3 connection, with fault injection."""

    def __init__(self, conn, fail_commit=False, fail_on=()):
        self._conn = conn
        self.fail_commit = fail_commit
        self.fail_on = tuple(fail_on)

    def execute(self, sql, params=()):
        for sub in self.fail_on:
            if sub in sql:
                raise sqlite3.OperationalError(f"pin: fail on {sub!r}")
        return self._conn.execute(sql, params)

    def commit(self):
        if self.fail_commit:
            raise sqlite3.OperationalError("pin: commit fails")
        return self._conn.commit()


def _fresh_db():
    conn = sqlite3.connect(":memory:")
    pr.ensure_schema(conn)
    return conn


def _ts_hour(hour):
    base = _dt.datetime(2026, 1, 1, tzinfo=_dt.timezone.utc)
    return (base + _dt.timedelta(hours=hour)).timestamp()


def test_pin_note_trigger_commit():
    db = _ProxyDB(_fresh_db(), fail_commit=True)
    pr.note_trigger(db, "probe-trigger")
    assert "presence.note_trigger_commit" in _sites()


def test_pin_nudge_due_commit():
    db = _ProxyDB(_fresh_db(), fail_commit=True)
    assert pr._nudge_due(db, "probe-key") is True
    assert "presence.nudge_due_commit" in _sites()


def test_pin_interruptibility_quiet_hours(monkeypatch):
    monkeypatch.setattr(_patterns, "quiet_hours", _boom)
    res = pr.interruptibility(_fresh_db(), ts=_ts_hour(10))
    assert res["quiet_hours"] == []
    assert res["score"] == 0.75
    assert "presence.interruptibility_quiet_hours" in _sites()


def test_pin_queue_candidate_commit():
    db = _ProxyDB(_fresh_db(), fail_commit=True)
    assert isinstance(pr.queue_candidate(db, "k", "t"), int)
    assert "presence.queue_candidate_commit" in _sites()


def test_pin_drop_candidate_commit():
    conn = _fresh_db()
    pr.queue_candidate(conn, "k", "t")
    db = _ProxyDB(conn, fail_commit=True)
    assert pr.drop_candidate(db, 1) is True
    assert "presence.drop_candidate_commit" in _sites()


def test_pin_serendipity_novelty():
    db = _ProxyDB(_fresh_db(),
                  fail_on=("FROM presence_surface ORDER BY id DESC",))
    res = pr.serendipity_score(db, "k", "t")
    assert res["novelty"] == 0.7
    assert "presence.serendipity_novelty" in _sites()


def test_pin_record_feedback_commit():
    conn = _fresh_db()
    pr.surface(conn, "k", "t")
    db = _ProxyDB(conn, fail_commit=True)
    res = pr.record_surface_feedback(db, 1, "engaged")
    assert res.get("ok") is True
    assert "presence.record_feedback_commit" in _sites()


def test_pin_sense_interruptibility(monkeypatch):
    monkeypatch.setattr(pr, "interruptibility", _boom)
    res = pr.sense(_fresh_db())
    assert res["interruptibility"] == {"score": 0.5,
                                       "interruptible": True}
    assert "presence.sense_interruptibility" in _sites()


def test_pin_surface_commit():
    db = _ProxyDB(_fresh_db(), fail_commit=True)
    res = pr.surface(db, "k", "t")
    assert res.get("ok") is True
    assert "presence.surface_commit" in _sites()


def test_pin_heartbeat_publish(monkeypatch):
    # Trap pin (#155): the publish handler sits inside heartbeat's big
    # outer try; the record landing proves no name was deleted.
    monkeypatch.setattr(global_bus, "publish", _boom)
    did = pr.heartbeat(_fresh_db())
    assert isinstance(did, dict)
    assert "presence.heartbeat_publish" in _sites()


def test_pin_maybe_surface_commit():
    conn = _fresh_db()
    pr.queue_candidate(conn, "k", "due-title", priority="normal",
                       ttl_hours=1, ts=_ts_hour(0) - 100000)
    db = _ProxyDB(conn, fail_commit=True)
    ctx = {"weaknesses": [], "rising": [], "clusters": []}
    assert pr._maybe_surface(db, ctx, 0.9) == "due-title"
    assert "presence.maybe_surface_commit" in _sites()


def test_pin_mark_surface_commit():
    conn = _fresh_db()
    pr.surface(conn, "k", "t")
    db = _ProxyDB(conn, fail_commit=True)
    assert pr.mark_surface(db, 1, "delivered") is True
    assert "presence.mark_surface_commit" in _sites()


def test_pin_digest_audit_commit():
    db = _ProxyDB(_fresh_db(), fail_commit=True)
    assert isinstance(pr.digest(db), str)
    assert "presence.digest_audit_commit" in _sites()


# ---- NOT-recording pins (exclusions stay silent) --------------------------

def test_pin_not_recording_idle_state(monkeypatch):
    # Exclusion 281: failure score == documented no-history score.
    monkeypatch.setattr(_patterns, "quiet_hours", lambda *a, **k: [])
    monkeypatch.setattr(_idle, "idle_state", _boom)
    res_fail = pr.interruptibility(_fresh_db(), ts=_ts_hour(10))
    monkeypatch.setattr(_idle, "idle_state",
                        lambda *a, **k: {"idle_seconds": 0.0,
                                         "last_activity_ts": 0.0})
    res_nohist = pr.interruptibility(_fresh_db(), ts=_ts_hour(10))
    assert res_fail["score"] == res_nohist["score"]
    assert _sites() == []


def test_pin_not_recording_relevance(monkeypatch):
    # Exclusion 424: failure relevance == designed no-data value.
    monkeypatch.setattr(_patterns, "current_interests", _boom)
    res = pr.serendipity_score(_fresh_db(), "k", "t")
    assert res["relevance"] == 0.3
    assert _sites() == []


def test_pin_not_recording_anticipation(monkeypatch):
    # Exclusion 538: enhancement probe, failure == no-trigger baseline.
    db = _fresh_db()
    pr.note_trigger(db, "morning-pulse")
    monkeypatch.setattr(_patterns, "likely_next", _boom)
    assert pr.sense(db)["anticipation"] == []
    assert _sites() == []


def test_pin_not_recording_workspace_dir(monkeypatch):
    # Exclusion 937: unreadable parameter == designed cwd fallback.
    db = _fresh_db()
    got = {}
    monkeypatch.setattr(_idle, "workspace_db",
                        lambda d: got.setdefault("dir", d) or db)
    pr._heartbeat_fire(None, None)
    assert got["dir"] == os.getcwd()
    assert _sites() == []
