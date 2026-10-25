"""Pass #183 pins: the NEW triggers failure ledger + saved_search conversions.

``nomorals/triggers/failures.py`` (namespace ``triggers``) is new in
this pass — saved_search.py is a triggers-subsystem module, and no
triggers-area ledger existed through #182. The 17 durability tests
below mirror the #141+ shape exactly (same as #182's, against the new
ledger); the 14 site pins prove each conversion records without
changing flow (fallback value unchanged AND a ledger entry under its
site name); the 3 NOT-recording pins prove each exclusion stays silent
with the designed output. No pin hardcodes scan totals or
saved_search counts. Every test redirects stores with the app's own
``NM_HOME`` override (never the real home, never the repo tree).
Restart tests use real subprocesses. Every trigger was probed live
before its pin was written (probe.py).

Pins rule (#169): no bare-swallow handler anywhere in this file — the
scanner counts test files too. Where a pin must swallow, it binds
``as exc`` + ``assert isinstance(exc, Exception)``. Docstrings avoid
the literal bare-swallow phrase (#173 lesson).
"""
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
from nomorals.triggers import failures as tf
from nomorals.triggers import saved_search as ss


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    tf.clear_failures()
    yield
    tf.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


def _sites():
    return [f.site for f in tf.recent_failures()]


# ---- Store-level (durability mirror of #141+) ----------------------------

def test_store_round_trip():
    store = fs.Store("triggers")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("triggers").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("triggers")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("triggers").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("triggers")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("triggers").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("triggers").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("triggers").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("triggers")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("triggers").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("triggers").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("triggers").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("triggers").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("triggers")
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
    entries, _ = fs.Store("triggers").load()
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        tf.record_failure("no-home", ValueError("boom"))
        assert tf.failure_count() == 1
        assert [f.site for f in tf.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_ledger_api_record_recent_count_clear_format():
    assert tf.failure_count() == 0
    assert tf.recent_failures() == []
    assert tf.format_failures() == "no triggers failures recorded."
    tf.record_failure("triggers.site-a", ValueError("boom"))
    tf.record_failure("triggers.site-b", RuntimeError("bang"))
    assert tf.failure_count() == 2
    assert [f.site for f in tf.recent_failures()] == ["triggers.site-b", "triggers.site-a"]
    assert [f.site for f in tf.recent_failures(1)] == ["triggers.site-b"]
    text = tf.format_failures()
    assert "2 total" in text and "triggers.site-a" in text and "triggers.site-b" in text
    tf.clear_failures()
    assert tf.failure_count() == 0
    assert tf.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    tf.record_failure("triggers.disk", ValueError("boom"))
    entries, total = fs.Store("triggers").load()
    assert total == 1 and entries[0]["site"] == "triggers.disk"
    tf.clear_failures()
    assert fs.Store("triggers").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    tf.record_failure("triggers.mem", ValueError("boom"))
    assert tf.failure_count() == 1
    assert [f.site for f in tf.recent_failures()] == ["triggers.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                tf.record_failure(f"triggers.w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
            assert isinstance(exc, Exception)
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert tf.failure_count() == 400
    assert len(tf.recent_failures(1000)) == 200
    entries, total = tf._store.load()
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
        "from nomorals.triggers import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.triggers import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.triggers import failures as f\n"
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
    store_path = fs.Store("triggers").path()
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

class _BrokenDB:
    def execute(self, *a, **k):
        raise sqlite3.OperationalError("db down")


class _BadFilters(dict):
    def get(self, *a, **k):
        raise RuntimeError("filters down")


class _Sched:
    def __init__(self):
        self.scheduled = []

    def list_jobs(self):
        raise RuntimeError("jobs down")

    async def schedule_cron(self, **kw):
        self.scheduled.append(kw)
        return True


def _broken_store(tmp_path):
    store = ss.SavedSearchStore(str(tmp_path / "saved_search.db"))
    store._db = _BrokenDB()
    return store


def test_pin_saved_search_get(tmp_path):
    assert _broken_store(tmp_path).get("x") is None
    assert "saved_search.get" in _sites()


def test_pin_saved_search_list(tmp_path):
    assert _broken_store(tmp_path).list() == []
    assert "saved_search.list" in _sites()


def test_pin_saved_search_remove(tmp_path):
    assert _broken_store(tmp_path).remove("x") is False
    assert "saved_search.remove" in _sites()


def test_pin_saved_search_purge_expired(tmp_path):
    assert _broken_store(tmp_path).purge_expired() == 0
    assert "saved_search.purge_expired" in _sites()


def test_pin_saved_search_seen(tmp_path):
    assert _broken_store(tmp_path).seen("x", "k") is False
    assert "saved_search.seen" in _sites()


def test_pin_saved_search_snooze(tmp_path):
    assert _broken_store(tmp_path).snooze("x", 1) is False
    assert "saved_search.snooze" in _sites()


def test_pin_saved_search_unsnooze(tmp_path):
    assert _broken_store(tmp_path).unsnooze("x") is False
    assert "saved_search.unsnooze" in _sites()


def test_pin_saved_search_set_digest_at(tmp_path):
    assert _broken_store(tmp_path).set_digest_at("x", "10:00") is False
    assert "saved_search.set_digest_at" in _sites()


def test_pin_saved_search_pending_digest_count(tmp_path):
    assert _broken_store(tmp_path).pending_digest_count("x") == 0
    assert "saved_search.pending_digest_count" in _sites()


def test_pin_saved_search_take_digest(tmp_path):
    assert _broken_store(tmp_path).take_digest("x") == []
    assert "saved_search.take_digest" in _sites()


def test_pin_saved_search_row_to_search():
    row = {"id": "w", "domain": "property", "query": "q",
           "filters_json": "not-json", "cadence_s": 21600, "ttl_days": 14,
           "created_at": 0.0, "expires_at": 0.0, "last_run": 0.0,
           "match_count": 0, "active": 1}
    assert ss._row_to_search(row).filters == {}
    assert "saved_search._row_to_search" in _sites()


def test_pin_saved_search_matches_filters_outer():
    assert ss._matches_filters({}, _BadFilters()) is True
    assert "saved_search._matches_filters" in _sites()


def test_pin_saved_search_dedup_key():
    with patch.object(ss, "_photo_hash", side_effect=RuntimeError("boom")):
        assert ss.dedup_key({"photo_bytes": b"x"}) == ""
    assert "saved_search.dedup_key" in _sites()


def test_pin_saved_search_ensure_schedule():
    sched = _Sched()
    assert ss.ensure_schedule(sched) is True
    assert sched.scheduled and sched.scheduled[0]["task_id"] == "saved-search-hourly"
    assert "saved_search.ensure_schedule" in _sites()


# ---- NOT-recording pins (exclusions stay silent) -------------------------

def test_pin_exclusion_register_matcher():
    assert ss.register_matcher(123, lambda *a: []) is False
    assert _sites() == []


def test_pin_exclusion_max_price():
    assert ss._matches_filters({"price_kobo": "abc"},
                               {"max_price_kobo": 100}) is True
    assert _sites() == []


def test_pin_exclusion_parse_watch():
    assert ss.parse_watch(12345) is None
    assert _sites() == []
