"""Pass #207 pins: the travel ledger records the one convert, and
the 12 exclusions stay silent.

``nomorals/travel/groups.py`` is a "never raises on reads" store:
failed reads degrade to empty lists, None, or zero. This pass converts
exactly one site — ``list_expenses`` (621), whose silent empty list
fed both the settle command's "Everyone's settled up" verdict and the
leave-trip settle-guard's fail-open pass. The 17 durability tests
below mirror the #141+ shape exactly (same as #206's, against the NEW
travel ledger), the behavior pin proves 621 records while keeping its
``[]`` fallback, and the 12 NOT-recording pins prove each exclusion
stays silent with its designed output. No pin hardcodes scan totals
or groups.py counts. Every test redirects stores with the app's own
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
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs
from nomorals.travel import failures as tf
from nomorals.travel import groups as gg


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    tf.clear_failures()
    yield
    tf.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


def _assert_silent():
    assert tf.failure_count() == 0
    assert tf.recent_failures() == []


# ---- Store-level (durability mirror of #141+) ----------------------------

def test_store_round_trip():
    store = fs.Store("travel")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("travel").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("travel")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("travel").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("travel")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()


def test_store_namespaces_do_not_clobber():
    assert fs.Store("travel").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("travel").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("travel")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("travel").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("travel").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("travel").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("travel").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("travel")
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
    entries, _ = fs.Store("travel").load()
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
    assert tf.format_failures() == "no travel failures recorded."
    tf.record_failure("travel.site-a", ValueError("boom"))
    tf.record_failure("travel.site-b", RuntimeError("bang"))
    assert tf.failure_count() == 2
    assert [f.site for f in tf.recent_failures()] == ["travel.site-b", "travel.site-a"]
    assert [f.site for f in tf.recent_failures(1)] == ["travel.site-b"]
    text = tf.format_failures()
    assert "2 total" in text and "travel.site-a" in text and "travel.site-b" in text
    tf.clear_failures()
    assert tf.failure_count() == 0
    assert tf.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    tf.record_failure("travel.disk", ValueError("boom"))
    entries, total = fs.Store("travel").load()
    assert total == 1 and entries[0]["site"] == "travel.disk"
    tf.clear_failures()
    assert fs.Store("travel").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    tf.record_failure("travel.mem", ValueError("boom"))
    assert tf.failure_count() == 1
    assert [f.site for f in tf.recent_failures()] == ["travel.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                tf.record_failure(f"travel.w{k}-{i}", ValueError("boom"))
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
        "from nomorals.travel import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.travel import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.travel import failures as f\n"
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
    store_path = fs.Store("travel").path()
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


# ---- Behavior pin (the one convert) --------------------------------------

class _RaisingDB:
    """Every DB op raises: the store's whole-layer-broken mode."""

    def execute(self, *a, **k):
        raise RuntimeError("db down")


def _store(tmp_path):
    return gg.GroupTripStore("pin", db_path=str(tmp_path / "t.db"))


def test_behavior_621_expenses_read_failure_returns_empty_and_records(tmp_path):
    store = _store(tmp_path)
    trip = store.create_trip("Trip", ["Alice"], created_by="Alice")
    store.add_expense(trip.id, "Alice", 500000, ["Alice"], "suya")
    assert len(store.list_expenses(trip.id)) == 1
    store._db = _RaisingDB()
    assert store.list_expenses(trip.id) == []
    assert tf.failure_count() == 1
    rec = tf.recent_failures()[0]
    assert rec.site == "groups.list_expenses"
    assert rec.exc_type == "RuntimeError"


# ---- NOT-recording pins (one per exclusion) ------------------------------

def test_not_recording_308_double_init_alter_failures_pass(tmp_path):
    path = str(tmp_path / "t.db")
    gg.GroupTripStore("pin", db_path=path)
    gg.GroupTripStore("pin", db_path=path)  # ALTERs now fail duplicate-column
    _assert_silent()


def test_not_recording_370_list_trips_failure_returns_empty(tmp_path):
    store = _store(tmp_path)
    store._db = _RaisingDB()
    assert store.list_trips() == []
    _assert_silent()


def test_not_recording_458_get_poll_failure_returns_none(tmp_path):
    store = _store(tmp_path)
    store._db = _RaisingDB()
    assert store.get_poll("nope") is None
    _assert_silent()


class _SelectiveCursor:
    def __init__(self, row=None):
        self._row = row

    def fetchone(self):
        return self._row

    def fetchall(self):
        return []


class _VotesDownDB:
    """Polls read fine; only the votes query fails (site 502)."""

    def execute(self, sql, params=()):
        if "FROM votes" in sql:
            raise RuntimeError("db down")
        row = {"id": "p1", "trip_id": "t1", "question": "Q?",
               "options_json": '["a", "b"]', "deadline": 0.0,
               "created_by": "Alice", "created_at": 0.0}
        return _SelectiveCursor(row)


def test_not_recording_502_votes_failure_reads_as_zero_votes(tmp_path):
    store = _store(tmp_path)
    store._db = _VotesDownDB()
    res = store.poll_result("p1")
    assert res is not None
    assert res["counts"] == [0, 0] and res["total"] == 0
    assert res["winner"] is None
    _assert_silent()


def test_not_recording_525_closing_soon_failure_returns_empty(tmp_path):
    store = _store(tmp_path)
    store._db = _RaisingDB()
    assert store.polls_closing_soon("t1") == []
    _assert_silent()


def test_not_recording_674_payments_failure_returns_empty(tmp_path):
    store = _store(tmp_path)
    store._db = _RaisingDB()
    assert store.list_payments("t1") == []
    _assert_silent()


def test_not_recording_702_budget_failure_returns_zero(tmp_path):
    store = _store(tmp_path)
    store._db = _RaisingDB()
    assert store.get_budget("t1", "Alice") == 0
    _assert_silent()


def test_not_recording_816_legs_failure_returns_empty(tmp_path):
    store = _store(tmp_path)
    store._db = _RaisingDB()
    assert store.list_legs("t1") == []
    _assert_silent()


def test_not_recording_849_agree_failure_returns_none(tmp_path):
    store = _store(tmp_path)
    store._db = _RaisingDB()
    assert store.agree("t1", "p1") is None
    _assert_silent()


def test_not_recording_873_proposals_failure_returns_empty(tmp_path):
    store = _store(tmp_path)
    store._db = _RaisingDB()
    assert store.list_proposals("t1") == []
    _assert_silent()


def _gtrip_env(tmp_path):
    chat = SimpleNamespace(kind="group", platform="test", id="g1")
    ctx = SimpleNamespace(
        settings=SimpleNamespace(community_dir=str(tmp_path)))
    return chat, ctx


def _gtrip_db(tmp_path):
    chat, _ = _gtrip_env(tmp_path)
    return gg.GroupTripStore(gg._group_key(chat), db_path=str(
        tmp_path / "group_trips"
        / f"{gg._sanitize_group_key(gg._group_key(chat))}.db"))


def test_not_recording_1142_bad_split_falls_back_to_equal(tmp_path):
    chat, ctx = _gtrip_env(tmp_path)
    out = gg.control_gtrip("new Trip", ctx, chat, sender="Alice")
    assert "Group trip created" in out
    store = _gtrip_db(tmp_path)
    trip = store.list_trips()[0]
    gg.control_gtrip(f"join {trip.id}", ctx, chat, sender="Bob")
    out = gg.control_gtrip("expense 5k suya split 1.2.3", ctx, chat,
                           sender="Alice")
    assert "Logged" in out
    exp = store.list_expenses(trip.id)[0]
    assert exp.split == gg.SPLIT_EQUAL
    _assert_silent()


def test_not_recording_1273_leg_parse_failure_keeps_raw_text(tmp_path):
    chat, ctx = _gtrip_env(tmp_path)
    gg.control_gtrip("new Trip", ctx, chat, sender="Alice")
    with patch.object(gg, "parse_confirmation",
                      side_effect=RuntimeError("no parse")):
        out = gg.control_gtrip("leg BA075 LOS to LHR departs 22:45",
                               ctx, chat, sender="Alice")
    assert "Leg added" in out
    store = _gtrip_db(tmp_path)
    trip = store.list_trips()[0]
    leg = store.list_legs(trip.id)[0]
    assert leg.flight["flight_number"] == "BA075 LOS to LHR departs 22:45"
    _assert_silent()
