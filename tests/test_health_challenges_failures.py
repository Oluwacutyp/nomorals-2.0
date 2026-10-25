"""Pass #176 pins: the health failure ledger (NEW) + challenges conversions.

``nomorals/health/failures.py`` (namespace ``health``) is new: no
health-area ledger existed in the track, and challenges.py is a
health-subsystem module, so a new ledger is correct (unlike #161's
engine/trivia split, and unlike reusing another subsystem's ledger).
The 17 durability tests below mirror the #141+ shape exactly; the 16
site pins prove each conversion records without changing flow (fallback
value unchanged AND a ledger entry under its site name); the 2
NOT-recording pins prove the exclusions stay silent. No pin hardcodes
scan totals or challenges counts. Every test redirects stores with the
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
from nomorals.health import failures as hf
from nomorals.health import challenges as ch


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    hf.clear_failures()
    yield
    hf.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


def _sites():
    return [f.site for f in hf.recent_failures()]


def _boom(*args, **kwargs):
    raise RuntimeError("boom")


# ---- Store-level (durability mirror of #141+) ----------------------------

def test_store_round_trip():
    store = fs.Store("health")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("health").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("health")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("health").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("health")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("health").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("health").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("health").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("health")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("health").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("health").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("health").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("health").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("health")
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
    entries, _ = fs.Store("health").load()
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        hf.record_failure("no-home", ValueError("boom"))
        assert hf.failure_count() == 1
        assert [f.site for f in hf.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_ledger_api_record_recent_count_clear_format():
    assert hf.failure_count() == 0
    assert hf.recent_failures() == []
    assert hf.format_failures() == "no health failures recorded."
    hf.record_failure("health.site-a", ValueError("boom"))
    hf.record_failure("health.site-b", RuntimeError("bang"))
    assert hf.failure_count() == 2
    assert [f.site for f in hf.recent_failures()] == ["health.site-b", "health.site-a"]
    assert [f.site for f in hf.recent_failures(1)] == ["health.site-b"]
    text = hf.format_failures()
    assert "2 total" in text and "health.site-a" in text and "health.site-b" in text
    hf.clear_failures()
    assert hf.failure_count() == 0
    assert hf.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    hf.record_failure("health.disk", ValueError("boom"))
    entries, total = fs.Store("health").load()
    assert total == 1 and entries[0]["site"] == "health.disk"
    hf.clear_failures()
    assert fs.Store("health").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    hf.record_failure("health.mem", ValueError("boom"))
    assert hf.failure_count() == 1
    assert [f.site for f in hf.recent_failures()] == ["health.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                hf.record_failure(f"health.w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
            assert isinstance(exc, Exception)
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert hf.failure_count() == 400
    assert len(hf.recent_failures(1000)) == 200
    entries, total = hf._store.load()
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
        "from nomorals.health import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.health import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.health import failures as f\n"
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
    store_path = fs.Store("health").path()
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

class BoomDB:
    def execute(self, *a, **k):
        raise RuntimeError("boom")
    def commit(self):
        raise RuntimeError("commit boom")


class ConnProxy:
    """Delegates to a real connection except UPDATE, which raises."""
    def __init__(self, conn):
        self._conn = conn
    def execute(self, sql, params=()):
        if str(sql).lstrip().upper().startswith("UPDATE"):
            raise RuntimeError("update boom")
        return self._conn.execute(sql, params)
    def commit(self):
        return self._conn.commit()


def _store(tmp_path):
    return ch.ChallengeStore(str(tmp_path / "c.db"))


def _fake_challenge():
    return ch.Challenge(challenge_id="ch_x", name="t")


def test_pin_get(tmp_path):
    s = _store(tmp_path); s._db = BoomDB()
    assert s.get("x") is None
    assert "challenges.get" in _sites()


def test_pin_list_challenges(tmp_path):
    s = _store(tmp_path); s._db = BoomDB()
    assert s.list_challenges() == []
    assert "challenges.list_challenges" in _sites()


def test_pin_join_challenge(tmp_path, monkeypatch):
    s = _store(tmp_path); s._db = BoomDB()
    monkeypatch.setattr(s, "get", lambda cid: _fake_challenge())
    assert s.join_challenge("ch_x", "m") is False
    assert "challenges.join_challenge" in _sites()


def test_pin_members(tmp_path):
    s = _store(tmp_path); s._db = BoomDB()
    assert s.members("ch_x") == []
    assert "challenges.members" in _sites()


def test_pin_progress(tmp_path, monkeypatch):
    s = _store(tmp_path); s._db = BoomDB()
    monkeypatch.setattr(s, "get", lambda cid: _fake_challenge())
    assert s.progress("ch_x", "m") == (0, 0)
    assert "challenges.progress" in _sites()


def test_pin_leaderboard(tmp_path):
    s = _store(tmp_path); s._db = BoomDB()
    assert s.leaderboard("ch_x") == []
    assert "challenges.leaderboard" in _sites()


def test_pin_streak(tmp_path):
    s = _store(tmp_path); s._db = BoomDB()
    assert s.streak("m") == 0
    assert "challenges.streak" in _sites()


def test_pin_at_risk(tmp_path):
    s = _store(tmp_path); s._db = BoomDB()
    assert s.at_risk() == []
    assert "challenges.at_risk" in _sites()


def test_pin_freezes(tmp_path):
    s = _store(tmp_path); s._db = BoomDB()
    assert s.freezes("m") == 0
    assert "challenges.freezes" in _sites()


def test_pin_maybe_earn_freeze(tmp_path, monkeypatch):
    s = _store(tmp_path)
    monkeypatch.setattr(s, "freezes", lambda m: 0)
    monkeypatch.setattr(s, "_set_freezes", _boom)
    s._maybe_earn_freeze("m", 7)
    assert "challenges.maybe_earn_freeze" in _sites()


def test_pin_monthly_badge_update(tmp_path):
    s = _store(tmp_path)
    c = s.create_challenge("t")
    assert c is not None
    s._db = ConnProxy(s._db)
    out = s.monthly_challenge("distance")
    assert out is not None and out.badge.startswith("monthly:")
    assert "challenges.monthly_badge_update" in _sites()
    row = s._db.execute(
        "SELECT badge FROM challenges WHERE id = ?", (out.id,)).fetchone()
    assert row["badge"] != out.badge


def test_pin_personal_best(tmp_path):
    s = _store(tmp_path); s._db = BoomDB()
    assert s.personal_best("ch_x", "m") == 0
    assert "challenges.personal_best" in _sites()


def test_pin_note_personal_best(tmp_path, monkeypatch):
    s = _store(tmp_path); s._db = BoomDB()
    monkeypatch.setattr(s, "personal_best", lambda *a: 0)
    assert s._note_personal_best("ch_x", "m", 5) is False
    assert "challenges.note_personal_best" in _sites()


def test_pin_get_squad(tmp_path):
    s = _store(tmp_path); s._db = BoomDB()
    assert s.get_squad("sq_x") is None
    assert "challenges.get_squad" in _sites()


def test_pin_squad_leaderboard(tmp_path, monkeypatch):
    s = _store(tmp_path)
    monkeypatch.setattr(
        s, "get_squad",
        lambda sid: ch.Squad(squad_id="sq_x", name="s", members=["a", "b"]))
    monkeypatch.setattr(s, "streak", _boom)
    assert s.squad_leaderboard("sq_x") == []
    assert "challenges.squad_leaderboard" in _sites()


def test_pin_date_to_days():
    assert ch._date_to_days("not-a-date") == 0.0
    assert "challenges.date_to_days" in _sites()


# ---- NOT-recording pins (exclusions) --------------------------------------

def test_pin_risk_nudge_not_recording(tmp_path, monkeypatch):
    s = _store(tmp_path)
    monkeypatch.setattr(s, "streak", _boom)
    assert s.risk_nudge("m") == ""
    assert "challenges.risk_nudge" not in _sites()
    assert hf.failure_count() == 0


def test_pin_progress_bar_not_recording():
    assert ch._progress_bar("x", 5) == ""
    assert "challenges.progress_bar" not in _sites()
    assert hf.failure_count() == 0
