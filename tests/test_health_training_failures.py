"""Pass #177 pins: the health failure ledger (REUSED from #176) + training conversions.

``nomorals/health/failures.py`` (namespace ``health``) is reused from
#176: training.py is a health-subsystem module, so reuse is the expected
call (unlike #161's engine/trivia split). The 17 durability tests below
mirror the #141+ shape exactly (same ledger as #176); the 16 site pins
prove each conversion records without changing flow (fallback value
unchanged AND a ledger entry under its site name); the 2 NOT-recording
pins prove the exclusions stay silent. No pin hardcodes scan totals or
training counts. Every test redirects stores with the app's own
``NM_HOME`` override (never the real home, never the repo tree).
Restart tests use real subprocesses. Every trigger was probed live
before its pin was written (probe.py).

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
from nomorals.health import training as tr


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


def _coach(tmp_path, **kw):
    return tr.TrainingCoach(store_path=str(tmp_path / "t.db"), **kw)


def test_pin_readiness_coach_import(tmp_path, monkeypatch):
    dummy = types.ModuleType("coach-broken")
    monkeypatch.setitem(sys.modules, "nomorals.health.coach", dummy)
    tc = _coach(tmp_path)
    info = tc.readiness()
    assert info.level == "unknown" and info.has_data is False
    assert "training.readiness_coach_import" in _sites()


def test_pin_get_plan(tmp_path):
    tc = _coach(tmp_path)
    tc._db = BoomDB()
    assert tc.get_plan("x") is None
    assert "training.get_plan" in _sites()


def test_pin_latest_plan(tmp_path):
    tc = _coach(tmp_path)
    tc._db = BoomDB()
    assert tc.latest_plan() is None
    assert "training.latest_plan" in _sites()


def test_pin_last_loads(tmp_path):
    tc = _coach(tmp_path)
    tc._db = BoomDB()
    assert tc._last_loads("push") == {}
    assert "training.last_loads" in _sites()


def test_pin_set_feedback_insert(tmp_path):
    tc = _coach(tmp_path)
    tc._db.execute("DROP TABLE set_feedback")
    tc._db.commit()
    w = tr.Workout(date="2026-10-25", kind="push",
                   exercises=[tr.Exercise(key="bench", name="Bench", sets=3,
                                          reps="8", rest_secs=90,
                                          intensity="moderate")])
    assert tc.log_workout(w, True, rpe=8, set_rpes={"bench": 8}) is True
    assert "training.set_feedback_insert" in _sites()


def test_pin_progression_advice(tmp_path):
    tc = _coach(tmp_path)
    tc._db = BoomDB()
    assert tc.progression_advice("bench") == "couldn't read progression history."
    assert "training.progression_advice" in _sites()


def test_pin_history(tmp_path):
    tc = _coach(tmp_path)
    tc._db = BoomDB()
    assert tc.history() == []
    assert "training.history" in _sites()


def test_pin_weekly_sets_loads(tmp_path):
    import datetime
    tc = _coach(tmp_path)
    kind = list(tr._DAY_LIBRARY)[0]
    tc._db.execute(
        "INSERT INTO workout_log VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("wl_pin", datetime.date.today().isoformat(), "", kind,
         "moderate", 1, 8, "not-json{{", "", 0.0))
    tc._db.commit()
    assert isinstance(tc.weekly_sets(), dict)
    assert "training.weekly_sets_loads" in _sites()


def test_pin_volume_report(tmp_path, monkeypatch):
    tc = _coach(tmp_path)
    monkeypatch.setattr(tc, "weekly_sets", _boom)
    assert tc.volume_report() == "couldn't build the volume report."
    assert "training.volume_report" in _sites()


def test_pin_personal_records_loads(tmp_path):
    tc = _coach(tmp_path)
    tc._db.execute(
        "INSERT INTO workout_log VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("wl_pin2", "2026-10-25", "", "push", "moderate", 1, 8,
         "corrupt[[", "", 0.0))
    tc._db.commit()
    assert tc.personal_records() == []
    assert "training.personal_records_loads" in _sites()


def test_pin_format_prs(tmp_path, monkeypatch):
    tc = _coach(tmp_path)
    monkeypatch.setattr(tc, "personal_records", _boom)
    assert tc.format_prs() == "couldn't read PRs right now."
    assert "training.format_prs" in _sites()


def test_pin_deload_readiness(tmp_path, monkeypatch):
    tc = _coach(tmp_path)

    def ws(weeks_back=0):
        return {"chest": {0: 10, 1: 5, 2: 0}[weeks_back]}

    monkeypatch.setattr(tc, "weekly_sets", ws)
    monkeypatch.setattr(tc, "readiness", _boom)
    assert tc.deload_advisory() == ""
    assert "training.deload_readiness" in _sites()


def test_pin_deload_advisory(tmp_path, monkeypatch):
    tc = _coach(tmp_path)
    monkeypatch.setattr(tc, "weekly_sets", _boom)
    assert tc.deload_advisory() == ""
    assert "training.deload_advisory" in _sites()


def test_pin_format_workout(tmp_path):
    tc = _coach(tmp_path)
    assert tc.format_workout(None) == "workout unavailable."
    assert "training.format_workout" in _sites()


def test_pin_format_plan(tmp_path):
    tc = _coach(tmp_path)
    assert tc.format_plan(None) == "plan unavailable."
    assert "training.format_plan" in _sites()


def test_pin_bump_load():
    assert tr._bump_load(80, 1.0) == 80
    assert "training.bump_load" in _sites()


# ---- NOT-recording pins (exclusions) --------------------------------------

class _BadAnalysis:
    @property
    def issues(self):
        raise RuntimeError("boom")


def test_pin_weak_point_notes_not_recording(tmp_path):
    tc = _coach(tmp_path)
    assert tc.weak_point_notes([_BadAnalysis()]) == ""
    assert _sites() == []


def test_pin_speak_cues_tts_not_recording(tmp_path):
    tc = _coach(tmp_path, tts=_boom)
    w = tr.Workout(date="2026-10-25", kind="push",
                   exercises=[tr.Exercise(key="bench", name="Bench", sets=3,
                                          reps="8", rest_secs=90,
                                          intensity="moderate")])
    got = tc.speak_cues(w)
    assert got == [c.text for c in tr.coaching_cues(w)]
    assert _sites() == []
