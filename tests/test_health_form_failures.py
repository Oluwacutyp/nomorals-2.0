"""Pass #187 pins: the health failure ledger (REUSED from #176) + form conversions.

``nomorals/health/failures.py`` (namespace ``health``) is reused from
#176/#177: form.py is a health-subsystem module, so reuse is the expected
call. The 17 durability tests below mirror the #141+ shape exactly (same
ledger as #176/#177); the 5 site pins prove each conversion records
without changing flow (fallback value unchanged AND a ledger entry under
its site name); the 11 NOT-recording pins prove the exclusions stay
silent. No pin hardcodes scan totals or form counts. Every test redirects
stores with the app's own ``NM_HOME`` override (never the real home, never
the repo tree). Restart tests use real subprocesses. Every trigger was
probed live before its pin was written (probe.py).

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
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs
from nomorals.health import failures as hf
from nomorals.health import form as fm
from nomorals.health.form import FormAnalysis, FormIssue
from nomorals.health.training import Workout


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
        except Exception as exc:
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
    hf.record_failure("form.site-a", ValueError("boom"))
    hf.record_failure("form.site-b", RuntimeError("bang"))
    assert hf.failure_count() == 2
    assert [f.site for f in hf.recent_failures()] == ["form.site-b", "form.site-a"]
    assert [f.site for f in hf.recent_failures(1)] == ["form.site-b"]
    text = hf.format_failures()
    assert "2 total" in text and "form.site-a" in text and "form.site-b" in text
    hf.clear_failures()
    assert hf.failure_count() == 0
    assert hf.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    hf.record_failure("form.disk", ValueError("boom"))
    entries, total = fs.Store("health").load()
    assert total == 1 and entries[0]["site"] == "form.disk"
    hf.clear_failures()
    assert fs.Store("health").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    hf.record_failure("form.mem", ValueError("boom"))
    assert hf.failure_count() == 1
    assert [f.site for f in hf.recent_failures()] == ["form.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                hf.record_failure(f"form.w{k}-{i}", ValueError("boom"))
        except Exception as exc:
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

class _BoomDB:
    def execute(self, *a, **k):
        raise RuntimeError("boom")

    def commit(self):
        raise RuntimeError("commit boom")


class _BoomCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _RowsDB:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, *a, **k):
        return _BoomCursor(self._rows)


def _store():
    s = fm.FormStore.__new__(fm.FormStore)
    s._db = _BoomDB()
    return s


class _RaisingCoach:
    def today(self, plan):
        raise RuntimeError("coach down")


class _CountingCoach:
    """First today() returns a real workout; the retry raises."""

    def __init__(self):
        self.n = 0

    def today(self, plan):
        self.n += 1
        if self.n == 1:
            return Workout(date="2026-10-25", kind="plan", exercises=[])
        raise RuntimeError("coach down")


class _SliceRaisingIssues(list):
    """Iterates fine but raises on slicing — forces the prepend block's
    issues-slice line to raise after mobility_for succeeds."""

    def __getitem__(self, key):
        if isinstance(key, slice):
            raise TypeError("no slicing")
        return super().__getitem__(key)


def test_pin_mobility_for():
    out = fm.mobility_for(SimpleNamespace(exercise="squat", issues=[object()]))
    assert out == []
    assert "form.mobility_for" in _sites()


def test_pin_augment_today():
    w = fm.augment_today_workout(
        _RaisingCoach(),
        SimpleNamespace(available=False, exercise="squat", issues=[]))
    assert w.kind == "general" and list(w.exercises) == []
    assert "form.augment_today" in _sites()


def test_pin_augment_fallback():
    issues = _SliceRaisingIssues([FormIssue(checkpoint="depth", label="Depth",
                                            observed="shallow",
                                            fix="go deeper")])
    analysis = SimpleNamespace(available=True, exercise="squat", issues=issues)
    w = fm.augment_today_workout(_CountingCoach(), analysis)
    assert w.kind == "general" and list(w.exercises) == []
    assert "form.augment_fallback" in _sites()


def test_pin_store_get():
    assert _store().get("x") is None
    assert "form.store_get" in _sites()


def test_pin_store_latest():
    assert _store().latest("squat") is None
    assert "form.store_latest" in _sites()


# ---- NOT-recording pins (exclusions) --------------------------------------

def test_pin_no_record_mediapipe_available(monkeypatch):
    monkeypatch.setitem(sys.modules, "mediapipe", None)
    assert fm.mediapipe_available() is False
    assert _sites() == []


def test_pin_no_record_read_image_rgb_inner():
    assert fm._read_image_rgb("/nonexistent/x.jpg") is None
    assert _sites() == []


def test_pin_no_record_read_image_rgb_outer():
    assert fm._read_image_rgb(None) is None
    assert _sites() == []


def test_pin_no_record_joint_angle():
    assert fm.joint_angle("x", "y", "z") is None
    assert _sites() == []


def test_pin_no_record_pose_angles():
    assert fm._pose_angles(None, "squat") == {}
    assert _sites() == []


def test_pin_no_record_count_reps():
    assert fm.count_reps(None, "squat") == 0
    assert _sites() == []


def test_pin_no_record_form_trend_inner():
    assert fm.form_trend(SimpleNamespace(_db=_BoomDB()), "squat") == ""
    assert _sites() == []


def test_pin_no_record_form_trend_outer():
    rows = [{"created_at": 1.0,
             "analysis_json": '{"score": 0.5, "band": "needs work"}'},
            {"created_at": 2.0,
             "analysis_json": '{"score": 0.8, "band": "good"}'}]
    with patch.object(fm, "guard_coaching", side_effect=RuntimeError("guard down")):
        assert fm.form_trend(SimpleNamespace(_db=_RowsDB(rows)), "squat") == ""
    assert _sites() == []


def test_pin_no_record_frames_for_inner(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not a video")
    with patch.object(fm.subprocess, "run", side_effect=RuntimeError("ffmpeg down")):
        assert fm._frames_for(str(video)) == (
            None, "frame extraction failed for this video.")
    assert _sites() == []


def test_pin_no_record_parse_checkpoint_lines():
    assert fm._parse_checkpoint_lines(b"depth: FLAG x", "squat") == ([], [])
    assert _sites() == []


def test_pin_no_record_store_save():
    analysis = FormAnalysis(id="x", exercise="squat", available=True,
                            score=0.5, band="needs work", note="",
                            created_at=0.0)
    assert _store().save(analysis) is False
    assert _sites() == []
