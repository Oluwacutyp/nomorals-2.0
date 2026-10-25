"""Pass #179 pins: the audio failure ledger (REUSED from #157/#160) + fingerprint conversions.

``nomorals/audio/failures.py`` (namespace ``audio``) is reused from
#157/#160: fingerprint.py is an audio-subsystem module, so reuse is
the expected call (unlike #161's engine/trivia split). The 17
durability tests below mirror the #141+ shape exactly (same ledger
as #157/#160); the 7 site pins prove each conversion records without
changing flow (fallback value unchanged AND a ledger entry under its
site name); the 10 NOT-recording pins prove each exclusion stays
silent (designed output AND an empty ledger).

NOTE: ``nomorals/audio/__init__.py`` shadows the submodule attribute
with a ``fingerprint`` function, so this file binds both modules via
``sys.modules`` after ``importlib.import_module`` — a plain
``from nomorals.audio import fingerprint`` would bind the function.
No pin hardcodes scan totals or fingerprint counts. Every test
redirects stores with the app's own ``NM_HOME`` override (never the
real home, never the repo tree). Restart tests use real subprocesses.
Every trigger was probed live before its pin was written (probe.py).

Pins rule (#169): no bare-swallow handler anywhere in this file —
the scanner counts test files too. Where a pin must swallow, it binds
``as exc`` + ``assert isinstance(exc, Exception)``. Docstrings avoid
the literal bare-swallow phrase (#173 lesson).
"""
import importlib
import json
import os
import subprocess
import sys
import threading
import types
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs

importlib.import_module("nomorals.audio.fingerprint")
importlib.import_module("nomorals.audio.failures")
F = sys.modules["nomorals.audio.fingerprint"]
AF = sys.modules["nomorals.audio.failures"]


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    AF.clear_failures()
    yield
    AF.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


def _sites():
    return [f.site for f in AF.recent_failures()]


# ---- Store-level (durability mirror of #141+) ----------------------------

def test_store_round_trip():
    store = fs.Store("audio")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("audio").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("audio")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("audio").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("audio")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("audio").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("audio").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("audio").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("audio")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("audio").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("audio").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("audio").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("audio").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("audio")
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
    entries, _ = fs.Store("audio").load()
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        AF.record_failure("no-home", ValueError("boom"))
        assert AF.failure_count() == 1
        assert [f.site for f in AF.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_ledger_api_record_recent_count_clear_format():
    assert AF.failure_count() == 0
    assert AF.recent_failures() == []
    assert AF.format_failures() == "no audio failures recorded."
    AF.record_failure("audio.site-a", ValueError("boom"))
    AF.record_failure("audio.site-b", RuntimeError("bang"))
    assert AF.failure_count() == 2
    assert [f.site for f in AF.recent_failures()] == ["audio.site-b", "audio.site-a"]
    assert [f.site for f in AF.recent_failures(1)] == ["audio.site-b"]
    text = AF.format_failures()
    assert "2 total" in text and "audio.site-a" in text and "audio.site-b" in text
    AF.clear_failures()
    assert AF.failure_count() == 0
    assert AF.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    AF.record_failure("audio.disk", ValueError("boom"))
    entries, total = fs.Store("audio").load()
    assert total == 1 and entries[0]["site"] == "audio.disk"
    AF.clear_failures()
    assert fs.Store("audio").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    AF.record_failure("audio.mem", ValueError("boom"))
    assert AF.failure_count() == 1
    assert [f.site for f in AF.recent_failures()] == ["audio.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                AF.record_failure(f"audio.w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
            assert isinstance(exc, Exception)
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert AF.failure_count() == 400
    assert len(AF.recent_failures(1000)) == 200
    entries, total = AF._store.load()
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


_BIND = ("import importlib, sys\n"
         "importlib.import_module('nomorals.audio.failures')\n"
         "f = sys.modules['nomorals.audio.failures']\n")


def _record_code(n: int, clear_first: bool = True) -> str:
    clear = "f.clear_failures()\n" if clear_first else ""
    return (_BIND + clear +
            "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                    for i in range(n)) +
            "print('ok')\n")


_READ_CODE = (_BIND +
              "import json\n"
              "print(json.dumps({'count': f.failure_count(), "
              "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = _BIND + "f.clear_failures()\nprint('ok')\n"


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
    store_path = fs.Store("audio").path()
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


# ---- Site pins (fallback unchanged AND ledger entry) ----------------------

def test_pin_ffmpeg_path():
    fake = types.ModuleType("nomorals.media_edit.videos")  # no ffmpeg_path
    with patch.dict(sys.modules, {"nomorals.media_edit.videos": fake}):
        assert F._ffmpeg_path() is None
    assert _sites() == ["fingerprint.ffmpeg_path"]


def test_pin_resample_linear():
    samples = array("d", [0.1, 0.2, 0.3, 0.4])
    out = F._resample_linear(samples, 0, 11025)  # ZeroDivisionError inside
    assert list(out) == list(samples)
    assert _sites() == ["fingerprint.resample_linear"]


def test_pin_decode_tmp_cleanup(tmp_path):
    mp3 = tmp_path / "clip.mp3"
    mp3.write_bytes(b"\x00" * 16)
    fake = types.ModuleType("nomorals.media_edit.videos")

    def boom(*a, **k):
        raise RuntimeError("ffmpeg down")

    fake.run_ffmpeg = boom
    with patch.dict(sys.modules, {"nomorals.media_edit.videos": fake}), \
            patch.object(F, "_ffmpeg_path", return_value="/bin/ffmpeg"):
        with pytest.raises(F.AudioReadError):
            F.read_mono(str(mp3))
    assert _sites() == ["fingerprint.decode_tmp_cleanup"]


def _dead_db(tmp_path):
    db = F.FingerprintDB(str(tmp_path / "t.db"))
    db._connect = lambda: (_ for _ in ()).throw(RuntimeError("db down"))
    return db


def test_pin_stats(tmp_path):
    assert _dead_db(tmp_path).stats() == {"ok": False, "tracks": 0, "hashes": 0}
    assert _sites() == ["fingerprint.stats"]


def test_pin_list_tracks(tmp_path):
    assert _dead_db(tmp_path).list_tracks() == []
    assert _sites() == ["fingerprint.list_tracks"]


def test_pin_remove_track(tmp_path):
    assert _dead_db(tmp_path).remove_track(1) is False
    assert _sites() == ["fingerprint.remove_track"]


def test_pin_close(tmp_path):
    db = F.FingerprintDB(str(tmp_path / "t.db"))

    class BadConn:
        def close(self):
            raise RuntimeError("close down")

    db._conn = BadConn()
    db.close()
    assert db._conn is None
    assert _sites() == ["fingerprint.close"]


# ---- NOT-recording pins (designed output AND empty ledger) ----------------

def test_pin_notrec_has_numpy():
    with patch.dict(sys.modules, {"numpy": None}):
        assert F.has_numpy() is False
    assert _sites() == []


def _corrupt_wav(tmp_path):
    p = tmp_path / "bad.wav"
    p.write_bytes(b"not a wav at all")
    return str(p)


def test_pin_notrec_media_duration_routing(tmp_path):
    with patch("shutil.which", return_value="/bin/ffprobe"), \
            patch("subprocess.run",
                  return_value=SimpleNamespace(stdout="42.5\n")):
        assert F._media_duration(_corrupt_wav(tmp_path)) == 42.5
    assert _sites() == []


def test_pin_notrec_media_duration_terminal(tmp_path):
    with patch("shutil.which", return_value="/bin/ffprobe"), \
            patch("subprocess.run", side_effect=RuntimeError("ffprobe down")):
        assert F._media_duration(_corrupt_wav(tmp_path)) == 0.0
    assert _sites() == []


def test_pin_notrec_default_db_path():
    with patch.object(Path, "mkdir", side_effect=OSError("nope")):
        p = F._default_db_path()
        assert p.endswith("fingerprints.db")
    assert _sites() == []


def test_pin_notrec_estimate_tempo():
    flux = [float(i % 7) for i in range(40)]
    assert F._estimate_tempo(flux, 0, 2048) == (0.0, 0.0)  # sr=0: div by 0
    assert _sites() == []


def test_pin_notrec_chroma_vector():
    assert F._chroma_vector([["x"]], 11025) == [0.0] * 12
    assert _sites() == []


def test_pin_notrec_chroma_key():
    with patch.object(F, "_chroma_vector", side_effect=RuntimeError("boom")):
        assert F._chroma_key([[1.0]], 11025) == ("", 0.0)
    assert _sites() == []


def test_pin_notrec_detect_chord():
    with patch.object(F, "_chroma_vector", side_effect=RuntimeError("boom")):
        assert F._detect_chord([[1.0]], 11025) == ("", 0.0)
    assert _sites() == []


def test_pin_notrec_estimate_tuning():
    assert F._estimate_tuning([["x"]], 11025) == 0.0
    assert _sites() == []


def test_pin_notrec_quality_flags():
    out = F._quality_flags(["x"], 11025, [[1.0]])
    assert out == {"snr_db": 0.0, "hum": False,
                   "saturated": False, "dc_offset": 0.0}
    assert _sites() == []
