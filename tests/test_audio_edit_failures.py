"""Pass #196 pins: the audio failure ledger (REUSED from #157) + edit.py
conversions.

The audio ledger already exists (``nomorals/audio/failures.py``,
namespace ``audio``, created #157, reused #160/#179); this pass reuses
it unchanged. The 17 durability tests below mirror the #141+ shape for
this ledger; the 2 site pins prove each conversion records without
changing flow (fallback behavior unchanged AND a ledger entry under
its site name); the 12 NOT-recording pins prove each exclusion stays
silent on its failure path. No pin hardcodes scan totals or edit.py
counts. Every test redirects stores with the app's own ``NM_HOME``
override (never the real home, never the repo tree). Restart tests use
real subprocesses. Every trigger was probed live before its pin was
written.
"""
import json
import os
import subprocess
import sys
import threading
import types
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.audio import edit
from nomorals.audio import failures as af
from nomorals.core import failure_store as fs


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    af.clear_failures()
    yield
    af.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


def _sites():
    return [f.site for f in af.recent_failures()]


def _boom(*args, **kwargs):
    raise RuntimeError("boom")


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
        except Exception as exc:
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


# ---- Ledger-level (in-process) -------------------------------------------

def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        af.record_failure("no-home", ValueError("boom"))
        assert af.failure_count() == 1
        assert [f.site for f in af.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_ledger_api_record_recent_count_clear_format():
    assert af.failure_count() == 0
    assert af.recent_failures() == []
    assert af.format_failures() == "no audio failures recorded."
    af.record_failure("audio.site-a", ValueError("boom"))
    af.record_failure("audio.site-b", RuntimeError("bang"))
    assert af.failure_count() == 2
    assert [f.site for f in af.recent_failures()] == ["audio.site-b", "audio.site-a"]
    assert [f.site for f in af.recent_failures(1)] == ["audio.site-b"]
    text = af.format_failures()
    assert "2 total" in text and "audio.site-a" in text and "audio.site-b" in text
    af.clear_failures()
    assert af.failure_count() == 0
    assert af.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    af.record_failure("audio.disk", ValueError("boom"))
    entries, total = fs.Store("audio").load()
    assert total == 1 and entries[0]["site"] == "audio.disk"
    af.clear_failures()
    assert fs.Store("audio").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    af.record_failure("audio.mem", ValueError("boom"))
    assert af.failure_count() == 1
    assert [f.site for f in af.recent_failures()] == ["audio.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                af.record_failure(f"audio.w{k}-{i}", ValueError("boom"))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert af.failure_count() == 400
    assert len(af.recent_failures(1000)) == 200
    entries, total = af._store.load()
    assert len(entries) == 200
    assert total == 400


def test_ledger_ring_cap_drops_oldest_silently():
    for i in range(250):
        af.record_failure(f"audio.ring{i}", ValueError("boom"))
    assert af.failure_count() == 250
    ring = af.recent_failures(1000)
    assert len(ring) == 200
    assert ring[0].site == "audio.ring249"
    assert ring[-1].site == "audio.ring50"


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
        "from nomorals.audio import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.audio import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.audio import failures as f\n"
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
    # Corrupt via a subprocess (the store path resolves against NM_HOME
    # at call time, and the subprocess shares this test's home).
    corrupt = (
        "from pathlib import Path\n"
        "from nomorals.core import failure_store as fs\n"
        "fs.Store('audio').path().write_text('{not json', encoding='utf-8')\n"
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

def _boom_unlink(self, missing_ok=False):
    raise OSError("pin unlink failure")


def test_pin_splice_cleanup():
    calls = {"n": 0}

    def fake_trim(audio, s, e, out_dir=None, suffix="", ext=".wav"):
        calls["n"] += 1
        return {"output": f"/tmp/pin-part{calls['n']}.wav"}

    def fake_concat(parts, out_dir=None, suffix="", ext=".wav"):
        return {"output": "/tmp/pin-spliced.wav"}

    with patch.object(edit, "_ffmpeg", lambda: "ffmpeg"), \
            patch("nomorals.media_edit.videos.trim", side_effect=fake_trim), \
            patch("nomorals.media_edit.videos.concat", side_effect=fake_concat), \
            patch.object(Path, "unlink", _boom_unlink):
        res = edit._splice("voice.wav", [(0.0, 1.0), (2.0, 3.0)])
    assert res.get("ok") is True
    assert res.get("mode") == "trim+concat"
    assert "edit.splice_cleanup" in _sites()


def test_pin_concat_cleanup(tmp_path):
    wav = tmp_path / "voice.wav"
    wav.write_bytes(b"")
    calls = {"n": 0}

    def fake_trim(audio, s, e, out_dir=None, suffix="", ext=".wav"):
        calls["n"] += 1
        return {"output": f"/tmp/pin-keep{calls['n']}.wav"}

    def fake_concat(parts, out_dir=None, suffix="", ext=".wav"):
        return {"output": "/tmp/pin-edited.wav"}

    t = edit.EditableTranscript(audio_path=str(wav))
    with patch.object(edit, "_audio_duration", lambda p: 10.0), \
            patch("nomorals.media_edit.videos.trim", side_effect=fake_trim), \
            patch("nomorals.media_edit.videos.concat", side_effect=fake_concat), \
            patch.object(Path, "unlink", _boom_unlink):
        res = edit.apply_edits(t, [edit.Edit.delete(1, 2),
                                   edit.Edit.delete(5, 6)])
    assert res.get("ok") is True
    assert "edit.concat_cleanup" in _sites()


# ---- NOT-recording pins (exclusions stay silent on failure) --------------

def test_pin_no_record_find_fillers():
    with patch.object(edit, "fillers_for", _boom):
        assert edit.find_fillers([{"text": "hello"}]) == []
    assert af.recent_failures() == []


class _BadFloat:
    def __float__(self):
        raise RuntimeError("pin boom")


def test_pin_no_record_merge_ranges():
    assert edit._merge_ranges([(_BadFloat(), 1.0)]) == []
    assert af.recent_failures() == []


def test_pin_no_record_ffmpeg():
    with patch("nomorals.media_edit.videos.ffmpeg_path", _boom):
        assert edit._ffmpeg() is None
    assert af.recent_failures() == []


class _RaisingWord:
    def __init__(self):
        self._text = "um"

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        raise RuntimeError("pin boom")


def test_pin_no_record_correct_text():
    t = edit.EditableTranscript(words=[_RaisingWord()])
    assert t.correct_text("um", "uh") == 0
    assert af.recent_failures() == []


class _RaisingEnd:
    @property
    def end(self):
        raise RuntimeError("pin boom")

    start = 0.0


def test_pin_no_record_words_in():
    t = edit.EditableTranscript(words=[_RaisingEnd()])
    assert t.words_in(0, 1) == []
    assert af.recent_failures() == []


def test_pin_no_record_transcript_edit():
    assert edit.transcript_edit("/tmp/does-not-exist-196.wav") is None
    assert af.recent_failures() == []


def test_pin_no_record_find_silences():
    with patch("nomorals.audio.fingerprint.read_mono", _boom):
        assert edit.find_silences("x.wav") == []
    assert af.recent_failures() == []


def test_pin_no_record_sample_rate():
    with patch("nomorals.media_edit.videos.video_probe", _boom):
        assert edit._audio_sample_rate("x.wav") == 0
    assert af.recent_failures() == []


def test_pin_no_record_silence_wav():
    with patch("nomorals.media_edit.videos.run_ffmpeg", _boom):
        assert edit._silence_wav(1.0, 24000, "/tmp") is None
    assert af.recent_failures() == []


def test_pin_no_record_normalize_wav():
    with patch("nomorals.media_edit.videos.run_ffmpeg", _boom):
        assert edit._normalize_wav("/tmp/x.wav", 24000, "/tmp") is None
    assert af.recent_failures() == []


def test_pin_no_record_list_fx(monkeypatch):
    bare = types.ModuleType("nomorals.audio.dsp")
    monkeypatch.setitem(sys.modules, "nomorals.audio.dsp", bare)
    assert edit.list_fx() == "effects unavailable"
    assert af.recent_failures() == []


def test_pin_no_record_nl_intent():
    with patch.object(edit, "_detect_lang", _boom):
        assert edit.nl_audio_intent("hello") is None
    assert af.recent_failures() == []
