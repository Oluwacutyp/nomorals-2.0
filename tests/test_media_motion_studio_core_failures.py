"""Pass #205 pins: media-ledger durability (reused by motion_studio) +
the motion_studio convert and the NOT-recording exclusions.

The motion_studio convert (site 368) records through the REUSED media
ledger (``nomorals.media.failures``, namespace ``media``) — see
SURVEY.md for the layering arithmetic (motion_studio/_core.py resolves
to the media package mapping L4, so a same-package media-ledger import
is layering-safe). Site name: ``motion_studio._core.record_ledger``.

Every test runs with NM_HOME redirected at a scratch dir and the
ledger's in-memory state reset, so no test touches the real home.
No test in this file uses a bare handler: probes assert on return
values and on the ledger's record set.
"""

import json
import os
import subprocess
import sys
import threading

import pytest

from nomorals.media import failures as led
from nomorals.core import failure_store as store_mod
from nomorals.media.motion_studio import _core as core


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch, tmp_path):
    """Redirect NM_HOME and reset ledger memory before every test."""
    monkeypatch.setenv("NM_HOME", str(tmp_path))
    led._ledger.clear()
    led._total = 0
    led._loaded = False
    yield
    led._ledger.clear()
    led._total = 0
    led._loaded = False


def _store(ns="media"):
    return store_mod.Store(ns)


# ── durability (17, mirroring #141+) ─────────────────────────────────


def test_round_trip():
    led.record_failure("motion_studio.probe", ValueError("boom"))
    recs = led.recent_failures(5)
    assert len(recs) == 1
    assert recs[0].site == "motion_studio.probe"
    assert recs[0].exc_type == "ValueError"
    assert recs[0].message == "boom"
    assert led.failure_count() == 1


def test_store_cap_250_to_200_total_preserved():
    st = _store("cap-probe")
    rows = [{"ts": float(i), "site": f"s{i}", "exc_type": "E",
             "message": "m"} for i in range(250)]
    assert st.save(rows, 250) is True
    entries, total = st.load()
    assert len(entries) == 200
    assert total == 250
    assert entries[0]["site"] == "s50"
    assert entries[-1]["site"] == "s249"


def test_ledger_cap_250_200_total_preserved():
    for i in range(250):
        led.record_failure(f"motion_studio.bulk{i}", RuntimeError("x"))
    assert len(led.recent_failures(500)) == 200
    assert led.failure_count() == 250


def test_clear_gone():
    led.record_failure("motion_studio.probe", ValueError("x"))
    assert led.failure_count() == 1
    led.clear_failures()
    assert led.recent_failures() == []
    assert led.failure_count() == 0
    assert not _store("media").path().exists()


def test_namespace_isolation():
    _store("ns-a").save([{"ts": 1.0, "site": "a", "exc_type": "E",
                          "message": "m"}], 1)
    entries, total = _store("ns-b").load()
    assert entries == [] and total == 0
    assert _store("ns-a").path() != _store("ns-b").path()


def test_corrupt_recovery_store_level():
    st = _store("corrupt")
    st.path().parent.mkdir(parents=True, exist_ok=True)
    st.path().write_text("{not json", encoding="utf-8")
    entries, total = st.load()
    assert entries == [] and total == 0


def test_save_failure_ancestor_is_file(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("file, not dir", encoding="utf-8")
    st = store_mod.Store("x")
    before = store_mod.write_failures()
    import nomorals.core.failure_store as fsm
    old = os.environ.get("NM_HOME")
    os.environ["NM_HOME"] = str(blocker / "sub")
    try:
        assert st.save([{"ts": 1.0, "site": "s", "exc_type": "E",
                         "message": "m"}], 1) is False
    finally:
        if old is None:
            del os.environ["NM_HOME"]
        else:
            os.environ["NM_HOME"] = old
    assert store_mod.write_failures() == before + 1


def test_save_failure_failures_dir_is_file(tmp_path, monkeypatch):
    (tmp_path / "failures").write_text("file, not dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(tmp_path))
    st = _store("x")
    assert st.save([{"ts": 1.0, "site": "s", "exc_type": "E",
                     "message": "m"}], 1) is False


def test_save_failure_counter_bumps(tmp_path, monkeypatch):
    (tmp_path / "failures").write_text("file, not dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(tmp_path))
    before = store_mod.write_failures()
    assert _store("x").save([{"ts": 1.0, "site": "s", "exc_type": "E",
                              "message": "m"}], 1) is False
    assert store_mod.write_failures() == before + 1
    assert store_mod.last_error() is not None


def test_concurrent_saves():
    st = _store("conc")
    errors = []

    def worker(n):
        ok = st.save([{"ts": float(n), "site": f"w{n}",
                       "exc_type": "E", "message": "m"}], n + 1)
        if not ok:
            errors.append(n)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    entries, total = st.load()
    assert isinstance(entries, list) and total >= 1
    for row in entries:
        assert set(row) == {"ts", "site", "exc_type", "message"}


def test_disk_hit():
    led.record_failure("motion_studio.disk", ValueError("on-disk"))
    text = _store("media").path().read_text(encoding="utf-8")
    assert "motion_studio.disk" in text
    assert "on-disk" in text


def test_concurrent_records():
    errors = []

    def worker(n):
        for i in range(25):
            led.record_failure(f"motion_studio.t{n}-{i}", ValueError("c"))

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert led.failure_count() == 200


def test_subprocess_restart(tmp_path):
    code_write = (
        "import sys; sys.path.insert(0, '.');"
        "from nomorals.media.failures import record_failure;"
        "record_failure('motion_studio.sub', ValueError('persisted'));"
        "print('wrote')"
    )
    code_read = (
        "import sys; sys.path.insert(0, '.');"
        "from nomorals.media.failures import recent_failures, failure_count;"
        "recs = recent_failures(5);"
        "print(recs[0].site if recs else 'EMPTY');"
        "print(failure_count())"
    )
    env = dict(os.environ, NM_HOME=str(tmp_path))
    tree = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cwd = tree if os.path.isdir(os.path.join(tree, "nomorals")) else "."
    w = subprocess.run([sys.executable, "-c", code_write], cwd=cwd,
                       env=env, capture_output=True, text=True, timeout=120)
    assert w.returncode == 0, w.stderr
    r = subprocess.run([sys.executable, "-c", code_read], cwd=cwd,
                       env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.strip().splitlines()
    assert lines[0] == "motion_studio.sub"
    assert lines[1] == "1"


def test_broken_home_ledger_level(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("file, not dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker / "sub"))
    # Disk writes fail, but the in-memory ledger still records.
    led.record_failure("motion_studio.mem", ValueError("mem-only"))
    assert led.failure_count() == 1
    assert led.recent_failures(1)[0].site == "motion_studio.mem"


def test_format_failures():
    assert led.format_failures() == "no recorded media failures"
    led.record_failure("motion_studio.fmt", ValueError("hello"))
    out = led.format_failures()
    assert "motion_studio.fmt" in out
    assert "ValueError" in out and "hello" in out


def test_clear_gone_on_reload_subprocess(tmp_path):
    code_write = (
        "import sys; sys.path.insert(0, '.');"
        "from nomorals.media.failures import record_failure, clear_failures;"
        "record_failure('motion_studio.tmp', ValueError('gone'));"
        "clear_failures(); print('cleared')"
    )
    code_read = (
        "import sys; sys.path.insert(0, '.');"
        "from nomorals.media.failures import recent_failures, failure_count;"
        "print(len(recent_failures())); print(failure_count())"
    )
    env = dict(os.environ, NM_HOME=str(tmp_path))
    tree = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cwd = tree if os.path.isdir(os.path.join(tree, "nomorals")) else "."
    w = subprocess.run([sys.executable, "-c", code_write], cwd=cwd,
                       env=env, capture_output=True, text=True, timeout=120)
    assert w.returncode == 0, w.stderr
    r = subprocess.run([sys.executable, "-c", code_read], cwd=cwd,
                       env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines() == ["0", "0"]


def test_total_preserved_across_restart(tmp_path):
    code_write = (
        "import sys; sys.path.insert(0, '.');"
        "from nomorals.media.failures import record_failure;"
        "[record_failure(f'motion_studio.t{i}', ValueError('v'))"
        " for i in range(3)]; print('wrote3')"
    )
    code_read = (
        "import sys; sys.path.insert(0, '.');"
        "from nomorals.media.failures import failure_count;"
        "print(failure_count())"
    )
    env = dict(os.environ, NM_HOME=str(tmp_path))
    tree = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cwd = tree if os.path.isdir(os.path.join(tree, "nomorals")) else "."
    w = subprocess.run([sys.executable, "-c", code_write], cwd=cwd,
                       env=env, capture_output=True, text=True, timeout=120)
    assert w.returncode == 0, w.stderr
    r = subprocess.run([sys.executable, "-c", code_read], cwd=cwd,
                       env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "3"


# ── behavior pin for the one convert ─────────────────────────────────


def test_convert_368_records_and_never_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "LEDGER_PATH", tmp_path / "ledger.json")

    def _boom():
        raise RuntimeError("migration exploded")

    monkeypatch.setattr(core, "_ensure_archive", _boom)
    core.record_ledger({"kind": "probe-368"})
    recs = led.recent_failures(5)
    assert any(r.site == "motion_studio._core.record_ledger"
               for r in recs)
    assert any(r.exc_type == "RuntimeError" for r in recs)
    win = json.loads((tmp_path / "ledger.json").read_text())
    assert any(w.get("kind") == "probe-368" for w in win)


# ── NOT-recording pins (one per exclusion) ───────────────────────────


def test_excl_180_ease_directed_failure_no_record(monkeypatch):
    from nomorals.media.directed import motion_score

    def _boom(name, t):
        raise RuntimeError("directed down")

    monkeypatch.setattr(motion_score, "ease_value", _boom)
    assert core.ease("smooth", 0.5) == 0.5
    assert led.failure_count() == 0


def test_excl_213_find_font_candidates_no_record():
    font = core.find_font(137, "definitely-not-a-font-xyz.ttf")
    assert font is not None
    assert led.failure_count() == 0


def test_excl_226_text_size_fallback_no_record():
    class _Draw:
        def textbbox(self, *args, **kwargs):
            raise RuntimeError("bbox down")

        def textlength(self, text, font=None):
            return 42.0

    class _Font:
        size = 24

    assert core.text_size(_Draw(), "hi", _Font()) == (42.0, 24)
    assert led.failure_count() == 0


def test_excl_265_window_corrupt_no_record(monkeypatch, tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text("{corrupt", encoding="utf-8")
    monkeypatch.setattr(core, "LEDGER_PATH", ledger)
    assert core._read_window_file() == []
    assert led.failure_count() == 0


def test_excl_283_archive_torn_line_no_record(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "LEDGER_PATH", tmp_path / "ledger.json")
    (tmp_path / "ledger.archive.jsonl").write_text(
        "{torn\n{\"kind\": \"kept\"}\n", encoding="utf-8")
    assert core._read_archive() == [{"kind": "kept"}]
    assert led.failure_count() == 0


def test_excl_287_archive_unreadable_no_record(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "LEDGER_PATH", tmp_path / "ledger.json")
    (tmp_path / "ledger.archive.jsonl").mkdir()
    assert core._read_archive() is None
    assert led.failure_count() == 0


def test_excl_296_meta_corrupt_no_record(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "LEDGER_PATH", tmp_path / "ledger.json")
    (tmp_path / "ledger.meta.json").write_text("{corrupt", encoding="utf-8")
    assert core._read_meta() == {}
    assert led.failure_count() == 0


def test_excl_397_read_ledger_migration_failure_no_record(monkeypatch, tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps([{"kind": "win"}]), encoding="utf-8")
    monkeypatch.setattr(core, "LEDGER_PATH", ledger)

    def _boom():
        raise RuntimeError("migrate down")

    monkeypatch.setattr(core, "_ensure_archive", _boom)
    assert core.read_ledger() == [{"kind": "win"}]
    assert led.failure_count() == 0


def test_excl_401_read_ledger_archive_raise_no_record(monkeypatch, tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps([{"kind": "win"}]), encoding="utf-8")
    monkeypatch.setattr(core, "LEDGER_PATH", ledger)

    def _boom():
        raise RuntimeError("archive read down")

    monkeypatch.setattr(core, "_read_archive", _boom)
    assert core.read_ledger() == [{"kind": "win"}]
    assert led.failure_count() == 0


def test_excl_494_encode_kill_failure_no_record(monkeypatch, tmp_path):
    import numpy as np

    class _Stdin:
        def write(self, data):
            raise BrokenPipeError("ffmpeg died")

        def close(self):
            pass

    class _Stderr:
        def read(self):
            return b"fake"

    class _FakeProc:
        def __init__(self, *args, **kwargs):
            self.stdin = _Stdin()
            self.stderr = _Stderr()
            self.returncode = 1

        def wait(self, timeout=None):
            return None

        def kill(self):
            raise RuntimeError("kill down")

    monkeypatch.setattr(core.subprocess, "Popen", _FakeProc)
    monkeypatch.setattr(core, "ffmpeg_path", lambda: "ffmpeg")
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(core.MotionStudioError):
        core.encode_frames([frame], tmp_path / "out.mp4", 2, 2, 24.0)
    assert led.failure_count() == 0


def test_excl_538_decode_unlink_failure_no_record(monkeypatch, tmp_path):
    import wave as _wave

    def _fake_run(cmd, timeout=None):
        out = cmd[-1]
        with _wave.open(out, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(b"\x00\x00" * 100)

    def _boom(path):
        raise OSError("unlink down")

    monkeypatch.setattr(core, "run_ffmpeg", _fake_run)
    monkeypatch.setattr(os, "unlink", _boom)
    src = tmp_path / "in.mp3"
    src.write_bytes(b"junk")
    data = core.decode_audio_mono(src)
    assert len(data) == 100
    assert led.failure_count() == 0


def test_excl_545_probe_failure_no_record(monkeypatch):
    def _boom(path):
        raise RuntimeError("probe down")

    monkeypatch.setattr(core, "video_probe", _boom)
    assert core.probe_duration("nope.mp4") == 0.0
    assert led.failure_count() == 0
