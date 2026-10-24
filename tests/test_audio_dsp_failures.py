"""Pass #157 pins: the audio failure ledger (NEW) + dsp conversions.

No audio ledger existed before this pass: ``nomorals/audio/failures.py``
(namespace ``audio``) is created here, durability-wired through the shared
``nomorals/core/failure_store.py`` exactly like the #141–#156 ledgers. The
17 durability tests below mirror #156's shape exactly; the 25 site pins
prove each conversion records without changing flow (fallback value
unchanged AND a ledger entry under its site name). There are 0
exclusions. No pin hardcodes scan totals or dsp counts.
Every test redirects stores with the app's own ``NM_HOME`` override (never
the real home, never the repo tree). Restart tests use real subprocesses.
Every trigger was probed live before its pin was written.
"""
import json
import os
import subprocess
import sys
import threading
from array import array
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.audio import dsp
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


def _samples():
    return array("d", [0.1, -0.2, 0.3, 0.0, 0.5, -0.4, 0.2])


SR = 22050


# ---- Store-level (durability mirror of #156) -----------------------------

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
        except Exception as exc:  # noqa: BLE001 — test must surface it
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


# ---- Converted-site behavior: fallback unchanged + recorded -------------

def _break_array_once(monkeypatch):
    """Patch dsp.array so its NEXT call raises, then delegates.

    Lets a pin trip a handler whose only triggerable input is the
    samples buffer itself (remove_dc has no scalar parameter): the
    try-body array call raises, the fallback array call succeeds.
    """
    real = dsp.array
    state = {"n": 0}

    def one_shot(*args, **kwargs):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("boom")
        return real(*args, **kwargs)

    monkeypatch.setattr(dsp, "array", one_shot)


def test_converted_list_presets_records_and_returns_unavailable(monkeypatch):
    class _BoomChain:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(dsp, "EffectChain", _BoomChain)
    assert dsp.list_presets() == "presets unavailable"
    assert _sites() == ["dsp.list_presets"]


def test_converted_normalize_peak_records_and_returns_input():
    inp = _samples()
    assert list(dsp.normalize_peak(inp, "bad")) == list(inp)
    assert _sites() == ["dsp.normalize_peak"]


def test_converted_remove_dc_records_and_returns_input(monkeypatch):
    _break_array_once(monkeypatch)
    inp = _samples()
    assert list(dsp.remove_dc(inp)) == list(inp)
    assert _sites() == ["dsp.remove_dc"]


def test_converted_trim_silence_records_and_returns_input():
    inp = _samples()
    assert list(dsp.trim_silence(inp, "bad")) == list(inp)
    assert _sites() == ["dsp.trim_silence"]


def test_converted_fade_edges_records_and_returns_input():
    inp = _samples()
    assert list(dsp.fade_edges(inp, "bad")) == list(inp)
    assert _sites() == ["dsp.fade_edges"]


def test_converted_biquad_records_and_returns_input():
    inp = _samples()
    assert list(dsp.biquad(inp, "bad", "notch", 1000.0)) == list(inp)
    assert _sites() == ["dsp.biquad"]


def test_converted_telephone_records_and_returns_input(monkeypatch):
    monkeypatch.setattr(dsp, "highpass", _boom)
    inp = _samples()
    assert list(dsp.telephone(inp, SR)) == list(inp)
    assert _sites() == ["dsp.telephone"]


def test_converted_gain_db_records_and_returns_input():
    inp = _samples()
    assert list(dsp.gain_db(inp, SR, "bad")) == list(inp)
    assert _sites() == ["dsp.gain_db"]


def test_converted_bitcrush_records_and_returns_input():
    inp = _samples()
    assert list(dsp.bitcrush(inp, SR, bits="bad")) == list(inp)
    assert _sites() == ["dsp.bitcrush"]


def test_converted_distortion_records_and_returns_input():
    inp = _samples()
    assert list(dsp.distortion(inp, SR, tone="bad")) == list(inp)
    assert _sites() == ["dsp.distortion"]


def test_converted_chorus_records_and_returns_input():
    inp = _samples()
    assert list(dsp.chorus(inp, SR, rate_hz="bad")) == list(inp)
    assert _sites() == ["dsp.chorus"]


def test_converted_tremolo_records_and_returns_input():
    inp = _samples()
    assert list(dsp.tremolo(inp, SR, depth="bad")) == list(inp)
    assert _sites() == ["dsp.tremolo"]


def test_converted_vibrato_records_and_returns_input():
    inp = _samples()
    assert list(dsp.vibrato(inp, SR, depth_ms="bad")) == list(inp)
    assert _sites() == ["dsp.vibrato"]


def test_converted_phaser_records_and_returns_input():
    inp = _samples()
    assert list(dsp.phaser(inp, SR, stages="bad")) == list(inp)
    assert _sites() == ["dsp.phaser"]


def test_converted_delay_records_and_returns_input():
    inp = _samples()
    assert list(dsp.delay(inp, SR, delay_ms="bad")) == list(inp)
    assert _sites() == ["dsp.delay"]


def test_converted_autolevel_records_and_returns_input():
    inp = _samples()
    assert list(dsp.autolevel(inp, SR, target_rms_db="bad")) == list(inp)
    assert _sites() == ["dsp.autolevel"]


def test_converted_duck_under_records_and_returns_bed():
    bed = _samples()
    voice = _samples()
    assert list(dsp.duck_under(bed, voice, SR, duck_db="bad")) == list(bed)
    assert _sites() == ["dsp.duck_under"]


def test_converted_dehum_records_and_returns_input():
    inp = _samples()
    assert list(dsp.dehum(inp, SR, freqs=("bad",))) == list(inp)
    assert _sites() == ["dsp.dehum"]


def test_converted_compressor_records_and_returns_input():
    inp = _samples()
    assert list(dsp.compressor(inp, SR, threshold_db="bad")) == list(inp)
    assert _sites() == ["dsp.compressor"]


def test_converted_soft_limiter_records_and_returns_input():
    inp = _samples()
    assert list(dsp.soft_limiter(inp, ceiling="bad")) == list(inp)
    assert _sites() == ["dsp.soft_limiter"]


def test_converted_convolve_fft_records_on_numpy_failure():
    # Bad kernel element: the numpy FFT path raises (ValueError on the
    # float64 conversion), the record fires, then the direct-convolution
    # fallback itself raises on the same bad element. The record is the
    # assertion — the fallback's own TypeError is expected either way
    # (and on a numpy-less box the import failure takes the same path).
    x = array("d", [0.5, 0.25, 0.125])
    with pytest.raises(TypeError):
        dsp._convolve_fft(x, [1.0, "bad"])
    assert _sites() == ["dsp.convolve_fft"]


def test_converted_echo_records_and_returns_input():
    inp = _samples()
    assert list(dsp.echo(inp, SR, delay_ms="bad")) == list(inp)
    assert _sites() == ["dsp.echo"]


def test_converted_varispeed_records_and_returns_input():
    inp = _samples()
    assert list(dsp._varispeed(inp, "bad")) == list(inp)
    assert _sites() == ["dsp.varispeed"]


def test_converted_pitch_shift_records_and_returns_input():
    inp = _samples()
    assert list(dsp.pitch_shift(inp, SR, "bad")) == list(inp)
    assert _sites() == ["dsp.pitch_shift"]


def test_converted_mix_under_records_and_returns_bed():
    bed = _samples()
    top = _samples()
    assert list(dsp.mix_under(bed, top, gain="bad")) == list(bed)
    assert _sites() == ["dsp.mix_under"]
