"""Tail-76 pins — nomorals/media/dj_engine.py silent-failure sites.

Module contract (its own docstring): "Never raises out of the
public functions — analysis degrades honestly (``bpm=None``
means "couldn't detect"...)".  Every pin asserts the CURRENT
behaviour at one census site: the guarded step degrades to
the module's documented no-data / unknown / neutral value.
The six sites record nothing — a caplog pin proves the
_read_mono failure surfaces zero log records (its neighbours
detect_bpm / detect_key DO log, and are not census sites).
"""
import logging
import math
import struct
import wave
from array import array

from nomorals.media import dj_engine as D


def test_read_mono_missing_file_returns_empty(caplog):
    # site: _read_mono except -> (array("d"), _SR)
    with caplog.at_level(logging.DEBUG,
                         logger="nomorals.media.dj_engine"):
        samples, sr = D._read_mono("/nonexistent/definitely-missing.wav")
    assert len(samples) == 0
    assert sr == D._SR
    assert caplog.records == []  # silent site: nothing recorded
    # the sole consumer's no-data path: honest empty analysis
    a = D.analyze_track("/nonexistent/definitely-missing.wav")
    assert a.bpm is None
    assert a.duration_s == 0.0
    assert a.bpm_source == ""


def test_goertzel_mag_bad_samples_returns_zero():
    # site: _goertzel_mag except -> 0.0
    sine = array("d", [math.sin(2 * math.pi * 440 * i / 22050)
                       for i in range(4096)])
    assert D._goertzel_mag(sine, 0, 4096, 440.0, 22050) > 0  # control
    assert D._goertzel_mag("abcd", 0, 4, 440.0, 22050) == 0.0


def test_track_energy_failure_returns_neutral_default():
    # site: track_energy except -> 0.5
    # 0.5 is the documented neutral: the TrackAnalysis default
    # AND the empty-input return — not a measured claim.
    assert D.track_energy(array("d")) == 0.5
    assert D.track_energy(["a", "b"]) == 0.5


def test_analyze_track_resample_failure_keeps_native_samples(
        tmp_path, monkeypatch):
    # site: analyze_track resample guard except -> pass
    # (optional consistent-rate copy; failure keeps the native
    # samples + true sample rate and detection proceeds).
    path = tmp_path / "t8k.wav"
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(8000)
        f.writeframes(b"".join(
            struct.pack("<h", int(12000 * math.sin(
                2 * math.pi * 220 * i / 8000)))
            for i in range(8000 * 5)))
    import nomorals.media.vocal_lite as vl

    def _boom(*a, **k):
        raise RuntimeError("no resample")

    monkeypatch.setattr(vl, "_resample_linear", _boom)
    a = D.analyze_track(str(path))
    assert a.duration_s == 5.0  # analysis completed unresampled
    assert a.bpm_source in ("detected", "unknown")


def test_camelot_code_bad_key_returns_unknown():
    # site: camelot_code except -> ""
    assert D.camelot_code("C", "major") == "8B"  # control
    assert D.camelot_code(123, "major") == ""
    # "" is also the documented unknown-key value
    assert D.camelot_code("H", "major") == ""


def test_camelot_parts_unparseable_returns_none():
    # site: _camelot_parts except -> None
    assert D._camelot_parts("8A") == (8, "A")  # control
    assert D._camelot_parts("X") is None
    assert D._camelot_parts(None) is None
    # consumer contract: unknown code -> no path, neutral score
    assert D.harmonic_path("X", "8A") == []
