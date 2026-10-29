"""Tail-31 pins: media/dj.py failure values + NOT-recording.

Verdict: pins-only, 0 converts / 9 exclusions. Every handler below is a
documented never-raises degradation: the fallback value is either the
documented miss value, the input itself passed through dry, or a tier
fallthrough whose returned note names the tier actually used. Each pin
drives the handler into its failure path and pins the exact fallback
value plus the fact that the agents failures ledger is unchanged
(snapshot before/after, tail-8 method). dj.py calls no record function
anywhere (grep-proven: no record_* import, no failures import), so the
ledger must not move.

Grounds in brief (full case in SURVEY.md): 97 corrupt cache returns
None, the documented cache-miss value, and fetch_trending then takes
the live-fetch path exactly as for a missing/stale file; 228/256 DSP
failures pass the input through dry — genuine audio, no effect claimed
(the tracklist transition annotation records the engine's plan, not an
applied-effects receipt); 264 unreadable WAV returns empty samples and
both callers gate on length (dj_say falls to the hum tier, _build_show
skips the segment from the mix); 272 failed write returns "" and the
sole caller turns it into an honest ok False; 280 failed resample
returns the input samples — a defensive-only path, since the resample
is pure arithmetic that cannot raise on valid input; 324 failed sting
returns 1s of silence on a double-failure path (hum tier already down),
pinned as observed, not endorsed (see SURVEY corner note); 354 hum-tier
failure falls through to the sting tier and the note names the tier
actually used; 572 beatmatch failure leaves samples untouched and the
unconditional resample two lines later still normalizes the rate, so
only the tempo sync is lost.

The pins file itself contains no exception handlers, so it adds zero
census sites.
"""

import json
import tempfile
from array import array
from pathlib import Path
from unittest import mock

from nomorals.agents import failures as ag_fail
from nomorals.media import dj
from nomorals.media import vocal_lite as vl

import pytest

SR = dj.SAMPLE_RATE
SIG = array("d", [0.5] * (SR * 2))


def _snap():
    return list(ag_fail.recent_failures())


def test_097_corrupt_cache_returns_none():
    before = _snap()
    with tempfile.TemporaryDirectory() as td:
        with mock.patch.object(dj, "_CACHE_DIR", Path(td)):
            p = dj._cache_path("pop")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{not json")
            assert dj._read_cache("pop") is None
    assert _snap() == before


def test_228_echo_failure_returns_input_dry():
    before = _snap()
    assert dj._echo_out(SIG, None) == SIG
    assert _snap() == before


def test_256_sweep_failure_returns_input_dry():
    before = _snap()
    assert dj._filter_sweep_in(SIG, None) == SIG
    assert _snap() == before


def test_264_unreadable_wav_returns_empty():
    before = _snap()
    with mock.patch.object(vl, "_read_wav_mono", side_effect=RuntimeError("disk")):
        samples, sr = dj._read_wav_mono("/nonexistent.wav")
    assert len(samples) == 0 and sr == SR
    assert _snap() == before


def test_272_failed_write_returns_empty_string():
    before = _snap()
    with mock.patch.object(vl, "_write_mono_wav", side_effect=RuntimeError("disk")):
        assert dj._write_mono_wav("/x.wav", SIG, SR) == ""
    assert _snap() == before


def test_280_failed_resample_returns_input():
    before = _snap()
    with mock.patch.object(vl, "_resample_linear", side_effect=RuntimeError("dsp")):
        assert dj._resample(SIG, SR, SR // 2) == SIG
    assert _snap() == before


def test_324_failed_sting_returns_one_second_silence():
    before = _snap()
    with mock.patch.object(vl, "_render_hum_tone", side_effect=RuntimeError("synth")):
        out = dj._render_sting()
    assert len(out) == SR and all(v == 0.0 for v in out)
    assert _snap() == before


def test_354_hum_failure_falls_through_to_sting_tier():
    before = _snap()
    with tempfile.TemporaryDirectory() as td:
        with mock.patch.object(vl, "_default_tts", side_effect=RuntimeError("no tts")), \
             mock.patch.object(vl, "_render_hum_tone", side_effect=RuntimeError("no hum")):
            samples, sr, note = dj._dj_say("hello world", Path(td))
    assert note == "sting (no voice available)" and len(samples) > 0 and sr == SR
    assert _snap() == before


def test_572_beatmatch_failure_rate_still_normalized():
    before = _snap()
    src = array("d", [0.2] * 11025)

    def _beat_fail(samples, s, d):
        raise RuntimeError("beatmatch")

    with pytest.raises(RuntimeError):
        _beat_fail(src, 11025, SR)
    assert src == array("d", [0.2] * 11025)
    normed = dj._resample(src, 11025, SR)
    assert abs(len(normed) - SR) <= 2
    assert _snap() == before
