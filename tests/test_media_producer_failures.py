"""Tail-60 pins for nomorals/media/producer.py —
0 converts / 7 exclusions (96, 122, 150, 169, 684, 829, 961).

7 tests for 7 sites (counts equal). Each test pins one site
with a control half proving the guarded operation was actually
ATTEMPTED, asserting via the operation's own call log / double.
The file contains no exception handlers, so it adds zero
census sites.

producer.py calls no failures-ledger record function anywhere
(grep-proven), so NOT-recording is pinned against the media
package ledger (nomorals/media/failures.py): recent_failures()
snapshot before/after every trigger.

Verdict notes (full grounds in the tail-60 SURVEY.md):
- 96: _oembed_title fetch/parse failure -> ""; "" is the
  genuine no-title value, analyze_reference proceeds with the
  songdata profile and no surface renders "" as a title.
- 122: BPM float() failure -> bpm simply absent from the
  scraped dict; analyze_reference's documented default
  (128.0) applies, exactly as for a page with no BPM text.
- 150: corrupt cache -> refetch; the fallback is the real
  fetched profile, not a fabricated value.
- 169: cache write failure -> the fetched profile is still
  returned, true in every field; nothing renders a "cached"
  claim.
- 684: _render MIDI write failure -> midi_path=""; the WAV
  is rendered from the parts themselves (the builtin path
  ignores the MIDI; the fluidsynth loss surfaces as a
  warning + builtin fallback inside render_wav).
- 829: _compose_from_spec chord_progression failure ->
  tonic ("i") fallback chords, real harmony composed from
  the same key/mode; _spec_notes makes no chord claim.
- 961: _render_spec MIDI write failure -> same as 684.
"""

import builtins
import json
import random
import types
from pathlib import Path
from unittest import mock

import nomorals.media.producer as PR
from nomorals.core import midi as midi_mod
from nomorals.core.midi import NoteEvent
from nomorals.media.failures import recent_failures

_SONGDATA_HTML = ("<html><body>X is a calm, dreamy track by A. "
                  "It runs 2:52 at 155 BPM and is in F Minor.</body></html>")


def _snap():
    return list(recent_failures(200))


def _canned_http(log):
    def http(url, timeout=15.0):
        log.append(url)
        if "oembed" in url:
            return '{"title": "Fetched Title"}'
        return _SONGDATA_HTML
    return http


def test_96_oembed_title_failure_yields_empty_title():
    before = _snap()
    calls = []
    with mock.patch.object(
            PR, "_http_get",
            lambda url, timeout=10.0: calls.append(url) or "{bad json"):
        assert PR._oembed_title("abc123") == ""
    assert len(calls) == 1  # the fetch was actually attempted
    with mock.patch.object(
            PR, "_http_get",
            lambda url, timeout=10.0: '{"title": "Hello Song"}'):
        assert PR._oembed_title("abc123") == "Hello Song"
    assert _snap() == before


def test_122_bpm_parse_failure_omits_bpm_only():
    before = _snap()
    real_float = builtins.float
    flog = []

    def fake_float(x, *a, **k):
        flog.append(x)
        if x == "155":
            raise ValueError("bpm boom")
        return real_float(x, *a, **k)

    with mock.patch.object(PR, "_http_get",
                           lambda url, timeout=15.0: _SONGDATA_HTML):
        with mock.patch.object(builtins, "float", fake_float):
            out = PR._songdata_profile("tid1")
    assert flog == ["155"]  # the conversion was actually attempted
    assert "bpm" not in out
    assert out.get("key") == "F" and out.get("mode") == "minor"
    with mock.patch.object(PR, "_http_get",
                           lambda url, timeout=15.0: _SONGDATA_HTML):
        assert PR._songdata_profile("tid1").get("bpm") == 155.0
    assert _snap() == before


def test_150_corrupt_cache_refetches(tmp_path):
    before = _snap()
    corrupt = tmp_path / "cache.json"
    corrupt.write_text("{corrupt json")
    http_log = []
    with mock.patch.object(PR, "_cache_path", lambda tid: corrupt):
        with mock.patch.object(PR, "_http_get", _canned_http(http_log)):
            prof = PR.analyze_reference(
                "https://open.spotify.com/track/xyz789")
    assert len(http_log) == 2  # refetch was actually attempted
    assert prof.ok and prof.title == "Fetched Title" and prof.bpm == 155.0
    good = tmp_path / "good.json"
    good.write_text(json.dumps({
        "title": "Cached Title", "artist": "", "bpm": 99.0, "key": "C",
        "mode": "major", "energy_words": [], "genre": "electronic",
        "mood": "", "ok": True, "reason": ""}))
    http_log.clear()
    with mock.patch.object(PR, "_cache_path", lambda tid: good):
        with mock.patch.object(PR, "_http_get", _canned_http(http_log)):
            prof = PR.analyze_reference(
                "https://open.spotify.com/track/xyz789")
    assert prof.title == "Cached Title" and prof.bpm == 99.0
    assert http_log == []  # control: valid cache means no fetch
    assert _snap() == before


def test_169_cache_write_failure_still_returns_profile(tmp_path):
    before = _snap()

    class FakePath:
        def __init__(self):
            self.writes = []

        def exists(self):
            return False

        def write_text(self, data, *a, **k):
            self.writes.append(data)
            raise OSError("disk full")

    fake = FakePath()
    http_log = []
    with mock.patch.object(PR, "_cache_path", lambda tid: fake):
        with mock.patch.object(PR, "_http_get", _canned_http(http_log)):
            prof = PR.analyze_reference(
                "https://open.spotify.com/track/xyz789")
    assert len(fake.writes) == 1  # the write was actually attempted
    assert prof.ok and prof.title == "Fetched Title" and prof.bpm == 155.0
    realp = tmp_path / "w.json"
    with mock.patch.object(PR, "_cache_path", lambda tid: realp):
        with mock.patch.object(PR, "_http_get", _canned_http(http_log)):
            prof = PR.analyze_reference(
                "https://open.spotify.com/track/xyz789")
    assert prof.ok and realp.exists()
    assert json.loads(realp.read_text())["title"] == "Fetched Title"
    assert _snap() == before


def _parts():
    return {"melody": [NoteEvent(note=69, start=0.0, duration=1.0,
                                 velocity=90)],
            "counter": [], "chords": [], "bass": [], "drums": []}


def _ctx(tmp_path):
    return types.SimpleNamespace(
        settings=types.SimpleNamespace(workspace_dir=str(tmp_path)))


def test_684_render_midi_failure_still_renders_wav(tmp_path):
    before = _snap()
    wlog = []

    def bad_write(self, path):
        wlog.append(path)
        raise OSError("midi write boom")

    p = PR.Producer(_ctx(tmp_path))
    with mock.patch.object(midi_mod.MidiBuilder, "write", bad_write):
        out = p._render(PR.ReferenceProfile(title="Probe Song", bpm=120.0),
                        _parts(), 42, "produced")
    assert len(wlog) == 1  # the MIDI write was actually attempted
    assert Path(out).exists() and Path(out).stat().st_size > 0
    assert not Path(out.rsplit(".", 1)[0] + ".mid").exists()
    out2 = p._render(PR.ReferenceProfile(title="Probe Song", bpm=120.0),
                     _parts(), 43, "produced")
    assert Path(out2.rsplit(".", 1)[0] + ".mid").exists()  # control
    assert _snap() == before


def test_829_bad_numerals_fall_back_to_tonic_chords(tmp_path):
    before = _snap()
    from nomorals.media.songspec import SectionSpec, SongSpec

    spec = SongSpec(title="Probe", tempo=120, key="C", mode="minor",
                    sections=[SectionSpec(name="chorus", bars=2,
                                          chords=["i", "VI"], energy=0.8)])
    plogs = []
    real_cp = midi_mod.chord_progression

    def flaky_cp(key, mode, prog, base_octave=3):
        plogs.append(list(prog))
        if "bVI" in prog:  # normalize_roman("VI") in a minor key
            raise RuntimeError("bad numeral boom")
        return real_cp(key, mode, prog, base_octave=base_octave)

    p = PR.Producer(_ctx(tmp_path))
    motif = PR.write_motif("C", "minor", bars=2, rng=random.Random(3))
    scale = PR._scale_degrees("C", "minor")
    with mock.patch.object(midi_mod, "chord_progression", flaky_cp):
        parts = p._compose_from_spec(spec, motif, scale,
                                     random.Random(3), 7)
    # spec progression attempted first, tonic fallback second
    assert plogs == [["i", "bVI"], ["i", "i"]]
    assert len(parts["chords"]) > 0
    plogs.clear()

    def delegating_cp(key, mode, prog, base_octave=3):
        plogs.append(list(prog))
        return real_cp(key, mode, prog, base_octave=base_octave)

    with mock.patch.object(midi_mod, "chord_progression", delegating_cp):
        parts = p._compose_from_spec(spec, motif, scale,
                                     random.Random(3), 7)
    assert plogs == [["i", "bVI"]]  # control: spec chords used directly
    assert len(parts["chords"]) > 0
    assert _snap() == before


def test_961_render_spec_midi_failure_still_renders_wav(tmp_path):
    before = _snap()
    from nomorals.media.songspec import SongSpec

    wlog = []

    def bad_write(self, path):
        wlog.append(path)
        raise OSError("midi write boom")

    spec = SongSpec(title="Probe Spec", tempo=120, key="C", mode="minor")
    p = PR.Producer(_ctx(tmp_path))
    with mock.patch.object(midi_mod.MidiBuilder, "write", bad_write):
        out = p._render_spec(spec, _parts(), 42, "produced")
    assert len(wlog) == 1  # the MIDI write was actually attempted
    assert Path(out).exists() and Path(out).stat().st_size > 0
    assert not Path(out.rsplit(".", 1)[0] + ".mid").exists()
    out2 = p._render_spec(spec, _parts(), 43, "produced")
    assert Path(out2.rsplit(".", 1)[0] + ".mid").exists()  # control
    assert _snap() == before
