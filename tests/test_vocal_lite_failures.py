"""Pass #119 pins: vocal_lite's 12 silent handlers now record.

Covers the ledger basics (ring bound, unprintable-exc safety, format,
empty state), the direct-import surface, the unchanged ``__all__``,
one real converted site per pin (fallback value held AND site
recorded), and a suppression-off scan pin asserting the remainder is
exactly the 8 debug-honest sites (the scanner's ``debug`` blind spot:
they log with ``exc_info=True`` but the scanner's "uses" set lacks
``debug`` — honest, not convertible).
"""

import ast
import wave
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from nomorals.media import vocal_lite
from nomorals.media.vocal_lite_failures import (
    clear_failures,
    failure_count,
    format_failures,
    recent_failures,
    record_failure,
)

SITES = [
    "vocal_lite._profile_kind",
    "vocal_lite.song_duration_beats",
    "vocal_lite._phrases:event",
    "vocal_lite._phrases",
    "vocal_lite._phrase_median_midi",
    "vocal_lite._tts_line_f0_midi",
    "vocal_lite._resample_linear",
    "vocal_lite.render_sung_vocal:section_lines",
    "vocal_lite.render_sung_vocal:phrase_start",
    "vocal_lite._event_in_chorus",
    "vocal_lite.render_hummed_vocal:event",
    "vocal_lite.add_vocal_track:cap_mib",
]


@pytest.fixture(autouse=True)
def _clean_ledger():
    clear_failures()
    yield
    clear_failures()


def _sites():
    return [f.site for f in recent_failures(500)]


def _silent_wav(path: Path, frames: int = 2205) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(b"\x00\x00" * frames)
    return str(path)


class _Section:
    def __init__(self, name, bars, lyrics):
        self.name = name
        self.bars = bars
        self.lyrics = lyrics


class _Song:
    def __init__(self, sections, tempo=100):
        self.sections = sections
        self.tempo = tempo


# --- ledger basics -------------------------------------------------------


def test_ledger_record_recent_count_clear_format():
    assert recent_failures() == []
    assert format_failures() == "no recorded vocal_lite failures"
    record_failure("vocal_lite._phrases", ValueError("bad event"))
    assert failure_count() == 1
    entries = recent_failures()
    assert len(entries) == 1
    assert entries[0].site == "vocal_lite._phrases"
    assert entries[0].exc_type == "ValueError"
    assert "bad event" in format_failures()
    clear_failures()
    assert failure_count() == 0
    assert recent_failures() == []


def test_ledger_bound_keeps_newest_200():
    for i in range(250):
        record_failure("vocal_lite._phrases", ValueError(f"e{i}"))
    assert failure_count() == 250
    kept = recent_failures(500)
    assert len(kept) == 200
    assert kept[0].message == "e249"
    assert kept[-1].message == "e50"


def test_ledger_unprintable_exc_safe():
    class _Boom(Exception):
        def __str__(self):
            raise RuntimeError("no str for you")

    record_failure("vocal_lite._phrases", _Boom())
    assert failure_count() == 1
    assert "unprintable" in recent_failures()[0].message


def test_direct_import_surface_and_all_unchanged():
    # Direct import is the surface: vocal_lite declares an explicit
    # __all__ (the #102 lesson says not to widen a declared surface).
    assert vocal_lite.__all__ == [
        "hook_lyrics",
        "chorus_regions",
        "section_regions",
        "song_duration_beats",
        "render_hook_vocal",
        "render_sung_vocal",
        "render_hummed_vocal",
        "add_vocal_track",
        "mix_vocal_track",
        "mix_hum_track",
    ]


# --- converted sites, one real pin each ----------------------------------


def test_profile_kind_failure_records_and_falls_back(monkeypatch):
    from nomorals.core import profile

    monkeypatch.setattr(profile, "detect_profile",
                        mock.Mock(side_effect=RuntimeError("no profile")))
    assert vocal_lite._profile_kind() == ""
    assert _sites() == ["vocal_lite._profile_kind"]


def test_song_duration_beats_failure_records_and_falls_back():
    class _BadSong:
        @property
        def sections(self):
            raise RuntimeError("sections unreadable")

    assert vocal_lite.song_duration_beats(_BadSong()) == 0.0
    assert _sites() == ["vocal_lite.song_duration_beats"]


def test_phrases_event_skip_records_and_keeps_good_events():
    class _FlakyStart:
        def __init__(self):
            self._n = 0

        @property
        def start(self):
            # Sorts fine on the first read (the sorted() key), then fails
            # on the loop's re-read — that is the inner handler's path.
            self._n += 1
            if self._n > 1:
                raise RuntimeError("start unreadable on re-read")
            return 0.0

        duration = 1.0

    good = SimpleNamespace(start=0.0, duration=1.0)
    phrases = vocal_lite._phrases([good, _FlakyStart()])
    assert phrases == [[good]]
    assert _sites() == ["vocal_lite._phrases:event"]


def test_phrases_outer_failure_records_and_falls_back():
    def _gen():
        yield SimpleNamespace(start=0.0, duration=1.0)
        raise RuntimeError("stream broke")

    assert vocal_lite._phrases(_gen()) == []
    assert _sites() == ["vocal_lite._phrases"]


def test_phrase_median_midi_failure_records_and_falls_back():
    class _BadNote:
        @property
        def note(self):
            raise RuntimeError("note unreadable")

    assert vocal_lite._phrase_median_midi([_BadNote()]) == 0.0
    assert _sites() == ["vocal_lite._phrase_median_midi"]


def test_tts_line_f0_failure_records_and_falls_back():
    with mock.patch("nomorals.voice.tts.voice_print",
                    side_effect=RuntimeError("voice_print boom")):
        assert vocal_lite._tts_line_f0_midi("/nonexistent.wav") == 0.0
    assert _sites() == ["vocal_lite._tts_line_f0_midi"]


def test_resample_linear_failure_records_and_returns_input():
    samples = array("d", [0.1, 0.2, 0.3])
    out = vocal_lite._resample_linear(samples, 0, 22050)
    assert list(out) == [0.1, 0.2, 0.3]
    assert _sites() == ["vocal_lite._resample_linear"]


def test_render_sung_vocal_section_lines_skip_records(tmp_path):
    class _RaisingLyricsSection:
        name = "chorus"
        bars = 4

        @property
        def lyrics(self):
            raise RuntimeError("lyrics unreadable")

    song = _Song([_RaisingLyricsSection()])
    res = vocal_lite.render_sung_vocal(song, str(tmp_path), [])
    assert res["ok"] is False
    assert "no lyric lines could be sung" in res["reason"]
    assert _sites() == ["vocal_lite.render_sung_vocal:section_lines"]


def test_render_sung_vocal_phrase_start_fallback_records(tmp_path, monkeypatch):
    class _FlakyStart:
        note = 60

        @property
        def start(self):
            raise RuntimeError("phrase start unreadable")

    wav = _silent_wav(tmp_path / "line.wav")
    monkeypatch.setattr(vocal_lite, "_phrases",
                        lambda events, gap_beats=1.0: [[_FlakyStart()]])
    monkeypatch.setattr(vocal_lite, "_tts_line_f0_midi",
                        lambda path: 0.0)
    song = _Song([_Section("chorus", 4, ["la"])])
    res = vocal_lite.render_sung_vocal(
        song, str(tmp_path), [SimpleNamespace(start=0.0, duration=1.0)],
        tts_fn=lambda text, out_path: {"ok": True, "path": wav})
    assert res["ok"] is True
    assert res["lines"] == 1
    assert _sites() == ["vocal_lite.render_sung_vocal:phrase_start"]


def test_event_in_chorus_failure_records_and_answers_false():
    class _BadEvent:
        @property
        def start(self):
            raise RuntimeError("start unreadable")

    assert vocal_lite._event_in_chorus(_BadEvent(), [(0.0, 16.0)]) is False
    assert _sites() == ["vocal_lite._event_in_chorus"]


def test_render_hummed_vocal_event_skip_records_and_keeps_good(tmp_path):
    class _BadNote:
        start = 0.0

        @property
        def note(self):
            raise RuntimeError("note unreadable")

        duration = 1.0
        velocity = 96

    good = SimpleNamespace(start=0.0, note=60, duration=1.0, velocity=96)
    song = _Song([_Section("chorus", 4, [])])
    res = vocal_lite.render_hummed_vocal(
        song, str(tmp_path), [good, _BadNote()])
    assert res["ok"] is True
    assert _sites() == ["vocal_lite.render_hummed_vocal:event"]


def test_add_vocal_track_cap_mib_failure_records(tmp_path, monkeypatch):
    from nomorals.media import caps

    class _BadInt:
        def __floordiv__(self, other):
            raise RuntimeError("cap unreadable")

    bed = _silent_wav(tmp_path / "bed.wav", frames=2205)
    voc = _silent_wav(tmp_path / "hook.wav", frames=2205)
    monkeypatch.setattr(caps, "check_write_size",
                        lambda *a, **k: (False, "test cap"))
    monkeypatch.setattr(caps, "MAX_AUDIO_WRITE_BYTES", _BadInt())
    song = _Song([_Section("chorus", 4, ["sing this hook"])])
    res = vocal_lite.add_vocal_track(
        song, bed, str(tmp_path),
        tts_fn=lambda text, out_path: {"ok": True, "path": voc})
    assert res["ok"] is False
    assert "0 MiB" in res["reason"]
    assert _sites() == ["vocal_lite.add_vocal_track:cap_mib"]


# --- scan pin ------------------------------------------------------------


def test_scan_remainder_is_empty_debug_honest_recognized(monkeypatch):
    from nomorals.tools import error_scan as es

    monkeypatch.setattr(es, "_noqa_suppresses",
                        lambda line, rule: False)
    path = Path(vocal_lite.__file__)
    report = es.scan([str(path)])
    findings = [f for f in report.findings if f.file == str(path)]
    # #140: the scanner now recognizes exc_info=True logging, so all 8
    # debug-honest sites (hook_lyrics, section_regions, chorus_regions,
    # _tile_vocal, _render_hum_tone, _write_mono_wav, mix_hum_track,
    # _mix_sung_track) no longer count, and none of the 12 converted
    # sites remains either: the remainder is empty.
    assert findings == []
