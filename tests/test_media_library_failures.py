"""Tail-91 pins: nomorals/media/library.py census sites.

Module posture (its own docstring): metadata extraction is a
best-effort chain — mutagen, then ffprobe, then the file name —
and "metadata extraction never blocks a library write"; a file
that defeats every reader still enters the library with its
file name as the title. ClipIndex extraction likewise leaves a
failed signal at the documented 0.0 ("honest") and still
indexes the entry. Every site is held as an exclusion:

67 — ``mutagen.File`` crash -> None in ``_metadata_mutagen``.
None is the rung's designed give-up value: the chain falls
through to ffprobe, then to the literal file name. The failure
can only ever cost tag richness, never fabricate a tag.
76 — per-key tag-read crash -> continue in the mutagen ``tag``
helper. One odd tag backend must not lose the other keys:
probe-proven, a crashing "title" key still yields the artist
and the duration.
162 — reader crash -> got=None in ``read_metadata``. The
chain-level guard (the readers also self-guard; this is the
belt-and-braces that keeps read_metadata's "Never raises"
contract absolute). Failure lands on the filename fallback —
a literal, not an invention.
720 — ``segment_movie`` crash -> bounds [(0.0, 0.0)] in
``ClipIndex.index_file``. Segmentation is best-effort: the
entry still indexes with zeroed signals. Tension, stated
plainly: the fallback bounds bin NO transcript words
(0 <= t < 0 is empty), so the entry also loses its transcript
— wider than the docstring's "still indexes with its
transcript" line admits. Held: every field of the entry is an
absent/zero value, no measurement is fabricated, and inventing
fallback bounds would be a design change, not an
error-handling fix.
724 — ``extract_features`` crash -> feats=[]. Each feature
signal reads the documented absent value 0.0 while the entry
keeps its transcript and bounds — probe-proven against the
control, where real features land.
"""
from __future__ import annotations

import logging
import sys
import types

from nomorals.media import library as L
import nomorals.media.scene_intel.segment as seg_mod
import nomorals.media.highlight_model.features as feat_mod


# ── 67: mutagen crash gives up the rung, not the chain ──────────

def test_67_mutagen_file_crash_returns_none(monkeypatch):
    fake_mut = types.ModuleType("mutagen")

    def _boom(*a, **k):
        raise RuntimeError("tag boom")

    fake_mut.File = _boom
    monkeypatch.setitem(sys.modules, "mutagen", fake_mut)
    assert L._metadata_mutagen("/x.mp3") is None


def test_67_chain_falls_through_to_filename(monkeypatch):
    fake_mut = types.ModuleType("mutagen")

    def _boom(*a, **k):
        raise RuntimeError("tag boom")

    fake_mut.File = _boom
    monkeypatch.setitem(sys.modules, "mutagen", fake_mut)
    monkeypatch.setattr(L, "_metadata_ffprobe", lambda p: None)
    meta = L.read_metadata("/some/dir/Plain Name.mp3")
    assert meta["title"] == "Plain Name"
    assert meta["artist"] == ""


# ── 76: one bad tag key costs only that key ─────────────────────

def test_76_tag_key_crash_continues_to_next_key(monkeypatch):
    fake_audio = types.SimpleNamespace(
        info=types.SimpleNamespace(length=12.5))

    def fake_get(k):
        if k == "title":
            raise RuntimeError("odd backend")
        if k == "artist":
            return ["A Artist"]
        return None

    fake_audio.get = fake_get
    fake_mut = types.ModuleType("mutagen")
    fake_mut.File = lambda *a, **k: fake_audio
    monkeypatch.setitem(sys.modules, "mutagen", fake_mut)
    meta = L._metadata_mutagen("/x.mp3")
    assert meta is not None
    assert meta["title"] == ""
    assert meta["artist"] == "A Artist"
    assert meta["duration"] == 12.5


# ── 162: reader crash lands on the filename fallback ────────────

def test_162_both_readers_crash_filename_fallback(monkeypatch):
    def _boom(p):
        raise RuntimeError("reader boom")

    monkeypatch.setattr(L, "_metadata_mutagen", _boom)
    monkeypatch.setattr(L, "_metadata_ffprobe", _boom)
    meta = L.read_metadata("/some/dir/My Song.mp3")
    assert meta["title"] == "My Song"
    assert meta["artist"] == ""


def test_162_control_missing_file_filename_fallback():
    meta = L.read_metadata("/some/dir/Real Tune.flac")
    assert meta["title"] == "Real Tune"


# ── 720: segmentation crash indexes a zeroed entry ──────────────

def test_720_segmentation_crash_indexes_zero_entry(monkeypatch):
    def _boom(p):
        raise RuntimeError("seg boom")

    monkeypatch.setattr(seg_mod, "segment_movie", _boom)
    monkeypatch.setattr(feat_mod, "extract_features", lambda p: [])
    idx = L.ClipIndex()
    entries = idx.index_file(
        "/movie.mp4", transcript=[{"start": 0.5, "text": "hello"}])
    assert len(entries) == 1
    entry = entries[0]
    assert entry["start"] == 0.0 and entry["end"] == 0.0
    assert entry["audio_energy"] == 0.0
    assert entry["motion"] == 0.0
    # the tension, pinned: (0,0) bounds bin no transcript words
    assert entry["transcript"] == ""


# ── 724: feature crash zeroes signals, keeps the entry ──────────

class _FakeSeg:
    def __init__(self, s, e, kind="scene"):
        self.start, self.end, self.kind = s, e, kind


def test_724_feature_crash_zeroes_signals_keeps_transcript(monkeypatch):
    monkeypatch.setattr(
        seg_mod, "segment_movie",
        lambda p: types.SimpleNamespace(segments=[_FakeSeg(0.0, 2.0)]))

    def _boom(p):
        raise RuntimeError("feat boom")

    monkeypatch.setattr(feat_mod, "extract_features", _boom)
    idx = L.ClipIndex()
    entries = idx.index_file(
        "/movie.mp4", transcript=[{"start": 0.5, "text": "hello world"}])
    entry = entries[0]
    assert entry["transcript"] == "hello world"
    assert entry["audio_energy"] == 0.0
    assert entry["motion"] == 0.0
    assert entry["aesthetic"] == 0.0


def test_724_control_real_features_land(monkeypatch):
    monkeypatch.setattr(
        seg_mod, "segment_movie",
        lambda p: types.SimpleNamespace(segments=[_FakeSeg(0.0, 2.0)]))
    monkeypatch.setattr(feat_mod, "extract_features",
                        lambda p: [(0.9, 0.5, 0.3)])
    entry = L.ClipIndex().index_file("/movie.mp4")[0]
    assert abs(entry["audio_energy"] - 0.9) < 1e-9
    assert abs(entry["motion"] - 0.5) < 1e-9


# ── NOT-recording: the metadata sites log nothing ───────────────

def test_sites_record_nothing(caplog):
    with caplog.at_level(logging.DEBUG,
                         logger="nomorals.media.library"):
        L.read_metadata("/some/dir/My Song.mp3")
    assert [r for r in caplog.records
            if r.name == "nomorals.media.library"] == []
