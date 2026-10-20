"""Pass 35 (music creation engine) — regression tests for verified defects.

Each test names the defect it pins. All were reproduced against pristine
HEAD (cebe4fb4) before the fix; see the pass-35 report for transcripts.
"""
from __future__ import annotations

import hashlib
import math
import os
import struct
import types
import wave
from unittest.mock import MagicMock

import pytest

from nomorals.core.midi import (
    MidiBuilder, NoteEvent, _parse_root, chord_progression, scale_notes,
    transpose_root,
)


# ── helpers ──────────────────────────────────────────────────────────────

def _ctx(tmp_path):
    """Minimal context whose workspace is the test tmp dir."""
    return types.SimpleNamespace(
        settings=types.SimpleNamespace(workspace_dir=str(tmp_path)))


def _parse_smf(path):
    """Parse an SMF file → [(track_name, [(note, channel), ...]), ...]."""
    data = open(path, "rb").read()
    assert data[:4] == b"MThd"
    _fmt, ntrk, _div = struct.unpack(">HHH", data[8:14])
    pos = 14
    tracks = []
    for _ in range(ntrk):
        assert data[pos:pos + 4] == b"MTrk"
        ln = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        chunk = data[pos + 8:pos + 8 + ln]
        pos += 8 + ln
        name, notes = "", []
        i, running = 0, None
        while i < len(chunk):
            while chunk[i] & 0x80:  # delta VLQ
                i += 1
            i += 1
            b0 = chunk[i]
            if b0 == 0xFF:
                mtype = chunk[i + 1]
                j = i + 2
                lnb = 0
                while True:
                    lnb = lnb * 128 + (chunk[j] & 0x7F)
                    j += 1
                    if not chunk[j - 1] & 0x80:
                        break
                if mtype == 0x03:
                    name = chunk[j:j + lnb].decode("utf-8", "replace")
                i = j + lnb
                running = None
            elif b0 in (0xF0, 0xF7):
                j = i + 1
                lnb = 0
                while True:
                    lnb = lnb * 128 + (chunk[j] & 0x7F)
                    j += 1
                    if not chunk[j - 1] & 0x80:
                        break
                i = j + lnb
                running = None
            else:
                if b0 & 0x80:
                    running = b0
                    i += 1
                status = running
                hi = status & 0xF0
                ndata = 1 if hi in (0xC0, 0xD0) else 2
                if hi == 0x90 and chunk[i + 1] > 0:
                    notes.append((chunk[i], status & 0x0F))
                i += ndata
        tracks.append((name, notes))
    return tracks


def _small_spec():
    from nomorals.media.songspec import SongSpec, SectionSpec
    return SongSpec(
        title="Pass35 Spec", tempo=120, key="C", mode="minor",
        sections=[SectionSpec(name="chorus", bars=2,
                              chords=["i", "VI"], energy=0.8),
                  SectionSpec(name="break", bars=2,
                              chords=["i", "VI"], energy=0.3)])


# ── F1: flat keys rendered a semitone sharp (core/midi._parse_root) ─────

@pytest.mark.parametrize("key,want", [
    ("Bb", 10), ("Eb", 3), ("Ab", 8), ("Db", 1), ("Gb", 6),
    ("bb", 10), ("eb", 3),
    # enharmonic edges: the offset is literal, so Cb sits one semitone
    # below C and B# one above B — consumers add it absolutely or mod it
    ("Cb", -1), ("B#", 12),
    ("F#", 6), ("C#", 1), ("G#", 8), ("E#", 5),
    ("C", 0), ("D", 2), ("E", 4), ("F", 5), ("G", 7), ("A", 9), ("B", 11),
])
def test_parse_root_accidentals(key, want):
    assert _parse_root(key) == want


def test_flat_key_scale_and_chords_are_in_the_named_key():
    assert scale_notes("Bb", "major", octaves=1)[0] == 70  # Bb4
    assert scale_notes("Eb", "major", octaves=1)[0] == 63  # Eb4
    triad = chord_progression("Eb", "major", ["I"])[0]
    assert sorted(n % 12 for n in triad) == [3, 7, 10]     # Eb G Bb
    assert transpose_root("Bb", 0) == "Bb"
    assert transpose_root("Eb", 2) == "F"


def test_rnb_profile_key_really_is_bb():
    """Producer assigns Bb to rnb/soul; the scale must start on Bb."""
    from nomorals.media.producer import _scale_degrees, describe_to_profile
    prof = describe_to_profile("smooth rnb slow jam")
    assert prof.key == "Bb"
    assert _scale_degrees(prof.key, prof.mode)[0] % 12 == 10


# ── F2: SMF time-signature denominator byte must be log2 ────────────────

@pytest.mark.parametrize("ts", [(4, 4), (3, 4), (6, 8), (2, 2), (12, 8)])
def test_time_signature_meta_uses_log2_denominator(ts):
    data = MidiBuilder(tempo=120, time_signature=ts).build()
    i = data.find(b"\xff\x58\x04")
    assert i != -1
    nn, dd = data[i + 3], data[i + 4]
    assert nn == ts[0]
    assert dd == int(math.log2(ts[1])), (
        f"{ts} written as {nn}/{2 ** dd} — dd is a power of two in SMF")


# ── F3: MidiBuilder track names must reach the file (FF 03) ─────────────

def test_track_names_are_serialized():
    b = MidiBuilder(tempo=120)
    tr = b.new_track("drums")
    b.add_notes(tr, [NoteEvent(36, 0.0, 0.25, 100, 9)])
    data = b.build()
    assert b"\xff\x03\x05drums" in data
    assert b"\xff\x03\x06melody" in data  # builder's own melody track


# ── F4: producer MIDI must carry every part, drums on channel 9 ─────────

def test_produce_from_spec_midi_is_complete(tmp_path):
    from nomorals.media.producer import Producer
    res = Producer(_ctx(tmp_path)).produce_from_spec(
        _small_spec(), seed=12, workdir="produced")
    assert res["ok"], res.get("reason")
    mid = res["path"].rsplit(".", 1)[0] + ".mid"
    tracks = dict(_parse_smf(mid))
    for part in ("melody", "counter", "chords", "bass", "drums"):
        assert tracks.get(part), f"{part} part missing/empty in MIDI"
    drum_channels = {ch for _n, ch in tracks["drums"]}
    assert drum_channels == {9}, (
        f"drums on channels {drum_channels}, not the GM percussion channel")


def test_produce_from_profile_midi_is_complete(tmp_path):
    from nomorals.media.producer import Producer
    res = Producer(_ctx(tmp_path)).produce(
        "dark driving edm test", seed=11, workdir="produced")
    assert res["ok"], res.get("reason")
    mid = res["path"].rsplit(".", 1)[0] + ".mid"
    tracks = dict(_parse_smf(mid))
    for part in ("melody", "counter", "chords", "bass", "drums"):
        assert tracks.get(part), f"{part} part missing/empty in MIDI"
    drum_channels = {ch for _n, ch in tracks["drums"]}
    assert drum_channels == {9}


# ── F5: a refused render write must surface as failure, not ok=True ─────

def test_render_cap_refusal_fails_honestly(tmp_path, monkeypatch):
    from nomorals.media import caps
    from nomorals.media.producer import Producer
    p = Producer(_ctx(tmp_path))
    ok_res = p.produce_from_spec(_small_spec(), seed=5, workdir="produced")
    assert ok_res["ok"] and os.path.getsize(ok_res["path"]) > 0
    monkeypatch.setattr(caps, "MAX_AUDIO_WRITE_BYTES", 64)
    bad = p.produce_from_spec(_small_spec(), seed=6, workdir="produced")
    assert bad["ok"] is False, (
        "render was refused by the write cap but produce reported success "
        f"with path {bad.get('path')!r} (exists: "
        f"{os.path.exists(bad.get('path', ''))})")
    assert bad.get("reason")


def test_synth_backend_raises_when_builtin_write_is_refused(
        tmp_path, monkeypatch):
    from nomorals.media import caps
    from nomorals.media.synth_backend import render_wav
    monkeypatch.setenv("NM_SYNTH_BACKEND", "builtin")
    monkeypatch.setattr(caps, "MAX_AUDIO_WRITE_BYTES", 64)
    parts = {"melody": [NoteEvent(60, 0.0, 1.0, 96, 0)]}
    with pytest.raises(RuntimeError):
        render_wav(parts, 120.0, 1, "", str(tmp_path / "x.wav"),
                   context=None)


# ── F6: chain LoRA must reach the ACE-Step backend it claims ────────────

class _FakeBedBackend:
    instances: list = []

    def __init__(self, **_kw):
        self.lora_path = ""
        self.lora_at_generate = None
        _FakeBedBackend.instances.append(self)

    def generate(self, lyrics, style, *, duration_s=120, seed=0,
                 title="", workdir=""):
        self.lora_at_generate = self.lora_path
        path = os.path.join(workdir, "fake-bed.wav")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(22050)
            w.writeframes(b"\x00\x00" * 2205)
        return MagicMock(audio_path=path, variant="fake-variant")


def test_chain_lora_reaches_backend(tmp_path, monkeypatch):
    import nomorals.media.ace_step as ace_mod
    from nomorals.media.musicgen_chain import ChainRequest, run_chain
    monkeypatch.setattr(ace_mod, "ACEStepBackend", _FakeBedBackend)
    _FakeBedBackend.instances.clear()
    req = ChainRequest(title="t", lyrics="la la la", language="en",
                       lora="/tmp/my-style-lora", workdir=str(tmp_path))
    res = run_chain(req)
    bed = next(s for s in res.stages if s.name == "bed")
    assert _FakeBedBackend.instances[-1].lora_at_generate == \
        "/tmp/my-style-lora"
    assert "+ LoRA /tmp/my-style-lora" in bed.detail
    # control: no LoRA requested → none attached, none claimed
    req2 = ChainRequest(title="t", lyrics="la la la", language="en",
                        workdir=str(tmp_path))
    res2 = run_chain(req2)
    bed2 = next(s for s in res2.stages if s.name == "bed")
    assert _FakeBedBackend.instances[-1].lora_at_generate == ""
    assert "LoRA" not in bed2.detail


def test_chain_tool_exposes_lora(tmp_path, monkeypatch):
    import nomorals.media.ace_step as ace_mod
    from nomorals.media import musicgen_chain
    monkeypatch.setattr(ace_mod, "ACEStepBackend", _FakeBedBackend)
    _FakeBedBackend.instances.clear()

    class _Reg:
        context = None

        def __init__(self):
            self.tools = {}

        def register(self, name, **_kw):
            def deco(fn):
                self.tools[name] = fn
                return fn
            return deco

    reg = _Reg()
    musicgen_chain.register(reg)
    out = reg.tools["musicgen_chain"](
        action="run", title="t", lyrics="la la la", language="en",
        lora="/tmp/tool-lora", workdir=str(tmp_path))
    assert out["ok"] is True
    assert _FakeBedBackend.instances[-1].lora_at_generate == "/tmp/tool-lora"


# ── F7: draft_melody must have the documented phrase arch ───────────────

def test_draft_melody_phrase_arch():
    from nomorals.media.musicgen_chain import draft_melody
    lyrics = " ".join(["la"] * 24)
    mid, end = [], []
    for seed in range(40):
        pitches = [p for p, _s, _d in draft_melody(lyrics, "en", seed=seed)]
        n = len(pitches)
        mid.append(sum(pitches[n // 3:2 * n // 3]) / (n // 3))
        end.append(sum(pitches[2 * n // 3:]) / (n - 2 * n // 3))
    assert sum(mid) / len(mid) > sum(end) / len(end) + 0.3, (
        "no phrase arch — contour is a bare random walk "
        f"(middle {sum(mid) / len(mid):.2f} vs end {sum(end) / len(end):.2f})")


def test_draft_melody_is_seeded_deterministic():
    from nomorals.media.musicgen_chain import draft_melody
    a = draft_melody("mo dé ayọ̀ dé", "yo", seed=7)
    b = draft_melody("mo dé ayọ̀ dé", "yo", seed=7)
    assert a == b


# ── F8: ABC parser must honor the L: unit-length header ─────────────────

def test_abc_L_header_sets_unit_length():
    from nomorals.media.abc_melody import parse_abc_melody
    # exactly what ABC_MELODY_PROMPT teaches the LLM to write
    abc = "X:1\nL:1/4\nK:C\nC2 D2 | E2 z2 | G2 E2 | C4 |"
    notes = parse_abc_melody(abc)
    total = max(s + d for _n, s, d in notes)
    assert total == pytest.approx(16.0), (
        f"L:1/4 melody rendered {total} beats, want 16 — the L: header "
        "was stripped and the 0.5-beat default applied instead")
    assert notes[-1][2] == pytest.approx(4.0)  # C4 is a full bar


def test_abc_default_length_unchanged_without_L():
    from nomorals.media.abc_melody import parse_abc_melody
    notes = parse_abc_melody("C D E F")
    assert [d for _n, _s, d in notes] == [0.5, 0.5, 0.5, 0.5]
    notes8 = parse_abc_melody("L:1/8\nC D E F")
    assert [d for _n, _s, d in notes8] == [0.5, 0.5, 0.5, 0.5]


# ── F9: video storyboard must use the caller's active draft ─────────────

def test_video_board_uses_callers_draft(monkeypatch):
    import nomorals.llm.brain as brain_mod
    from nomorals.tools import music as music_tools

    seen = {}

    def fake_brain_for(_ctx):
        class _Brain:
            def chat(self, messages, _params, task_kind=""):
                seen["prompt"] = messages[0].content \
                    if hasattr(messages[0], "content") else str(messages[0])
                return types.SimpleNamespace(text="board")
        return _Brain()

    monkeypatch.setattr(brain_mod, "brain_for", fake_brain_for)
    draft = types.SimpleNamespace(title="Neon Skyline", topic="lagos nights")
    music_tools._ACTIVE_DRAFTS["chatX"] = draft
    try:
        ctx = types.SimpleNamespace(
            loop_ctx=types.SimpleNamespace(chat_key="chatX"))
        out = music_tools._video_board("the song", ctx)
    finally:
        music_tools._ACTIVE_DRAFTS.pop("chatX", None)
    assert out.startswith("🎬")
    assert "Neon Skyline" in seen["prompt"], (
        "storyboard did not see the caller's active draft "
        "(looked up under the wrong draft key)")


# ── regression guards: seeded reproducibility / honest success ──────────

def test_seeded_produce_is_byte_reproducible(tmp_path):
    from nomorals.media.producer import Producer
    p = Producer(_ctx(tmp_path))
    a = p.produce_from_spec(_small_spec(), seed=77, workdir="produced")
    b = p.produce_from_spec(_small_spec(), seed=77, workdir="produced")
    assert a["ok"] and b["ok"]
    ha = hashlib.sha256(open(a["path"], "rb").read()).hexdigest()
    hb = hashlib.sha256(open(b["path"], "rb").read()).hexdigest()
    assert ha == hb
    c = p.produce_from_spec(_small_spec(), seed=78, workdir="produced")
    hc = hashlib.sha256(open(c["path"], "rb").read()).hexdigest()
    assert hc != ha
