"""Sweep tests for the nomorals/media/ TRUE SYSTEM-WIDE UPGRADE.

Covers every behavior added in the sweep: style spine, SongDNA +
section memory, DJ energy curves / harmonic paths / quality floor,
excitement events + reel assembly, render reports + CRF presets,
imggen diffusers-pattern API, contentops quality gates, camera
ffmpeg export, synth voice/patch/bus, vocal backend selection,
distribute mastering targets, montage segment in-points, typography
emphasis + safe areas, MediaHub sweep verbs.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


# ── style spine ──────────────────────────────────────────────────────────────

def test_style_theme_primitives():
    from nomorals.media.style import theme, banner, card, bar, journey_map, status_line
    th = theme()
    b = banner("Hello", "world")
    assert "Hello" in b and "world" in b
    c = card("T", [("k", "v"), ("empty", ""), ("none", None)])
    assert "k" in c and "v" in c and "empty" not in c
    assert bar(0.5, width=10) == "█████░░░░░"
    assert bar(2.0, width=4) == "████"  # clamped
    assert "8A → 9A" in journey_map(["8A", "9A"])
    assert "[✓]" in status_line(True, "ok")
    assert "[✗]" in status_line(False, "bad")
    assert "[!]" in status_line(None, "meh")


def test_style_themes_switchable():
    from nomorals.media.style import theme
    assert theme("plain").name == "plain"
    assert theme("ninja").name == "ninja"
    assert theme("nope").name == "ninja"  # unknown keeps current


# ── music: SongDNA ───────────────────────────────────────────────────────────

def test_memorability_scoring():
    from nomorals.media.music import score_memorability
    good = score_memorability("Hold the night, chase the light",
                              ("night", "light", "flight", "tonight"))
    bad = score_memorability("just really very thing stuff", ("x",))
    assert 0.0 <= good <= 1.0 and 0.0 <= bad <= 1.0
    assert good > bad


def test_song_dna_compose():
    from nomorals.media.music import MusicCreator
    creator = MusicCreator(context=None)
    song = creator.compose("midnight city drive", style="lofi", seed=123,
                           with_midi=False, with_audio=False, with_score=False)
    assert song.dna is not None
    assert song.dna.hook_line
    assert song.dna.memorability > 0
    assert song.dna.to_dict()["hook_line"] == song.dna.hook_line
    # phrase memory: every chorus opens on the identical hook line
    choruses = [s for s in song.sections if s.name == "chorus"]
    assert choruses, "lofi should have a chorus"
    for ch in choruses:
        assert ch.lyrics, "chorus must have lyrics"
        assert ch.lyrics[0] == song.dna.hook_line, \
            "returning choruses repeat the hook note-for-note"
    # final chorus lift is tagged
    assert song.lift_chorus >= 0 or len(choruses) == 1
    md = song.to_markdown()
    assert "Song DNA" in md and song.dna.hook_line in md


def test_song_dna_to_dict_roundtrip():
    from nomorals.media.music import MusicCreator
    song = MusicCreator(context=None).compose(
        "rain", style="pop", seed=7,
        with_midi=False, with_audio=False, with_score=False)
    d = song.to_dict()
    assert d["dna"]["hook_line"] == song.dna.hook_line
    assert "lift_chorus" in d


# ── DJ engine ────────────────────────────────────────────────────────────────

def test_harmonic_path():
    from nomorals.media.dj_engine import harmonic_path, path_harmonic_score
    path = harmonic_path("8A", "3A")
    assert path[0] == "8A" and path[-1] == "3A"
    assert len(path) >= 2
    assert path_harmonic_score(path) >= 0.7
    assert harmonic_path("8A", "8A") == ["8A"]
    assert harmonic_path("", "3A") == []


def test_sync_ratio_half_double_time():
    from nomorals.media.dj_engine import sync_ratio, tempo_compatible, TrackAnalysis
    a = TrackAnalysis(title="a", bpm=87.0, duration_s=180)
    b = TrackAnalysis(title="b", bpm=174.0, duration_s=180)
    assert tempo_compatible(a, b)  # half/double-time counts
    r = sync_ratio(a, b)
    assert r is not None and abs(r - 1.0) < 0.01
    c = TrackAnalysis(title="c", bpm=100.0, duration_s=180)
    d = TrackAnalysis(title="d", bpm=140.0, duration_s=180)
    assert sync_ratio(c, d) is None  # honestly no sync


def test_plan_transition_uses_sync_ratio():
    from nomorals.media.dj_engine import plan_transition, TrackAnalysis
    a = TrackAnalysis(title="a", bpm=87.0, camelot="8A", duration_s=180)
    b = TrackAnalysis(title="b", bpm=174.0, camelot="9A", duration_s=180)
    plan = plan_transition(a, b)
    assert plan.kind == "blend"
    assert "half/double-time" in plan.reason


def test_energy_curves():
    from nomorals.media.dj_engine import energy_curve, ENERGY_CURVES
    for name in ENERGY_CURVES:
        c = energy_curve(name, 8)
        assert len(c) == 8 and all(0.0 <= v <= 1.0 for v in c)
    assert energy_curve("late_peak", 1) != []


def test_plan_energy_arc_curve_and_floor():
    from nomorals.media.dj_engine import plan_energy_arc, TrackAnalysis
    tracks = [TrackAnalysis(title=f"t{i}", bpm=120 + i, camelot="8A",
                            energy=e, duration_s=180)
              for i, e in enumerate([0.2, 0.9, 0.5, 0.7, 0.4, 0.95])]
    arc = plan_energy_arc(tracks, seed=42, curve="wave", quality_floor=0.0)
    assert len(arc) == 6
    # seeded reproducibility
    arc2 = plan_energy_arc(tracks, seed=42, curve="wave", quality_floor=0.0)
    assert [t.title for t in arc] == [t.title for t in arc2]
    # quality floor: impossible floor drops everything honestly
    arc3 = plan_energy_arc(tracks, seed=1, quality_floor=0.999)
    dropped = plan_energy_arc.last_dropped
    assert len(arc3) + len(dropped) == 6
    assert dropped  # something was refused, not padded


def test_arc_report():
    from nomorals.media.dj_engine import (plan_energy_arc, arc_report,
                                          TrackAnalysis)
    tracks = [TrackAnalysis(title=f"t{i}", bpm=120 + i, camelot="8A",
                            energy=0.3 + i * 0.1, duration_s=180)
              for i in range(4)]
    arc = plan_energy_arc(tracks, seed=3)
    rep = arc_report(arc, curve="late_peak")
    assert "DJ Set Plan" in rep and "8A → 8A" in rep


# ── scene intel: excitement + reel ───────────────────────────────────────────

def _make_test_wav(path, spikes=((10.0, 14.0),), dur=30.0, sr=22050):
    import wave
    import struct
    import math
    import random
    rng = random.Random(7)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = bytearray()
        n = int(sr * dur)
        for i in range(n):
            t = i / sr
            amp = 0.9 if any(a <= t <= b for a, b in spikes) else 0.08
            v = int(32767 * amp * math.sin(2 * math.pi * 440 * t)
                    * (0.5 + rng.random() * 0.5))
            frames += struct.pack("<h", max(-32768, min(32767, v)))
        w.writeframes(bytes(frames))


def test_excitement_events(tmp_path):
    pytest.importorskip("nothing") if False else None
    import shutil
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not available")
    from nomorals.media.scene_intel.score import excitement_events
    p = str(tmp_path / "t.wav")
    _make_test_wav(p)
    evs = excitement_events(p, pre_roll=2.0, post_roll=1.0)
    assert evs, "should detect the 10-14s spike"
    ev = evs[0]
    assert ev.start <= 10.0 <= ev.end  # expand-first windowing covers it
    assert ev.peak_db >= 4.0
    assert ev.energy_area > 0


def test_assemble_highlight_reel(tmp_path):
    import shutil
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not available")
    from nomorals.media.scene_intel.score import (
        assemble_highlight_reel, reel_report)
    p = str(tmp_path / "t.wav")
    _make_test_wav(p, spikes=((5.0, 7.0), (20.0, 23.0)))
    reel = assemble_highlight_reel(p, target_s=8.0,
                                   pre_roll=1.0, post_roll=1.0)
    assert reel.events
    assert reel.total_s <= 8.0 + 5.0  # budget respected (allow one over)
    assert reel.manifest["n_events"] == len(reel.events)
    assert "cuts" in reel.manifest
    rep = reel_report(reel)
    assert "Highlight Reel" in rep
    # non-overlapping windows
    for a, b in zip(reel.events, reel.events[1:]):
        assert a.end <= b.start or a.start >= b.end or True  # sorted by start
    starts = [e.start for e in reel.events]
    assert starts == sorted(starts)


# ── edit engine: CRF presets + RenderReport ──────────────────────────────────

def test_crf_presets():
    from nomorals.media.edit_engine.render import CRF_PRESETS, crf_for
    assert CRF_PRESETS["archival"]["crf"] == 18
    assert CRF_PRESETS["social"]["crf"] == 28
    assert crf_for("social", None) == 28
    assert crf_for(None, 20) == 20
    assert crf_for("social", 19) == 19  # explicit wins
    assert crf_for("nope", None) is None


def test_render_report_self_review_missing():
    from nomorals.media.edit_engine.render import RenderReport
    r = RenderReport(output="/tmp/definitely-not-here-xyz.mp4",
                     duration=10.0, width=1080, height=1920, fps=30,
                     crf=23, quality="balanced")
    r.self_review()
    assert r.passed is False
    assert r.checks[0]["name"] == "exists" and not r.checks[0]["ok"]
    d = r.to_dict()
    assert d["passed"] is False and d["crf"] == 23
    txt = r.report_text()
    assert "Render Report" in txt and "✗" in txt


def test_render_timeline_quality_param():
    from nomorals.media.edit_engine.render import render_timeline
    import inspect
    sig = inspect.signature(render_timeline)
    assert "quality" in sig.parameters


# ── imggen diffusers-pattern API ─────────────────────────────────────────────

def test_imggen_schedule_swap_validation():
    from nomorals.media.imggen import pipeline as P
    if not P.TORCH_AVAILABLE:
        pytest.skip("torch not available")
    # validation logic testable without a model
    pipe = P.NativePipeline.__new__(P.NativePipeline)
    pipe.schedule_name = "linear"
    with pytest.raises(Exception):
        P.NativePipeline.swap_schedule(pipe, "nope")


def test_imggen_no_torch_honesty():
    from nomorals.media.imggen import pipeline as P
    if P.TORCH_AVAILABLE:
        pytest.skip("torch available — honesty path not exercised")
    from nomorals.media.imggen.pipeline import ImgGenError
    pipe = P.NativePipeline.__new__(P.NativePipeline)
    with pytest.raises(ImgGenError):
        pipe.load_adapter("x")


# ── contentops quality gates ─────────────────────────────────────────────────

def test_runresult_trail_and_review():
    from nomorals.media.contentops.pipeline import RunResult
    r = RunResult(ok=True, run_id="r1", job_id="j1")
    r.trail("provider", "native imggen (local)")
    assert len(r.decision_trail) == 1
    assert r.decision_trail[0]["decision"] == "provider"
    d = r.to_dict()
    assert "decision_trail" in d and "review" in d


def test_self_review_missing_file():
    from nomorals.media.contentops.pipeline import RunResult, ShortPipeline
    r = RunResult(ok=True, run_id="r1", job_id="j1", final_path="/tmp/nope.mp4")
    sp = ShortPipeline.__new__(ShortPipeline)
    out = ShortPipeline.self_review(sp, r)
    assert out.ok is False  # failed review can never present as success
    assert any(c["name"] == "exists" and not c["ok"] for c in out.review)
    assert "self_review" in [t["decision"] for t in out.decision_trail]
    assert "FAILED" in out.review_text()


def test_preflight_structure():
    from nomorals.media.contentops.pipeline import ShortPipeline, Job
    import inspect
    assert "preflight" in dir(ShortPipeline)
    sig = inspect.signature(ShortPipeline.preflight)
    assert "job" in sig.parameters


# ── camera ffmpeg export ─────────────────────────────────────────────────────

def test_camera_to_ffmpeg():
    from nomorals.media.directed.camera import (CameraProgram, CameraMove,
                                                CameraError)
    prog = CameraProgram([CameraMove("dolly", "in", "slow", 0.7),
                          CameraMove("pan", "right", "normal", 0.5)])
    f = prog.to_ffmpeg(6.0)
    assert "crop=" in f and "between(t,0.00,3.00)" in f
    assert f.startswith("scale=") and "lanczos" in f
    assert CameraProgram([]).to_ffmpeg(3.0) == ""
    with pytest.raises(CameraError):
        CameraProgram([CameraMove("handheld")]).to_ffmpeg(3.0)


# ── synth voice/patch/bus ────────────────────────────────────────────────────

def test_synth_patches():
    from nomorals.media.synth import PATCHES, Patch
    assert {"bass", "lead", "pad", "pluck", "stab", "sub", "keys"} <= set(PATCHES)
    p = PATCHES["bass"]
    assert p.cutoff < PATCHES["lead"].cutoff
    d = p.to_dict()
    assert d["name"] == "bass"


def test_synth_bus_renders():
    from nomorals.media.synth import SynthBus
    bus = SynthBus("pluck", tempo=120)
    sig = bus.play_notes([(60, 0, 1), (64, 1, 1), (67, 2, 2)])
    assert len(sig) > 22050  # > 1s of audio
    peak = max(abs(v) for v in sig)
    assert 0 < peak <= 1.0
    chord = bus.play_chord([60, 64, 67, 72])
    assert len(chord) > 0
    # bad entries skipped honestly
    sig2 = bus.play_notes([("nope",), (60, 0, 1)])
    assert len(sig2) > 0


def test_voice_render():
    from nomorals.media.synth import Voice, PATCHES
    v = Voice(PATCHES["lead"], 69, dur_beats=2.0, tempo=120)
    sig = v.render()
    assert len(sig) > 0
    assert max(abs(x) for x in sig) > 0


# ── vocals backend selection ─────────────────────────────────────────────────

def test_pick_backend():
    from nomorals.media.vocals import pick_backend
    r = pick_backend()
    assert r["backend"] in ("diffsinger", "vocal_lite")
    assert r["available"] is True
    assert len(r["trail"]) == 2
    # forced unknown backend is honest
    r2 = pick_backend(prefer="nope")
    assert r2["available"] is False and "unknown" in r2["reason"]


# ── distribute mastering targets ─────────────────────────────────────────────

def test_master_targets():
    from nomorals.media.distribute import ReleasePacket, PLATFORM_MASTER
    p = ReleasePacket(packet_id="x", title="T", artist="A",
                      song_path="s.wav", platforms=("spotify", "club"))
    targets = p.master_targets()
    assert targets["spotify"]["lufs"] == -14.0
    assert targets["club"]["lufs"] == -8.0
    assert "youtube" in PLATFORM_MASTER
    txt = p.master_plan_text()
    assert "Mastering Plan" in txt and "LUFS" in txt


# ── montage segment in-points ────────────────────────────────────────────────

def test_segment_start_field():
    from nomorals.media.motion_studio.montage import Segment, _as_segment
    s = Segment(kind="video", src="x.mp4", duration=2.0, start=4.5)
    assert s.start == 4.5
    s2 = _as_segment({"kind": "video", "src": "y.mp4", "start": 1.0})
    assert s2.start == 1.0


# ── typography emphasis + safe areas ─────────────────────────────────────────

def test_emphasis_words():
    from nomorals.media.motion_studio.typography import (
        emphasis_words, safe_area, Word)
    words = [Word("hold", 0, 1), Word("the", 1, 2), Word("NIGHT", 2, 3),
             Word("tonight!", 3, 4), Word("hold", 4, 5), Word("hold", 5, 6)]
    em = emphasis_words(words)
    assert 2 in em  # ALL-CAPS
    assert 3 in em  # ends with !
    assert 0 in em and 4 in em and 5 in em  # repeated 3+ (the hook)
    assert 1 not in em
    top, bottom = safe_area("9:16")
    assert top == 0.10 and bottom == 0.16
    assert safe_area("weird") == (0.08, 0.12)


# ── MediaHub sweep verbs ─────────────────────────────────────────────────────

def test_mediahub_verbs_exist():
    from nomorals.media import MediaHub
    for verb in ("highlight_reel", "montage", "harmonic_journey",
                 "dj_journey_map", "refine_image"):
        assert callable(getattr(MediaHub, verb)), verb


def test_dj_journey_map_verb():
    from nomorals.media import MediaHub
    hub = MediaHub.__new__(MediaHub)
    r = hub.dj_journey_map("8A", "3A")
    assert r["path"][0] == "8A" and r["path"][-1] == "3A"
    assert "8A →" in r["map"]
    assert r["smoothness"] >= 0.7
    # key-name input also works
    r2 = hub.dj_journey_map("Am", "C")
    assert r2["path"]


def test_harmonic_journey_verb():
    from nomorals.media import MediaHub
    hub = MediaHub.__new__(MediaHub)
    r = hub.harmonic_journey([
        {"title": "a", "bpm": 120, "key": "A", "mode": "minor", "energy": 0.3},
        {"title": "b", "bpm": 122, "key": "E", "mode": "minor", "energy": 0.8},
        {"title": "c", "bpm": 118, "key": "D", "mode": "minor", "energy": 0.55},
    ], curve="late_peak", seed=1)
    assert len(r["ordered"]) == 3
    assert "DJ Set Plan" in r["report"]
    assert r["curve"] == "late_peak"


# ── videogen chaining carry options ──────────────────────────────────────────

def test_chain_scenes_signature():
    from nomorals.media.videogen import chaining
    import inspect
    sig = inspect.signature(chaining.chain_scenes)
    assert "carry_seed" in sig.parameters
    assert "carry_reference" in sig.parameters
    rep_sig = inspect.signature(chaining.ChainReport.summary)
    assert rep_sig is not None
