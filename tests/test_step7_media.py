"""Tests for Step 7 Media: transcript→EditSpec, pure render, voice
direction, gen ops, faceswap guard, render queue (mission queue).

Every module has one invariant test (marked INVARIANT) plus behavior
tests. No test touches ffmpeg, a model, or the network: renders use a
fake renderer, swaps use a fake swap, the queue uses a real
MissionStore on an in-memory DB.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest
from PIL import Image

from nomorals.media.edit_engine import (
    EditSpec,
    apply_genop,
    apply_pack,
    canonical_spec_dict,
    compile_transcript_edit,
    genop_to_spec,
    keeps_for,
    proxy_spec,
    render_cache_key,
    render_proxy_then_full,
    render_pure,
    validate_genop,
)
from nomorals.media_edit.captions import Word
from nomorals.media_edit.faceswap_guard import (
    FaceSwapGuardError,
    FaceswapRequest,
    guarded_swap,
    verify_guarded_output,
)
from nomorals.media_edit.videos import MediaEditError
from nomorals.missions import MissionStore
from nomorals.missions.render_queue import (
    advance_render,
    enqueue_render,
    render_mission_status,
)
from nomorals.storage.db import Database
from nomorals.voice.direction import (
    DirectionCue,
    VoiceDirectionTrack,
    render_for_tts,
    segments,
    synthesize_with_direction,
)


def W(s, e, t):
    return Word(start=s, end=e, text=t)


@pytest.fixture()
def words():
    return [
        W(0.0, 0.4, "hello"), W(0.4, 0.6, "um"), W(0.6, 1.0, "world"),
        W(1.0, 1.4, "this"), W(1.4, 1.6, "is"), W(1.6, 2.0, "an"),
        W(2.0, 2.4, "umbrella"), W(2.4, 2.8, "uh"), W(2.8, 3.2, "test"),
        W(3.2, 3.8, "you"), W(3.8, 4.2, "know"), W(4.2, 4.6, "yeah"),
    ]


@pytest.fixture()
def store():
    db = Database(":memory:")
    db.migrate()
    return MissionStore(db)


def _fake_renderer_factory(calls: list):
    def fake(spec: EditSpec) -> str:
        calls.append(spec)
        out = Path(spec.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake-video:" + spec.to_dict().__str__().encode()[:8])
        return str(out)
    return fake


# ── transcript → EditSpec ────────────────────────────────────────────────────

class TestTranscriptCompiler:
    def test_remove_fillers_keeps(self, words):
        keeps = keeps_for(words, duration=4.6,
                          ops=[{"op": "remove_fillers", "pad": 0.0}])
        # fillers at (0.4,0.6), (2.4,2.8), (3.2,4.2) cut out
        assert keeps == [(0.0, 0.4), (0.6, 2.4), (2.8, 3.2), (4.2, 4.6)]

    def test_umbrella_is_not_um(self, words):
        keeps = keeps_for(words, duration=4.6,
                          ops=[{"op": "remove_fillers", "pad": 0.0}])
        assert keeps[1] == (0.6, 2.4)  # umbrella span survives

    def test_keep_topic(self, words):
        keeps = keeps_for(words, duration=4.6,
                          ops=[{"op": "keep_topic", "topic": "umbrella",
                                "context_s": 0.0}])
        assert keeps == [(2.0, 2.4)]

    def test_cut_ranges_and_keep_ranges(self, words):
        keeps = keeps_for(words, duration=4.6,
                          ops=[{"op": "cut_ranges", "ranges": [[0.0, 1.0]]}])
        assert keeps == [(1.0, 4.6)]
        keeps = keeps_for(words, duration=4.6,
                          ops=[{"op": "keep_ranges", "ranges": [[1.0, 2.0]]}])
        assert keeps == [(1.0, 2.0)]

    def test_unknown_op_and_empty_edit_raise(self, words):
        with pytest.raises(MediaEditError):
            keeps_for(words, duration=4.6, ops=[{"op": "nope"}])
        with pytest.raises(MediaEditError):
            keeps_for(words, duration=4.6,
                      ops=[{"op": "cut_ranges", "ranges": [[0.0, 4.6]]}])

    def test_invariant_segments_cover_exactly_the_keeps(self, words):
        """INVARIANT: spec segments == keep-ranges, same bounds, same total."""
        spec = compile_transcript_edit(
            "clip.mp4", words, duration=4.6,
            ops=[{"op": "remove_fillers", "pad": 0.1}])
        keeps = keeps_for(words, duration=4.6,
                          ops=[{"op": "remove_fillers", "pad": 0.1}])
        assert len(spec.clips) == len(keeps) == len(spec.segments or [])
        for clip, (s, e), (idx, dur) in zip(spec.clips, keeps,
                                            spec.segments or []):
            assert clip.path == "clip.mp4"
            assert (clip.start, clip.end) == (s, e)
            assert dur == pytest.approx(e - s)
        total = sum(e - s for s, e in keeps)
        assert sum(d for _, d in (spec.segments or [])) == pytest.approx(total)
        assert all(0 <= s < e <= 4.6 for s, e in keeps)


# ── pure render ──────────────────────────────────────────────────────────────

def _spec(**over):
    base = {"clips": [{"path": "a.mp4", "start": 0.0, "end": 2.0}],
            "segments": [[0, 2.0]], "fps": 30, "width": 1080,
            "height": 1920, "seed": 0}
    base.update(over)
    return base


class TestPureRender:
    def test_canonical_strips_profile_and_out(self):
        a = canonical_spec_dict(_spec(profile="fast", out="/tmp/x.mp4"))
        b = canonical_spec_dict(_spec(profile="slow", out="/tmp/y.mp4"))
        assert a == b
        assert "profile" not in a and "out" not in a

    def test_invariant_same_triple_same_key(self, tmp_path):
        """INVARIANT: cache key depends only on (sources, spec, pack)."""
        src = tmp_path / "a.mp4"
        src.write_bytes(b"data")
        k1 = render_cache_key([src], _spec(profile="fast"), {"style": "phonk"})
        k2 = render_cache_key([src], _spec(profile="slow"), {"style": "phonk"})
        k3 = render_cache_key([src], _spec(), {"style": "vlog"})
        assert k1 == k2  # profile is a machine hint, not identity
        assert k1 != k3  # pack swap = different look = different key

    def test_proxy_spec_scales_to_480p_even(self):
        proxy = proxy_spec(_spec())
        assert proxy["height"] == 480
        assert proxy["width"] == 270
        assert proxy["width"] % 2 == 0 and proxy["height"] % 2 == 0
        assert proxy["segments"] == [[0, 2.0]]  # same cut

    def test_apply_pack_validates_style_and_keys(self):
        with pytest.raises(MediaEditError):
            apply_pack(_spec(), {"style": "nope"})
        with pytest.raises(MediaEditError):
            apply_pack(_spec(), {"spec_overrides": {"clips": []}})
        out = apply_pack(_spec(), {"style": "phonk",
                                   "spec_overrides": {"crf": 20}})
        assert out["crf"] == 20
        assert _spec().get("crf") is None  # input untouched

    def test_render_pure_caches(self, tmp_path):
        calls: list = []
        src = tmp_path / "a.mp4"
        src.write_bytes(b"data")
        r1 = render_pure([src], _spec(), {"style": "minimal"},
                         cache_dir=tmp_path / "cache",
                         renderer=_fake_renderer_factory(calls))
        assert r1["cached"] is False and len(calls) == 1
        r2 = render_pure([src], _spec(), {"style": "minimal"},
                         cache_dir=tmp_path / "cache",
                         renderer=_fake_renderer_factory(calls))
        assert r2["cached"] is True and len(calls) == 1  # no re-render
        assert r1["output"] == r2["output"]
        assert Path(r1["output"]).exists()

    def test_render_proxy_then_full(self, tmp_path):
        calls: list = []
        src = tmp_path / "a.mp4"
        src.write_bytes(b"data")
        res = render_proxy_then_full([src], _spec(), {"style": "minimal"},
                                     cache_dir=tmp_path / "cache",
                                     renderer=_fake_renderer_factory(calls))
        assert len(calls) == 2  # proxy + full
        assert res["proxy"]["output"] != res["full"]["output"]
        # second run: both cached
        res2 = render_proxy_then_full([src], _spec(), {"style": "minimal"},
                                      cache_dir=tmp_path / "cache",
                                      renderer=_fake_renderer_factory(calls))
        assert res2["proxy"]["cached"] and res2["full"]["cached"]
        assert len(calls) == 2


# ── voice direction ──────────────────────────────────────────────────────────

class TestVoiceDirection:
    def test_segments_split_by_cue(self):
        track = VoiceDirectionTrack(
            cues=[DirectionCue(word_start=2, word_end=4, emotion="excited")])
        segs = segments("one two three four five", track)
        assert [s["text"] for s in segs] == ["one two", "three four", "five"]
        assert segs[1]["emotion"] == "excited"
        assert segs[0]["emotion"] == "neutral"

    def test_overlapping_cues_raise(self):
        with pytest.raises(Exception):
            VoiceDirectionTrack(cues=[
                DirectionCue(word_start=0, word_end=3),
                DirectionCue(word_start=2, word_end=4)])

    def test_invariant_join_reproduces_dialogue(self):
        """INVARIANT: joining segments reproduces the dialogue verbatim."""
        dialogue = "hello brave new world of sound"
        track = VoiceDirectionTrack(
            cues=[DirectionCue(word_start=1, word_end=3, emotion="sad",
                               pause_after_s=0.5),
                  DirectionCue(word_start=4, word_end=6, emotion="excited")])
        segs = segments(dialogue, track)
        assert " ".join(s["text"] for s in segs) == " ".join(dialogue.split())
        assert " ".join(s["text"] for s in segments(dialogue,
                        VoiceDirectionTrack())) == " ".join(dialogue.split())

    def test_render_for_tts_keeps_dialogue_separate(self):
        track = VoiceDirectionTrack(
            cues=[DirectionCue(word_start=0, word_end=2, emotion="excited")])
        out = render_for_tts("hello world again", track, backend="plain")
        assert out[0]["text"] == "hello world"
        assert "rendered" in out[0] and "emotion" in out[0]

    def test_synthesize_passes_dialogue_and_track_separately(self):
        seen: dict = {}

        def fake_tts(*, rendered_segments, dialogue, track, **kw):
            seen["dialogue"] = dialogue
            seen["track"] = track
            seen["segments"] = rendered_segments
            return "audio-bytes"

        track = VoiceDirectionTrack(
            cues=[DirectionCue(word_start=0, word_end=1, emotion="excited")])
        res = synthesize_with_direction("hello world", track,
                                        tts_fn=fake_tts, backend="plain")
        assert res == "audio-bytes"
        assert seen["dialogue"] == "hello world"  # verbatim, undamaged
        assert seen["track"]["cues"][0]["emotion"] == "excited"
        assert seen["segments"][0]["text"] == "hello"


# ── gen ops ──────────────────────────────────────────────────────────────────

class TestGenOps:
    def test_motion_still_compiles(self):
        spec = genop_to_spec({"op": "motion_still", "image": "still.jpg",
                              "duration": 3.0, "move": "zoom_in"})
        assert spec.clips[0].path == "still.jpg"
        assert spec.clips[0].effects[0].name == "ken_burns"
        assert spec.segments == [(0, 3.0)]

    def test_restyle_compiles(self):
        spec = genop_to_spec({"op": "restyle", "source": "clip.mp4",
                              "grade": "teal_orange", "duration": 2.0})
        assert spec.clips[0].effects[0].name == "color_grade"
        assert spec.clips[0].effects[0].params["grade"] == "teal_orange"

    def test_apply_genop_appends_and_restyle_targets(self):
        base = genop_to_spec({"op": "motion_still", "image": "a.jpg",
                              "duration": 2.0})
        out = apply_genop(base, {"op": "restyle", "source": "a.jpg",
                                 "grade": "noir"})
        assert out.clips[0].effects[-1].name == "color_grade"
        out2 = apply_genop(base, {"op": "motion_still", "image": "b.jpg",
                                  "duration": 1.0})
        assert len(out2.clips) == 2 and out2.segments == [(0, 2.0), (1, 1.0)]

    def test_validation_rejects_unknowns(self):
        with pytest.raises(MediaEditError):
            validate_genop({"op": "teleport"})
        with pytest.raises(MediaEditError):
            validate_genop({"op": "restyle", "source": "x.mp4",
                            "grade": "chartreuse"})
        with pytest.raises(MediaEditError):
            validate_genop({"op": "motion_still", "image": "x.jpg",
                            "move": "sideways"})

    def test_invariant_compile_is_deterministic(self):
        """INVARIANT: same op → equal spec dicts, all effects registered."""
        from nomorals.media.edit_engine import EFFECTS
        op = {"op": "motion_still", "image": "a.jpg", "duration": 2.0,
              "move": "zoom_in"}
        a = genop_to_spec(op).to_dict()
        b = genop_to_spec(op).to_dict()
        assert a == b
        for clip in genop_to_spec(op).clips:
            for fx in clip.effects:
                assert fx.name in EFFECTS


# ── faceswap guard ───────────────────────────────────────────────────────────

def _png(path: Path, color: tuple[int, int, int]) -> Path:
    img = Image.new("RGB", (64, 64), color)
    img.save(str(path))
    return path


class TestFaceswapGuard:
    def test_consent_required(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        src = _png(vault / "ref.png", (255, 0, 0))
        tgt = _png(tmp_path / "tgt.png", (0, 0, 255))
        req = FaceswapRequest(source_ref=src, target=tgt, vault_dir=vault,
                              identity_id="id-1", consent=False)
        with pytest.raises(FaceSwapGuardError):
            guarded_swap(req, swap_fn=lambda s, t, restore=True: t.copy())

    def test_reference_must_be_in_vault(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        src = _png(tmp_path / "outside.png", (255, 0, 0))
        tgt = _png(tmp_path / "tgt.png", (0, 0, 255))
        req = FaceswapRequest(source_ref=src, target=tgt, vault_dir=vault,
                              identity_id="id-1", consent=True)
        with pytest.raises(FaceSwapGuardError):
            guarded_swap(req, swap_fn=lambda s, t, restore=True: t.copy())

    def test_vault_inside_repo_refused(self, tmp_path):
        repo_root = Path(__file__).resolve().parent.parent
        tgt = _png(tmp_path / "tgt.png", (0, 0, 255))
        req = FaceswapRequest(source_ref=repo_root / "x.png", target=tgt,
                              vault_dir=repo_root, identity_id="id-1",
                              consent=True)
        with pytest.raises(FaceSwapGuardError):
            guarded_swap(req, swap_fn=lambda s, t, restore=True: t.copy())

    def test_identity_lock_blocks_retarget(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        ref1 = _png(vault / "ref1.png", (255, 0, 0))
        ref2 = _png(vault / "ref2.png", (0, 255, 0))
        tgt = _png(tmp_path / "tgt.png", (0, 0, 255))
        swap = lambda s, t, restore=True: t.copy()
        guarded_swap(FaceswapRequest(source_ref=ref1, target=tgt,
                                      vault_dir=vault, identity_id="id-1",
                                      consent=True), swap_fn=swap)
        with pytest.raises(FaceSwapGuardError):
            guarded_swap(FaceswapRequest(source_ref=ref2, target=tgt,
                                          vault_dir=vault, identity_id="id-1",
                                          consent=True), swap_fn=swap)

    def test_invariant_sidecar_hash_matches_output(self, tmp_path):
        """INVARIANT: sidecar output_sha256 == sha256(output file)."""
        vault = tmp_path / "vault"
        vault.mkdir()
        ref = _png(vault / "ref.png", (255, 0, 0))
        tgt = _png(tmp_path / "tgt.png", (0, 0, 255))
        res = guarded_swap(
            FaceswapRequest(source_ref=ref, target=tgt, vault_dir=vault,
                            identity_id="id-1", consent=True,
                            watermark_text="test-swap"),
            swap_fn=lambda s, t, restore=True: t.copy())
        sidecar = json.loads(Path(res["sidecar"]).read_text())
        actual = hashlib.sha256(Path(res["output"]).read_bytes()).hexdigest()
        assert sidecar["output_sha256"] == actual == res["output_sha256"]
        assert sidecar["watermark"] == res["watermark"]
        assert verify_guarded_output(res["output"])["ok"] is True
        # tamper → verification fails
        Path(res["output"]).write_bytes(b"tampered")
        with pytest.raises(FaceSwapGuardError):
            verify_guarded_output(res["output"])


# ── render queue = mission queue ─────────────────────────────────────────────

class TestRenderQueue:
    def test_enqueue_advance_resume(self, store, tmp_path):
        calls: list = []
        src = tmp_path / "a.mp4"
        src.write_bytes(b"data")

        def render_fn(*, stage, state):
            calls.append(stage)
            return {"output": f"/tmp/{stage}.mp4"}

        mission = enqueue_render(store, sources=[src], spec=_spec(),
                                 pack={"style": "minimal"})
        assert "render" in mission.tags and "media" in mission.tags
        r1 = advance_render(store, mission.id, render_fn=render_fn)
        assert r1["stage"] == "proxy" and r1["done"] is False
        # simulate a crash: a fresh advance resumes at the next stage
        r2 = advance_render(store, mission.id, render_fn=render_fn)
        assert r2["stage"] == "full" and r2["done"] is True
        assert calls == ["proxy", "full"]  # proxy never re-ran
        status = render_mission_status(store, mission.id)
        assert status["completed_stages"] == ["full", "proxy"]
        assert status["done"] is True
        # checkpoints recorded
        assert len(store.checkpoint_history(mission.id)) == 2

    def test_invariant_completed_stages_never_rerun(self, store, tmp_path):
        """INVARIANT: crash-resume replays from the last checkpoint."""
        calls: list = []
        src = tmp_path / "a.mp4"
        src.write_bytes(b"data")

        def render_fn(*, stage, state):
            calls.append(stage)
            if stage == "proxy" and len(calls) == 1:
                raise RuntimeError("simulated crash mid-proxy")
            return {"output": f"/tmp/{stage}.mp4"}

        mission = enqueue_render(store, sources=[src], spec=_spec())
        with pytest.raises(RuntimeError):
            advance_render(store, mission.id, render_fn=render_fn)
        # proxy did not complete → it runs again; nothing is half-recorded
        assert render_mission_status(store, mission.id)["completed_stages"] == []
        advance_render(store, mission.id, render_fn=render_fn)
        advance_render(store, mission.id, render_fn=render_fn)
        assert calls == ["proxy", "proxy", "full"]

    def test_budget_exhausted_fails_without_rendering(self, store, tmp_path):
        calls: list = []
        src = tmp_path / "a.mp4"
        src.write_bytes(b"data")

        def render_fn(*, stage, state):
            calls.append(stage)
            return {"output": "/tmp/x.mp4"}

        mission = enqueue_render(store, sources=[src], spec=_spec(),
                                 budget_wall=0.001)
        # spend the budget directly, then advance
        m = store.get(mission.id)
        m.charge(wall=1.0)
        store.save(m)
        res = advance_render(store, mission.id, render_fn=render_fn)
        assert res["status"] == "failed"
        assert calls == []
