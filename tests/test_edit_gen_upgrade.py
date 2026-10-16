"""EDIT_GEN_UPGRADE tests: 2.1–2.10. Pure / offline — no ffmpeg, no models.

Covers: one document (views, cross-modal, hash, undo, diff, A/B),
intent front door + learning, non-destructive derivatives,
render queue (resume, profile guard, honest failure, work-stealing),
identity + look lock + continuity gate, shot semantic search,
audio missing middle (emotion, stems, loudness, spatial),
image four, video four, quality gate.
"""

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nomorals.media.edit_engine import (
    Document, DocTrack, UndoStack, diff_documents,
    image_document, audio_document, voiceover_document, transcript_document,
    RenderQueue, Segment, segments_for_ranges, resolve_render_profile,
    RenderFailure,
    EmotionOp, LoudnessSpec, loudness_spec_for, check_loudness,
    SpatialParams, spatial_filter, stem_mix, StemSet,
    relight_after_composite, multi_ref_identity, pose_control,
    reframe, reframe_dims, match_cut_score, tighten_keeps, tighten,
    QualityGate, quality_gate, auto_fix,
    Clip, Effect,
)
from nomorals.media_edit.intent import (
    parse_instruction, parse_document_intent, CorrectionStore,
    INTENT_PRESET_STACKS,
)
from nomorals.media.videogen.consistency import (
    IdentityLock, LookLock, palette_distance, embedding_similarity,
    continuity_check,
)
from nomorals.media.directed.validate import (
    validate_identity_continuity, validate_look_lock,
    validate_document_continuity,
)
from nomorals.media.library import ClipIndex


WORDS = [
    {"start": 0.0, "end": 1.0, "text": "we"},
    {"start": 1.0, "end": 2.0, "text": "say"},
    {"start": 2.0, "end": 3.0, "text": "never"},
    {"start": 3.0, "end": 4.0, "text": "again"},
    {"start": 4.0, "end": 5.0, "text": "um"},
    {"start": 5.0, "end": 8.0, "text": "today"},
]


def _doc():
    return Document(
        tracks=[DocTrack(kind="video", clips=[
            Clip(path="clip.mp4", start=0.0, end=8.0, duration=8.0)])],
        transcript=list(WORDS))


class TestOneDocument(unittest.TestCase):
    def test_specialisations_are_views(self):
        img = image_document("pic.jpg")
        self.assertEqual(img.track("image").clips[0].path, "pic.jpg")
        aud = audio_document(["a.wav", "b.wav"])
        self.assertEqual(len(aud.track("audio").clips), 2)
        self.assertIsNone(aud.track("video"))
        vo = voiceover_document("vo.wav", WORDS)
        self.assertEqual(vo.track("voice").clips[0].path, "vo.wav")
        self.assertEqual(len(vo.transcript), 6)
        td = transcript_document("clip.mp4", WORDS)
        self.assertEqual(td.transcript[2]["text"], "never")

    def test_roundtrip_and_hash(self):
        doc = _doc()
        doc2 = Document.from_dict(json.loads(json.dumps(doc.to_dict())))
        self.assertEqual(doc2.to_dict(), doc.to_dict())
        self.assertEqual(doc.compute_hash(), doc2.compute_hash())
        doc2.title = "changed"
        self.assertNotEqual(doc.compute_hash(), doc2.compute_hash())

    def test_cross_modal_cut_and_bed(self):
        doc = _doc()
        doc.cut_video_where_said("never again")
        self.assertEqual(len(doc.track("video").clips), 1)
        self.assertEqual(doc.track("video").clips[0].start, 2.0)
        self.assertEqual(doc.track("video").clips[0].end, 4.0)
        self.assertTrue(doc.removed)  # removed is recoverable
        doc.use_audio_as_music_bed("clip.mp4", start=2.0, end=4.0)
        self.assertIsNotNone(doc.track("music"))
        self.assertTrue(doc.audio_mix.layers)
        # restore the removed range
        doc.restore_removed()
        self.assertEqual(len(doc.track("video").clips), 2)

    def test_undo_redo_diff_ab(self):
        doc = _doc()
        stack = UndoStack(doc)
        stack.push(doc)
        doc2 = doc.copy()
        doc2.title = "v2"
        stack.push(doc2)
        self.assertTrue(stack.can_undo)
        back = stack.undo()
        self.assertEqual(back.title, "")
        fwd = stack.redo()
        self.assertEqual(fwd.title, "v2")
        d = diff_documents(doc, doc2)
        self.assertIn("title", d["changed"])
        # A/B: same edit, different look only
        a = doc.ab_variant({"grade": "warm", "grain": 4.0})
        b = doc.ab_variant({"grade": "cool", "grain": 0.0})
        self.assertEqual(a.track("video").clips, b.track("video").clips)
        self.assertNotEqual(a.look_lock, b.look_lock)
        # with_output: new derivative, same edit
        proxy = doc.with_output(width=540, height=960, profile="termux")
        self.assertEqual(proxy.track("video").clips,
                         doc.track("video").clips)
        self.assertEqual(proxy.output["width"], 540)
        der = doc.derivative(profile="termux", output_path="out.mp4")
        self.assertEqual(der["doc_version"], doc.version)

    def test_to_edit_spec(self):
        spec = _doc().to_edit_spec()
        self.assertEqual(spec.clips[0].path, "clip.mp4")
        back = Document.from_edit_spec(spec)
        self.assertEqual(back.track("video").clips[0].path, "clip.mp4")


class TestIntentFrontDoor(unittest.TestCase):
    def test_sunday_morning(self):
        hit = parse_document_intent("make it feel Sunday morning")
        self.assertIsNotNone(hit)
        self.assertEqual(hit.kind, "document")
        self.assertTrue(any(o.get("op") == "look_lock" for o in hit.ops))
        self.assertIn("sunday_morning", INTENT_PRESET_STACKS)

    def test_tired_voice(self):
        hit = parse_document_intent("make him sound tired")
        self.assertTrue(any(o.get("emotion") == "tired" for o in hit.ops))

    def test_tighten_third(self):
        hit = parse_document_intent("tighten by a third")
        self.assertTrue(any(o.get("op") == "tighten" for o in hit.ops))
        self.assertAlmostEqual(
            [o for o in hit.ops if o.get("op") == "tighten"][0]["fraction"],
            1 / 3)
        hit2 = parse_document_intent("tighten by 50%")
        self.assertAlmostEqual(
            [o for o in hit2.ops if o.get("op") == "tighten"][0]["fraction"],
            0.5)

    def test_match_look_and_cross_modal(self):
        self.assertIsNotNone(
            parse_document_intent("match look of that shot"))
        cut = parse_document_intent(
            "cut video where he says 'never again'")
        self.assertTrue(any(o.get("op") == "cut_where_said"
                            for o in cut.ops))
        bed = parse_document_intent("use that audio as music bed")
        self.assertTrue(any(o.get("op") == "use_audio_as_music_bed"
                            for o in bed.ops))

    def test_parse_instruction_front_door(self):
        hit = parse_instruction("make it feel Sunday morning")
        self.assertEqual(hit.kind, "document")

    def test_corrections_learn(self):
        store = CorrectionStore()
        before = [{"op": "grade", "temperature": 0}]
        after = store.learn_from_correction("no, warmer", before)
        self.assertTrue(any(o.get("temperature") == 800 for o in after))
        store.record_correction("make it warmer", "no, warmer",
                                before, after)
        self.assertEqual(store.preferences()["pairs"], 1)
        self.assertEqual(store.preferences()["corrections"]["no, warmer"], 1)
        with tempfile.TemporaryDirectory() as td:
            p = str(Path(td) / "corr.json")
            store.save(p)
            self.assertEqual(CorrectionStore.load(p).preferences()["pairs"], 1)


class TestRenderQueue(unittest.TestCase):
    def test_profile_guard_never_silent(self):
        r = resolve_render_profile(profile="termux", height=1080)
        self.assertTrue(r["degraded"])
        self.assertEqual(r["height"], 480)
        self.assertIn("allow_full_res", r["reason"])
        r2 = resolve_render_profile(profile="termux", height=1080,
                                    allow_full_res=True)
        self.assertFalse(r2["degraded"])
        self.assertEqual(r2["height"], 1080)
        r3 = resolve_render_profile(profile="workstation", height=1080)
        self.assertFalse(r3["degraded"])

    def test_resume_and_progress(self):
        with tempfile.TemporaryDirectory() as td:
            state = str(Path(td) / "q.json")
            q = RenderQueue(segments_for_ranges([(0, 2), (2, 4), (4, 6)]),
                            state_path=state)
            calls = []
            def fn(seg):
                calls.append(seg.id)
                return f"{seg.id}.mp4"
            events = list(q.run(fn, worker_id="w1"))
            self.assertEqual(q.progress, 1.0)
            self.assertEqual(len(calls), 3)
            # resume: nothing left to do
            q2 = RenderQueue(state_path=state)
            self.assertEqual(q2.progress, 1.0)
            events2 = list(q2.run(fn, worker_id="w1"))
            self.assertEqual(len(calls), 3)  # no re-render
            self.assertEqual(events2[-1]["state"], "finished")

    def test_honest_failure_and_steal(self):
        q = RenderQueue(segments_for_ranges([(0, 1)]),
                        max_attempts=1)
        def boom(seg):
            raise RuntimeError("CUDA out of memory on model X")
        with self.assertRaises(RenderFailure) as ctx:
            list(q.run(boom, worker_id="w1"))
        self.assertIn("CUDA", str(ctx.exception))
        self.assertEqual(q.failed[0].id, "seg-0")
        # stale claim is stolen by another worker
        q2 = RenderQueue([Segment(id="s1", start=0, end=1,
                                  state="claimed", worker="dead",
                                  claimed_at=0.0, attempts=1)],
                         claim_ttl_s=1.0, max_attempts=3)
        stolen = q2.claim("w2", location="mesh-node-2")
        self.assertIsNotNone(stolen)
        self.assertEqual(stolen.worker, "w2")


class TestIdentityLookLock(unittest.TestCase):
    def test_palette_and_embedding(self):
        self.assertAlmostEqual(palette_distance([[0, 0, 0]], [[0, 0, 0]]), 0.0)
        self.assertAlmostEqual(
            embedding_similarity([1, 0], [1, 0]), 1.0)
        self.assertLess(embedding_similarity([1, 0], [0, 1]), 0.01)

    def test_continuity_flags_not_ships(self):
        lock = IdentityLock(character_id="hero",
                            face_embedding=[1.0, 0.0],
                            wardrobe="red jacket",
                            palette=[[200, 30, 30]])
        look = LookLock(name="warm", grade="warm",
                        palette=[[200, 30, 30]], grain=4.0, lens="soft")
        good = {"id": "s1", "character_id": "hero",
                "face_embedding": [1.0, 0.0], "wardrobe": "red jacket",
                "palette": [[200, 30, 30]], "grade": "warm"}
        bad = {"id": "s2", "character_id": "hero",
               "face_embedding": [0.0, 1.0], "wardrobe": "blue coat",
               "palette": [[0, 0, 255]], "grade": "cool"}
        res = continuity_check([good, bad], identity_locks=[lock],
                               look_lock=look)
        self.assertFalse(res["ok"])
        self.assertTrue(all(m["shot"] == "s2" for m in res["mismatches"]))
        rep = validate_identity_continuity([good, bad],
                                           identity_locks=[lock.to_dict()],
                                           look_lock=look.to_dict())
        self.assertFalse(rep.ok)
        self.assertTrue(rep.errors)
        rep2 = validate_identity_continuity([good],
                                             identity_locks=[lock.to_dict()],
                                             look_lock=look.to_dict())
        self.assertTrue(rep2.ok)
        self.assertFalse(validate_look_lock(good, look.to_dict()).errors)
        doc = {"identity_locks": [lock.to_dict()],
               "look_lock": look.to_dict(),
               "shot_measurements": [good]}
        self.assertTrue(validate_document_continuity(doc).ok)
        self.assertTrue(validate_document_continuity({}).warnings)


class TestShotSearch(unittest.TestCase):
    def _index(self):
        idx = ClipIndex()
        idx.index_clip({"clip_id": "a.mp4", "start": 0, "end": 10,
                        "transcript": "he laughs at the joke",
                        "characters": ["hero"], "tags": ["laughing"],
                        "audio_energy": 0.9, "framing_width": 0.8,
                        "aesthetic": 0.7})
        idx.index_clip({"clip_id": "a.mp4", "start": 10, "end": 20,
                        "transcript": "quiet scene",
                        "audio_energy": 0.1, "framing_width": 0.2,
                        "aesthetic": 0.5})
        idx.index_clip({"clip_id": "a.mp4", "start": 20, "end": 30,
                        "transcript": "the music plays on",
                        "audio_energy": 0.85, "framing_width": 0.5})
        return idx

    def test_laugh_query(self):
        hits = self._index().query("find shot where he laughs")
        self.assertEqual(hits[0]["start"], 0)

    def test_widest_last_minute(self):
        hits = self._index().query("widest shot last minute")
        self.assertEqual(hits[0]["framing_width"], 0.2)

    def test_music_drop(self):
        hits = self._index().query("where music drops")
        self.assertTrue(hits)
        self.assertEqual(hits[0]["start"], 10)  # 0.9 -> 0.1 fall
        self.assertEqual(ClipIndex.from_dict(
            self._index().to_dict()).query("laughs")[0]["start"], 0)


class TestAudioMissingMiddle(unittest.TestCase):
    def test_emotion_ops(self):
        op = EmotionOp(line_index=0, start=0.0, end=1.0,
                       emotion="tired", intensity=1.0)
        filt = op.to_ffmpeg_filter()
        self.assertIn("atempo", filt)
        self.assertIn("lowpass", filt)
        ops = __import__("nomorals.media.edit_engine.audio_ops",
                         fromlist=["emotion_timeline"]).emotion_timeline(
            WORDS, "tired")
        self.assertEqual(len(ops), 6)
        with self.assertRaises(Exception):
            EmotionOp(line_index=0, start=0, end=1, emotion="nope")

    def test_stem_mix(self):
        stems = StemSet(source_path="mix.wav", vocals_path="v.wav",
                        drums_path="d.wav", bass_path="b.wav",
                        other_path="o.wav")
        mix = stem_mix(stems, gains={"vocals": 1.2}, solo="vocals")
        self.assertEqual(len(mix.layers), 1)
        self.assertAlmostEqual(mix.layers[0].volume, 1.2)
        mix2 = stem_mix(stems, muted=["drums"])
        self.assertEqual(len(mix2.layers), 3)

    def test_loudness_enforced(self):
        spec = loudness_spec_for("spotify")
        self.assertAlmostEqual(spec.target_lufs, -14.0)
        self.assertTrue(check_loudness(
            {"integrated_lufs": -14.0, "true_peak_dbtp": -1.5}, spec)["ok"])
        bad = check_loudness(
            {"integrated_lufs": -20.0, "true_peak_dbtp": 0.5}, spec)
        self.assertFalse(bad["ok"])
        self.assertAlmostEqual(bad["gain_db"], 6.0)
        self.assertTrue(any("true peak" in i for i in bad["issues"]))
        with self.assertRaises(Exception):
            loudness_spec_for("not-a-platform")

    def test_spatial(self):
        filt = spatial_filter(SpatialParams(pan=-1.0, distance_m=2.0,
                                            room=0.5))
        self.assertIn("pan=", filt)
        self.assertIn("aecho", filt)
        with self.assertRaises(Exception):
            SpatialParams(pan=2.0)


class TestImageFour(unittest.TestCase):
    def test_relight(self):
        from PIL import Image
        img = Image.new("RGB", (32, 32), (128, 128, 128))
        out = relight_after_composite(img, light_dir="top-left",
                                      intensity=0.4, warmth=0.5)
        self.assertEqual(out.size, (32, 32))
        self.assertNotEqual(list(out.getdata())[0],
                            list(img.getdata())[0])
        with self.assertRaises(Exception):
            relight_after_composite(img, light_dir="nowhere")

    def test_pose_control(self):
        res = pose_control("wave", n_frames=4)
        self.assertEqual(res["action"], "wave")
        self.assertEqual(res["control_image"].size, (512, 512))
        self.assertEqual(res["track"].n_frames, 4)

    def test_multi_ref(self):
        ident = multi_ref_identity([
            {"palette": [[10, 20, 30]], "embedding": [1.0, 0.0],
             "wardrobe": "red jacket"},
            {"palette": [[12, 22, 32]], "embedding": [0.9, 0.1],
             "wardrobe": "red jacket"}])
        self.assertEqual(ident["n_refs"], 2)
        self.assertEqual(ident["wardrobe"], "red jacket")
        self.assertAlmostEqual(ident["palette"][0][0], 11.0)
        with self.assertRaises(Exception):
            multi_ref_identity([{"palette": [[0, 0, 0]]}])


class TestVideoFour(unittest.TestCase):
    def test_reframe(self):
        clip = Clip(path="c.mp4", duration=5.0)
        out = reframe(clip, "9:16")
        self.assertTrue(any(e.name == "reframe" for e in out.effects))
        self.assertEqual(reframe_dims("16:9", height=720), (1280, 720))
        tracked = reframe(clip, "1:1",
                          subject_track=[{"t": 0, "bbox": (0.2, 0.2, 0.6, 0.8)}])
        params = [e.params for e in tracked.effects
                  if e.name == "reframe"][0]
        self.assertTrue(params["subject_tracking"])
        self.assertAlmostEqual(params["anchor_x"], 0.4)

    def test_match_cuts(self):
        a = {"id": "a", "motion": 0.05}
        b = {"id": "b", "motion": 0.05}
        res = match_cut_score(a, b)
        self.assertGreater(res["breath"], 0.9)
        self.assertEqual(res["verdict"], "match")

    def test_tighten(self):
        keeps = [(0.0, 8.0)]
        words = [dict(w) for w in WORDS]
        # gap 5->8 is not in words; add explicit pause via word ends
        res = tighten_keeps(keeps, words, fraction=1 / 3)
        self.assertLess(res["duration_after"], res["duration_before"])
        self.assertAlmostEqual(res["removed_fraction"], 1 / 3, places=1)
        self.assertTrue(res["removed"])
        doc = _doc()
        out = tighten(doc, fraction=0.25)
        self.assertTrue(out.removed)
        self.assertLess(out.duration, 8.0)


class TestQualityGate(unittest.TestCase):
    def test_ship_when_all_pass(self):
        doc = _doc()
        doc.loudness = LoudnessSpec(platform="spotify").to_dict()
        rep = quality_gate(doc, {
            "exposure": 0.95, "loudness": {"integrated_lufs": -14.0,
                                          "true_peak_dbtp": -1.5},
            "black_frame_fraction": 0.0, "caption_drift_s": 0.02,
            "identity_scores": [0.9], "aesthetic": 0.8})
        self.assertTrue(rep["ship"])
        self.assertEqual(rep["verdict"], "ship")

    def test_unknown_axes_block_ship_honestly(self):
        rep = quality_gate(_doc(), {})
        self.assertFalse(rep["ship"])
        self.assertTrue(rep["unknown"])
        self.assertTrue(rep["weaknesses"])

    def test_auto_fix(self):
        doc = _doc()
        doc.loudness = LoudnessSpec(platform="spotify").to_dict()
        rep = quality_gate(doc, {
            "exposure": 0.95, "loudness": {"integrated_lufs": -20.0,
                                          "true_peak_dbtp": -1.5},
            "black_frame_fraction": 0.0, "caption_drift_s": 0.02,
            "identity_scores": [0.9], "aesthetic": 0.8})
        self.assertFalse(rep["ship"])
        fixed = auto_fix(doc, rep)
        self.assertTrue(fixed["applied"])
        self.assertAlmostEqual(fixed["document"].loudness["gain_db"], 6.0)


if __name__ == "__main__":
    unittest.main()
