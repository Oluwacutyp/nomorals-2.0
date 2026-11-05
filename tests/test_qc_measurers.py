"""§1.5c tail — QC measurers + the five-pass contact sheet.

The acceptance shape the plan demands: measurers read a *finished
render file* (no hand-fed measurement dicts anywhere in this file —
the only stub is the §2.4 Transcriber protocol seam, the same fake
the caption wave pins), the gate judges what they measured, a
planted defect fails the specific axis it belongs to, and the
contact sheet's frames provably come from the render.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nomorals.media.edit_engine import (  # noqa: E402
    Clip, Document, DocTrack,
)
from nomorals.media.edit_engine.audio import loudness_measure  # noqa: E402
from nomorals.media.edit_engine.audio_ops import LoudnessSpec  # noqa: E402
from nomorals.media.edit_engine.qc import (  # noqa: E402
    CONTACT_SHEET_PASSES, MEASURER_RUNGS, build_contact_sheet,
    measure_render, probe_render, qc_render,
)
from nomorals.media.edit_engine.render import RenderReport  # noqa: E402
from nomorals.media_edit.videos import run_ffmpeg  # noqa: E402

W, H, FPS = 320, 180, 30
WORDS = [{"start": float(i), "end": float(i) + 0.8,
          "text": t}
         for i, t in enumerate(
             ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"])]


class _FakeTranscriber:
    """The §2.4 protocol seam: measured word timings, supplied."""

    def __init__(self, words):
        self._words = words

    def transcribe_words(self, path):
        return [dict(w) for w in self._words]


def _render(tmp_path, name, segments, *, audio="sine", seconds=None):
    """A real finished render via ffmpeg lavfi segments + concat."""
    total = sum(d for _c, d in segments)
    args: list[str] = []
    for color, dur in segments:
        args += ["-f", "lavfi", "-i",
                 f"color=c={color}:s={W}x{H}:r={FPS}:d={dur}"]
    n = len(segments)
    if audio == "sine":
        args += ["-f", "lavfi", "-i",
                 f"sine=frequency=440:duration={total}"]
    elif audio == "silence":
        args += ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo"]
    filt = "".join(f"[{i}:v]" for i in range(n)) + \
        f"concat=n={n}:v=1:a=0[v]"
    args += ["-filter_complex", filt, "-map", "[v]"]
    if audio in ("sine", "silence"):
        args += ["-map", f"{n}:a", "-c:a", "aac", "-shortest"]
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p",
             str(tmp_path / name)]
    run_ffmpeg(args)
    return str(tmp_path / name)


def _good_render(tmp_path):
    return _render(tmp_path, "good.mp4",
                   [("0x7a6a53", 2.0), ("0x536a7a", 2.0),
                    ("0x6a7a53", 2.0)])


def _doc(*, words=WORDS, target_lufs=None, locks=None, seconds=6.0):
    doc = Document(
        tracks=[DocTrack(kind="video", clips=[
            Clip(path="render.mp4", start=0.0, end=seconds,
                 duration=seconds)])],
        transcript=[dict(w) for w in words])
    if target_lufs is not None:
        doc.loudness = LoudnessSpec(target_lufs=target_lufs).to_dict()
    if locks is not None:
        doc.identity_locks = locks
    return doc


def _target_for(path):
    """The document's loudness target, from a reference measurement
    of the fixture (the spec is authored data; the gate still only
    ever sees the QC measurer's own reading of the file)."""
    return round(loudness_measure(path)["input_i"], 2)


# ── measurers read the file ───────────────────────────────────────────────────

class TestMeasureRender:
    def test_all_measurable_axes_measured_from_the_file(self, tmp_path):
        path = _good_render(tmp_path)
        doc = _doc(target_lufs=_target_for(path))
        bundle = measure_render(doc, path,
                                transcriber=_FakeTranscriber(WORDS))
        assert bundle["ok"] is True
        m = bundle["measurements"]
        # every gate input came out of the file — nothing hand-fed
        assert m["exposure"] == 1.0            # muted tones, no clipping
        assert m["black_frame_fraction"] == 0.0
        assert m["loudness"]["integrated_lufs"] == pytest.approx(
            _target_for(path), abs=0.05)
        assert m["caption_drift_s"] == pytest.approx(0.0, abs=0.01)
        assert m["image_features"]["saturation"] > 0.1
        # identity: no locks → not applicable, and NOT in measurements
        assert "identity_scores" not in m
        assert bundle["unmeasured"] == {}
        # every axis names its rung
        assert set(bundle["methods"]) == set(MEASURER_RUNGS)
        assert "studio_color" in bundle["methods"]["exposure_clipping"]
        assert "loudnorm" in bundle["methods"]["loudness"]

    def test_probe_render_facts(self, tmp_path):
        path = _good_render(tmp_path)
        probe = probe_render(path)
        assert probe["duration_s"] == pytest.approx(6.0, abs=0.2)
        assert probe["has_audio"] and probe["has_video"]
        assert (probe["width"], probe["height"]) == (W, H)

    def test_missing_render_is_all_unmeasured_never_raises(self, tmp_path):
        bundle = measure_render(_doc(), tmp_path / "nope.mp4")
        assert bundle["ok"] is False
        assert set(bundle["unmeasured"]) == set(MEASURER_RUNGS)
        res = qc_render(_doc(), tmp_path / "nope.mp4")
        assert res["ship"] is False
        assert res["contact_sheet"]["ok"] is False
        assert res["contact_sheet"]["reason"]

    def test_no_audio_stream_loudness_unmeasured(self, tmp_path):
        path = _render(tmp_path, "mute.mp4", [("0x7a6a53", 3.0)],
                       audio="none")
        bundle = measure_render(_doc(words=[]), path)
        assert "loudness" not in bundle["measurements"]
        assert "no audio stream" in bundle["unmeasured"]["loudness"]

    def test_unmeasured_axis_blocks_ship_from_measured_bundle(
            self, tmp_path):
        # the plan's pin, now against measurer output instead of a
        # hand-fed dict: a doc with no caption words leaves the
        # caption axis unmeasured → the gate cannot ship.
        path = _good_render(tmp_path)
        doc = _doc(words=[], target_lufs=_target_for(path))
        res = qc_render(doc, path, transcriber=_FakeTranscriber(WORDS))
        assert res["ship"] is False
        assert "caption_sync_drift" in res["gate"]["unknown"]
        assert "caption" in res["measure"]["unmeasured"][
            "caption_sync_drift"]


# ── end-to-end: gate ships a good render, fails a degraded one ───────────────

class TestQcRenderEndToEnd:
    def test_good_render_ships_with_sheet_attached(self, tmp_path):
        path = _good_render(tmp_path)
        doc = _doc(target_lufs=_target_for(path))
        report = RenderReport(output=path)
        res = qc_render(doc, path, report=report,
                        transcriber=_FakeTranscriber(WORDS))
        assert res["ship"] is True
        assert res["gate"]["verdict"] == "ship"
        sheet = res["contact_sheet"]
        assert sheet["ok"] is True
        assert os.path.isfile(sheet["path"])
        assert len(sheet["passes"]) == CONTACT_SHEET_PASSES == 5
        spans = [p["span_s"] for p in sheet["passes"]]
        assert spans[0][0] == 0.0
        assert spans[-1][1] == pytest.approx(6.0, abs=0.01)
        for p in sheet["passes"]:
            assert len(p["frames"]) == 3
            for f in p["frames"]:
                assert p["span_s"][0] - 0.06 <= f["t"] <= p["span_s"][1]
                assert f["grabbed"] is True
        assert sheet["failing_frames"] == []
        # attached to the report in its established shape
        assert report.contact_sheet is sheet
        assert report.to_dict()["contact_sheet"]["kind"] == \
            "five_pass_frame_qc"
        assert report.to_dict()["contact_sheet"]["axes"][
            "loudness"]["status"] == "pass"

    def test_degraded_render_fails_the_specific_axes(self, tmp_path):
        # half the picture black, digital-silence audio, and the
        # file 3s shorter than the document's timeline expects.
        path = _render(tmp_path, "bad.mp4",
                       [("black", 3.0), ("0x7a6a53", 3.0)],
                       audio="silence")
        doc = _doc(target_lufs=-23.0)
        res = qc_render(doc, path, expected_duration=9.0,
                        transcriber=_FakeTranscriber(WORDS))
        gate = res["gate"]
        assert res["ship"] is False
        assert "black_frames" in gate["failed"]
        assert "loudness" in gate["failed"]
        bundle = res["measure"]
        # duration measured honestly against the expectation
        assert bundle["duration"]["delta_s"] == pytest.approx(-3.0,
                                                              abs=0.3)
        assert bundle["duration"]["within_tolerance"] is False
        # black fraction covers the black half AND the missing tail
        assert bundle["measurements"]["black_frame_fraction"] >= 0.6
        assert bundle["evidence"]["black_frames"]["missing"] > 0
        # silence reads at the named floor, not a fabricated number
        assert bundle["measurements"]["loudness"][
            "integrated_lufs"] == -70.0
        assert any("silence" in n for n in
                   bundle["evidence"]["loudness"]["notes"])
        # the sheet shows the frames that failed the black axis
        failing = res["contact_sheet"]["failing_frames"]
        assert failing
        assert all("black_frames" in f["axes"] for f in failing)
        assert all(f["t"] < 3.1 for f in failing)

    def test_caption_drift_fails_its_axis(self, tmp_path):
        path = _good_render(tmp_path)
        doc = _doc(target_lufs=_target_for(path))
        heard = [dict(w, start=w["start"] - 0.45, end=w["end"] - 0.45)
                 for w in WORDS]
        res = qc_render(doc, path, transcriber=_FakeTranscriber(heard))
        assert "caption_sync_drift" in res["gate"]["failed"]
        ev = res["measure"]["evidence"]["caption_sync_drift"]
        assert ev["median_offset_s"] == pytest.approx(-0.45, abs=0.02)
        assert ev["matched"] == len(WORDS)


# ── identity: honest unknown without an embedding rung ───────────────────────

class TestIdentity:
    LOCKS = [{"character_id": "lead", "face_embedding": [1.0, 0.0, 0.0],
              "palette": [[122.0, 106.0, 83.0]], "wardrobe": ""}]

    def test_locks_without_embeddings_stay_unmeasured(self, tmp_path):
        path = _good_render(tmp_path)
        doc = _doc(target_lufs=_target_for(path), locks=self.LOCKS)
        bundle = measure_render(doc, path,
                                transcriber=_FakeTranscriber(WORDS))
        assert "identity_scores" not in bundle["measurements"]
        reason = bundle["unmeasured"]["identity_drift"]
        assert "face-embedding" in reason
        # palette evidence was measured from the frames instead
        pal = bundle["evidence"]["identity_drift"]["palette"]
        assert pal and "lead" in pal[0]["palette_dist"]
        res = qc_render(doc, path, transcriber=_FakeTranscriber(WORDS))
        assert res["ship"] is False
        assert "identity_drift" in res["gate"]["unknown"]

    def test_supplied_embeddings_are_measured(self, tmp_path):
        path = _good_render(tmp_path)
        doc = _doc(target_lufs=_target_for(path), locks=self.LOCKS)
        bundle = measure_render(
            doc, path, transcriber=_FakeTranscriber(WORDS),
            shot_embeddings=[[1.0, 0.0, 0.0]])
        assert bundle["measurements"]["identity_scores"] == [1.0]
        assert "check_identity" in bundle["methods"]["identity_drift"]


# ── the sheet's frames come from the render ──────────────────────────────────

class TestContactSheetContent:
    def test_sheet_pixels_are_the_render_frames(self, tmp_path):
        from PIL import Image
        path = _render(tmp_path, "sheet.mp4",
                       [("0x7a6a53", 4.0), ("black", 2.0)])
        sheet = build_contact_sheet(path, tmp_path / "sheet.png",
                                    duration=6.0)
        assert sheet["ok"] is True
        img = Image.open(sheet["path"]).convert("RGB")
        first = sheet["passes"][0]["frames"][0]
        x, y, w, h = first["cell"]
        px = img.getpixel((x + w // 2, y + h // 2))
        assert px == pytest.approx((0x7A, 0x6A, 0x53), abs=14)
        last = sheet["passes"][4]["frames"][-1]
        assert last["black"] is True
        x, y, w, h = last["cell"]
        px = img.getpixel((x + w // 2, y + h // 2))
        assert max(px) <= 12
        assert any("black_frames" in f["axes"]
                   for f in sheet["failing_frames"])

    def test_sheet_is_deterministic(self, tmp_path):
        import hashlib
        path = _good_render(tmp_path)
        a = build_contact_sheet(path, tmp_path / "a.png", duration=6.0)
        b = build_contact_sheet(path, tmp_path / "b.png", duration=6.0)
        ha = hashlib.sha256((tmp_path / "a.png").read_bytes()).hexdigest()
        hb = hashlib.sha256((tmp_path / "b.png").read_bytes()).hexdigest()
        assert ha == hb
        strip = ("path",)
        assert ({k: v for k, v in a.items() if k not in strip}
                == {k: v for k, v in b.items() if k not in strip})
