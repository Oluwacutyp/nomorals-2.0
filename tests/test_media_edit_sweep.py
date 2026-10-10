"""Sweep tests for the media_edit module upgrade.

Covers every behavior added in the sweep: new image ops, studio presets /
LUT video grading / themed reports, caption styles + pop animation, video
loudnorm/watermark/rotate/flip/normalized concat, intent patterns + themed
plans, generate styles + prompt polish, transcript VTT/chapters, job
cancel/retry, layer transforms, edittext removal, avatar Wav2Lip backend,
upscale asset routing, segment model table, ComfyUI Wan i2v, LTX routing.
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

from nomorals.media_edit import images
from nomorals.media_edit import studio
from nomorals.media_edit import captions
from nomorals.media_edit import videos
from nomorals.media_edit import intent
from nomorals.media_edit import generate
from nomorals.media_edit import transcript_edit
from nomorals.media_edit import jobs
from nomorals.media_edit import layers
from nomorals.media_edit import segment
from nomorals.media_edit import upscale
from nomorals.media_edit import edittext
from nomorals.media_edit import avatar
from nomorals.media_edit import faceswap
from nomorals.media_edit import comfy
from nomorals.media_edit import video_models


def _img(w=120, h=80, color="red") -> Image.Image:
    return Image.new("RGB", (w, h), color)


def _words():
    return [captions.Word(text=t, start=i * 0.4, end=i * 0.4 + 0.35)
            for i, t in enumerate(["hey", "check", "this", "out"])]


@pytest.fixture()
def workdir(tmp_path: Path) -> Path:
    return tmp_path


# ---------------------------------------------------------------------------
# images: duotone / round_corners / gradient_overlay
# ---------------------------------------------------------------------------

class TestNewImageOps:
    def test_duotone_maps_luminance(self):
        out = images.op_duotone(_img(color="white"), dark="black",
                                light="white")
        assert out.mode == "RGB" and out.size == (120, 80)
        # white input → light color; black input → dark color
        assert images.op_duotone(_img(color="black")).getpixel((0, 0)) == (0, 0, 0)

    def test_round_corners_alpha(self):
        out = images.op_round_corners(_img(), radius=20)
        assert out.mode == "RGBA"
        assert out.getpixel((0, 0))[3] == 0      # corner transparent
        assert out.getpixel((60, 40))[3] == 255  # center opaque

    def test_round_corners_clamped(self):
        out = images.op_round_corners(_img(w=40, h=40), radius=999)
        assert out.size == (40, 40)

    def test_gradient_overlay_modes(self):
        for direction in ("vertical", "horizontal", "diagonal"):
            out = images.op_gradient_overlay(
                _img(), colors=("#000000", "#ffffff"),
                direction=direction, opacity=0.5)
            assert out.mode == "RGB" and out.size == (120, 80)
        with pytest.raises(images.MediaEditError):
            images.op_gradient_overlay(_img(), direction="sideways")
        with pytest.raises(images.MediaEditError):
            images.op_gradient_overlay(_img(), colors=("#000",))

    def test_allowlisted_and_chainable(self):
        for name in ("duotone", "round_corners", "gradient_overlay"):
            assert name in images.OP_ALLOWLIST
            assert name in images._OP_FUNCS
        out = images.apply_chain(
            _img(), [{"op": "duotone"}, {"op": "round_corners", "radius": 8}])
        assert out.mode == "RGBA"
        images.validate_ops([{"op": "gradient_overlay",
                              "colors": ["#000", "#fff"]}])


# ---------------------------------------------------------------------------
# studio: presets, lut chain, themed report, video LUT grade
# ---------------------------------------------------------------------------

class TestStudioSweep:
    @pytest.mark.parametrize("preset", ["cyberpunk", "moody", "pastel-pop",
                                        "street", "clean-film", "anime-soft"])
    def test_new_presets_apply(self, preset):
        out = studio.op_filter(_img(), preset, strength=0.8)
        assert out.size == (120, 80)

    def test_preset_strength_zero_is_noop(self):
        src = _img()
        out = studio.op_filter(src, "cyberpunk", strength=0.0)
        assert list(out.getdata()) == list(src.getdata())

    def test_lut_op_chains(self):
        st = studio.EditStudio()
        st.lut("film.cube")
        assert st.ops[-1]["op"] == "cube_lut"
        assert st.ops[-1]["params"]["lut"] == "film.cube"

    def test_report_themes(self):
        st = studio.EditStudio(name="demo", kind="image")
        st.filter("cinematic").grade(temperature=0.1)
        rich = st.report(theme="rich")
        assert "🎬" in rich and "demo" in rich and "cinematic" in rich
        plain = st.report(theme="plain")
        assert plain == st.describe()
        with pytest.raises(images.MediaEditError):
            st.report(theme="fancy")

    def test_grade_video_lut(self, workdir):
        src = workdir / "src.mp4"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=10",
             "-pix_fmt", "yuv420p", str(src)], check=True)
        cube = workdir / "identity.cube"
        cube.write_text(
            'TITLE "identity"\nLUT_3D_SIZE 2\n'
            "0.0 0.0 0.0\n1.0 0.0 0.0\n0.0 1.0 0.0\n1.0 1.0 0.0\n"
            "0.0 0.0 1.0\n1.0 0.0 1.0\n0.0 1.0 1.0\n1.0 1.0 1.0\n")
        res = studio.grade_video_lut(src, cube, strength=1.0,
                                     out_dir=workdir)
        assert Path(res["output"]).exists()
        assert res["strength"] == 1.0
        with pytest.raises(images.MediaEditError):
            studio.grade_video_lut(workdir / "nope.mp4", cube)


# ---------------------------------------------------------------------------
# captions: new styles + pop animation
# ---------------------------------------------------------------------------

class TestCaptionSweep:
    def test_new_styles_listed(self):
        styles = captions.caption_styles()
        for s in ("neon", "podcast", "beast-bounce"):
            assert s in styles

    @pytest.mark.parametrize("style", ["neon", "podcast", "beast-bounce"])
    def test_new_styles_render_valid_ass(self, style):
        ass = captions.words_to_ass(_words(), style=style)
        assert ass.startswith("[Script Info]")
        assert "Dialogue:" in ass

    def test_pop_animation_tags(self):
        ass = captions.words_to_ass(_words(), style="hormozi",
                                    animate="pop")
        assert "\\t(" in ass and "\\fscx" in ass
        plain = captions.words_to_ass(_words(), style="hormozi")
        assert "\\t(" not in plain
        with pytest.raises(captions.MediaEditError):
            captions.words_to_ass(_words(), animate="wiggle")


# ---------------------------------------------------------------------------
# videos: loudnorm / watermark / rotate / flip / normalized concat
# ---------------------------------------------------------------------------

@pytest.fixture()
def av_src(workdir: Path) -> Path:
    src = workdir / "av.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=duration=2:size=160x120:rate=10",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-pix_fmt", "yuv420p", "-shortest", str(src)], check=True)
    return src


class TestVideoSweep:
    def test_loudnorm(self, av_src, workdir):
        res = videos.loudnorm(av_src, out_dir=workdir)
        assert Path(res["output"]).exists()
        assert res["target_lufs"] == -16.0
        assert "loudnorm" in res["filter"]

    def test_watermark(self, av_src, workdir):
        logo = workdir / "logo.png"
        Image.new("RGBA", (40, 20), (255, 0, 0, 255)).save(logo)
        res = videos.watermark(av_src, logo, out_dir=workdir,
                               position="top-left", opacity=0.5,
                               scale=0.2)
        assert Path(res["output"]).exists()
        assert res["position"] == "top-left"
        with pytest.raises(videos.MediaEditError):
            videos.watermark(av_src, logo, position="middle-earth")
        with pytest.raises(videos.MediaEditError):
            videos.watermark(av_src, workdir / "missing.png")

    def test_rotate_flip(self, av_src, workdir):
        r = videos.rotate_video(av_src, 90, out_dir=workdir)
        assert Path(r["output"]).exists() and r["angle"] == 90
        f = videos.flip_video(av_src, direction="vertical",
                              out_dir=workdir)
        assert Path(f["output"]).exists()
        with pytest.raises(videos.MediaEditError):
            videos.rotate_video(av_src, 45)
        with pytest.raises(videos.MediaEditError):
            videos.flip_video(av_src, direction="diagonal")

    def test_concat_normalized(self, av_src, workdir):
        res = videos.concat_normalized([av_src, av_src], out_dir=workdir,
                                       width=160, height=120, fps=10)
        assert Path(res["output"]).exists()
        info = videos.video_probe(res["output"])
        assert info["width"] == 160 and info["height"] == 120


# ---------------------------------------------------------------------------
# intent: new patterns + themed plans
# ---------------------------------------------------------------------------

class TestIntentSweep:
    @pytest.mark.parametrize("phrase,op", [
        ("remove the background", "bg_remove_v2"),
        ("cartoonize this photo", "cartoonize"),
        ("give it a pencil sketch look", "pencil_sketch"),
        ("upscale it", "upscale_sr"),
        ("make it duotone", "duotone"),
        ("round corners", "round_corners"),
        ("add a gradient overlay", "gradient_overlay"),
        ("apply lut film.cube", "cube_lut"),
    ])
    def test_new_image_patterns(self, phrase, op):
        parsed = intent.parse_instruction(phrase, kind="image")
        assert any(o["op"] == op for o in parsed.ops), parsed.ops

    def test_lut_carries_filename(self):
        parsed = intent.parse_instruction("apply lut film.cube",
                                          kind="image")
        lut_op = next(o for o in parsed.ops if o["op"] == "cube_lut")
        assert lut_op["lut"] == "film.cube"

    def test_parsed_ops_validate(self):
        parsed = intent.parse_instruction(
            "remove the background and cartoonize it", kind="image")
        images.validate_ops(parsed.ops)

    def test_format_plan_themes(self):
        parsed = intent.parse_instruction("make it square and upscale it",
                                          kind="image")
        plain = intent.format_plan(parsed, theme="plain")
        assert plain == intent.describe_plan(parsed) == parsed.describe()
        rich = intent.format_plan(parsed, theme="rich")
        assert "🗺️" in rich and "upscale_sr" in rich
        v = intent.parse_instruction("trim the first 30 seconds",
                                     kind="video")
        assert "[background job]" in intent.format_plan(v, theme="rich")
        with pytest.raises(intent.AmbiguousInstructionError):
            intent.format_plan(parsed, theme="fancy")


# ---------------------------------------------------------------------------
# generate: styles + prompt polish
# ---------------------------------------------------------------------------

class TestGenerateSweep:
    def test_new_styles(self):
        styles = generate.list_styles()
        for s in ("noir-film", "ghibli", "ukiyo-e", "double-exposure",
                  "isometric", "vaporwave", "storybook", "blueprint"):
            assert s in styles
            assert generate._resolve_image_style(s) == s

    def test_enhance_prompt(self):
        p = generate.enhance_prompt("a cat")
        assert "highly detailed" in p
        # idempotent
        assert generate.enhance_prompt(p) == p

    def test_resolve_gen_params_enhance(self):
        prompt, w, h, steps = generate._resolve_gen_params(
            "a cat", style="ghibli", enhance=True)
        assert "ghibli" in prompt or "Ghibli" in prompt or \
            "studio ghibli" in prompt
        assert "highly detailed" in prompt
        # default path unchanged
        p2, _, _, _ = generate._resolve_gen_params("a cat", style="ghibli")
        assert "highly detailed" not in p2


# ---------------------------------------------------------------------------
# transcript_edit: vtt + chapters
# ---------------------------------------------------------------------------

class TestTranscriptSweep:
    def test_export_vtt(self, workdir):
        words = [transcript_edit.Word(text=t, start=i * 1.0,
                                      end=i * 1.0 + 0.8)
                 for i, t in enumerate(["hello", "world", "again"])]
        out = transcript_edit.export_vtt(words, workdir / "v.mp4",
                                         out_path=workdir / "v.vtt")
        text = out.read_text()
        assert text.startswith("WEBVTT")
        assert "-->" in text and "00:00:00.000" in text

    def test_auto_chapters(self):
        # two clusters separated by a long pause
        words = []
        t = 0.0
        for i in range(10):
            words.append(transcript_edit.Word(text=f"w{i}", start=t,
                                              end=t + 0.4))
            t += 0.6
        t += 10.0  # long pause → chapter break
        for i in range(10, 20):
            words.append(transcript_edit.Word(text=f"w{i}", start=t,
                                              end=t + 0.4))
            t += 0.6
        chapters = transcript_edit.auto_chapters(words, min_gap_s=3.0)
        assert len(chapters) == 2
        assert chapters[0]["start"] == 0.0
        assert chapters[1]["title"].startswith("w10")
        assert transcript_edit.auto_chapters([]) == []


# ---------------------------------------------------------------------------
# jobs: cancel + retry
# ---------------------------------------------------------------------------

class TestJobsSweep:
    def test_cancel_queued(self):
        mgr = jobs.JobManager()
        gate = threading.Event()

        def _blocker(progress):
            gate.wait(10)
            return {"output": "x"}

        def _quick(progress):
            return {"output": "y"}

        j1 = mgr.submit("video", "blocker", _blocker)
        j2 = mgr.submit("video", "victim", _quick)
        time.sleep(0.3)  # let the worker pick up j1
        info = mgr.cancel(j2)
        assert info["status"] == "cancelled"
        gate.set()
        assert mgr.wait(j1)["status"] == "done"
        assert mgr.wait(j2)["status"] == "cancelled"

    def test_cancel_running(self):
        mgr = jobs.JobManager()

        def _slow(progress):
            for i in range(20):
                progress(i / 20)
                time.sleep(0.05)
            return {"output": "never"}

        jid = mgr.submit("video", "slow", _slow)
        time.sleep(0.3)
        mgr.cancel(jid)
        assert mgr.wait(jid, timeout=10)["status"] == "cancelled"

    def test_retry(self):
        mgr = jobs.JobManager()
        calls = {"n": 0}

        def _flaky(progress):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            progress(1.0)
            return {"output": "ok"}

        jid = mgr.submit("image", "flaky", _flaky)
        assert mgr.wait(jid)["status"] == "failed"
        assert mgr.retry(jid) == jid
        done = mgr.wait(jid)
        assert done["status"] == "done"
        assert done["result"]["output"] == "ok"

    def test_retry_running_rejected(self):
        mgr = jobs.JobManager()
        gate = threading.Event()

        def _blocker(progress):
            gate.wait(10)
            return {"output": "x"}

        jid = mgr.submit("video", "blocker", _blocker)
        time.sleep(0.3)
        with pytest.raises(images.MediaEditError):
            mgr.retry(jid)
        gate.set()
        mgr.wait(jid)


# ---------------------------------------------------------------------------
# layers: duplicate / nudge / scale
# ---------------------------------------------------------------------------

class TestLayersSweep:
    def _stack(self):
        st = layers.LayerStack((200, 200), bg="white")
        lid = st.add_image(_img(60, 40), name="pic", position=(10, 10))
        return st, lid

    def test_nudge(self):
        st, lid = self._stack()
        st.nudge(lid, 5, -3)
        info = next(l for l in st.layer_info() if l["id"] == lid)
        assert info["position"] == [15, 7]

    def test_scale_layer(self):
        st, lid = self._stack()
        st.scale_layer(lid, 2.0)
        layer = st._get(lid)
        assert layer.params["scale"] == 2.0
        with pytest.raises(images.MediaEditError):
            st.scale_layer(lid, 0)
        tid = st.add_text("hi")
        with pytest.raises(images.MediaEditError):
            st.scale_layer(tid, 2.0)

    def test_duplicate(self):
        st, lid = self._stack()
        dup = st.duplicate(lid, dx=4, dy=4)
        assert dup != lid and len(st) == 2
        info = next(l for l in st.layer_info() if l["id"] == dup)
        assert info["position"] == [14, 14]
        assert info["name"] == "pic copy"
        # flatten still works with the duplicate present
        flat = st.flatten()
        assert flat.size == (200, 200)


# ---------------------------------------------------------------------------
# edittext / faceswap / avatar: honest fail-fast without heavy deps
# ---------------------------------------------------------------------------

class TestHeavyDepFailFast:
    def test_remove_text_needs_tesseract(self):
        with pytest.raises(Exception) as exc:
            edittext.remove_text(_img(), "hello")
        assert "tesseract" in str(exc.value).lower()

    def test_swap_all_faces_needs_insightface(self):
        with pytest.raises(Exception) as exc:
            faceswap.swap_all_faces(_img(), _img())
        assert "insightface" in str(exc.value).lower()

    def test_wav2lip_not_configured(self):
        ok, reason = avatar.wav2lip_available()
        assert ok is False
        assert reason  # names what's missing
        with pytest.raises(avatar.NotConfigured):
            avatar.wav2lip_repo()
        names = [b["name"] for b in avatar.list_backends()]
        assert "wav2lip" in names


# ---------------------------------------------------------------------------
# upscale / segment: routing tables
# ---------------------------------------------------------------------------

class TestRoutingTables:
    def test_upscale_auto_kinds(self):
        assert upscale.ASSET_MODELS["anime"]["model"] == \
            "RealESRGAN_x4plus_anime_6B"
        assert upscale.ASSET_MODELS["document"]["model"] == \
            "RealESRGAN_x2plus"
        assert upscale.ASSET_MODELS["photo"]["model"] == "RealESRGAN_x4plus"
        with pytest.raises(upscale.UpscaleError):
            upscale.upscale_auto(_img(), kind="hologram")

    def test_segment_model_table(self):
        names = {m["name"] for m in segment.list_models()}
        for m in ("u2net_human_seg", "birefnet-portrait", "isnet-anime",
                  "birefnet-general", "isnet-general-use"):
            assert m in names


# ---------------------------------------------------------------------------
# comfy: wan22 i2v workflow
# ---------------------------------------------------------------------------

class TestComfySweep:
    def test_i2v_template_valid(self):
        raw = Path(comfy.__file__).parent / "workflows" / "wan22_i2v.json"
        wf = json.loads(raw.read_text())
        kinds = {n["class_type"] for n in wf.values()}
        assert {"WanVideoModelLoader", "WanVideoVAELoader",
                "WanVideoTextEncode", "LoadImage", "WanVideoImageEncode",
                "WanVideoSampler", "WanVideoDecode",
                "VHS_VideoCombine"} <= kinds

    def test_i2v_workflow_parameterized(self):
        wf = comfy._wan22_i2v_workflow(
            image_name="start.png", prompt="a cat walks",
            negative_prompt="blurry", width=960, height=544,
            num_frames=48, steps=12, seed=7, cfg=5.0,
            model_file=None, vae_file=None)
        img_node = comfy._single(wf, "LoadImage")["inputs"]
        assert img_node["image"] == "start.png"
        enc = comfy._single(wf, "WanVideoTextEncode")["inputs"]
        assert enc["positive_prompt"] == "a cat walks"
        assert enc["negative_prompt"] == "blurry"
        sampler = comfy._single(wf, "WanVideoSampler")["inputs"]
        assert sampler["seed"] == 7 and sampler["num_frames"] == 48
        assert sampler["steps"] == 12
        assert hasattr(comfy.ComfyUIBackend, "img2video")

    def test_i2v_rejects_bad_frames(self):
        with pytest.raises(comfy.GenerativeEditError):
            comfy._wan22_i2v_workflow(
                image_name="x.png", prompt="p", negative_prompt=None,
                width=None, height=None, num_frames=0, steps=None,
                seed=None, cfg=None, model_file=None, vae_file=None)


# ---------------------------------------------------------------------------
# video_models: LTX route
# ---------------------------------------------------------------------------

class TestVideoModelSweep:
    def test_ltx_listed(self):
        router = video_models.VideoModelRouter()
        entry = next(m for m in router.list_models()
                     if m["name"] == "ltx-2.3")
        assert entry["backend"] == "comfy"
        assert entry["speed"] == "fast"
        assert entry["paid"] is False

    def test_pick_ltx(self):
        router = video_models.VideoModelRouter()
        route = router._pick_ltx()
        assert route.model == "ltx-2.3"
        assert route.backend == "comfy"
        assert route.workflow is None  # honest: no template ships
        assert "workflow" in route.reason.lower()

    def test_prefer_validated(self):
        router = video_models.VideoModelRouter()
        with pytest.raises(video_models.VideoModelError):
            router.route(prefer="sora", profile="termux")
