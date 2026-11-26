"""StableAnimator — third pose-driven character-video backend.

Failing-first: every new name is imported at module level, so the
whole file collection-errors on pristine (the backend does not exist
there). On head the tests run against the real router, the real
provision specs, and a faked inference subprocess (no torch/CUDA on
this box — the neural render itself is proven by the fake-subprocess
argv + real ffmpeg assembly, and live on a GPU workstation).
"""
import os
import subprocess
from pathlib import Path

from PIL import Image

import nomorals.media.directed.animator as animator_mod
from nomorals.media.directed.animator import (
    ModelUnavailable,
    animate_and_speak,
    direct_animate,
    stableanimator_animate,
    stableanimator_dwpose_dir,
    stableanimator_provision,
    stableanimator_resource_specs,
    stableanimator_status,
)
from nomorals.media.directed.pose_rig import build_track, render_pose_frames


def _person_png(tmp_path, size=128):
    img = Image.new("RGB", (size, size), (40, 40, 40))
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    d.rectangle([size // 2 - 8, size // 4, size // 2 + 8,
                 size - size // 8], fill=(200, 150, 120))
    p = tmp_path / "person.png"
    img.save(p)
    return str(p)


# ── status + router (no torch/CUDA on this box) ───────────────────────

def test_status_unavailable_names_gate():
    st = stableanimator_status()
    assert st["available"] is False
    assert "StableAnimator" in st["reason"]
    assert "torch" in st["reason"].lower()
    assert "provision" in st["reason"]
    assert "entry" in st["missing"] or "unet" in st["missing"]


def test_neural_raises_naming_both_backends():
    try:
        direct_animate("/tmp/x.png", build_track("nod", n_frames=4),
                       prefer="neural")
    except ModelUnavailable as exc:
        assert "StableAnimator" in str(exc)
        assert "MimicMotion" in str(exc)
    else:
        raise AssertionError("expected ModelUnavailable")


def test_stableanimator_forced_raises():
    try:
        direct_animate("/tmp/x.png", build_track("nod", n_frames=4),
                       prefer="stableanimator")
    except ModelUnavailable as exc:
        assert "StableAnimator" in str(exc)
    else:
        raise AssertionError("expected ModelUnavailable")


def test_router_auto_falls_to_warp(tmp_path):
    img = _person_png(tmp_path)
    res = direct_animate(img, build_track("nod", n_frames=4),
                         workdir=str(tmp_path))
    assert res.backend == "warp"
    assert os.path.exists(res.path)
    assert os.path.getsize(res.path) > 1000


def test_chain_forced_stableanimator_raises(tmp_path):
    img = _person_png(tmp_path)
    try:
        animate_and_speak(img, build_track("nod", n_frames=4),
                          "hello there", "some-voice",
                          prefer="stableanimator")
    except ModelUnavailable as exc:
        assert "StableAnimator" in str(exc)
    else:
        raise AssertionError("expected ModelUnavailable")


# ── pose frames (the control-folder contract) ─────────────────────────

def test_render_pose_frames(tmp_path):
    track = build_track("wave", n_frames=8)
    out = render_pose_frames(track, tmp_path / "poses")
    names = sorted(p.name for p in out.iterdir())
    assert names == [f"frame_{i}.png" for i in range(8)]
    # the upstream loader sorts by int(name.split('_')[1]) — the file
    # order IS the track order under that key.
    ordered = sorted(names, key=lambda n: int(n.split("_")[1].split(".")[0]))
    assert ordered == names
    for p in out.iterdir():
        im = Image.open(p)
        assert im.size == (track.width, track.height)
        # skeleton drawn: not a black frame
        assert max(im.convert("L").getdata()) > 0


# ── fake inference: argv + real ffmpeg assembly ──────────────────────

def _fake_install(tmp_path):
    base = tmp_path / "sa"
    (base / "repo").mkdir(parents=True)
    (base / "repo" / "inference_basic.py").write_text("# fake entry\n")
    (base / "Animation").mkdir()
    for name in ("unet.pth", "pose_net.pth", "face_encoder.pth"):
        (base / "Animation" / name).write_bytes(b"\x00")
    svd = base / "stable-video-diffusion-img2vid-xt"
    (svd / "image_encoder").mkdir(parents=True)
    (svd / "unet").mkdir()
    (svd / "vae").mkdir()
    for rel in ("model_index.json",
                "image_encoder/model.safetensors",
                "unet/diffusion_pytorch_model.safetensors",
                "vae/diffusion_pytorch_model.safetensors"):
        (svd / rel).write_bytes(b"\x00")
    face = base / "repo" / "models" / "antelopev2"
    face.mkdir(parents=True)
    (face / "glintr100.onnx").write_bytes(b"\x00")
    return base


def test_fake_inference_argv_and_mp4(tmp_path, monkeypatch):
    base = _fake_install(tmp_path)
    monkeypatch.setattr(animator_mod, "STABLEANIMATOR_MODELS_DIR", base)
    monkeypatch.setattr(
        animator_mod, "stableanimator_status",
        lambda: {"available": True, "torch": True, "cuda": True,
                 "repo": str(base / "repo"), "weights": "",
                 "missing": [], "reason": ""})
    seen: dict = {}
    real_run = subprocess.run

    def fake_run(argv, **kw):
        if argv[1] == "inference_basic.py":
            seen["argv"] = list(argv)
            seen["cwd"] = kw.get("cwd")
            out_dir = Path(argv[argv.index("--output_dir") + 1])
            frames = out_dir / "animated_images"
            frames.mkdir(parents=True)
            for i in range(4):
                Image.new("RGB", (512, 512),
                          (10 + i * 20, 20, 30)).save(
                              frames / f"frame_{i}.png")
            return subprocess.CompletedProcess(argv, 0, "", "")
        # ffmpeg assembly runs for real
        return real_run(argv, **kw)

    monkeypatch.setattr(subprocess, "run", fake_run)
    img = _person_png(tmp_path, size=512)
    track = build_track("nod", n_frames=4)
    out = str(tmp_path / "out.mp4")
    res = stableanimator_animate(img, track, out_path=out,
                                 workdir=str(tmp_path / "wd"))
    assert res.backend == "stableanimator"
    assert res.path == out
    assert os.path.exists(out)
    assert os.path.getsize(out) > 1000
    with open(out, "rb") as fh:
        assert fh.read(8)[4:8] == b"ftyp"  # real mp4 container
    argv = seen["argv"]
    assert argv[0] == "python"
    assert argv[1] == "inference_basic.py"
    assert seen["cwd"] == str(base / "repo")
    assert argv[argv.index("--pretrained_model_name_or_path") + 1] == \
        str(base / "stable-video-diffusion-img2vid-xt")
    assert argv[argv.index("--validation_image") + 1] == img
    poses = Path(argv[argv.index("--validation_control_folder") + 1])
    assert sorted(p.name for p in poses.iterdir()) == \
        [f"frame_{i}.png" for i in range(4)]
    assert argv[argv.index("--unet_model_name_or_path") + 1] == \
        str(base / "Animation" / "unet.pth")
    assert argv[argv.index("--posenet_model_name_or_path") + 1] == \
        str(base / "Animation" / "pose_net.pth")
    assert argv[argv.index("--face_encoder_model_name_or_path") + 1] == \
        str(base / "Animation" / "face_encoder.pth")
    assert argv[argv.index("--width") + 1] == "512"
    assert argv[argv.index("--height") + 1] == "512"
    assert "--gradient_checkpointing" in argv


# ── provision specs (exact HF inventory, verified 2026-11-26) ────────

def test_provision_specs(tmp_path, monkeypatch):
    monkeypatch.setattr(animator_mod, "STABLEANIMATOR_MODELS_DIR",
                        tmp_path / "sa")
    monkeypatch.setattr(animator_mod, "MIMIC_MODELS_DIR",
                        tmp_path / "mm")
    specs = stableanimator_resource_specs()
    assert len(specs) == 19  # 12 SVD+Animation + 5 face pack + 2 DWPose
    prefix = ("https://huggingface.co/FrancisRing/StableAnimator"
              "/resolve/main/")
    for spec in specs:
        assert spec.urls[0].startswith(prefix), spec.urls[0]
        assert spec.expected_bytes is not None and spec.expected_bytes > 0
        assert "svd_xt" not in spec.name
        assert "glintr100_torch" not in spec.name
        assert str(spec.dest).startswith(str(tmp_path))
    by_name = {s.name: s for s in specs}
    assert by_name["stableanimator/Animation/unet.pth"].expected_bytes == \
        6201376518
    assert by_name["stableanimator/Animation/pose_net.pth"].expected_bytes == \
        829269
    assert by_name["stableanimator/Animation/face_encoder.pth"].expected_bytes == \
        228754558
    assert by_name["stableanimator/stable-video-diffusion-img2vid-xt/unet/diffusion_pytorch_model.safetensors"].expected_bytes == \
        6098682464


def test_provision_gate_refuses_cpu_box():
    # No local CUDA GPU on this box: the gate refuses BEFORE any
    # download — no 16 GB fetch onto a box that cannot render.
    rep = stableanimator_provision()
    assert rep["ok"] is False
    assert rep["status"] == "unavailable"
    assert "VRAM" in rep["detail"]


def test_dwpose_shared_with_mimicmotion(tmp_path, monkeypatch):
    mm = tmp_path / "mm" / "DWPose"
    sa = tmp_path / "sa"
    monkeypatch.setattr(animator_mod, "MIMIC_MODELS_DIR", tmp_path / "mm")
    monkeypatch.setattr(animator_mod, "STABLEANIMATOR_MODELS_DIR", sa)
    # MimicMotion already holds both onnx files: shared, no dup.
    mm.mkdir(parents=True)
    (mm / "dw-ll_ucoco_384.onnx").write_bytes(b"\x00")
    (mm / "yolox_l.onnx").write_bytes(b"\x00")
    assert stableanimator_dwpose_dir() == mm
    # One missing: the StableAnimator dir, provisioned by us.
    (mm / "yolox_l.onnx").unlink()
    assert stableanimator_dwpose_dir() == sa / "DWPose"


# ── videogen engine registration ──────────────────────────────────────

def test_videogen_engine_registered():
    from nomorals.media.videogen import registry as vg
    assert "stableanimator" in vg.BACKENDS
    entries = [e for e in vg.list_models()
               if e["name"] == "stableanimator-basic"]
    assert len(entries) == 1
    e = entries[0]
    assert e["backend"] == "stableanimator"
    assert e["source"] == "FrancisRing/StableAnimator"
    assert e["vram_gb"] == 16.0
    assert e["license_key"] == "stableanimator"
    assert e["loadable"] is True
    assert vg.resolve("stableanimator-basic")["name"] == "stableanimator-basic"


def test_videogen_generate_unavailable(tmp_path):
    from nomorals.media.videogen import pipeline as vp
    img = _person_png(tmp_path)
    try:
        vp.generate("a warrior waves", backend="stableanimator",
                    image=img, action="wave")
    except vp.VideogenError as exc:
        assert "stableanimator" in str(exc).lower()
    else:
        raise AssertionError("expected VideogenError (never motion fallback)")


def test_hero_clip_unavailable(tmp_path):
    from nomorals.media.videogen import pipeline as vp
    img = _person_png(tmp_path)
    out = vp.request_hero_clip("a warrior waves",
                               prefer="stableanimator",
                               image=img, action="nod")
    assert out["ok"] is False
    assert "stableanimator" in out["reason"].lower()


def test_hero_clip_needs_image_and_pose(tmp_path):
    from nomorals.media.videogen import pipeline as vp
    img = _person_png(tmp_path)
    no_image = vp.request_hero_clip("x", prefer="stableanimator",
                                    action="nod")
    assert no_image["ok"] is False
    assert "image" in no_image["reason"]
    no_pose = vp.request_hero_clip("x", prefer="stableanimator", image=img)
    assert no_pose["ok"] is False
    assert "pose" in no_pose["reason"]


# ── license + tool wiring ─────────────────────────────────────────────

def test_license_entry():
    from nomorals.media.model_license import can_ship
    rep = can_ship("stableanimator")
    assert rep["licence"] == "MIT"
    assert rep["commercial_use"] is True


def test_tool_animate_photo_say_wiring(tmp_path):
    from nomorals.tools import directed as mod
    from nomorals.tools.registry import ToolRegistry
    reg = ToolRegistry()
    mod.register(reg)
    img = _person_png(tmp_path)
    # say without voice: honest error, no render attempted.
    half = reg.call("animate_photo", image=img, action="nod",
                    say="hello there")
    assert half.ok
    assert half.value["ok"] is False
    assert "both" in half.value["error"]
    # unknown backend: honest error.
    bad = reg.call("animate_photo", image=img, action="nod", backend="nope")
    assert bad.ok
    assert bad.value["ok"] is False
    assert "unknown backend" in bad.value["error"]
    # say + unknown voice: warp renders, then the voice step fails
    # honestly naming the voice (no fake speech, no crash).
    voiced = reg.call("animate_photo", image=img, action="nod",
                      say="hello there", voice="no-such-voice")
    assert voiced.ok
    assert voiced.value["ok"] is False
    assert "no-such-voice" in voiced.value["error"]
