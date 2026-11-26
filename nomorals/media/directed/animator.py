"""Animation render backends — neural when available, honest warp on CPU.

- StableAnimatorBackend: Francis-Rings/StableAnimator, reference image +
  pose frames -> ID-preserving directed video (SVD-XT + Face Encoder +
  ID Adapter, no face post-processing). Workstation (16GB VRAM). Top
  neural rung when identity fidelity matters most.
- MimicMotionBackend: Tencent/MimicMotion, pose video + reference image ->
  photoreal directed video. Workstation (16GB VRAM). Wired now, not later.
- WarpAnimator: pose-guided mesh warp on CPU. REAL directed motion (the arm
  region rises, the hand morphs) via PIL MESH displacement driven by the
  same keypoint trajectories. Crude but honest — labeled `warp`, never
  claimed as neural.

direct_animate() routes: best neural installed first
(stableanimator -> mimicmotion), warp fallback. The pose currency is
shared: PoseTrack + render_pose_video() / render_pose_frames().
Both neural backends read the same DWPose weights for pose extraction
(the template-motion path); at inference time Devon renders the pose
frames itself, so neither backend extracts poses to render.
"""

from __future__ import annotations
from .failures import record_failure

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

NUMPY_AVAILABLE = np is not None


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError(
            "nomorals.media.directed.animator is unavailable: numpy is not installed "
            "(pip install numpy)")

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]

PIL_AVAILABLE = Image is not None


def _require_pil() -> None:
    if Image is None:
        raise RuntimeError(
            "nomorals.media.directed.animator is unavailable: Pillow is not installed "
            "(pip install Pillow)")


from .pose_rig import PoseTrack, render_pose_video


class ModelUnavailable(RuntimeError):
    """Raised when a neural backend isn't installed. Carries the fix."""


@dataclass
class AnimResult:
    path: str
    backend: str          # "stableanimator" | "mimicmotion" | "warp"
    action: str = ""
    note: str = ""


@dataclass
class ChainResult:
    """Pose -> animate -> lip-sync -> voice, one receipt.

    ``path`` is the final dubbed video. ``animate_backend`` names the
    animation rung that rendered the body; ``lipsync_backend`` names
    the lip-sync rung; ``voice`` names the catalogue voice that spoke.
    """
    path: str
    animate_backend: str  # "stableanimator" | "mimicmotion" | "warp"
    lipsync_backend: str  # "latentsync" | "wav2lip" | "warp"
    voice: str = ""
    action: str = ""
    note: str = ""


def _ffmpeg() -> str | None:
    from shutil import which
    return which("ffmpeg")


# ── MimicMotion backend ──────────────────────────────────────────────
MIMIC_REPO = "https://github.com/Tencent/MimicMotion"
MIMIC_MODELS_DIR = Path.home() / ".devon-models" / "mimicmotion"


def mimicmotion_status() -> dict:
    """Check whether MimicMotion is installed and usable."""
    try:
        import torch  # noqa: F401
        has_torch = True
        cuda = torch.cuda.is_available()
    except Exception as e:
        record_failure("animator.mimicmotion_status", e)
        has_torch, cuda = False, False
    repo = MIMIC_MODELS_DIR / "repo"
    weights = MIMIC_MODELS_DIR / "MimicMotion_1-1.pth"
    dwpose = MIMIC_MODELS_DIR / "DWPose"
    return {
        "available": bool(has_torch and cuda and repo.is_dir()
                          and weights.exists()),
        "torch": has_torch, "cuda": cuda,
        "repo": str(repo), "weights": str(weights),
        "reason": (
            "MimicMotion needs torch+CUDA and the repo weights. "
            f"Install: git clone {MIMIC_REPO} {repo} && "
            "download MimicMotion_1-1.pth + DWPose onnx files into "
            f"{MIMIC_MODELS_DIR} (see DIRECTED_MINING.md §5)."
        ),
    }


def mimicmotion_animate(image: str, track: PoseTrack, *,
                        out_path: str | None = None,
                        workdir: str | None = None) -> AnimResult:
    """Render directed video with MimicMotion (neural, workstation)."""
    st = mimicmotion_status()
    if not st["available"]:
        raise ModelUnavailable(st["reason"])
    wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="mm_"))
    wd.mkdir(parents=True, exist_ok=True)
    pose_vid = render_pose_video(track, str(wd / "pose.mp4"))
    out = out_path or str(wd / "directed.mp4")
    cfg = wd / "infer.yaml"
    cfg.write_text(
        "ref_image: %s\npose_video: %s\noutput: %s\n" % (image, pose_vid, out))
    repo = Path(st["repo"])
    proc = subprocess.run(
        ["python", "inference.py", "--inference_config", str(cfg)],
        cwd=str(repo), capture_output=True, timeout=3600)
    if proc.returncode != 0 or not os.path.exists(out):
        raise RuntimeError(
            "MimicMotion inference failed: "
            + proc.stderr.decode()[-2000:])
    return AnimResult(path=out, backend="mimicmotion", action=track.action,
                      note="neural pose-guided render (MimicMotion)")


# ── StableAnimator backend ───────────────────────────────────────────
# Francis-Rings/StableAnimator (CVPR2025, MIT): reference image + a
# folder of pose skeleton frames -> ID-preserving video. No face
# post-processing at all (no FaceFusion, no GFP-GAN/CodeFormer) — the
# Face Encoder + ID Adapter carry identity inside the denoise.
#
# Verified against the live repo (2026-11-26, main) and the HF weight
# repo FrancisRing/StableAnimator — every path, flag and byte count
# below comes from those reads, not from the survey:
# - entry: inference_basic.py; --validation_image (reference),
#   --validation_control_folder (pose PNG folder, files named
#   <word>_<N>.png — the loader sorts by int(name.split('_')[1])),
#   --pretrained_model_name_or_path (the SVD diffusers dir),
#   --posenet/--face_encoder/--unet_model_name_or_path (.pth files),
#   --output_dir, --width/--height, --guidance_scale,
#   --num_inference_steps, --decode_chunk_size, --gradient_checkpointing.
# - the script writes <output_dir>/animated_images/frame_<N>.png + an
#   animation_video.gif — it does NOT write an mp4. Devon assembles
#   the mp4 from the frames with ffmpeg (the README's own recipe).
# - inference loads the SVD SUBFOLDER weights (unet / vae /
#   image_encoder .safetensors + configs), NOT the single-file
#   svd_xt.safetensors / svd_xt_image_decoder.safetensors at the SVD
#   root — those two ~9.5GB files are NOT provisioned.
# - face embedding comes from insightface antelopev2 glintr100.onnx
#   (FaceModel, root='.' — insightface downloads the pack into
#   <repo>/models/antelopev2 on first run); glintr100_torch.pth in
#   Animation/ is NOT referenced by inference_basic.py and is NOT
#   provisioned.
# - DWPose is preprocessing only (DWPose/skeleton_extraction.py):
#   inference never touches it. Both backends' preprocessing reads
#   the SAME two onnx files (dw-ll_ucoco_384.onnx + yolox_l.onnx —
#   the MimicMotion weights in DIRECTED_MINING.md §5 are byte-identical
#   names), so the provisioner provisions them ONCE and both backends
#   resolve to the same files (see stableanimator_dwpose_dir).
STABLEANIMATOR_REPO = "https://github.com/Francis-Rings/StableAnimator"
STABLEANIMATOR_HF_REPO = "FrancisRing/StableAnimator"
STABLEANIMATOR_MODELS_DIR = Path.home() / ".devon-models" / "stableanimator"
#: VRAM floor for the basic checkpoint (README: 576x1024 or 512x512;
#: --gradient_checkpointing exists for tight cards).
STABLEANIMATOR_MIN_VRAM_GB = 16.0

#: Weight specs, in provision order. expected_bytes are the exact sizes
#: from the HF API listing (2026-11-26). URLs are the public HF resolve
#: endpoints — no login, no key. Files the inference script never loads
#: are deliberately absent: svd_xt.safetensors (9,559,625,980 B),
#: svd_xt_image_decoder.safetensors (9,503,252,964 B), the fp16
#: variants, glintr100_torch.pth, inference.zip (eval samples).
STABLEANIMATOR_WEIGHT_FILES: tuple[tuple[str, int], ...] = (
    # Animation/ — the StableAnimator weights themselves.
    ("Animation/unet.pth", 6201376518),
    ("Animation/pose_net.pth", 829269),
    ("Animation/face_encoder.pth", 228754558),
    # SVD diffusers dir — the subfolder weights + configs the script
    # actually loads (from_pretrained(..., subfolder=...)).
    ("stable-video-diffusion-img2vid-xt/model_index.json", 496),
    ("stable-video-diffusion-img2vid-xt/feature_extractor/preprocessor_config.json", 518),
    ("stable-video-diffusion-img2vid-xt/image_encoder/config.json", 685),
    ("stable-video-diffusion-img2vid-xt/image_encoder/model.safetensors", 2528371296),
    ("stable-video-diffusion-img2vid-xt/scheduler/scheduler_config.json", 533),
    ("stable-video-diffusion-img2vid-xt/unet/config.json", 984),
    ("stable-video-diffusion-img2vid-xt/unet/diffusion_pytorch_model.safetensors", 6098682464),
    ("stable-video-diffusion-img2vid-xt/vae/config.json", 607),
    ("stable-video-diffusion-img2vid-xt/vae/diffusion_pytorch_model.safetensors", 391017740),
)

#: insightface antelopev2 pack — provisioned straight into the repo's
#: models/antelopev2/ (exactly where insightface's own first-run
#: download lands, minus the nested-dir bug the README documents).
STABLEANIMATOR_FACE_FILES: tuple[tuple[str, int], ...] = (
    ("models/antelopev2/1k3d68.onnx", 143607619),
    ("models/antelopev2/2d106det.onnx", 5030888),
    ("models/antelopev2/genderage.onnx", 1322532),
    ("models/antelopev2/glintr100.onnx", 260665334),
    ("models/antelopev2/scrfd_10g_bnkps.onnx", 16923827),
)

#: DWPose weights — shared with MimicMotion (same files, same bytes).
STABLEANIMATOR_DWPOSE_FILES: tuple[tuple[str, int], ...] = (
    ("DWPose/dw-ll_ucoco_384.onnx", 134399116),
    ("DWPose/yolox_l.onnx", 216746733),
)

#: pip deps for the repo's own environment (requirements.txt, plural —
#: the README's file listing says `requirement.txt` but the file on
#: main is `requirements.txt`). Torch is NOT in this list: Devon never
#: auto-installs torch — the status reason names torch 2.5.1 cu124 +
#: xformers as the owner's honest gate.
STABLEANIMATOR_PIP_PACKAGES: tuple[str, ...] = (
    "diffusers", "transformers==4.35.2", "accelerate==0.25.0",
    "timm==0.4.12", "decord", "einops", "scipy", "pandas",
    "coloredlogs", "flatbuffers", "numpy==1.26.4", "packaging",
    "protobuf", "sympy", "imageio-ffmpeg", "insightface", "facexlib",
    "opencv-python-headless", "gradio", "onnxruntime-gpu",
)


def _hf_url(rel: str) -> str:
    return (f"https://huggingface.co/{STABLEANIMATOR_HF_REPO}"
            f"/resolve/main/{rel}")


def stableanimator_resource_specs(
        *, include_dwpose: bool = True) -> list:
    """One ResourceSpec per weight file (provision.py seam).

    Layout mirrors the upstream tree so the repo's own scripts find
    their paths: weights under ~/.devon-models/stableanimator/
    (Animation/ + the SVD dir), the face pack under <repo>/models/.
    DWPose specs are included by default (shared — see
    stableanimator_dwpose_dir; when MimicMotion already holds them
    the provisioner skips the download, it never fetches twice).
    """
    from ...core.provision import ResourceSpec
    specs: list = []
    for rel, size in STABLEANIMATOR_WEIGHT_FILES:
        specs.append(ResourceSpec(
            name=f"stableanimator/{rel}", urls=(_hf_url(rel),),
            dest=STABLEANIMATOR_MODELS_DIR / rel, expected_bytes=size))
    repo = STABLEANIMATOR_MODELS_DIR / "repo"
    for rel, size in STABLEANIMATOR_FACE_FILES:
        specs.append(ResourceSpec(
            name=f"stableanimator/{rel}", urls=(_hf_url(rel),),
            dest=repo / rel, expected_bytes=size))
    if include_dwpose:
        dw = stableanimator_dwpose_dir()
        for rel, size in STABLEANIMATOR_DWPOSE_FILES:
            specs.append(ResourceSpec(
                name=f"stableanimator/{rel}", urls=(_hf_url(rel),),
                dest=dw / Path(rel).name, expected_bytes=size))
    return specs


def stableanimator_dwpose_dir() -> Path:
    """Where the DWPose weights live — shared with MimicMotion.

    When MimicMotion's DWPose dir already holds both onnx files, that
    IS the dir (same bytes, zero duplicate download). Otherwise the
    StableAnimator dir, which the provisioner fills.
    """
    mm = MIMIC_MODELS_DIR / "DWPose"
    if ((mm / "dw-ll_ucoco_384.onnx").is_file()
            and (mm / "yolox_l.onnx").is_file()):
        return mm
    return STABLEANIMATOR_MODELS_DIR / "DWPose"


def _stableanimator_paths() -> dict:
    """Repo + weight paths the status check and the driver agree on."""
    base = STABLEANIMATOR_MODELS_DIR
    repo = base / "repo"
    svd = base / "stable-video-diffusion-img2vid-xt"
    return {
        "repo": repo,
        "entry": repo / "inference_basic.py",
        "svd": svd,
        "unet": base / "Animation" / "unet.pth",
        "posenet": base / "Animation" / "pose_net.pth",
        "face_encoder": base / "Animation" / "face_encoder.pth",
        "face_pack": repo / "models" / "antelopev2" / "glintr100.onnx",
        "dwpose": stableanimator_dwpose_dir(),
    }


def stableanimator_status() -> dict:
    """Check whether StableAnimator is installed and usable."""
    try:
        import torch  # noqa: F401
        has_torch = True
        cuda = torch.cuda.is_available()
    except Exception as e:
        record_failure("animator.stableanimator_status", e)
        has_torch, cuda = False, False
    p = _stableanimator_paths()
    missing = [k for k in ("entry", "unet", "posenet", "face_encoder",
                           "face_pack")
               if not p[k].exists()]
    svd_ok = ((p["svd"] / "model_index.json").is_file()
              and (p["svd"] / "image_encoder" / "model.safetensors").is_file()
              and (p["svd"] / "unet" / "diffusion_pytorch_model.safetensors").is_file()
              and (p["svd"] / "vae" / "diffusion_pytorch_model.safetensors").is_file())
    if not svd_ok:
        missing.append("svd")
    return {
        "available": bool(has_torch and cuda and not missing),
        "torch": has_torch, "cuda": cuda,
        "repo": str(p["repo"]), "weights": str(p["unet"]),
        "missing": missing,
        "reason": (
            "StableAnimator needs torch+CUDA and the repo weights. "
            f"Install: git clone {STABLEANIMATOR_REPO} {p['repo']} && "
            "provision the weights (stableanimator_provision) — "
            "Animation/unet.pth + pose_net.pth + face_encoder.pth and "
            "the SVD diffusers dir, ~16 GB from Hugging Face, no login. "
            "Torch itself is never auto-installed: torch 2.5.1 cu124 + "
            "xformers + the repo requirements.txt go in the repo's own "
            "environment."
            + (f" Missing now: {', '.join(missing)}." if missing else "")
        ),
    }


def stableanimator_animate(image: str, track: PoseTrack, *,
                           out_path: str | None = None,
                           workdir: str | None = None,
                           num_inference_steps: int = 25,
                           guidance_scale: float = 3.0) -> AnimResult:
    """Render directed video with StableAnimator (neural, workstation).

    Same call shape as mimicmotion_animate: reference image + pose
    track in, mp4 out. The differences are upstream's, not Devon's:
    the pose input is a FOLDER of skeleton frames (render_pose_frames
    writes frame_<N>.png, the naming the loader sorts on) and the
    script emits PNG frames, which Devon assembles into the mp4 with
    ffmpeg (the README's own recipe) at the track's fps.
    """
    st = stableanimator_status()
    if not st["available"]:
        raise ModelUnavailable(st["reason"])
    from .pose_rig import render_pose_frames
    wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="sa_"))
    wd.mkdir(parents=True, exist_ok=True)
    pose_dir = render_pose_frames(track, wd / "poses")
    out_dir = wd / "sa_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = _stableanimator_paths()
    proc = subprocess.run(
        ["python", "inference_basic.py",
         "--pretrained_model_name_or_path", str(p["svd"]),
         "--validation_image", image,
         "--validation_control_folder", str(pose_dir),
         "--output_dir", str(out_dir),
         "--width", str(track.width),
         "--height", str(track.height),
         "--guidance_scale", str(guidance_scale),
         "--num_inference_steps", str(num_inference_steps),
         "--posenet_model_name_or_path", str(p["posenet"]),
         "--face_encoder_model_name_or_path", str(p["face_encoder"]),
         "--unet_model_name_or_path", str(p["unet"]),
         "--decode_chunk_size", "4",
         "--gradient_checkpointing"],
        cwd=str(p["repo"]), capture_output=True, timeout=3600)
    frames_dir = out_dir / "animated_images"
    frames = sorted(frames_dir.glob("frame_*.png")) if frames_dir.is_dir() else []
    if proc.returncode != 0 or not frames:
        raise RuntimeError(
            "StableAnimator inference failed: "
            + proc.stderr.decode()[-2000:])
    out = out_path or str(wd / "stableanimator.mp4")
    ff = _ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found — mp4 assembly needs ffmpeg")
    proc2 = subprocess.run(
        [ff, "-hide_banner", "-loglevel", "error", "-y",
         "-framerate", str(track.fps), "-i",
         str(frames_dir / "frame_%d.png"),
         "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", out],
        capture_output=True, timeout=600)
    if proc2.returncode != 0 or not os.path.exists(out):
        raise RuntimeError(
            "StableAnimator mp4 assembly failed: "
            + proc2.stderr.decode()[-2000:])
    return AnimResult(
        path=out, backend="stableanimator", action=track.action,
        note=("neural pose-guided ID-preserving render (StableAnimator, "
              "SVD-XT + Face Encoder + ID Adapter — no face "
              "post-processing)"))


def stableanimator_provision(*, box: dict | None = None) -> dict:
    """Provision the StableAnimator install: clone + weights.

    Returns a /models-shaped report (never raises): per-resource
    statuses installed / already-installed / unavailable / failed.
    The gate is honest and first: no local CUDA GPU with 16GB+ VRAM
    means no 16GB download — the report says so in plain words and
    stops. Torch is NEVER installed here (the status reason names
    torch 2.5.1 cu124 + xformers + requirements.txt as the remaining
    manual step); the weights and the repo are what Devon fetches
    herself. DWPose weights land in the MimicMotion dir when it
    already holds them (shared, never fetched twice).
    """
    from ...core.provision import ensure_resource
    steps: list[str] = []
    import shutil
    if not shutil.which("git"):
        return {"ok": False, "status": "unavailable",
                "detail": "git is not on this box — StableAnimator clones "
                          f"from {STABLEANIMATOR_REPO}",
                "steps": steps}
    try:
        from ...core.gpu import detect_gpu
        gpu = detect_gpu()
    except Exception as exc:  # noqa: BLE001 - probe, unavailable is valid
        record_failure("animator.stableanimator_provision", exc)
        gpu = None
    if gpu is None or gpu.path != "local" or not (gpu.vram_gb or 0) >= STABLEANIMATOR_MIN_VRAM_GB:
        vram = (gpu.vram_gb if gpu is not None else 0) or 0
        return {"ok": False, "status": "unavailable",
                "detail": (f"StableAnimator needs an NVIDIA GPU with "
                           f"{STABLEANIMATOR_MIN_VRAM_GB:.0f} GB+ VRAM — "
                           f"this box has {vram} GB local VRAM. The ~16 GB "
                           "weights are not downloaded onto a box that "
                           "cannot render them."),
                "steps": steps}
    repo = STABLEANIMATOR_MODELS_DIR / "repo"
    if (repo / "inference_basic.py").is_file():
        steps.append("repo already present, skipping clone")
    else:
        proc = subprocess.run(
            ["git", "clone", "--depth", "1", STABLEANIMATOR_REPO,
             str(repo)], capture_output=True, text=True, timeout=900)
        steps.append(f"git clone: rc={proc.returncode}")
        if proc.returncode != 0:
            steps.append(f"clone stderr: {(proc.stderr or '')[-300:]}")
            return {"ok": False, "status": "failed",
                    "detail": "StableAnimator clone failed",
                    "steps": steps}
    reports: dict[str, str] = {}
    failed = False
    for spec in stableanimator_resource_specs():
        try:
            facts = ensure_resource(spec)
            reports[spec.name] = ("already-installed" if facts.get("already")
                                  else "installed")
            steps.append(f"{spec.name}: {reports[spec.name]} "
                         f"({facts.get('bytes', 0)} bytes)")
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            reports[spec.name] = "failed"
            steps.append(f"{spec.name}: failed — {exc}")
            failed = True
    if failed:
        return {"ok": False, "status": "failed",
                "detail": "some StableAnimator weights did not provision — "
                          "see the steps",
                "steps": steps, "resources": reports}
    return {"ok": True, "status": "installed",
            "detail": (f"StableAnimator installed in "
                       f"{STABLEANIMATOR_MODELS_DIR} — ID-preserving "
                       "pose-driven video (MIT). Torch 2.5.1 cu124 + "
                       "xformers + requirements.txt still go in the "
                       "repo's own environment; Devon never installs "
                       "torch herself."),
            "steps": steps, "resources": reports}


def animate_and_speak(image: str, track: PoseTrack, text: str,
                      voice_name: str, *,
                      out_path: str | None = None,
                      workdir: str | None = None,
                      prefer: str = "auto",
                      lip_prefer: str = "auto") -> ChainResult:
    """The full chain: pose -> animate -> lip-sync -> voice -> video.

    1. direct_animate renders the body (stableanimator -> mimicmotion
       -> warp by profile; prefer forces a rung).
    2. face_box_from_pose reads the face box straight off the pose
       track — no detector needed, we KNOW where the face is.
    3. dub_video speaks the text in a catalogue voice and lip-syncs
       it onto the footage (TTS + lip sync + mux, one organ).
    The voice is the catalogue voice (UniversalTTS backends); the RVC
    timbre swap lives in the singing path, not the speak path — the
    chain voices honestly through dub_video and says so in the note.
    """
    res = direct_animate(image, track, out_path=None, workdir=workdir,
                         prefer=prefer)
    from .lipsync import dub_video, face_box_from_pose
    wd = Path(workdir) if workdir else Path(res.path).parent
    box = face_box_from_pose(track)
    dubbed = dub_video(res.path, text, voice_name, face_box=box,
                       workdir=str(wd), prefer=lip_prefer)
    out = out_path or dubbed.path
    if dubbed.path != out:
        import shutil
        shutil.move(dubbed.path, out)
    return ChainResult(
        path=out, animate_backend=res.backend,
        lipsync_backend=dubbed.backend, voice=voice_name,
        action=track.action,
        note=(f"{res.note} | dubbed with voice '{voice_name}' "
              f"({dubbed.backend} lip sync)"))


# ── CPU warp animator ────────────────────────────────────────────────
def _displacement_field(kp0: np.ndarray, kp1: np.ndarray, W: int, H: int,
                        grid: int = 10, radius: float = 0.25) -> tuple:
    """Full-res displacement field from keypoint motion (numpy, vectorized).

    Each grid vertex moves by the distance-weighted average of nearby
    keypoint deltas (gaussian falloff); the vertex field is bilinearly
    upsampled to full resolution. Vertices far from any motion stay put —
    background doesn't swim.
    """
    _require_numpy()
    delta = (kp1 - kp0) * np.array([W, H])  # normalized -> px
    gx = np.linspace(0, W, grid + 1)
    gy = np.linspace(0, H, grid + 1)
    GVx, GVy = np.meshgrid(gx, gy)  # (G+1, G+1)
    r2 = (radius * max(W, H)) ** 2
    dx = np.zeros_like(GVx)
    dy = np.zeros_like(GVy)
    kpx = (kp0[:, 0] * W)[None, None, :]
    kpy = (kp0[:, 1] * H)[None, None, :]
    d2 = (kpx - GVx[:, :, None]) ** 2 + (kpy - GVy[:, :, None]) ** 2
    w = np.exp(-d2 / (r2 + 1e-9))
    wsum = w.sum(axis=2, keepdims=True) + 1e-9
    dx = (w * delta[None, None, :, 0]).sum(axis=2) / wsum[:, :, 0]
    dy = (w * delta[None, None, :, 1]).sum(axis=2) / wsum[:, :, 0]
    # bilinear upsample to full res via PIL (float mode)
    _require_pil()
    dx_full = np.array(Image.fromarray(
        dx.astype(np.float32), mode="F").resize((W, H), Image.BILINEAR))
    dy_full = np.array(Image.fromarray(
        dy.astype(np.float32), mode="F").resize((W, H), Image.BILINEAR))
    return dx_full, dy_full


def _bilinear_sample(arr: np.ndarray, sx: np.ndarray,
                     sy: np.ndarray) -> np.ndarray:
    """Vectorized bilinear sample of HxWxC array at float coords."""
    H, W = arr.shape[:2]
    _require_numpy()
    sx = np.clip(sx, 0, W - 1.001)
    sy = np.clip(sy, 0, H - 1.001)
    x0 = sx.astype(int)
    y0 = sy.astype(int)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    fx = (sx - x0)[..., None]
    fy = (sy - y0)[..., None]
    a = arr[y0, x0]
    b = arr[y0, x1]
    c = arr[y1, x0]
    d = arr[y1, x1]
    return (a * (1 - fx) * (1 - fy) + b * fx * (1 - fy)
            + c * (1 - fx) * fy + d * fx * fy)


def _mesh_for_frame(base_img: Image.Image, kp0: np.ndarray, kp1: np.ndarray,
                    grid: int = 10, radius: float = 0.25,
                    strength: float = 1.0) -> Image.Image:
    """Warp base_img toward keypoint motion. Returns the warped frame."""
    W, H = base_img.size
    dx, dy = _displacement_field(kp0, kp1, W, H, grid=grid, radius=radius)
    _require_numpy()
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    sx = xs - dx * strength
    sy = ys - dy * strength
    arr = np.array(base_img)
    out = _bilinear_sample(arr, sx, sy).astype(np.uint8)
    _require_pil()
    return Image.fromarray(out)


# ── depth-aware warp ─────────────────────────────────────────────────
# Layered foreground/background warp. The gaussian-falloff displacement
# core (_displacement_field) is unchanged; depth only modulates HOW MUCH
# each layer follows it (near = 1.0x, far = 0.35x) and each layer is
# warped + composited back-to-front, so a moving foreground reveals the
# (less-moving) background instead of smearing it. Depth is a heuristic
# pseudo-depth map (documented, not neural) unless the caller supplies
# a real one.


def estimate_depth(image: Image.Image,
                   keypoints: np.ndarray | None = None) -> np.ndarray:
    """Heuristic pseudo-depth map, HxW float32 in [0,1] (1 = near).

    Base field: bottom-of-frame is nearer + center weighting, heavily
    smoothed (no hard edges). When ``keypoints`` (N,2 normalized) are
    given, gaussian splats at each joint pull the map near — the
    person reads as foreground. This is a compositional heuristic for
    layered warping, NOT neural depth (no MiDaS/Depth-Anything here).
    """
    W, H = image.size
    # work small, then upsample — the map must be smooth
    sw, sh = 64, 64
    _require_numpy()
    ys, xs = np.mgrid[0:sh, 0:sw].astype(np.float64)
    # bottom nearer + gentle center pull
    depth = 0.35 + 0.45 * (ys / sh)
    cx = np.exp(-(((xs / sw - 0.5) * 2) ** 2
                  + ((ys / sh - 0.5) * 2) ** 2) * 1.2)
    depth = depth * 0.7 + cx * 0.3
    if keypoints is not None:
        kp = np.asarray(keypoints, dtype=np.float64)
        kx = (kp[:, 0] * sw)[None, None, :]
        ky = (kp[:, 1] * sh)[None, None, :]
        d2 = (kx - xs[:, :, None]) ** 2 + (ky - ys[:, :, None]) ** 2
        splat = np.exp(-d2 / (2 * (sw * 0.06) ** 2)).max(axis=2)
        depth = np.maximum(depth, splat * 0.95 + 0.05)
    depth = np.clip(depth, 0, 1)
    _require_pil()
    big = Image.fromarray((depth * 255).astype(np.uint8)).resize(
        (W, H), Image.BILINEAR)
    return np.array(big).astype(np.float64) / 255.0


def depth_aware_warp(base_img: Image.Image, kp0: np.ndarray, kp1: np.ndarray,
                     depth: np.ndarray | None = None,
                     n_layers: int = 3, grid: int = 10,
                     radius: float = 0.25,
                     strength: float = 1.0) -> Image.Image:
    """Warp with layered foreground/background separation.

    The displacement field comes from the shared gaussian-falloff core;
    each depth layer follows it at 0.35x (far) .. 1.0x (near) and layers
    composite back-to-front, so foreground motion uncovers background
    instead of dragging it along. ``depth``: HxW [0,1], 1 = near.
    """
    W, H = base_img.size
    if depth is None:
        depth = estimate_depth(base_img, kp0)
    _require_numpy()
    depth = np.asarray(depth, dtype=np.float64)
    _require_pil()
    if depth.shape != (H, W):
        depth = np.array(Image.fromarray(
            (np.clip(depth, 0, 1) * 255).astype(np.uint8)
        ).resize((W, H), Image.BILINEAR)).astype(np.float64) / 255.0
    depth = np.clip(depth, 0, 1)

    n_layers = max(1, int(n_layers))
    mean_d = float(depth.mean())
    if n_layers == 1 or float(depth.max() - depth.min()) < 1e-6:
        # degenerate depth: single warp, weight from mean depth
        w = 0.35 + 0.65 * mean_d
        return _mesh_for_frame(base_img, kp0, kp1, grid=grid,
                               radius=radius, strength=strength * w)

    dx, dy = _displacement_field(kp0, kp1, W, H, grid=grid, radius=radius)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    arr = np.array(base_img)
    edges = np.quantile(depth, np.linspace(0, 1, n_layers + 1))
    feather = 0.08
    composite: np.ndarray | None = None
    for li in range(n_layers - 1, -1, -1):  # far -> near
        lo, hi = float(edges[li]), float(edges[li + 1])
        m = (np.clip((depth - (lo - feather)) / max(1e-9, feather), 0, 1)
             * np.clip(((hi + feather) - depth) / max(1e-9, feather), 0, 1))
        if m.max() <= 0:
            continue
        w = 0.35 + 0.65 * (li / max(1, n_layers - 1))
        sx = xs - dx * strength * w
        sy = ys - dy * strength * w
        warped = _bilinear_sample(arr, sx, sy)
        if composite is None:
            composite = warped
        else:
            composite = warped * m[..., None] + composite * (1 - m[..., None])
    out = composite if composite is not None else arr.astype(np.float64)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def warp_animate(image: str, track: PoseTrack, *,
                 out_path: str | None = None,
                 workdir: str | None = None,
                 grid: int = 10,
                 depth_layers: int = 0) -> AnimResult:
    """CPU fallback: pose-guided mesh warp. Real directed motion, honest label.

    Each frame warps the still toward the cumulative keypoint displacement
    from frame 0. The arm rises, the hand morphs — driven by the actual
    pose track, not random motion.

    ``depth_layers`` >= 2 enables the depth-aware layered warp
    (foreground/background separation from a pose-derived pseudo-depth
    map); 0/1 keeps the classic single-grid warp.
    """
    ff = _ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found — warp animator needs ffmpeg")
    _require_pil()
    base = Image.open(image).convert("RGB")
    W, H = base.size
    wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="warp_"))
    wd.mkdir(parents=True, exist_ok=True)
    kp0 = track.frames[0]
    n = track.n_frames
    depth = None
    if depth_layers >= 2:
        depth = estimate_depth(base, kp0)
    for i in range(n):
        # cumulative displacement from rest; warp is directed by the track
        if depth is not None:
            frame = depth_aware_warp(base, kp0, track.frames[i],
                                     depth=depth, n_layers=depth_layers,
                                     grid=grid)
        else:
            frame = _mesh_for_frame(base, kp0, track.frames[i], grid=grid)
        frame.save(wd / f"w_{i:04d}.png")
    out = out_path or str(wd / "warp.mp4")
    subprocess.run(
        [ff, "-hide_banner", "-loglevel", "error", "-y",
         "-framerate", str(track.fps), "-i", str(wd / "w_%04d.png"),
         "-pix_fmt", "yuv420p", out],
        check=True, capture_output=True, timeout=600)
    return AnimResult(
        path=out, backend="warp", action=track.action,
        note=("CPU mesh-warp"
              + (" (depth-layered)" if depth is not None else "")
              + ": directed by the pose track (arm rises, hand "
                "morphs). Honest 2.5D warp — not neural. For photoreal, run "
                "on a workstation with MimicMotion."))


def direct_animate(image: str, track: PoseTrack, *,
                   out_path: str | None = None,
                   workdir: str | None = None,
                   prefer: str = "auto") -> AnimResult:
    """Route to the best available animator. Never fakes it.

    prefer: "auto" | "neural" | "stableanimator" | "mimicmotion" |
    "warp". "auto" tries stableanimator (best ID fidelity) ->
    mimicmotion -> warp. "neural" tries the neural rungs in order and
    raises ModelUnavailable (naming both reasons) instead of silently
    falling back. An explicit rung name forces that rung.
    """
    if prefer in ("auto", "neural", "stableanimator"):
        st = stableanimator_status()
        if st["available"]:
            return stableanimator_animate(image, track, out_path=out_path,
                                          workdir=workdir)
        if prefer == "stableanimator":
            raise ModelUnavailable(st["reason"])
        sa_reason = st["reason"]
    else:
        sa_reason = ""
    if prefer in ("auto", "neural", "mimicmotion"):
        st = mimicmotion_status()
        if st["available"]:
            return mimicmotion_animate(image, track, out_path=out_path,
                                       workdir=workdir)
        if prefer == "mimicmotion":
            raise ModelUnavailable(st["reason"])
        if prefer == "neural":
            raise ModelUnavailable(
                "no neural animator available. "
                f"StableAnimator: {sa_reason} "
                f"MimicMotion: {st['reason']}")
    return warp_animate(image, track, out_path=out_path, workdir=workdir)
