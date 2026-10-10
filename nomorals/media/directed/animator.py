"""Animation render backends — neural when available, honest warp on CPU.

- MimicMotionBackend: Tencent/MimicMotion, pose video + reference image ->
  photoreal directed video. Workstation (16GB VRAM). Wired now, not later.
- WarpAnimator: pose-guided mesh warp on CPU. REAL directed motion (the arm
  region rises, the hand morphs) via PIL MESH displacement driven by the
  same keypoint trajectories. Crude but honest — labeled `warp`, never
  claimed as neural.

direct_animate() routes: neural when the model is installed, warp fallback.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .pose_rig import PoseTrack, render_pose_video


class ModelUnavailable(RuntimeError):
    """Raised when a neural backend isn't installed. Carries the fix."""


@dataclass
class AnimResult:
    path: str
    backend: str          # "mimicmotion" | "warp"
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
    except Exception:
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


# ── CPU warp animator ────────────────────────────────────────────────
def _displacement_field(kp0: np.ndarray, kp1: np.ndarray, W: int, H: int,
                        grid: int = 10, radius: float = 0.25) -> tuple:
    """Full-res displacement field from keypoint motion (numpy, vectorized).

    Each grid vertex moves by the distance-weighted average of nearby
    keypoint deltas (gaussian falloff); the vertex field is bilinearly
    upsampled to full resolution. Vertices far from any motion stay put —
    background doesn't swim.
    """
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
    dx_full = np.array(Image.fromarray(
        dx.astype(np.float32), mode="F").resize((W, H), Image.BILINEAR))
    dy_full = np.array(Image.fromarray(
        dy.astype(np.float32), mode="F").resize((W, H), Image.BILINEAR))
    return dx_full, dy_full


def _bilinear_sample(arr: np.ndarray, sx: np.ndarray,
                     sy: np.ndarray) -> np.ndarray:
    """Vectorized bilinear sample of HxWxC array at float coords."""
    H, W = arr.shape[:2]
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
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    sx = xs - dx * strength
    sy = ys - dy * strength
    arr = np.array(base_img)
    out = _bilinear_sample(arr, sx, sy).astype(np.uint8)
    return Image.fromarray(out)


def warp_animate(image: str, track: PoseTrack, *,
                 out_path: str | None = None,
                 workdir: str | None = None,
                 grid: int = 10) -> AnimResult:
    """CPU fallback: pose-guided mesh warp. Real directed motion, honest label.

    Each frame warps the still toward the cumulative keypoint displacement
    from frame 0. The arm rises, the hand morphs — driven by the actual
    pose track, not random motion.
    """
    ff = _ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found — warp animator needs ffmpeg")
    base = Image.open(image).convert("RGB")
    W, H = base.size
    wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="warp_"))
    wd.mkdir(parents=True, exist_ok=True)
    kp0 = track.frames[0]
    n = track.n_frames
    for i in range(n):
        # cumulative displacement from rest; warp is directed by the track
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
        note=("CPU mesh-warp: directed by the pose track (arm rises, hand "
              "morphs). Honest 2.5D warp — not neural. For photoreal, run "
              "on a workstation with MimicMotion."))


def direct_animate(image: str, track: PoseTrack, *,
                   out_path: str | None = None,
                   workdir: str | None = None,
                   prefer: str = "auto") -> AnimResult:
    """Route to the best available animator. Never fakes it.

    prefer: "auto" | "neural" | "warp". "neural" raises ModelUnavailable
    with install instructions instead of silently falling back.
    """
    if prefer in ("auto", "neural"):
        st = mimicmotion_status()
        if st["available"]:
            return mimicmotion_animate(image, track, out_path=out_path,
                                       workdir=workdir)
        if prefer == "neural":
            raise ModelUnavailable(st["reason"])
    return warp_animate(image, track, out_path=out_path, workdir=workdir)
