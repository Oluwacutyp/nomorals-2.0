"""Lip sync — audio-driven lip movement on video/photo subjects.

Backends (mined, see LIPSYNC_MINING.md):
- Wav2Lip (zdh6090/Wav2Lip): video -> lip-synced video. ~2GB VRAM, 8GB GPU.
  Research license (non-commercial) — noted in status.
- LatentSync 1.5 (ByteDance, Apache 2.0): higher quality, 8GB VRAM.
  v1.6 needs 18GB — wired as alt.
- SadTalker: single photo + audio -> talking head with head motion.
- envelope_warp (CPU): audio RMS envelope -> jaw-region warp. REAL
  audio-driven motion (mouth opens on energy, closes on silence),
  honestly labeled. Needs a face box (or pose-derived).

lip_sync() routes to the best available. dub_video() orchestrates
translate-text -> TTS voice -> lip sync -> mux for dubbing.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

MODELS_DIR = Path.home() / ".devon-models"


class ModelUnavailable(RuntimeError):
    """Neural lip-sync backend not installed. Carries the fix."""


@dataclass
class LipSyncResult:
    path: str
    backend: str          # "wav2lip" | "latentsync" | "sadtaker" | "warp"
    note: str = ""


def _ffmpeg() -> str | None:
    from shutil import which
    return which("ffmpeg")


def _has_torch_cuda() -> tuple[bool, bool]:
    try:
        import torch
        return True, torch.cuda.is_available()
    except Exception:
        return False, False


# ── backend status checks ────────────────────────────────────────────
def wav2lip_status() -> dict:
    has_torch, cuda = _has_torch_cuda()
    repo = MODELS_DIR / "wav2lip" / "repo"
    weights = MODELS_DIR / "wav2lip" / "wav2lip.pth"
    sfd = MODELS_DIR / "wav2lip" / "s3fd.pth"
    ok = has_torch and cuda and repo.is_dir() and weights.exists()
    return {
        "available": bool(ok), "torch": has_torch, "cuda": cuda,
        "license": "research/non-commercial (Wav2Lip upstream)",
        "reason": (
            "Wav2Lip needs torch+CUDA, the repo, and weights "
            "(wav2lip.pth ~350MB + s3fd.pth ~100MB). "
            "Install: git clone https://github.com/zdh6090/Wav2Lip "
            f"{repo} && download weights into {MODELS_DIR / 'wav2lip'}. "
            "See LIPSYNC_MINING.md."
        ),
    }


def latentsync_status() -> dict:
    has_torch, cuda = _has_torch_cuda()
    repo = MODELS_DIR / "latentsync" / "repo"
    unet = MODELS_DIR / "latentsync" / "latentsync_unet.pt"
    ok = has_torch and cuda and repo.is_dir() and unet.exists()
    vram = 0
    if cuda:
        try:
            import torch
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        except Exception:
            pass
    return {
        "available": bool(ok and vram >= 7.5), "torch": has_torch,
        "cuda": cuda, "vram_gb": round(vram, 1),
        "license": "Apache 2.0 (commercial OK)",
        "reason": (
            "LatentSync 1.5 needs torch+CUDA (8GB+ VRAM), the repo, and "
            "checkpoints (latentsync_unet.pt + whisper tiny.pt). "
            "Install: git clone https://github.com/bytedance/LatentSync "
            f"{repo} && download checkpoints into "
            f"{MODELS_DIR / 'latentsync'}. v1.6 needs 18GB VRAM."
        ),
    }


def sadtaker_status() -> dict:
    has_torch, cuda = _has_torch_cuda()
    repo = MODELS_DIR / "sadtaker" / "repo"
    ok = has_torch and cuda and repo.is_dir()
    return {
        "available": bool(ok), "torch": has_torch, "cuda": cuda,
        "reason": (
            "SadTalker needs torch+CUDA and the repo + checkpoints. "
            "Install: git clone https://github.com/OpenTalker/SadTalker "
            f"{repo} && follow its checkpoint download script."
        ),
    }


# ── neural backends (subprocess to the repos) ────────────────────────
def wav2lip_sync(video: str, audio: str, *, out_path: str | None = None,
                 workdir: str | None = None) -> LipSyncResult:
    st = wav2lip_status()
    if not st["available"]:
        raise ModelUnavailable(st["reason"])
    wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="w2l_"))
    wd.mkdir(parents=True, exist_ok=True)
    out = out_path or str(wd / "lipsync.mp4")
    repo = MODELS_DIR / "wav2lip" / "repo"
    ckpt = MODELS_DIR / "wav2lip" / "wav2lip.pth"
    proc = subprocess.run(
        ["python", "inference.py",
         "--checkpoint_path", str(ckpt),
         "--face", video, "--audio", audio,
         "--outfile", out],
        cwd=str(repo), capture_output=True, timeout=3600)
    if proc.returncode != 0 or not os.path.exists(out):
        raise RuntimeError("Wav2Lip failed: "
                           + proc.stderr.decode()[-2000:])
    return LipSyncResult(path=out, backend="wav2lip",
                         note="neural lip sync (Wav2Lip, 96px mouth region)")


def latentsync_sync(video: str, audio: str, *, out_path: str | None = None,
                    workdir: str | None = None) -> LipSyncResult:
    st = latentsync_status()
    if not st["available"]:
        raise ModelUnavailable(st["reason"])
    wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="ls_"))
    wd.mkdir(parents=True, exist_ok=True)
    out = out_path or str(wd / "lipsync.mp4")
    repo = MODELS_DIR / "latentsync" / "repo"
    proc = subprocess.run(
        ["bash", "inference.sh", video, audio, out],
        cwd=str(repo), capture_output=True, timeout=3600)
    if proc.returncode != 0 or not os.path.exists(out):
        raise RuntimeError("LatentSync failed: "
                           + proc.stderr.decode()[-2000:])
    return LipSyncResult(path=out, backend="latentsync",
                         note="neural lip sync (LatentSync, diffusion)")


def sadtaker_animate(image: str, audio: str, *,
                     out_path: str | None = None,
                     workdir: str | None = None) -> LipSyncResult:
    """Single photo + audio -> talking head with head motion."""
    st = sadtaker_status()
    if not st["available"]:
        raise ModelUnavailable(st["reason"])
    wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="st_"))
    wd.mkdir(parents=True, exist_ok=True)
    out = out_path or str(wd / "talking_head.mp4")
    repo = MODELS_DIR / "sadtaker" / "repo"
    proc = subprocess.run(
        ["python", "inference.py",
         "--source_image", image, "--driven_audio", audio,
         "--result_dir", str(wd)],
        cwd=str(repo), capture_output=True, timeout=3600)
    # SadTalker names output itself; find the mp4
    mp4s = sorted(wd.glob("*.mp4"))
    if proc.returncode != 0 or not mp4s:
        raise RuntimeError("SadTalker failed: "
                           + proc.stderr.decode()[-2000:])
    if str(mp4s[0]) != out:
        import shutil
        shutil.move(str(mp4s[0]), out)
    return LipSyncResult(path=out, backend="sadtaker",
                         note="talking head from photo (SadTalker, 3DMM)")


# ── CPU envelope warp ────────────────────────────────────────────────
def audio_envelope(audio: str, fps: float, n_frames: int) -> np.ndarray:
    """Per-video-frame mouth openness 0..1 from audio energy.

    RMS energy in 1/fps windows -> normalized -> envelope follower
    (fast attack, slower release) -> silence gate. REAL audio-driven
    signal: mouth opens on speech energy, closes on silence.
    """
    ff = _ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found")
    wd = Path(tempfile.mkdtemp(prefix="env_"))
    wav = str(wd / "a.wav")
    subprocess.run(
        [ff, "-hide_banner", "-loglevel", "error", "-y", "-i", audio,
         "-ac", "1", "-ar", "16000", "-f", "f32le", wav],
        check=True, capture_output=True, timeout=300)
    raw = np.fromfile(wav, dtype=np.float32)
    sr = 16000
    win = max(1, int(sr / fps))
    n = min(n_frames, len(raw) // win)
    rms = np.array([np.sqrt(np.mean(raw[i * win:(i + 1) * win] ** 2) + 1e-9)
                    for i in range(n)])
    # pad if audio shorter than video
    if n < n_frames:
        rms = np.pad(rms, (0, n_frames - n))
    # normalize by 95th percentile (robust to spikes)
    p95 = np.percentile(rms, 95) + 1e-9
    norm = np.clip(rms / p95, 0, 1)
    # envelope follower: fast attack, slow release
    env = np.zeros_like(norm)
    for i, v in enumerate(norm):
        prev = env[i - 1] if i else 0.0
        env[i] = v if v > prev else prev * 0.82 + v * 0.18
    # silence gate: below 8% energy -> closed
    env[env < 0.08] = 0.0
    return np.clip(env, 0, 1)


def envelope_warp_sync(video: str, audio: str,
                       face_box: tuple[float, float, float, float],
                       *, out_path: str | None = None,
                       workdir: str | None = None,
                       fps: int = 24) -> LipSyncResult:
    """CPU lip sync: audio envelope -> jaw-region warp.

    face_box: (x0, y0, x1, y1) normalized. The mouth region (lower
    third of the face box) stretches vertically with mouth openness.
    Honest label: envelope warp, not neural.
    """
    from .camera import read_frames, write_frames
    frames, _ = read_frames(video, fps=fps)
    if not frames:
        raise RuntimeError("no frames extracted")
    openness = audio_envelope(audio, float(fps), len(frames))
    W, H = frames[0].size
    fx0, fy0, fx1, fy1 = face_box
    x0, y0, x1, y1 = int(fx0 * W), int(fy0 * H), int(fx1 * W), int(fy1 * H)
    # mouth region: lower third of face box
    my0 = y0 + int((y1 - y0) * 0.55)
    my1 = y1
    out_frames = []
    for img, op in zip(frames, openness):
        if op < 0.02:
            out_frames.append(img)
            continue
        a = np.array(img)
        # vertical stretch of mouth region: scale 1 -> 1+0.9*openness
        region = a[my0:my1, x0:x1].copy()
        rh = region.shape[0]
        new_h = int(rh * (1 + 0.9 * op))
        stretched = np.array(
            Image.fromarray(region).resize((x1 - x0, new_h), Image.BICUBIC))
        # paste anchored at top of mouth region (jaw drops down)
        canvas = a.copy()
        end = min(my0 + new_h, H)
        canvas[my0:end, x0:x1] = stretched[:end - my0]
        # slight horizontal widen for natural open-mouth shape
        out_frames.append(Image.fromarray(canvas))
    out = out_path or str(
        Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="ew_"))
        / "warp_sync.mp4")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_frames(out_frames, out, fps=fps, audio_src=audio)
    return LipSyncResult(
        path=out, backend="warp",
        note=("CPU envelope warp: mouth opens on speech energy, closes on "
              "silence. Honest 2D warp — for neural quality use Wav2Lip / "
              "LatentSync on a GPU machine."))


# ── pose-derived face box (directed animation integration) ───────────
def face_box_from_pose(track) -> tuple[float, float, float, float]:
    """Estimate a face box from a pose track's head keypoints.

    Uses nose/eyes/ears (indices 0, 14-17). Returns normalized
    (x0, y0, x1, y1). This is how lip sync plugs into directed
    animation: we KNOW where the face is from the pose track.
    """
    kp = track.frames[len(track.frames) // 2]  # mid-action frame
    head = kp[[0, 14, 15, 16, 17]]
    x0, y0 = head.min(axis=0)
    x1, y1 = head.max(axis=0)
    # pad: face is bigger than the keypoint spread
    padx, pady = (x1 - x0) * 0.9 + 0.02, (y1 - y0) * 1.6 + 0.03
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 - 0.01
    return (max(0.0, cx - padx), max(0.0, cy - pady),
            min(1.0, cx + padx), min(1.0, cy + pady))


# ── router ───────────────────────────────────────────────────────────
def lip_sync(video: str, audio: str, *,
             face_box: tuple[float, float, float, float] | None = None,
             out_path: str | None = None,
             workdir: str | None = None,
             prefer: str = "auto",
             fps: int = 24) -> LipSyncResult:
    """Best-available lip sync. Never fakes it.

    prefer: "auto" | "wav2lip" | "latentsync" | "warp".
    Neural choices raise ModelUnavailable (with install instructions)
    instead of silently degrading.
    """
    order = {"auto": ["latentsync", "wav2lip", "warp"],
             "wav2lip": ["wav2lip"], "latentsync": ["latentsync"],
             "warp": ["warp"]}.get(prefer, ["latentsync", "wav2lip", "warp"])
    last_err = ""
    for b in order:
        try:
            if b == "latentsync":
                return latentsync_sync(video, audio, out_path=out_path,
                                       workdir=workdir)
            if b == "wav2lip":
                return wav2lip_sync(video, audio, out_path=out_path,
                                    workdir=workdir)
            if b == "warp":
                if face_box is None:
                    raise ModelUnavailable(
                        "warp lip sync needs a face_box (x0,y0,x1,y1 "
                        "normalized) — no face detector on this machine. "
                        "Pass the face region explicitly.")
                return envelope_warp_sync(video, audio, face_box,
                                          out_path=out_path,
                                          workdir=workdir, fps=fps)
        except ModelUnavailable as exc:
            last_err = str(exc)
            continue
    raise ModelUnavailable(
        "no lip-sync backend available. " + last_err)


# ── dubbing ──────────────────────────────────────────────────────────
def dub_video(video: str, translated_text: str, voice_name: str, *,
              face_box: tuple[float, float, float, float] | None = None,
              out_path: str | None = None,
              workdir: str | None = None,
              prefer: str = "auto", fps: int = 24) -> LipSyncResult:
    """Dub: translated text -> TTS in a catalogue voice -> lip sync -> mux.

    translated_text: the translation (the brain translates via LLM; no
    offline translator is wired — passing untranslated text dubs it
    as-is, honestly).
    """
    from ...voice.catalogue import default_catalogue
    wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="dub_"))
    wd.mkdir(parents=True, exist_ok=True)
    cat = default_catalogue()
    tts_out = str(wd / "dub_audio.wav")
    res = cat.speak_as(translated_text, voice_name, out_path=tts_out)
    audio_path = res.get("path") or tts_out
    if not os.path.exists(audio_path):
        raise RuntimeError(f"TTS produced no audio: {res}")
    synced = lip_sync(video, audio_path, face_box=face_box,
                      workdir=str(wd), prefer=prefer, fps=fps)
    out = out_path or str(wd / "dubbed.mp4")
    if synced.path != out:
        import shutil
        shutil.move(synced.path, out)
    synced.path = out
    synced.note += f" | dubbed with voice '{voice_name}'"
    return synced
