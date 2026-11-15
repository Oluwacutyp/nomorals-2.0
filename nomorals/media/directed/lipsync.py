"""Lip sync — audio-driven lip movement on video/photo subjects.

Backends (mined, see LIPSYNC_MINING.md):
- Wav2Lip (Rudrabha/Wav2Lip fork of zdh6090/Wav2Lip): video ->
  lip-synced video. Runs on CPU (~25 fps) — no GPU needed.
  Research license (non-commercial) — noted in status.
- LatentSync 1.6 (ByteDance, Apache 2.0): higher quality, 18GB VRAM.
  The 1.5 8GB config is NOT provisioned — the provision installs 1.6.
- SadTalker (OpenTalker, Apache 2.0): single photo + audio ->
  talking head with head motion, 4GB VRAM.
- envelope_warp (CPU): audio RMS envelope -> jaw-region warp. REAL
  audio-driven motion (mouth opens on energy, closes on silence),
  honestly labeled. Needs a face box (or pose-derived).

lip_sync() routes to the best available. dub_video() orchestrates
translate-text -> TTS voice -> lip sync -> mux for dubbing.
"""

from __future__ import annotations
from .failures import record_failure

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from ...core.gpu import detect_gpu

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

NUMPY_AVAILABLE = np is not None


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError(
            "nomorals.media.directed.lipsync is unavailable: numpy is not installed "
            "(pip install numpy)")

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]

PIL_AVAILABLE = Image is not None


def _require_pil() -> None:
    if Image is None:
        raise RuntimeError(
            "nomorals.media.directed.lipsync is unavailable: Pillow is not installed "
            "(pip install Pillow)")


MODELS_DIR = Path.home() / ".devon-models"

#: Canonical home for the provisioned SadTalker checkout (avatar
#: canonical shape: ~/Repo). The directed organ resolves the avatar
#: homes for LatentSync / Wav2Lip first (where the provision installs)
#: and falls back to the legacy ~/.devon-models layout, so a manual
#: install there keeps working.
SADTALKER_DIR_NAME = "SadTalker"


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
    """(torch installed, local CUDA available). Never raises.

    Delegates the CUDA half to the shared detector
    (:mod:`nomorals.core.gpu`) — the lip-sync drivers run as local
    subprocesses, so only the local path counts; a forwarded GPU is
    not local CUDA. Torch presence is a package question, answered
    with ``find_spec`` — the old code imported torch, paying a full
    torch import + CUDA init for a yes/no. A torch that cannot be
    found is still recorded under this site (Pass #106: a fallback
    leaves a trace), exactly as the old failed import was.
    """
    err: Exception | None = None
    try:
        has_torch = find_spec("torch") is not None
    except Exception as e:  # noqa: BLE001 - probe, False is a valid answer
        has_torch = False
        err = e
    if not has_torch:
        record_failure("lipsync._has_torch_cuda",
                       err or ImportError("torch is not installed"))
    try:
        cuda = detect_gpu().path == "local"
    except Exception:  # noqa: BLE001 - probe, False is a valid answer
        cuda = False
    return has_torch, cuda


# ── backend resolution ─────────────────────────────────────────────
# The avatar organ provisions LatentSync + Wav2Lip to its canonical
# homes (~/LatentSync, ~/Wav2Lip); the directed organ resolves those
# FIRST and falls back to the legacy ~/.devon-models layout, so a
# manual install there keeps working. SadTalker has no avatar home —
# the directed provision installs it to ~/SadTalker.

def _avatar_dirs() -> tuple[Path | None, Path | None]:
    """(latentsync_dir, wav2lip_dir) from the avatar organ, or (None, None)."""
    try:
        from ...media_edit.avatar import latentsync_dir, wav2lip_dir
        return latentsync_dir(), wav2lip_dir()
    except Exception as exc:
        record_failure("lipsync._avatar_dirs", exc)
        return None, None


def _wav2lip_repo() -> Path:
    _, w2l = _avatar_dirs()
    if w2l is not None and (w2l / "inference.py").is_file():
        return w2l
    return MODELS_DIR / "wav2lip" / "repo"


def _wav2lip_ckpt() -> Path:
    _, w2l = _avatar_dirs()
    if w2l is not None:
        canon = w2l / "checkpoints" / "wav2lip_gan.pth"
        if canon.is_file():
            return canon
    return MODELS_DIR / "wav2lip" / "wav2lip.pth"


def _wav2lip_s3fd() -> Path:
    _, w2l = _avatar_dirs()
    if w2l is not None:
        canon = w2l / "face_detection" / "detection" / "sfd" / "s3fd.pth"
        if canon.is_file():
            return canon
    return MODELS_DIR / "wav2lip" / "s3fd.pth"


def _latentsync_repo() -> Path:
    ls, _ = _avatar_dirs()
    if ls is not None and (ls / "scripts" / "inference.py").is_file():
        return ls
    return MODELS_DIR / "latentsync" / "repo"


def _latentsync_unet() -> Path:
    ls, _ = _avatar_dirs()
    if ls is not None:
        canon = ls / "checkpoints" / "latentsync_unet.pt"
        if canon.is_file():
            return canon
    return MODELS_DIR / "latentsync" / "latentsync_unet.pt"


def _latentsync_tiny() -> Path:
    ls, _ = _avatar_dirs()
    if ls is not None:
        canon = ls / "checkpoints" / "whisper" / "tiny.pt"
        if canon.is_file():
            return canon
    return MODELS_DIR / "latentsync" / "whisper" / "tiny.pt"


def _sadtaker_repo() -> Path:
    canon = Path.home() / SADTALKER_DIR_NAME
    if (canon / "inference.py").is_file():
        return canon
    return MODELS_DIR / "sadtaker" / "repo"


def _wav2lip_python() -> str:
    """Venv python for the avatar Wav2Lip repo, else sys.executable."""
    try:
        from ...media_edit.avatar import wav2lip_python
        return wav2lip_python()
    except Exception as exc:
        record_failure("lipsync._wav2lip_python", exc)
        return sys.executable


def _latentsync_python() -> str:
    """Venv python for the avatar LatentSync repo, else sys.executable."""
    try:
        from ...media_edit.avatar import latentsync_python
        return latentsync_python()
    except Exception as exc:
        record_failure("lipsync._latentsync_python", exc)
        return sys.executable


def _venv_python(repo: Path) -> str | None:
    """The repo's own .venv python, when one exists."""
    if os.name == "nt":
        candidate = repo / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = repo / ".venv" / "bin" / "python"
    return str(candidate) if candidate.is_file() else None


def _sadtaker_python() -> str:
    return _venv_python(_sadtaker_repo()) or sys.executable


# ── backend status checks ────────────────────────────────────────────
def wav2lip_status() -> dict:
    has_torch, cuda = _has_torch_cuda()
    repo = _wav2lip_repo()
    weights = _wav2lip_ckpt()
    sfd = _wav2lip_s3fd()
    # Wav2Lip runs on CPU — CUDA is NOT required (the avatar provision
    # installs it for CPU boxes). s3fd is required: without the face
    # detector the driver cannot render, so an install missing it is
    # honestly "not installed" and the provision fills the gap.
    ok = has_torch and repo.is_dir() and weights.exists() and sfd.exists()
    return {
        "available": bool(ok), "torch": has_torch, "cuda": cuda,
        "license": "research/non-commercial (Wav2Lip upstream)",
        "reason": (
            "Wav2Lip is not installed on this box — Devon installs it "
            "herself with the lip-sync provision (clone + deps + "
            "checkpoints, no setup needed from you). It runs on CPU, so "
            "it fits boxes without a GPU."
        ),
    }


def latentsync_status() -> dict:
    has_torch, cuda = _has_torch_cuda()
    repo = _latentsync_repo()
    unet = _latentsync_unet()
    tiny = _latentsync_tiny()
    ok = (has_torch and cuda and repo.is_dir() and unet.exists()
          and tiny.exists())
    # VRAM from the shared detector (local path only) — the same
    # measurement _has_torch_cuda just made, not a second torch import
    # for the same number. The detector never raises, but the status
    # check stays never-raises either way. A forwarded GPU that passed
    # attestation reports its attested VRAM here too, so the 18 GB
    # floor reads the proven number — but the driver renders locally,
    # so ``cuda`` (and therefore ``available``) stays a local question:
    # an attested remote GPU clears the floor without claiming this
    # box can render on it.
    vram = 0.0
    if cuda:
        try:
            vram = detect_gpu().vram_gb
        except Exception:  # noqa: BLE001 - probe, 0.0 is a valid answer
            vram = 0.0
    else:
        try:
            info = detect_gpu()
            if info.path == "forwarded" and info.attested:
                vram = info.vram_gb
        except Exception:  # noqa: BLE001 - probe, 0.0 is a valid answer
            vram = 0.0
    # The provision installs the 1.6 config (18GB floor), matching the
    # avatar gate — the 7.5GB gate was for the 1.5 config nothing
    # provisions and the driver never drove.
    return {
        "available": bool(ok and vram >= 18.0), "torch": has_torch,
        "cuda": cuda, "vram_gb": round(vram, 1),
        "license": "Apache 2.0 (commercial OK)",
        "reason": (
            "LatentSync 1.6 is not installed on this box — Devon "
            "installs it herself with the lip-sync provision (clone + "
            "deps + checkpoints, no setup needed from you). It needs "
            "torch+CUDA (18GB+ VRAM)."
        ),
    }


def sadtaker_status() -> dict:
    has_torch, cuda = _has_torch_cuda()
    repo = _sadtaker_repo()
    ok = (has_torch and cuda and repo.is_dir()
          and (repo / "inference.py").is_file()
          and bool(list((repo / "checkpoints").glob("SadTalker*.safetensors"))))
    return {
        "available": bool(ok), "torch": has_torch, "cuda": cuda,
        "reason": (
            "SadTalker is not installed on this box — Devon installs it "
            "herself with the lip-sync provision (clone + deps + "
            "checkpoints, no setup needed from you). It needs torch+CUDA "
            "(4GB+ VRAM)."
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
    repo = _wav2lip_repo()
    ckpt = _wav2lip_ckpt()
    proc = subprocess.run(
        [_wav2lip_python(), "inference.py",
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
    repo = _latentsync_repo()
    unet = _latentsync_unet()
    # The repo's inference.sh takes NO arguments — it runs the demo
    # files with defaults, so the old call could never lip-sync an
    # arbitrary video. Drive the documented module CLI with explicit
    # args, same invocation the avatar organ already proves.
    from ...media_edit.avatar import (
        DEFAULT_GUIDANCE,
        DEFAULT_STEPS,
        DEFAULT_UNET_CONFIG,
        LATENTSYNC_INFERENCE_MODULE,
    )
    unet_cfg = ((os.environ.get("LATENTSYNC_UNET_CONFIG") or "").strip()
                or DEFAULT_UNET_CONFIG)
    ckpt = (os.environ.get("LATENTSYNC_CKPT") or "").strip()
    if ckpt and not Path(ckpt).is_absolute():
        ckpt_path = repo / ckpt
    else:
        ckpt_path = Path(ckpt) if ckpt else unet
    steps = ((os.environ.get("LATENTSYNC_STEPS") or "").strip()
             or str(DEFAULT_STEPS))
    guidance = ((os.environ.get("LATENTSYNC_GUIDANCE") or "").strip()
                or str(DEFAULT_GUIDANCE))
    proc = subprocess.run(
        [_latentsync_python(), "-m", LATENTSYNC_INFERENCE_MODULE,
         "--unet_config_path", unet_cfg,
         "--inference_ckpt_path", str(ckpt_path),
         "--video_path", video,
         "--audio_path", audio,
         "--video_out_path", out,
         "--inference_steps", str(steps),
         "--guidance_scale", str(guidance)],
        cwd=str(repo), capture_output=True, timeout=3600)
    if proc.returncode != 0 or not os.path.exists(out):
        raise RuntimeError("LatentSync failed: "
                           + proc.stderr.decode()[-2000:])
    return LipSyncResult(path=out, backend="latentsync",
                         note="neural lip sync (LatentSync 1.6, diffusion)")


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
    repo = _sadtaker_repo()
    proc = subprocess.run(
        [_sadtaker_python(), "inference.py",
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


# ── CPU visemes: phoneme -> mouth-shape map ──────────────────────────
# The CPU fallback used to be jaw-only (mouth opens on energy). Now it
# renders real viseme shapes — jaw openness, lip width, lip rounding —
# driven either by phonemes (from text, uniform timing — honest, not
# forced alignment) or by an acoustic heuristic (energy + spectral
# centroid) when no text is available. Neural backends still win on
# quality; this is the honest CPU tier.

#: viseme -> (jaw_open 0..1, lip_width ~0.5..1.2, lip_round 0..1)
VISEME_SHAPES: dict[str, tuple[float, float, float]] = {
    "sil": (0.00, 1.00, 0.00),   # closed / silence
    "MBP": (0.00, 0.95, 0.10),   # m, b, p — lips pressed
    "FV":  (0.15, 1.00, 0.00),   # f, v — teeth on lower lip
    "TDN": (0.30, 1.00, 0.00),   # t, d, n — tongue to ridge
    "SZ":  (0.20, 1.05, 0.00),   # s, z — narrow hiss
    "TH":  (0.35, 0.95, 0.00),   # th — tongue between teeth
    "L":   (0.35, 1.00, 0.00),   # l
    "SH":  (0.45, 0.70, 0.70),   # sh, ch — rounded
    "IY":  (0.25, 1.20, 0.00),   # beat — wide smile
    "IH":  (0.30, 1.05, 0.00),   # bit
    "EH":  (0.45, 1.10, 0.00),   # bed
    "AE":  (0.75, 1.15, 0.00),   # cat — wide open
    "AH":  (0.90, 1.00, 0.00),   # father — tall open
    "AA":  (0.80, 1.05, 0.00),   # hot
    "AO":  (0.70, 0.90, 0.40),   # bought — open rounded
    "ER":  (0.40, 0.85, 0.30),   # bird — tight mid
    "OW":  (0.55, 0.65, 0.80),   # boat — rounded mid
    "UW":  (0.35, 0.60, 0.90),   # boot — small round
    "W":   (0.30, 0.55, 1.00),   # we — puckered
    "R":   (0.40, 0.80, 0.40),   # red — slight round
    "KG":  (0.60, 0.95, 0.10),   # k, g — back open
}

#: ARPAbet phoneme -> viseme
PHONEME_TO_VISEME: dict[str, str] = {
    "AA": "AA", "AE": "AE", "AH": "AH", "AO": "AO", "AW": "OW",
    "AY": "AH", "EH": "EH", "ER": "ER", "EY": "EH", "IH": "IH",
    "IY": "IY", "OW": "OW", "OY": "OW", "UH": "UW", "UW": "UW",
    "B": "MBP", "CH": "SH", "D": "TDN", "DH": "TH", "F": "FV",
    "G": "KG", "HH": "sil", "JH": "SH", "K": "KG", "L": "L",
    "M": "MBP", "N": "TDN", "NG": "KG", "P": "MBP", "R": "R",
    "S": "SZ", "SH": "SH", "T": "TDN", "TH": "TH", "V": "FV",
    "W": "W", "Y": "IY", "Z": "SZ", "ZH": "SH",
}


def phoneme_to_viseme(phoneme: str) -> str:
    """ARPAbet phoneme -> viseme name. Unknown -> silence (closed)."""
    return PHONEME_TO_VISEME.get((phoneme or "").strip().upper(), "sil")


def mouth_shape_for_viseme(viseme: str) -> tuple[float, float, float]:
    """Viseme name -> (jaw_open, lip_width, lip_round)."""
    return VISEME_SHAPES.get((viseme or "").strip().upper(), VISEME_SHAPES["sil"])


# ── compact English G2P ──────────────────────────────────────────────
# Common-word ARPAbet dictionary + rule-based fallback. Covers everyday
# speech well; long-tail words get a reasonable approximation. This is a
# heuristic front-end — for broadcast quality plug in a real G2P
# (e.g. g2p-en / phonemizer) via text_to_phonemes()'s interface.
_G2P_WORDS: dict[str, tuple[str, ...]] = {
    "the": ("DH", "AH"), "a": ("AH",), "an": ("AE", "N"),
    "and": ("AE", "N", "D"), "or": ("AO", "R"), "but": ("B", "AH", "T"),
    "to": ("T", "UW"), "of": ("AH", "V"), "in": ("IH", "N"),
    "on": ("AA", "N"), "for": ("F", "AO", "R"), "with": ("W", "IH", "DH"),
    "is": ("IH", "Z"), "are": ("AA", "R"), "was": ("W", "AH", "Z"),
    "were": ("W", "ER"), "be": ("B", "IY"), "been": ("B", "IH", "N"),
    "have": ("HH", "AE", "V"), "has": ("HH", "AE", "Z"),
    "had": ("HH", "AE", "D"), "do": ("D", "UW"), "does": ("D", "AH", "Z"),
    "did": ("D", "IH", "D"), "will": ("W", "IH", "L"),
    "would": ("W", "UH", "D"), "can": ("K", "AE", "N"),
    "could": ("K", "UH", "D"), "should": ("SH", "UH", "D"),
    "i": ("AY",), "you": ("Y", "UW"), "he": ("HH", "IY"),
    "she": ("SH", "IY"), "we": ("W", "IY"), "they": ("DH", "EY"),
    "it": ("IH", "T"), "this": ("DH", "IH", "S"), "that": ("DH", "AE", "T"),
    "these": ("DH", "IY", "Z"), "those": ("DH", "OW", "Z"),
    "my": ("M", "AY"), "your": ("Y", "AO", "R"), "his": ("HH", "IH", "Z"),
    "her": ("HH", "ER"), "our": ("AW", "ER"), "their": ("DH", "EH", "R"),
    "me": ("M", "IY"), "him": ("HH", "IH", "M"), "us": ("AH", "S"),
    "them": ("DH", "EH", "M"), "what": ("W", "AH", "T"),
    "when": ("W", "EH", "N"), "where": ("W", "EH", "R"),
    "who": ("HH", "UW"), "why": ("W", "AY"), "how": ("HH", "AW"),
    "not": ("N", "AA", "T"), "no": ("N", "OW"), "yes": ("Y", "EH", "S"),
    "hello": ("HH", "EH", "L", "OW"), "hi": ("HH", "AY"),
    "hey": ("HH", "EY"), "thanks": ("TH", "AE", "NG", "K", "S"),
    "thank": ("TH", "AE", "NG", "K"), "please": ("P", "L", "IY", "Z"),
    "good": ("G", "UH", "D"), "bad": ("B", "AE", "D"),
    "new": ("N", "UW"), "old": ("OW", "L", "D"), "big": ("B", "IH", "G"),
    "small": ("S", "M", "AO", "L"), "great": ("G", "R", "EY", "T"),
    "love": ("L", "AH", "V"), "like": ("L", "AY", "K"),
    "want": ("W", "AA", "N", "T"), "need": ("N", "IY", "D"),
    "know": ("N", "OW"), "think": ("TH", "IH", "NG", "K"),
    "see": ("S", "IY"), "look": ("L", "UH", "K"), "come": ("K", "AH", "M"),
    "go": ("G", "OW"), "going": ("G", "OW", "IH", "NG"),
    "make": ("M", "EY", "K"), "take": ("T", "EY", "K"),
    "give": ("G", "IH", "V"), "get": ("G", "EH", "T"),
    "got": ("G", "AA", "T"), "say": ("S", "EY"), "said": ("S", "EH", "D"),
    "tell": ("T", "EH", "L"), "talk": ("T", "AO", "K"),
    "speak": ("S", "P", "IY", "K"), "listen": ("L", "IH", "S", "AH", "N"),
    "watch": ("W", "AA", "CH"), "play": ("P", "L", "EY"),
    "work": ("W", "ER", "K"), "time": ("T", "AY", "M"),
    "day": ("D", "EY"), "night": ("N", "AY", "T"), "today": ("T", "UW", "D", "EY"),
    "now": ("N", "AW"), "here": ("HH", "IY", "R"), "there": ("DH", "EH", "R"),
    "very": ("V", "EH", "R", "IY"), "really": ("R", "IY", "L", "IY"),
    "just": ("JH", "AH", "S", "T"), "also": ("AO", "L", "S", "OW"),
    "well": ("W", "EH", "L"), "so": ("S", "OW"), "too": ("T", "UW"),
    "people": ("P", "IY", "P", "AH", "L"), "man": ("M", "AE", "N"),
    "woman": ("W", "UH", "M", "AH", "N"), "child": ("CH", "AY", "L", "D"),
    "world": ("W", "ER", "L", "D"), "life": ("L", "AY", "F"),
    "house": ("HH", "AW", "S"), "home": ("HH", "OW", "M"),
    "water": ("W", "AO", "T", "ER"), "food": ("F", "UW", "D"),
    "money": ("M", "AH", "N", "IY"), "car": ("K", "AA", "R"),
    "phone": ("F", "OW", "N"), "video": ("V", "IH", "D", "IY", "OW"),
    "music": ("M", "Y", "UW", "Z", "IH", "K"), "song": ("S", "AO", "NG"),
    "dance": ("D", "AE", "N", "S"), "party": ("P", "AA", "R", "T", "IY"),
}

_G2P_DIGRAPHS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sh", ("SH",)), ("ch", ("CH",)), ("th", ("TH",)), ("ph", ("F",)),
    ("wh", ("W",)), ("ng", ("NG",)), ("ck", ("K",)), ("dge", ("JH",)),
    ("tch", ("CH",)), ("ee", ("IY",)), ("oo", ("UW",)), ("ea", ("IY",)),
    ("ai", ("EY",)), ("ay", ("EY",)), ("ie", ("IY",)), ("oa", ("OW",)),
    ("ow", ("AW",)), ("ou", ("AW",)), ("oi", ("OY",)), ("oy", ("OY",)),
    ("au", ("AO",)), ("aw", ("AO",)), ("ew", ("UW",)), ("qu", ("K", "W")),
)

_G2P_VOWELS: dict[str, tuple[str, ...]] = {
    "a": ("AE",), "e": ("EH",), "i": ("IH",), "o": ("AA",), "u": ("AH",),
    "y": ("IY",),
}


def _g2p_rules(word: str) -> list[str]:
    """Rule-based fallback: digraphs, magic-e, c/g softening."""
    w = word.lower()
    out: list[str] = []
    i = 0
    magic_e = w.endswith("e") and len(w) > 2
    core = w[:-1] if magic_e else w
    while i < len(core):
        matched = False
        for dg, ph in _G2P_DIGRAPHS:
            if core.startswith(dg, i):
                out.extend(ph)
                i += len(dg)
                matched = True
                break
        if matched:
            continue
        ch = core[i]
        nxt = core[i + 1] if i + 1 < len(core) else ""
        if ch == "c":
            out.append("S" if nxt in "eiy" else "K")
        elif ch == "g":
            out.append("JH" if nxt in "eiy" else "G")
        elif ch == "x":
            out.extend(("K", "S"))
        elif ch in _G2P_VOWELS:
            v = _G2P_VOWELS[ch]
            if magic_e and ch == "a":
                v = ("EY",)
            elif magic_e and ch == "i":
                v = ("AY",)
            elif magic_e and ch == "o":
                v = ("OW",)
            out.extend(v)
        elif ch == "r":
            # r colors the previous vowel; keep simple: append R
            out.append("R")
        elif ch.isalpha():
            out.append({"b": "B", "d": "D", "f": "F", "h": "HH",
                        "j": "JH", "k": "K", "l": "L", "m": "M",
                        "n": "N", "p": "P", "s": "S", "t": "T",
                        "v": "V", "w": "W", "z": "Z"}.get(ch, "HH"))
        i += 1
    # collapse doubles
    return [p for j, p in enumerate(out) if j == 0 or p != out[j - 1]]


def text_to_phonemes(text: str) -> list[str]:
    """English text -> ARPAbet phoneme list.

    Dictionary for common words, rule-based fallback otherwise. Word
    boundaries emit nothing (timing is uniform downstream); sentence
    punctuation emits a short silence.
    """
    phonemes: list[str] = []
    for tok in (text or "").lower().split():
        word = "".join(c for c in tok if c.isalpha() or c == "'")
        if not word:
            continue
        seq = _G2P_WORDS.get(word)
        phonemes.extend(seq if seq else _g2p_rules(word))
        if tok and tok[-1] in ".!?":
            phonemes.append("sil")
    return phonemes


def phonemes_to_visemes(phonemes: list[str]) -> list[str]:
    """Phoneme list -> viseme name list."""
    return [phoneme_to_viseme(p) if p != "sil" else "sil" for p in phonemes]


def viseme_track(n_frames: int, phonemes: list[str], duration_s: float,
                 fps: float) -> list[tuple[float, float, float]]:
    """Per-frame mouth shapes from a phoneme list.

    Uniform timing (each phoneme gets an equal slice — honest, NOT
    forced alignment) with a 3-frame moving-average smooth so shapes
    flow into each other instead of snapping.
    """
    shapes = [mouth_shape_for_viseme(v)
              for v in phonemes_to_visemes(phonemes)]
    if not shapes:
        return [VISEME_SHAPES["sil"]] * n_frames
    per = max(1e-6, duration_s / len(shapes))
    raw = []
    for i in range(n_frames):
        t = i / max(1e-6, fps)
        idx = min(len(shapes) - 1, int(t / per))
        raw.append(shapes[idx])
    # coarticulation smoothing: moving average over shapes
    sm = []
    for i in range(n_frames):
        lo, hi = max(0, i - 1), min(n_frames, i + 2)
        win = raw[lo:hi]
        sm.append(tuple(sum(s[j] for s in win) / len(win) for j in range(3)))
    return sm


def acoustic_viseme(openness: float, centroid: float) -> str:
    """Heuristic viseme from audio features when no text is available.

    openness: 0..1 energy envelope; centroid: 0..1 spectral centroid.
    A rough guess — bright spectra read as front vowels, dark as
    rounded back vowels, high energy as open vowels. NOT phoneme
    recognition; labeled as such wherever it is used.
    """
    if openness < 0.12:
        return "sil"
    if openness > 0.62:
        return "AH" if centroid < 0.5 else "AE"
    if centroid > 0.62:
        return "IY" if openness < 0.40 else "EH"
    if centroid < 0.30:
        return "OW" if openness > 0.35 else "UW"
    return "EH" if openness < 0.45 else "AH"


# ── CPU envelope warp ────────────────────────────────────────────────
def audio_features(audio: str, fps: float,
                   n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-video-frame (mouth openness, spectral centroid) from audio.

    openness: 0..1 RMS-energy envelope (fast attack, slow release,
    silence-gated) — REAL audio-driven signal. centroid: 0..1 spectral
    centroid (brightness) for the acoustic-viseme heuristic.
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
    _require_numpy()
    raw = np.fromfile(wav, dtype=np.float32)
    sr = 16000
    win = max(1, int(sr / fps))
    n = min(n_frames, len(raw) // win)
    rms = np.zeros(n_frames)
    cent = np.zeros(n_frames)
    freqs = np.fft.rfftfreq(win, 1.0 / sr) / (sr / 2)
    hann = np.hanning(win)
    for i in range(n):
        seg = raw[i * win:(i + 1) * win]
        rms[i] = np.sqrt(np.mean(seg ** 2) + 1e-9)
        mag = np.abs(np.fft.rfft(seg * hann)) + 1e-9
        cent[i] = float(np.sum(freqs * mag) / np.sum(mag))
    # normalize energy by 95th percentile (robust to spikes)
    p95 = np.percentile(rms, 95) + 1e-9
    norm = np.clip(rms / p95, 0, 1)
    # envelope follower: fast attack, slow release
    env = np.zeros_like(norm)
    for i, v in enumerate(norm):
        prev = env[i - 1] if i else 0.0
        env[i] = v if v > prev else prev * 0.82 + v * 0.18
    # silence gate: below 8% energy -> closed
    env[env < 0.08] = 0.0
    return np.clip(env, 0, 1), np.clip(cent, 0, 1)


def audio_envelope(audio: str, fps: float, n_frames: int) -> np.ndarray:
    """Per-video-frame mouth openness 0..1 from audio energy.

    Kept for backwards compatibility; see audio_features() for the
    (openness, centroid) pair the viseme warp uses.
    """
    return audio_features(audio, fps, n_frames)[0]


def _warp_mouth(frame_arr: np.ndarray,
                box: tuple[int, int, int, int],
                shape: tuple[float, float, float]) -> np.ndarray:
    """Reshape the mouth region to a viseme mouth shape.

    shape: (jaw_open, lip_width, lip_round). Jaw drops the region
    downward, width spreads/squeezes it horizontally, rounding puckers
    it (narrower + slight vertical pinch). Anchored at the top of the
    mouth so the upper lip stays put — the jaw does the moving.
    """
    x0, y0, x1, y1 = box
    jaw_open, lip_width, lip_round = shape
    H, W = frame_arr.shape[:2]
    x0, x1 = max(0, x0), min(W, x1)
    y0, y1 = max(0, y0), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return frame_arr
    region = frame_arr[y0:y1, x0:x1]
    rh, rw = region.shape[:2]
    new_w = max(1, int(rw * lip_width * (1.0 - 0.25 * lip_round)))
    new_h = max(1, int(rh * (1.0 + 1.1 * jaw_open)
                       * (1.0 - 0.15 * lip_round)))
    if new_w == rw and new_h == rh:
        return frame_arr
    _require_pil()
    _require_numpy()
    warped = np.array(
        Image.fromarray(region).resize((new_w, new_h), Image.BICUBIC))
    canvas = frame_arr.copy()
    cx = (x0 + x1) // 2
    px0 = cx - new_w // 2
    ye = min(y0 + new_h, H)
    xs0, xs1 = max(0, px0), min(W, px0 + new_w)
    if xs1 > xs0 and ye > y0:
        canvas[y0:ye, xs0:xs1] = warped[:ye - y0, xs0 - px0:xs0 - px0 + (xs1 - xs0)]
    return canvas


def envelope_warp_sync(video: str, audio: str,
                       face_box: tuple[float, float, float, float],
                       *, out_path: str | None = None,
                       workdir: str | None = None,
                       fps: int = 24,
                       phonemes: list[str] | None = None,
                       text: str | None = None) -> LipSyncResult:
    """CPU lip sync: audio -> viseme mouth shapes -> warp.

    face_box: (x0, y0, x1, y1) normalized. The mouth region (lower
    third of the face box) is reshaped per frame.

    Drive, best first:
    1. ``phonemes`` (ARPAbet) or ``text`` (compact G2P): real viseme
       shapes with uniform timing — honest, not forced alignment.
    2. neither: acoustic heuristic (energy + spectral centroid) picks a
       plausible viseme per frame — labeled as heuristic.

    Honest label: viseme warp, not neural.
    """
    from .camera import read_frames, write_frames
    frames, _ = read_frames(video, fps=fps)
    if not frames:
        raise RuntimeError("no frames extracted")
    if text and not phonemes:
        phonemes = text_to_phonemes(text)
    openness, centroid = audio_features(audio, float(fps), len(frames))
    if phonemes:
        shapes = viseme_track(len(frames), phonemes,
                              len(frames) / float(fps), float(fps))
        drive = f"text-driven visemes ({len(phonemes)} phonemes, uniform timing)"
    else:
        shapes = [mouth_shape_for_viseme(acoustic_viseme(op, ce))
                  for op, ce in zip(openness, centroid)]
        drive = "acoustic-heuristic visemes (energy + spectral centroid)"
    W, H = frames[0].size
    fx0, fy0, fx1, fy1 = face_box
    x0, y0, x1, y1 = int(fx0 * W), int(fy0 * H), int(fx1 * W), int(fy1 * H)
    # mouth region: lower third of face box
    my0 = y0 + int((y1 - y0) * 0.55)
    my1 = y1
    out_frames = []
    _require_pil()
    _require_numpy()
    for img, op, shape in zip(frames, openness, shapes):
        if op < 0.02:
            out_frames.append(img)
            continue
        a = np.array(img)
        out_frames.append(Image.fromarray(
            _warp_mouth(a, (x0, my0, x1, my1), shape)))
    out = out_path or str(
        Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="ew_"))
        / "warp_sync.mp4")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_frames(out_frames, out, fps=fps, audio_src=audio)
    return LipSyncResult(
        path=out, backend="warp",
        note=("CPU viseme warp (" + drive + "): mouth shapes follow the "
              "audio. Honest 2D warp — for neural quality use Wav2Lip / "
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
             fps: int = 24,
             text: str | None = None,
             phonemes: list[str] | None = None) -> LipSyncResult:
    """Best-available lip sync. Never fakes it.

    prefer: "auto" | "wav2lip" | "latentsync" | "warp".
    Neural choices raise ModelUnavailable (with the provision pointer)
    instead of silently degrading.
    ``text``/``phonemes`` drive viseme shapes in the CPU warp backend.
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
                                          workdir=workdir, fps=fps,
                                          text=text, phonemes=phonemes)
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
                      workdir=str(wd), prefer=prefer, fps=fps,
                      text=translated_text)
    out = out_path or str(wd / "dubbed.mp4")
    if synced.path != out:
        import shutil
        shutil.move(synced.path, out)
    synced.path = out
    synced.note += f" | dubbed with voice '{voice_name}'"
    return synced


# ── self-provision: Devon installs the backends herself ─────────────
# Same honest-reporting shape as /models (tools/models_provision) and
# the avatar provision (media_edit/avatar.provision_backends):
# per-backend report dicts, statuses installed / already-installed /
# unavailable / failed, never raises, never a manual-step list.
# LatentSync + Wav2Lip DELEGATE to the avatar provision (one
# provisioner, not two); SadTalker is the directed organ's own backend
# and gets its own provisioner here. Seams are module attributes so
# tests can replace them. The only genuine walls are a GPU the box
# does not have (LatentSync on a CPU-only box, SadTalker likewise)
# and a login or payment — no backend here needs one; all repos and
# checkpoints are public.

SADTALKER_REPO_URL = "https://github.com/OpenTalker/SadTalker"
SADTALKER_MIN_VRAM_GB = 4.0

#: family, label, download MB, disk floor GB, RAM floor GB. The
#: latentsync / wav2lip numbers mirror the avatar provision so the
#: wrapper's disk accounting stays consistent with its delegate.
_BACKEND_INFO: dict[str, dict[str, Any]] = {
    "latentsync": {"family": "lip-sync",
                   "label": "LatentSync 1.6 lip-sync (diffusion, GPU)",
                   "size_mb": 9000, "disk_gb": 12.0, "ram_gb": 4.0},
    "wav2lip": {"family": "lip-sync",
                "label": "Wav2Lip lip-sync (lightweight, CPU)",
                "size_mb": 2000, "disk_gb": 4.0, "ram_gb": 2.0},
    "sadtaker": {"family": "lip-sync",
                 "label": "SadTalker talking head (photo + audio, GPU)",
                 "size_mb": 6000, "disk_gb": 8.0, "ram_gb": 4.0},
}

BACKENDS: tuple[str, ...] = ("latentsync", "wav2lip", "sadtaker")

#: Modern deps for SadTalker's venv. Upstream documents torch 1.12.0
#: (2022, Python 3.8-3.10) — no wheels exist for the owner's stack, so
#: the provision installs the current set instead (same reasoning as
#: the avatar's Wav2Lip). The first real render is the live proof it
#: runs the 2023-era inference code.
SADTALKER_PIP_PACKAGES: tuple[str, ...] = (
    "torch", "torchvision", "torchaudio", "facexlib", "gfpgan",
    "insightface", "onnxruntime", "moviepy", "opencv-python-headless",
    "imageio[ffmpeg]", "yacs", "kornia", "safetensors", "pydub",
    "librosa", "numpy", "scipy", "Pillow", "tqdm",
)


def _run(args, timeout=None, cwd=None):
    return subprocess.run(args, capture_output=True, text=True,
                          timeout=timeout,
                          cwd=str(cwd) if cwd is not None else None)


def _measure_box() -> dict[str, Any]:
    from ...characters.provision import _measure_box as _mb
    return _mb()


def _probe_gpu() -> dict[str, Any]:
    from ...media.trellis import probe_gpu
    return probe_gpu()


def _report(ok, status, detail, steps=None):
    return {"ok": ok, "status": status, "detail": detail,
            "steps": list(steps or [])}


def _floor_refusal(name, box, disk_left):
    info = _BACKEND_INFO[name]
    return _report(
        False, "unavailable",
        f"{info['label']} needs {info['disk_gb']:.0f} GB disk and "
        f"{info['ram_gb']:.0f} GB RAM — this box has "
        f"{disk_left:.1f} GB free and {box.get('total_ram_gb', '?')} GB "
        f"total RAM ({box.get('free_ram_gb', '?')} GB free). It cannot "
        f"install here; it stays unavailable.")


def _latentsync_installed() -> bool:
    """File-level check (no torch import): repo + entry + checkpoints."""
    repo = _latentsync_repo()
    return ((repo / "scripts" / "inference.py").is_file()
            and _latentsync_unet().is_file()
            and _latentsync_tiny().is_file())


def _wav2lip_installed() -> bool:
    repo = _wav2lip_repo()
    return ((repo / "inference.py").is_file()
            and _wav2lip_ckpt().is_file()
            and _wav2lip_s3fd().is_file())


def _sadtaker_installed() -> bool:
    repo = _sadtaker_repo()
    return ((repo / "inference.py").is_file()
            and bool(list((repo / "checkpoints").glob("SadTalker*.safetensors"))))


def sadtaker_gate(box) -> tuple[bool, str]:
    """True/False + plain reason, for a box measured by _measure_box."""
    box = dict(box or {})
    profile = str(box.get("profile") or "")
    if profile == "termux":
        return (False, "SadTalker cannot install on termux — talking head needs a GPU host")
    platform = str(box.get("platform") or sys.platform)
    cpus = box.get("cpus", "?")
    ram = box.get("total_ram_gb", "?")
    if not platform.startswith("linux"):
        return (False, f"SadTalker is tested only on Linux — this box runs "
                       f"{platform} ({cpus} CPUs, {ram} GB RAM); it cannot install here")
    gpu = _probe_gpu()
    if not gpu.get("has_gpu"):
        return (False, f"SadTalker needs an NVIDIA GPU with "
                       f"{SADTALKER_MIN_VRAM_GB:.0f} GB+ VRAM — no NVIDIA GPU "
                       f"detected on this box (0 GB VRAM, {cpus} CPUs, "
                       f"{ram} GB RAM)")
    vram = float(gpu.get("vram_gb") or 0.0)
    if vram < SADTALKER_MIN_VRAM_GB:
        return (False, f"SadTalker needs an NVIDIA GPU with "
                       f"{SADTALKER_MIN_VRAM_GB:.0f} GB+ VRAM — this box has "
                       f"{gpu.get('name')} with {vram} GB")
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 9) or (major, minor) >= (3, 13):
        return (False, f"SadTalker's torch needs Python 3.9-3.12 — "
                       f"this box runs Python {major}.{minor}")
    return (True, f"GPU {gpu.get('name')} with {vram} GB VRAM — meets "
                  f"the {SADTALKER_MIN_VRAM_GB:.0f} GB floor")


def _make_venv(repo, steps):
    """repo/.venv via the venv module, with pip bootstrapped."""
    venv_dir = repo / ".venv"
    if not (venv_dir / "pyvenv.cfg").is_file():
        proc = _run([sys.executable, "-m", "venv", str(venv_dir)],
                    timeout=600)
        steps.append(f"python -m venv: rc={proc.returncode}")
        if proc.returncode != 0:
            steps.append(f"venv stderr: {(proc.stderr or '')[-300:]}")
            return None
    python = (venv_dir / "Scripts" / "python.exe" if os.name == "nt"
              else venv_dir / "bin" / "python")
    if not python.is_file():
        steps.append(f"venv python not found at {python}")
        return None
    proc = _run([str(python), "-m", "pip", "install", "-q",
                 "--upgrade", "pip"], timeout=600)
    steps.append(f"pip upgrade: rc={proc.returncode}")
    if proc.returncode != 0:
        steps.append(f"pip upgrade stderr: {(proc.stderr or '')[-300:]}")
        return None
    return str(python)


def _provision_sadtaker(box: dict[str, Any]) -> dict[str, Any]:
    """SadTalker: clone + venv + modern deps + download script + verify."""
    steps: list[str] = []
    import shutil

    gate_ok, gate_reason = sadtaker_gate(box)
    if not gate_ok:
        return _report(False, "unavailable", gate_reason, steps)
    if not shutil.which("git"):
        return _report(False, "unavailable",
                       "git is not on this box — SadTalker clones from "
                       "github.com/OpenTalker/SadTalker", steps)
    repo = Path.home() / SADTALKER_DIR_NAME
    if (repo / "inference.py").is_file():
        steps.append("repo already present, skipping clone")
    else:
        proc = _run(["git", "clone", "--depth", "1",
                     SADTALKER_REPO_URL, str(repo)], timeout=900)
        steps.append(f"git clone: rc={proc.returncode}")
        if proc.returncode != 0:
            steps.append(f"clone stderr: {(proc.stderr or '')[-300:]}")
            return _report(False, "failed", "SadTalker clone failed", steps)
    venv_python = _make_venv(repo, steps)
    if venv_python is None:
        return _report(False, "failed", "SadTalker venv creation failed", steps)
    proc = _run([venv_python, "-m", "pip", "install", "-q",
                 *SADTALKER_PIP_PACKAGES], timeout=1800, cwd=str(repo))
    steps.append(f"pip install deps: rc={proc.returncode}")
    if proc.returncode != 0:
        steps.append(f"pip stderr: {(proc.stderr or '')[-300:]}")
        return _report(False, "failed", "SadTalker deps install failed", steps)
    if _sadtaker_installed():
        steps.append("checkpoints already present, skipping download")
    else:
        script = repo / "scripts" / "download_models.sh"
        if not script.is_file():
            script = repo / "download_models.sh"
        if not script.is_file():
            return _report(False, "failed",
                           "SadTalker checkpoint script not found in the repo", steps)
        proc = _run(["bash", str(script)], timeout=1800, cwd=str(repo))
        steps.append(f"download checkpoints: rc={proc.returncode}")
        if proc.returncode != 0:
            steps.append(f"download stderr: {(proc.stderr or '')[-300:]}")
            return _report(False, "failed", "SadTalker checkpoints did not download", steps)
    if not _sadtaker_installed():
        return _report(False, "failed",
                       "checkpoints downloaded but the install cannot be verified", steps)
    verify = _run([venv_python, "-c", "import torch, librosa, cv2"],
                  timeout=300)
    steps.append(f"import torch, librosa, cv2: rc={verify.returncode}")
    if verify.returncode != 0:
        return _report(False, "failed",
                       "SadTalker deps installed but do not import — see the steps above", steps)
    return _report(True, "installed",
                   f"SadTalker installed in {repo} — talking head from a "
                   f"photo (Apache 2.0)", steps)


def provision_backends(backends=None, *, box=None) -> dict[str, Any]:
    """Provision the named lip-sync backends (default: all three).

    LatentSync + Wav2Lip delegate to the avatar provision (one
    provisioner, not two); SadTalker is provisioned here. Returns
    ``{"ok", "box", "backends"}`` in the /models report shape. Never
    raises — a backend that blows up reports ``failed``. Never emits
    a manual-step list.
    """
    wanted = list(BACKENDS if backends is None else backends)
    box = dict(box) if box is not None else _measure_box()
    try:
        disk_left = float(box.get("free_disk_gb") or 0)
    except (TypeError, ValueError):
        disk_left = 0.0
    try:
        total_ram = float(box.get("total_ram_gb") or 0)
    except (TypeError, ValueError):
        total_ram = 0.0

    reports: dict[str, dict[str, Any]] = {}
    unknown = [b for b in wanted if b not in _BACKEND_INFO]
    # Shared backends: the directed wrapper checks its OWN availability
    # first (canonical OR legacy) and short-circuits already-installed
    # — otherwise the avatar provision would re-install to the
    # canonical path, or report "unavailable" on a floor-refused box
    # where a legacy install exists.
    to_delegate = []
    for name in wanted:
        if name in unknown:
            continue
        if name in ("latentsync", "wav2lip"):
            installed = (_latentsync_installed() if name == "latentsync"
                         else _wav2lip_installed())
            if installed:
                reports[name] = _report(True, "already-installed",
                                        f"{name} already installed")
            else:
                to_delegate.append(name)
    if to_delegate:
        from ...media_edit.avatar import provision_backends as _avatar_provision
        try:
            shared_result = _avatar_provision(to_delegate, box=box)
            for name in to_delegate:
                rep = (shared_result.get("backends") or {}).get(name)
                reports[name] = (rep if rep is not None else _report(
                    False, "failed",
                    f"avatar provision returned no report for {name}"))
            for name in to_delegate:
                if reports[name].get("ok"):
                    disk_left = max(
                        0.0, disk_left - _BACKEND_INFO[name]["size_mb"] / 1024.0)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            for name in to_delegate:
                reports[name] = _report(False, "failed",
                                        f"avatar provision crashed: {exc}")
    for name in wanted:
        if name in unknown or name in reports:
            continue
        # Only sadtaker reaches here.
        if _sadtaker_installed():
            reports[name] = _report(True, "already-installed",
                                    "sadtaker already installed")
            continue
        gate_ok, gate_reason = sadtaker_gate(box)
        if not gate_ok:
            reports[name] = _report(False, "unavailable", gate_reason)
            continue
        info = _BACKEND_INFO[name]
        if total_ram < info["ram_gb"] or disk_left < info["disk_gb"]:
            reports[name] = _floor_refusal(name, box, disk_left)
            continue
        try:
            reports[name] = _provision_sadtaker(box)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            reports[name] = _report(False, "failed",
                                    f"provision crashed: {exc}")
        if reports[name].get("ok"):
            disk_left = max(0.0, disk_left - info["size_mb"] / 1024.0)
    ok = bool(reports) and all(r["ok"] for r in reports.values()) and not unknown
    result: dict[str, Any] = {"ok": ok, "box": box, "backends": reports}
    if unknown:
        result["error"] = "unknown backend(s): " + ", ".join(unknown)
    return result


def provision_report_text(result: dict[str, Any]) -> str:
    """Plain-text per-backend report for the chat reply."""
    box = result.get("box") or {}
    lines = ["🎬 lip-sync backends — provision"]
    if box:
        lines.append(
            f"box: {box.get('total_ram_gb', '?')} GB RAM "
            f"({box.get('free_ram_gb', '?')} GB free), "
            f"{box.get('free_disk_gb', '?')} GB disk free, "
            f"profile {box.get('profile') or '?'}")
    for name in BACKENDS:
        rep = (result.get("backends") or {}).get(name)
        if rep is None:
            continue
        lines.append(f"• {name}: {rep.get('status')} — {rep.get('detail')}")
        for step in rep.get("steps") or []:
            lines.append(f"    {step}")
    if result.get("error"):
        lines.append(str(result["error"]))
    lines.append("")
    lines.append("Wav2Lip weights are research-only (no commercial use) — "
                 "prefer LatentSync for anything public.")
    lines.append("LatentSync's VAE downloads on first render (~335 MB, public).")
    lines.append("SadTalker is Apache 2.0 (commercial OK).")
    return "\n".join(lines)
