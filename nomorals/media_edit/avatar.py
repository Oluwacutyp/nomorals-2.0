"""Talking-head avatar pipeline: photo + audio → lip-synced video.

``talking_head(photo, audio_or_text)`` — text goes through Devon's TTS
(``nomorals/voice/tts.py``) to a wav, then:

1. photo → still video (ffmpeg, duration = audio length) — LatentSync
   needs a *video* input, not a photo;
2. still video + audio → LatentSync lip-sync (local, zero per-minute cost);
3. output mp4 verified (exists, non-empty, duration ≈ audio duration).

LatentSync is a GitHub repo (bytedance/LatentSync) with no pip API, so
this drives its documented inference entry point::

    python -m scripts.inference \\
        --unet_config_path "configs/unet/stage2_512.yaml" \\
        --inference_ckpt_path "checkpoints/latentsync_unet.pt" \\
        --video_path <still.mp4> --audio_path <audio.wav> \\
        --video_out_path <out.mp4> \\
        --inference_steps 20 --guidance_scale 1.5

Setup is self-provisioned — Devon installs both backends herself
(``provision_backends()``): she clones the repos into
``~/LatentSync`` / ``~/Wav2Lip``, creates a venv in each repo,
installs the deps, downloads the checkpoints and wires the paths.
No setup is asked of the owner. LatentSync needs a GPU host (18 GB+
VRAM for the 1.6 512 config — its inference hardcodes CUDA), so on a
box without one it is honestly refused with the measured numbers;
Wav2Lip runs on CPU and is the fallback there. Optional env
overrides: ``LATENTSYNC_REPO`` / ``WAV2LIP_REPO`` (a checkout
elsewhere), ``LATENTSYNC_PYTHON`` / ``WAV2LIP_PYTHON`` (the python
to run the repo with — the repo's own ``.venv`` when deps live
there), ``LATENTSYNC_UNET_CONFIG``, ``LATENTSYNC_CKPT``,
``LATENTSYNC_STEPS``, ``LATENTSYNC_GUIDANCE``, ``WAV2LIP_CKPT``.

Upgrade path: ``backend="echomimic"`` (EchoMimicV2, half-body gestures)
is an honest stub — not implemented, no install path yet, never fake
video.
API fallback: ``backend="api"`` (PrunaAI P-Video) is an honest stub —
not wired, no install path yet — the termux path.

Nothing here fakes a video: every unavailable path raises AvatarError
with exact remediation.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


class AvatarError(Exception):
    """Talking-head generation unavailable or failed."""


class NotConfigured(AvatarError):
    """A backend exists on paper but isn't set up on this machine."""


# ---------------------------------------------------------------------------
# setup / availability
# ---------------------------------------------------------------------------

LATENTSYNC_INFERENCE_MODULE = "scripts.inference"
DEFAULT_UNET_CONFIG = "configs/unet/stage2_512.yaml"
DEFAULT_CKPT = "checkpoints/latentsync_unet.pt"
DEFAULT_STEPS = 20
DEFAULT_GUIDANCE = 1.5

#: Canonical homes for the provisioned checkouts (trellis/unirig/
#: ace-step shape: ~/Repo, env var overrides).
LATENTSYNC_DIR_NAME = "LatentSync"
WAV2LIP_DIR_NAME = "Wav2Lip"

LATENTSYNC_SETUP = (
    "LatentSync is not installed on this box — Devon installs it "
    "herself with the avatar provision (clone + deps + checkpoints, "
    "no setup needed from you). It needs a GPU host (18 GB+ VRAM), "
    "so on a box without one it is honestly refused instead.")


def latentsync_dir() -> Path:
    """Where the LatentSync checkout lives (env override or canonical)."""
    raw = (os.environ.get("LATENTSYNC_REPO") or "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / LATENTSYNC_DIR_NAME


def wav2lip_dir() -> Path:
    """Where the Wav2Lip checkout lives (env override or canonical)."""
    raw = (os.environ.get("WAV2LIP_REPO") or "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / WAV2LIP_DIR_NAME


def _venv_python(repo: Path) -> str | None:
    """The repo's own venv python, when the provision created one."""
    if os.name == "nt":
        candidate = repo / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = repo / ".venv" / "bin" / "python"
    return str(candidate) if candidate.is_file() else None


def latentsync_repo() -> Path:
    """Resolve the LatentSync checkout. Raises NotConfigured when missing."""
    repo = latentsync_dir()
    script = repo / "scripts" / "inference.py"
    if not script.exists():
        raise NotConfigured(
            f"no LatentSync checkout at {repo} "
            "(no scripts/inference.py) — " + LATENTSYNC_SETUP)
    return repo


def latentsync_python() -> str:
    """Python interpreter to run the repo with (venv-aware)."""
    env = (os.environ.get("LATENTSYNC_PYTHON") or "").strip()
    if env:
        return env
    return _venv_python(latentsync_dir()) or sys.executable


def avatar_available() -> tuple[bool, str]:
    """(True, '') when a local talking head can render right now."""
    from ..core.profiles import get_profile_kind
    kind = get_profile_kind()
    if kind == "termux":
        return (False, "talking head needs a GPU host — the API "
                       "fallback (PrunaAI P-Video) is not available either")
    try:
        repo = latentsync_repo()
    except NotConfigured as exc:
        return (False, str(exc).splitlines()[0])
    return (True, f"LatentSync ready at {repo}")


# ---------------------------------------------------------------------------
# audio: text → TTS wav, or an existing audio file
# ---------------------------------------------------------------------------

_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac"}


def _resolve_audio(audio_or_text: str | os.PathLike[str],
                   *, voice: str | None,
                   workdir: Path) -> tuple[Path, bool]:
    """Return (audio_path, was_synthesized). Never fakes audio."""
    candidate = Path(str(audio_or_text)).expanduser()
    if candidate.suffix.lower() in _AUDIO_SUFFIXES and candidate.exists():
        _log.info("avatar: using provided audio %s", candidate)
        return candidate, False
    text = str(audio_or_text or "").strip()
    if not text:
        raise AvatarError("talking head needs text to speak or an audio file")
    _log.info("avatar: synthesizing speech (%d chars)", len(text))
    from ..voice.tts import UniversalTTS
    engine = UniversalTTS(backend="auto")
    out = workdir / f"avatar-tts-{int(time.time() * 1000)}.wav"
    try:
        result = engine.speak(text, voice_name=voice, out_path=str(out))
    except Exception as exc:
        raise AvatarError(f"TTS failed, cannot voice the avatar: {exc}") from exc
    path = Path(result.get("path", ""))
    if not path.exists() or path.stat().st_size == 0:
        raise AvatarError("TTS produced no audio — avatar has no voice")
    _log.info("avatar: TTS done via %s → %s",
              result.get("backend"), path)
    return path, True


def _audio_duration_s(audio: Path) -> float:
    """Duration in seconds. wav via stdlib; anything else via ffprobe."""
    if audio.suffix.lower() == ".wav":
        try:
            import wave
            with wave.open(str(audio), "rb") as w:
                frames, rate = w.getnframes(), w.getframerate()
                if rate > 0:
                    return max(0.5, frames / rate)
        except Exception:  # noqa: BLE001 - fall through to ffprobe
            pass
    from .videos import run_ffmpeg  # noqa: F401  (import check only)
    import json as _json
    from .videos import ffmpeg_path
    proc = subprocess.run(
        [ffmpeg_path().replace("ffmpeg", "ffprobe"), "-v", "quiet",
         "-print_format", "json", "-show_format", str(audio)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    try:
        dur = float(_json.loads(proc.stdout)["format"]["duration"])
        return max(0.5, dur)
    except Exception as exc:
        raise AvatarError(
            f"could not read audio duration from {audio}: {exc}") from exc


# ---------------------------------------------------------------------------
# photo → still video (LatentSync needs video input)
# ---------------------------------------------------------------------------

def _photo_to_video(photo: Path, duration_s: float, workdir: Path) -> Path:
    """Loop a photo into a 25fps still video of ``duration_s`` seconds."""
    from .videos import run_ffmpeg
    if not photo.exists():
        raise AvatarError(f"no such photo: {photo}")
    out = workdir / f"avatar-still-{int(time.time() * 1000)}.mp4"
    res = run_ffmpeg([
        "-loop", "1", "-framerate", "25", "-i", str(photo),
        "-t", f"{duration_s:.2f}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", "scale=512:512:force_original_aspect_ratio=decrease,"
               "pad=512:512:(ow-iw)/2:(oh-ih)/2",
        str(out),
    ], timeout=max(120.0, duration_s * 4))
    if not out.exists() or out.stat().st_size == 0:
        raise AvatarError(
            f"ffmpeg could not turn the photo into video: {res.get('error')}")
    return out


# ---------------------------------------------------------------------------
# LatentSync inference
# ---------------------------------------------------------------------------

def _run_latentsync(video: Path, audio: Path, out: Path) -> Path:
    """Drive the repo's documented inference CLI. Verifies the output."""
    repo = latentsync_repo()
    unet_cfg = (os.environ.get("LATENTSYNC_UNET_CONFIG") or "").strip() \
        or DEFAULT_UNET_CONFIG
    ckpt = (os.environ.get("LATENTSYNC_CKPT") or "").strip() or DEFAULT_CKPT
    steps = (os.environ.get("LATENTSYNC_STEPS") or "").strip() \
        or str(DEFAULT_STEPS)
    guidance = (os.environ.get("LATENTSYNC_GUIDANCE") or "").strip() \
        or str(DEFAULT_GUIDANCE)
    ckpt_path = repo / ckpt if not Path(ckpt).is_absolute() else Path(ckpt)
    if not ckpt_path.exists():
        raise NotConfigured(
            f"LatentSync UNet checkpoint not found at {ckpt_path} — "
            "download latentsync_unet.pt into the repo's checkpoints/ "
            "(see the LatentSync README).")
    cmd = [
        latentsync_python(), "-m", LATENTSYNC_INFERENCE_MODULE,
        "--unet_config_path", unet_cfg,
        "--inference_ckpt_path", str(ckpt_path),
        "--video_path", str(video),
        "--audio_path", str(audio),
        "--video_out_path", str(out),
        "--inference_steps", str(steps),
        "--guidance_scale", str(guidance),
    ]
    _log.info("avatar: running LatentSync (%s steps, guidance %s)",
              steps, guidance)
    try:
        proc = subprocess.run(
            cmd, cwd=str(repo), capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=3600)
    except OSError as exc:
        raise AvatarError(
            f"could not start LatentSync ({latentsync_python()}): {exc}"
        ) from exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-1500:]
        raise AvatarError(
            f"LatentSync failed (exit {proc.returncode}):\n{tail}")
    if not out.exists() or out.stat().st_size == 0:
        raise AvatarError(
            "LatentSync exited 0 but produced no video — "
            "check the repo's logs above for face-detection failures")
    # duration sanity: output should roughly match the audio length
    try:
        out_dur = _audio_duration_s(out)
        aud_dur = _audio_duration_s(audio)
        if abs(out_dur - aud_dur) > max(2.0, aud_dur * 0.2):
            _log.warning("avatar: output duration %.1fs vs audio %.1fs — "
                         "lip-sync may be partial", out_dur, aud_dur)
    except AvatarError as _exc:
        _log.debug("avatar._run_latentsync failed", exc_info=True)
    return out


# ---------------------------------------------------------------------------
# Wav2Lip backend (lightweight alternative to LatentSync)
# ---------------------------------------------------------------------------
# Wav2Lip (Rudrabha/Wav2Lip, IIIT-H) is the strongest *lightweight* lip-sync
# baseline: accurate sync at ~25 fps on CPU-class hardware, at the cost of a
# static head (it repaints the mouth region onto the still video instead of
# diffusing the whole frame). Pick it when the GPU can't run LatentSync.
# License note: Wav2Lip's weights bar commercial use — the backend warns.

WAV2LIP_SETUP = (
    "Wav2Lip is not installed on this box — Devon installs it "
    "herself with the avatar provision (clone + deps + checkpoints, "
    "no setup needed from you). It runs on CPU, so it fits boxes "
    "without a GPU.")
WAV2LIP_CKPT = "checkpoints/wav2lip_gan.pth"
WAV2LIP_S3FD = "face_detection/detection/sfd/s3fd.pth"


def wav2lip_repo() -> Path:
    """Resolve the Wav2Lip checkout. Raises NotConfigured when missing."""
    repo = wav2lip_dir()
    script = repo / "inference.py"
    if not script.exists():
        raise NotConfigured(
            f"no Wav2Lip checkout at {repo} (no inference.py) — "
            + WAV2LIP_SETUP)
    return repo


def wav2lip_python() -> str:
    """Python interpreter to run the repo with (venv-aware)."""
    env = (os.environ.get("WAV2LIP_PYTHON") or "").strip()
    if env:
        return env
    return _venv_python(wav2lip_dir()) or sys.executable


def wav2lip_available() -> tuple[bool, str]:
    """(True, '') when Wav2Lip can render right now."""
    try:
        repo = wav2lip_repo()
    except NotConfigured as exc:
        return (False, str(exc).splitlines()[0])
    ckpt = repo / WAV2LIP_CKPT
    if not ckpt.exists():
        return (False, f"Wav2Lip checkpoint missing at {ckpt}")
    s3fd = repo / WAV2LIP_S3FD
    if not s3fd.exists():
        return (False, f"Wav2Lip face detector missing at {s3fd}")
    return (True, f"Wav2Lip ready at {repo}")


def _run_wav2lip(video: Path, audio: Path, out: Path) -> Path:
    """Drive Wav2Lip's documented inference CLI. Verifies the output."""
    repo = wav2lip_repo()
    ckpt = (os.environ.get("WAV2LIP_CKPT") or "").strip()
    ckpt_path = Path(ckpt) if ckpt else repo / WAV2LIP_CKPT
    if not ckpt_path.exists():
        raise NotConfigured(
            f"Wav2Lip checkpoint not found at {ckpt_path} — "
            "download wav2lip_gan.pth into the repo's checkpoints/ "
            "(see the Wav2Lip README).")
    _log.warning("avatar: Wav2Lip weights are research-only (no commercial "
                 "use) — prefer LatentSync for anything public")
    cmd = [
        wav2lip_python(), "inference.py",
        "--checkpoint_path", str(ckpt_path),
        "--face", str(video),
        "--audio", str(audio),
        "--outfile", str(out),
    ]
    _log.info("avatar: running Wav2Lip")
    try:
        proc = subprocess.run(
            cmd, cwd=str(repo), capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=3600)
    except OSError as exc:
        raise AvatarError(
            f"could not start Wav2Lip ({wav2lip_python()}): {exc}"
        ) from exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-1500:]
        raise AvatarError(
            f"Wav2Lip failed (exit {proc.returncode}):\n{tail}")
    if not out.exists() or out.stat().st_size == 0:
        raise AvatarError(
            "Wav2Lip exited 0 but produced no video — "
            "check the repo's logs above for face-detection failures")
    return out


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def talking_head(photo: str | os.PathLike[str],
                 audio_or_text: str | os.PathLike[str],
                 *, voice: str | None = None,
                 backend: str = "latentsync",
                 out_path: str | os.PathLike[str] | None = None,
                 workdir: str | os.PathLike[str] | None = None) -> Path:
    """Photo + audio/text → talking-head mp4.

    ``backend="latentsync"``: local LatentSync (zero per-minute cost,
    best quality).
    ``backend="wav2lip"``: local Wav2Lip — lighter/faster, static head,
    research-only license. Pick it when the GPU can't run LatentSync.
    ``backend="echomimic"``: EchoMimicV2 upgrade path — honest stub.
    ``backend="api"``: PrunaAI P-Video — honest stub.
    Returns the output mp4 Path. Never fakes a video.
    """
    backend = (backend or "latentsync").lower()
    if backend == "wav2lip":
        photo_p = Path(str(photo)).expanduser()
        work = Path(workdir).expanduser() if workdir else Path(
            tempfile.mkdtemp(prefix="avatar-"))
        work.mkdir(parents=True, exist_ok=True)
        _log.info("avatar: stage 1/3 — audio")
        audio_path, _synth = _resolve_audio(audio_or_text, voice=voice,
                                            workdir=work)
        duration = _audio_duration_s(audio_path)
        _log.info("avatar: stage 2/3 — still video (%.1fs)", duration)
        still = _photo_to_video(photo_p, duration, work)
        dest = Path(str(out_path)).expanduser() if out_path else \
            work / f"talking-head-{int(time.time())}.mp4"
        _log.info("avatar: stage 3/3 — Wav2Lip lip-sync")
        result = _run_wav2lip(still, audio_path, dest)
        _log.info("avatar: done → %s", result)
        return result
    if backend == "echomimic":
        raise NotConfigured(
            "EchoMimicV2 is not available on this box (upgrade path for "
            "half-body gestures). There is no install path for it yet — "
            "Devon cannot provision it herself, so it stays unavailable "
            "until a provision is built for it.")
    if backend == "api":
        raise NotConfigured(
            "PrunaAI P-Video API fallback is not available on this box "
            "(cloud talking-head backend). There is no install path for "
            "it yet — Devon cannot provision it herself, so it stays "
            "unavailable until a provision is built for it.")
    if backend != "latentsync":
        raise AvatarError(
            f"unknown avatar backend {backend!r}; "
            "use latentsync|wav2lip|echomimic|api")

    photo_p = Path(str(photo)).expanduser()
    work = Path(workdir).expanduser() if workdir else Path(
        tempfile.mkdtemp(prefix="avatar-"))
    work.mkdir(parents=True, exist_ok=True)

    _log.info("avatar: stage 1/3 — audio")
    audio_path, _synth = _resolve_audio(audio_or_text, voice=voice,
                                        workdir=work)
    duration = _audio_duration_s(audio_path)
    _log.info("avatar: stage 2/3 — still video (%.1fs)", duration)
    still = _photo_to_video(photo_p, duration, work)
    dest = Path(str(out_path)).expanduser() if out_path else \
        work / f"talking-head-{int(time.time())}.mp4"
    _log.info("avatar: stage 3/3 — LatentSync lip-sync")
    result = _run_latentsync(still, audio_path, dest)
    _log.info("avatar: done → %s", result)
    return result


# ---------------------------------------------------------------------------
# self-provision: Devon installs both backends herself
# ---------------------------------------------------------------------------
# Same honest-reporting shape as /models (tools/models_provision):
# per-backend report dicts, statuses installed / already-installed /
# unavailable / failed, never raises, never a manual-step list.
# Seams are module attributes so tests can replace them.
# The owner is never asked to clone, install or export anything:
# the only genuine walls are a GPU the box does not have (LatentSync
# on a CPU-only box) and a login or payment (neither backend needs
# one — both repos and checkpoints are public).

LATENTSYNC_REPO_URL = "https://github.com/bytedance/LatentSync"
WAV2LIP_REPO_URL = "https://github.com/Rudrabha/Wav2Lip"
LATENTSYNC_HF_REPO = "ByteDance/LatentSync-1.6"
LATENTSYNC_MIN_VRAM_GB = 18.0
WAV2LIP_GAN_DRIVE_URL = (
    "https://drive.google.com/file/d/15G3U08c8xsCkOqQxE38Z2XXDnPcOptNk/view")
WAV2LIP_S3FD_URL = (
    "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth")

#: family, label, download MB, disk floor GB, RAM floor GB.
_BACKEND_INFO: dict[str, dict[str, Any]] = {
    "latentsync": {"family": "talking-head",
                   "label": "LatentSync 1.6 lip-sync (diffusion, GPU)",
                   "size_mb": 9000, "disk_gb": 12.0, "ram_gb": 4.0},
    "wav2lip": {"family": "talking-head",
                "label": "Wav2Lip lip-sync (lightweight, CPU)",
                "size_mb": 2000, "disk_gb": 4.0, "ram_gb": 2.0},
}

BACKENDS: tuple[str, ...] = ("latentsync", "wav2lip")

#: Modern deps for Wav2Lip's venv. Upstream's requirements.txt pins
#: torch==1.1.0 / numpy==1.17.1 (2020, Python 3.6) — no wheels exist
#: for the owner's stack, so the provision installs the current set
#: instead. The first real render is the live proof it runs the
#: 2020-era inference code.
WAV2LIP_PIP_PACKAGES: tuple[str, ...] = (
    "torch", "librosa", "opencv-python", "scipy", "numpy", "tqdm",
    "face-detection", "Pillow",
)


def _run(args: list[str], timeout: int | None = None,
         cwd: str | Path | None = None) -> Any:
    """Subprocess seam (same shape as tools/models_provision._run)."""
    return subprocess.run(args, capture_output=True, text=True,
                          timeout=timeout,
                          cwd=str(cwd) if cwd is not None else None)


def _measure_box() -> dict[str, Any]:
    """Box measure seam — the characters provision's, lazily imported
    so this module stays importable without the characters organ."""
    from ..characters.provision import _measure_box as _mb

    return _mb()


def _probe_gpu() -> dict[str, Any]:
    """GPU probe seam — trellis's, the one GPU probe in the codebase."""
    from ..media.trellis import probe_gpu

    return probe_gpu()


def _download_url(url: str, dest: Path) -> Path:
    """Download a URL to dest (stdlib urllib). Returns dest."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Devon/1.0"})
    with urllib.request.urlopen(req, timeout=900) as resp, \
            open(dest, "wb") as fh:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    return dest


def _report(ok: bool, status: str, detail: str,
            steps: list[str] | None = None) -> dict[str, Any]:
    return {"ok": ok, "status": status, "detail": detail,
            "steps": list(steps or [])}


def _latentsync_installed() -> bool:
    """Repo + inference entry + both checkpoints present."""
    repo = latentsync_dir()
    return ((repo / "scripts" / "inference.py").is_file()
            and (repo / "checkpoints" / "latentsync_unet.pt").is_file()
            and (repo / "checkpoints" / "whisper" / "tiny.pt").is_file())


def _wav2lip_installed() -> bool:
    """Repo + inference entry + both checkpoints present."""
    repo = wav2lip_dir()
    return ((repo / "inference.py").is_file()
            and (repo / "checkpoints" / "wav2lip_gan.pth").is_file()
            and (repo / "face_detection" / "detection" / "sfd"
                 / "s3fd.pth").is_file())


def latentsync_gate(box: dict[str, Any]) -> tuple[bool, str]:
    """The install gate: Linux + NVIDIA GPU + 18 GB VRAM + Python 3.9-3.12.

    Returns (ok, reason) — the reason always names the measured
    numbers, so a refusal on EC2 (Windows, no GPU) or this box (no
    GPU) says exactly what is missing. Upstream's inference hardcodes
    device="cuda", so CPU is not a degraded path — it is a refusal.
    """
    box = dict(box or {})
    profile = str(box.get("profile") or "")
    if profile == "termux":
        return (False, "LatentSync cannot install on termux — talking "
                       "head needs a GPU host; use the API fallback")
    platform = str(box.get("platform") or sys.platform)
    cpus = box.get("cpus", "?")
    ram = box.get("total_ram_gb", "?")
    if not platform.startswith("linux"):
        return (False, f"LatentSync is tested only on Linux — this box "
                       f"runs {platform} ({cpus} CPUs, {ram} GB RAM); "
                       f"it cannot install here")
    gpu = _probe_gpu()
    if not gpu.get("has_gpu"):
        return (False, f"LatentSync needs an NVIDIA GPU with "
                       f"{LATENTSYNC_MIN_VRAM_GB:.0f} GB+ VRAM — no "
                       f"NVIDIA GPU detected on this box (0 GB VRAM, "
                       f"{cpus} CPUs, {ram} GB RAM)")
    vram = float(gpu.get("vram_gb") or 0.0)
    if vram < LATENTSYNC_MIN_VRAM_GB:
        return (False, f"LatentSync needs an NVIDIA GPU with "
                       f"{LATENTSYNC_MIN_VRAM_GB:.0f} GB+ VRAM — this "
                       f"box has {gpu.get('name')} with {vram} GB")
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 9) or (major, minor) >= (3, 13):
        return (False, f"LatentSync's torch 2.5.1 needs Python 3.9-3.12 "
                       f"— this box runs Python {major}.{minor}")
    return (True, f"GPU {gpu.get('name')} with {vram} GB VRAM — meets "
                  f"the {LATENTSYNC_MIN_VRAM_GB:.0f} GB floor")


def wav2lip_gate(box: dict[str, Any]) -> tuple[bool, str]:
    """The install gate: not termux. No GPU gate — upstream runs on
    CPU (device falls back when CUDA is absent)."""
    profile = str((box or {}).get("profile") or "")
    if profile == "termux":
        return (False, "Wav2Lip cannot install on termux — talking "
                       "head needs a GPU host; use the API fallback")
    return (True, "CPU install — no GPU needed")


def _floor_refusal(name: str, box: dict[str, Any],
                   disk_left: float) -> dict[str, Any]:
    info = _BACKEND_INFO[name]
    return _report(False, "unavailable",
                   f"needs {info['disk_gb']} GB free disk and "
                   f"{info['ram_gb']} GB RAM; box has {disk_left} GB disk "
                   f"free and {box.get('total_ram_gb', '?')} GB RAM total")


def _pip_into(package: str, steps: list[str]) -> bool:
    """pip install one package into Devon's own interpreter."""
    proc = _run([sys.executable, "-m", "pip", "install", "-q", package],
                timeout=1200)
    steps.append(f"pip install {package}: rc={proc.returncode}")
    if proc.returncode != 0:
        steps.append(f"pip stderr: {(proc.stderr or '')[-300:]}")
    return proc.returncode == 0


def _make_venv(repo: Path, steps: list[str]) -> str | None:
    """Create the repo's .venv (or reuse it). Returns its python."""
    venv_python = _venv_python(repo)
    if venv_python is not None:
        steps.append("venv already present, reusing it")
        return venv_python
    proc = _run([sys.executable, "-m", "venv", str(repo / ".venv")],
                timeout=600)
    steps.append(f"venv create: rc={proc.returncode}")
    if proc.returncode != 0:
        steps.append(f"venv stderr: {(proc.stderr or '')[-300:]}")
        return None
    return _venv_python(repo)


def _provision_latentsync(box: dict[str, Any]) -> dict[str, Any]:
    """LatentSync: clone + venv + requirements + HF checkpoints."""
    steps: list[str] = []
    import shutil

    gate_ok, gate_reason = latentsync_gate(box)
    if not gate_ok:
        return _report(False, "unavailable", gate_reason, steps)
    if not shutil.which("git"):
        return _report(False, "unavailable",
                       "git is not on this box — LatentSync clones from "
                       "github.com/bytedance/LatentSync", steps)
    repo = latentsync_dir()
    if (repo / "scripts" / "inference.py").is_file():
        steps.append("repo already present, skipping clone")
    else:
        proc = _run(["git", "clone", "--depth", "1",
                     LATENTSYNC_REPO_URL, str(repo)], timeout=900)
        steps.append(f"git clone: rc={proc.returncode}")
        if proc.returncode != 0:
            steps.append(f"clone stderr: {(proc.stderr or '')[-300:]}")
            return _report(False, "failed", "LatentSync clone failed",
                           steps)
    venv_python = _make_venv(repo, steps)
    if venv_python is None:
        return _report(False, "failed",
                       "LatentSync venv creation failed", steps)
    req = repo / "requirements.txt"
    proc = _run([venv_python, "-m", "pip", "install", "-q",
                 "-r", str(req)], timeout=1800, cwd=str(repo))
    steps.append(f"pip install -r requirements.txt: rc={proc.returncode}")
    if proc.returncode != 0:
        steps.append(f"pip stderr: {(proc.stderr or '')[-300:]}")
        return _report(False, "failed",
                       "LatentSync requirements install failed", steps)
    ckpt_dir = repo / "checkpoints"
    unet = ckpt_dir / "latentsync_unet.pt"
    tiny = ckpt_dir / "whisper" / "tiny.pt"
    if unet.is_file() and tiny.is_file():
        steps.append("checkpoints already present, skipping download")
    else:
        probe = _run([sys.executable, "-c", "import huggingface_hub"],
                     timeout=120)
        if probe.returncode != 0 and not _pip_into("huggingface-hub",
                                                   steps):
            return _report(False, "failed",
                           "huggingface_hub is needed for the "
                           "checkpoint download", steps)
        snippet = (
            "from huggingface_hub import snapshot_download; "
            f"snapshot_download(repo_id={LATENTSYNC_HF_REPO!r}, "
            "allow_patterns=['latentsync_unet.pt', 'whisper/tiny.pt'], "
            f"local_dir={str(ckpt_dir)!r})")
        proc = _run([sys.executable, "-c", snippet], timeout=1800)
        steps.append(f"download checkpoints: rc={proc.returncode}")
        if proc.returncode != 0:
            steps.append(f"download stderr: {(proc.stderr or '')[-300:]}")
            return _report(False, "failed",
                           "LatentSync checkpoints did not download", steps)
    if not _latentsync_installed():
        return _report(False, "failed",
                       "checkpoints downloaded but the install cannot "
                       "be verified", steps)
    verify = _run([venv_python, "-c",
                   "import torch, diffusers, omegaconf"], timeout=300)
    steps.append("import torch, diffusers, omegaconf: "
                 f"rc={verify.returncode}")
    if verify.returncode != 0:
        return _report(False, "failed",
                       "LatentSync deps installed but do not import — "
                       "see the steps above", steps)
    return _report(True, "installed",
                   f"LatentSync installed in {repo} — lip-sync via the "
                   f"talking head; the VAE downloads on first render",
                   steps)


def _provision_wav2lip(box: dict[str, Any]) -> dict[str, Any]:
    """Wav2Lip: clone + venv + modern deps + both checkpoints."""
    steps: list[str] = []
    import shutil

    gate_ok, gate_reason = wav2lip_gate(box)
    if not gate_ok:
        return _report(False, "unavailable", gate_reason, steps)
    if not shutil.which("git"):
        return _report(False, "unavailable",
                       "git is not on this box — Wav2Lip clones from "
                       "github.com/Rudrabha/Wav2Lip", steps)
    repo = wav2lip_dir()
    if (repo / "inference.py").is_file():
        steps.append("repo already present, skipping clone")
    else:
        proc = _run(["git", "clone", "--depth", "1",
                     WAV2LIP_REPO_URL, str(repo)], timeout=900)
        steps.append(f"git clone: rc={proc.returncode}")
        if proc.returncode != 0:
            steps.append(f"clone stderr: {(proc.stderr or '')[-300:]}")
            return _report(False, "failed", "Wav2Lip clone failed", steps)
    venv_python = _make_venv(repo, steps)
    if venv_python is None:
        return _report(False, "failed",
                       "Wav2Lip venv creation failed", steps)
    proc = _run([venv_python, "-m", "pip", "install", "-q",
                 "--extra-index-url",
                 "https://download.pytorch.org/whl/cpu",
                 *WAV2LIP_PIP_PACKAGES], timeout=1800, cwd=str(repo))
    steps.append(f"pip install deps: rc={proc.returncode}")
    if proc.returncode != 0:
        steps.append(f"pip stderr: {(proc.stderr or '')[-300:]}")
        return _report(False, "failed",
                       "Wav2Lip deps install failed", steps)
    ckpt = repo / "checkpoints" / "wav2lip_gan.pth"
    if ckpt.is_file():
        steps.append("wav2lip_gan.pth already present, skipping download")
    else:
        probe = _run([sys.executable, "-c", "import gdown"], timeout=120)
        if probe.returncode != 0 and not _pip_into("gdown", steps):
            return _report(False, "failed",
                           "gdown is needed for the checkpoint "
                           "download", steps)
        proc = _run([sys.executable, "-m", "gdown", "--fuzzy",
                     WAV2LIP_GAN_DRIVE_URL, "-O", str(ckpt)],
                    timeout=1800)
        steps.append(f"gdown wav2lip_gan.pth: rc={proc.returncode}")
        if proc.returncode != 0:
            steps.append(f"gdown stderr: {(proc.stderr or '')[-300:]}")
            return _report(False, "failed",
                           "wav2lip_gan.pth did not download", steps)
    s3fd = repo / "face_detection" / "detection" / "sfd" / "s3fd.pth"
    if s3fd.is_file():
        steps.append("s3fd.pth already present, skipping download")
    else:
        try:
            _download_url(WAV2LIP_S3FD_URL, s3fd)
            steps.append("download s3fd.pth: ok")
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            return _report(False, "failed",
                           f"s3fd.pth did not download: {exc}", steps)
    if not _wav2lip_installed():
        return _report(False, "failed",
                       "checkpoints downloaded but the install cannot "
                       "be verified", steps)
    verify = _run([venv_python, "-c",
                   "import torch, cv2, librosa, face_detection"],
                  timeout=300)
    steps.append("import torch, cv2, librosa, face_detection: "
                 f"rc={verify.returncode}")
    if verify.returncode != 0:
        return _report(False, "failed",
                       "Wav2Lip deps installed but do not import — "
                       "see the steps above", steps)
    return _report(True, "installed",
                   f"Wav2Lip installed in {repo} — lip-sync via the "
                   f"talking head (CPU); research-only weights, no "
                   f"commercial use", steps)


def provision_backends(backends: Any = None, *,
                       box: dict[str, Any] | None = None
                       ) -> dict[str, Any]:
    """Provision the named talking-head backends (default: both).

    Returns ``{"ok", "box", "backends"}`` in the /models report shape.
    Never raises — a backend that blows up reports ``failed``. Never
    emits a manual-step list: the owner is never asked to clone,
    install or export anything.
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
    for name in wanted:
        if name in unknown:
            continue
        installed = (_latentsync_installed() if name == "latentsync"
                     else _wav2lip_installed())
        if installed:
            reports[name] = _report(True, "already-installed",
                                    f"{name} already installed")
            continue
        gate = (latentsync_gate if name == "latentsync"
                else wav2lip_gate)
        gate_ok, gate_reason = gate(box)
        if not gate_ok:
            reports[name] = _report(False, "unavailable", gate_reason)
            continue
        info = _BACKEND_INFO[name]
        if total_ram < info["ram_gb"] or disk_left < info["disk_gb"]:
            reports[name] = _floor_refusal(name, box, disk_left)
            continue
        try:
            if name == "latentsync":
                reports[name] = _provision_latentsync(box)
            else:
                reports[name] = _provision_wav2lip(box)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            _log.debug("avatar provision %s failed", name, exc_info=True)
            reports[name] = _report(False, "failed",
                                    f"provision crashed: {exc}")
        if reports[name].get("ok"):
            disk_left = max(0.0, disk_left - info["size_mb"] / 1024.0)
    ok = bool(reports) and all(r["ok"] for r in reports.values()) \
        and not unknown
    result: dict[str, Any] = {"ok": ok, "box": box, "backends": reports}
    if unknown:
        result["error"] = "unknown backend(s): " + ", ".join(unknown)
    return result


def provision_report_text(result: dict[str, Any]) -> str:
    """Plain-text per-backend report for the chat reply."""
    box = result.get("box") or {}
    lines = ["🗣️ talking-head backends — provision"]
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
        lines.append(f"• {name}: {rep.get('status')} — "
                     f"{rep.get('detail')}")
        for step in rep.get("steps") or []:
            lines.append(f"    {step}")
    if result.get("error"):
        lines.append(str(result["error"]))
    lines.append("")
    lines.append("Wav2Lip weights are research-only (no commercial "
                 "use) — prefer LatentSync for anything public.")
    lines.append("LatentSync's VAE downloads on first render "
                 "(~335 MB, public).")
    return "\n".join(lines)


def list_backends() -> list[dict[str, Any]]:
    """Introspection: what avatar backends exist and their state."""
    ok, reason = avatar_available()
    wok, wreason = wav2lip_available()
    return [
        {"name": "latentsync", "kind": "local", "cost": "free",
         "available": ok, "note": reason or "ready"},
        {"name": "wav2lip", "kind": "local", "cost": "free",
         "available": wok,
         "note": (wreason or "ready — lighter/faster, static head, "
                  "research-only license")},
        {"name": "echomimic", "kind": "local", "cost": "free",
         "available": False,
         "note": "upgrade path (half-body) — not implemented"},
        {"name": "api", "kind": "api", "cost": "unknown",
         "available": False,
         "note": "PrunaAI P-Video — not implemented"},
    ]
