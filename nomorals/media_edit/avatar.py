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

Setup (fail-closed with these exact steps when missing):

    git clone https://github.com/bytedance/LatentSync
    cd LatentSync && pip install -r requirements.txt
    # download checkpoints/latentsync_unet.pt + whisper tiny.pt
    # (see the repo README) then:
    export LATENTSYNC_REPO=/path/to/LatentSync

Optional env overrides: ``LATENTSYNC_PYTHON`` (python to run the repo
with — use the repo's venv when deps live there), ``LATENTSYNC_UNET_CONFIG``,
``LATENTSYNC_CKPT``, ``LATENTSYNC_STEPS``, ``LATENTSYNC_GUIDANCE``.

Upgrade path: ``backend="echomimic"`` (EchoMimicV2, half-body gestures)
is an honest stub — raises with setup steps, never fake video.
API fallback: ``backend="api"`` (PrunaAI P-Video) is an honest stub with
the endpoint, key env var, and pricing — the termux path.

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

LATENTSYNC_SETUP = """LatentSync is not configured. Set it up:
  git clone https://github.com/bytedance/LatentSync
  cd LatentSync && pip install -r requirements.txt
  # download checkpoints/latentsync_unet.pt and whisper tiny.pt
  # (see the LatentSync README "Download checkpoints" section)
  export LATENTSYNC_REPO=/path/to/LatentSync
  # if the repo lives in its own venv:
  export LATENTSYNC_PYTHON=/path/to/LatentSync/.venv/bin/python"""


def latentsync_repo() -> Path:
    """Resolve the LatentSync checkout. Raises NotConfigured when missing."""
    raw = (os.environ.get("LATENTSYNC_REPO") or "").strip()
    if not raw:
        raise NotConfigured(LATENTSYNC_SETUP)
    repo = Path(raw).expanduser()
    script = repo / "scripts" / "inference.py"
    if not script.exists():
        raise NotConfigured(
            f"LATENTSYNC_REPO={repo} has no scripts/inference.py — "
            "is it a real LatentSync checkout?\n" + LATENTSYNC_SETUP)
    return repo


def latentsync_python() -> str:
    """Python interpreter to run the repo with (venv-aware)."""
    return (os.environ.get("LATENTSYNC_PYTHON") or "").strip() or sys.executable


def avatar_available() -> tuple[bool, str]:
    """(True, '') when a local talking head can render right now."""
    from ..core.profiles import get_profile_kind
    kind = get_profile_kind()
    if kind == "termux":
        return (False, "talking head needs a GPU host — use the API "
                       "fallback (backend='api', PrunaAI P-Video)")
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
        capture_output=True, text=True, timeout=30)
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
            cmd, cwd=str(repo), capture_output=True, text=True,
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
    except AvatarError:
        pass  # duration check is advisory, not fatal
    return out


# ---------------------------------------------------------------------------
# Wav2Lip backend (lightweight alternative to LatentSync)
# ---------------------------------------------------------------------------
# Wav2Lip (Rudrabha/Wav2Lip, IIIT-H) is the strongest *lightweight* lip-sync
# baseline: accurate sync at ~25 fps on CPU-class hardware, at the cost of a
# static head (it repaints the mouth region onto the still video instead of
# diffusing the whole frame). Pick it when the GPU can't run LatentSync.
# License note: Wav2Lip's weights bar commercial use — the backend warns.

WAV2LIP_SETUP = """Wav2Lip is not configured. Set it up:
  git clone https://github.com/Rudrabha/Wav2Lip
  cd Wav2Lip && pip install -r requirements.txt
  # download checkpoints/wav2lip_gan.pth (see the Wav2Lip README)
  export WAV2LIP_REPO=/path/to/Wav2Lip
  # if the repo lives in its own venv:
  export WAV2LIP_PYTHON=/path/to/Wav2Lip/.venv/bin/python"""
WAV2LIP_CKPT = "checkpoints/wav2lip_gan.pth"


def wav2lip_repo() -> Path:
    """Resolve the Wav2Lip checkout. Raises NotConfigured when missing."""
    raw = (os.environ.get("WAV2LIP_REPO") or "").strip()
    if not raw:
        raise NotConfigured(WAV2LIP_SETUP)
    repo = Path(raw).expanduser()
    script = repo / "inference.py"
    if not script.exists():
        raise NotConfigured(
            f"WAV2LIP_REPO={repo} has no inference.py — "
            "is it a real Wav2Lip checkout?\n" + WAV2LIP_SETUP)
    return repo


def wav2lip_python() -> str:
    """Python interpreter to run the repo with (venv-aware)."""
    return (os.environ.get("WAV2LIP_PYTHON") or "").strip() or sys.executable


def wav2lip_available() -> tuple[bool, str]:
    """(True, '') when Wav2Lip can render right now."""
    try:
        repo = wav2lip_repo()
    except NotConfigured as exc:
        return (False, str(exc).splitlines()[0])
    ckpt = repo / WAV2LIP_CKPT
    if not ckpt.exists():
        return (False, f"Wav2Lip checkpoint missing at {ckpt}")
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
        "--checkpoint", str(ckpt_path),
        "--face", str(video),
        "--audio", str(audio),
        "--outfile", str(out),
    ]
    _log.info("avatar: running Wav2Lip")
    try:
        proc = subprocess.run(
            cmd, cwd=str(repo), capture_output=True, text=True,
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
            "EchoMimicV2 is not wired yet (upgrade path for half-body "
            "gestures). Setup when ready:\n"
            "  git clone https://github.com/antgroup/echomimic_v2\n"
            "  # follow its README (needs ~24GB VRAM), then ask for the "
            "backend to be implemented against its inference CLI.")
    if backend == "api":
        raise NotConfigured(
            "PrunaAI P-Video API fallback is not wired yet. When you want "
            "it:\n"
            "  endpoint: https://api.pruna.ai (see PrunaAI docs for the "
            "current lip-sync route)\n"
            "  key env: PRUNAAI_API_KEY\n"
            "  pricing: ~$0.025/sec of output (check current pricing)\n"
            "Then ask for the API backend to be implemented.")
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
        {"name": "api", "kind": "api", "cost": "~$0.025/sec",
         "available": False,
         "note": "PrunaAI P-Video — not implemented"},
    ]
