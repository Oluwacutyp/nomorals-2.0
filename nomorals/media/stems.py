"""Demucs stem separation — vocals / drums / bass / other.

Phased vocal pipeline (#58), stage 1: Demucs (MIT) is the easiest win
and the highest utility — karaoke tracks, remixes, vocal recovery from
a finished mix.

Fail-closed everywhere: if Demucs isn't installed or there's no usable
device, :func:`separate` raises :class:`DemucsUnavailable` with an
install recipe. Never fake stems.
"""

from __future__ import annotations

import logging
import os
import wave
from dataclasses import dataclass, field
from pathlib import Path

_log = logging.getLogger(__name__)

__all__ = [
    "DemucsUnavailable",
    "ProbeResult",
    "Stems",
    "INSTALL_HINT",
    "MODEL_URL",
    "STEM_NAMES",
    "probe_demucs",
    "separate",
]

MODEL_URL = "https://github.com/facebookresearch/demucs"
INSTALL_HINT = (
    "pip install demucs  (MIT)\n"
    "then: python -m demucs --mp3 <track>  (CLI)\n"
    f"or let Devon call it in-process. {MODEL_URL}"
)

#: The four stems Demucs always produces, in save order.
STEM_NAMES = ("vocals", "drums", "bass", "other")

#: Profiles too small for a 100M+ parameter separator.
TOO_SMALL = frozenset({"termux", "mobile", "embedded"})

#: Default Demucs model — hybrid transformer, best quality/speed trade.
DEFAULT_MODEL = "htdemucs"


class DemucsUnavailable(RuntimeError):
    """Demucs can't run here — with the install recipe attached."""


@dataclass
class ProbeResult:
    available: bool
    reason: str = ""          # human-readable when not available
    detail: str = ""          # extra context (install hint, device)
    profile: str = ""


@dataclass
class Stems:
    ok: bool
    source_path: str = ""
    vocals_path: str = ""
    drums_path: str = ""
    bass_path: str = ""
    other_path: str = ""
    model: str = DEFAULT_MODEL
    sample_rate: int = 44100
    error: str = ""
    note: str = ""

    def paths(self) -> dict[str, str]:
        return {
            "vocals": self.vocals_path,
            "drums": self.drums_path,
            "bass": self.bass_path,
            "other": self.other_path,
        }

    def missing(self) -> list[str]:
        return [k for k, v in self.paths().items() if not v]


def _importable(name: str) -> bool:
    try:
        __import__(name)
    except Exception:  # noqa: BLE001
        return False
    return True


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _detect_profile_kind() -> str:
    try:
        from ..core.profile import detect_profile
        return str(detect_profile().kind).lower()
    except Exception:  # noqa: BLE001
        return ""


def probe_demucs(profile: str = "") -> ProbeResult:
    """Check whether Demucs can run here. Never raises."""
    prof = (profile or "").strip().lower() or _detect_profile_kind()
    if prof in TOO_SMALL:
        return ProbeResult(
            available=False, profile=prof,
            reason=("Demucs needs a real machine — the separator won't "
                    "fit this device."),
        )
    if not _importable("demucs"):
        return ProbeResult(
            available=False, profile=prof,
            reason="Demucs isn't installed here.",
            detail=INSTALL_HINT,
        )
    device = "cuda" if _cuda_available() else "cpu"
    return ProbeResult(
        available=True, profile=prof,
        detail=f"demucs importable; device={device} "
               f"(CPU works, CUDA is ~10x faster)",
    )


def _write_wav_mono(path: str, samples: "object", sample_rate: int) -> None:
    """Mono float samples → 16-bit WAV, stdlib + numpy only."""
    import numpy as np
    arr = np.asarray(samples, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise DemucsUnavailable("separator returned an empty stem")
    peak = float(abs(arr).max()) or 1.0
    pcm16 = (arr / peak * 32767).astype("<i2")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())


class _RealSeparator:
    """Thin wrapper over ``demucs.api.Separator``.

    Real API (verified against the public repo):
    ``Separator(model=..., device=...)`` →
    ``separate_audio_file(path)`` → ``(origin, separated)`` where
    ``separated`` is a dict {vocals, drums, bass, other} of
    (channels, samples) torch tensors.
    """

    name = "demucs-api"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model
        self._sep = None

    def load(self) -> None:
        from demucs.api import Separator
        device = "cuda" if _cuda_available() else "cpu"
        self._sep = Separator(model=self.model, device=device)

    def unload(self) -> None:
        self._sep = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def separate(self, audio_path: str) -> tuple[dict[str, object], int]:
        import numpy as np
        assert self._sep is not None
        _origin, separated = self._sep.separate_audio_file(audio_path)
        sr = int(getattr(self._sep, "samplerate", 44100))
        out: dict[str, object] = {}
        for name in STEM_NAMES:
            tensor = separated[name]
            mono = tensor.detach().cpu().numpy().mean(axis=0)
            out[name] = np.asarray(mono, dtype=np.float64)
        return out, sr


def separate(audio_path: str, *, out_dir: str = "stems",
             model: str = DEFAULT_MODEL, profile: str = "",
             separator: "_RealSeparator | None" = None) -> Stems:
    """Split a finished mix into vocals/drums/bass/other.

    ``separator`` is injectable (tests). Otherwise the real Demucs
    adapter is used. Fails closed — never fake stems.
    """
    src = (audio_path or "").strip()
    if not src or not os.path.isfile(src):
        raise DemucsUnavailable(f"no such audio file: {src!r}")
    probe = probe_demucs(profile)
    if separator is None and not probe.available:
        raise DemucsUnavailable(
            probe.reason + (f"\n{probe.detail}" if probe.detail else ""))
    sep = separator if separator is not None else _RealSeparator(model)
    try:
        sep.load()
        stems, sr = sep.separate(src)
    finally:
        try:
            sep.unload()
        except Exception:  # noqa: BLE001
            _log.debug("separator unload failed", exc_info=True)
    out = Path(out_dir)
    result = Stems(ok=True, source_path=src, model=model, sample_rate=sr,
                   note="Demucs stem separation — 4 stems, no audio faked")
    for name in STEM_NAMES:
        dest = str(out / f"{Path(src).stem}_{name}.wav")
        _write_wav_mono(dest, stems[name], sr)
        setattr(result, f"{name}_path", dest)
    if result.missing():
        raise DemucsUnavailable(
            f"separator produced no stems: missing {result.missing()}")
    return result
