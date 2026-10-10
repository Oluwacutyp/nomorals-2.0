"""RVC bridge — retrieval-based voice conversion as a post-stage.

The key architectural move of this whole phase: RVC (MIT,
RVC-Project/Retrieval-based-Voice-Conversion-WebUI) is speech→speech
conversion that PRESERVES the source's emotion, prosody, timing and melody
while swapping the identity. That makes it the universal carrier for:

- neural emotion: synthesize an expressive performance in any voice
  (expressive backend, or even DSP-shaped), RVC it into the target identity
  → the emotion transfers neurally, no pitch-hack.
- accent: synthesize in the target accent (multilingual backends),
  RVC identity transfer keeps the accent.
- singing: DiffSinger renders the melody in a voicebank voice,
  RVC swaps in the user's timbre.

Design: this module owns *detection + invocation* only. It shells to the
installed RVC (the reference WebUI CLI or `rvc-python`), never reimplements
the model. When RVC is absent, every caller gets an honest
:class:`RVCUnavailable` — never a fake conversion. All audio I/O is
16-bit PCM wav via stdlib `wave`, so nothing here needs numpy.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "RVCUnavailable",
    "RVCModel",
    "detect_rvc",
    "convert",
    "list_models",
]


class RVCUnavailable(Exception):
    """Raised when RVC is not installed — the caller must fall back honestly."""


@dataclass
class RVCModel:
    """One trained RVC voice model (.pth + .index)."""
    name: str
    pth: str
    index: str = ""


def _rvc_root() -> str:
    return os.environ.get("RVC_HOME", os.path.expanduser("~/.nomorals/rvc"))


def detect_rvc() -> dict[str, Any]:
    """Probe for a usable RVC install. Never raises.

    Returns {"ok": bool, "method": "cli"|"python"|"", "detail": str}.
    """
    # Method 1: rvc-python / the RVC WebUI infer CLI on PATH
    for exe in ("rvc", "rvc-infer"):
        if shutil.which(exe):
            return {"ok": True, "method": "cli", "detail": exe}
    # Method 2: RVC WebUI checkout with infer-web.py
    webui = os.path.join(_rvc_root(), "infer-web.py")
    if os.path.exists(webui):
        return {"ok": True, "method": "webui", "detail": webui}
    # Method 3: importable rvc_python
    try:
        import importlib.util
        if importlib.util.find_spec("rvc_python") is not None:
            return {"ok": True, "method": "python", "detail": "rvc_python"}
    except Exception:
        pass
    return {
        "ok": False, "method": "",
        "detail": ("RVC not installed. Install RVC-Project/Retrieval-based-"
                   "Voice-Conversion-WebUI (MIT) and set RVC_HOME, or "
                   "`pip install rvc-python`."),
    }


def list_models(models_dir: str = "") -> list[RVCModel]:
    """Trained .pth models under models_dir (default RVC_HOME/models)."""
    d = models_dir or os.path.join(_rvc_root(), "models")
    out: list[RVCModel] = []
    try:
        names = sorted(os.listdir(d))
    except OSError:
        return out
    for n in names:
        p = os.path.join(d, n)
        if os.path.isdir(p):
            pths = [f for f in os.listdir(p) if f.endswith(".pth")]
            idxs = [f for f in os.listdir(p) if f.endswith(".index")]
            if pths:
                out.append(RVCModel(
                    name=n, pth=os.path.join(p, sorted(pths)[0]),
                    index=os.path.join(p, sorted(idxs)[0]) if idxs else ""))
        elif n.endswith(".pth"):
            stem = n[:-4]
            idx = os.path.join(d, stem + ".index")
            out.append(RVCModel(name=stem, pth=p,
                                index=idx if os.path.exists(idx) else ""))
    return out


def _read_wav(path: str) -> tuple[array, int]:
    with wave.open(path, "rb") as w:
        n = w.getnchannels()
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    samp = array("h", raw)
    if n > 1:  # mono mix
        mono = array("h", [0]) * (len(samp) // n)
        for i in range(len(mono)):
            mono[i] = sum(samp[i * n:(i + 1) * n]) // n
        samp = mono
    return samp, sr


def _write_wav(path: str, samples: array, sr: int) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.tobytes())


def convert(input_wav: str, model: RVCModel | str, *,
            f0_up_key: int = 0,
            index_rate: float = 0.75,
            protect: float = 0.33,
            out_path: str = "") -> dict[str, Any]:
    """Convert input_wav's identity to the RVC model, preserving delivery.

    ``model``: RVCModel or a model name resolvable by list_models().
    Returns {"ok", "path", "method"} or raises RVCUnavailable / RuntimeError
    (honest — the caller falls back).
    """
    probe = detect_rvc()
    if not probe["ok"]:
        raise RVCUnavailable(probe["detail"])
    if isinstance(model, str):
        found = [m for m in list_models() if m.name == model]
        if not found:
            raise RVCUnavailable(
                f"RVC model '{model}' not found under "
                f"{os.path.join(_rvc_root(), 'models')}")
        model = found[0]
    dest = out_path or tempfile.mktemp(suffix="_rvc.wav", prefix="rvc_")
    method = probe["method"]
    try:
        if method == "cli":
            cmd = [
                probe["detail"], "infer",
                "--input", input_wav,
                "--model", model.pth,
                "--output", dest,
                "--f0-up-key", str(f0_up_key),
                "--index-rate", str(index_rate),
                "--protect", str(protect),
            ]
            if model.index:
                cmd += ["--index", model.index]
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=600)
            if r.returncode != 0 or not os.path.exists(dest):
                raise RuntimeError(
                    f"RVC CLI failed: {(r.stderr or r.stdout or '')[:300]}")
        elif method == "webui":
            cmd = [
                "python", probe["detail"], "--infer",
                input_wav, model.pth, dest,
                str(f0_up_key), str(index_rate), str(protect),
            ]
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=600)
            if r.returncode != 0 or not os.path.exists(dest):
                raise RuntimeError(
                    f"RVC WebUI failed: {(r.stderr or r.stdout or '')[:300]}")
        else:  # python
            from rvc_python import RVCInference  # type: ignore
            inf = RVCInference(model.pth, index_path=model.index or None)
            samp, sr = _read_wav(input_wav)
            out = inf.convert(samp, sr, f0_up_key=f0_up_key,
                              index_rate=index_rate, protect=protect)
            _write_wav(dest, array("h", out), sr)
    except RVCUnavailable:
        raise
    except Exception as exc:
        raise RuntimeError(f"RVC conversion failed: {exc}") from exc
    _log.info("rvc convert ok: %s -> %s (%s)", input_wav, dest, model.name)
    return {"ok": True, "path": dest, "method": method, "model": model.name}
