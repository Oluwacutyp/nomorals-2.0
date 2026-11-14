"""One command that installs every missing model.

The owner asked (2026-11-14): "I meant a command that generally
install every missing model." The per-area commands already exist
(``/character provision`` for the three character backends,
``/dj desk provision`` for madmom) — this module is the ONE command
that covers all of Devon's models: ``/models``.

Inventory (from the code, not a wishlist):

* characters — xtts / rvc / faceswap, DELEGATED to
  ``characters.provision.provision_backends`` (never duplicated)
* DJ — madmom, DELEGATED to ``media.dj_desk.provision_madmom``
* voice STT — faster-whisper / parakeet / whisper-cpp, profile order
  from ``audio.pipeline._STT_ORDER``
* voice TTS — chatterbox / kokoro / piper / f5tts, profile order from
  ``audio.pipeline._TTS_ORDER`` (xtts is the character backend)
* image — sd15 (Stable Diffusion 1.5 weights) / ace-step (ACE-Step
  1.5 engine + weights)
* small ML — sklearn / lightgbm / xgboost / catboost, via
  ``llm.specialists.backend_available``

Journal-trained learners (the ladder's data-owned jobs) are STATUS
ONLY — they train from journal rows, not downloads. ``learner_status``
reports rows held vs the job's floor and trained-or-not-yet, never
"installed".

Codebeast and the big-merge specialists are OUT OF SCOPE (separate
programmes) — the report says so, it does not fold them in.

Same shape as the character provision: explicit only (importing this
module installs nothing), per-backend report dicts, never raises, an
honest box check with floors naming measured numbers. No real
downloads happen in tests — the seams are the module-level ``_run``,
``_measure_box``, ``_detect_*`` functions and the two delegates.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..audio.pipeline import _STT_ORDER, _TTS_ORDER, detect_profile
from ..characters import provision as _char_provision
from ..characters.store import default_character_dir
from ..llm.specialists import backend_available, default_models_dir
from ..media import dj_desk as _dj_desk
from ..media.ace_step import CHECKPOINT_ENV
from ..media.dj_engine import read_feedback_log
from ..media.imggen import checkpoint_dir as _imggen_checkpoint_dir
from ..media.imggen import registry as _imggen_registry

# NOTE: nomorals.voice.fetch is imported lazily inside _piper_voice_path:
# importing the nomorals.voice package runs its __init__, which probes
# the CPU (platform.processor() shells out to `uname -p`). The
# orchestrator's import must stay inert.

_log = logging.getLogger(__name__)

#: The two delegates — module attributes so tests can replace them.
_char_provision_backends = _char_provision.provision_backends
_madmom_provision = _dj_desk.provision_madmom
_measure_box = _char_provision._measure_box

#: Installable backends, in install order.
BACKENDS: tuple[str, ...] = (
    "xtts", "rvc", "faceswap", "madmom",
    "faster-whisper", "parakeet", "whisper-cpp",
    "chatterbox", "kokoro", "piper", "f5tts",
    "sd15", "ace-step",
    "sklearn", "lightgbm", "xgboost", "catboost",
)

#: Order-name (as the pipeline spells it) -> backend id.
_ORDER_TO_BACKEND = {
    "parakeet-onnx": "parakeet",
    "whisper-cpp": "whisper-cpp",
    "faster-whisper": "faster-whisper",
    "chatterbox": "chatterbox",
    "kokoro": "kokoro",
    "piper": "piper",
    "xtts": "xtts",
    "f5tts": "f5tts",
}

#: family, label, download MB, disk floor GB, RAM floor GB.
_BACKEND_INFO: dict[str, dict[str, Any]] = {
    "xtts": {"family": "characters",
             "label": "XTTS v2 voice cloning (Coqui)",
             "size_mb": 1800, "disk_gb": 3.0, "ram_gb": 4.0},
    "rvc": {"family": "characters",
            "label": "RVC voice swap engine",
            "size_mb": 50, "disk_gb": 1.0, "ram_gb": 2.0},
    "faceswap": {"family": "characters",
                 "label": "face swap (InsightFace inswapper + GFPGAN)",
                 "size_mb": 880, "disk_gb": 3.0, "ram_gb": 2.0},
    "madmom": {"family": "dj",
               "label": "madmom beat tracker (DJ desk)",
               "size_mb": 60, "disk_gb": 1.0, "ram_gb": 2.0},
    "faster-whisper": {"family": "voice-stt",
                       "label": "faster-whisper STT (large-v3-turbo)",
                       "size_mb": 850, "disk_gb": 2.0, "ram_gb": 2.0},
    "parakeet": {"family": "voice-stt",
                 "label": "Parakeet TDT 0.6B v3 STT (onnx-asr)",
                 "size_mb": 2400, "disk_gb": 4.0, "ram_gb": 2.0},
    "whisper-cpp": {"family": "voice-stt",
                    "label": "whisper.cpp STT (base GGUF)",
                    "size_mb": 150, "disk_gb": 1.0, "ram_gb": 2.0},
    "chatterbox": {"family": "voice-tts",
                   "label": "Chatterbox TTS",
                   "size_mb": 1000, "disk_gb": 2.0, "ram_gb": 2.0},
    "kokoro": {"family": "voice-tts",
               "label": "Kokoro TTS",
               "size_mb": 300, "disk_gb": 1.0, "ram_gb": 2.0},
    "piper": {"family": "voice-tts",
              "label": "Piper TTS + en_US-lessac-medium voice",
              "size_mb": 100, "disk_gb": 1.0, "ram_gb": 1.0},
    "f5tts": {"family": "voice-tts",
              "label": "F5-TTS voice cloning",
              "size_mb": 1000, "disk_gb": 2.0, "ram_gb": 2.0},
    "sd15": {"family": "image",
             "label": "Stable Diffusion 1.5 weights",
             "size_mb": 4300, "disk_gb": 6.0, "ram_gb": 2.0},
    "ace-step": {"family": "image",
                 "label": "ACE-Step 1.5 music engine + weights",
                 "size_mb": 15000, "disk_gb": 16.0, "ram_gb": 4.0},
    "sklearn": {"family": "small-ml",
                "label": "scikit-learn (small-ML default)",
                "size_mb": 30, "disk_gb": 1.0, "ram_gb": 1.0},
    "lightgbm": {"family": "small-ml",
                 "label": "LightGBM (small-ML gradient boosting)",
                 "size_mb": 10, "disk_gb": 1.0, "ram_gb": 1.0},
    "xgboost": {"family": "small-ml",
               "label": "XGBoost (small-ML gradient boosting)",
               "size_mb": 100, "disk_gb": 1.0, "ram_gb": 1.0},
    "catboost": {"family": "small-ml",
                 "label": "CatBoost (small-ML gradient boosting)",
                 "size_mb": 100, "disk_gb": 1.0, "ram_gb": 1.0},
}

#: pip package + import module + warm-load snippet per engine backend.
#: The warm load is the install: it downloads the weights the backend
#: needs, so a provisioned backend works on first use.
_ENGINE_INSTALL: dict[str, dict[str, str]] = {
    "faster-whisper": {
        "pip": "faster-whisper", "module": "faster_whisper",
        "warm": ("from faster_whisper import WhisperModel; "
                 "WhisperModel('large-v3-turbo', device='cpu', "
                 "compute_type='int8')")},
    "parakeet": {
        "pip": "onnx-asr[cpu,hub]", "module": "onnx_asr",
        "warm": ("import onnx_asr; "
                 "onnx_asr.load_model('nemo-parakeet-tdt-0.6b-v3')")},
    "whisper-cpp": {
        "pip": "pywhispercpp", "module": "pywhispercpp",
        "warm": ("from pywhispercpp.model import Model; Model('base')")},
    "chatterbox": {
        "pip": "chatterbox-tts", "module": "chatterbox",
        "warm": ("from chatterbox.tts_turbo import ChatterboxTurboTTS; "
                 "ChatterboxTurboTTS.from_pretrained(device='cpu')")},
    "kokoro": {
        "pip": "kokoro", "module": "kokoro",
        "warm": ("from kokoro import KPipeline; "
                 "KPipeline(lang_code='a')")},
    "f5tts": {
        "pip": "f5-tts", "module": "f5_tts",
        "warm": ("from f5_tts.api import F5TTS; "
                 "F5TTS(model='F5TTS_v1_Base')")},
    "sklearn": {"pip": "scikit-learn", "module": "sklearn", "warm": ""},
    "lightgbm": {"pip": "lightgbm", "module": "lightgbm", "warm": ""},
    "xgboost": {"pip": "xgboost", "module": "xgboost", "warm": ""},
    "catboost": {"pip": "catboost", "module": "catboost", "warm": ""},
}

#: SD1.5 weights on Hugging Face (public, ungated).
_SD15_REPO = "runwayml/stable-diffusion-v1-5"
_SD15_FILE = "v1-5-pruned-emaonly.safetensors"

#: Default Piper voice (the backend's own default).
_PIPER_DEFAULT_VOICE = "en_US-lessac-medium"

#: ACE-Step repo clone target — the layout ace_step.py itself expects
#: (``~/ACE-Step-1.5/checkpoints`` is its repo-default weights dir).
_ACE_REPO_DIR = Path.home() / "ACE-Step-1.5"

#: Learner jobs: the ladder's data-owned jobs. ``store`` is the
#: ModelStore job name (None = no versioned model: online or
#: synthetic), ``prefix`` matches per-instance jobs in the store,
#: ``floor`` is the job's declared min_examples.
LEARNER_JOBS: tuple[dict[str, Any], ...] = (
    {"job": "character_dialogue", "store": None, "prefix": "character-line:",
     "floor": 30},
    {"job": "character_tactics", "store": None, "prefix": "tactics:",
     "floor": 40},
    {"job": "dj_sound_selection", "store": None, "prefix": None,
     "floor": 30},
    {"job": "dj_cue_suggest", "store": "dj_cue_suggest", "prefix": None,
     "floor": 30},
    {"job": "npc_line_scorer", "store": None, "prefix": "npc-line:",
     "floor": 30},
    {"job": "npc_tactics", "store": None, "prefix": "tactics:",
     "floor": 40},
    {"job": "pumpdesk_win", "store": "pumpdesk-win", "prefix": None,
     "floor": 30},
    {"job": "error_triage", "store": "error-triage", "prefix": None,
     "floor": 30},
    {"job": "morning_rank", "store": "morning-rank", "prefix": None,
     "floor": 30},
)

OUT_OF_SCOPE_NOTE = (
    "Out of scope for /models (separate programmes): codebeast, the "
    "3.8B specialist model, and the big-merge specialists (vision, "
    "coding, charts, multitask, agentic, chat/roleplay) — those install "
    "through their own programmes, not this command.")


# ── seams ────────────────────────────────────────────────────────────

def _run(args: list[str], timeout: int | None = None) -> Any:
    """Subprocess seam (same shape as characters.provision._run)."""
    return subprocess.run(args, capture_output=True, text=True,
                          timeout=timeout)


def _pip(package: str, steps: list[str]) -> bool:
    """pip install one package. True on rc 0."""
    proc = _run([sys.executable, "-m", "pip", "install", "-q", package],
                timeout=1200)
    steps.append(f"pip install {package}: rc={proc.returncode}")
    if proc.returncode != 0:
        steps.append(f"pip stderr: {(proc.stderr or '')[-300:]}")
    return proc.returncode == 0


def _verify_import(module: str, steps: list[str]) -> bool:
    """Import check in a clean interpreter. True on rc 0."""
    proc = _run([sys.executable, "-c", f"import {module}"],
                timeout=300)
    steps.append(f"import {module}: rc={proc.returncode}")
    return proc.returncode == 0


def _report(ok: bool, status: str, detail: str,
            steps: list[str] | None = None) -> dict[str, Any]:
    return {"ok": ok, "status": status, "detail": detail,
            "steps": list(steps or [])}


# ── detection ────────────────────────────────────────────────────────

def _detect_xtts() -> bool:
    try:
        return bool(_char_provision.backend_status()["xtts"]["installed"])
    except Exception:  # noqa: BLE001 - detection is best-effort
        return False


def _detect_rvc() -> bool:
    try:
        return bool(_char_provision.backend_status()["rvc"]["installed"])
    except Exception:  # noqa: BLE001 - detection is best-effort
        return False


def _detect_faceswap() -> bool:
    try:
        return bool(
            _char_provision.backend_status()["faceswap"]["installed"])
    except Exception:  # noqa: BLE001 - detection is best-effort
        return False


def _detect_madmom() -> bool:
    try:
        return bool(_dj_desk.madmom_available()[0])
    except Exception:  # noqa: BLE001 - detection is best-effort
        return False


def _detect_faster_whisper() -> bool:
    return _verify_import("faster_whisper", [])


def _detect_parakeet() -> bool:
    return _verify_import("onnx_asr", [])


def _detect_whisper_cpp() -> bool:
    return _verify_import("pywhispercpp", [])


def _detect_chatterbox() -> bool:
    return _verify_import("chatterbox", [])


def _detect_kokoro() -> bool:
    return _verify_import("kokoro", [])


def _detect_f5tts() -> bool:
    return _verify_import("f5_tts", [])


def _piper_voice_path() -> Path:
    """Where the default Piper voice lives (backend's own resolution)."""
    from ..voice.fetch import default_cache_dir

    explicit = (os.environ.get("PIPER_VOICE", "") or "").strip()
    if explicit:
        return Path(explicit)
    base = (os.environ.get("PIPER_VOICES_DIR", "") or "").strip()
    if base:
        return Path(base) / f"{_PIPER_DEFAULT_VOICE}.onnx"
    return (Path(default_cache_dir("piper"))
            / f"{_PIPER_DEFAULT_VOICE}.onnx")


def _detect_piper() -> bool:
    try:
        if not _verify_import("piper", []):
            return False
        return _piper_voice_path().is_file()
    except Exception:  # noqa: BLE001 - detection is best-effort
        return False


def _detect_sd15() -> bool:
    """A loadable SD1.5 entry in the image registry."""
    try:
        return any(e.get("model_class") == "sd15" and e.get("loadable")
                   for e in _imggen_registry.list_models())
    except Exception:  # noqa: BLE001 - detection is best-effort
        return False


def _ace_weights_dir() -> Path | None:
    """Where ACE-Step weights live, if anywhere yet."""
    env = (os.environ.get(CHECKPOINT_ENV, "") or "").strip()
    candidates = ([Path(env)] if env else [])
    candidates += [
        Path.home() / ".nomorals" / "models" / "acestep-1.5",
        _ACE_REPO_DIR / "checkpoints",
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand
    return None


def _ace_state() -> str:
    """missing | engine-only | installed (weights present)."""
    weights = _ace_weights_dir()
    if weights is not None and any(
            p.suffix in (".safetensors", ".ckpt", ".pt", ".bin")
            for p in weights.iterdir()):
        return "installed"
    if (_ACE_REPO_DIR / "pyproject.toml").is_file():
        return "engine-only"
    return "missing"


def _detect_ace_step() -> bool:
    return _ace_state() == "installed"


def _detect_sklearn() -> bool:
    try:
        return bool(backend_available("gradboost")[0])
    except Exception:  # noqa: BLE001 - detection is best-effort
        return False


def _detect_lightgbm() -> bool:
    try:
        return bool(backend_available("lightgbm")[0])
    except Exception:  # noqa: BLE001 - detection is best-effort
        return False


def _detect_xgboost() -> bool:
    try:
        return bool(backend_available("xgboost")[0])
    except Exception:  # noqa: BLE001 - detection is best-effort
        return False


def _detect_catboost() -> bool:
    try:
        return bool(backend_available("catboost")[0])
    except Exception:  # noqa: BLE001 - detection is best-effort
        return False


_DETECTORS: dict[str, Any] = {
    "xtts": _detect_xtts, "rvc": _detect_rvc,
    "faceswap": _detect_faceswap, "madmom": _detect_madmom,
    "faster-whisper": _detect_faster_whisper,
    "parakeet": _detect_parakeet, "whisper-cpp": _detect_whisper_cpp,
    "chatterbox": _detect_chatterbox, "kokoro": _detect_kokoro,
    "piper": _detect_piper, "f5tts": _detect_f5tts,
    "sd15": _detect_sd15, "ace-step": _detect_ace_step,
    "sklearn": _detect_sklearn, "lightgbm": _detect_lightgbm,
    "xgboost": _detect_xgboost, "catboost": _detect_catboost,
}


def _detector(name: str) -> Any:
    """The current ``_detect_*`` for a backend.

    Resolved off the module globals at call time (not the _DETECTORS
    snapshot above) so a patched detector takes effect — the tests
    patch ``models_provision._detect_<name>`` per test.
    """
    fn_name = "_detect_" + name.replace("-", "_")
    return globals().get(fn_name) or _DETECTORS[name]


# ── inventory ────────────────────────────────────────────────────────

def _profile_order(profile: str) -> dict[str, list[str]]:
    """Voice backends in this profile's preference order, mapped to ids."""
    out: dict[str, list[str]] = {"voice-stt": [], "voice-tts": []}
    for order, key in ((_STT_ORDER, "voice-stt"),
                       (_TTS_ORDER, "voice-tts")):
        for name in order.get(profile, ()):
            backend = _ORDER_TO_BACKEND.get(name)
            if backend and backend not in out[key]:
                out[key].append(backend)
    return out


def inventory(profile: str | None = None) -> list[dict[str, Any]]:
    """Every installable backend: installed?, in this profile's order?

    Voice rows follow the profile's preference order; every other
    family follows BACKENDS order. Backends outside the profile's
    order are still listed (``in_profile`` False) — the inventory is
    complete, the install is profile-gated.
    """
    box = _measure_box()
    prof = (profile or box.get("profile") or detect_profile() or "pc")
    orders = _profile_order(prof)
    rows: list[dict[str, Any]] = []

    def _row(name: str) -> dict[str, Any]:
        info = _BACKEND_INFO[name]
        fam = info["family"]
        in_profile = True
        if fam in ("voice-stt", "voice-tts"):
            in_profile = name in orders[fam]
        try:
            installed = bool(_detector(name)())
        except Exception:  # noqa: BLE001 - a detector never kills inventory
            installed = False
        detail = ""
        if name == "ace-step":
            state = _ace_state()
            if state == "engine-only":
                detail = ("engine installed, weights download on first "
                          "generation")
        return {"id": name, "family": fam, "label": info["label"],
                "installed": installed, "in_profile": in_profile,
                "size_mb": info["size_mb"], "disk_gb": info["disk_gb"],
                "ram_gb": info["ram_gb"], "detail": detail}

    # Families in BACKENDS first-appearance order; voice rows follow the
    # profile's preference order inside their family (xtts is listed
    # once, under characters — the TTS order's xtts is the same backend).
    families: list[str] = []
    for name in BACKENDS:
        fam = _BACKEND_INFO[name]["family"]
        if fam not in families:
            families.append(fam)
    for fam in families:
        if fam in ("voice-stt", "voice-tts"):
            for name in orders[fam]:
                if name == "xtts":
                    continue
                rows.append(_row(name))
            for name in BACKENDS:
                if (_BACKEND_INFO[name]["family"] == fam
                        and name not in orders[fam] and name != "xtts"):
                    rows.append(_row(name))
        else:
            for name in BACKENDS:
                if _BACKEND_INFO[name]["family"] == fam:
                    rows.append(_row(name))
    return rows


def backend_status() -> dict[str, dict[str, Any]]:
    """Installed? per backend. Never raises."""
    out: dict[str, dict[str, Any]] = {}
    for row in inventory():
        out[row["id"]] = {"installed": row["installed"],
                          "detail": row["detail"]
                          or ("installed" if row["installed"]
                              else "not installed")}
    return out


# ── provisioners ─────────────────────────────────────────────────────

def _provision_engine(name: str, box: dict[str, Any]) -> dict[str, Any]:
    """pip install + warm load (downloads weights) + import verify."""
    spec = _ENGINE_INSTALL[name]
    steps: list[str] = []
    if not _pip(spec["pip"], steps):
        return _report(False, "failed",
                       f"pip install {spec['pip']} failed", steps)
    warm = spec.get("warm") or ""
    if warm:
        proc = _run([sys.executable, "-c", warm], timeout=1200)
        steps.append(f"warm load: rc={proc.returncode}")
        if proc.returncode != 0:
            steps.append(f"warm stderr: {(proc.stderr or '')[-300:]}")
            return _report(False, "failed",
                           f"{name} weights did not download", steps)
    if not _verify_import(spec["module"], steps):
        return _report(False, "failed",
                       f"{spec['module']} still not importable", steps)
    return _report(True, "installed", f"{name} installed", steps)


def _provision_piper(box: dict[str, Any]) -> dict[str, Any]:
    """Piper engine + the default voice. Termux uses the prebuilt binary."""
    steps: list[str] = []
    profile = str(box.get("profile") or "")
    if profile != "termux" and not _pip("piper-tts", steps):
        return _report(False, "failed", "pip install piper-tts failed",
                       steps)
    if profile == "termux":
        steps.append("termux: prebuilt piper binary, no pip install")
    if not _verify_import("huggingface_hub", steps) and not _pip(
            "huggingface-hub", steps):
            return _report(False, "failed",
                           "huggingface_hub is needed to fetch the voice",
                           steps)
    fetch = ("from nomorals.voice.fetch import fetch_piper_voice; "
             f"fetch_piper_voice({_PIPER_DEFAULT_VOICE!r})")
    proc = _run([sys.executable, "-c", fetch], timeout=900)
    steps.append(f"fetch voice {_PIPER_DEFAULT_VOICE}: rc={proc.returncode}")
    if proc.returncode != 0:
        steps.append(f"fetch stderr: {(proc.stderr or '')[-300:]}")
        return _report(False, "failed",
                       f"voice {_PIPER_DEFAULT_VOICE} did not download",
                       steps)
    if not _verify_import("piper", steps):
        return _report(False, "failed", "piper still not importable",
                       steps)
    voice = _piper_voice_path()
    check = _run([sys.executable, "-c",
                  f"import os; raise SystemExit(0 if os.path.isfile({str(voice)!r}) else 1)"],
                 timeout=60)
    if check.returncode != 0:
        return _report(False, "failed",
                       f"voice file missing at {voice}", steps)
    return _report(True, "installed",
                   f"piper installed, voice {_PIPER_DEFAULT_VOICE} ready",
                   steps)


def _provision_sd15(box: dict[str, Any]) -> dict[str, Any]:
    """Stable Diffusion 1.5 weights from Hugging Face into the registry."""
    steps: list[str] = []
    if not _verify_import("huggingface_hub", steps) and not _pip(
            "huggingface-hub", steps):
            return _report(False, "failed",
                           "huggingface_hub is needed for the download",
                           steps)
    dest = _imggen_checkpoint_dir()
    snippet = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id={_SD15_REPO!r}, "
        f"allow_patterns=[{_SD15_FILE!r}], local_dir={dest!r})")
    proc = _run([sys.executable, "-c", snippet], timeout=1800)
    steps.append(f"download {_SD15_FILE}: rc={proc.returncode}")
    if proc.returncode != 0:
        steps.append(f"download stderr: {(proc.stderr or '')[-300:]}")
        return _report(False, "failed",
                       f"{_SD15_FILE} did not download", steps)
    verify = ("from nomorals.media.imggen.registry import list_models; "
              "raise SystemExit(0 if any(e.get('model_class') == 'sd15' "
              "and e.get('loadable') for e in list_models()) else 1)")
    if _run([sys.executable, "-c", verify], timeout=300).returncode != 0:
        return _report(False, "failed",
                       "weights downloaded but the registry cannot load "
                       "them", steps)
    return _report(True, "installed",
                   f"sd15 weights installed in {dest}", steps)


def _provision_ace_step(box: dict[str, Any]) -> dict[str, Any]:
    """ACE-Step 1.5: clone + uv sync. Weights download on first run."""
    steps: list[str] = []
    import shutil

    if not shutil.which("git"):
        return _report(False, "unavailable",
                       "git is not on this box — ACE-Step clones from "
                       "github.com/ace-step/ACE-Step-1.5", steps)
    repo = _ACE_REPO_DIR
    if (repo / "pyproject.toml").is_file():
        steps.append("repo already present, skipping clone")
    else:
        proc = _run(["git", "clone", "--depth", "1",
                     "https://github.com/ace-step/ACE-Step-1.5",
                     str(repo)], timeout=900)
        steps.append(f"git clone: rc={proc.returncode}")
        if proc.returncode != 0:
            steps.append(f"clone stderr: {(proc.stderr or '')[-300:]}")
            return _report(False, "failed", "ACE-Step clone failed", steps)
    uv = shutil.which("uv")
    if not uv:
        if not _pip("uv", steps):
            return _report(False, "failed",
                           "uv is needed to sync ACE-Step", steps)
        uv = "uv"
    proc = _run([str(uv), "sync"], timeout=1800)
    steps.append(f"uv sync: rc={proc.returncode}")
    if proc.returncode != 0:
        steps.append(f"sync stderr: {(proc.stderr or '')[-300:]}")
        return _report(False, "failed", "ACE-Step uv sync failed", steps)
    gpu = _run([sys.executable, "-c",
                "import torch; print(torch.cuda.is_available())"],
               timeout=120)
    gpu_note = ""
    if gpu.returncode != 0 or "True" not in (gpu.stdout or ""):
        gpu_note = " — generation needs an NVIDIA GPU, none detected here"
    return _report(True, "installed",
                   "ACE-Step engine installed; weights (~5-15 GB) "
                   f"download on first generation{gpu_note}", steps)


# ── the command ──────────────────────────────────────────────────────

def _floor_refusal(name: str, box: dict[str, Any],
                   disk_left: float) -> dict[str, Any]:
    info = _BACKEND_INFO[name]
    return _report(False, "unavailable",
                   f"needs {info['disk_gb']} GB free disk and "
                   f"{info['ram_gb']} GB RAM; box has {disk_left} GB disk "
                   f"free and {box.get('total_ram_gb', '?')} GB RAM total")


def provision_models(backends: Any = None, *,
                     box: dict[str, Any] | None = None) -> dict[str, Any]:
    """Provision the named backends (default: every missing one).

    Returns ``{"ok", "box", "backends", "learners"}`` in the character
    provision's report shape, plus the learners' status. Never raises
    — a backend that blows up reports ``failed``.
    """
    wanted = list(BACKENDS if backends is None else backends)
    box = dict(box) if box is not None else _measure_box()
    profile = str(box.get("profile") or detect_profile() or "pc")
    orders = _profile_order(profile)
    in_profile = {
        name: (True if _BACKEND_INFO[name]["family"]
               not in ("voice-stt", "voice-tts")
               else name in orders[_BACKEND_INFO[name]["family"]])
        for name in BACKENDS}
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
    char_to_provision: list[str] = []
    for name in wanted:
        if name in unknown:
            continue
        info = _BACKEND_INFO[name]
        if not in_profile[name]:
            reports[name] = _report(
                True, "skipped",
                f"not in the {profile} profile's order — skipped")
            continue
        try:
            installed = bool(_detector(name)())
        except Exception:  # noqa: BLE001 - a detector never kills the run
            installed = False
        if installed:
            reports[name] = _report(True, "already-installed",
                                    f"{name} already installed")
            continue
        if total_ram < info["ram_gb"] or disk_left < info["disk_gb"]:
            reports[name] = _floor_refusal(name, box, disk_left)
            continue
        if name in ("xtts", "rvc", "faceswap"):
            # Floor-checked above; the delegate installs the batch.
            char_to_provision.append(name)
            continue
        try:
            if name == "madmom":
                raw = _madmom_provision()
                if raw.get("ok"):
                    reports[name] = _report(True, "installed",
                                            "madmom installed",
                                            raw.get("steps") or [])
                else:
                    reports[name] = _report(
                        False, "failed",
                        str(raw.get("error") or "madmom provision failed"),
                        raw.get("steps") or [])
            elif name == "piper":
                reports[name] = _provision_piper(box)
            elif name == "sd15":
                reports[name] = _provision_sd15(box)
            elif name == "ace-step":
                reports[name] = _provision_ace_step(box)
            elif name in _ENGINE_INSTALL:
                reports[name] = _provision_engine(name, box)
            else:
                reports[name] = _report(False, "failed",
                                        f"no provisioner for {name}")
        except Exception as exc:  # noqa: BLE001 - a backend never kills
            _log.debug("models provision %s failed", name, exc_info=True)
            reports[name] = _report(False, "failed",
                                    f"provision crashed: {exc}")
        if reports[name].get("ok"):
            disk_left = max(0.0, disk_left
                            - info["size_mb"] / 1024.0)
    # Characters go through their own provisioner, once, for the batch
    # that passed this command's floor check (the delegate re-checks
    # its own floors too — the two agree by construction).
    if char_to_provision:
        try:
            char_result = _char_provision_backends(char_to_provision)
            for b in char_to_provision:
                rep = (char_result.get("backends") or {}).get(b)
                if rep is None:
                    continue
                reports[b] = _report(
                    bool(rep.get("ok")), str(rep.get("status") or "failed"),
                    str(rep.get("detail") or ""),
                    list(rep.get("steps") or []))
        except Exception as exc:  # noqa: BLE001 - a delegate never kills
            _log.debug("models character provision failed", exc,
                       exc_info=True)
            for b in char_to_provision:
                if b not in reports:
                    reports[b] = _report(False, "failed",
                                         f"character provision crashed: "
                                         f"{exc}")
    ok = bool(reports) and all(r["ok"] for r in reports.values()) \
        and not unknown
    result: dict[str, Any] = {"ok": ok, "box": box, "backends": reports,
                              "learners": learner_status()}
    if unknown:
        result["error"] = "unknown backend(s): " + ", ".join(unknown)
    return result


def models_report_text(result: dict[str, Any]) -> str:
    """Plain-text per-backend report for the chat reply."""
    box = result.get("box") or {}
    lines = ["📦 models — every missing model"]
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
    learners = result.get("learners") or []
    if learners:
        lines.append("")
        lines.append("learners — status only, they train from journals:")
        for row in learners:
            lines.append(f"• {row['job']}: {row['status']} — "
                         f"{row['detail']}")
    if result.get("error"):
        lines.append(str(result["error"]))
    lines.append("")
    lines.append(OUT_OF_SCOPE_NOTE)
    lines.append("XTTS speaks only for the owner's own characters "
                 "(private use only). Face swap runs outside the game "
                 "only — NPCs get no faces.")
    return "\n".join(lines)


def status_text() -> str:
    """Installed vs missing per model, plus the learners' status."""
    rows = inventory()
    lines = ["📦 models — status"]
    for row in rows:
        state = "installed" if row["installed"] else "missing"
        extra = f" — {row['detail']}" if row["detail"] else ""
        lines.append(f"• {row['id']}: {state}{extra}")
    lines.append("")
    lines.append("learners — status only, they train from journals:")
    for row in learner_status():
        lines.append(f"• {row['job']}: {row['status']} — {row['detail']}")
    lines.append("")
    lines.append(OUT_OF_SCOPE_NOTE)
    return "\n".join(lines)


# ── journal-trained learners: status only ────────────────────────────

def _failure_grades_path() -> Path:
    """The error-triage grade journal — the learner's own path.

    Mirrors agents/failure_models.DEFAULT_GRADES_PATH (pinned by test).
    """
    override = (os.environ.get("NM_FAILURE_GRADES") or "").strip()
    if override:
        return Path(os.path.expanduser(override))
    return Path.home() / ".nomorals" / "failure_grades.jsonl"


def _npc_learn_dir() -> Path:
    """The NPC line journal dir — the learner's own path.

    Mirrors games/npc_learn._default_npc_dir (pinned by test).
    """
    override = (os.environ.get("NM_NPC_LEARN_DIR") or "").strip()
    if override:
        return Path(os.path.expanduser(override))
    return Path.home() / ".nomorals" / "npc_learn"


def _tactics_dir() -> Path:
    """The combat-policy journal dir — the learner's own path.

    Mirrors games/npc_learn._default_tactics_dir.
    """
    override = (os.environ.get("NM_TACTICS_DIR") or "").strip()
    if override:
        return Path(os.path.expanduser(override))
    return Path.home() / ".nomorals" / "tactics"


def _pumpdesk_dir() -> Path:
    """The PumpDesk desk dir — the learner's own path.

    Mirrors agents/pumpdesk/store.pumpdesk_dir (pinned by test).
    """
    override = (os.environ.get("NOMORALS_PUMPDESK_DIR", "") or "").strip()
    if override:
        return Path(override)
    return Path.home() / ".config" / "nomorals" / "pumpdesk"


def _character_learn_dir() -> Path | None:
    """Per-character learn docs — characters/learn.LearnStore's dir."""
    try:
        return Path(default_character_dir()) / "learn"
    except Exception:  # noqa: BLE001 - status is best-effort
        return None


def _count_jsonl(path: Path) -> int | None:
    """Non-empty lines in a JSONL journal. None when it cannot be read."""
    try:
        if not path.is_file():
            return 0
        count = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    count += 1
        return count
    except Exception:  # noqa: BLE001 - status is best-effort
        return None


def _count_json_docs(directory: Path | None, key: str,
                     graded_key: str | None = None) -> int | None:
    """Entries across a dir of JSON journal docs.

    ``graded_key`` counts only entries carrying that key (the fit's
    input); without it, every entry counts. Missing dir = 0.
    """
    if directory is None:
        return None
    try:
        if not directory.is_dir():
            return 0
        total = 0
        for path in sorted(directory.glob("*.json")):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - a corrupt doc is skipped
                continue
            if not isinstance(doc, dict):
                continue
            entries = doc.get(key) or []
            if graded_key is None:
                total += len(entries)
            else:
                total += sum(1 for e in entries
                             if isinstance(e, dict)
                             and e.get(graded_key) is not None)
        return total
    except Exception:  # noqa: BLE001 - status is best-effort
        return None


def _morning_rank_rows() -> int | None:
    """Per-briefing section history rows (briefing_engagement table).

    Read-only sqlite count against the app's database file; the file
    must already exist (never created by a status read) and the table
    must exist. None = unknown.
    """
    try:
        from ..core.config import load_settings

        db_path = load_settings().db_path
    except Exception:  # noqa: BLE001 - status is best-effort
        return None
    try:
        if not Path(db_path).is_file():
            return None
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=10)
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM briefing_engagement").fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 - status is best-effort
        return None


def _dj_taste_rows() -> int | None:
    """DJ taste feedback rows (compose rows are not taste judgments)."""
    try:
        rows = read_feedback_log()
        return sum(1 for r in rows if r.get("source") != "compose")
    except Exception:  # noqa: BLE001 - status is best-effort
        return None


def _store_manifest(job: str) -> dict[str, Any]:
    """The versioned store's manifest for one job ({} when untrained)."""
    try:
        path = Path(default_models_dir()) / job / "manifest.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:  # noqa: BLE001 - no manifest = not trained
        return {}


def _trained_instances(prefix: str) -> list[dict[str, Any]]:
    """Trained per-instance models (character-line:/npc-line:/tactics:)."""
    out: list[dict[str, Any]] = []
    try:
        base = Path(default_models_dir())
        if not base.is_dir():
            return out
        for job_dir in sorted(base.iterdir()):
            if not job_dir.is_dir():
                continue
            if not job_dir.name.startswith(prefix.rstrip(":")):
                continue
            manifest = _store_manifest(job_dir.name)
            if int(manifest.get("latest") or 0) > 0:
                out.append({"job": job_dir.name,
                            "version": int(manifest["latest"]),
                            "n_rows": int(manifest.get("n_rows") or 0)})
    except Exception:  # noqa: BLE001 - status is best-effort
        return out
    return out


def _learner_rows(job: str) -> int | None:
    """Journal rows held for one learner job. None = unknown."""
    if job == "error_triage":
        return _count_jsonl(_failure_grades_path())
    if job == "pumpdesk_win":
        return _count_jsonl(_pumpdesk_dir() / "pump_calls.jsonl")
    if job == "morning_rank":
        return _morning_rank_rows()
    if job == "dj_sound_selection":
        return _dj_taste_rows()
    if job == "npc_line_scorer":
        return _count_json_docs(_npc_learn_dir(), "entries",
                                graded_key="grade")
    if job in ("npc_tactics", "character_tactics"):
        return _count_json_docs(_tactics_dir(), "rounds")
    if job == "character_dialogue":
        return _count_json_docs(_character_learn_dir(), "entries")
    if job == "dj_cue_suggest":
        return None  # synthetic rows, fitted on demand
    return None


def learner_status(store: Any = None) -> list[dict[str, Any]]:
    """Status per journal-trained learner. Never raises.

    Each row: job, floor, rows held (None = unknown), trained?,
    version/score when a model exists, status, detail. No learner is
    ever "installed" — they train from data, not downloads.
    """
    rows: list[dict[str, Any]] = []
    for spec in LEARNER_JOBS:
        job = spec["job"]
        floor = int(spec["floor"])
        held = _learner_rows(job)
        trained = False
        version = 0
        score = None
        instances: list[dict[str, Any]] = []
        if spec.get("store"):
            manifest = _store_manifest(str(spec["store"]))
            version = int(manifest.get("latest") or 0)
            trained = version > 0
            if trained:
                report = manifest.get("report") or {}
                score = report.get("score")
        if spec.get("prefix"):
            instances = _trained_instances(str(spec["prefix"]))
            if instances and not trained:
                trained = True
                version = max(i["version"] for i in instances)
        if trained:
            if instances:
                detail = (f"trained — {len(instances)} instance(s), "
                          f"latest v{version}")
            else:
                detail = f"trained — v{version}"
                if score is not None:
                    detail += f", held-out {score}"
            if held is not None:
                detail += f", {held} journal rows held"
            status = "trained"
        elif held is None:
            if job == "dj_cue_suggest":
                status = "not trained yet"
                detail = (f"synthetic rows — fitted on demand, "
                          f"floor {floor}")
            else:
                status = "unknown"
                detail = (f"not trained yet — journal rows unknown, "
                          f"floor {floor}")
        else:
            status = "not trained yet"
            detail = f"{held}/{floor} journal rows held"
        rows.append({"job": job, "floor": floor, "rows": held,
                     "trained": trained, "version": version,
                     "score": score, "instances": instances,
                     "status": status, "detail": detail})
    return rows
