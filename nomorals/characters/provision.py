"""Self-provision the character generative backends (family (b)).

The owner asked (2026-11-13): "Weren't you supposed to give me a
certain command to install the models?" Until this module, the answer
was no — the character specialists in ``specialists.py`` were slots
only, honestly unavailable, with no install path anywhere in the repo.
This module is that path, in the same shape as ``provision_madmom()``
(``media/dj_desk.py``) and its ``/dj desk provision`` verb:

* explicit only — importing this module installs nothing;
* one report dict per backend, plus an overall report — never raises;
* slow by nature (torch + multi-hundred-MB weights) — the caller says so.

Backends, each independently reported:

* ``xtts`` — XTTS v2 voice cloning: ``pip install TTS`` (Coqui),
  then a warm load of ``tts_models/multilingual/multi-dataset/xtts_v2``
  (~1.8 GB of weights, downloaded on first use). XTTS v2 is
  NON-COMMERCIAL — private (owner's own) use only
  (``audio/pipeline.py`` ``_NONCOMMERCIAL``; the audience rule removes
  it for public audiences). The provision installs it for the owner's
  own characters on his own box, and says so in the report.
* ``rvc`` — RVC voice swap: ``pip install rvc-python`` (the runtime
  ``voice/rvc_bridge.py`` detects). Character voice MODELS (.pth) are
  NOT downloadable — each character's model is trained/supplied by the
  owner under ``~/.nomorals/rvc/models``; the provision installs the
  runtime and lists the models already there.
* ``faceswap`` — guarded face swap: ``pip install insightface
  onnxruntime gfpgan`` plus the weights the swap warms itself
  (inswapper_128.onnx ~530 MB, GFPGANv1.4.pth ~350 MB; the buffalo_l
  analysis pack downloads on first use). Termux is refused by the
  same profile gate ``media_edit/faceswap.py`` enforces, and face swap
  stays outside the game — NPCs get no faces.

Installs go into Devon's own interpreter (``sys.executable -m pip``),
because that is what the existing detection expects: ``voice/tts.py``
probes ``find_spec("TTS")`` in-process, ``rvc_bridge.detect_rvc()``
probes an importable ``rvc_python``, and ``faceswap`` imports
insightface in-process. A venv install would stay invisible to all
three. Once provisioned, ``character_voice`` / ``swap_voice`` /
``swap_face`` light up with no further code change.

Honest box check: the provision MEASURES the box (RAM, free disk,
CPUs, profile) and refuses a backend whose floor the box cannot meet,
naming the measured numbers — never a fake install. The verify step
(an actual import / warm load in a subprocess) is the real proof; a
box that passes the floors but fails the load reports ``failed``.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

__all__ = ["BACKENDS", "preflight", "backend_status", "provision_backends",
           "provision_report_text"]

_log = get_logger(__name__)

BACKENDS = ("xtts", "rvc", "faceswap")

#: Floors per backend (GB). Floors, not measurements — the verify step
#: is the real proof. XTTS v2's weights alone are ~1.8 GB on disk and
#: load into RAM with torch on top, so a 2 GB box cannot run it
#: honestly; the lighter backends fit smaller boxes.
_FLOORS: dict[str, dict[str, float]] = {
    "xtts": {"total_ram_gb": 4.0, "free_ram_gb": 2.0, "free_disk_gb": 8.0},
    "rvc": {"total_ram_gb": 2.0, "free_ram_gb": 1.0, "free_disk_gb": 3.0},
    "faceswap": {"total_ram_gb": 2.0, "free_ram_gb": 1.0,
                 "free_disk_gb": 3.0},
}

_XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
_XTTS_WEIGHTS_NOTE = "XTTS v2 weights ~1.8 GB"


# ── seams (tests patch these; production never does) ───────────────────────

def _run(args: list[str], timeout: int | None = None) -> Any:
    """Run a subprocess, capturing output. One seam for the tests."""
    return subprocess.run(args, capture_output=True, text=True,
                          timeout=timeout)


def _measure_box() -> dict[str, Any]:
    """Measure this box: RAM, free disk, CPUs, platform, profile.

    Never raises — unmeasurable fields come back 0.0 with ok False.
    """
    total_gb = 0.0
    free_gb = 0.0
    try:
        import psutil  # type: ignore[import-not-found]

        mem = psutil.virtual_memory()
        total_gb = float(mem.total) / 1e9
        free_gb = float(mem.available) / 1e9
    except Exception:  # noqa: BLE001 - fall through to the stdlib probes
        total_gb, free_gb = _measure_ram_stdlib()
    try:
        free_disk_gb = shutil.disk_usage(Path.home()).free / 1e9
    except Exception:  # noqa: BLE001 - a missing home is its own wall
        free_disk_gb = 0.0
    try:
        from ..core.profiles import get_profile_kind

        profile = get_profile_kind()
    except Exception:  # noqa: BLE001 - profile is advisory here
        profile = ""
    return {"ok": total_gb > 0.0,
            "total_ram_gb": round(total_gb, 2),
            "free_ram_gb": round(free_gb, 2),
            "free_disk_gb": round(free_disk_gb, 2),
            "cpus": os.cpu_count() or 0,
            "platform": sys.platform,
            "profile": profile,
            "detail": "box measured" if total_gb > 0.0
                      else "RAM could not be measured on this box"}


def _measure_ram_stdlib() -> tuple[float, float]:
    """RAM in GB without psutil: Windows ctypes, else /proc/meminfo."""
    try:
        if sys.platform == "win32":
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            st = _MemStatus()
            st.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return (float(st.ullTotalPhys) / 1e9,
                        float(st.ullAvailPhys) / 1e9)
            return 0.0, 0.0
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        fields: dict[str, float] = {}
        for line in meminfo.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].endswith(":"):
                try:
                    fields[parts[0][:-1]] = float(parts[1]) / 1e6
                except ValueError:
                    continue
        return fields.get("MemTotal", 0.0), fields.get("MemAvailable", 0.0)
    except Exception:  # noqa: BLE001 - unmeasurable is 0.0, not a crash
        return 0.0, 0.0


# ── detection (the same probes the specialists use) ───────────────────────

def _detect_xtts() -> bool:
    return importlib.util.find_spec("TTS") is not None


def _detect_rvc() -> bool:
    try:
        from ..voice import rvc_bridge

        return bool(rvc_bridge.detect_rvc().get("ok"))
    except Exception:  # noqa: BLE001 - a broken probe is "not installed"
        return False


def _detect_faceswap() -> bool:
    pkgs = all(importlib.util.find_spec(name) is not None
               for name in ("insightface", "onnxruntime", "gfpgan"))
    if not pkgs:
        return False
    try:
        from ..media_edit import _model_cache as mc

        cache = mc.cache_dir()
        return ((cache / "inswapper_128.onnx").is_file()
                and (cache / "GFPGANv1.4.pth").is_file())
    except Exception:  # noqa: BLE001 - a broken cache probe is "not ready"
        return False


_DETECTORS = {"xtts": _detect_xtts, "rvc": _detect_rvc,
              "faceswap": _detect_faceswap}


def _gate(name: str, box: dict[str, Any]) -> str:
    """The box-check refusal for ``name``, or "" when the box passes.

    Names the measured numbers — an honest refusal, not a bare no.
    """
    if name == "faceswap" and (box.get("profile") or "") == "termux":
        return ("face swap needs a laptop/workstation (insightface + "
                "ONNX want real RAM/CPU); termux can't run it honestly")
    floors = _FLOORS[name]
    missing = []
    if box.get("total_ram_gb", 0.0) < floors["total_ram_gb"]:
        missing.append(
            f"total RAM {box.get('total_ram_gb', 0.0)} GB "
            f"(needs {floors['total_ram_gb']} GB)")
    if box.get("free_ram_gb", 0.0) < floors["free_ram_gb"]:
        missing.append(
            f"free RAM {box.get('free_ram_gb', 0.0)} GB "
            f"(needs {floors['free_ram_gb']} GB)")
    if box.get("free_disk_gb", 0.0) < floors["free_disk_gb"]:
        missing.append(
            f"free disk {box.get('free_disk_gb', 0.0)} GB "
            f"(needs {floors['free_disk_gb']} GB)")
    if not missing:
        return ""
    return ("this box cannot run " + name + ": " + "; ".join(missing)
            + " — measured now; a bigger box unblocks it")


def _pip(packages: list[str], label: str, steps: list[str]) -> tuple[bool, str]:
    """pip-install into Devon's own interpreter. Returns (ok, tail).

    The failure tail rides in the steps AND the return value — a failed
    install must name its reason, never just its rc.
    """
    try:
        proc = _run([sys.executable, "-m", "pip", "install", "-q",
                     *packages], timeout=1800)
    except Exception as exc:  # noqa: BLE001 - a failed launch is a step
        steps.append(f"{label}: launch failed: {exc}")
        return False, str(exc)
    steps.append(f"{label}: rc={proc.returncode}")
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-300:]
        steps.append(f"{label} output: {tail}")
        return False, tail
    return True, ""


def _verify(args: list[str], label: str, steps: list[str],
            timeout: int) -> tuple[bool, str]:
    """Run the warm-load verify in a subprocess. (ok, tail)."""
    try:
        proc = _run([sys.executable, "-c", *args], timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - a failed launch is a step
        steps.append(f"{label}: launch failed: {exc}")
        return False, str(exc)
    steps.append(f"{label}: rc={proc.returncode}")
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-300:]
        return False, tail
    return True, (proc.stdout or "").strip()


# ── per-backend provision ────────────────────────────────────────────────

def _provision_xtts(box: dict[str, Any]) -> dict[str, Any]:
    steps: list[str] = []
    if _detect_xtts():
        return {"ok": True, "status": "already-installed",
                "detail": "XTTS (Coqui TTS) is already installed — "
                          "private use only: XTTS v2 is non-commercial, "
                          "so this backend speaks only for the owner's "
                          "own characters on his own box",
                "steps": steps}
    refusal = _gate("xtts", box)
    if refusal:
        return {"ok": False, "status": "unavailable", "detail": refusal,
                "steps": steps}
    ok, tail = _pip(["TTS"], "pip install TTS", steps)
    if not ok:
        return {"ok": False, "status": "failed",
                "detail": "pip install TTS failed: " + (tail or "see steps"),
                "steps": steps}
    try:
        import importlib

        importlib.invalidate_caches()
    except Exception:  # noqa: BLE001 - cache refresh is best-effort
        pass
    ok, tail = _verify(
        ["from TTS.api import TTS; "
         "TTS('" + _XTTS_MODEL + "', gpu=False); print('xtts-ok')"],
        "warm load XTTS v2 (" + _XTTS_WEIGHTS_NOTE + ")", steps, 1800)
    if not ok:
        return {"ok": False, "status": "failed",
                "detail": "XTTS warm load failed: " + tail, "steps": steps}
    return {"ok": True, "status": "installed",
            "detail": "XTTS v2 installed and warm-loaded (Coqui TTS, "
                      + _XTTS_WEIGHTS_NOTE + ") — private use only: "
                      "XTTS v2 is non-commercial, so this backend "
                      "speaks only for the owner's own characters on "
                      "his own box",
            "steps": steps}


def _provision_rvc(box: dict[str, Any]) -> dict[str, Any]:
    steps: list[str] = []
    if _detect_rvc():
        return {"ok": True, "status": "already-installed",
                "detail": "RVC runtime is already installed",
                "steps": steps}
    refusal = _gate("rvc", box)
    if refusal:
        return {"ok": False, "status": "unavailable", "detail": refusal,
                "steps": steps}
    ok, tail = _pip(["rvc-python"], "pip install rvc-python", steps)
    if not ok:
        return {"ok": False, "status": "failed",
                "detail": "pip install rvc-python failed: "
                          + (tail or "see steps"),
                "steps": steps}
    try:
        import importlib

        importlib.invalidate_caches()
    except Exception:  # noqa: BLE001 - cache refresh is best-effort
        pass
    ok, tail = _verify(
        ["import rvc_python; print('rvc-ok')"],
        "verify rvc_python import", steps, 300)
    if not ok:
        return {"ok": False, "status": "failed",
                "detail": "rvc-python import failed: " + tail,
                "steps": steps}
    try:
        root = Path(os.path.expanduser("~")) / ".nomorals" / "rvc" / "models"
        root.mkdir(parents=True, exist_ok=True)
        steps.append(f"models dir ready: {root}")
    except Exception as exc:  # noqa: BLE001 - the dir is a nicety
        steps.append(f"models dir not created: {exc}")
    try:
        from ..voice import rvc_bridge

        known = [m.name for m in rvc_bridge.list_models()]
    except Exception:  # noqa: BLE001 - listing is a nicety
        known = []
    detail = ("RVC runtime installed (rvc-python). Character voice "
              "models (.pth) are NOT downloadable — each character's "
              "model is trained/supplied by the owner under "
              "~/.nomorals/rvc/models")
    if known:
        detail += f"; models already there: {', '.join(known)}"
    return {"ok": True, "status": "installed", "detail": detail,
            "steps": steps}


def _warm_faceswap_models() -> dict[str, Any]:
    """Download the two swap weights via the model cache. Never raises."""
    try:
        from ..media_edit import _model_cache as mc
        from ..media_edit.faceswap import GFPGAN_FILE, GFPGAN_URL, INSWAPPER_FILE, INSWAPPER_URL

        inswapper = mc.model_path(INSWAPPER_FILE, url=INSWAPPER_URL,
                                  size_mb=530.0)
        gfpgan = mc.model_path(GFPGAN_FILE, url=GFPGAN_URL, size_mb=350.0)
        return {"ok": True,
                "detail": f"weights ready: {inswapper.name}, {gfpgan.name}"}
    except Exception as exc:  # noqa: BLE001 - a failed warm is a report
        return {"ok": False, "detail": f"weight download failed: {exc}"}


def _provision_faceswap(box: dict[str, Any]) -> dict[str, Any]:
    steps: list[str] = []
    if _detect_faceswap():
        return {"ok": True, "status": "already-installed",
                "detail": "face-swap backend is already installed "
                          "(insightface + onnxruntime + gfpgan, weights "
                          "present) — outside the game only; NPCs get "
                          "no faces",
                "steps": steps}
    refusal = _gate("faceswap", box)
    if refusal:
        return {"ok": False, "status": "unavailable", "detail": refusal,
                "steps": steps}
    ok, tail = _pip(["insightface", "onnxruntime", "gfpgan"],
                    "pip install insightface onnxruntime gfpgan", steps)
    if not ok:
        return {"ok": False, "status": "failed",
                "detail": "pip install insightface/onnxruntime/gfpgan "
                          "failed: " + (tail or "see steps"),
                "steps": steps}
    try:
        import importlib

        importlib.invalidate_caches()
    except Exception:  # noqa: BLE001 - cache refresh is best-effort
        pass
    ok, tail = _verify(
        ["import insightface, onnxruntime, gfpgan; "
         "print('faceswap-ok')"],
        "verify insightface/onnxruntime/gfpgan import", steps, 300)
    if not ok:
        return {"ok": False, "status": "failed",
                "detail": "face-swap import check failed: " + tail,
                "steps": steps}
    warmed = _warm_faceswap_models()
    steps.append("warm weights: " + warmed["detail"])
    if not warmed["ok"]:
        return {"ok": False, "status": "failed",
                "detail": warmed["detail"], "steps": steps}
    return {"ok": True, "status": "installed",
            "detail": "face-swap backend installed (insightface + "
                      "onnxruntime + gfpgan; inswapper_128 ~530 MB + "
                      "GFPGANv1.4 ~350 MB warmed; buffalo_l downloads "
                      "on first use) — outside the game only; NPCs get "
                      "no faces",
            "steps": steps}


_PROVISIONERS = {"xtts": _provision_xtts, "rvc": _provision_rvc,
                 "faceswap": _provision_faceswap}


# ── public surface ───────────────────────────────────────────────────────

def preflight() -> dict[str, Any]:
    """Measure this box. Never raises."""
    try:
        return _measure_box()
    except Exception as exc:  # noqa: BLE001 - measurement is best-effort
        _log.debug("provision preflight failed", exc_info=True)
        return {"ok": False, "total_ram_gb": 0.0, "free_ram_gb": 0.0,
                "free_disk_gb": 0.0, "cpus": 0, "platform": sys.platform,
                "profile": "", "detail": f"box measure failed: {exc}"}


def backend_status() -> dict[str, dict[str, Any]]:
    """What the specialists' own detection says right now. Never raises."""
    out: dict[str, dict[str, Any]] = {}
    try:
        out["xtts"] = {"installed": _detect_xtts(),
                       "detail": ("XTTS (Coqui TTS) present — private "
                                  "use only" if _detect_xtts()
                                  else "no TTS backend installed")}
    except Exception as exc:  # noqa: BLE001 - detection is best-effort
        out["xtts"] = {"installed": False, "detail": f"probe failed: {exc}"}
    try:
        from ..voice import rvc_bridge

        probe = rvc_bridge.detect_rvc()
        known = [m.name for m in rvc_bridge.list_models()]
        detail = str(probe.get("detail") or "")
        if known:
            detail += f" — models: {', '.join(known)}"
        out["rvc"] = {"installed": bool(probe.get("ok")), "detail": detail}
    except Exception as exc:  # noqa: BLE001 - detection is best-effort
        out["rvc"] = {"installed": False, "detail": f"probe failed: {exc}"}
    try:
        out["faceswap"] = {
            "installed": _detect_faceswap(),
            "detail": ("face-swap backend present (weights ready)"
                       if _detect_faceswap()
                       else "insightface/onnxruntime/gfpgan or the "
                            "swap weights are missing")}
    except Exception as exc:  # noqa: BLE001 - detection is best-effort
        out["faceswap"] = {"installed": False,
                           "detail": f"probe failed: {exc}"}
    return out


def provision_backends(backends: Any = None) -> dict[str, Any]:
    """Provision the named backends (default all three).

    Returns ``{"ok", "box", "backends": {name: report}}`` where each
    report is ``{"ok", "status", "detail", "steps"}`` with status one
    of installed / already-installed / failed / unavailable. Never
    raises — a backend that blows up reports ``failed``.
    """
    wanted = list(BACKENDS if backends is None else backends)
    box = preflight()
    reports: dict[str, dict[str, Any]] = {}
    unknown = [b for b in wanted if b not in _PROVISIONERS]
    for name in wanted:
        if name in unknown:
            continue
        try:
            reports[name] = _PROVISIONERS[name](box)
        except Exception as exc:  # noqa: BLE001 - a backend never kills
            _log.debug("provision %s failed", name, exc_info=True)
            try:
                from .failures import record_failure

                record_failure(f"characters.provision.{name}", exc)
            except Exception:  # noqa: BLE001 - the record is a nicety
                pass
            reports[name] = {"ok": False, "status": "failed",
                             "detail": f"provision crashed: {exc}",
                             "steps": []}
    ok = bool(reports) and all(r["ok"] for r in reports.values()) \
        and not unknown
    result: dict[str, Any] = {"ok": ok, "box": box, "backends": reports}
    if unknown:
        result["error"] = "unknown backend(s): " + ", ".join(unknown)
    return result


def provision_report_text(result: dict[str, Any]) -> str:
    """Plain-text per-backend report for the chat reply."""
    box = result.get("box") or {}
    lines = ["🎭 character models provision"]
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
    lines.append("XTTS speaks only for the owner's own characters "
                 "(private use only). Face swap runs outside the game "
                 "only — NPCs get no faces.")
    return "\n".join(lines)
