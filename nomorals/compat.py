"""Optional-dependency detection.

The core package runs with **zero** third-party imports. Accelerators are probed
once, cached, and every consumer has a pure-stdlib fallback. This module is the
single place that asks "is X installed?" so that behaviour is predictable and
reportable.
"""

from __future__ import annotations

import importlib
import importlib.util
import shutil
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

__all__ = [
    "Feature",
    "available",
    "feature_report",
    "load_optional",
    "native_section",
    "report_as_text",
    "require",
    "which",
]


@dataclass(frozen=True)
class Feature:
    """A named optional capability with an install hint."""

    name: str
    module: str
    purpose: str
    install: str = ""
    fallback: str = ""


FEATURES: tuple[Feature, ...] = (
    Feature(
        "numpy",
        "numpy",
        "Accelerated vector similarity and the native training loop.",
        "pip install numpy",
        "Pure-Python cosine over array('f'); ~30x slower, functionally identical.",
    ),
    Feature(
        "yt_dlp",
        "yt_dlp",
        "Video/audio download from ~1800 sites.",
        "pip install yt-dlp",
        "Direct-URL downloader only; site extraction unavailable.",
    ),
    Feature(
        "torch",
        "torch",
        "GPU fine-tuning backends.",
        "pip install torch",
        "Native pure-Python trainer (tiny models only) or generated external configs.",
    ),
    Feature(
        "transformers",
        "transformers",
        "Hugging Face tokenizers and model loading.",
        "pip install transformers",
        "Built-in BPE trainer/tokenizer in nomorals.training.tokenize.",
    ),
    Feature(
        "huggingface_hub",
        "huggingface_hub",
        "Resumable, cached HF downloads.",
        "pip install huggingface-hub",
        "Raw HTTP range-request downloader in nomorals.llm.download.",
    ),
    Feature(
        "peft",
        "peft",
        "LoRA/QLoRA adapters.",
        "pip install peft",
        "Full fine-tune configs only.",
    ),
    Feature(
        "PIL",
        "PIL",
        "Rich image decoding for the vision pipeline.",
        "pip install pillow",
        "Built-in PNG/JPEG/GIF/BMP header parser and raw pixel decode for PNG.",
    ),
)

_BINARIES: dict[str, str] = {
    "ffmpeg": "Audio/video muxing and transcoding (media post-processing).",
    "ffprobe": "Media stream inspection.",
    "git": "Backup versioning and push to a remote repository.",
    "yt-dlp": "Command-line fallback for video download.",
    "llama-server": "Local GGUF inference server.",
    "bwrap": "Bubblewrap sandbox for shell execution (stronger isolation).",
    "unshare": "Namespace isolation fallback for shell execution.",
}


@lru_cache(maxsize=None)
def available(name: str) -> bool:
    """Return True if the named optional feature is importable."""
    for feature in FEATURES:
        if feature.name == name:
            return importlib.util.find_spec(feature.module) is not None
    # Fall back to treating the argument as a module path.
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@lru_cache(maxsize=None)
def which(binary: str) -> str | None:
    """Absolute path to an external binary, or None."""
    return shutil.which(binary)


def load_optional(name: str) -> Any:
    """Import an optional module, returning None when unavailable."""
    if not available(name):
        return None
    target = name
    for feature in FEATURES:
        if feature.name == name:
            target = feature.module
            break
    try:
        return importlib.import_module(target)
    except Exception:  # pragma: no cover - broken install  # noqa: E104 - optional import probe; None means unavailable
        return None


def require(name: str) -> Any:
    """Import an optional module or raise with an actionable message."""
    mod = load_optional(name)
    if mod is None:
        feature = next((f for f in FEATURES if f.name == name), None)
        hint = feature.install if feature else f"pip install {name}"
        raise RuntimeError(
            f"optional dependency {name!r} is not installed. Install it with: {hint}"
        )
    return mod


def report_as_text(report: dict[str, Any]) -> str:
    """Human-readable rendering of :func:`feature_report`."""
    lines = [
        f"python:   {report['python']}",
        f"platform: {report['platform']}",
        "",
        "optional python packages:",
    ]
    for f in report["features"]:
        mark = "yes" if f["available"] else " no"
        lines.append(f"  [{mark}] {f['name']:<18} {f['purpose']}")
        if not f["available"] and f["install"]:
            lines.append(f"          install:  {f['install']}")
            lines.append(f"          fallback: {f['fallback']}")
    lines += ["", "external binaries:"]
    for b in report["binaries"]:
        mark = "yes" if b["path"] else " no"
        path = b["path"] or "-"
        lines.append(f"  [{mark}] {b['name']:<14} {path:<28} {b['purpose']}")
    nat = report.get("native")
    if nat:
        lines += ["", "native accelerators:"]
        for name, status in (nat.get("kernels") or {}).items():
            mark = "yes" if status == "native" else " no"
            lines.append(f"  [{mark}] {name:<14} {status:<11} backend: "
                         f"{'native-cpp' if status == 'native' else 'pure-python'}")
        overall = nat.get("overall", "unknown")
        if overall != "native":
            lines.append("          fallback: pure-python (correctness identical, "
                         "slower)")
            compiler = nat.get("compiler")
            if compiler:
                lines.append(f"          build:    nm native --build  "
                             f"(compiler: {compiler})")
            else:
                lines.append("          build:    no C++ compiler on PATH — "
                             "pure-python fallback is the supported path")
        else:
            lines.append("          all kernels on native-cpp"
                         + (f" ({nat.get('version')})" if nat.get("version") else ""))
    return "\n".join(lines)


def native_section(info: dict[str, Any]) -> dict[str, Any]:
    """Build the ``report['native']`` block from ``nomorals.native.info()``.

    Takes the info dict instead of importing the native package: this module
    is L0 and must not import the L2 native package (see test_layering).
    Callers that may import native (e.g. the ``nm doctor`` command) call
    ``native.info()`` and pass the result here.

    Per-kernel status: ``native`` (built+loaded), ``stale-arch`` (.so exists
    but would not load — wrong arch or corrupt), ``missing`` (not built).
    """
    kernels: dict[str, str] = {}
    vec = {"built": info.get("built", False), "loaded": info.get("loaded", False)}
    for name, block in (("vecsim", vec), ("mlptrain", info.get("mlp") or {}),
                        ("bpe", info.get("bpe") or {}),
                        ("memextract", info.get("mem") or {})):
        built = bool(block.get("built"))
        loaded = bool(block.get("loaded"))
        kernels[name] = ("native" if loaded
                         else "stale-arch" if built else "missing")
    loaded_count = sum(1 for s in kernels.values() if s == "native")
    overall = ("native" if loaded_count == len(kernels)
               else "partial" if loaded_count else "pure-python")
    return {
        "overall": overall,
        "kernels": kernels,
        "compiler": info.get("compiler"),
        "version": info.get("version"),
        "fallback": "pure-python",
    }


def feature_report() -> dict[str, Any]:
    """Machine-readable report of what is available in this environment."""
    import platform

    report: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "features": [],
        "binaries": [],
        # Filled in by callers that may import the native package (L2):
        # see compat.native_section().  None keeps the schema stable.
        "native": None,
    }
    for feature in FEATURES:
        report["features"].append(
            {
                "name": feature.name,
                "available": available(feature.name),
                "purpose": feature.purpose,
                "install": feature.install,
                "fallback": feature.fallback,
            }
        )
    for binary, purpose in _BINARIES.items():
        report["binaries"].append(
            {"name": binary, "path": which(binary), "purpose": purpose}
        )
    return report
