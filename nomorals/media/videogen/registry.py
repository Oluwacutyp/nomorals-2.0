"""Videogen model registry — every usable video model, one listing.

Sibling of :mod:`nomorals.media.imggen.registry` (same conventions:
discovery roots, entry dicts, honest ``loadable`` flags, ``resolve()``),
with the video shape: a model is a *backend + a diffusers source* (an
HF repo id or a local diffusers-format directory), and LoRAs are
separate installable data the backends apply at load time.

Discovery:

- **Built-ins** — the three stock checkpoints the backends ship with
  (LTX-Video base, LTX-Video 0.9.7 distilled, Wan 2.2 TI2V-5B), always
  listed.
- **Local model dirs** under the models root (``$DEVON_MODELS_DIR`` /
  ``~/.devon-models`` — the root other organs already expect): a
  directory holding ``model_index.json`` is a diffusers model; its
  backend is classified from the pipeline ``_class_name`` in that file
  (``LTXPipeline`` family → ``ltx``, ``WanPipeline`` family → ``wan``).
  An unrecognized pipeline class is *listed*, ``loadable=False``, with
  the class named — never silently omitted, never pretended runnable.
- **LoRAs** live under ``<models root>/loras/<backend>/<name>``: either
  a directory containing ``adapter_config.json`` and/or
  ``*.safetensors`` weights, or a single ``<name>.safetensors`` file.
  ``<backend>`` is ``ltx`` or ``wan`` — a LoRA only applies to its own
  backend's pipelines.

Classification is metadata; *loadability* is a separate, honest flag.
Licenses are labels from :mod:`nomorals.media.model_license`
(``license_key`` per entry) — metadata, never a gate.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable

from .capabilities import (
    LTX_MIN_VRAM_GB,
    WAN_MIN_VRAM_GB,
    VideogenError,
)
from .ltx_backend import BASE_MODEL_ID, DISTILLED_MODEL_ID
from .wan_backend import MODEL_ID as WAN_MODEL_ID

__all__ = [
    "models_root",
    "list_models",
    "list_loras",
    "resolve",
    "resolve_lora",
    "normalize_loras",
    "BACKENDS",
]

BACKENDS = ("ltx", "wan")

_BACKEND_FLOORS = {"ltx": LTX_MIN_VRAM_GB, "wan": WAN_MIN_VRAM_GB}

#: pipeline ``_class_name`` values in a diffusers ``model_index.json``
#: that each backend can actually load.
_PIPELINE_CLASSES = {
    "ltx": {"LTXPipeline", "LTXImageToVideoPipeline",
            "LTXConditionPipeline"},
    "wan": {"WanPipeline", "WanImageToVideoPipeline"},
}


def models_root() -> str:
    """Root scanned for local video models + LoRAs."""
    return os.environ.get("DEVON_MODELS_DIR") or os.path.expanduser(
        "~/.devon-models")


# ---------------------------------------------------------------------------
# Built-in (stock) entries
# ---------------------------------------------------------------------------

def _builtin_entries() -> list[dict]:
    return [
        {
            "name": "ltx-distilled", "backend": "ltx",
            "source": DISTILLED_MODEL_ID, "format": "diffusers-repo",
            "vram_gb": LTX_MIN_VRAM_GB, "license_key": "ltx",
            "loadable": True, "loadable_reason": "",
            "builtin": True, "default": True, "distilled": True,
        },
        {
            "name": "ltx-base", "backend": "ltx",
            "source": BASE_MODEL_ID, "format": "diffusers-repo",
            "vram_gb": LTX_MIN_VRAM_GB, "license_key": "ltx",
            "loadable": True, "loadable_reason": "",
            "builtin": True, "default": False, "distilled": False,
        },
        {
            "name": "wan-ti2v-5b", "backend": "wan",
            "source": WAN_MODEL_ID, "format": "diffusers-repo",
            "vram_gb": WAN_MIN_VRAM_GB, "license_key": "wan",
            "loadable": True, "loadable_reason": "",
            "builtin": True, "default": True, "distilled": False,
        },
    ]


# ---------------------------------------------------------------------------
# Local discovery
# ---------------------------------------------------------------------------

def _read_pipeline_class(dirpath: str) -> str:
    """``_class_name`` from a diffusers ``model_index.json`` ('' if none)."""
    try:
        with open(os.path.join(dirpath, "model_index.json"),
                  encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return ""
    if isinstance(data, dict):
        return str(data.get("_class_name") or "")
    return ""


def _backend_for_class(cls: str) -> str:
    for backend, classes in _PIPELINE_CLASSES.items():
        if cls in classes:
            return backend
    return ""


def _entry_for_model_dir(dirpath: str) -> dict:
    name = os.path.basename(os.path.normpath(dirpath))
    cls = _read_pipeline_class(dirpath)
    backend = _backend_for_class(cls)
    if backend:
        return {
            "name": name, "backend": backend, "source": dirpath,
            "format": "diffusers-dir",
            "vram_gb": _BACKEND_FLOORS[backend],
            "license_key": backend,
            "loadable": True, "loadable_reason": "",
            "builtin": False, "default": False, "distilled": False,
            "pipeline_class": cls,
        }
    reason = (f"pipeline class {cls!r} in model_index.json is not an "
              "LTX or Wan pipeline Devon's backends can load"
              if cls else
              "no model_index.json — not a diffusers model directory")
    return {
        "name": name, "backend": "", "source": dirpath,
        "format": "diffusers-dir", "vram_gb": None,
        "license_key": "", "loadable": False,
        "loadable_reason": reason,
        "builtin": False, "default": False, "distilled": False,
        "pipeline_class": cls,
    }


def _scan_local_models() -> list[dict]:
    root = models_root()
    out: list[dict] = []
    if not os.path.isdir(root):
        return out
    candidates: list[str] = []
    if os.path.isfile(os.path.join(root, "model_index.json")):
        candidates.append(root)
    for dirpath, dirnames, _filenames in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        if depth > 2:
            dirnames[:] = []
            continue
        # the loras tree holds adapters, not models
        dirnames[:] = [d for d in dirnames if d != "loras"]
        if dirpath != root and os.path.isfile(
                os.path.join(dirpath, "model_index.json")):
            candidates.append(dirpath)
            dirnames[:] = []  # a model dir's innards are not more models
    builtin_names = {e["name"].lower() for e in _builtin_entries()}
    for path in sorted(candidates):
        entry = _entry_for_model_dir(path)
        if entry["name"].lower() in builtin_names:
            entry["name"] = f"local-{entry['name']}"
        out.append(entry)
    return out


def _is_lora_dir(dirpath: str) -> bool:
    try:
        names = os.listdir(dirpath)
    except OSError:
        return False
    return ("adapter_config.json" in names
            or any(n.endswith(".safetensors") for n in names))


def list_loras(backend: str | None = None) -> list[dict]:
    """Installed LoRA adapters: ``{name, path, backend, format}``.

    Convention: ``<models root>/loras/<backend>/<name>/`` (a directory
    with ``adapter_config.json`` and/or ``*.safetensors``) or a single
    ``<models root>/loras/<backend>/<name>.safetensors`` file.
    """
    root = os.path.join(models_root(), "loras")
    out: list[dict] = []
    if os.path.isdir(root):
        for b in sorted(os.listdir(root)):
            if backend and b.lower() != backend.lower():
                continue
            bdir = os.path.join(root, b)
            if not os.path.isdir(bdir):
                continue
            for name in sorted(os.listdir(bdir)):
                path = os.path.join(bdir, name)
                if os.path.isdir(path) and _is_lora_dir(path):
                    out.append({"name": name, "path": path,
                                "backend": b.lower(), "format": "dir"})
                elif os.path.isfile(path) and name.endswith(".safetensors"):
                    out.append({
                        "name": os.path.splitext(name)[0], "path": path,
                        "backend": b.lower(), "format": "file"})
    return out


# ---------------------------------------------------------------------------
# Public listing + resolution
# ---------------------------------------------------------------------------

def list_models() -> list[dict]:
    """All video models: built-ins first, then discovered local dirs.

    Every entry: ``{name, backend, source, format, vram_gb,
    license_key, loras, loadable, loadable_reason, builtin, default,
    distilled}`` — ``loras`` names the installed adapters for the
    entry's backend (see :func:`list_loras`).
    """
    entries = _builtin_entries()
    entries.extend(_scan_local_models())
    loras = list_loras()
    for e in entries:
        e["loras"] = [l["name"] for l in loras
                      if l["backend"] == e["backend"]]
    return entries


def resolve(name_or_path: str) -> dict | None:
    """Resolve a registry name or a local dir path to a model entry.

    Name match is case-insensitive over built-ins and discovered local
    models. An existing directory path is classified on the spot.
    Returns None when nothing matches (a bare ``org/repo`` string is a
    raw HF id the caller may still use with an explicit backend).
    """
    if not name_or_path:
        return None
    if os.path.isdir(name_or_path):
        return _entry_for_model_dir(name_or_path)
    if os.path.exists(name_or_path):
        return {
            "name": os.path.splitext(os.path.basename(name_or_path))[0],
            "backend": "", "source": name_or_path, "format": "file",
            "vram_gb": None, "license_key": "", "loras": [],
            "loadable": False,
            "loadable_reason": "video models are diffusers directories "
                               "or HF repo ids, not single files",
            "builtin": False, "default": False, "distilled": False,
        }
    want = name_or_path.lower()
    matches = [e for e in list_models() if e["name"].lower() == want]
    if not matches:
        return None
    loadable = [e for e in matches if e["loadable"]]
    return (loadable or matches)[0]


def resolve_lora(name_or_path: str) -> dict | None:
    """Resolve a LoRA registry name or a filesystem path to an entry.

    Returns ``{name, path, backend, format}`` — ``backend`` is "" for a
    raw path outside the ``loras/<backend>/`` convention (the applying
    backend will fail honestly if the adapter doesn't fit it).
    """
    if not name_or_path:
        return None
    if os.path.isdir(name_or_path) and _is_lora_dir(name_or_path):
        return {"name": os.path.basename(os.path.normpath(name_or_path)),
                "path": name_or_path, "backend": "", "format": "dir"}
    if os.path.isfile(name_or_path) and name_or_path.endswith(
            ".safetensors"):
        return {"name": os.path.splitext(
                    os.path.basename(name_or_path))[0],
                "path": name_or_path, "backend": "", "format": "file"}
    if os.path.exists(name_or_path):
        return None
    want = name_or_path.lower()
    for entry in list_loras():
        if entry["name"].lower() == want:
            return entry
    return None


def _split_weight(text: str) -> tuple[str, float]:
    """``"name:0.8"`` → (``"name"``, 0.8); plain ``"name"`` → weight 1.0."""
    head, sep, tail = text.rpartition(":")
    if sep and head:
        try:
            return head, float(tail)
        except ValueError:
            pass
    return text, 1.0


def normalize_loras(spec: Any) -> list[dict]:
    """Normalize a LoRA selection to ``[{name, source, weight, backend}]``.

    Accepted shapes: a single name/path string (``"anime"`` or
    ``"anime:0.8"``), or a list whose items are strings, ``(spec,
    weight)`` pairs, or dicts ``{"name"|"path"|"source", "weight"}``.
    Names resolve through the registry; unknown names raise
    :class:`VideogenError` naming the LoRA — never a silent skip.
    """
    if spec is None:
        return []
    items: Iterable[Any]
    if isinstance(spec, (str, tuple, dict)):
        items = [spec]
    else:
        items = list(spec)
    out: list[dict] = []
    for item in items:
        weight = 1.0
        ref: Any = item
        if isinstance(item, dict):
            ref = (item.get("name") or item.get("path")
                   or item.get("source") or "")
            weight = float(item.get("weight", 1.0))
        elif isinstance(item, (tuple, list)):
            if len(item) != 2:
                raise VideogenError(
                    f"LoRA spec {item!r} must be (name-or-path, weight)")
            ref, weight = str(item[0]), float(item[1])
        ref = str(ref).strip()
        if isinstance(item, str):
            ref, weight = _split_weight(ref)
        if not ref:
            raise VideogenError("empty LoRA reference in selection")
        entry = resolve_lora(ref)
        if entry is None:
            known = ", ".join(l["name"] for l in list_loras()) or "none"
            raise VideogenError(
                f"unknown LoRA {ref!r} — installed LoRAs: {known}; "
                "LoRAs live under "
                f"{os.path.join(models_root(), 'loras')}/<backend>/<name>")
        out.append({"name": entry["name"], "source": entry["path"],
                    "weight": weight, "backend": entry["backend"]})
    return out
