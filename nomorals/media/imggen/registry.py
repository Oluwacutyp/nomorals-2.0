"""Imggen model registry — every installed image model, one listing.

Devon's image weights live in two places:

- ``checkpoint_dir()`` (``$DEVON_IMGGEN_DIR`` / ``~/.nomorals/imggen``)
  — Devon's own trained runs plus any dropped-in checkpoints;
- ``~/.devon-models/`` — the root other organs already expect
  (e.g. ``media/directed/filler.py`` looks for SD weights there).

This module *discovers* what is installed and classifies safetensors
files by **header key inspection only** (the safetensors header is a
length-prefixed JSON block; reading it loads no tensors):

- keys containing ``conditioner.embedders.1`` → ``sdxl``
- keys containing ``double_blocks`` / ``single_transformer_blocks``
  → ``flux``
- LDM layout (``model.diffusion_model.*``) or diffusers layout
  (``unet.*`` + ``text_encoder.*``, no ``text_encoder_2``) → ``sd15``
- anything else → ``unknown``

Classification is metadata; *loadability* is a separate, honest flag.
In this pass only Devon's native checkpoints and LDM-format SD1.5
safetensors can actually load — SDXL/FLUX entries are listed with
``loadable=False`` and a reason naming the class and the missing
loader, never silently omitted and never pretended runnable.
"""

from __future__ import annotations

import json
import os
import struct

from . import checkpoint_dir

__all__ = ["list_models", "resolve", "registry_roots"]

_FORMAT_NATIVE = "devon-imggen-1"
_FORMAT_LDM = "ldm-safetensors"
_FORMAT_DIFFUSERS = "diffusers-safetensors"
_FORMAT_UNKNOWN = "safetensors"

_CLASS_ORDER = {"native": 0, "sd15": 1, "sdxl": 2, "flux": 3,
                "unknown": 4}

_NOT_LOADABLE_REASONS = {
    "sdxl": "SDXL checkpoints are registered but the SDXL loader "
            "arrives in a later pass (A2)",
    "flux": "FLUX checkpoints are registered but the FLUX loader "
            "arrives in a later pass (B)",
    "unknown": "unrecognized safetensors layout — cannot classify "
               "this file as a known image model family",
}


def registry_roots() -> list[str]:
    """Roots scanned for image models, in discovery order."""
    return [checkpoint_dir(), os.path.expanduser("~/.devon-models")]


# ---------------------------------------------------------------------------
# Safetensors header inspection (no tensor data is read)
# ---------------------------------------------------------------------------

def _safetensors_keys(path: str) -> set[str] | None:
    """Key set from a safetensors header; None if unreadable/not one."""
    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                return None
            (hlen,) = struct.unpack("<Q", raw)
            if hlen <= 0 or hlen > 1 << 31:
                return None
            header = json.loads(f.read(hlen).decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(header, dict):
        return None
    return {k for k in header if k != "__metadata__"}


def _classify_keys(keys: set[str]) -> tuple[str, str]:
    """(model_class, format) from a safetensors key set."""
    if any("conditioner.embedders.1" in k for k in keys) \
            or any(k.startswith("text_encoder_2.") for k in keys):
        return "sdxl", _FORMAT_LDM if any(
            k.startswith("model.diffusion_model.") for k in keys) \
            else _FORMAT_DIFFUSERS
    if any("double_blocks" in k or "single_transformer_blocks" in k
           for k in keys):
        return "flux", _FORMAT_LDM if any(
            k.startswith("model.diffusion_model.") for k in keys) \
            else _FORMAT_DIFFUSERS
    if any(k.startswith("model.diffusion_model.") for k in keys):
        return "sd15", _FORMAT_LDM
    if any(k.startswith("unet.") for k in keys) \
            and any(k.startswith("text_encoder.") for k in keys):
        return "sd15", _FORMAT_DIFFUSERS
    return "unknown", _FORMAT_UNKNOWN


def _entry_for_safetensors(path: str) -> dict:
    name = os.path.splitext(os.path.basename(path))[0]
    keys = _safetensors_keys(path)
    if keys is None:
        return {
            "name": name, "path": path, "model_class": "unknown",
            "format": _FORMAT_UNKNOWN,
            "bytes": os.path.getsize(path),
            "loadable": False,
            "loadable_reason": "could not read a safetensors header "
                               "from this file",
        }
    model_class, fmt = _classify_keys(keys)
    loadable = model_class == "sd15" and fmt == _FORMAT_LDM
    if loadable:
        reason = ""
    elif model_class == "sd15":
        reason = ("SD1.5 checkpoint in diffusers layout — the loader "
                  "speaks compvis LDM format; convert the checkpoint "
                  "to a single LDM .safetensors first")
    else:
        reason = _NOT_LOADABLE_REASONS[model_class]
    return {
        "name": name, "path": path, "model_class": model_class,
        "format": fmt, "bytes": os.path.getsize(path),
        "loadable": loadable, "loadable_reason": reason,
    }


def _entry_for_native(path: str, run: str = "") -> dict:
    return {
        "name": run or os.path.splitext(os.path.basename(path))[0],
        "path": path, "model_class": "native",
        "format": _FORMAT_NATIVE,
        "bytes": os.path.getsize(path) if os.path.exists(path) else 0,
        "loadable": True, "loadable_reason": "",
        **({"run": run} if run else {}),
    }


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _scan_safetensors() -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for root in registry_roots():
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            depth = dirpath[len(root):].count(os.sep)
            if depth > 2:
                dirnames[:] = []
                continue
            for fn in sorted(filenames):
                if not fn.endswith(".safetensors"):
                    continue
                path = os.path.join(dirpath, fn)
                real = os.path.realpath(path)
                if real in seen:
                    continue
                seen.add(real)
                out.append(_entry_for_safetensors(path))
    return out


def list_models() -> list[dict]:
    """All installed image models: native runs + safetensors entries.

    Sorted loadable-first (Devon's native class before SD1.5, then
    the registered-but-not-yet-loadable classes), newest first within
    a class. Every entry: ``{name, path, model_class, format, bytes,
    loadable, loadable_reason}`` (native entries also carry ``run``).
    """
    from .pipeline import list_native_checkpoints

    entries: list[dict] = [
        _entry_for_native(c["path"], c["run"])
        for c in list_native_checkpoints()
    ]
    entries.extend(_scan_safetensors())
    def _key(e: dict):
        try:
            mtime = os.path.getmtime(e["path"])
        except OSError:
            mtime = 0.0
        return (not e["loadable"],
                _CLASS_ORDER.get(e["model_class"], 9),
                -mtime, e["name"])
    entries.sort(key=_key)
    return entries


def resolve(name_or_path: str) -> dict | None:
    """Resolve a registry name or a filesystem path to an entry.

    Name match is the filename stem (safetensors) or run name
    (native), case-insensitive. An existing filesystem path resolves
    directly and is classified on the spot, wherever it lives.
    Returns None when nothing matches.
    """
    if not name_or_path:
        return None
    if os.path.exists(name_or_path):
        if name_or_path.endswith(".safetensors"):
            return _entry_for_safetensors(name_or_path)
        if name_or_path.endswith(".pt"):
            return _entry_for_native(name_or_path)
        return {
            "name": os.path.splitext(
                os.path.basename(name_or_path))[0],
            "path": name_or_path, "model_class": "unknown",
            "format": "unknown", "bytes": os.path.getsize(name_or_path),
            "loadable": False,
            "loadable_reason": "not a recognized image model file "
                               "(.safetensors or Devon .pt checkpoint)",
        }
    want = name_or_path.lower()
    matches = [e for e in list_models()
               if e["name"].lower() == want
               or e.get("run", "").lower() == want]
    if not matches:
        return None
    loadable = [e for e in matches if e["loadable"]]
    return (loadable or matches)[0]
