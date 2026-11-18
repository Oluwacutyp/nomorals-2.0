"""The corpus oracle — the PSD decoder checked against real files.

PhotoCraft's testing discipline, in Devon's idiom: the corpus is
real Adobe Photoshop files, and every file carries Photoshop's own
rendering of itself (the merged composite, saved with Maximize
Compatibility). The oracle decodes each file three ways and
compares:

1. **Structure** — the file parses (header, layers, channels) or
   fails with a named reason. A crash is never acceptable.
2. **Flatten parity** — Devon's own compositor over the parsed
   layers, compared pixel-for-pixel against the file's composite
   (Photoshop's rendering of the same stack). Exact for files
   without adjustment layers, non-normal blends, layer effects,
   live text or smart objects; files WITH any of those are
   reported as ``differs-adjustment``, ``differs-blend`` or
   ``differs-unrendered``, with the diff measured and the cause
   named — the flatten does not render any of them yet, and the
   oracle says so instead of hiding it.
3. **Round-trip** — decode → encode → decode → flatten must match
   the first flatten exactly (pixels + layer names + opacity).

The harness runs over any directory of ``.psd`` files. The full
258-file corpus lives in the mining checkout (env
``NM_PHOTOCRAFT_CORPUS`` overrides the path); the vendored
fixtures in ``tests/fixtures/photocraft/`` always run.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .document import PhotoDocument
from .psd import PsdError, PsdFile, encode_psd

__all__ = ["default_corpus_dir", "run_oracle", "summarize"]

#: The mining checkout's corpus, when present on this box.
_DEFAULT_CORPUS = Path.home() / "workspace" / "echelon-mining" \
    / "repos" / "storytold" / "photocraft-corpus" / "photoshop"


def default_corpus_dir() -> str:
    return os.environ.get("NM_PHOTOCRAFT_CORPUS",
                          str(_DEFAULT_CORPUS))


def _max_diff(a: Any, b: Any) -> int:
    import numpy as np

    x = np.asarray(a.convert("RGBA")).astype(int)
    y = np.asarray(b.convert("RGBA")).astype(int)
    if x.shape != y.shape:
        return -1
    return int(abs(x - y).max()) if x.size else 0


def _check_file(path: Path) -> dict:
    entry: dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "status": "failed",
        "mode": "",
        "depth": 0,
        "width": 0,
        "height": 0,
        "layers": 0,
        "adjustments": [],
        "blends": [],
        "effects": False,
        "text": False,
        "smartobjects": False,
        "exact": False,
        "max_diff": -1,
        "roundtrip_ok": False,
        "note": "",
    }
    try:
        data = path.read_bytes()
    except OSError as exc:
        entry["note"] = f"unreadable: {exc}"
        return entry
    try:
        psd = PsdFile.from_bytes(data)
    except PsdError as exc:
        entry["note"] = f"parse failed: {exc}"
        return entry
    entry["mode"] = psd.mode
    entry["depth"] = psd.depth
    entry["width"] = psd.width
    entry["height"] = psd.height
    entry["layers"] = len(psd.layers)
    entry["adjustments"] = sorted({
        key for layer in psd.layers
        for key in layer.adjustment_keys})
    entry["blends"] = sorted({
        layer.blend_key for layer in psd.layers
        if layer.blend_key not in ("norm", "pass")
        and not layer.is_group_start and not layer.is_group_end})
    # Layer effects (lfx2/lrFX: bevel, shadow, glow — Photoshop
    # renders them into the composite; the stored pixels are the
    # flat shape), live text (TySh: the composite re-renders the
    # vector glyphs; the stored raster differs on edges by up to
    # ~40 levels), and smart objects (PlLd/SoLd: smart filters
    # render into the composite; the stored pixels are the
    # unfiltered source) are all parsed but not rendered yet —
    # named causes, same class as adjustments and blends.
    entry["effects"] = any(
        "lfx2" in layer.block_keys or "lrFX" in layer.block_keys
        for layer in psd.layers)
    entry["text"] = any(
        "TySh" in layer.block_keys for layer in psd.layers)
    entry["smartobjects"] = any(
        "PlLd" in layer.block_keys or "SoLd" in layer.block_keys
        for layer in psd.layers)
    try:
        flat = psd.flatten_layers()
        comp = psd.composite_rgba()
    except Exception as exc:  # noqa: BLE001 - one file's render
        entry["note"] = f"render failed: {type(exc).__name__}: {exc}"
        return entry
    entry["max_diff"] = _max_diff(flat, comp)
    entry["exact"] = entry["max_diff"] == 0
    if entry["adjustments"]:
        entry["status"] = "differs-adjustment"
        entry["note"] = (
            "flatten differs from the vendor composite by "
            f"{entry['max_diff']} per channel — adjustment layers "
            f"({', '.join(entry['adjustments'])}) are parsed but "
            "not rendered yet")
    elif entry["blends"]:
        entry["status"] = "differs-blend"
        entry["note"] = (
            "flatten differs from the vendor composite by "
            f"{entry['max_diff']} per channel — non-normal blend "
            f"mode(s) ({', '.join(entry['blends'])}) are parsed "
            "but rendered as normal, not rendered yet")
    elif entry["effects"] or entry["text"] or entry["smartobjects"]:
        causes = []
        if entry["effects"]:
            causes.append("layer effects (bevel/shadow/glow)")
        if entry["text"]:
            causes.append("live text (vector glyphs re-rendered)")
        if entry["smartobjects"]:
            causes.append("smart objects (smart filters)")
        entry["status"] = "differs-unrendered"
        entry["note"] = (
            "flatten differs from the vendor composite by "
            f"{entry['max_diff']} per channel — "
            + ", ".join(causes)
            + " are parsed but not rendered yet")
    elif entry["exact"]:
        entry["status"] = "exact"
    else:
        entry["status"] = "differs"
        entry["note"] = (
            "flatten differs from the vendor composite by "
            f"{entry['max_diff']} per channel with no adjustment "
            "layers to blame")
    try:
        doc = PhotoDocument.from_psd_bytes(data)
        again = PsdFile.from_bytes(encode_psd(doc))
        entry["roundtrip_ok"] = (
            _max_diff(again.flatten_layers(), flat) == 0
            and [layer.display_name for layer in again.layers]
            == [layer.display_name for layer in psd.layers]
            and [layer.blend_key for layer in again.layers]
            == [layer.blend_key for layer in psd.layers])
        if (not entry["roundtrip_ok"] and not psd.layers
                and [layer.display_name for layer in again.layers]
                == ["Background"]
                and _max_diff(again.flatten_layers(), flat) == 0):
            # A flat file opens as one synthesized Background
            # layer (the document model has no layerless state),
            # so its round-trip legitimately gains that layer.
            entry["roundtrip_ok"] = True
        if not entry["roundtrip_ok"]:
            entry["note"] += "; round-trip flatten differs"
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        entry["roundtrip_ok"] = False
        entry["note"] += f"; round-trip failed: {exc}"
    return entry


def run_oracle(corpus_dir: str | os.PathLike[str],
               *, limit: int | None = None) -> dict:
    """Run the oracle over every ``.psd`` under ``corpus_dir``."""
    root = Path(corpus_dir)
    files = sorted(root.rglob("*.psd"))
    if limit is not None:
        files = files[:limit]
    entries = [_check_file(path) for path in files]
    summary = {
        "files": len(entries),
        "exact": sum(1 for e in entries
                     if e["status"] == "exact"),
        "differs_adjustment": sum(
            1 for e in entries
            if e["status"] == "differs-adjustment"),
        "differs_blend": sum(
            1 for e in entries
            if e["status"] == "differs-blend"),
        "differs_unrendered": sum(
            1 for e in entries
            if e["status"] == "differs-unrendered"),
        "differs": sum(1 for e in entries
                       if e["status"] == "differs"),
        "failed": sum(1 for e in entries
                      if e["status"] == "failed"),
        "roundtrip_ok": sum(1 for e in entries
                            if e["roundtrip_ok"]),
    }
    return {"corpus": str(root), "summary": summary,
            "files": entries}


def summarize(report: dict) -> str:
    summary = report["summary"]
    lines = [
        f"corpus oracle: {report['corpus']}",
        f"  files: {summary['files']}",
        f"  exact (flatten == vendor composite): {summary['exact']}",
        "  differs, adjustment layers unrendered: "
        f"{summary['differs_adjustment']}",
        "  differs, non-normal blends unrendered: "
        f"{summary['differs_blend']}",
        "  differs, effects/text/smart objects unrendered: "
        f"{summary['differs_unrendered']}",
        f"  differs, unexplained: {summary['differs']}",
        f"  failed to parse/render: {summary['failed']}",
        f"  round-trip ok: {summary['roundtrip_ok']}",
    ]
    for entry in report["files"]:
        if entry["status"] in ("differs", "failed"):
            lines.append(
                f"  {entry['name']}: {entry['status']} — "
                f"{entry['note']}")
    return "\n".join(lines)
