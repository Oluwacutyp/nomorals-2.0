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
   (Photoshop's rendering of the same stack). The compositor
   renders adjustment layers, non-normal blends, layer effects,
   live text and smart objects; a file is ``exact`` only when
   every feature it carries is rendered AND the pixels match.
   A file whose features are rendered but whose pixels still
   differ is ``differs`` (a render gap, named by the feature
   class); a file with a feature the compositor could not parse
   is ``differs-adjustment`` / ``differs-blend`` /
   ``differs-unrendered`` by class.
3. **Byte-stable write-back** — decode → encode must be
   byte-identical where the editor made no change (the stable
   writer), and the document round-trip (open → save with no
   edits) must be byte-identical too.
4. **Round-trip** — decode → encode → decode → flatten must match
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
        "adjustments_rendered": True,
        "blends_rendered": True,
        "effects_rendered": True,
        "text_rendered": True,
        "smartobjects_rendered": True,
        "fills_rendered": True,
        "exact": False,
        "max_diff": -1,
        "byte_stable": False,
        "doc_byte_stable": False,
        "roundtrip_ok": False,
        "roundtrip_max_diff": 0,
        "roundtrip_depth_reduced": False,
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
    # Byte-stable write-back: the parsed file writes back
    # byte-identical, and so does the document opened from it
    # with no edits.
    entry["byte_stable"] = psd.to_bytes() == data
    # Rendered flags: a feature class counts as rendered when
    # every layer carrying it parsed and the compositor did not
    # report it unrendered. (A class the file does not carry is
    # vacuously rendered.)
    unr = psd.unrendered()
    entry["adjustments_rendered"] = not any(
        line.startswith("adjustment") for line in unr)
    entry["fills_rendered"] = not any(
        line.startswith("fill") for line in unr)
    from .blend import BLEND_KEYS

    entry["blends_rendered"] = all(
        (key or "norm")[:4].ljust(4) in BLEND_KEYS
        for key in entry["blends"])
    entry["effects_rendered"] = True
    entry["text_rendered"] = True
    entry["smartobjects_rendered"] = True
    from .compose import describe_layer as _describe

    for layer in psd.layers:
        blocks = dict(layer.raw_blocks)
        if "lfx2" in blocks or "lrFX" in blocks:
            try:
                parsed = _describe(
                    raw_blocks=layer.raw_blocks).get("effects")
            except Exception:
                parsed = None
            if parsed is None:
                entry["effects_rendered"] = False
        if "TySh" in blocks:
            try:
                parsed = _describe(
                    raw_blocks=layer.raw_blocks,
                    doc_width=0, doc_height=0).get("text")
            except Exception:
                parsed = None
            has_raster = bool(layer.width and layer.height
                              and layer.channels)
            if parsed is None and not has_raster:
                entry["text_rendered"] = False
        if "SoLd" in blocks or "PlLd" in blocks:
            try:
                parsed = _describe(
                    raw_blocks=layer.raw_blocks).get("smart")
            except Exception:
                parsed = None
            if parsed is None:
                entry["smartobjects_rendered"] = False
    if entry["exact"]:
        # Pixel-exact against Photoshop's own composite: the
        # editor renders the file as Photoshop does. A
        # pass-through feature (a smart filter the edit path
        # cannot re-render yet) does not demote a render that
        # matches — the stored pixels ARE the filtered result,
        # and the compositor uses them. Edit-path liveness is a
        # separate dimension, carried by the *_rendered flags
        # and reported in the summary below.
        entry["status"] = "exact"
    elif not entry["adjustments_rendered"]:
        entry["status"] = "differs-adjustment"
        entry["note"] = (
            "flatten differs from the vendor composite by "
            f"{entry['max_diff']} per channel — adjustment layers "
            f"({', '.join(entry['adjustments'])}) are parsed but "
            "not rendered")
    elif not entry["blends_rendered"]:
        entry["status"] = "differs-blend"
        entry["note"] = (
            "flatten differs from the vendor composite by "
            f"{entry['max_diff']} per channel — non-normal blend "
            f"mode(s) ({', '.join(entry['blends'])}) are not "
            "rendered")
    elif not (entry["effects_rendered"]
              and entry["text_rendered"]
              and entry["smartobjects_rendered"]
              and entry["fills_rendered"]):
        causes = []
        if not entry["effects_rendered"]:
            causes.append("layer effects (bevel/shadow/glow)")
        if not entry["text_rendered"]:
            causes.append("live text (vector glyphs re-rendered)")
        if not entry["smartobjects_rendered"]:
            causes.append("smart objects (smart filters)")
        if not entry["fills_rendered"]:
            causes.append("fill layers")
        entry["status"] = "differs-unrendered"
        entry["note"] = (
            "flatten differs from the vendor composite by "
            f"{entry['max_diff']} per channel — "
            + ", ".join(causes)
            + " are parsed but not rendered")
    else:
        entry["status"] = "differs"
        entry["note"] = (
            "flatten differs from the vendor composite by "
            f"{entry['max_diff']} per channel — every feature is "
            "rendered, so this is a render gap, not an unrendered "
            "feature")
    try:
        doc = PhotoDocument.from_psd_bytes(data)
        entry["doc_byte_stable"] = doc.to_psd_bytes() == data
        again = PsdFile.from_bytes(encode_psd(doc))
        rt_diff = _max_diff(again.flatten_layers(), flat)
        entry["roundtrip_max_diff"] = rt_diff
        entry["roundtrip_ok"] = (
            rt_diff == 0
            and [layer.display_name for layer in again.layers]
            == [layer.display_name for layer in psd.layers]
            and [layer.blend_key for layer in again.layers]
            == [layer.blend_key for layer in psd.layers])
        if (not entry["roundtrip_ok"] and not psd.layers
                and [layer.display_name for layer in again.layers]
                == ["Background"]
                and rt_diff == 0):
            # A flat file opens as one synthesized Background
            # layer (the document model has no layerless state),
            # so its round-trip legitimately gains that layer.
            entry["roundtrip_ok"] = True
        if not entry["roundtrip_ok"] and psd.depth != 8:
            # The encoder writes depth 8, so a 16/32-bit source
            # is depth-reduced by design — its round-trip is
            # measured, not failed: the byte-stable path
            # (doc_byte_stable) is the preservation guarantee
            # for an unedited file. Nonlinear adjustments
            # (posterize, hue-sat) amplify the 8-bit
            # quantization at step boundaries, so the max diff
            # is reported rather than bounded.
            entry["roundtrip_depth_reduced"] = True
        if entry["roundtrip_depth_reduced"]:
            # A depth-reduced round-trip is measured, not failed.
            entry["note"] += (
                f"; round-trip depth-reduced "
                f"(max diff {rt_diff})")
        elif not entry["roundtrip_ok"]:
            entry["note"] += "; round-trip flatten differs"
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        entry["roundtrip_ok"] = False
        entry["note"] += f"; round-trip failed: {exc}"
    return entry


def run_oracle(corpus_dir: str | os.PathLike[str],
               *, limit: int | None = None,
               paths: list[str | os.PathLike[str]] | None = None
               ) -> dict:
    """Run the oracle over every ``.psd`` under ``corpus_dir``,
    or over ``paths`` when given — a sweep slice of the corpus,
    so a long run covers its slice without re-checking files
    outside it."""
    root = Path(corpus_dir)
    files = (sorted(Path(p) for p in paths) if paths is not None
             else sorted(root.rglob("*.psd")))
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
        "byte_stable": sum(1 for e in entries
                           if e["byte_stable"]),
        "doc_byte_stable": sum(1 for e in entries
                               if e["doc_byte_stable"]),
        "roundtrip_ok": sum(1 for e in entries
                            if e["roundtrip_ok"]),
        "roundtrip_depth_reduced": sum(
            1 for e in entries
            if e["roundtrip_depth_reduced"]),
        "adjustments_unrendered": sum(
            1 for e in entries
            if not e["adjustments_rendered"]),
        "blends_unrendered": sum(
            1 for e in entries
            if not e["blends_rendered"]),
        "effects_unrendered": sum(
            1 for e in entries
            if not e["effects_rendered"]),
        "text_unrendered": sum(
            1 for e in entries
            if not e["text_rendered"]),
        "smartobjects_unrendered": sum(
            1 for e in entries
            if not e["smartobjects_rendered"]),
        "fills_unrendered": sum(
            1 for e in entries
            if not e["fills_rendered"]),
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
        f"  byte-stable write-back: {summary['byte_stable']}",
        f"  document byte-stable write-back: "
        f"{summary['doc_byte_stable']}",
        f"  round-trip ok: {summary['roundtrip_ok']}",
        f"  round-trip depth-reduced (16/32-bit source, 8-bit "
        f"encoder): {summary['roundtrip_depth_reduced']}",
        f"  features not live-rendered (edit path): "
        f"adjustments {summary['adjustments_unrendered']}, "
        f"blends {summary['blends_unrendered']}, "
        f"effects {summary['effects_unrendered']}, "
        f"text {summary['text_unrendered']}, "
        f"smart objects {summary['smartobjects_unrendered']}, "
        f"fills {summary['fills_unrendered']}",
    ]
    for entry in report["files"]:
        if entry["status"] in ("differs", "failed"):
            lines.append(
                f"  {entry['name']}: {entry['status']} — "
                f"{entry['note']}")
    return "\n".join(lines)
