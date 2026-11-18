"""Background-removal tool: the one-shot cutout surface for the brain.

"Remove the background" over the Studio — not a standalone tool.
The remove action builds an :class:`ImageDocument` through
``Studio.remove_background_document``, saves the document (so
Studio work can continue from the cutout: more ops, undo, branch,
re-render), and renders it through ``Studio.render_document``.
The organ (:mod:`nomorals.media.imggen.bgremove`) owns the
composed ladder (rembg model rung where provisioned, then the
classical mask ladder); the Studio owns the document flow; this
module is only the surface between the brain and the Studio.

``action=remove`` takes a photo path and writes a transparent PNG
(the owner sends a photo and says "remove the background"; the
brain passes the saved attachment path). ``action=status`` reports
which rungs this machine can actually run; ``action=models`` lists
the rembg models and the classical rungs with their reasons.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..core.policy import Capability
from ..media.imggen.bgremove import bgremove_status, cutout_coverage

__all__ = ["register"]


def _home_of(context: Any) -> str | None:
    settings = (getattr(context, "settings", None)
                if context is not None else None)
    home = (getattr(settings, "home", None)
            if settings is not None else None)
    return str(home) if home else None


def _rung_from_sidecar(png_path: str) -> tuple[str, str]:
    """Read the rung back off the Studio's own sidecar record
    (``<png>.txt``): ``rung: remove_background: bgremove (<tier>)
    — <detail>``. Defensive — an unreadable sidecar yields "".
    """
    try:
        text = Path(str(png_path) + ".txt").read_text(
            encoding="utf-8")
    except OSError:
        return "", ""
    for line in text.splitlines():
        if not line.startswith("rung: remove_background: "):
            continue
        rest = line[len("rung: remove_background: "):]
        label, _, detail = rest.partition(" — ")
        tier = label.strip()
        if tier.startswith("bgremove (") and tier.endswith(")"):
            tier = tier[len("bgremove ("):-1]
        return tier, detail.strip()
    return "", ""


def register(registry: Any) -> None:
    """Attach the ``remove_background`` tool."""
    context = registry.context

    @registry.register(
        "remove_background",
        description=(
            "Remove the background from a photo: a transparent PNG "
            "cutout of the foreground subject, as a Studio image "
            "document the owner can keep editing. Call this when "
            "the owner sends a photo and says 'remove the "
            "background', 'cut out the subject' or 'make the "
            "background transparent'. action=remove takes path=<the "
            "saved photo path> and writes the cutout (out=<optional "
            "PNG path>; default is <home>/generations/bgcut-<time>"
            ".png) plus the document beside it (<same stem>.json) "
            "so Studio work can continue from the cutout. "
            "tier=auto runs the best rung the machine can run "
            "(rembg model first where provisioned, then the "
            "classical mask ladder: sam, grabcut, flood, "
            "box+feather); tier=rembg|sam|grabcut|flood|box forces "
            "one rung, which refuses by name when it cannot run. "
            "model= picks the rembg model (empty = profile "
            "default). The answer names the rung that served the "
            "cutout and its coverage. action=status reports which "
            "rungs this machine can run and why; action=models "
            "lists the rembg models and the classical rungs."
        ),
        capability=Capability.MEDIA,
    )
    def remove_background_tool(action: str = "remove", path: str = "",
                               out: str = "", tier: str = "auto",
                               model: str = "") -> dict[str, Any]:
        verb = (action or "remove").strip().lower()
        if verb == "status":
            result = bgremove_status()
            result["ok"] = True
            result["action"] = "status"
            return result
        if verb == "models":
            from ..media.imggen.masks import mask_status
            from ..media_edit.segment import list_models

            return {"ok": True, "action": "models",
                    "rembg_models": list_models(),
                    "classical_rungs": mask_status()}
        if verb != "remove":
            raise ValueError(
                f"unknown remove_background action {action!r}; "
                "use remove|status|models")
        src = (path or "").strip()
        if not src:
            raise ValueError(
                "remove_background needs a photo path — pass "
                "path=<the saved photo>")
        if not Path(src).exists():
            raise ValueError(f"no such photo: {src}")
        target = (out or "").strip()
        if not target:
            home = _home_of(context)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            base = (Path(home) / "generations"
                    if home else Path("."))
            target = str(base / f"bgcut-{stamp}.png")
        target_path = Path(target)
        if target_path.suffix.lower() != ".png":
            target_path = target_path.with_suffix(".png")
        from ..media.imggen.studio import Studio

        studio = Studio()
        doc = studio.remove_background_document(
            src, tier=tier or "auto", model=model or "")
        doc_path = target_path.with_suffix(".json")
        doc.save(doc_path)
        paths = studio.render_document(doc, save_to=str(target_path))
        out_png = Path(paths[0])
        rung, detail = _rung_from_sidecar(str(out_png))
        from PIL import Image

        with Image.open(out_png) as rendered:
            coverage = cutout_coverage(rendered)
        return {
            "ok": True,
            "action": "remove",
            "input": src,
            "output": str(out_png),
            "document": str(doc_path),
            "rung": rung,
            "detail": detail,
            "coverage": round(coverage, 4),
            "bytes": out_png.stat().st_size,
        }
