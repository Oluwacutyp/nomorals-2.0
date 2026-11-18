"""Image-optimisation tool: the Squoosh rung for the brain.

"Compress this for WhatsApp" over the Studio — not a standalone
tool. The ``optimize`` action runs the codec ladder (Pillow
codecs first; ffmpeg AVIF / heif-enc where the box provides
them), measures every trial (bytes, dimensions held, PSNR where
actually computed) and recommends the best trial under the
target's constraint, with the full trial table shown. The organ
(:mod:`nomorals.media.imggen.optimize`) owns the ladder; the
Studio owns the document flow; this module is only the surface
between the brain and the Studio.

The optimised image is written as a Studio image document
(``<same stem>.json`` beside the file), so Studio work can
continue from it: more ops, undo, branch, re-render. The file's
actual on-disk size is reported, not the trial's measured size
(they agree; a mismatch is an error, never papered over).
``action=status`` reports which codec rungs this machine can
actually run and why the others are unavailable.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..core.policy import Capability
from ..media.imggen.optimize import document_from_report, optimize_image

__all__ = ["register"]


def _home_of(context: Any) -> str | None:
    settings = (getattr(context, "settings", None)
                if context is not None else None)
    home = (getattr(settings, "home", None)
            if settings is not None else None)
    return str(home) if home else None


def register(registry: Any) -> None:
    """Attach the ``optimize_image`` tool."""
    context = registry.context

    @registry.register(
        "optimize_image",
        description=(
            "Optimise (compress) a photo for a destination: runs "
            "the codec ladder (JPEG, WebP, PNG, AVIF where the box "
            "provides it), measures every trial (bytes, dimensions, "
            "PSNR where actually computed) and saves the best trial "
            "under the target's constraint, as a Studio image "
            "document the owner can keep editing. Call this when "
            "the owner sends a photo and says 'compress this for "
            "WhatsApp', 'make this smaller', 'optimise this for "
            "the web' or 'shrink this'. action=optimize takes "
            "path=<the saved photo path>; target=whatsapp (default, "
            "~1 MB, longest edge 1920, JPEG preferred) | telegram "
            "(10 MB) | web (500 KB, WebP preferred) | archive (no "
            "cap, smallest lossless) | smallest (smallest file "
            "with PSNR >= 30 dB); max_bytes=<optional byte budget "
            "overriding the target's>; out=<optional output path "
            "without an extension — the winning codec's extension "
            "is added; default is <home>/generations/optimized-"
            "<time>>. The answer carries the full trial table and "
            "names the recommended trial and the actual on-disk "
            "size. action=status reports which codec rungs this "
            "machine can run and why the others are unavailable."
        ),
        capability=Capability.MEDIA,
    )
    def optimize_image_tool(action: str = "optimize", path: str = "",
                            out: str = "", target: str = "whatsapp",
                            max_bytes: int = 0) -> dict[str, Any]:
        verb = (action or "optimize").strip().lower()
        if verb == "status":
            from ..media.imggen.optimize import optimize_status

            result = optimize_status()
            result["ok"] = True
            result["action"] = "status"
            return result
        if verb != "optimize":
            raise ValueError(
                f"unknown optimize_image action {action!r}; "
                "use optimize|status")
        src = (path or "").strip()
        if not src:
            raise ValueError(
                "optimize_image needs a photo path — pass "
                "path=<the saved photo>")
        if not Path(src).exists():
            raise ValueError(f"no such photo: {src}")
        want = (target or "whatsapp").strip().lower()
        budget = int(max_bytes or 0)
        if budget < 0:
            raise ValueError("max_bytes must be >= 0")
        report = optimize_image(src, target=want, max_bytes=budget)
        if report.recommended is None:
            near = (report.trials[report.closest]
                    if report.closest is not None else None)
            detail = ""
            if near is not None:
                detail = (f" — closest is {near.codec} "
                          f"{'q' + str(near.quality) if near.quality is not None else 'lossless'} "
                          f"at {near.size_bytes} bytes")
            raise ValueError(
                f"no trial fits {want} "
                f"(budget {report.target_note}){detail}")
        best = report.trials[report.recommended]
        target_path = (out or "").strip()
        if not target_path:
            home = _home_of(context)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            base = (Path(home) / "generations"
                    if home else Path("."))
            target_path = str(base / f"optimized-{stamp}")
        from ..media.imggen.studio import Studio

        studio = Studio()
        doc = document_from_report(report, max_bytes=budget)
        paths = studio.render_document(
            doc, save_to=target_path,
            prompt=f"optimize for {want}")
        out_file = Path(paths[0])
        doc_path = out_file.with_suffix(".json")
        doc.save(doc_path)
        actual = out_file.stat().st_size
        if actual != best.size_bytes:
            raise ValueError(
                f"optimised file wrote {actual} bytes but the "
                f"winning trial measured {best.size_bytes} — "
                "refusing to report a size that is not on disk")
        return {
            "ok": True,
            "action": "optimize",
            "input": src,
            "output": str(out_file),
            "document": str(doc_path),
            "target": want,
            "target_note": report.target_note,
            "recommended": best.to_dict(),
            "trials": [t.to_dict() for t in report.trials],
            "table": report.format_table(),
            "bytes": actual,
            "source_bytes": report.source_bytes,
        }
