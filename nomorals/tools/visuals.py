"""Scene tool: ``render_scene`` — the brain's visual surface.

Devon composes a structured scene spec herself whenever an answer
is spatial — a diagram, a layout, a composition — and renders it
into a deliverable she can use: a PNG image for her own image
outputs, a live HTML page, or terminal text. The validated spec
comes back in the result, so the next stage (the code-video rung)
consumes it without re-authoring it.

The tool is a thin surface over :mod:`nomorals.visuals`; the organ
owns the spec, the renderers, and the profile pick.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.policy import Capability
from ..visuals.render import render_scene as _render_scene

__all__ = ["register"]


def register(registry: Any) -> None:
    """Attach the ``render_scene`` tool to a registry."""
    context = registry.context

    @registry.register(
        "render_scene",
        description=(
            "Render a structured scene spec (diagram, overlay, or 3D "
            "scene) into a deliverable: a PNG image, a live HTML page, "
            "or terminal text. Devon composes the spec herself whenever "
            "an answer is spatial — a diagram, a layout, a composition — "
            "and uses the rendered output in her own images and videos. "
            "target= auto|live|chat|terminal; auto sends diagrams and "
            "overlays to PNG and 3D scenes to the live page. Returns the "
            "file path and the validated spec for the next stage."
        ),
        capability=Capability.MEDIA,
    )
    def render_scene_tool(spec: dict | None = None,
                          target: str = "auto",
                          title: str = "",
                          out_dir: str = "") -> dict[str, Any]:
        if not isinstance(spec, dict):
            raise ValueError("render_scene needs a spec dict")
        out: str | None = None
        if out_dir:
            out = out_dir
        else:
            settings = (getattr(context, "settings", None)
                        if context is not None else None)
            home = (getattr(settings, "home", None)
                    if settings is not None else None)
            if home:
                out = str(Path(home) / "visuals")
        return _render_scene(spec, target=target, title=title,
                             out_dir=out)
