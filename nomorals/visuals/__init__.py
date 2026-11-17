"""Scene-spec organ: the brain's visual surface.

Devon composes a structured scene spec and uses it to drive her
own outputs — a PNG image, a live HTML page, a scene handed to the
code-video rung — instead of only prompting a diffusion model with
text and hoping. The display ladder (live page, PNG snapshot,
terminal text) is the second half, co-equal: the Clicky approach,
kept, with the organ built natively underneath it.

The spec is the interchange format. Everything in this package is
pure data in, deliverable out.
"""

from __future__ import annotations

from .codegraph import callgraph_scene
from .impossible import dossier_scene, render_dossier_png
from .profile import pick_target
from .render import (
    default_output_dir,
    render_live_html,
    render_png,
    render_scene,
    render_text,
    write_live_page,
)
from .spec import (
    SceneSpecError,
    auto_layout,
    diagram_from_parts,
    validate_scene_spec,
)

__all__ = [
    "SceneSpecError",
    "auto_layout",
    "callgraph_scene",
    "default_output_dir",
    "diagram_from_parts",
    "dossier_scene",
    "pick_target",
    "render_dossier_png",
    "render_live_html",
    "render_png",
    "render_scene",
    "render_text",
    "validate_scene_spec",
    "write_live_page",
]
