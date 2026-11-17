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

from .annotations import Annotation, AnnotationBoard, get_board
from .bridge import (
    DEFAULT_BRIDGE_PORT,
    SceneBridgeServer,
    bridge_fingerprint,
    load_or_create_token,
    start_bridge,
    stop_bridge,
    token_path,
)
from .codegraph import callgraph_scene
from .flight import flight_duration, flight_frames
from .idle import (
    DEFAULT_IDLE_SECONDS,
    IdleTrigger,
    ensure_visual_idle_job,
    get_idle_trigger,
    run_visual_idle_tick_once,
)
from .impossible import dossier_scene, render_dossier_png
from .interactive import (
    page_for_board,
    render_interactive_page,
    write_interactive_page,
)
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
from .tags import ParsedTags, Tag, parse_tags, strip_tags, tags_to_annotations

__all__ = [
    "Annotation",
    "AnnotationBoard",
    "DEFAULT_BRIDGE_PORT",
    "DEFAULT_IDLE_SECONDS",
    "IdleTrigger",
    "ParsedTags",
    "SceneBridgeServer",
    "SceneSpecError",
    "Tag",
    "auto_layout",
    "bridge_fingerprint",
    "callgraph_scene",
    "default_output_dir",
    "diagram_from_parts",
    "dossier_scene",
    "ensure_visual_idle_job",
    "flight_duration",
    "flight_frames",
    "get_board",
    "get_idle_trigger",
    "load_or_create_token",
    "page_for_board",
    "parse_tags",
    "pick_target",
    "render_dossier_png",
    "render_interactive_page",
    "render_live_html",
    "render_png",
    "render_scene",
    "render_text",
    "run_visual_idle_tick_once",
    "start_bridge",
    "stop_bridge",
    "strip_tags",
    "tags_to_annotations",
    "token_path",
    "validate_scene_spec",
    "write_interactive_page",
    "write_live_page",
]
