"""FigCraft — Devon's collaborative interface-design organ (lane 105).

Ported from storytold/figcraft (dual MIT/Apache-2.0) at upstream
commit 6f78021. The .figcraft format is a ZIP package carrying
``document.json`` (format "figcraft", version 1) plus image parts
and a thumbnail. See ``nomorals/media/design-attribution.toml``
for the per-unit SHA-256 record.
"""

from .attribution import verify_attribution
from .collab import History, Replica, apply_batch, frac_between, frac_first
from .components import detach_instance, instance_props, resolve_instance, variants
from .edit import DesignEditor
from .errors import DesignError, DesignErrorKind
from .layout import compute_layout, measure_text
from .model import DesignDocument
from .parity import run_parity
from .proto import add_interaction, flows, interactions, validate
from .render import render_frame
from .session import SessionClient, SessionServer

__all__ = [
    "DesignDocument",
    "DesignEditor",
    "DesignError",
    "DesignErrorKind",
    "History",
    "Replica",
    "SessionClient",
    "SessionServer",
    "add_interaction",
    "apply_batch",
    "compute_layout",
    "detach_instance",
    "flows",
    "frac_between",
    "frac_first",
    "instance_props",
    "interactions",
    "measure_text",
    "render_frame",
    "resolve_instance",
    "run_parity",
    "validate",
    "variants",
    "verify_attribution",
]
