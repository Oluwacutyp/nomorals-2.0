"""CAD — computer-aided design and drafting (CadCraft port).

Devon's CAD organ: a real CAD document model (sketches, lines/arcs/
circles/ellipses/splines/points/polylines/text, layers, blocks,
dimensions, constraints) in upstream's ``.cadcraft`` zip shape, a
real constraint solver ported from upstream's ``crates/constraints``
(flat parameter vector, residual forms per constraint kind, damped
weighted minimum-norm Gauss–Newton, DOF from the Jacobian's rank and
null space), typed command state machines with per-step panic
guards, real drafting operations (trim, extend, offset, fillet,
chamfer, mirror, array) computed from real geometry, a sketch
renderer to PNG, real DXF import/export, and machine-checked parity
with SHA-256 attribution.

Port of the *design* of storytold/cadcraft (an open-source,
clean-room CAD application in Rust, dual MIT/Apache-2.0, upstream
commit 2099288f5449b1602d2e2eedca92fe99fb19b973). The code here is
Devon's own Python; the design, the solver algorithm and the format
shape follow upstream, and every ported unit is pinned in
``cad-attribution.toml``.
"""

from __future__ import annotations

from .attribution import load_attribution, verify_attribution
from .codec import (
    export_dxf,
    import_drawing,
    import_dxf,
    parse_document,
    to_document_bytes,
)
from .commands import CommandSession, Guarded, command_specs, list_commands
from .edit import CadEditor
from .errors import CadError, CadErrorKind
from .geom import (
    circle_circle,
    line_circle,
    line_line,
)
from .model import (
    Arc,
    CadDocument,
    CadPoint,
    CadText,
    Circle,
    Constraint,
    ConstraintKind,
    Dimension,
    Ellipse,
    GeomRef,
    Insert,
    Layer,
    Line,
    Polyline,
    Spline,
    Sub,
)
from .parity import run_parity
from .render import render_document, render_to_image
from .solver import dof_report, measure, purge_invalid, solve, validate_constraint

__all__ = [
    "Arc", "CadDocument", "CadEditor", "CadError", "CadErrorKind",
    "CadPoint", "CadText", "Circle", "CommandSession", "Constraint",
    "ConstraintKind", "Dimension", "Ellipse", "GeomRef", "Guarded",
    "Insert", "Layer", "Line", "Polyline", "Spline", "Sub",
    "circle_circle", "command_specs", "dof_report", "export_dxf",
    "import_drawing", "import_dxf", "line_circle", "line_line",
    "list_commands", "load_attribution", "measure", "parse_document",
    "purge_invalid", "render_document", "render_to_image",
    "run_parity", "solve", "to_document_bytes", "validate_constraint",
    "verify_attribution",
]
