"""ImpossibleMadePossible renders its working.

The module's whole job is making structure visible — text walls
are its weakest surface. Three diagrams come straight out of a
dossier, no hand-authoring: the prior-art map (what exists, what
failed), the constraint diagram (what binds the goal), and the
before/after reframe (the problem as asked, the problem as
achievable). Nodes are laid out by the organ, never placed by hand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .render import render_png
from .spec import diagram_from_parts, validate_scene_spec

__all__ = [
    "dossier_scene",
    "render_dossier_png",
]

_DOSSIER_KINDS = ("prior_art", "constraint", "reframe")


def _short(text: Any, limit: int = 48) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _found_entries(results: dict[str, Any]) -> list[tuple[str, int]]:
    """Per-source found-hit counts, in source order."""
    counts: list[tuple[str, int]] = []
    for name, rows in (results or {}).items():
        if not isinstance(rows, list):
            continue
        hits = sum(1 for row in rows
                   if isinstance(row, dict)
                   and row.get("status") == "found")
        if hits:
            counts.append((str(name), hits))
    return counts


def _prior_art_scene(dossier: dict[str, Any]) -> dict[str, Any]:
    formulation = dossier.get("formulation") or {}
    art = dossier.get("prior_art") or {}
    problem = _short(formulation.get("raw") or "problem", 48)
    nodes = [{"id": "problem", "label": problem, "shape": "circle"}]
    edges: list[dict[str, Any]] = []
    for name, hits in _found_entries(art.get("results") or {}):
        node_id = f"src-{name}"
        nodes.append({"id": node_id,
                      "label": f"{name} · {hits} hit{'s' if hits != 1 else ''}"})
        edges.append({"from": "problem", "to": node_id,
                      "label": f"{hits} hit{'s' if hits != 1 else ''}"})
    failed = art.get("failed_attempts") or []
    failed = [row for row in failed if isinstance(row, dict)]
    if failed:
        nodes.append({"id": "failed",
                      "label": f"failed attempts · {len(failed)}"})
        edges.append({"from": "problem", "to": "failed",
                      "label": "abandoned"})
    if len(nodes) == 1:
        nodes.append({"id": "none",
                      "label": "no prior art found in these sources"})
        edges.append({"from": "problem", "to": "none", "label": "0 hits"})
    return diagram_from_parts(
        f"prior art: {problem}", nodes, edges)


def _constraint_scene(dossier: dict[str, Any]) -> dict[str, Any]:
    formulation = dossier.get("formulation") or {}
    goal = _short(formulation.get("goal_function")
                  or formulation.get("raw") or "goal", 48)
    nodes = [{"id": "goal", "label": goal, "shape": "circle"}]
    edges: list[dict[str, Any]] = []
    constraints = formulation.get("constraints") or []
    for i, constraint in enumerate(constraints):
        node_id = f"c{i}"
        nodes.append({"id": node_id, "label": _short(constraint, 48)})
        edges.append({"from": "goal", "to": node_id,
                      "label": "must satisfy"})
    return diagram_from_parts(f"constraints: {goal}", nodes, edges)


def _reframe_scene(dossier: dict[str, Any]) -> dict[str, Any]:
    formulation = dossier.get("formulation") or {}
    reframed = dossier.get("reframed") or {}
    before = _short(formulation.get("raw") or "problem", 48)
    after = _short(reframed.get("reframed_goal") or "achievable", 48)
    nodes = [
        {"id": "before", "label": before},
        {"id": "after", "label": after, "shape": "circle"},
    ]
    edges = [{"from": "before", "to": "after", "label": "reframe"}]
    ache = _short(reframed.get("ache") or "", 48)
    if ache:
        nodes.append({"id": "ache", "label": f"ache: {ache}"})
        edges.append({"from": "after", "to": "ache",
                      "label": "ache underneath"})
    return diagram_from_parts("reframe: before → after", nodes, edges)


def dossier_scene(dossier: dict[str, Any],
                  kind: str = "prior_art") -> dict[str, Any]:
    """A dossier → a validated diagram spec.

    ``kind`` is ``prior_art`` (the map), ``constraint`` (what binds
    the goal) or ``reframe`` (before/after). The spec is the
    deliverable — render it, send it, or hand it to the video rung.
    """
    if kind not in _DOSSIER_KINDS:
        from .spec import SceneSpecError

        raise SceneSpecError(
            f"scene spec: dossier kind must be one of {_DOSSIER_KINDS}, "
            f"got {kind!r}")
    if kind == "prior_art":
        return _prior_art_scene(dossier)
    if kind == "constraint":
        return _constraint_scene(dossier)
    return _reframe_scene(dossier)


def render_dossier_png(dossier: dict[str, Any], *,
                       kind: str = "prior_art",
                       out_dir: str | Path,
                       size: tuple[int, int] = (1280, 720)) -> str:
    """Render one dossier diagram to a PNG file; return the path.

    The filename is deterministic per claim + kind, so re-rendering
    the same dossier overwrites the same file instead of littering
    the visuals dir.
    """
    spec = dossier_scene(dossier, kind=kind)
    claim = str(dossier.get("claim_id") or "dossier")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-"
                   for ch in claim)[:40] or "dossier"
    target = Path(out_dir) / f"impossible-{kind}-{safe}.png"
    return render_png(spec, target, size=size)


def dossier_diagram_spec(dossier: dict[str, Any],
                         kind: str = "prior_art") -> dict[str, Any]:
    """Alias kept short for callers that only need the spec."""
    return validate_scene_spec(dossier_scene(dossier, kind=kind))
