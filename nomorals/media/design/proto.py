"""FigCraft prototyping — interactions and flows (lane 105).

Upstream's prototype model (``crates/doc`` ``proto.rs``): a node
carries ``interactions`` — ``{"trigger", "action", "destination",
"transition"}`` — and a page carries ``flows`` (named entry
points). This module wires interactions with validation (a
destination must exist), lists them, walks the reachable screens
from a start node, and validates a whole document's prototype
graph, naming every dangling destination instead of failing on
the first.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from .errors import DesignError
from .model import DesignDocument

__all__ = ["add_interaction", "interactions", "flows", "validate"]

_TRIGGERS = frozenset({
    "on-click", "on-drag", "on-hover", "on-press", "on-key",
    "after-delay", "mouse-enter", "mouse-leave", "on-tap",
})
_ACTIONS = frozenset({
    "navigate", "back", "close", "overlay", "swap", "scroll-to",
    "open-url", "set-variable",
})
_TRANSITIONS = frozenset({
    "instant", "push", "slide", "dissolve", "move-in", "move-out",
    "pop", "none",
})


def add_interaction(doc: DesignDocument, node_id: int, trigger: str,
                    action: str, destination: int | None,
                    transition: str = "instant") -> dict[str, Any]:
    """Wire an interaction onto a node (validates as it goes)."""
    if trigger not in _TRIGGERS:
        raise DesignError(f"unknown trigger {trigger!r}", kind="invalid_argument")
    if action not in _ACTIONS:
        raise DesignError(f"unknown action {action!r}", kind="invalid_argument")
    if transition not in _TRANSITIONS:
        raise DesignError(f"unknown transition {transition!r}",
                          kind="invalid_argument")
    node = doc.node(node_id)
    if action in ("navigate", "overlay", "swap", "scroll-to"):
        if destination is None:
            raise DesignError(f"action {action!r} needs a destination",
                              kind="invalid_argument")
        if not doc.has_node(int(destination)):
            raise DesignError(f"interaction destination {destination} "
                              f"does not exist", kind="not_found")
    entry = {"trigger": trigger, "action": action,
             "destination": destination, "transition": transition}
    node.setdefault("interactions", []).append(entry)
    doc.mark_dirty()
    return entry


def interactions(doc: DesignDocument, node_id: int) -> list[dict[str, Any]]:
    """The interactions wired onto a node (possibly empty)."""
    return list(doc.node(node_id).get("interactions") or [])


def flows(doc: DesignDocument, start_id: int) -> list[int]:
    """Screens reachable from ``start_id`` via prototype edges.

    Breadth-first over navigate/overlay/swap/back edges, start
    first, each screen once, in visit order.
    """
    doc.node(start_id)  # raises not_found for a bad start
    seen: list[int] = []
    queue: deque[int] = deque([int(start_id)])
    while queue:
        current = queue.popleft()
        if current in seen:
            continue
        seen.append(current)
        for entry in interactions(doc, current):
            dest = entry.get("destination")
            if dest is not None and int(dest) not in seen:
                queue.append(int(dest))
    return seen


def validate(doc: DesignDocument) -> list[str]:
    """Prototype problems in the whole document (empty = clean)."""
    problems: list[str] = []
    for nid, node in doc.nodes.items():
        for entry in node.get("interactions") or []:
            dest = entry.get("destination")
            if entry.get("action") in ("navigate", "overlay", "swap", "scroll-to"):
                if dest is None:
                    problems.append(
                        f"node {nid}: {entry.get('action')} has no destination")
                elif not doc.has_node(int(dest)):
                    problems.append(
                        f"node {nid}: destination {dest} does not exist")
    for page_id in doc.pages:
        for flow in doc.node(page_id).get("flows") or []:
            start = flow.get("start")
            if start is not None and not doc.has_node(int(start)):
                problems.append(f"page {page_id}: flow start {start} "
                                f"does not exist")
    return problems
