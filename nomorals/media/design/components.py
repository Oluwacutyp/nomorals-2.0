"""FigCraft components — variants, instances, overrides (lane 105).

Follows upstream's component model (``crates/doc`` + the
components parity doc): a ``componentSet`` holds variant
``component`` children keyed by their ``props``; an ``instance``
names its ``source`` (a component or a componentSet) plus the
``props`` that pick a variant, and carries ``overrides`` —
``{"node": <component-subtree id>, "prop": <field>, "value": ...}``
rows that replace a value inside the resolved tree. Resolution
is honest: an instance with props that match no variant resolves
to the set's first variant and says so; overrides that name a
node or property that does not exist are refused, never silently
dropped.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .errors import DesignError
from .model import DesignDocument

__all__ = ["variants", "resolve_instance", "instance_props", "detach_instance"]


def variants(doc: DesignDocument, component_set_id: int) -> list[dict[str, Any]]:
    """The variant components of a componentSet, in order."""
    node = doc.node(component_set_id)
    if node.get("type") != "componentSet":
        raise DesignError(f"node {component_set_id} is not a componentSet",
                          kind="invalid_argument")
    return [c for c in doc.children(component_set_id)
            if c.get("type") == "component"]


def _pick_variant(doc: DesignDocument, source_id: int,
                  props: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Pick the variant a ``props`` map selects.

    Returns (variant, exact): ``exact`` is False when no variant
    matches and the set's first variant is used as the fallback.
    """
    source = doc.node(source_id)
    if source.get("type") == "component":
        return source, True
    if source.get("type") != "componentSet":
        raise DesignError(f"node {source_id} is not a component source",
                          kind="invalid_argument")
    options = variants(doc, source_id)
    if not options:
        raise DesignError(f"componentSet {source_id} has no variants",
                          kind="invalid_argument")
    for variant in options:
        vprops = variant.get("props") or {}
        if all(str(vprops.get(k)) == str(v) for k, v in props.items()):
            return variant, True
    return options[0], False


def instance_props(doc: DesignDocument, instance_id: int) -> dict[str, Any]:
    """The resolved props of an instance (its own props map)."""
    node = doc.node(instance_id)
    if node.get("type") != "instance":
        raise DesignError(f"node {instance_id} is not an instance",
                          kind="invalid_argument")
    return dict(node.get("props") or {})


def resolve_instance(doc: DesignDocument, instance_id: int) -> dict[str, Any]:
    """Expand an instance into its resolved tree.

    Returns ``{"component": <variant id>, "props": {...},
    "exact": bool, "tree": [node dicts]}`` — the variant's
    subtree, deep-copied, with the instance's overrides applied
    to the matching subtree nodes.
    """
    node = doc.node(instance_id)
    if node.get("type") != "instance":
        raise DesignError(f"node {instance_id} is not an instance",
                          kind="invalid_argument")
    source = node.get("source")
    if source is None:
        raise DesignError(f"instance {instance_id} names no source",
                          kind="invalid_argument")
    variant, exact = _pick_variant(doc, int(source), node.get("props") or {})
    tree: list[dict[str, Any]] = []

    def collect(nid: int) -> None:
        item = deepcopy(doc.node(nid))
        tree.append(item)
        for child in doc.children(nid):
            collect(int(child["id"]))

    collect(int(variant["id"]))
    by_id = {int(n["id"]): n for n in tree}
    for override in node.get("overrides") or []:
        target = by_id.get(int(override.get("node", -1)))
        if target is None:
            raise DesignError(
                f"override names unknown node {override.get('node')}",
                kind="not_found")
        prop = override.get("prop")
        if prop not in target and prop not in ("characters", "fills", "width",
                                               "height", "name", "opacity"):
            raise DesignError(
                f"override names unknown property {prop!r} on node "
                f"{override.get('node')}", kind="invalid_argument")
        target[prop] = deepcopy(override.get("value"))
    return {"component": int(variant["id"]),
            "props": dict(node.get("props") or {}),
            "exact": exact, "tree": tree}


def detach_instance(doc: DesignDocument, instance_id: int) -> int:
    """Replace an instance with plain nodes (its resolved tree).

    The resolved subtree is copied in with fresh ids under the
    instance's parent, at the instance's position; the instance
    node is removed. Returns the new root id.
    """
    node = doc.node(instance_id)
    if node.get("type") != "instance":
        raise DesignError(f"node {instance_id} is not an instance",
                          kind="invalid_argument")
    resolved = resolve_instance(doc, instance_id)
    parent_id = node.get("parent")
    if parent_id is None:
        raise DesignError(f"instance {instance_id} has no parent",
                          kind="invalid_argument")
    doc.mark_dirty()
    parent = doc.node(int(parent_id))
    position = list(parent.get("children") or []).index(instance_id)
    id_map: dict[int, int] = {}
    new_nodes: list[dict[str, Any]] = []
    for item in resolved["tree"]:
        fresh = doc.alloc_id()
        id_map[int(item["id"])] = fresh
        copy = deepcopy(item)
        copy["id"] = fresh
        if copy.get("type") == "component":
            copy["type"] = "frame"
        copy.pop("props", None)
        new_nodes.append(copy)
    for copy in new_nodes:
        old_children = copy.get("children") or []
        copy["children"] = [id_map[int(c)] for c in old_children
                            if int(c) in id_map]
        if copy.get("parent") is not None and int(copy["parent"]) in id_map:
            copy["parent"] = id_map[int(copy["parent"])]
    root_copy = new_nodes[0]
    root_copy["parent"] = int(parent_id)
    root_copy["name"] = node.get("name", root_copy.get("name", "Detached"))
    root_copy["transform"] = deepcopy(node.get("transform") or
                                      [1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    root_copy["width"] = node.get("width", root_copy.get("width"))
    root_copy["height"] = node.get("height", root_copy.get("height"))
    nodes = doc._data.setdefault("nodes", {})
    for copy in new_nodes:
        nodes[str(copy["id"])] = copy
    children = list(parent.get("children") or [])
    children[position] = root_copy["id"]
    parent["children"] = children
    del nodes[str(instance_id)]
    return int(root_copy["id"])
