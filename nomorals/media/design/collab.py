"""FigCraft collab — the multiplayer core (lane 105).

Follows upstream's collab protocol (``docs/collab-protocol.md``,
``crates/collab``) in Devon's Python:

* The document is objects with properties. A change says
  "property P of object O is now V" — ``Create`` / ``Set`` /
  ``Delete`` for nodes, ``SetDoc`` / ``Put`` / ``Remove`` for
  document-level data. Values are JSON.
* Tree structure is two properties on the child: ``parent`` (an
  object id) and ``position`` (a fractional key). A parent's
  children are its children sorted by ``(position, id)``, so two
  clients inserting into the same frame never conflict, and a
  move is one property write. The server rejects a ``parent``
  write that would create a cycle or point at a deleted object.
* Ordering is the server's: conflicting writes to the same
  property resolve last-writer-wins in server order. Clients
  apply their own edits optimistically and reconcile when the
  server echoes or overrides them.
* Keys are fractional-index strings over a 62-digit alphabet
  (``0-9A-Za-z``), ordered lexicographically: ``between(a, b)``
  always finds a key strictly between two adjacent keys, and a
  parent whose keys would grow past 48 characters is re-spaced.
  (Upstream's ``frac`` is base-62 digits after a length-coded
  integer part; Devon's keys use the same alphabet and the same
  ordering guarantees, with rational midpoint generation.)

``Replica`` holds the synced state and applies batches strictly
(stop at the first invalid change — the server's mode) or
leniently (skip what no longer applies — a client replaying its
pending edits). ``History`` keeps per-user undo/redo and can
revert to any sequence number.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .errors import DesignError
from .model import _canonical

__all__ = [
    "Change", "Replica", "History", "apply_batch", "diff",
    "frac_between", "frac_first", "MAX_KEY_LEN",
]

_DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
MAX_KEY_LEN = 48

_NODE_PROPS = frozenset({
    "name", "type", "parent", "position", "transform", "width", "height",
    "fills", "strokes", "stroke", "corners", "clip", "layout",
    "child_layout", "characters", "style", "component", "source",
    "props", "overrides", "bound", "effects", "guides", "interactions",
    "overflow", "background", "flows", "prototype", "contents_hidden",
    "count", "inner_ratio", "network", "description",
})

Change = dict[str, Any]


def _val(char: str) -> int:
    return _DIGITS.index(char)


def _ch(value: int) -> str:
    return _DIGITS[value]


def _before(key: str) -> str:
    """A key strictly smaller than ``key`` (non-empty)."""
    if not key:
        raise DesignError("no key sorts before the empty key",
                          kind="invalid_argument")
    head = _val(key[0])
    if head > 0:
        return _ch(head - 1)
    if len(key) == 1:
        raise DesignError("no key sorts before a lone smallest digit",
                          kind="invalid_argument")
    return "0" * (len(key) - 1)


def frac_first() -> str:
    """The first key in a fresh ordering (the alphabet's middle)."""
    return "V"


def frac_between(before: str | None, after: str | None) -> str:
    """A key strictly between ``before`` and ``after`` (either may be None).

    ``None`` before means "before everything", ``None`` after means
    "after everything". Raises DesignError for the one pair that
    has no key between it — a key and that key plus a lone smallest
    digit — which the generator below never produces.
    """
    if before is None and after is None:
        return frac_first()
    if before is None:
        return _before(after)  # type: ignore[arg-type]
    if after is None:
        return before + "V"
    if not before < after:
        raise DesignError(f"frac keys out of order: {before!r} >= {after!r}",
                          kind="invalid_argument")
    i = 0
    while i < len(before) and i < len(after) and before[i] == after[i]:
        i += 1
    prefix = before[:i]
    low = before[i] if i < len(before) else None
    high = after[i] if i < len(after) else None
    if low is None:
        # ``before`` is a prefix of ``after``.
        d = _val(high)  # type: ignore[arg-type]
        if d > 1:
            return before + _ch(d // 2)
        if d == 1:
            return before + "0V"
        # ``after`` is ``before`` + "0" + rest: recurse past the zero.
        return before + "0" + _before(after[i + 1:])
    # Both have a digit here and low < high (before < after).
    if _val(high) - _val(low) > 1:  # type: ignore[arg-type]
        return prefix + _ch((_val(low) + _val(high)) // 2)  # type: ignore[arg-type]
    return before + "V"


def _respread(count: int) -> list[str]:
    """``count`` evenly spaced keys (re-spacing a crowded parent)."""
    keys: list[str] = []
    for i in range(1, count + 1):
        frac = i / (count + 1)
        digits: list[str] = []
        value = frac
        for _ in range(12):
            value *= 62
            digit = int(value)
            digits.append(_ch(digit))
            value -= digit
            if value == 0:
                break
        keys.append("".join(digits) or "0")
    return keys


def _default_node(node_type: str, nid: int) -> dict[str, Any]:
    node: dict[str, Any] = {"id": nid, "name": node_type, "type": node_type,
                            "parent": None, "children": [],
                            "transform": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                            "width": 0.0, "height": 0.0}
    if node_type == "text":
        node["characters"] = ""
        node["style"] = {"font_family": "Inter", "font_style": "Regular",
                         "font_size": 16.0,
                         "line_height": {"unit": "percent", "value": 140.0},
                         "letter_spacing": {"unit": "percent", "value": 0.0},
                         "decoration": "none", "case": "original",
                         "fills": None, "link": None}
    return node


class Replica:
    """The synced state of one document: nodes + fractional keys."""

    def __init__(self, doc: dict[str, Any]):
        self._doc = deepcopy(doc)
        self.nodes: dict[int, dict[str, Any]] = {
            int(k): deepcopy(v) for k, v in doc.get("nodes", {}).items()}
        self.keys: dict[int, dict[int, str]] = {}
        self.deleted: set[int] = set()
        for node in self.nodes.values():
            parent = node.get("parent")
            if parent is not None:
                self.keys.setdefault(int(parent), {})
        for node in self.nodes.values():
            parent = node.get("parent")
            if parent is None:
                continue
            siblings = [int(c) for c in
                        self.nodes[int(parent)].get("children", [])]
            spread = _respread(len(siblings))
            for child_id, key in zip(siblings, spread, strict=True):
                self.keys[int(parent)][child_id] = key
        self.next_id = int(doc.get("next_id", 1))
        self._counters: dict[int, int] = {}

    # -- ids ----------------------------------------------------------
    def alloc_id(self, client_id: int) -> int:
        space = int(client_id) << 32
        highest = self._counters.get(int(client_id), -1)
        for nid in list(self.nodes) + list(self.deleted):
            if (nid >> 32) == int(client_id) and (nid & 0xFFFFFFFF) > highest:
                highest = nid & 0xFFFFFFFF
        counter = highest + 1
        self._counters[int(client_id)] = counter
        fresh = space | counter
        if fresh >= self.next_id:
            self.next_id = fresh + 1
        return fresh

    # -- materialisation ----------------------------------------------
    def materialize(self) -> dict[str, Any]:
        data = deepcopy(self._doc)
        nodes: dict[str, Any] = {}
        for nid, node in self.nodes.items():
            copy = deepcopy(node)
            copy.pop("position", None)
            children = [c for c in self.nodes
                        if self.nodes[c].get("parent") == nid]
            order = self.keys.get(nid, {})
            children.sort(key=lambda c: (order.get(c, "V"), c))
            copy["children"] = children
            nodes[str(nid)] = copy
        data["nodes"] = nodes
        data["next_id"] = self.next_id
        return data

    # -- validation ---------------------------------------------------
    def _is_descendant(self, nid: int, ancestor: int) -> bool:
        seen: set[int] = set()
        current = nid
        while current is not None and current not in seen:
            seen.add(current)
            if current == ancestor:
                return True
            node = self.nodes.get(current)
            current = None if node is None else node.get("parent")
            current = None if current is None else int(current)
        return False

    def _validate(self, change: Change) -> str | None:
        """Return an error reason, or None when the change applies."""
        kind = change.get("kind")
        if kind == "create":
            nid = int(change["id"])
            if nid in self.nodes:
                return f"node {nid} already exists"
            parent = change.get("parent")
            if parent is None:
                return "create needs a parent"
            if int(parent) not in self.nodes:
                return f"parent {parent} does not exist"
            if int(parent) == nid:
                return "a node cannot be its own parent"
            return None
        if kind == "set":
            nid = int(change["id"])
            if nid not in self.nodes:
                return f"node {nid} does not exist"
            prop = change.get("prop")
            if prop not in _NODE_PROPS:
                return f"unknown property {prop!r}"
            if prop == "parent":
                new_parent = change.get("value")
                if new_parent is None:
                    return "parent cannot be None for a non-page node"
                if int(new_parent) not in self.nodes:
                    return f"parent {new_parent} does not exist"
                if int(new_parent) == nid or self._is_descendant(int(new_parent), nid):
                    return "parent write would create a cycle"
            return None
        if kind == "delete":
            return None  # deleting what is already gone is a no-op
        if kind in ("set_doc", "put", "remove"):
            return None
        return f"unknown change kind {kind!r}"

    def _fix_for(self, change: Change, reason: str) -> Change:
        """The current value to roll a rejected change back to."""
        kind = change.get("kind")
        if kind == "set":
            nid = int(change["id"])
            node = self.nodes.get(nid)
            value = None if node is None else node.get(change.get("prop"))
            return {"kind": "set", "id": nid, "prop": change.get("prop"),
                    "value": deepcopy(value), "reason": reason}
        if kind == "create":
            return {"kind": "delete", "id": int(change["id"]),
                    "reason": reason}
        return {"kind": "noop", "reason": reason}

    # -- applying -----------------------------------------------------
    def _apply_one(self, change: Change) -> None:
        kind = change.get("kind")
        if kind == "create":
            nid = int(change["id"])
            node = _default_node(str(change.get("node_type", "rectangle")), nid)
            node.update(deepcopy(change.get("props") or {}))
            node["name"] = (change.get("props") or {}).get("name", node["type"])
            node["parent"] = int(change["parent"])
            node["children"] = []
            self.nodes[nid] = node
            self.deleted.discard(nid)
            position = change.get("position")
            if position is None:
                siblings = self.keys.setdefault(int(change["parent"]), {})
                position = frac_between(
                    max(siblings.values()) if siblings else None, None)
            self.keys.setdefault(int(change["parent"]), {})[nid] = str(position)
            self._maybe_respacing(int(change["parent"]))
            return
        if kind == "set":
            nid = int(change["id"])
            node = self.nodes.get(nid)
            if node is None:
                return
            prop = change.get("prop")
            if prop == "parent":
                old_parent = node.get("parent")
                new_parent = int(change["value"])
                if old_parent is not None:
                    self.keys.get(int(old_parent), {}).pop(nid, None)
                node["parent"] = new_parent
                position = change.get("position")
                if position is None:
                    siblings = self.keys.setdefault(new_parent, {})
                    position = frac_between(
                        max(siblings.values()) if siblings else None, None)
                self.keys.setdefault(new_parent, {})[nid] = str(position)
                self._maybe_respacing(new_parent)
            elif prop == "position":
                parent = node.get("parent")
                if parent is not None:
                    self.keys.setdefault(int(parent), {})[nid] = str(change["value"])
                    self._maybe_respacing(int(parent))
            else:
                node[prop] = deepcopy(change.get("value"))
            return
        if kind == "delete":
            nid = int(change["id"])
            if nid not in self.nodes:
                return
            subtree = [nid] + self._subtree(nid)
            for victim in subtree:
                parent = self.nodes[victim].get("parent")
                if parent is not None:
                    self.keys.get(int(parent), {}).pop(victim, None)
                self.deleted.add(victim)
                del self.nodes[victim]
            return
        if kind == "set_doc":
            self._doc[change["prop"]] = deepcopy(change.get("value"))
            return
        if kind == "put":
            entry = str(change["entry"])
            bucket, _, key = entry.partition(":")
            if bucket == "style":
                styles = self._doc.setdefault("styles", [])
                self._put_by_id(styles, key, change.get("value"))
            elif bucket == "comment":
                comments = self._doc.setdefault("comments", [])
                self._put_by_id(comments, key, change.get("value"))
            return
        if kind == "remove":
            entry = str(change["entry"])
            bucket, _, key = entry.partition(":")
            target = self._doc.get("styles" if bucket == "style" else "comments", [])
            self._doc["styles" if bucket == "style" else "comments"] = [
                item for item in target
                if str(item.get("id")) != key]

    @staticmethod
    def _put_by_id(items: list[Any], key: str, value: Any) -> None:
        for i, item in enumerate(items):
            if isinstance(item, dict) and str(item.get("id")) == key:
                items[i] = deepcopy(value)
                return
        items.append(deepcopy(value))

    def _subtree(self, nid: int) -> list[int]:
        out: list[int] = []
        for other, node in self.nodes.items():
            if node.get("parent") == nid:
                out.append(other)
                out.extend(self._subtree(other))
        return out

    def _maybe_respacing(self, parent: int) -> None:
        siblings = self.keys.get(parent, {})
        if any(len(k) > MAX_KEY_LEN for k in siblings.values()):
            ordered = sorted(siblings, key=lambda c: (siblings[c], c))
            for child_id, key in zip(ordered, _respread(len(ordered)),
                                     strict=True):
                siblings[child_id] = key

    def apply(self, batch: list[Change], strict: bool = True
              ) -> list[Change]:
        """Apply a batch; return fixes for what was rejected.

        Strict (the server's mode): the whole batch is validated
        first — one invalid change rejects the batch with a fix
        naming the current value, and nothing is applied. Lenient
        (a client replaying pending edits): invalid changes are
        skipped, the rest apply.
        """
        if strict:
            for change in batch:
                reason = self._validate(change)
                if reason is not None:
                    return [self._fix_for(change, reason)]
            for change in batch:
                self._apply_one(change)
            return []
        fixes: list[Change] = []
        for change in batch:
            reason = self._validate(change)
            if reason is not None:
                # Lenient: skip what no longer applies, silently.
                continue
            self._apply_one(change)
        return fixes

    def inverse(self, batch: list[Change]) -> list[Change]:
        """The batch that undoes ``batch`` against current state."""
        out: list[Change] = []
        for change in reversed(batch):
            kind = change.get("kind")
            if kind == "set":
                node = self.nodes.get(int(change["id"]))
                value = None if node is None else node.get(change.get("prop"))
                out.append({"kind": "set", "id": int(change["id"]),
                            "prop": change.get("prop"),
                            "value": deepcopy(value)})
            elif kind == "create":
                out.append({"kind": "delete", "id": int(change["id"])})
            elif kind == "delete":
                node = self.nodes.get(int(change["id"]))
                if node is not None:
                    out.append({"kind": "create", "id": int(change["id"]),
                                "node_type": node.get("type"),
                                "parent": node.get("parent"),
                                "position": self.keys.get(
                                    int(node.get("parent") or 0), {}).get(
                                        int(change["id"])),
                                "props": deepcopy(node)})
            elif kind == "set_doc":
                out.append({"kind": "set_doc", "prop": change.get("prop"),
                            "value": deepcopy(self._doc.get(change.get("prop")))})
        return out

    # -- diff ---------------------------------------------------------
    def diff(self, other: dict[str, Any]) -> list[Change]:
        """The batch that turns this replica into ``other`` (file form)."""
        other_nodes = {int(k): v for k, v in other.get("nodes", {}).items()}
        mine = self.nodes
        batch: list[Change] = []
        # Creates first (parents before children), in file order.
        created: list[int] = []
        for nid in sorted(other_nodes, key=lambda n: (other_nodes[n].get("parent") is not None, n)):
            if nid not in mine and nid not in self.deleted:
                created.append(nid)
        # Order creates so parents come before their children.
        ordered: list[int] = []
        pending = set(created)
        while pending:
            progressed = False
            for nid in sorted(pending):
                parent = other_nodes[nid].get("parent")
                if parent is None or int(parent) not in pending:
                    ordered.append(nid)
                    pending.discard(nid)
                    progressed = True
            if not progressed:  # cycle in the new data; take the rest as-is
                ordered.extend(sorted(pending))
                break
        # Positions for new nodes: keys between their new siblings.
        sibling_keys: dict[int, dict[int, str]] = {}
        for nid in ordered:
            parent = other_nodes[nid].get("parent")
            node = other_nodes[nid]
            position: str | None = None
            if parent is not None:
                parent = int(parent)
                siblings = [int(c) for c in other_nodes[parent].get("children", [])]
                keys_here = sibling_keys.setdefault(parent, dict(self.keys.get(parent, {})))
                idx = siblings.index(nid)
                prev_key = None
                next_key = None
                for j in range(idx - 1, -1, -1):
                    if siblings[j] in keys_here:
                        prev_key = keys_here[siblings[j]]
                        break
                for j in range(idx + 1, len(siblings)):
                    if siblings[j] in keys_here:
                        next_key = keys_here[siblings[j]]
                        break
                position = frac_between(prev_key, next_key)
                keys_here[nid] = position
            batch.append({"kind": "create", "id": nid,
                          "node_type": node.get("type", "rectangle"),
                          "parent": parent, "position": position,
                          "props": {k: deepcopy(v) for k, v in node.items()
                                    if k not in ("id", "parent", "children")}})
        # Moves and sets for shared nodes.
        for nid, node in other_nodes.items():
            if nid not in mine or nid in created:
                continue
            mine_node = mine[nid]
            if node.get("parent") != mine_node.get("parent"):
                batch.append({"kind": "set", "id": nid, "prop": "parent",
                              "value": node.get("parent")})
            for key, value in node.items():
                if key in ("id", "parent", "children"):
                    continue
                if _canonical(mine_node.get(key)) != _canonical(value):
                    batch.append({"kind": "set", "id": nid, "prop": key,
                                  "value": deepcopy(value)})
        # Position changes where the sibling order moved.
        for nid, node in other_nodes.items():
            if nid not in mine or nid in created:
                continue
            parent = node.get("parent")
            if parent is None:
                continue
            siblings = [int(c) for c in other_nodes[int(parent)].get("children", [])]
            idx = siblings.index(nid)
            my_keys = self.keys.get(int(parent), {})
            my_order = sorted(my_keys, key=lambda c: (my_keys[c], c))
            if idx < len(my_order) and my_order[idx] != nid:
                prev_key = None
                next_key = None
                for j in range(idx - 1, -1, -1):
                    if siblings[j] in my_keys and siblings[j] != nid:
                        prev_key = my_keys[siblings[j]]
                        break
                for j in range(idx + 1, len(siblings)):
                    if siblings[j] in my_keys and siblings[j] != nid:
                        next_key = my_keys[siblings[j]]
                        break
                batch.append({"kind": "set", "id": nid, "prop": "position",
                              "value": frac_between(prev_key, next_key)})
        # Deletes last (one per topmost deleted subtree).
        gone = [nid for nid in mine if nid not in other_nodes]
        gone_set = set(gone)
        for nid in sorted(gone):
            parent = mine[nid].get("parent")
            if parent is None or int(parent) not in gone_set:
                batch.append({"kind": "delete", "id": nid})
        # Document-level data.
        for key in ("name", "current_page", "styles", "variables",
                    "comments", "document_colors", "imageIndex"):
            if _canonical(self._doc.get(key)) != _canonical(other.get(key)):
                batch.append({"kind": "set_doc", "prop": key,
                              "value": deepcopy(other.get(key))})
        return batch


def apply_batch(replica: Replica, batch: list[Change],
                strict: bool = True) -> list[Change]:
    """Apply ``batch`` to ``replica``; return fixes for rejections."""
    return replica.apply(batch, strict=strict)


def diff(replica: Replica, other: dict[str, Any]) -> list[Change]:
    """The batch that turns ``replica`` into ``other`` (file form)."""
    return replica.diff(other)


class History:
    """Per-user undo/redo + revert-to-sequence over recorded batches."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []
        self._undo: dict[str, list[dict[str, Any]]] = {}
        self._redo: dict[str, list[dict[str, Any]]] = {}

    def record(self, seq: int, author: str, batch: list[Change],
               inverse: list[Change]) -> None:
        entry = {"seq": seq, "author": author,
                 "batch": deepcopy(batch), "inverse": deepcopy(inverse)}
        self._entries.append(entry)
        self._undo.setdefault(author, []).append(entry)
        self._redo.setdefault(author, []).clear()

    def undo_for(self, author: str) -> list[Change] | None:
        stack = self._undo.get(author, [])
        if not stack:
            return None
        entry = stack.pop()
        self._redo.setdefault(author, []).append(entry)
        return deepcopy(entry["inverse"])

    def redo_for(self, author: str) -> list[Change] | None:
        stack = self._redo.get(author, [])
        if not stack:
            return None
        entry = stack.pop()
        self._undo.setdefault(author, []).append(entry)
        return deepcopy(entry["batch"])

    def revert_to(self, seq: int) -> list[list[Change]]:
        """Compensating batches that rewind state to ``seq``."""
        out: list[list[Change]] = []
        for entry in reversed(self._entries):
            if entry["seq"] > seq:
                out.append(deepcopy(entry["inverse"]))
        return out

    @property
    def entries(self) -> list[dict[str, Any]]:
        return deepcopy(self._entries)
