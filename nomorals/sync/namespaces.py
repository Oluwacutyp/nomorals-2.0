"""Sync namespaces: typed replicated state with visible conflicts.

The :mod:`~nomorals.sync.engine` replicates opaque records with
last-write-wins.  Namespaces sit one level up: a :class:`Namespace` is a
*typed* replicated value — ``map`` / ``list`` / ``counter`` — with merge
semantics matched to the type, so concurrent edits that the record
layer would flatten into one winner instead compose:

* **map** — last-write-wins per key, ordered by ``(timestamp,
  device_id)``.  A same-timestamp write of a *different* value is a
  real conflict: the deterministic winner (higher device id) is kept
  and the loser is recorded in :meth:`Namespace.conflicts` — never
  silently discarded.
* **list** — append-only union by item id.  Concurrent appends are
  *both* kept (a list merge never loses an append).  The same item id
  written with a different value keeps *both* entries, flags them
  conflicted, and records a conflict — the reader sees the fork.
* **counter** — a grow-only per-device counter (G-counter CRDT):
  merge takes the per-device max, so increments from every device
  survive and the total is the sum.  Counters never conflict.

Invariants, enforced by the tests in
``tests/test_remaining_tier3.py``:

1. ``merge()`` never silently discards a conflict — every fork lands
   in :meth:`conflicts` with both sides' values.
2. A list merge never loses an append.
3. ``merge()`` is idempotent: merging the same state twice changes
   nothing the second time (no duplicate list entries, no duplicate
   conflicts).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Namespace",
    "NamespaceSet",
    "Conflict",
    "NAMESPACE_TYPES",
    "merge_namespaces",
]

NAMESPACE_TYPES = ("map", "list", "counter")


@dataclass
class Conflict:
    """One fork the merge could not silently resolve."""

    namespace: str = ""
    kind: str = ""          # "map" | "list"
    key: str = ""           # map key, or list item id
    values: list[Any] = field(default_factory=list)  # both sides, in order
    devices: list[str] = field(default_factory=list)
    resolved_to: Any = None  # the value kept in the namespace
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "kind": self.kind,
            "key": self.key,
            "values": list(self.values),
            "devices": list(self.devices),
            "resolved_to": self.resolved_to,
            "timestamp": self.timestamp,
        }


def _now() -> float:
    return time.time()


class Namespace:
    """One typed replicated value."""

    def __init__(self, name: str, kind: str = "map",
                 *, device_id: str = "local") -> None:
        if kind not in NAMESPACE_TYPES:
            raise ValueError(
                f"unknown namespace type {kind!r}; expected {NAMESPACE_TYPES}")
        if not name:
            raise ValueError("namespace name is required")
        self.name = name
        self.kind = kind
        self.device_id = device_id or "local"
        # map: key -> (value, ts, device)
        self._map: dict[str, tuple[Any, float, str]] = {}
        # list: entries {id, value, ts, device, conflicted}
        self._list: list[dict[str, Any]] = []
        self._list_ids: set[str] = set()
        # counter: device -> count (G-counter)
        self._counter: dict[str, int] = {}
        self._conflicts: list[Conflict] = []
        self._conflict_keys: set[tuple[str, str]] = set()

    # ── local writes ────────────────────────────────────────────────────

    def put(self, key: str, value: Any, *,
            ts: float | None = None, device: str | None = None) -> None:
        """Set a map key. Raises on a non-map namespace."""
        if self.kind != "map":
            raise TypeError(f"put() needs a map namespace, got {self.kind}")
        self._map[str(key)] = (value, ts if ts is not None else _now(),
                               device or self.device_id)

    def append(self, value: Any, *, item_id: str = "",
               ts: float | None = None,
               device: str | None = None) -> str:
        """Append to a list namespace. Returns the item id. Raises on a
        non-list namespace. The id is stable across merges (same id =
        same logical append), so callers that re-append after a crash
        must reuse the id to stay idempotent."""
        if self.kind != "list":
            raise TypeError(f"append() needs a list namespace, got {self.kind}")
        iid = item_id or f"{device or self.device_id}:{_now():.6f}:{len(self._list)}"
        entry = {"id": iid, "value": value,
                 "ts": ts if ts is not None else _now(),
                 "device": device or self.device_id, "conflicted": False}
        self._list.append(entry)
        self._list_ids.add(iid)
        return iid

    def increment(self, amount: int = 1, *,
                  device: str | None = None) -> int:
        """Add to this device's counter shard. Returns the new total."""
        if self.kind != "counter":
            raise TypeError(
                f"increment() needs a counter namespace, got {self.kind}")
        if amount < 0:
            raise ValueError("counter increments must be >= 0 (G-counter)")
        dev = device or self.device_id
        self._counter[dev] = self._counter.get(dev, 0) + int(amount)
        return self.value()

    # ── reads ───────────────────────────────────────────────────────────

    def value(self) -> Any:
        """The current value: dict for map, list of values for list,
        int total for counter."""
        if self.kind == "map":
            return {k: v for k, (v, _ts, _d) in self._map.items()}
        if self.kind == "list":
            return [e["value"] for e in self._list]
        return sum(self._counter.values())

    def entries(self) -> list[dict[str, Any]]:
        """List entries with their metadata (ids, devices, flags)."""
        if self.kind != "list":
            raise TypeError("entries() is for list namespaces")
        return [dict(e) for e in self._list]

    def conflicts(self) -> list[Conflict]:
        """Every conflict this namespace has ever recorded — the full
        fork history, newest last.  Nothing here was silently dropped."""
        return list(self._conflicts)

    # ── merge ───────────────────────────────────────────────────────────

    def merge(self, other: "Namespace") -> list[Conflict]:
        """Merge ``other`` into this namespace. Returns the *new*
        conflicts this merge recorded (``[]`` when the states were
        already consistent).  Idempotent: merging the same ``other``
        twice records each conflict once and changes nothing the
        second time."""
        if other.name != self.name:
            raise ValueError(
                f"cannot merge namespace {other.name!r} into {self.name!r}")
        if other.kind != self.kind:
            raise ValueError(
                f"cannot merge {other.kind!r} into {self.kind!r} namespace")
        if self.kind == "map":
            return self._merge_map(other)
        if self.kind == "list":
            return self._merge_list(other)
        return self._merge_counter(other)

    def _record(self, conflict: Conflict) -> Conflict | None:
        """Record a conflict once per (kind, key, values) — the
        idempotency guard that keeps re-merges from duplicating."""
        marker = (conflict.kind, conflict.key,
                  tuple(repr(v) for v in conflict.values))
        if marker in self._conflict_keys:
            return None
        self._conflict_keys.add(marker)
        self._conflicts.append(conflict)
        return conflict

    def _merge_map(self, other: "Namespace") -> list[Conflict]:
        new: list[Conflict] = []
        for key, (value, ts, device) in other._map.items():
            current = self._map.get(key)
            if current is None:
                self._map[key] = (value, ts, device)
                continue
            cur_value, cur_ts, cur_device = current
            if (ts, device) > (cur_ts, cur_device):
                if ts == cur_ts and value != cur_value:
                    # Same logical time, different values: a genuine
                    # fork. Deterministic winner (higher device id, per
                    # the (ts, device) order) is kept; the loser lives
                    # on in the conflict record.
                    rec = self._record(Conflict(
                        namespace=self.name, kind="map", key=key,
                        values=[cur_value, value],
                        devices=[cur_device, device], resolved_to=value,
                        timestamp=ts))
                    if rec is not None:
                        new.append(rec)
                self._map[key] = (value, ts, device)
            elif ts == cur_ts and value != cur_value and device != cur_device:
                # Incoming lost the tiebreak but disagrees: still a
                # conflict, still recorded — the losing value is data.
                rec = self._record(Conflict(
                    namespace=self.name, kind="map", key=key,
                    values=[cur_value, value],
                    devices=[cur_device, device],
                    resolved_to=cur_value, timestamp=ts))
                if rec is not None:
                    new.append(rec)
        return new

    def _merge_list(self, other: "Namespace") -> list[Conflict]:
        new: list[Conflict] = []
        by_id = {e["id"]: e for e in self._list}
        for entry in other._list:
            mine = by_id.get(entry["id"])
            if mine is None:
                # A genuinely new append — never lost, always kept.
                copied = dict(entry)
                self._list.append(copied)
                self._list_ids.add(copied["id"])
                by_id[copied["id"]] = copied
                continue
            if mine["value"] != entry["value"]:
                # Same append id, different value: a fork. Keep BOTH
                # entries (the append is never lost) and flag them.
                fork = dict(entry)
                fork["conflicted"] = True
                mine["conflicted"] = True
                self._list.append(fork)
                rec = self._record(Conflict(
                    namespace=self.name, kind="list", key=entry["id"],
                    values=[mine["value"], entry["value"]],
                    devices=[mine["device"], entry["device"]],
                    resolved_to=[mine["value"], entry["value"]],
                    timestamp=max(mine["ts"], entry["ts"])))
                if rec is not None:
                    new.append(rec)
                # by_id keeps the first entry; the fork rides alongside
        return new

    def _merge_counter(self, other: "Namespace") -> list[Conflict]:
        for device, count in other._counter.items():
            if count > self._counter.get(device, 0):
                self._counter[device] = count
        return []  # G-counters cannot conflict

    # ── serialisation ───────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "device_id": self.device_id,
            "map": {k: {"value": v, "ts": ts, "device": d}
                    for k, (v, ts, d) in self._map.items()},
            "list": [dict(e) for e in self._list],
            "counter": dict(self._counter),
            "conflicts": [c.to_dict() for c in self._conflicts],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Namespace":
        ns = cls(data["name"], data.get("kind", "map"),
                 device_id=data.get("device_id", "local"))
        for key, item in (data.get("map") or {}).items():
            ns._map[key] = (item["value"], item["ts"], item["device"])
        for entry in data.get("list") or []:
            ns._list.append(dict(entry))
            ns._list_ids.add(entry["id"])
        ns._counter.update(data.get("counter") or {})
        for c in data.get("conflicts") or []:
            ns._conflicts.append(Conflict(
                namespace=c.get("namespace", ns.name),
                kind=c.get("kind", ns.kind), key=c.get("key", ""),
                values=list(c.get("values", [])),
                devices=list(c.get("devices", [])),
                resolved_to=c.get("resolved_to"),
                timestamp=c.get("timestamp", 0.0)))
            ns._conflict_keys.add(
                (c.get("kind", ns.kind), c.get("key", ""),
                 tuple(repr(v) for v in c.get("values", []))))
        return ns


def merge_namespaces(local: Namespace, remote: Namespace) -> list[Conflict]:
    """Merge ``remote`` into ``local`` (in place). Returns new conflicts."""
    return local.merge(remote)


class NamespaceSet:
    """A named collection of namespaces — one per sync domain."""

    def __init__(self, device_id: str = "local") -> None:
        self.device_id = device_id or "local"
        self._namespaces: dict[str, Namespace] = {}

    def get_or_create(self, name: str, kind: str = "map") -> Namespace:
        ns = self._namespaces.get(name)
        if ns is None:
            ns = Namespace(name, kind, device_id=self.device_id)
            self._namespaces[name] = ns
        elif ns.kind != kind:
            raise ValueError(
                f"namespace {name!r} already exists as {ns.kind!r}, "
                f"not {kind!r}")
        return ns

    def merge(self, other: "NamespaceSet") -> list[Conflict]:
        """Merge every namespace in ``other`` into this set."""
        new: list[Conflict] = []
        for ns in other._namespaces.values():
            mine = self.get_or_create(ns.name, ns.kind)
            new.extend(mine.merge(ns))
        return new

    def conflicts(self) -> list[Conflict]:
        """All conflicts across all namespaces."""
        out: list[Conflict] = []
        for ns in self._namespaces.values():
            out.extend(ns.conflicts())
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"device_id": self.device_id,
                "namespaces": {n: ns.to_dict()
                               for n, ns in self._namespaces.items()}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NamespaceSet":
        store = cls(data.get("device_id", "local"))
        for name, ns_data in (data.get("namespaces") or {}).items():
            store._namespaces[name] = Namespace.from_dict(ns_data)
        return store
