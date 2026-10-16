"""Commitment ledger -> world-graph projection.

Projects :class:`Commitment` records from
``nomorals.commitments.ledger`` into the
:class:`WorldGraph <nomorals.planning.graph.WorldGraph>` so promises
show up in planning: who they are owed to, when they are due, and
whether they are still open.

Mapping
-------
* One commitment -> one ``commitment`` node, ``external_ref =
  "commitment:<id>"``.  The node label is the promised ``what``.
* ``promisee`` -> one ``person`` node, ``external_ref =
  "person:<promisee>"`` (label = promisee), linked by
  ``commitment --owned_by--> person``.
* ``deadline`` -> one ``deadline`` node, ``external_ref =
  "deadline:<commitment_id>"`` with a ``due_ts`` attr, linked by
  ``commitment --due--> deadline``.
* Non-open commitments (kept / broken / cancelled / superseded) keep
  their node, with ``attrs["status"]`` updated and ``attrs["closed"]``
  set — the node is never deleted, so the graph keeps the history.

Rules (same discipline as the graph itself):

* **Idempotent.** Re-syncing the same commitment returns the same
  node ids and creates no new nodes or edges.  Identity is the
  ``external_ref``; the graph's own dedup is the source of truth and
  this module additionally refreshes attrs on re-sync so status
  changes (open -> kept, ...) land on the existing node.
* **Never raises.** Every public function catches its own failures
  and reports them in its return value (``{"ok": False, ...}``,
  ``0``, or ``[]``).  A broken commitment must never break planning.
* **Projection only.** This module never writes back to the ledger
  (except ``sync_extracted_to_graph``, whose whole job is to call the
  ledger's own ``extract_and_record`` intake) and never rewrites
  graph nodes other than the commitment/person/deadline nodes it owns.
* **Stdlib only, duck-typed.** ``graph``, ``ledger`` and
  ``commitment`` are accepted as duck-typed objects — no imports from
  sibling packages — so this module cannot create an import cycle or
  a layering violation.  Only the standard library is imported.
"""

from __future__ import annotations

import json
import time
from typing import Any

__all__ = [
    "sync_commitment_to_graph",
    "sync_ledger_to_graph",
    "sync_extracted_to_graph",
    "overdue_commitments_in_graph",
]

# Statuses the ledger uses. Anything that is not "open" counts as
# closed for graph purposes.
_OPEN = "open"


# ── small helpers ────────────────────────────────────────────────────

def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` off a dataclass/attrs object or a plain dict."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_float(value: Any) -> float | None:
    """Best-effort float coercion; None when there is no number."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError, OverflowError):
            return None
    if hasattr(value, "timestamp") and callable(value.timestamp):
        try:
            return float(value.timestamp())
        except (TypeError, ValueError, OverflowError):
            return None
    if isinstance(value, str):
        try:
            return float(value.strip())
        except (TypeError, ValueError):
            return None
    return None


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _commitment_attrs(commitment: Any) -> dict[str, Any]:
    """Attrs stored on a commitment node (and refreshed on re-sync)."""
    status = _clean_str(_field(commitment, "status", _OPEN)) or _OPEN
    deadline = _as_float(_field(commitment, "deadline", None))
    attrs: dict[str, Any] = {
        "commitment_id": _clean_str(_field(commitment, "id", "")),
        "what": _clean_str(_field(commitment, "what", "")),
        "promisor": _clean_str(_field(commitment, "promisor", "")),
        "promisee": _clean_str(_field(commitment, "promisee", "")),
        "status": status,
        "closed": status != _OPEN,
        "source": "commitments",
        "language": _clean_str(_field(commitment, "language", "")),
        "source_text": _clean_str(_field(commitment, "source_text", "")),
        "source_message_id": _field(commitment, "source_message_id", None),
        "evidence": _clean_str(_field(commitment, "evidence", "")),
    }
    created = _as_float(_field(commitment, "created_at", None))
    if created is not None:
        attrs["created_at"] = created
    closed_at = _as_float(_field(commitment, "closed_at", None))
    if closed_at is not None:
        attrs["closed_at"] = closed_at
    if deadline is not None:
        attrs["deadline"] = deadline
        attrs["due_ts"] = deadline
    return attrs


def _find_by_external_ref(graph: Any, external_ref: str) -> Any | None:
    """Find a graph node by external_ref via the public list API."""
    if not external_ref:
        return None
    try:
        list_nodes = getattr(graph, "list_nodes", None)
        if not callable(list_nodes):
            return None
        for node in list_nodes() or []:
            if _field(node, "external_ref", "") == external_ref:
                return node
    except Exception:  # noqa: BLE001 — never raise
        return None
    return None


def _persist_attrs(graph: Any, node: Any, attrs: dict[str, Any]) -> Any | None:
    """Persist merged attrs on an existing node; return fresh node or None.

    The graph has no public update API, so the primary path is a
    targeted UPDATE on its SQLite backing store (same package, same
    table the graph itself writes).  The fallback mutates the node
    object in place, which persists for in-memory / stub graphs that
    hand back their stored object.  Either way the caller gets the
    freshest node available.
    """
    node_id = _field(node, "node_id", "")
    if not node_id:
        return None
    merged = dict(_field(node, "attrs", {}) or {})
    merged.update(attrs)
    # Primary: direct UPDATE on the graph's own backing store.
    try:
        db = getattr(graph, "_db", None)
        if db is not None:
            db.execute(
                "UPDATE graph_nodes SET attrs = ?, updated_at = ? "
                "WHERE node_id = ?",
                (json.dumps(merged), time.time(), node_id),
            )
            db.commit()
            get = getattr(graph, "get", None)
            if callable(get):
                fresh = get(node_id)
                if fresh is not None:
                    return fresh
    except Exception:  # noqa: BLE001 — fall through to in-place mutate
        pass
    # Fallback: mutate the node object in place (stub graphs).
    try:
        if isinstance(_field(node, "attrs", None), dict):
            node.attrs.update(attrs)
        else:
            node.attrs = merged
    except Exception:  # noqa: BLE001
        pass
    return node


def _upsert_node(graph: Any, type: str, label: str,
                 attrs: dict[str, Any], external_ref: str) -> Any | None:
    """Idempotent node upsert: reuse by external_ref, refresh attrs."""
    if not external_ref:
        # No stable identity -> plain insert (graph may still dedup).
        try:
            return graph.add_node(type, label, attrs)
        except Exception:  # noqa: BLE001
            return None
    existing = _find_by_external_ref(graph, external_ref)
    if existing is not None:
        current = dict(_field(existing, "attrs", {}) or {})
        # Refresh only what changed (keeps any attrs other writers set).
        changed = {k: v for k, v in attrs.items() if current.get(k) != v}
        # A dropped deadline: remove stale due_ts/deadline so overdue
        # checks never fire on a deadline that no longer exists.
        if "due_ts" not in attrs and "due_ts" in current:
            changed["due_ts"] = None
        if "deadline" not in attrs and "deadline" in current:
            changed["deadline"] = None
        if changed:
            refreshed = _persist_attrs(graph, existing, changed)
            return refreshed if refreshed is not None else existing
        return existing
    try:
        node = graph.add_node(type, label, attrs,
                              external_ref=external_ref)
    except TypeError:
        # Older / stub add_node without external_ref support.
        try:
            node = graph.add_node(type, label, attrs)
        except Exception:  # noqa: BLE001
            return None
    except Exception:  # noqa: BLE001
        return None
    # A race / pre-existing node with the same ref can still return a
    # duplicate-free result via re-lookup; prefer the looked-up node.
    if node is None and external_ref:
        return _find_by_external_ref(graph, external_ref)
    return node


def _add_edge(graph: Any, from_id: str, to_id: str, type: str) -> bool:
    try:
        add_edge = getattr(graph, "add_edge", None)
        if not callable(add_edge):
            return False
        return bool(add_edge(from_id, to_id, type))
    except Exception:  # noqa: BLE001
        return False


# ── public API ───────────────────────────────────────────────────────

def sync_commitment_to_graph(graph: Any, commitment: Any) -> dict[str, Any]:
    """Project one commitment into the graph. Idempotent. Never raises.

    Returns ``{"ok", "commitment_node", "person_node",
    "deadline_node", "commitment_id", "status"}`` where the ``*_node``
    values are graph node ids (or ``None`` when that node does not
    apply / could not be created).
    """
    result: dict[str, Any] = {
        "ok": False,
        "commitment_node": None,
        "person_node": None,
        "deadline_node": None,
        "commitment_id": None,
        "status": None,
    }
    try:
        if graph is None or commitment is None:
            result["error"] = "graph and commitment are required"
            return result
        commitment_id = _clean_str(_field(commitment, "id", ""))
        if not commitment_id:
            result["error"] = "commitment has no id"
            return result
        result["commitment_id"] = commitment_id
        status = _clean_str(_field(commitment, "status", _OPEN)) or _OPEN
        result["status"] = status

        what = _clean_str(_field(commitment, "what", ""))
        label = what or _clean_str(_field(commitment, "source_text", ""))
        if label:
            label = label.split("\n")[0].strip()[:200]
        if not label:
            label = "commitment"
        attrs = _commitment_attrs(commitment)

        node = _upsert_node(graph, "commitment", label, attrs,
                            f"commitment:{commitment_id}")
        if node is None:
            result["error"] = "could not create commitment node"
            return result
        commitment_node_id = _field(node, "node_id", "")
        result["commitment_node"] = commitment_node_id
        if not commitment_node_id:
            result["error"] = "commitment node has no id"
            return result

        # Person node for the promisee + owned_by edge.
        promisee = _clean_str(_field(commitment, "promisee", ""))
        if promisee:
            person = _upsert_node(
                graph, "person", promisee,
                {"name": promisee, "source": "commitments"},
                f"person:{promisee}")
            if person is not None:
                person_id = _field(person, "node_id", "")
                if person_id:
                    result["person_node"] = person_id
                    _add_edge(graph, commitment_node_id, person_id,
                              "owned_by")

        # Deadline node + due edge, when a deadline is present.
        deadline = _as_float(_field(commitment, "deadline", None))
        if deadline is not None:
            deadline_node = _upsert_node(
                graph, "deadline", f"Due: {label}"[:200],
                {"due_ts": deadline, "deadline": deadline,
                 "commitment_id": commitment_id, "status": status,
                 "closed": status != _OPEN, "source": "commitments"},
                f"deadline:{commitment_id}")
            if deadline_node is not None:
                deadline_id = _field(deadline_node, "node_id", "")
                if deadline_id:
                    result["deadline_node"] = deadline_id
                    _add_edge(graph, commitment_node_id, deadline_id,
                              "due")

        result["ok"] = True
        return result
    except Exception as exc:  # noqa: BLE001 — never raise
        result["error"] = str(exc)
        return result


def _iter_ledger_commitments(ledger: Any) -> list[Any]:
    """All commitments in a ledger: open + closed, deduped by id.

    Prefers the ledger's own snapshot (``commitments`` mapping) when
    available — that is the full state, open and closed.  Falls back
    to ``open_commitments()`` alone so a minimal ledger still syncs.
    """
    by_id: dict[str, Any] = {}

    def _take(rows: Any) -> None:
        try:
            items = list(rows or [])
        except TypeError:
            return
        for item in items:
            cid = _clean_str(_field(item, "id", ""))
            if cid and cid not in by_id:
                by_id[cid] = item

    # Full snapshot first (open + kept + broken + cancelled + superseded).
    try:
        snapshot = _field(ledger, "commitments", None)
        if isinstance(snapshot, dict):
            _take(snapshot.values())
        elif isinstance(snapshot, (list, tuple)):
            _take(snapshot)
    except Exception:  # noqa: BLE001
        pass
    # Open commitments via the public query (covers ledgers whose
    # snapshot is not exposed, and picks up anything the snapshot
    # missed).
    for method in ("open_commitments", "all_commitments",
                   "list_commitments"):
        try:
            fn = getattr(ledger, method, None)
            if callable(fn):
                _take(fn())
                break
        except Exception:  # noqa: BLE001
            continue
    # Last resort: entries-based ledgers sometimes expose all rows.
    if not by_id:
        for attr in ("all", "rows"):
            try:
                _take(_field(ledger, attr, None))
            except Exception:  # noqa: BLE001
                pass
    return list(by_id.values())


def sync_ledger_to_graph(ledger: Any, graph: Any) -> int:
    """Sync every commitment in ``ledger`` into ``graph``.

    Syncs open commitments plus kept / broken / cancelled / superseded
    ones when the ledger exposes them, so closed promises update
    their graph nodes instead of going stale.  Returns the number of
    commitments successfully projected.  Never raises.
    """
    try:
        if ledger is None or graph is None:
            return 0
        count = 0
        for commitment in _iter_ledger_commitments(ledger):
            try:
                if sync_commitment_to_graph(graph, commitment).get("ok"):
                    count += 1
            except Exception:  # noqa: BLE001 — one bad row, not the sync
                continue
        return count
    except Exception:  # noqa: BLE001 — never raise
        return 0


def sync_extracted_to_graph(message: Any, ledger: Any,
                            graph: Any) -> dict[str, Any]:
    """Extract commitments from ``message`` and project them to graph.

    Calls ``ledger.extract_and_record(message)`` (as a list — the
    ledger's intake takes a list of message dicts) and syncs every
    newly recorded commitment into ``graph``.

    ``message`` may be a single message dict or a list of them.
    Returns ``{"ok", "added", "commitment_nodes", "commitment_ids"}``.
    Never raises.
    """
    result: dict[str, Any] = {
        "ok": False,
        "added": 0,
        "commitment_nodes": [],
        "commitment_ids": [],
    }
    try:
        if ledger is None or graph is None:
            result["error"] = "ledger and graph are required"
            return result
        extract = getattr(ledger, "extract_and_record", None)
        if not callable(extract):
            result["error"] = "ledger has no extract_and_record"
            return result
        messages = message if isinstance(message, list) else [message]
        try:
            added = extract(messages)
        except TypeError:
            # A ledger whose intake takes a single message dict.
            added = extract(message)  # type: ignore[operator]
        if added is None:
            added = []
        try:
            added_list = list(added)
        except TypeError:
            added_list = [added]
        synced = 0
        for commitment in added_list:
            try:
                sync = sync_commitment_to_graph(graph, commitment)
            except Exception:  # noqa: BLE001 — never raise
                continue
            if sync.get("ok"):
                synced += 1
                node_id = sync.get("commitment_node")
                if node_id:
                    result["commitment_nodes"].append(node_id)
                cid = sync.get("commitment_id") or _clean_str(
                    _field(commitment, "id", ""))
                if cid:
                    result["commitment_ids"].append(cid)
        result["added"] = synced
        result["ok"] = True
        return result
    except Exception as exc:  # noqa: BLE001 — never raise
        result["error"] = str(exc)
        return result


def overdue_commitments_in_graph(graph: Any, now: Any = None) -> list[Any]:
    """Commitment nodes past due and still open.

    For scheduler reminders / weekly-review slippage: a node counts
    when its ``due_ts`` (or ``deadline``) attr is before ``now`` and
    its ``status`` attr is ``open`` (missing status counts as open —
    nodes projected before status tracking existed stay visible).
    Returns graph nodes sorted by due time, earliest first.
    Never raises.
    """
    try:
        if graph is None:
            return []
        now_ts = _as_float(now)
        if now_ts is None:
            now_ts = time.time()
        list_nodes = getattr(graph, "list_nodes", None)
        if not callable(list_nodes):
            return []
        try:
            nodes = list_nodes("commitment")
        except TypeError:
            nodes = list_nodes()
        overdue: list[tuple[float, Any]] = []
        for node in nodes or []:
            try:
                attrs = _field(node, "attrs", {}) or {}
                if isinstance(attrs, dict):
                    status = _clean_str(attrs.get("status", _OPEN)) or _OPEN
                    due = _as_float(attrs.get("due_ts",
                                              attrs.get("deadline",
                                                        attrs.get("due"))))
                else:
                    status, due = _OPEN, None
                if status != _OPEN:
                    continue
                if due is None or not due < now_ts:
                    continue
                overdue.append((due, node))
            except Exception:  # noqa: BLE001 — one bad node, not the query
                continue
        overdue.sort(key=lambda pair: pair[0])
        return [node for _, node in overdue]
    except Exception:  # noqa: BLE001 — never raise
        return []
