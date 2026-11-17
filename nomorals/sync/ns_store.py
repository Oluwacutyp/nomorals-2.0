"""Namespace persistence on top of the sync store.

Each namespace lives as one sync record under key ``ns:<name>`` with
value :meth:`Namespace.to_dict`.  That buys two things for free:

* plain ``nm sync push`` / ``pull`` replicate already-merged
  namespace state over the real transport (record layer), and an
  :class:`AutoSync` loop re-syncs on every namespace write because
  writes go through :meth:`SyncStore.put`, which notifies
  subscribers;
* ``ns merge`` is the one place where two devices' namespace
  states are combined at namespace granularity.  It must be: the
  record layer merges per *top-level field of the record value*,
  and a namespace's whole ``map`` dict is a single field, so a
  plain pull would last-write-wins the entire map and silently
  drop the loser's keys.  The namespace merge (different-field
  edits compose, same-field forks become visible conflicts) only
  happens here.
"""
from __future__ import annotations

import logging
from typing import Any

from .namespaces import Conflict, Namespace, NamespaceSet

_log = logging.getLogger(__name__)

__all__ = [
    "NS_KEY_PREFIX",
    "ns_key",
    "load_set",
    "load_set_from_peer",
    "save_namespace",
    "merge_peer_namespaces",
]

NS_KEY_PREFIX = "ns:"


def ns_key(name: str) -> str:
    """The sync-record key a namespace persists under."""
    return f"{NS_KEY_PREFIX}{name}"


def _namespace_from_record_value(value: Any) -> Namespace | None:
    """Rebuild a namespace from a record value, or None when the
    value is not a namespace at all (honest skip, never a raise —
    one bad row must not lose the whole set)."""
    if not isinstance(value, dict) or not value.get("name"):
        return None
    try:
        return Namespace.from_dict(value)
    except Exception:  # noqa: BLE001 - corrupt row; skip, keep going
        _log.warning("sync ns: skipping unreadable namespace record",
                     exc_info=True)
        return None


def load_set(store: Any, device_id: str) -> NamespaceSet:
    """Load every namespace persisted in ``store``.

    Tombstoned keys never appear in ``store.keys()``; malformed
    rows are skipped with a warning, not a crash.
    """
    result = NamespaceSet(device_id=device_id)
    keys = store.keys()
    for key in keys:
        if not key.startswith(NS_KEY_PREFIX):
            continue
        rec = store.get(key)
        if rec is None:
            continue
        ns = _namespace_from_record_value(rec.value)
        if ns is not None:
            result.adopt(ns)
    return result


def load_set_from_peer(peer: Any, device_id: str) -> NamespaceSet:
    """Load a peer's namespaces over the :class:`SyncPeer` contract.

    Reads the peer's records from the start of its sequence space
    (``fetch_since_seq(0)``, falling back to ``fetch_since(0)`` for
    legacy peers) and keeps the ``ns:`` ones.  Deleted records are
    skipped — a namespace deleted on the peer is not resurrected
    by a merge.
    """
    try:
        records = peer.fetch_since_seq(0)
    except NotImplementedError:
        records = peer.fetch_since(0.0)
    result = NamespaceSet(device_id=device_id)
    for rec in records:
        if not rec.key.startswith(NS_KEY_PREFIX) or rec.deleted:
            continue
        ns = _namespace_from_record_value(rec.value)
        if ns is not None:
            result.adopt(ns)
    return result


def save_namespace(store: Any, ns: Namespace) -> None:
    """Persist one namespace as a sync record (a plain ``put`` — so
    subscribers, AutoSync included, see the write)."""
    store.put(ns_key(ns.name), ns.to_dict())


def merge_peer_namespaces(local_set: NamespaceSet,
                          peer_set: NamespaceSet,
                          store: Any) -> list[Conflict]:
    """Merge ``peer_set`` into ``local_set`` at namespace granularity
    and persist every merged namespace back to ``store``.

    Returns the *new* conflicts this merge recorded.  Different-field
    edits compose; same-field forks are recorded on the merged
    namespace (never silently dropped) and stay visible until an
    owner resolves them.
    """
    new = local_set.merge(peer_set)
    for name in peer_set.names():
        ns = local_set.get(name)
        if ns is not None:
            save_namespace(store, ns)
    return new
