"""``nm sync`` — multi-device state replication."""
from __future__ import annotations

import json
import sys
from typing import Any


def _engine(context: Any):
    from ...storage.db import Database
    from ...sync import SyncEngine, SyncStore
    db = getattr(context, "db", None)
    if db is None:
        from pathlib import Path
        home = Path.home() / ".nomorals"
        home.mkdir(parents=True, exist_ok=True)
        db = Database(str(home / "nomorals.db"))
    import socket
    device_id = getattr(context, "device_id", None) or socket.gethostname()
    store = SyncStore(db, device_id=device_id)
    return SyncEngine(db, store), store


def _http_peer(context: Any, words: list[str]):
    """Resolve the hub sync peer: ``--hub`` flag or NM_HUB_URL setting."""
    settings = getattr(context, "settings", None)
    hub_cfg = getattr(settings, "hub", None)
    hub_url = (getattr(hub_cfg, "url", "") or "").strip()
    use_hub = "--hub" in words or bool(hub_url)
    if not use_hub:
        return None
    url = hub_url
    for i, w in enumerate(words):
        if w == "--hub-url" and i + 1 < len(words):
            url = words[i + 1].strip()
    if not url:
        raise ValueError(
            "sync: --hub needs a hub URL — set NM_HUB_URL or pass "
            "--hub-url <url>")
    from ...sync import HttpSyncPeer
    return HttpSyncPeer(
        url,
        token=(getattr(hub_cfg, "token", "") or ""),
        timeout=float(getattr(hub_cfg, "request_timeout", 30.0) or 30.0),
    )


def _peer_store(args: Any, context: Any):
    """Resolve the peer store: --peer-db path (the hub peer is resolved
    separately by _http_peer)."""
    from ...storage.db import Database
    from ...sync import SyncStore
    peer_db = getattr(args, "peer_db", None)
    if peer_db:
        peer = Database(peer_db)
        return SyncStore(peer, device_id="peer")
    return None


def _cmd_sync(args: Any, context: Any) -> int:
    """Route ``nm sync <verb>``."""
    words = list(getattr(args, "task", None) or [])
    if not words:
        print("usage: nm sync status [--json]\n"
              "       nm sync put <key> <json-object>\n"
              "       nm sync get <key> [--json]\n"
              "       nm sync delete <key>\n"
              "       nm sync keys [--json]\n"
              "       nm sync push --peer-db PATH [--json]\n"
              "       nm sync pull --peer-db PATH [--json]\n"
              "       nm sync push --hub [--hub-url URL] [--json]  (via device hub)\n"
              "       nm sync pull --hub [--hub-url URL] [--json]  (via device hub)\n"
              "       nm sync ns list|create|get|put|append|incr|merge|conflicts|resolve",
              file=sys.stderr)
        return 2
    verb = words[0]
    engine, store = _engine(context)
    as_json = bool(getattr(args, "json", False))
    if verb == "status":
        st = engine.status()
        if as_json:
            print(json.dumps(st, indent=2, default=str))
        else:
            print(f"device: {st['device_id']}")
            print(f"local keys: {st['local_keys']}")
            print(f"pending push: {st['pending_push']}")
        return 0
    if verb == "put":
        if len(words) < 3:
            print("usage: nm sync put <key> <json-object>", file=sys.stderr)
            return 2
        try:
            value = json.loads(words[2])
        except json.JSONDecodeError as exc:
            print(f"invalid JSON value: {exc}", file=sys.stderr)
            return 2
        if not isinstance(value, dict):
            print("value must be a JSON object", file=sys.stderr)
            return 2
        try:
            rec = store.put(words[1], value)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"ok seq={rec.seq}")
        return 0
    if verb == "get":
        if len(words) < 2:
            print("usage: nm sync get <key>", file=sys.stderr)
            return 2
        rec = store.get(words[1])
        if rec is None:
            print(f"no such key: {words[1]}", file=sys.stderr)
            return 1
        if as_json:
            print(json.dumps(rec.to_dict(), indent=2, default=str))
        else:
            print(json.dumps(rec.value, indent=2, default=str))
        return 0
    if verb == "delete":
        if len(words) < 2:
            print("usage: nm sync delete <key>", file=sys.stderr)
            return 2
        rec = store.delete(words[1])
        print(f"ok seq={rec.seq} (tombstone, replicates on next push)")
        return 0
    if verb == "keys":
        keys = store.keys()
        if as_json:
            print(json.dumps(keys, indent=2))
        else:
            for k in keys:
                print(k)
        return 0
    if verb in ("push", "pull"):
        from ...sync import LocalPeer
        peer_store = _peer_store(args, context)
        try:
            http_peer = _http_peer(context, words) if peer_store is None else None
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if peer_store is None and http_peer is None:
            print("sync %s: need --peer-db PATH, or --hub / NM_HUB_URL "
                  "for the device hub" % verb, file=sys.stderr)
            return 3
        if http_peer is not None:
            peer, peer_id = http_peer, "hub"
        else:
            peer = LocalPeer(peer_store)
            # push-then-pull is one atomic sync(); for pull-only we still run
            # the full sync (push is idempotent when nothing changed).
            peer_path = getattr(peer_store.db, "path", None)
            peer_id = str(peer_path) if peer_path else "peer"
        result = engine.sync(peer, peer_id=peer_id)
        if as_json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"pushed {result.pushed}, pulled {result.pulled}, "
                  f"conflicts {result.conflicts_resolved}")
        return 0
    if verb == "ns":
        return _cmd_sync_ns(words[1:], args, context, engine, store)
    print(f"unknown sync verb: {verb}", file=sys.stderr)
    return 2


def _ns_flag(words: list[str], flag: str, default: str = "") -> str:
    """The value after ``flag`` in a word list, or ``default``."""
    for i, w in enumerate(words):
        if w == flag and i + 1 < len(words):
            return words[i + 1]
    return default


def _ns_load(context: Any, store: Any):
    """The local namespace set, read from the sync store."""
    import socket

    from ...sync.ns_store import load_set
    device_id = getattr(context, "device_id", None) or socket.gethostname()
    return load_set(store, device_id)


def _ns_save(store: Any, ns: Any) -> None:
    from ...sync.ns_store import save_namespace
    save_namespace(store, ns)


def _ns_get(context: Any, store: Any, name: str):
    """The namespace, or None (caller prints the honest refusal)."""
    return _ns_load(context, store).get(name)


def _cmd_sync_ns(words: list[str], args: Any, context: Any,
                 engine: Any, store: Any) -> int:
    """Route ``nm sync ns <subverb>`` — typed replicated state.

    Namespaces persist as sync records (``ns:<name>``), so plain
    push/pull move already-merged state; ``merge`` is the one place
    where two devices' states combine at namespace granularity.
    """
    from ...sync.namespaces import NAMESPACE_TYPES
    from ...sync.ns_store import (
        load_set,
        load_set_from_peer,
        merge_peer_namespaces,
    )
    as_json = bool(getattr(args, "json", False))
    if not words:
        print("usage: nm sync ns list [--json]\n"
              "       nm sync ns create <name> [--type map|list|counter]\n"
              "       nm sync ns get <name> [--json]\n"
              "       nm sync ns put <name> <key> <json-value>\n"
              "       nm sync ns append <name> <json-value> [--id ID]\n"
              "       nm sync ns incr <name> [amount]\n"
              "       nm sync ns merge --peer-db PATH [--json]\n"
              "       nm sync ns merge --hub [--hub-url URL] [--json]\n"
              "       nm sync ns conflicts [name] [--json]\n"
              "       nm sync ns resolve <name> <key> <json-value>",
              file=sys.stderr)
        return 2
    sub = words[0]
    if sub == "list":
        nset = _ns_load(context, store)
        if as_json:
            print(json.dumps(nset.to_dict(), indent=2, default=str))
        else:
            for name in nset.names():
                ns = nset.get(name)
                print(f"{name} ({ns.kind}): {json.dumps(ns.value(), default=str)}")
        return 0
    if sub == "create":
        if len(words) < 2:
            print("usage: nm sync ns create <name> [--type map|list|counter]",
                  file=sys.stderr)
            return 2
        kind = _ns_flag(words, "--type", "map")
        if kind not in NAMESPACE_TYPES:
            print(f"unknown namespace type {kind!r}; "
                  f"expected {NAMESPACE_TYPES}", file=sys.stderr)
            return 2
        nset = _ns_load(context, store)
        existing = nset.get(words[1])
        if existing is not None:
            if existing.kind != kind:
                print(f"namespace {words[1]!r} already exists as "
                      f"{existing.kind!r}, not {kind!r}", file=sys.stderr)
                return 2
            print(f"namespace {words[1]!r} already exists ({kind})")
            return 0
        ns = nset.get_or_create(words[1], kind)
        _ns_save(store, ns)
        print(f"created namespace {words[1]!r} ({kind})")
        return 0
    if sub == "get":
        if len(words) < 2:
            print("usage: nm sync ns get <name>", file=sys.stderr)
            return 2
        ns = _ns_get(context, store, words[1])
        if ns is None:
            print(f"no such namespace: {words[1]}", file=sys.stderr)
            return 1
        print(json.dumps(ns.value(), indent=2, default=str))
        return 0
    if sub == "put":
        if len(words) < 4:
            print("usage: nm sync ns put <name> <key> <json-value>",
                  file=sys.stderr)
            return 2
        ns = _ns_get(context, store, words[1])
        if ns is None:
            print(f"no such namespace: {words[1]}", file=sys.stderr)
            return 1
        try:
            value = json.loads(words[3])
        except json.JSONDecodeError as exc:
            print(f"invalid JSON value: {exc}", file=sys.stderr)
            return 2
        try:
            ns.put(words[2], value)
        except TypeError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        _ns_save(store, ns)
        print(f"ok {words[1]}.{words[2]} = {json.dumps(value, default=str)}")
        return 0
    if sub == "append":
        if len(words) < 3:
            print("usage: nm sync ns append <name> <json-value> [--id ID]",
                  file=sys.stderr)
            return 2
        ns = _ns_get(context, store, words[1])
        if ns is None:
            print(f"no such namespace: {words[1]}", file=sys.stderr)
            return 1
        try:
            value = json.loads(words[2])
        except json.JSONDecodeError as exc:
            print(f"invalid JSON value: {exc}", file=sys.stderr)
            return 2
        try:
            iid = ns.append(value, item_id=_ns_flag(words, "--id", ""))
        except TypeError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        _ns_save(store, ns)
        print(f"ok {words[1]} appended {iid}")
        return 0
    if sub == "incr":
        if len(words) < 2:
            print("usage: nm sync ns incr <name> [amount]", file=sys.stderr)
            return 2
        ns = _ns_get(context, store, words[1])
        if ns is None:
            print(f"no such namespace: {words[1]}", file=sys.stderr)
            return 1
        try:
            amount = int(words[2]) if len(words) > 2 else 1
        except ValueError:
            print(f"incr amount must be an integer, got {words[2]!r}",
                  file=sys.stderr)
            return 2
        try:
            total = ns.increment(amount)
        except (TypeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        _ns_save(store, ns)
        print(f"ok {words[1]} = {total}")
        return 0
    if sub == "merge":
        peer_store = _peer_store(args, context)
        try:
            http_peer = _http_peer(context, words) if peer_store is None else None
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if peer_store is None and http_peer is None:
            print("sync ns merge: need --peer-db PATH, or --hub / NM_HUB_URL "
                  "for the device hub", file=sys.stderr)
            return 3
        local = _ns_load(context, store)
        if peer_store is not None:
            peer_set = load_set(peer_store, "peer")
        else:
            peer_set = load_set_from_peer(http_peer, "peer")
        new = merge_peer_namespaces(local, peer_set, store)
        merged_names = peer_set.names()
        if as_json:
            print(json.dumps({
                "merged": merged_names,
                "new_conflicts": [c.to_dict() for c in new],
            }, indent=2, default=str))
        else:
            names = ", ".join(merged_names) if merged_names else "—"
            line = f"merged {len(merged_names)} namespace(s): {names}"
            if new:
                line += f", {len(new)} new conflict(s)"
            print(line)
            for c in new:
                print(f"conflict {c.namespace}.{c.key}: "
                      f"{json.dumps(c.values, default=str)} "
                      f"(kept {json.dumps(c.resolved_to, default=str)})")
        return 0
    if sub == "conflicts":
        nset = _ns_load(context, store)
        if len(words) > 1:
            ns = nset.get(words[1])
            if ns is None:
                print(f"no such namespace: {words[1]}", file=sys.stderr)
                return 1
            open_cs = ns.open_conflicts()
        else:
            # NamespaceSet.conflicts() is the full history; the open
            # list is the per-namespace view with resolutions applied.
            open_cs = [c for ns in (nset.get(n) for n in nset.names())
                       for c in ns.open_conflicts()]
        if as_json:
            print(json.dumps([c.to_dict() for c in open_cs],
                             indent=2, default=str))
        else:
            if not open_cs:
                print("no open conflicts")
            for c in open_cs:
                print(f"conflict {c.namespace} {c.kind} {c.key}: "
                      f"{json.dumps(c.values, default=str)} "
                      f"(kept {json.dumps(c.resolved_to, default=str)})")
        return 0
    if sub == "resolve":
        if len(words) < 4:
            print("usage: nm sync ns resolve <name> <key> <json-value>",
                  file=sys.stderr)
            return 2
        ns = _ns_get(context, store, words[1])
        if ns is None:
            print(f"no such namespace: {words[1]}", file=sys.stderr)
            return 1
        try:
            value = json.loads(words[3])
        except json.JSONDecodeError as exc:
            print(f"invalid JSON value: {exc}", file=sys.stderr)
            return 2
        try:
            record = ns.resolve(words[2], value)
        except (TypeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        _ns_save(store, ns)
        print(f"resolved {words[1]}.{words[2]} = "
              f"{json.dumps(record['resolved_to'], default=str)}")
        return 0
    print(f"unknown sync ns verb: {sub}", file=sys.stderr)
    return 2
