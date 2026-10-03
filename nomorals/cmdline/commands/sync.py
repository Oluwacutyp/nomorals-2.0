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


def _peer_store(args: Any, context: Any):
    """Resolve the peer store: --peer-db path, or settings hub (not yet)."""
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
              "       nm sync push --peer-db PATH [--json]\n"
              "       nm sync pull --peer-db PATH [--json]",
              file=sys.stderr)
        return 2
    verb = words[0]
    engine, _ = _engine(context)
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
    if verb in ("push", "pull"):
        from ...sync import LocalPeer
        peer_store = _peer_store(args, context)
        if peer_store is None:
            print("sync %s: need --peer-db PATH (hub URL transport pending)"
                  % verb, file=sys.stderr)
            return 3
        peer = LocalPeer(peer_store)
        # push-then-pull is one atomic sync(); for pull-only we still run
        # the full sync (push is idempotent when nothing changed).
        result = engine.sync(peer, peer_id=peer_store.db.path if hasattr(
            peer_store.db, "path") else "peer")
        if as_json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"pushed {result.pushed}, pulled {result.pulled}, "
                  f"conflicts {result.conflicts_resolved}")
        return 0
    print(f"unknown sync verb: {verb}", file=sys.stderr)
    return 2
