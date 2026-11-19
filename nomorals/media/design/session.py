"""FigCraft sessions — the multiplayer server and clients (lane 105).

Follows upstream's collab protocol (``docs/collab-protocol.md``,
``crates/collab-net``) with an in-process transport: the
``SessionServer`` is the single authority for ordering — it
assigns every accepted change a sequence number and rebroadcasts
it; clients apply their own edits immediately (optimistically)
and reconcile by applying broadcasts in server order, leniently.
Cursors, selections and viewports are ephemeral presence
messages that are never stored. A client that drops off queues
its edits and resumes with a catch-up delta on reconnect.

What is NOT built here: a real network transport. There is no
WebSocket server in this module — the session is in-process, and
the messages are the upstream message shapes (Hello / Welcome /
Changes / Presence / Ack / Reject / PeerJoined / PeerLeft) as
plain dicts, so a transport can carry them later without
changing the protocol. That wall is named in the survey.
"""

from __future__ import annotations

import secrets
from copy import deepcopy
from typing import Any

from .collab import History, Replica
from .edit import DesignEditor
from .errors import DesignError
from .model import DesignDocument

__all__ = ["SessionServer", "SessionClient"]

_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no look-alikes
_CODE_LEN = 9


def _code() -> str:
    return "".join(secrets.choice(_CODE_ALPHABET) for _ in range(_CODE_LEN))


class SessionServer:
    """One document's session authority (in-process)."""

    def __init__(self, doc: DesignDocument):
        self.doc = doc
        self.replica = Replica(doc.to_dict())
        self.file_id = f"figcraft:{secrets.token_hex(8)}"
        self.seq = 0
        self.history = History()
        self._clients: dict[int, dict[str, Any]] = {}
        self._outbox: dict[int, list[dict[str, Any]]] = {}
        self._presence_outbox: dict[int, list[dict[str, Any]]] = {}
        self._presence: dict[int, dict[str, Any]] = {}
        self._links: dict[str, str] = {}
        self._codes: dict[str, str] = {}
        max_space = 0
        for nid in doc.nodes:
            max_space = max(max_space, nid >> 32)
        self._next_client_id = max_space + 1
        self._batch_counter = 0

    # -- joining ------------------------------------------------------
    def hello(self, name: str, color: str = "#4472c4",
              token: str | None = None,
              resume: dict[str, Any] | None = None) -> dict[str, Any]:
        access = self._links.get(token, "edit") if token else "edit"
        if token is not None and token not in self._links:
            raise DesignError("unknown share link", kind="permission")
        client_id = None
        if resume is not None:
            old = int(resume.get("client_id", 0))
            if old in self._clients:
                client_id = old
        if client_id is None:
            client_id = self._next_client_id
            self._next_client_id += 1
        self._clients[client_id] = {"name": name, "color": color,
                                    "access": access, "seq": self.seq}
        self._outbox.setdefault(client_id, [])
        self._presence_outbox.setdefault(client_id, [])
        peers = [{"client_id": cid, "name": c["name"], "color": c["color"]}
                 for cid, c in self._clients.items() if cid != client_id]
        for cid in self._clients:
            if cid != client_id:
                self._outbox[cid].append(
                    {"type": "peer_joined", "client_id": client_id,
                     "name": name, "color": color})
        catch_up: dict[str, Any]
        if resume is not None and int(resume.get("seq", 0)) < self.seq:
            missed = [e for e in self.history.entries
                      if e["seq"] > int(resume.get("seq", 0))]
            catch_up = {"kind": "delta",
                        "batches": [{"seq": e["seq"], "author": e["author"],
                                     "batch": deepcopy(e["batch"])}
                                    for e in missed]}
        else:
            catch_up = {"kind": "snapshot",
                        "document": self.replica.materialize()}
        return {"session": self.file_id, "client_id": client_id,
                "seq": self.seq, "catch_up": catch_up,
                "peers": peers, "access": access}

    def leave(self, client_id: int) -> None:
        info = self._clients.pop(client_id, None)
        self._outbox.pop(client_id, None)
        self._presence_outbox.pop(client_id, None)
        self._presence.pop(client_id, None)
        if info is not None:
            for cid in self._clients:
                self._outbox[cid].append(
                    {"type": "peer_left", "client_id": client_id,
                     "name": info["name"]})

    # -- sharing ------------------------------------------------------
    def create_link(self, access: str) -> str:
        if access not in ("edit", "view", "comment"):
            raise DesignError(f"unknown access {access!r}",
                              kind="invalid_argument")
        token = secrets.token_urlsafe(12)
        self._links[token] = access
        return token

    def create_code(self, access: str) -> str:
        if access not in ("edit", "view", "comment"):
            raise DesignError(f"unknown access {access!r}",
                              kind="invalid_argument")
        code = _code()
        while code in self._codes:
            code = _code()
        self._codes[code] = access
        return code

    def resolve_code(self, code: str) -> tuple[str, str]:
        access = self._codes.get(code)
        if access is None:
            raise DesignError(f"unknown join code {code!r}",
                              kind="not_found")
        return self.file_id, access

    # -- changes ------------------------------------------------------
    def submit(self, client_id: int, batch_id: str,
               changes: list[dict[str, Any]], base_seq: int) -> dict[str, Any]:
        client = self._clients.get(client_id)
        if client is None:
            raise DesignError("client is not in this session",
                              kind="permission")
        if client["access"] == "view":
            return {"reject": {"batch_id": batch_id,
                               "reason": "view link: editing is refused",
                               "fixes": []}}
        if client["access"] == "comment":
            bad = [c for c in changes
                   if not (c.get("kind") in ("put", "remove")
                           and str(c.get("entry", "")).startswith("comment"))
                   and not (c.get("kind") == "set_doc"
                            and c.get("prop") == "comments")]
            if bad:
                return {"reject": {
                    "batch_id": batch_id,
                    "reason": "comment link: only comment changes are allowed",
                    "fixes": []}}
        inverse = self.replica.inverse(changes)
        fixes = self.replica.apply(changes, strict=True)
        if fixes:
            return {"reject": {"batch_id": batch_id,
                               "reason": "batch refused",
                               "fixes": fixes}}
        self.seq += 1
        self.history.record(self.seq, client["name"], changes, inverse)
        self.doc = DesignDocument.from_dict(self.replica.materialize())
        message = {"type": "changes", "seq": self.seq,
                   "author": client["name"], "batch_id": batch_id,
                   "changes": deepcopy(changes)}
        for cid in self._clients:
            self._outbox[cid].append(deepcopy(message))
        return {"seq": self.seq, "broadcast": message}

    def drain(self, client_id: int) -> list[dict[str, Any]]:
        out = self._outbox.get(client_id, [])
        self._outbox[client_id] = []
        return out

    # -- presence -----------------------------------------------------
    def presence(self, client_id: int, cursor: tuple[float, float] | None = None,
                 selection: list[int] | None = None,
                 viewport: dict[str, Any] | None = None) -> None:
        client = self._clients.get(client_id)
        if client is None:
            return
        state = {"client_id": client_id, "name": client["name"],
                 "color": client["color"],
                 "cursor": list(cursor) if cursor is not None else None,
                 "selection": selection, "viewport": viewport}
        self._presence[client_id] = state
        for cid in self._clients:
            if cid != client_id:
                self._presence_outbox[cid].append(
                    {"type": "presence", **deepcopy(state)})

    def drain_presence(self, client_id: int) -> list[dict[str, Any]]:
        out = self._presence_outbox.get(client_id, [])
        self._presence_outbox[client_id] = []
        return out

    def ack(self, client_id: int, seq: int) -> None:
        client = self._clients.get(client_id)
        if client is not None:
            client["seq"] = max(int(client.get("seq", 0)), int(seq))

    def peers(self, except_id: int | None = None) -> list[dict[str, Any]]:
        return [{"client_id": cid, "name": c["name"], "color": c["color"]}
                for cid, c in self._clients.items() if cid != except_id]


class SessionClient:
    """One editor in a session (in-process transport)."""

    def __init__(self, server: SessionServer, name: str,
                 color: str = "#4472c4"):
        self.server = server
        self.name = name
        self.color = color
        self.client_id: int | None = None
        self.access = "edit"
        self.doc: DesignDocument | None = None
        self.replica: Replica | None = None
        self.seq = 0
        self._batch_counter = 0
        self._offline = False
        self._queued: list[list[dict[str, Any]]] = []
        self._history = History()

    # -- connection ---------------------------------------------------
    def connect(self, token: str | None = None) -> dict[str, Any]:
        resume = None
        if self.client_id is not None:
            resume = {"session": self.server.file_id,
                      "client_id": self.client_id, "seq": self.seq}
        welcome = self.server.hello(self.name, self.color, token, resume)
        self.client_id = int(welcome["client_id"])
        self.access = welcome["access"]
        catch_up = welcome["catch_up"]
        if catch_up["kind"] == "snapshot":
            self.doc = DesignDocument.from_dict(catch_up["document"])
            self.replica = Replica(catch_up["document"])
        else:
            assert self.replica is not None and self.doc is not None
            for entry in catch_up["batches"]:
                self.replica.apply(entry["batch"], strict=False)
            self.doc = DesignDocument.from_dict(self.replica.materialize())
        self.seq = int(welcome["seq"])
        return welcome

    def go_offline(self) -> None:
        self._offline = True

    def reconnect(self) -> dict[str, Any]:
        self._offline = False
        welcome = self.connect()
        queued, self._queued = self._queued, []
        for batch in queued:
            self._submit(batch)
        if queued and self.replica is not None:
            self.doc = DesignDocument.from_dict(self.replica.materialize())
        return welcome

    # -- editing ------------------------------------------------------
    def _editor(self) -> DesignEditor:
        assert self.doc is not None
        return DesignEditor(self.doc)

    def _flush(self) -> None:
        assert self.doc is not None and self.replica is not None
        batch = self.replica.diff(self.doc.to_dict())
        if not batch:
            return
        self._submit(batch)

    def _submit(self, batch: list[dict[str, Any]]) -> None:
        assert self.replica is not None and self.client_id is not None
        if self._offline:
            self._queued.append(deepcopy(batch))
            return
        self._batch_counter += 1
        batch_id = f"{self.client_id}:{self._batch_counter}"
        inverse = self.replica.inverse(batch)
        self.replica.apply(batch, strict=False)
        result = self.server.submit(self.client_id, batch_id, batch, self.seq)
        if "reject" in result:
            reason = result["reject"].get("reason", "batch refused")
            # Roll the optimistic edit back first, then apply any
            # server fix (for a permission refusal there are none).
            self.replica.apply(inverse, strict=False)
            for fix in result["reject"].get("fixes", []):
                self.replica.apply([fix], strict=False)
            self.doc = DesignDocument.from_dict(self.replica.materialize())
            raise DesignError(f"batch refused: {reason}", kind="permission")
        self._history.record(int(result["seq"]), self.name, batch, inverse)

    def set_text(self, node_id: int, text: str) -> None:
        self._editor().set_text(node_id, text)
        self._flush()

    def set_fill(self, node_id: int, hex_color: str) -> None:
        self._editor().set_fill(node_id, hex_color)
        self._flush()

    def add_rect(self, parent_id: int, name: str, x: float, y: float,
                 width: float, height: float, fill: str) -> int:
        node_id = None
        if self.replica is not None and self.client_id is not None:
            node_id = self.replica.alloc_id(self.client_id)
        nid = self._editor().add_rect(parent_id, name, x, y, width, height,
                                      fill, node_id=node_id)
        self._flush()
        return nid

    def undo(self) -> bool:
        inverse = self._history.undo_for(self.name)
        if inverse is None or self.replica is None:
            return False
        self.replica.apply(inverse, strict=False)
        self.doc = DesignDocument.from_dict(self.replica.materialize())
        self._submit(inverse)
        return True

    def redo(self) -> bool:
        batch = self._history.redo_for(self.name)
        if batch is None or self.replica is None:
            return False
        self.replica.apply(batch, strict=False)
        self.doc = DesignDocument.from_dict(self.replica.materialize())
        self._submit(batch)
        return True

    # -- receiving ----------------------------------------------------
    def receive(self) -> list[dict[str, Any]]:
        """Drain the server's broadcasts and apply them in order."""
        assert self.client_id is not None and self.replica is not None
        messages = self.server.drain(self.client_id)
        applied: list[dict[str, Any]] = []
        for message in messages:
            if message.get("type") != "changes":
                continue
            self.replica.apply(message["changes"], strict=False)
            self.seq = max(self.seq, int(message["seq"]))
            applied.append(message)
        if applied:
            assert self.doc is not None
            self.doc = DesignDocument.from_dict(self.replica.materialize())
            # Re-apply edits queued while offline on top of the new state.
            for batch in self._queued:
                self.replica.apply(batch, strict=False)
            if self._queued:
                self.doc = DesignDocument.from_dict(self.replica.materialize())
        return applied

    def receive_presence(self) -> list[dict[str, Any]]:
        assert self.client_id is not None
        return self.server.drain_presence(self.client_id)

    def presence(self, cursor: tuple[float, float] | None = None,
                 selection: list[int] | None = None,
                 viewport: dict[str, Any] | None = None) -> None:
        if self.client_id is None or self._offline:
            return
        self.server.presence(self.client_id, cursor, selection, viewport)
