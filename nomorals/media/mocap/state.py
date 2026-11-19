"""Cross-engine mocap state: one shared store every engine reads and
writes.

The pattern is artcraftx's tasks table, re-expressed for live mocap
state (see ``MOCAP_STATE_SURVEY.md`` in ``~/workspace/
mocap-state-verify/`` — pattern only, no upstream code):

* One store, one entry per channel. A channel is a named stream of
  mocap data (``skeleton``, ``pointcloud``, ``depth``, ``bvh``).
* One writer per channel. An engine claims a channel before writing;
  a write from anyone else is refused and the refusal is recorded
  (the upstream rule is one row per provider job, enforced by a
  unique index; here it is ownership, enforced in code).
* Every write carries a monotonic ``seq`` and a timestamp. Reads
  report staleness explicitly — age in seconds against the channel's
  limit — and never pretend stale data is fresh; ``require_fresh``
  refuses it by name.
* Subscribers register with an opaque payload that is re-emitted on
  every poll (the upstream ``frontend_subscriber_payload`` rule).
  A subscriber polls for changes since its last poll; if the bounded
  delta log rotated past what it last saw, the poll says how many
  writes it missed and the caller re-syncs from a snapshot.
* ``snapshot()`` gives a late-joining engine the current state of
  every channel — no history required to join.

The store is in-process state: it lives as long as the process does.
``save``/``load`` persist it as JSON for durability across restarts.
Payloads must be JSON-serialisable — the store refuses anything else
at write time, naming the value that failed.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .errors import (
    ChannelOwnershipError,
    MocapError,
    StaleChannelError,
    UnknownChannelError,
    UnknownEngineError,
    UnknownSubscriberError,
)

__all__ = [
    "MAX_DELTAS",
    "MocapStateStore",
]

#: Deltas kept per channel. Past this, a subscriber polling for
#: changes is told how many writes it missed and must re-sync.
MAX_DELTAS = 64


def _iso_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _require_jsonable(payload: Any, *, where: str) -> None:
    try:
        json.dumps(payload)
    except (TypeError, ValueError) as exc:
        raise MocapError(
            f"{where} payload is not JSON-serialisable: {exc}",
            kind="bad_payload") from exc


class MocapStateStore:
    """The shared mocap state store."""

    def __init__(self, *,
                 clock: Callable[[], float] | None = None,
                 max_deltas: int = MAX_DELTAS) -> None:
        self._clock = clock if clock is not None else time.monotonic
        self._max_deltas = max(1, int(max_deltas))
        self._engines: dict[str, dict[str, Any]] = {}
        self._channels: dict[str, dict[str, Any]] = {}
        self._subscribers: dict[str, dict[str, Any]] = {}

    # -- engines ---------------------------------------------------

    def register_engine(self, name: str, *, produces: tuple[str, ...] = (),
                        consumes: tuple[str, ...] = (),
                        configured: bool = True,
                        note: str = "") -> dict[str, Any]:
        """An engine joins the store, declaring what it produces and
        consumes. A duplicate name is refused."""
        if not name:
            raise MocapError("engine name must not be empty",
                             kind="bad_engine")
        if name in self._engines:
            raise MocapError(f"engine {name!r} is already registered",
                             kind="duplicate_engine")
        record = {
            "name": name,
            "produces": list(produces),
            "consumes": list(consumes),
            "configured": bool(configured),
            "note": note,
            "registered_at": _iso_now(),
        }
        self._engines[name] = record
        return dict(record)

    def unregister_engine(self, name: str) -> dict[str, Any]:
        """An engine leaves: its claims are released and it drops
        off the roster. Unknown engines are refused by name."""
        self._require_engine(name)
        for channel in self._channels.values():
            if channel["owner"] == name:
                channel["owner"] = None
        return self._engines.pop(name)

    def engines(self) -> list[dict[str, Any]]:
        """Every registered engine, in registration order."""
        return [dict(record) for record in self._engines.values()]

    def _require_engine(self, name: str) -> dict[str, Any]:
        try:
            return self._engines[name]
        except KeyError as exc:
            raise UnknownEngineError(name) from exc

    # -- ownership -------------------------------------------------

    def claim(self, channel: str, engine: str) -> dict[str, Any]:
        """Take sole write ownership of a channel. Refused — naming
        the owner — when someone else already holds it."""
        self._require_engine(engine)
        record = self._channel(channel, create=True)
        owner = record["owner"]
        if owner is not None and owner != engine:
            self._record_failure(
                record, "ownership",
                f"claim refused: channel {channel!r} owned by {owner!r}")
            raise ChannelOwnershipError(channel, owner=owner,
                                        action="claim")
        record["owner"] = engine
        return self._channel_view(channel, record)

    def release(self, channel: str, engine: str) -> dict[str, Any]:
        """Give up write ownership. Only the owner may release."""
        self._require_engine(engine)
        record = self._channel(channel, create=False)
        if record["owner"] != engine:
            self._record_failure(
                record, "ownership",
                f"release refused: channel {channel!r} owned by "
                f"{record['owner']!r}")
            raise ChannelOwnershipError(channel, owner=record["owner"],
                                        action="release")
        record["owner"] = None
        return self._channel_view(channel, record)

    def owner(self, channel: str) -> str | None:
        """Who owns a channel right now, or None when it is free."""
        return self._channel(channel, create=False)["owner"]

    # -- writes ----------------------------------------------------

    def write(self, channel: str, engine: str, payload: Any, *,
              kind: str = "snapshot", max_age_s: float | None = None,
              claim_if_free: bool = True) -> dict[str, Any]:
        """Write to a channel. The writer must own it — or, when the
        channel is free and ``claim_if_free`` holds, the write claims
        it. A write from a non-owner is refused, named, and recorded
        in the channel's failure ledger. ``kind`` is ``snapshot``
        (replaces the current value) or ``delta`` (an event on top
        of it)."""
        if kind not in ("snapshot", "delta"):
            raise MocapError(
                f"write kind must be 'snapshot' or 'delta', got "
                f"{kind!r}", kind="bad_kind")
        _require_jsonable(payload, where=f"write to {channel!r}")
        self._require_engine(engine)
        record = self._channel(channel, create=True)
        owner = record["owner"]
        if owner is not None and owner != engine:
            self._record_failure(
                record, "ownership",
                f"write refused: channel {channel!r} owned by "
                f"{owner!r}")
            raise ChannelOwnershipError(channel, owner=owner)
        if owner is None and claim_if_free:
            record["owner"] = engine
        record["seq"] += 1
        record["ts"] = self._clock()
        record["updated_at"] = _iso_now()
        record["writer"] = engine
        if kind == "snapshot":
            record["payload"] = payload
        record["latest"] = payload
        if max_age_s is not None:
            if max_age_s <= 0:
                raise MocapError("max_age_s must be positive",
                                 kind="bad_max_age")
            record["max_age_s"] = float(max_age_s)
        entry = {
            "seq": record["seq"],
            "ts": record["ts"],
            "writer": engine,
            "kind": kind,
            "payload": payload,
        }
        record["deltas"].append(entry)
        while len(record["deltas"]) > self._max_deltas:
            record["deltas"].pop(0)
            record["dropped"] += 1
        return {
            "channel": channel,
            "seq": record["seq"],
            "writer": engine,
            "kind": kind,
            "owner": record["owner"],
        }

    # -- reads -----------------------------------------------------

    def read(self, channel: str, *, since_seq: int | None = None,
             require_fresh: bool = False,
             max_age_s: float | None = None) -> dict[str, Any]:
        """Read a channel. The result always carries the writer, the
        sequence, the age in seconds, and whether the data is stale
        against its limit — stale data is flagged, never silently
        served as fresh. With ``since_seq``, the changes since that
        sequence come back too, plus a ``missed`` count when the
        bounded delta log rotated past what the caller last saw (the
        caller must then re-sync from a snapshot)."""
        record = self._channel(channel, create=False)
        if record["seq"] == 0:
            raise MocapError(
                f"channel {channel!r} has no writes yet",
                kind="empty_channel")
        now = self._clock()
        age_s = now - record["ts"]
        limit = (max_age_s if max_age_s is not None
                 else record["max_age_s"])
        stale = limit is not None and age_s > limit
        if stale and require_fresh:
            raise StaleChannelError(channel, age_s=age_s,
                                    max_age_s=limit)
        out: dict[str, Any] = {
            "channel": channel,
            "payload": record["payload"],
            "latest": record["latest"],
            "seq": record["seq"],
            "writer": record["writer"],
            "owner": record["owner"],
            "age_s": round(age_s, 6),
            "stale": stale,
            "max_age_s": limit,
            "updated_at": record["updated_at"],
            "failures": list(record["failures"]),
        }
        if since_seq is not None:
            kept = record["deltas"]
            first_kept = kept[0]["seq"] if kept else record["seq"] + 1
            missed = max(0, (first_kept - 1) - int(since_seq))
            out["changes"] = [dict(d) for d in kept
                              if d["seq"] > int(since_seq)]
            out["missed"] = missed
            out["dropped_total"] = record["dropped"]
        return out

    def snapshot(self) -> dict[str, Any]:
        """The current state of every channel — the late joiner's
        view. Channels with no writes yet are listed as empty."""
        channels = {}
        for name in sorted(self._channels):
            record = self._channels[name]
            if record["seq"] == 0:
                channels[name] = {
                    "seq": 0, "writer": None, "owner": record["owner"],
                    "payload": None, "stale": False, "empty": True,
                }
                continue
            view = self.read(name)
            view["empty"] = False
            channels[name] = view
        return {
            "channels": channels,
            "engines": self.engines(),
            "at": _iso_now(),
        }

    def channels(self) -> list[str]:
        """Channel names, in first-write order."""
        return list(self._channels)

    def failures(self, channel: str) -> list[dict[str, Any]]:
        """The failure ledger for a channel — type + message for
        every refused write, even non-terminal ones."""
        return list(self._channel(channel, create=False)["failures"])

    # -- subscribers -----------------------------------------------

    def subscribe(self, subscriber_id: str,
                  channels: tuple[str, ...] | list[str], *,
                  payload: Any = None) -> dict[str, Any]:
        """Watch channels. The opaque ``payload`` is stored and
        re-emitted on every poll, exactly as the subscriber set it —
        the upstream ``frontend_subscriber_payload`` rule. A
        subscriber starts caught up at the current sequence of each
        channel it watches."""
        if not subscriber_id:
            raise MocapError("subscriber id must not be empty",
                             kind="bad_subscriber")
        _require_jsonable(payload, where="subscribe")
        seen: dict[str, int] = {}
        for channel in channels:
            record = self._channel(channel, create=True)
            seen[channel] = record["seq"]
        self._subscribers[subscriber_id] = {
            "channels": list(channels),
            "payload": payload,
            "seen": seen,
            "subscribed_at": _iso_now(),
        }
        return {
            "subscriber": subscriber_id,
            "channels": list(channels),
            "caught_up_at": dict(seen),
        }

    def unsubscribe(self, subscriber_id: str) -> None:
        """Stop watching. Unknown subscribers are refused by name."""
        if subscriber_id not in self._subscribers:
            raise UnknownSubscriberError(subscriber_id)
        del self._subscribers[subscriber_id]

    def poll(self, subscriber_id: str) -> dict[str, Any]:
        """Changes since the subscriber's last poll, per watched
        channel, plus the subscriber's own payload re-emitted. A
        channel whose delta log rotated reports ``missed`` and the
        caller re-syncs that channel from a snapshot."""
        try:
            sub = self._subscribers[subscriber_id]
        except KeyError as exc:
            raise UnknownSubscriberError(subscriber_id) from exc
        updates: dict[str, Any] = {}
        for channel in sub["channels"]:
            view = self.read(channel, since_seq=sub["seen"][channel])
            sub["seen"][channel] = view["seq"]
            updates[channel] = view
        return {
            "subscriber": subscriber_id,
            "subscriber_payload": sub["payload"],
            "updates": updates,
            "at": _iso_now(),
        }

    def subscribers(self) -> list[str]:
        """Subscriber ids, in subscription order."""
        return list(self._subscribers)

    # -- persistence -----------------------------------------------

    def save(self, path: str | Path) -> dict[str, Any]:
        """Persist the whole store as JSON. The path is resolved to
        an absolute path and recorded in the result."""
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "engines": self._engines,
            "channels": self._channels,
            "subscribers": self._subscribers,
            "saved_at": _iso_now(),
        }
        target.write_text(json.dumps(blob, indent=1, sort_keys=True),
                          encoding="utf-8")
        return {"path": str(target), "channels": len(self._channels),
                "engines": len(self._engines)}

    @classmethod
    def load(cls, path: str | Path, *,
             clock: Callable[[], float] | None = None,
             max_deltas: int = MAX_DELTAS) -> MocapStateStore:
        """Restore a store saved with :meth:`save`. A missing file is
        refused by name; a corrupt file is refused with the parse
        error."""
        target = Path(path).expanduser().resolve()
        if not target.exists():
            raise MocapError(f"no saved store at {target}",
                             kind="missing_store")
        try:
            blob = json.loads(target.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise MocapError(
                f"saved store at {target} is unreadable: {exc}",
                kind="corrupt_store") from exc
        store = cls(clock=clock, max_deltas=max_deltas)
        store._engines = blob.get("engines", {})
        store._channels = blob.get("channels", {})
        store._subscribers = blob.get("subscribers", {})
        return store

    # -- internals -------------------------------------------------

    def _channel(self, channel: str, *, create: bool) -> dict[str, Any]:
        if channel not in self._channels:
            if not create:
                raise UnknownChannelError(channel)
            self._channels[channel] = {
                "owner": None,
                "seq": 0,
                "ts": 0.0,
                "updated_at": None,
                "writer": None,
                "payload": None,
                "latest": None,
                "max_age_s": None,
                "deltas": [],
                "dropped": 0,
                "failures": [],
            }
        return self._channels[channel]

    def _record_failure(self, record: dict[str, Any], kind: str,
                        message: str) -> None:
        record["failures"].append({
            "type": kind, "message": message, "at": _iso_now(),
        })

    def _channel_view(self, channel: str,
                      record: dict[str, Any]) -> dict[str, Any]:
        return {
            "channel": channel,
            "owner": record["owner"],
            "seq": record["seq"],
            "writer": record["writer"],
        }
