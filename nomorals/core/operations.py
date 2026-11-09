"""Chat-scoped operation targeting (Living Chat P1, 2026-11-08).

Interactive cards — numbered fallback lists, pick-lists, menus —
are OPERATIONS: a set of actions offered in one chat at one
moment. This store is the targeting substrate they bind to, so a
number or a tap lands on the operation the user is actually
looking at:

* Operations live in a slot keyed ``(chat_key, owner_key)``. The
  owner key is the user the card was offered to ("" = the chat's
  shared slot, e.g. a DM or a deliberately shared group card).
  Two users in one group hold separate slots; two chats never
  share one. A bare number from user B can therefore never
  consume user A's card.
* Opening an operation REPLACES the slot's current one (the
  newest card owns the numbers — the standing house rule). The
  replaced operation is kept briefly as a TOMBSTONE.
* Resolving by number acts on the slot's current operation.
  Resolving by a specific operation id that is no longer current
  — a tap on a stale card — answers :data:`PickStatus.STALE`
  instead of executing against the wrong operation, and the
  caller turns that into the honest "that card has been
  replaced" reply. An expired operation answers STALE the same
  way while its tombstone lasts; a slot that never held an
  operation answers :data:`PickStatus.MISSING`.
* A successful pick CONSUMES the operation (single-use); an
  out-of-range number leaves it intact — a typo must not burn
  the card's other actions.

In-memory and bounded, like the pick stores it generalizes: a
restart forgets open cards, which then answer as missing rather
than acting on phantom state. This module never raises.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "Operation",
    "OperationStore",
    "PickResult",
    "PickStatus",
    "DEFAULT_TTL_S",
    "MAX_SLOTS",
]

#: How long an offered operation stays actionable.
DEFAULT_TTL_S = 600.0
#: Upper bound on live slots + tombstones, LRU-evicted.
MAX_SLOTS = 512


class PickStatus(Enum):
    """The outcome of resolving a pick against the store."""

    OK = "ok"                #: command resolved; operation consumed
    STALE = "stale"          #: the referenced operation was replaced/expired
    MISSING = "missing"      #: no operation (live or tombstoned) matches
    OUT_OF_RANGE = "out_of_range"  #: live operation, number outside it


@dataclass
class Operation:
    """One offered card: its identity, owner slot and actions."""

    operation_id: str
    chat_key: str
    owner_key: str = ""
    kind: str = ""
    actions: tuple[str, ...] = ()
    revision: int = 0
    opened_at: float = 0.0
    expires_at: float = 0.0
    #: Once replaced/consumed/expired, how long the operation is
    #: still REMEMBERED, so a late reference answers STALE instead
    #: of MISSING. Set when the operation leaves the current seat.
    remember_until: float = 0.0

    def expired(self, now: float | None = None) -> bool:
        ts = time.time() if now is None else float(now)
        return ts >= self.expires_at

    def remembered(self, now: float | None = None) -> bool:
        ts = time.time() if now is None else float(now)
        return ts < self.remember_until


@dataclass
class PickResult:
    """A resolution outcome; ``command`` is set only on OK."""

    status: PickStatus
    command: str = ""
    operation_id: str = ""
    kind: str = ""


@dataclass
class _Slot:
    current: Operation | None = None
    tombstone: Operation | None = None
    revision: int = 0


class OperationStore:
    """Bounded, thread-safe (chat, owner) → operation store."""

    def __init__(self, *, ttl_s: float = DEFAULT_TTL_S,
                 max_slots: int = MAX_SLOTS) -> None:
        self._ttl_s = float(ttl_s)
        self._max_slots = int(max_slots)
        self._lock = threading.Lock()
        self._slots: OrderedDict[tuple[str, str], _Slot] = OrderedDict()

    # ── opening ────────────────────────────────────────────────

    def open(self, chat_key: str, actions: list[str] | tuple[str, ...],
             *, owner_key: str = "", kind: str = "",
             ttl_s: float | None = None,
             now: float | None = None,
             operation_id: str = "") -> Operation | None:
        """Offer a new operation in the slot, replacing the old.

        Returns the Operation (its id is the card's stable
        reference), or None for an empty chat key / action list.

        ``operation_id`` (Living Chat P2): a caller that already
        owns a durable identity for this card — a living card's
        persisted id — may name it, so the operation and the
        durable card are one identity across a restart. The id
        keeps this store's shape (short, url-safe); an unusable
        value falls back to a generated id.
        """
        key = (str(chat_key or ""), str(owner_key or ""))
        acts = tuple(str(a) for a in (actions or []) if str(a).strip())
        if not key[0] or not acts:
            return None
        ts = time.time() if now is None else float(now)
        ttl = self._ttl_s if ttl_s is None else float(ttl_s)
        with self._lock:
            slot = self._slots.get(key)
            if slot is None:
                slot = _Slot()
                self._slots[key] = slot
            if slot.current is not None:
                slot.current.remember_until = ts + self._ttl_s
                slot.tombstone = slot.current
            slot.revision += 1
            named = str(operation_id or "").strip()
            if not named or len(named) > 32 or not all(
                    c.isalnum() or c in "-_" for c in named):
                named = secrets.token_hex(4)
            op = Operation(
                operation_id=named,
                chat_key=key[0], owner_key=key[1], kind=str(kind or ""),
                actions=acts, revision=slot.revision,
                opened_at=ts, expires_at=ts + ttl)
            slot.current = op
            self._slots.move_to_end(key)
            while len(self._slots) > self._max_slots:
                self._slots.popitem(last=False)
            return op

    # ── reading ────────────────────────────────────────────────

    def current(self, chat_key: str, *, owner_key: str = "",
                now: float | None = None) -> Operation | None:
        """The slot's live operation, or None (expired counts)."""
        key = (str(chat_key or ""), str(owner_key or ""))
        with self._lock:
            slot = self._slots.get(key)
            if slot is None or slot.current is None:
                return None
            if slot.current.expired(now):
                self._retire(slot, now)
                return None
            return slot.current

    # ── resolving ──────────────────────────────────────────────

    def resolve_number(self, chat_key: str, number: int, *,
                       owner_key: str = "",
                       now: float | None = None) -> PickResult:
        """Resolve a bare number against the slot's current card.

        OK consumes the operation. STALE when the slot's card is
        gone but a tombstone remembers one the number could have
        meant; MISSING when the slot never held (or has fully
        forgotten) a card.
        """
        key = (str(chat_key or ""), str(owner_key or ""))
        with self._lock:
            slot = self._slots.get(key)
            if slot is None:
                return PickResult(PickStatus.MISSING)
            if slot.current is not None and slot.current.expired(now):
                self._retire(slot, now)
            if slot.current is not None:
                return self._pick(slot, slot.current, number, now)
            tomb = slot.tombstone
            if tomb is not None and tomb.remembered(now) \
                    and 1 <= int(number) <= len(tomb.actions):
                return PickResult(PickStatus.STALE,
                                  operation_id=tomb.operation_id,
                                  kind=tomb.kind)
            return PickResult(PickStatus.MISSING)

    def resolve_operation(self, chat_key: str, operation_id: str,
                          number: int, *, owner_key: str = "",
                          now: float | None = None) -> PickResult:
        """Resolve a pick that NAMES its operation (a card tap).

        The named operation must be the slot's current one; a
        replaced or expired one answers STALE so the caller can
        say "that card has been replaced" instead of acting on
        the wrong operation.
        """
        key = (str(chat_key or ""), str(owner_key or ""))
        op_id = str(operation_id or "")
        with self._lock:
            slot = self._slots.get(key)
            if slot is None or not op_id:
                return PickResult(PickStatus.MISSING)
            if slot.current is not None and slot.current.expired(now):
                self._retire(slot, now)
            cur = slot.current
            if cur is not None and cur.operation_id == op_id:
                return self._pick(slot, cur, number, now)
            tomb = slot.tombstone
            if tomb is not None and tomb.operation_id == op_id:
                return PickResult(PickStatus.STALE, operation_id=op_id,
                                  kind=tomb.kind)
            return PickResult(PickStatus.MISSING)

    # ── internals ──────────────────────────────────────────────

    def _retire(self, slot: _Slot, now: float | None) -> None:
        """Move an expired current operation to the tombstone."""
        if slot.current is not None:
            ts = time.time() if now is None else float(now)
            slot.current.remember_until = ts + self._ttl_s
            slot.tombstone = slot.current
            slot.current = None

    def _pick(self, slot: _Slot, op: Operation, number: int,
              now: float | None) -> PickResult:
        n = int(number)
        if n < 1 or n > len(op.actions):
            return PickResult(PickStatus.OUT_OF_RANGE,
                              operation_id=op.operation_id, kind=op.kind)
        ts = time.time() if now is None else float(now)
        op.remember_until = ts + self._ttl_s
        slot.current = None
        slot.tombstone = op
        return PickResult(PickStatus.OK, command=op.actions[n - 1],
                          operation_id=op.operation_id, kind=op.kind)
