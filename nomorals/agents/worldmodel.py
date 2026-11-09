"""The world model (L5) — versioned facts with provenance, on the board.

Doc 5 §4: a world model the whole system shares — a versioned map of
the codebase, runtime, devices, accounts and work state, where every
assertion is tagged **observed / inferred / proposed / unverified** and
linked to its source.  This extends the existing
:class:`~nomorals.agents.blackboard.Blackboard` (versioned entries,
audit snapshots, reactive links) rather than replacing it: each fact
is a board entry under ``worldmodel:<key>`` whose board version IS the
fact's version, and a ``worldmodel:__version`` counter entry versions
the map as a whole.

Readers get the version they read (:class:`FactSnapshot`), so a
decision can name the exact state it consumed; a writer asserting a
newer fact bumps both versions; :meth:`WorldModel.invalidate` posts a
journaled tombstone instead of deleting history.  Version history
(who wrote which version, when) comes from the board's audit trail —
which stores value *previews* by design, so history answers provenance
questions, while full values live on the current entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .blackboard import Blackboard

__all__ = [
    "WorldModel",
    "FactSnapshot",
    "KIND_OBSERVED",
    "KIND_INFERRED",
    "KIND_PROPOSED",
    "KIND_UNVERIFIED",
    "FACT_KINDS",
]

KIND_OBSERVED = "observed"      # directly seen: tool result, file read
KIND_INFERRED = "inferred"      # derived from observations
KIND_PROPOSED = "proposed"      # a plan/intent, not yet reality
KIND_UNVERIFIED = "unverified"  # asserted by someone, not yet checked
FACT_KINDS = (KIND_OBSERVED, KIND_INFERRED, KIND_PROPOSED,
              KIND_UNVERIFIED)

_PREFIX = "worldmodel:"
_VERSION_KEY = "worldmodel:__version"


@dataclass(frozen=True)
class FactSnapshot:
    """One fact as read: value + the version it was read at."""

    key: str
    value: Any
    version: int            # this fact's version (board entry version)
    world_version: int      # the map's version when this was read
    kind: str
    source: str
    author: str
    updated_at: float
    invalidated: bool = False
    invalidate_reason: str = ""


class WorldModel:
    """Provenance-tagged, versioned facts over a :class:`Blackboard`."""

    def __init__(self, board: Blackboard, *, author: str = "worldmodel"
                 ) -> None:
        self.board = board
        self.author = author

    # -- writes -----------------------------------------------------------

    def assert_fact(self, key: str, value: Any, *, kind: str,
                    source: str, author: str = "") -> FactSnapshot:
        """Assert (or update) a fact.  ``kind`` is one of FACT_KINDS;
        ``source`` names where the assertion came from — a claim
        without provenance does not enter the world model."""
        if kind not in FACT_KINDS:
            raise ValueError(f"fact kind must be one of {FACT_KINDS}, "
                             f"got {kind!r}")
        if not source:
            raise ValueError("a world-model fact needs a source")
        entry = self.board.post(
            _PREFIX + key,
            {"value": value, "kind": kind, "source": source,
             "invalidated": False},
            author=author or self.author, topic="worldmodel",
            metadata={"kind": kind, "source": source})
        self._bump_world_version(author or self.author)
        return self._snapshot(key, entry)

    def invalidate(self, key: str, *, reason: str, by: str = ""
                   ) -> FactSnapshot | None:
        """Journal a tombstone: the fact stays in history, flagged."""
        current = self.read_fact(key)
        if current is None:
            return None
        entry = self.board.post(
            _PREFIX + key,
            {"value": current.value, "kind": current.kind,
             "source": current.source, "invalidated": True,
             "invalidate_reason": reason, "invalidated_by": by},
            author=by or self.author, topic="worldmodel",
            metadata={"kind": current.kind, "source": current.source,
                      "invalidated": True})
        self._bump_world_version(by or self.author)
        return self._snapshot(key, entry)

    def _bump_world_version(self, author: str) -> int:
        current = self.board.get(_VERSION_KEY)
        value = int(current) + 1 if current is not None else 1
        self.board.post(_VERSION_KEY, value, author=author,
                        topic="worldmodel")
        return value

    # -- reads --------------------------------------------------------------

    def version(self) -> int:
        """The world model's current global version (0 = empty map)."""
        current = self.board.get(_VERSION_KEY)
        return int(current) if current is not None else 0

    def _snapshot(self, key: str, entry: Any) -> FactSnapshot:
        payload = entry.value if isinstance(entry.value, dict) else {}
        return FactSnapshot(
            key=key, value=payload.get("value"), version=entry.version,
            world_version=self.version(),
            kind=str(payload.get("kind") or KIND_UNVERIFIED),
            source=str(payload.get("source") or ""),
            author=entry.author, updated_at=entry.updated_at,
            invalidated=bool(payload.get("invalidated")),
            invalidate_reason=str(payload.get("invalidate_reason") or ""))

    def read_fact(self, key: str) -> FactSnapshot | None:
        entry = self.board.entry(_PREFIX + key)
        if entry is None:
            return None
        return self._snapshot(key, entry)

    def facts(self, prefix: str = "") -> list[FactSnapshot]:
        """All current facts, optionally narrowed by key prefix."""
        out = []
        for full_key in self.board.keys(_PREFIX + "*"):
            if full_key == _VERSION_KEY:
                continue
            key = full_key[len(_PREFIX):]
            if prefix and not key.startswith(prefix):
                continue
            entry = self.board.entry(full_key)
            if entry is not None:
                out.append(self._snapshot(key, entry))
        return sorted(out, key=lambda f: f.key)

    def history(self, key: str) -> list[dict[str, Any]]:
        """Version provenance for one fact, oldest first: every audit
        record the board kept (version, author, when, value preview)."""
        return self.board.history(_PREFIX + key, limit=200)
