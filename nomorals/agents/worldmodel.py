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

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
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
    "PersistentWorldModel",
    "WORLDMODEL_BOARD_ENV",
    "default_board_path",
    "default_world",
    "plan_event_sink",
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


# ---------------------------------------------------------------------
# The production home (2026-11-10 — the readers pass).
#
# Until this pass the world model existed only in tests: no production
# code constructed one, nothing fed it, nothing read it, and the
# blackboards production builds are per-run scratch that dies with the
# run.  The home below is the board's own persistence idiom
# (Blackboard.save/load, which never raises) at the settings-home data
# layout every other durable store uses:
# ``<home>/data/worldmodel-board.json``.

#: Env override for the board path (absolute as-is, relative resolved
#: against the settings home) — the same shape as the immune stores'.
WORLDMODEL_BOARD_ENV = "NM_WORLDMODEL_BOARD"


def _settings_home(settings: Any = None) -> Path:
    """The settings data dir, never the cwd (mirrors immune wiring)."""
    if settings is not None:
        try:
            return Path(settings.home_path)
        except Exception:  # noqa: BLE001 - duck-typed settings
            pass
    return Path.home() / ".nomorals"


def default_board_path(context: Any = None) -> Path:
    """Resolve the world-model board path: env override, then the
    context's settings, then the home default — the same file from
    chat, CLI and scheduler, next to the other durable stores."""
    settings = getattr(context, "settings", None) if context is not None \
        else None
    override = os.environ.get(WORLDMODEL_BOARD_ENV, "")
    if override:
        p = Path(os.path.expanduser(override))
        if p.is_absolute():
            return p
        return (_settings_home(settings) / p).resolve()
    if settings is not None:
        try:
            return Path(settings.resolve("data/worldmodel-board.json"))
        except Exception:  # noqa: BLE001 - fall through to home default
            pass
    return _settings_home(settings) / "data" / "worldmodel-board.json"


class PersistentWorldModel(WorldModel):
    """A WorldModel whose every write persists its board.

    Persistence is the board's own save (atomic tmp-replace, never
    raises), so a failed save degrades to in-memory state with the
    save's own warning logged — a write never breaks its caller.
    """

    def __init__(self, board: Blackboard, path: Path, *,
                 author: str = "worldmodel") -> None:
        super().__init__(board, author=author)
        self.path = Path(path)

    def assert_fact(self, key: str, value: Any, *, kind: str,
                    source: str, author: str = "") -> FactSnapshot:
        snap = super().assert_fact(key, value, kind=kind, source=source,
                                   author=author)
        self.board.save(self.path)
        return snap

    def invalidate(self, key: str, *, reason: str, by: str = ""
                   ) -> FactSnapshot | None:
        snap = super().invalidate(key, reason=reason, by=by)
        if snap is not None:
            self.board.save(self.path)
        return snap


_world_cache: dict[str, WorldModel] = {}


def default_world(context: Any = None, *, path: Any = None) -> WorldModel:
    """The shared production world model at its resolved board path.

    Cached per path: the chat runtime, the CLI and the scheduler in
    one process share one model over one board file; a board file
    that already exists is loaded (a restore is state recovery — the
    board's load never raises, and a missing/corrupt file simply
    starts an empty map)."""
    target = Path(path) if path is not None else default_board_path(context)
    cache_key = str(target)
    cached = _world_cache.get(cache_key)
    if cached is not None:
        return cached
    board = Blackboard()
    if target.exists():
        board.load(target)
    world = PersistentWorldModel(board, target)
    _world_cache[cache_key] = world
    return world


def plan_event_sink(world: WorldModel) -> Callable[[Mapping[str, Any]], None]:
    """The feeder sink the plan-contract seam reports into.

    One observed fact per settled/failed plan step, keyed the way the
    substrate already names the step's evidence
    (``plan-step:<run>/<step>``), carrying the fields an owner reads:
    the goal, the step, who worked it, the outcome, and the gate's
    reasons in the seam's own words.  The fact IS the record the
    settlement produced — kind OBSERVED, source named — never a
    summary invented after the fact."""

    def _sink(event: Mapping[str, Any]) -> None:
        mission = str(event.get("mission_id") or "run")
        step = str(event.get("step") or "step")
        outcome = str(event.get("outcome") or "")
        value: dict[str, Any] = {
            "goal": str(event.get("goal") or ""),
            "step": step,
            "worker": str(event.get("worker") or ""),
            "outcome": outcome,
        }
        reasons = [str(r) for r in (event.get("reasons") or ())]
        if reasons:
            value["reasons"] = reasons
        detail = str(event.get("detail") or "")
        if detail:
            value["detail"] = detail
        source = ("plan step failure (delegation contract)"
                  if outcome == "failed"
                  else "plan step settlement (delegation contract)")
        key = str(event.get("fact_key") or f"plan-step:{mission}/{step}")
        world.assert_fact(key, value, kind=KIND_OBSERVED, source=source,
                          author="plan_contracts")

    return _sink
