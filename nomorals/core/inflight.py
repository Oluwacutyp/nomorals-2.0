"""In-flight run deduplication (§3.9 of the self-provisioning law).

A provision run dedupes: while an install of one tool is in
flight, a second caller for the SAME tool ATTACHES to the
in-flight run — it waits for that run and receives its outcome —
instead of starting a parallel install of the same tool. Two
installers racing the same target (an MSI and a package manager,
two ``npm ci`` in one directory, two downloads extracting into
one managed dir) corrupt each other's work; one run, many
waiters, is the only sane shape.

Semantics, exactly:

* Per-key: different keys run fully in parallel — this is never
  a global provisioning lock.
* The first caller for a key leads: it runs the callable and
  publishes the outcome. Concurrent callers attach: they block
  until the leader publishes, then receive the SAME outcome —
  the leader's return value, or the leader's exception
  re-raised (the same instance: an attacher never earns a
  verdict the run itself did not produce, and N attachers never
  become N sequential retries of a failed run).
* A completed flight is forgotten the moment it lands. Callers
  arriving afterwards start a fresh flight, whose callable is
  the organ's own business (organs put their already-installed /
  already-attempted checks inside or before the flight, so a
  post-completion call takes the shipped path — this registry
  caches no results).
* If the leader thread dies without publishing an outcome (a
  staged flight abandoned by its owner, a fork stranding the
  registry), attachers do not hang: the first attacher to notice
  claims the orphaned flight, records the takeover (registry
  event + log + the caller's ``on_orphan`` hook), and leads a
  fresh run itself.
* A leader calling :meth:`InflightRegistry.run` for its own key
  again fails fast instead of deadlocking on itself.

Stdlib only, in-process threads: provisioning runs inside the
runtime process, driven from its handler threads. Cross-process
races are the organs' own on-disk state checks, unchanged.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any, TypeVar

from .logging_setup import get_logger

_log = get_logger(__name__)

T = TypeVar("T")

#: How long an attacher sleeps between leader-liveness checks.
_WAIT_SLICE_S = 0.05

#: Registry events kept for inspection (oldest drop off).
_EVENTS_KEPT = 200

#: Returned by _attach when the flight it waited on was orphaned
#: and this caller should contend to lead the fresh run.
_ORPHANED = object()


class Flight:
    """One in-flight run for a key: the shared outcome slot its
    leader publishes and its attachers read."""

    __slots__ = ("key", "done", "value", "error", "leader",
                 "attached", "orphaned")

    def __init__(self, key: str) -> None:
        self.key = key
        self.done = threading.Event()
        self.value: Any = None
        self.error: BaseException | None = None
        self.leader: threading.Thread | None = None
        self.attached = 0
        self.orphaned = False

    def outcome(self) -> Any:
        """The leader's result — or its exception, re-raised."""
        if self.error is not None:
            raise self.error
        return self.value


class InflightRegistry:
    """Keyed in-flight runs with attach semantics. One registry
    is shared per process (see :data:`default_registry`); tests
    build their own for isolation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._flights: dict[str, Flight] = {}
        #: Recent registry history — leader completions, attaches,
        #: failures, orphan takeovers — newest last.
        self.events: deque[dict[str, Any]] = deque(
            maxlen=_EVENTS_KEPT)

    # ── inspection ───────────────────────────────────────────────

    def in_flight(self, key: str) -> bool:
        with self._lock:
            return key in self._flights

    def _record(self, event: str, key: str) -> None:
        self.events.append(
            {"event": event, "key": key, "ts": time.time()})

    # ── flight lifecycle ─────────────────────────────────────────

    def begin(self, key: str) -> Flight | None:
        """Claim a flight for ``key`` as its leader, or return
        None when a flight is already registered (the caller
        should attach via :meth:`run` instead). Low-level: the
        claimant owns publishing the outcome and calling
        :meth:`_finish`. Staged tests use it to build an
        abandoned flight; organs use :meth:`run`."""
        with self._lock:
            if key in self._flights:
                return None
            flight = Flight(key)
            flight.leader = threading.current_thread()
            self._flights[key] = flight
            return flight

    def _finish(self, flight: Flight) -> None:
        """Publish a led flight: record, unregister, release the
        attachers — in that order, so a caller arriving after the
        run lands never attaches to a completed flight."""
        self._record(
            "failed" if flight.error is not None else "completed",
            flight.key)
        with self._lock:
            if self._flights.get(flight.key) is flight:
                del self._flights[flight.key]
        flight.done.set()

    def run(self, key: str, fn: Callable[[], T], *,
            on_orphan: Callable[[str], None] | None = None) -> T:
        """Run ``fn`` as the flight for ``key``, or attach to the
        flight already running and receive its outcome.

        ``on_orphan`` fires (with the key) when this caller takes
        over for a leader that died without an outcome, so the
        organ can record the fallback in its own journal; a hook
        that raises is logged, never allowed to break the run."""
        while True:
            flight = self.begin(key)
            if flight is not None:
                try:
                    flight.value = fn()
                except BaseException as exc:  # noqa: BLE001 - the
                    flight.error = exc  # outcome IS the contract
                self._finish(flight)
                return flight.outcome()
            outcome = self._attach(key, on_orphan)
            if outcome is not _ORPHANED:
                return outcome

    def _attach(self, key: str, on_orphan: Callable[[str],
                            None] | None) -> Any:
        with self._lock:
            flight = self._flights.get(key)
            if flight is not None:
                flight.attached += 1
        if flight is None:
            return _ORPHANED  # it landed between begin and here
        if flight.leader is threading.current_thread():
            raise RuntimeError(
                f"in-flight run for {key!r} cannot attach to "
                "itself")
        self._record("attached", key)
        while not flight.done.is_set():
            if flight.orphaned:
                return _ORPHANED
            flight.done.wait(_WAIT_SLICE_S)
            if flight.done.is_set() or flight.orphaned:
                continue
            leader = flight.leader
            if leader is None or leader.is_alive():
                continue
            # The leader is gone and no outcome was published:
            # claim the orphan so exactly one waiter takes over.
            with self._lock:
                claimed = (
                    self._flights.get(key) is flight
                    and not flight.done.is_set()
                    and not flight.orphaned)
                if claimed:
                    flight.orphaned = True
                    del self._flights[key]
            if claimed:
                self._record("orphaned", key)
                _log.warning(
                    "in-flight run for %s lost its leader without "
                    "an outcome; a waiting caller takes over with "
                    "a fresh run", key)
                if on_orphan is not None:
                    try:
                        on_orphan(key)
                    except Exception:  # noqa: BLE001 - the record
                        _log.exception(  # hook never breaks a run
                            "orphan hook for %s failed", key)
                return _ORPHANED
        return flight.outcome()


#: The process-wide registry provisioning organs share: a tool is
#: a machine-level fact, so its flight is too — two supervisor
#: instances racing one Node install attach to the same run.
default_registry = InflightRegistry()


def run_inflight(key: str, fn: Callable[[], T], *,
                 on_orphan: Callable[[str], None] | None
                 = None) -> T:
    """Run ``fn`` on the shared registry — see
    :meth:`InflightRegistry.run`."""
    return default_registry.run(key, fn, on_orphan=on_orphan)
