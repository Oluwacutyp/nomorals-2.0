"""Persisted organ event bus (Tier 0 wiring).

:mod:`nomorals.organs` is a SQLite event store (``organ_events`` table,
``consumed`` flag). This module is the durable file-backed complement
for callers that need a bus without a database: events are appended to
a JSONL log with ``fsync`` (a crash after ``publish`` returns cannot
lose the event), consumption is tracked in a sidecar state file, and
poison messages go to a dead-letter log instead of being retried
forever.

Delivery semantics (honest, per-instance):

* :meth:`PersistedBus.drain` — exactly once *per bus state*: drained
  events are marked consumed in the sidecar before they are returned,
  so a second drain (even from a new instance on the same files) never
  re-delivers them. A crash between the durable mark and the caller
  processing the event can still lose it — callers that cannot afford
  that use :meth:`drain_and_handle`.
* :meth:`PersistedBus.drain_and_handle` — at-least-once processing with
  exactly-once *completion*: an event is marked consumed only after
  its handler succeeds. A handler that keeps raising is retried up to
  ``max_attempts`` times, then dead-lettered with its reason.
* :meth:`dead_letter` / :meth:`dead_letter_review` — poison messages
  land in ``<path>.dead.jsonl`` with their reason and can be reviewed
  (and re-driven by hand) instead of wedging the bus.

Concurrency: all operations hold one process-local lock and rewrite
the state sidecar atomically (tmp file + ``os.replace`` + fsync).
Cross-process writers are not claimed — one bus instance owns its
files, like one SQLite connection owns its database.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from .tier0_rollback import record_promotion

__all__ = ["PersistedBus", "register_promotion", "promotion_active"]

PROMOTION_NAME = "organs_bus"

_promotion_active = False


def promotion_active() -> bool:
    """True while the persisted-bus promotion is active (not rolled back)."""
    return _promotion_active


def register_promotion() -> None:
    """Record this wiring in the Tier 0 rollback registry."""
    global _promotion_active
    _promotion_active = True

    def _rollback() -> None:
        global _promotion_active
        _promotion_active = False

    record_promotion(
        PROMOTION_NAME, _rollback,
        description="persisted JSONL organ event bus (exactly-once drain)",
        metadata={"module": __name__},
    )


class PersistedBus:
    """Durable JSONL event bus with exactly-once drain + dead letters."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_attempts: int = 3,
        dead_letter_path: str | Path | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path = self.path.with_suffix(
            self.path.suffix + ".state.json")
        self.dead_path = (
            Path(dead_letter_path) if dead_letter_path is not None
            else self.path.with_suffix(self.path.suffix + ".dead.jsonl"))
        self.max_attempts = max(1, int(max_attempts))
        self._lock = threading.RLock()
        self._closed = False

    # ── publish ──────────────────────────────────────────────────────────────
    def publish(
        self,
        event: dict[str, Any] | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        """Append one event durably (JSONL + fsync). Returns the event.

        ``event`` may carry ``id`` / ``ts``; missing ones are generated.
        ``kind`` defaults to ``"event"``; ``src``/``dst`` default to
        ``"unknown"``/``"broadcast"``. Extra keyword fields merge in.
        """
        if self._closed:
            raise RuntimeError("PersistedBus is closed")
        stored = dict(event or {})
        stored.update(fields)
        stored.setdefault("id", uuid.uuid4().hex)
        stored.setdefault("ts", time.time())
        stored.setdefault("kind", "event")
        stored.setdefault("src", "unknown")
        stored.setdefault("dst", "broadcast")
        stored.setdefault("payload", {})
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(stored, default=str) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        return dict(stored)

    # ── state sidecar ────────────────────────────────────────────────────────
    def _load_state(self) -> dict[str, Any]:
        try:
            with open(self.state_path, encoding="utf-8") as fh:
                state = json.load(fh)
            if isinstance(state, dict):
                state.setdefault("consumed", [])
                state.setdefault("attempts", {})
                return state
        except (OSError, ValueError):
            pass
        return {"consumed": [], "attempts": {}}

    def _save_state(self, state: dict[str, Any]) -> None:
        tmp = self.state_path.with_suffix(
            self.state_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.state_path)

    def _read_log(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        try:
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue  # a torn line is skipped, never fatal
                    if isinstance(row, dict) and row.get("id") is not None:
                        events.append(row)
        except OSError:
            pass
        return events

    def _pending(
        self, dst: str | None, state: dict[str, Any]
    ) -> list[dict[str, Any]]:
        consumed = set(state.get("consumed") or [])
        return [
            e for e in self._read_log()
            if e["id"] not in consumed
            and (dst is None or e.get("dst") == dst)
        ]

    # ── drain ────────────────────────────────────────────────────────────────
    def drain(
        self, dst: str | None = None, *, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Take up to ``limit`` pending events for ``dst`` (or all).

        Marks them consumed durably before returning — a second drain
        never re-delivers them. Prefer :meth:`drain_and_handle` when
        the caller cannot afford to lose an event to a crash between
        drain and processing.
        """
        if self._closed:
            raise RuntimeError("PersistedBus is closed")
        if limit <= 0:
            return []
        with self._lock:
            state = self._load_state()
            pending = self._pending(dst, state)[:limit]
            if pending:
                consumed = set(state.get("consumed") or [])
                consumed.update(e["id"] for e in pending)
                state["consumed"] = sorted(consumed)
                self._save_state(state)
            return [dict(e) for e in pending]

    def drain_and_handle(
        self,
        handler: Callable[[dict[str, Any]], Any],
        dst: str | None = None,
        *,
        limit: int = 50,
    ) -> dict[str, int]:
        """Drain through ``handler``: success → consumed; repeated
        failure → dead-lettered after ``max_attempts`` attempts.

        Returns ``{"delivered": n, "dead_lettered": n, "retried": n}``
        where ``retried`` counts events that failed but still have
        attempts left (they stay pending for the next call).
        """
        if self._closed:
            raise RuntimeError("PersistedBus is closed")
        if not callable(handler):
            raise TypeError("handler must be callable")
        delivered = dead = retried = 0
        with self._lock:
            state = self._load_state()
            pending = self._pending(dst, state)[:max(0, limit)]
            consumed = set(state.get("consumed") or [])
            attempts = dict(state.get("attempts") or {})
            for event in pending:
                try:
                    handler(dict(event))
                except Exception as exc:  # noqa: BLE001 - poison path
                    n = int(attempts.get(event["id"], 0)) + 1
                    attempts[event["id"]] = n
                    if n >= self.max_attempts:
                        self._dead_letter_locked(
                            event,
                            f"{type(exc).__name__}: {exc} "
                            f"(failed {n}/{self.max_attempts} attempts)")
                        consumed.add(event["id"])
                        dead += 1
                    else:
                        retried += 1
                else:
                    consumed.add(event["id"])
                    attempts.pop(event["id"], None)
                    delivered += 1
            state["consumed"] = sorted(consumed)
            state["attempts"] = attempts
            self._save_state(state)
        return {"delivered": delivered, "dead_lettered": dead,
                "retried": retried}

    # ── dead letters ─────────────────────────────────────────────────────────
    def dead_letter(self, event: dict[str, Any], reason: str) -> None:
        """Move one event to the dead-letter log with a reason.

        Also marks it consumed so plain :meth:`drain` never returns it
        again. Re-dead-lettering the same id is a no-op (the first
        reason stands).
        """
        if self._closed:
            raise RuntimeError("PersistedBus is closed")
        with self._lock:
            if any(d.get("event", {}).get("id") == event.get("id")
                   for d in self._read_dead()):
                return
            self._dead_letter_locked(event, reason)
            state = self._load_state()
            consumed = set(state.get("consumed") or [])
            if event.get("id") is not None:
                consumed.add(event["id"])
            state["consumed"] = sorted(consumed)
            self._save_state(state)

    def _dead_letter_locked(
        self, event: dict[str, Any], reason: str
    ) -> None:
        entry = {"event": dict(event), "reason": str(reason),
                 "dead_at": time.time()}
        with open(self.dead_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def _read_dead(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        try:
            with open(self.dead_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(row, dict):
                        out.append(row)
        except OSError:
            pass
        return out

    def dead_letter_review(self) -> list[dict[str, Any]]:
        """Every dead-lettered event with its reason and ``dead_at``."""
        with self._lock:
            return [dict(d) for d in self._read_dead()]

    # ── inspection / lifecycle ───────────────────────────────────────────────
    def pending_count(self, dst: str | None = None) -> int:
        """Pending (unconsumed) events for ``dst`` (or all)."""
        with self._lock:
            return len(self._pending(dst, self._load_state()))

    def close(self) -> None:
        """Mark closed. Files stay on disk — the bus reopens by path."""
        self._closed = True

    def __enter__(self) -> "PersistedBus":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


register_promotion()
