"""Session recording / replay for console sessions.

Records every console event (command, output, mission action, alert, …)
as an ordered, JSON-serializable entry so a session can be saved to
disk and replayed later — for debugging, demos, and audit.

Events are plain dicts: ``{seq, ts, kind, payload}``. ``seq`` is a
per-recorder monotonic counter assigned at record time, so replay order
is exactly record order even if two events share a timestamp.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator


class SessionRecorder:
    """Append-only event log with ordered replay + JSON persistence."""

    def __init__(self, *, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or time.time
        self._events: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self._seq = 0

    # ── record ──

    def record_event(self, kind: str, payload: Any = None,
                     *, ts: float | None = None) -> dict[str, Any]:
        """Append one event; returns the stored entry (with seq/ts)."""
        with self._lock:
            entry = {
                "seq": self._seq,
                "ts": self._clock() if ts is None else float(ts),
                "kind": str(kind),
                "payload": payload,
            }
            self._seq += 1
            self._events.append(entry)
            return dict(entry)

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    # ── replay ──

    def replay(self, *, from_seq: int = 0,
               kind: str | None = None) -> Iterator[dict[str, Any]]:
        """Yield stored events in record order (copies, never live refs)."""
        with self._lock:
            snapshot = list(self._events)
        for entry in snapshot:
            if entry["seq"] < from_seq:
                continue
            if kind is not None and entry["kind"] != kind:
                continue
            yield dict(entry)

    def events(self, *, kind: str | None = None) -> list[dict[str, Any]]:
        return list(self.replay(kind=kind))

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._seq = 0

    # ── persistence ──

    def dumps(self) -> str:
        with self._lock:
            return json.dumps(self._events, default=str)

    def loads(self, data: str) -> None:
        """Replace contents with events decoded from ``dumps()`` output."""
        rows = json.loads(data)
        if not isinstance(rows, list):
            raise ValueError("recording JSON must be a list of events")
        with self._lock:
            self._events = [dict(r) for r in rows]
            self._seq = max((int(r.get("seq", 0)) for r in rows), default=-1) + 1

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(self.dumps(), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path, *,
             clock: Callable[[], float] | None = None) -> "SessionRecorder":
        rec = cls(clock=clock)
        rec.loads(Path(path).read_text(encoding="utf-8"))
        return rec


__all__ = ["SessionRecorder"]
