"""Shared scratch space for cooperating agents.

A blackboard is the classic multi-agent coordination pattern: agents do not pass
messages point-to-point, they publish partial results to a shared store and read
what others have contributed. It suits this system because a research agent's
output is genuinely useful to the coding, writing, and critique agents alike, and
wiring that as explicit edges would make the graph unreadable.

Entries are versioned and access-counted, so the orchestrator can tell which
contributions were actually used — that signal feeds the reflector.

Reactive triggers (:meth:`Blackboard.link`) turn the board into the team's
nervous system: an agent links a key pattern to its work function, and every
matching post fires that function — agents trigger each other with no manual
commands and no polling. :meth:`Blackboard.save` / :meth:`load` persist the
board so shared state survives a restart.
"""

from __future__ import annotations

import fnmatch
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from ..core.ids import new_id
from ..core.logging_setup import get_logger

__all__ = ["Blackboard", "BlackboardEntry", "Subscription"]

_log = get_logger(__name__)


@dataclass
class BlackboardEntry:
    key: str
    value: Any
    author: str = ""
    topic: str = ""
    version: int = 1
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    reads: int = 0
    ttl: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def expired(self) -> bool:
        return bool(self.ttl) and time.time() > self.created_at + self.ttl

    def to_dict(self, *, include_value: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "key": self.key,
            "author": self.author,
            "topic": self.topic,
            "version": self.version,
            "reads": self.reads,
            "updated_at": self.updated_at,
            "expired": self.expired,
        }
        if include_value:
            payload["value"] = self.value
        return payload


class Subscription:
    """A handle on a reactive trigger created by :meth:`Blackboard.link`.

    ``cancel()`` detaches the trigger; the subscription also detaches itself
    when garbage-collected. ``fired`` counts how many posts triggered it.
    """

    def __init__(self, board: "Blackboard", token: object) -> None:
        self._board = board
        self._token = token
        self.fired = 0
        self.cancelled = False

    def cancel(self) -> None:
        if not self.cancelled:
            self.cancelled = True
            board = self._board
            if board is not None:
                board._unlink(self._token)

    def __del__(self) -> None:  # pragma: no cover - GC timing
        try:
            self.cancel()
        except Exception:  # noqa: BLE001 - never raise from a finalizer
            pass


class Blackboard:
    """Thread-safe shared store with watch support and reactive triggers."""

    def __init__(self, *, prune_expired: bool = True,
                 max_trigger_workers: int = 8) -> None:
        self._entries: dict[str, BlackboardEntry] = {}
        self._history: list[BlackboardEntry] = []
        self._watchers: list[tuple[str, Callable[[BlackboardEntry], None]]] = []
        self._lock = threading.RLock()
        # Wakes condition-based waiters the moment a key lands (wait_for no
        # longer has to poll).
        self._changed = threading.Condition(self._lock)
        # Reactive triggers: token -> (pattern, callback, subscription).
        self._links: dict[object, tuple[str, Callable[[BlackboardEntry], None], Subscription]] = {}
        self._trigger_pool: ThreadPoolExecutor | None = None
        self._max_trigger_workers = max(1, int(max_trigger_workers))
        self._prune = prune_expired
        self.stats = {"writes": 0, "reads": 0, "misses": 0,
                      "triggers_fired": 0}

    # ── writes ───────────────────────────────────────────────────────────────
    def post(
        self,
        key: str,
        value: Any,
        *,
        author: str = "",
        topic: str = "",
        ttl: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> BlackboardEntry:
        """Write or update a key. Updates increment the version."""
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                self._history.append(existing)
                existing.value = value
                existing.version += 1
                existing.updated_at = time.time()
                existing.author = author or existing.author
                existing.metadata.update(metadata or {})
                entry = existing
            else:
                entry = BlackboardEntry(
                    key=key, value=value, author=author, topic=topic, ttl=ttl,
                    metadata=dict(metadata or {}),
                )
                self._entries[key] = entry
                self._history.append(entry)
            self.stats["writes"] += 1
            watchers = [
                callback
                for pattern, callback in self._watchers
                if fnmatch.fnmatchcase(key, pattern)
            ]
            triggers = [
                (token, pattern, callback, sub)
                for token, (pattern, callback, sub) in self._links.items()
                if not sub.cancelled and fnmatch.fnmatchcase(key, pattern)
            ]
            # Wake condition-based waiters while still holding the lock.
            self._changed.notify_all()
        for callback in watchers:
            try:
                callback(entry)
            except Exception:  # noqa: BLE001 - a watcher must not break the write
                pass
        for token, pattern, callback, sub in triggers:
            self._fire_trigger(token, pattern, callback, sub, entry)
        return entry

    # ── reactive triggers ──────────────────────────────────────────────
    def _trigger_executor(self) -> ThreadPoolExecutor:
        with self._lock:
            if self._trigger_pool is None:
                self._trigger_pool = ThreadPoolExecutor(
                    max_workers=self._max_trigger_workers,
                    thread_name_prefix="board-trigger")
            return self._trigger_pool

    def _fire_trigger(self, token: object, pattern: str,
                      callback: Callable[[BlackboardEntry], None],
                      sub: Subscription, entry: BlackboardEntry) -> None:
        """Dispatch one trigger firing. Never raises; a failing agent must
        not break the post or poison the board."""
        def _run() -> None:
            if sub.cancelled:
                return
            try:
                callback(entry)
            except Exception as exc:  # noqa: BLE001 - isolate agent failures
                _log.warning("blackboard trigger %r failed on %s: %s",
                             pattern, entry.key, exc)
            else:
                sub.fired += 1
                with self._lock:
                    self.stats["triggers_fired"] += 1

        try:
            self._trigger_executor().submit(_run)
        except Exception as exc:  # noqa: BLE001 - pool shutdown etc.
            _log.debug("blackboard trigger dispatch failed: %s", exc)

    def link(self, pattern: str,
             callback: Callable[[BlackboardEntry], None]) -> Subscription:
        """React to posts: whenever a key matching ``pattern`` (glob) is
        written, ``callback(entry)`` runs on a worker thread.

        This is how agents trigger each other with no manual commands and
        no polling — agent A posts ``mission.alpha.done`` and every agent
        linked to ``mission.*.done`` fires. Callbacks must be quick and
        must never raise (a raising callback is logged and its firing is
        not counted). Returns a :class:`Subscription`; call ``cancel()``
        to detach.
        """
        token = object()
        sub = Subscription(self, token)
        with self._lock:
            self._links[token] = (pattern, callback, sub)
        return sub

    def _unlink(self, token: object) -> None:
        with self._lock:
            self._links.pop(token, None)

    def unlink_all(self) -> int:
        """Detach every reactive trigger. Returns how many were removed."""
        with self._lock:
            n = len(self._links)
            for _token, (_pattern, _cb, sub) in self._links.values():
                sub.cancelled = True
            self._links.clear()
            return n

    def close(self) -> None:
        """Detach triggers and shut down the trigger worker pool.

        Idempotent. The board stays readable/writable afterwards; only
        reactive dispatch stops.
        """
        with self._lock:
            pool, self._trigger_pool = self._trigger_pool, None
        self.unlink_all()
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    def append(self, key: str, value: Any, **kw: Any) -> BlackboardEntry:
        """Append to a list-valued key, creating it if absent."""
        with self._lock:
            existing = self._entries.get(key)
            current = list(existing.value) if existing and isinstance(existing.value, list) else []
            current.append(value)
            return self.post(key, current, **kw)

    # ── reads ────────────────────────────────────────────────────────────────
    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or (self._prune and entry.expired):
                self.stats["misses"] += 1
                return default
            entry.reads += 1
            self.stats["reads"] += 1
            return entry.value

    def entry(self, key: str) -> BlackboardEntry | None:
        with self._lock:
            return self._entries.get(key)

    def has(self, key: str) -> bool:
        with self._lock:
            entry = self._entries.get(key)
            return entry is not None and not entry.expired

    def keys(self, pattern: str = "*") -> list[str]:
        with self._lock:
            return sorted(
                k for k, e in self._entries.items()
                if fnmatch.fnmatchcase(k, pattern) and not e.expired
            )

    def topic(self, topic: str) -> dict[str, Any]:
        with self._lock:
            return {
                k: e.value for k, e in self._entries.items()
                if e.topic == topic and not e.expired
            }

    def snapshot(self, *, include_values: bool = True) -> dict[str, Any]:
        with self._lock:
            return {
                k: e.to_dict(include_value=include_values)
                for k, e in self._entries.items()
                if not e.expired
            }

    def most_read(self, limit: int = 10) -> list[BlackboardEntry]:
        with self._lock:
            return sorted(self._entries.values(), key=lambda e: -e.reads)[:limit]

    # ── mutation ─────────────────────────────────────────────────────────────
    def delete(self, key: str) -> bool:
        with self._lock:
            return self._entries.pop(key, None) is not None

    def prune(self) -> int:
        with self._lock:
            stale = [k for k, e in self._entries.items() if e.expired]
            for key in stale:
                del self._entries[key]
            return len(stale)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    # ── coordination ─────────────────────────────────────────────────────────
    def watch(self, pattern: str, callback: Callable[[BlackboardEntry], None]) -> None:
        """Fire ``callback`` whenever a matching key is written."""
        with self._lock:
            self._watchers.append((pattern, callback))

    def wait_for(self, key: str, *, timeout: float = 10.0) -> Any:
        """Block until ``key`` appears. Returns None on timeout.

        Wakes the instant the key is posted (condition variable), not on a
        poll interval.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        with self._changed:
            while True:
                entry = self._entries.get(key)
                if entry is not None and not (self._prune and entry.expired):
                    entry.reads += 1
                    self.stats["reads"] += 1
                    return entry.value
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.stats["misses"] += 1
                    return None
                self._changed.wait(timeout=remaining)

    def gather(self, keys: Iterable[str]) -> dict[str, Any]:
        """Read many keys at once, skipping absent ones."""
        out: dict[str, Any] = {}
        for key in keys:
            if self.has(key):
                out[key] = self.get(key)
        return out

    def stats_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {**self.stats, "entries": len(self._entries),
                    "watchers": len(self._watchers),
                    "triggers": len(self._links)}

    # ── persistence ────────────────────────────────────────────────────
    @staticmethod
    def _json_safe(value: Any) -> Any:
        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            pass
        if isinstance(value, dict):
            return {str(k): Blackboard._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [Blackboard._json_safe(v) for v in value]
        return str(value)

    def save(self, path: str | Path) -> dict[str, Any]:
        """Persist the board to ``path`` as JSON. Best-effort: values that
        are not JSON-serializable are coerced (dicts/lists recursed,
        everything else stringified). Returns ``{"ok", "entries",
        "path"}`` — never raises."""
        try:
            with self._lock:
                payload = {
                    "saved_at": time.time(),
                    "entries": [
                        {
                            "key": e.key,
                            "value": self._json_safe(e.value),
                            "author": e.author,
                            "topic": e.topic,
                            "version": e.version,
                            "created_at": e.created_at,
                            "updated_at": e.updated_at,
                            "reads": e.reads,
                            "ttl": e.ttl,
                            "metadata": self._json_safe(e.metadata),
                        }
                        for e in self._entries.values()
                        if not e.expired
                    ],
                }
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False),
                           encoding="utf-8")
            tmp.replace(target)
            return {"ok": True, "entries": len(payload["entries"]),
                    "path": str(target)}
        except Exception as exc:  # noqa: BLE001 - persistence never breaks the board
            _log.warning("blackboard save failed: %s", exc)
            return {"ok": False, "entries": 0, "path": str(path),
                    "error": str(exc)}

    def load(self, path: str | Path) -> dict[str, Any]:
        """Restore entries saved by :meth:`save`. Existing keys are
        overwritten (as new versions); watchers/triggers are *not*
        fired for restored entries — a restore is state recovery, not
        new work. Returns ``{"ok", "entries", "path"}`` — never raises."""
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            items = raw.get("entries") if isinstance(raw, dict) else None
            if not isinstance(items, list):
                return {"ok": False, "entries": 0, "path": str(path),
                        "error": "not a blackboard snapshot"}
            restored = 0
            with self._lock:
                for item in items:
                    if not isinstance(item, dict) or not item.get("key"):
                        continue
                    key = str(item["key"])
                    existing = self._entries.get(key)
                    if existing is not None:
                        self._history.append(existing)
                        existing.value = item.get("value")
                        existing.version = int(item.get("version") or 1) + 1
                        existing.updated_at = time.time()
                        existing.author = str(item.get("author") or "")
                        existing.topic = str(item.get("topic") or "")
                        existing.reads = int(item.get("reads") or 0)
                        existing.ttl = float(item.get("ttl") or 0.0)
                        meta = item.get("metadata")
                        existing.metadata = dict(meta) if isinstance(meta, dict) else {}
                    else:
                        self._entries[key] = BlackboardEntry(
                            key=key, value=item.get("value"),
                            author=str(item.get("author") or ""),
                            topic=str(item.get("topic") or ""),
                            version=int(item.get("version") or 1),
                            created_at=float(item.get("created_at") or time.time()),
                            updated_at=float(item.get("updated_at") or time.time()),
                            reads=int(item.get("reads") or 0),
                            ttl=float(item.get("ttl") or 0.0),
                            metadata=dict(item.get("metadata"))
                            if isinstance(item.get("metadata"), dict) else {},
                        )
                        self._history.append(self._entries[key])
                    restored += 1
                self._changed.notify_all()
            return {"ok": True, "entries": restored, "path": str(path)}
        except Exception as exc:  # noqa: BLE001 - restore never breaks the board
            _log.warning("blackboard load failed: %s", exc)
            return {"ok": False, "entries": 0, "path": str(path),
                    "error": str(exc)}

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, key: str) -> bool:
        return self.has(key)
