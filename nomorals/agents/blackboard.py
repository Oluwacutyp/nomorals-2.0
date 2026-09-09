"""Shared scratch space for cooperating agents.

A blackboard is the classic multi-agent coordination pattern: agents do not pass
messages point-to-point, they publish partial results to a shared store and read
what others have contributed. It suits this system because a research agent's
output is genuinely useful to the coding, writing, and critique agents alike, and
wiring that as explicit edges would make the graph unreadable.

Entries are versioned and access-counted, so the orchestrator can tell which
contributions were actually used — that signal feeds the reflector.
"""

from __future__ import annotations

import fnmatch
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from ..core.ids import new_id

__all__ = ["Blackboard", "BlackboardEntry"]


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


class Blackboard:
    """Thread-safe shared store with watch support."""

    def __init__(self, *, prune_expired: bool = True) -> None:
        self._entries: dict[str, BlackboardEntry] = {}
        self._history: list[BlackboardEntry] = []
        self._watchers: list[tuple[str, Callable[[BlackboardEntry], None]]] = []
        self._lock = threading.RLock()
        self._prune = prune_expired
        self.stats = {"writes": 0, "reads": 0, "misses": 0}

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
        for callback in watchers:
            try:
                callback(entry)
            except Exception:  # noqa: BLE001 - a watcher must not break the write
                pass
        return entry

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

    def wait_for(self, key: str, *, timeout: float = 10.0, poll: float = 0.02) -> Any:
        """Block until ``key`` appears. Returns None on timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.has(key):
                return self.get(key)
            time.sleep(poll)
        return self.get(key)

    def gather(self, keys: Iterable[str]) -> dict[str, Any]:
        """Read many keys at once, skipping absent ones."""
        out: dict[str, Any] = {}
        for key in keys:
            if self.has(key):
                out[key] = self.get(key)
        return out

    def stats_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {**self.stats, "entries": len(self._entries), "watchers": len(self._watchers)}

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, key: str) -> bool:
        return self.has(key)
