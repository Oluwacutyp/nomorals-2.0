"""Model lifecycle for voice backends — load / LRU-evict / recover.

The storyteller-ml pattern, ported as *policy*, not topology: in
FakeYou's stack a Rust orchestrator owned GPU memory — models were
dumb sidecars, loaded into an LRU cache, and the orchestrator could
instruct eviction (``maybe_clear_synthesizer_checkpoint_path``).
Devon's backends run in-process, so there is no orchestrator process
to build; what ports is the *governed seam*:

- **Policy as data** — :class:`LifecyclePolicy` (how many models may
  sit loaded, an optional total-bytes budget) is a frozen dataclass
  a caller can read, diff, and override, not constants buried in a
  backend.
- **One seam for load/evict** — a backend registers its models with
  :meth:`ModelLifecycleManager.acquire` instead of keeping an ad-hoc
  dict (PiperBackend's ``_voices`` was exactly that: unbounded, no
  eviction, no visibility). Eviction calls the entry's release hook.
- **Failure is a state, not a surprise** — a loader that raises
  leaves the entry ``failed`` with the reason recorded; the next
  ``acquire`` retries the load (recovery), and the attempt/failure
  counts stay on the entry.
- **State as data** — :meth:`ModelLifecycleManager.status` returns
  the whole picture (policy, per-entry state, recency, sizes,
  failure reasons) for capability reports. No prose.

Thread-safe: the live voice loop and file synthesis share engines,
so all mutations run under one lock. Loaders run under the lock too
— loads here are local model constructions (ONNX sessions), and a
half-loaded entry visible to another thread would be worse than a
brief wait. Stdlib only.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .failures import record_failure

__all__ = ["LifecyclePolicy", "ModelEntry", "ModelLifecycleManager"]


@dataclass(frozen=True)
class LifecyclePolicy:
    """How much may sit loaded at once. Data, not constants."""

    #: Maximum simultaneously loaded models; beyond it the
    #: least-recently-used *other* entry is evicted on acquire.
    max_loaded: int = 4
    #: Optional total budget across loaded models' declared sizes
    #: (bytes). 0 = no byte budget. Entries that declare no size
    #: count as 0 — the budget governs what it can measure.
    max_total_bytes: int = 0


@dataclass
class ModelEntry:
    """One model's lifecycle record. ``handle`` is the live object;
    it is never exposed through :meth:`ModelLifecycleManager.status`.
    """

    key: str
    state: str = "absent"          # loaded | failed | evicted | absent
    handle: Any = None
    size_bytes: int = 0
    loads: int = 0                 # successful loads
    attempts: int = 0              # load attempts (loads + failures)
    failures: int = 0
    evictions: int = 0
    last_used: float = 0.0         # clock() at last acquire/peek-hit
    loaded_at: float = 0.0
    last_error: str = ""           # most recent load failure, named
    _release: Optional[Callable[[Any], None]] = field(
        default=None, repr=False)


class ModelLifecycleManager:
    """Loads, caches, evicts, and recovers models under a policy."""

    def __init__(self, policy: LifecyclePolicy | None = None, *,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.policy = policy or LifecyclePolicy()
        self._clock = clock
        self._entries: dict[str, ModelEntry] = {}
        self._lock = threading.RLock()

    # -- core ------------------------------------------------------

    def acquire(self, key: str, loader: Callable[[], Any], *,
                release: Optional[Callable[[Any], None]] = None,
                size_bytes: int = 0) -> Any:
        """Return the loaded handle for ``key``, loading on need.

        Cache hit → the handle, recency refreshed. Miss / evicted /
        failed → ``loader()`` runs; success registers the entry
        ``loaded`` and enforces the policy (LRU evictions); failure
        registers it ``failed`` with the reason and re-raises the
        loader's own exception — the next ``acquire`` retries, which
        is the recovery path. ``release`` is stored and called with
        the handle when the entry is evicted.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.state == "loaded":
                entry.last_used = self._clock()
                return entry.handle
            if entry is None:
                entry = ModelEntry(key=key)
                self._entries[key] = entry
            entry.attempts += 1
            try:
                handle = loader()
            except Exception as exc:
                entry.state = "failed"
                entry.handle = None
                entry.failures += 1
                entry.last_error = f"{type(exc).__name__}: {exc}"
                record_failure(f"lifecycle.acquire:{key}", exc)
                raise
            entry.handle = handle
            entry.state = "loaded"
            entry.loads += 1
            entry.size_bytes = int(size_bytes or 0)
            entry._release = release
            entry.loaded_at = entry.last_used = self._clock()
            entry.last_error = ""
            self._enforce_policy(keep=key)
            return handle

    def peek(self, key: str) -> Any:
        """The loaded handle for ``key`` or None — never loads.

        Does not refresh recency (a peek is an observation, not a
        use); :meth:`acquire` is the use that counts for LRU.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.state == "loaded":
                return entry.handle
            return None

    def evict(self, key: str, *, reason: str = "manual") -> bool:
        """Evict one loaded entry, calling its release hook.

        Returns True when something was evicted. A release hook that
        raises is recorded in the voice failure ledger; the entry is
        still evicted — a model that refuses to close must not become
        immortal.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.state != "loaded":
                return False
            self._evict_entry(entry, reason)
            return True

    def evict_all(self, *, reason: str = "shutdown") -> int:
        """Evict every loaded entry. Returns the eviction count."""
        with self._lock:
            count = 0
            for entry in list(self._entries.values()):
                if entry.state == "loaded":
                    self._evict_entry(entry, reason)
                    count += 1
            return count

    def _evict_entry(self, entry: ModelEntry, reason: str) -> None:
        handle, release = entry.handle, entry._release
        entry.handle = None
        entry.state = "evicted"
        entry.evictions += 1
        if release is not None and handle is not None:
            try:
                release(handle)
            except Exception as exc:  # noqa: BLE001 — recorded, not raised
                record_failure(f"lifecycle.evict:{entry.key}:{reason}",
                               exc)

    def _enforce_policy(self, *, keep: str) -> None:
        """Evict LRU entries (never ``keep``) until policy holds."""
        while True:
            loaded = [e for e in self._entries.values()
                      if e.state == "loaded"]
            over_count = len(loaded) > self.policy.max_loaded
            total = sum(e.size_bytes for e in loaded)
            over_bytes = (self.policy.max_total_bytes > 0
                          and total > self.policy.max_total_bytes)
            if not (over_count or over_bytes):
                return
            victims = [e for e in loaded if e.key != keep]
            if not victims:
                return
            victim = min(victims, key=lambda e: e.last_used)
            self._evict_entry(victim, "lru")

    # -- observation -------------------------------------------------

    def state_of(self, key: str) -> str:
        """``loaded`` | ``failed`` | ``evicted`` | ``absent``."""
        with self._lock:
            entry = self._entries.get(key)
            return entry.state if entry is not None else "absent"

    def loaded_keys(self) -> list[str]:
        """Loaded keys, most-recently-used first."""
        with self._lock:
            loaded = [e for e in self._entries.values()
                      if e.state == "loaded"]
            loaded.sort(key=lambda e: e.last_used, reverse=True)
            return [e.key for e in loaded]

    def status(self) -> dict[str, Any]:
        """The lifecycle as data, for capability reports.

        Entries are listed most-recently-used first; handles are
        never included. ``last_used_ago_s`` is measured on the
        manager's clock at call time.
        """
        with self._lock:
            now = self._clock()
            entries = sorted(self._entries.values(),
                             key=lambda e: e.last_used, reverse=True)
            loaded = [e for e in entries if e.state == "loaded"]
            return {
                "policy": {
                    "max_loaded": self.policy.max_loaded,
                    "max_total_bytes": self.policy.max_total_bytes,
                },
                "loaded": len(loaded),
                "total_bytes": sum(e.size_bytes for e in loaded),
                "entries": [{
                    "key": e.key,
                    "state": e.state,
                    "size_bytes": e.size_bytes,
                    "loads": e.loads,
                    "attempts": e.attempts,
                    "failures": e.failures,
                    "evictions": e.evictions,
                    "last_used_ago_s": (round(now - e.last_used, 3)
                                        if e.last_used else None),
                    "last_error": e.last_error,
                } for e in entries],
            }
