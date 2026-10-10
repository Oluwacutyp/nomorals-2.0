"""Sync engine: push/pull replication between stores.

A device pushes its changes to a peer and pulls the peer's changes,
merging with last-write-wins (HLC-ordered, per-field). Progress is tracked
per peer as monotonic sequence cursors (see :mod:`.store`), so sync is
incremental and can never miss a backdated write. The peer is usually the
cloud hub; devices never need to talk to each other directly.

Reliability discipline (borrowed from CouchDB/PouchDB replication):

- **Chunked, checkpointed phases.** Pushes go out in bounded chunks and the
  push cursor is saved after every chunk; pulls save the pull cursor
  periodically while applying. A crash mid-sync resumes where it stopped —
  records are never stranded and never lost (re-push is idempotent).
- **Retry with backoff.** Transient transport failures retry with
  exponential backoff + jitter; auth failures fail fast (no point retrying
  a bad token).
- **History.** Every run is recorded in ``sync_history`` — ``status()``
  shows the last run, ``history()`` shows the recent past.
- **Live mode.** :class:`AutoSync` keeps a peer continuously synced:
  debounced re-sync on every local change plus interval polling, with
  backoff when the peer is down (PouchDB ``live:true, retry:true`` style).
"""

from __future__ import annotations

import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.events import Event, global_bus
from ..core.logging_setup import get_logger
from ..storage.db import Database
from .errors import SyncAuthError, SyncConnectionError, SyncError
from .store import SyncRecord, SyncStore, _records_equal

__all__ = [
    "SyncPeer", "LocalPeer", "SyncEngine", "SyncResult", "SyncPreview",
    "AutoSync", "PROGRESS_TABLE", "HISTORY_TABLE",
    "PUSH_CHUNK", "PULL_CHUNK",
]

_log = get_logger(__name__)


def _emit(topic: str, data: dict[str, Any]) -> None:
    """Publish a telemetry event. Best-effort: a broken bus or subscriber
    must never break a sync run (fail-open telemetry, fail-closed
    function)."""
    try:
        global_bus.publish(Event(topic=topic, data=data, source=__name__))
    except Exception:  # noqa: BLE001 - telemetry is fail-open
        _log.debug("event %s failed", topic, exc_info=True)


PROGRESS_TABLE = "sync_progress"
HISTORY_TABLE = "sync_history"

#: Records per push chunk. Each chunk is checkpointed separately, so a
#: crash can only ever re-push one chunk (idempotent).
PUSH_CHUNK = 500
#: Records applied between pull-cursor checkpoints.
PULL_CHUNK = 500
#: Keys shown in a preview before "+N more".
PREVIEW_KEY_CAP = 50

DIRECTIONS = ("push", "pull", "both")

# ── presentation themes ────────────────────────────────────────────────
# Glyph sets only — no aesthetic baked into the logic. "rich" for
# terminals, "plain" for ASCII logs, "compact" for one-line status bars.

_THEMES: dict[str, dict[str, str]] = {
    "rich": {
        "ok": "✓", "fail": "✗", "warn": "⚠", "dry": "◌",
        "up": "↑", "down": "↓", "merge": "⑂", "time": "⏱",
        "arrow": "→", "sep": "·",
    },
    "plain": {
        "ok": "[OK]", "fail": "[FAIL]", "warn": "[WARN]", "dry": "[DRY]",
        "up": "pushed", "down": "pulled", "merge": "conflicts",
        "time": "t=", "arrow": "->", "sep": "|",
    },
    "compact": {
        "ok": "+", "fail": "!", "warn": "~", "dry": "?",
        "up": "+", "down": "-", "merge": "~", "time": "",
        "arrow": ">", "sep": " ",
    },
}


def _theme(style: str) -> dict[str, str]:
    return _THEMES.get(style, _THEMES["rich"])


def _ago(ts: float, now: float | None = None) -> str:
    if not ts:
        return "never"
    d = (time.time() if now is None else now) - ts
    if d < 0:
        return "just now"
    if d < 60:
        return f"{int(d)}s ago"
    if d < 3600:
        return f"{int(d / 60)}m ago"
    if d < 86400:
        return f"{int(d / 3600)}h ago"
    return f"{int(d / 86400)}d ago"


class SyncPeer(ABC):
    """What the engine needs from the other side."""

    @abstractmethod
    def push_records(self, records: list[SyncRecord]) -> int:
        """Accept records. Returns count applied."""
        ...

    @abstractmethod
    def fetch_since(self, since: float) -> list[SyncRecord]:
        """Return records changed after ``since`` (timestamp cursor)."""
        ...

    def fetch_since_seq(self, seq: int) -> list[SyncRecord]:
        """Return records written after replication cursor ``seq``.

        Peers backed by a :class:`SyncStore` override this. The default
        raises :exc:`NotImplementedError` and the engine falls back to
        :meth:`fetch_since` — a legacy peer keeps working, it just can't
        see backdated writes.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support seq cursors")

    def ping(self) -> bool:
        """Cheap liveness probe. Optional — default is unsupported."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support ping")


class LocalPeer(SyncPeer):
    """A peer backed by another SyncStore (same machine, or tests)."""

    def __init__(self, store: SyncStore) -> None:
        self.store = store

    def push_records(self, records: list[SyncRecord]) -> int:
        n = 0
        for rec in records:
            if self.store.apply(rec):
                n += 1
        return n

    def fetch_since(self, since: float) -> list[SyncRecord]:
        return self.store.list_changed_since(since)

    def fetch_since_seq(self, seq: int) -> list[SyncRecord]:
        return self.store.list_since_seq(seq)

    def ping(self) -> bool:
        return True


@dataclass
class SyncResult:
    pushed: int = 0
    pulled: int = 0
    conflicts_resolved: int = 0
    duration_s: float = 0.0
    peer_id: str = "hub"
    direction: str = "both"
    ok: bool = True
    error: str = ""
    dry_run: bool = False

    def to_dict(self) -> dict:
        return {
            "peer_id": self.peer_id,
            "direction": self.direction,
            "pushed": self.pushed,
            "pulled": self.pulled,
            "conflicts_resolved": self.conflicts_resolved,
            "duration_s": round(self.duration_s, 3),
            "ok": self.ok,
            "error": self.error,
            "dry_run": self.dry_run,
        }

    @property
    def total(self) -> int:
        return self.pushed + self.pulled

    @property
    def rate(self) -> float:
        """Records per second (0 when nothing moved)."""
        if self.duration_s <= 0:
            return 0.0
        return self.total / self.duration_s

    def format(self, style: str = "rich") -> str:
        """Human-readable one- or two-line sync report."""
        t = _theme(style)
        if style == "compact":
            mark = t["ok"] if self.ok else t["fail"]
            bits = (f"{t['up']}{self.pushed} {t['down']}{self.pulled} "
                    f"{t['merge']}{self.conflicts_resolved} "
                    f"{self.duration_s:.2f}s")
            if self.dry_run:
                bits = f"{t['dry']} dry-run {bits}"
            if self.error:
                bits += f" {self.error}"
            return f"sync {self.peer_id} {mark} {bits}".strip()

        if not self.ok:
            return (f"{t['fail']} sync {self.peer_id} failed "
                    f"({self.duration_s:.2f}s): {self.error}")
        head = f"{t['ok']} sync {self.peer_id}"
        if self.dry_run:
            head = f"{t['dry']} dry run {self.peer_id} — nothing was sent"
        detail = (f"{t['up']} {self.pushed} pushed {t['sep']} "
                  f"{t['down']} {self.pulled} pulled {t['sep']} "
                  f"{t['merge']} {self.conflicts_resolved} conflicts")
        meta = f"{t['time']}{self.duration_s:.2f}s"
        if self.rate > 0:
            meta += f" {t['sep']} {self.rate:.1f} rec/s"
        if style == "plain":
            return f"{head} [{meta}]: {detail}"
        return f"{head}\n  {detail}\n  {meta}"


@dataclass
class SyncPreview:
    """What :meth:`SyncEngine.sync` would do, without doing it."""

    peer_id: str = "hub"
    direction: str = "both"
    push_keys: list[str] = field(default_factory=list)
    pull_keys: list[str] = field(default_factory=list)
    push_total: int = 0
    pull_total: int = 0
    would_conflict: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "peer_id": self.peer_id,
            "direction": self.direction,
            "push_total": self.push_total,
            "pull_total": self.pull_total,
            "push_keys": list(self.push_keys),
            "pull_keys": list(self.pull_keys),
            "would_conflict": list(self.would_conflict),
        }

    def format(self, style: str = "rich") -> str:
        t = _theme(style)

        def keys(keys: list[str], total: int) -> str:
            shown = ", ".join(keys[:PREVIEW_KEY_CAP])
            extra = total - len(keys[:PREVIEW_KEY_CAP])
            if extra > 0:
                shown += f" (+{extra} more)" if shown else f"+{extra} more"
            return shown or "—"

        if style == "compact":
            return (f"preview {self.peer_id}: "
                    f"{t['up']}{self.push_total} {t['down']}{self.pull_total} "
                    f"{t['merge']}{len(self.would_conflict)}")
        lines = [f"{t['dry']} preview {self.peer_id} ({self.direction})"]
        if self.direction in ("push", "both"):
            lines.append(f"  {t['up']} would push {self.push_total}: "
                         f"{keys(self.push_keys, self.push_total)}")
        if self.direction in ("pull", "both"):
            lines.append(f"  {t['down']} would pull {self.pull_total}: "
                         f"{keys(self.pull_keys, self.pull_total)}")
            if self.would_conflict:
                lines.append(f"  {t['merge']} would conflict "
                             f"({len(self.would_conflict)}): "
                             f"{keys(self.would_conflict, len(self.would_conflict))}")
        return "\n".join(lines)


class SyncEngine:
    """Incremental two-way sync between a local store and a peer."""

    def __init__(self, db: Database, store: SyncStore, *,
                 retry_attempts: int = 3,
                 retry_base_s: float = 1.0,
                 retry_max_s: float = 60.0) -> None:
        self.db = db
        self.store = store
        self.retry_attempts = max(0, int(retry_attempts))
        self.retry_base_s = max(0.0, float(retry_base_s))
        self.retry_max_s = max(self.retry_base_s, float(retry_max_s))
        self._ensure_schema()

    # ── schema ─────────────────────────────────────────────────────────

    def _ensure_schema(self) -> None:
        self.db.execute(
            f"""CREATE TABLE IF NOT EXISTS {PROGRESS_TABLE} (
                peer_id TEXT PRIMARY KEY,
                last_push REAL NOT NULL DEFAULT 0,
                last_pull REAL NOT NULL DEFAULT 0,
                push_seq INTEGER NOT NULL DEFAULT 0,
                pull_seq INTEGER NOT NULL DEFAULT 0
            )"""
        )
        self._ensure_seq_columns()
        self.db.execute(
            f"""CREATE TABLE IF NOT EXISTS {HISTORY_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                peer_id TEXT NOT NULL,
                started_at REAL NOT NULL,
                duration_s REAL NOT NULL DEFAULT 0,
                direction TEXT NOT NULL DEFAULT 'both',
                pushed INTEGER NOT NULL DEFAULT 0,
                pulled INTEGER NOT NULL DEFAULT 0,
                conflicts INTEGER NOT NULL DEFAULT 0,
                ok INTEGER NOT NULL DEFAULT 1,
                error TEXT NOT NULL DEFAULT ''
            )"""
        )
        self.db.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{HISTORY_TABLE}_peer "
            f"ON {HISTORY_TABLE}(peer_id, id)"
        )

    def _ensure_seq_columns(self) -> None:
        """Migration: pre-seq progress rows carry timestamp cursors.

        The push cursor is translated once into a seq cursor (max local seq
        at or before the old timestamp mark) — same seq space, exact. The
        pull cursor lives in the *peer's* seq space, which isn't visible
        here, so it resets to 0: the first post-upgrade pull re-fetches from
        the peer's start, and every already-seen record is an idempotent
        no-op on apply. Rows already carrying a seq cursor are untouched.
        """
        from .store import TABLE as STORE_TABLE, ensure_column

        for col in ("push_seq", "pull_seq"):
            ensure_column(self.db, PROGRESS_TABLE, col,
                          "INTEGER NOT NULL DEFAULT 0")
        with self.db.transaction():
            stale = self.db.query(
                f"""SELECT peer_id, last_push, push_seq
                    FROM {PROGRESS_TABLE}
                    WHERE push_seq = 0 AND last_push > 0"""
            )
            for row in stale:
                push_seq = int(self.db.scalar(
                    f"SELECT COALESCE(MAX(seq), 0) FROM {STORE_TABLE} "
                    "WHERE updated_at <= ?",
                    (row["last_push"],), default=0))
                self.db.execute(
                    f"""UPDATE {PROGRESS_TABLE}
                        SET push_seq = ?
                        WHERE peer_id = ?""",
                    (push_seq, row["peer_id"]),
                )
            if stale:
                _log.info("sync: migrated %d push cursors to seq",
                          len(stale))

    # ── progress ───────────────────────────────────────────────────────

    def _progress(self, peer_id: str) -> tuple[int, int, float, float]:
        """(push_seq, pull_seq, last_push_ts, last_pull_ts) for a peer."""
        row = self.db.query_one(
            f"SELECT * FROM {PROGRESS_TABLE} WHERE peer_id=?", (peer_id,)
        )
        if not row:
            return 0, 0, 0.0, 0.0
        return (
            int(row.get("push_seq") or 0),
            int(row.get("pull_seq") or 0),
            float(row.get("last_push") or 0.0),
            float(row.get("last_pull") or 0.0),
        )

    def _save_progress(
        self,
        peer_id: str,
        push_seq: int,
        pull_seq: int,
        last_push_ts: float = 0.0,
        last_pull_ts: float = 0.0,
    ) -> None:
        self.db.execute(
            f"""INSERT INTO {PROGRESS_TABLE}
                    (peer_id, last_push, last_pull, push_seq, pull_seq)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(peer_id) DO UPDATE SET
                    last_push=excluded.last_push,
                    last_pull=excluded.last_pull,
                    push_seq=excluded.push_seq,
                    pull_seq=excluded.pull_seq""",
            (peer_id, last_push_ts, last_pull_ts, push_seq, pull_seq),
        )

    # ── retry ──────────────────────────────────────────────────────────

    def _retryable(self, exc: BaseException) -> bool:
        if isinstance(exc, SyncAuthError):
            return False  # a bad token never heals itself
        if isinstance(exc, SyncConnectionError):
            return True
        return bool(getattr(exc, "retryable", False))

    def _with_retry(self, label: str, fn: Callable[[], Any]) -> Any:
        """Run ``fn``; retry transient failures with backoff + jitter."""
        attempt = 0
        while True:
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - classified below
                if not isinstance(exc, SyncError) or not self._retryable(exc):
                    raise
                attempt += 1
                if attempt > self.retry_attempts:
                    raise
                delay = min(self.retry_base_s * (2.0 ** (attempt - 1)),
                            self.retry_max_s)
                delay *= 0.5 + random.random()  # full-ish jitter
                _log.warning("sync %s failed (attempt %d/%d): %s — "
                             "retrying in %.1fs",
                             label, attempt, self.retry_attempts, exc, delay)
                time.sleep(delay)

    def _progress_event(self, phase: str, peer_id: str,
                        done: int, total: int) -> None:
        _emit(f"sync.{phase}_progress", {
            "peer_id": peer_id, "phase": phase,
            "done": done, "total": total,
        })

    # ── main entry points ──────────────────────────────────────────────

    def sync(
        self,
        peer: SyncPeer,
        peer_id: str = "hub",
        *,
        direction: str = "both",
        dry_run: bool = False,
        on_progress: Callable[..., None] | None = None,
        chunk_size: int = PUSH_CHUNK,
    ) -> SyncResult:
        """Replicate with a peer.

        ``direction`` is ``"push"``, ``"pull"`` or ``"both"`` (one-way
        modes skip the other phase entirely). ``dry_run`` returns what
        *would* happen without touching either side. ``on_progress`` is
        called as ``on_progress(phase, done, total, peer_id=...)`` with
        phase ``"push"``/``"pull"`` after every chunk, so live UIs can
        render real progress bars.
        """
        if not peer_id:
            raise ValueError("peer_id is required")
        direction = direction.lower()
        if direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}")
        chunk_size = max(1, int(chunk_size))
        started = time.time()
        _emit("sync.started", {"peer_id": peer_id, "direction": direction,
                               "dry_run": dry_run})

        if dry_run:
            preview = self.preview(peer, peer_id, direction=direction)
            return SyncResult(
                peer_id=peer_id, direction=direction, dry_run=True,
                pushed=preview.push_total, pulled=preview.pull_total,
                conflicts_resolved=len(preview.would_conflict),
                duration_s=time.time() - started,
            )

        do_push = direction in ("push", "both")
        do_pull = direction in ("pull", "both")
        pushed = pulled = conflicts = 0
        try:
            if do_push:
                pushed = self._push(peer, peer_id, on_progress, chunk_size)
            if do_pull:
                pulled, conflicts = self._pull(
                    peer, peer_id, on_progress, chunk_size)
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
            duration = time.time() - started
            self._record_run(peer_id, direction, started, duration,
                             pushed, pulled, conflicts,
                             ok=False, error=str(exc))
            _emit("sync.failed", {"peer_id": peer_id, "error": str(exc),
                                  "duration_s": duration})
            _log.warning("sync with %s failed: %s", peer_id, exc)
            raise

        duration = time.time() - started
        self._record_run(peer_id, direction, started, duration,
                         pushed, pulled, conflicts, ok=True)
        result = SyncResult(
            peer_id=peer_id, direction=direction,
            pushed=pushed, pulled=pulled, conflicts_resolved=conflicts,
            duration_s=duration,
        )
        _log.info("sync with %s: pushed=%d pulled=%d conflicts=%d (%.2fs)",
                  peer_id, pushed, pulled, conflicts, duration)
        _emit("sync.completed", {
            "peer_id": peer_id, "direction": direction,
            "pushed": pushed, "pulled": pulled,
            "conflicts_resolved": conflicts, "duration_s": duration,
        })
        return result

    def sync_all(self, peers: dict[str, SyncPeer],
                 **kwargs: Any) -> dict[str, SyncResult]:
        """Sync with several peers, capturing per-peer failures.

        A peer that raises gets ``SyncResult(ok=False, error=...)`` instead
        of aborting the rest — the returned dict always has one entry per
        peer id.
        """
        results: dict[str, SyncResult] = {}
        for pid, peer in peers.items():
            try:
                results[pid] = self.sync(peer, peer_id=pid, **kwargs)
            except Exception as exc:  # noqa: BLE001 - per-peer capture
                _log.warning("sync_all: peer %s failed: %s", pid, exc)
                results[pid] = SyncResult(
                    peer_id=pid, direction=kwargs.get("direction", "both"),
                    ok=False, error=str(exc))
        return results

    def preview(self, peer: SyncPeer, peer_id: str = "hub", *,
                direction: str = "both") -> SyncPreview:
        """What a sync would do — counts and key lists, no mutations.

        Note: the pull side genuinely fetches from the peer (network for
        HTTP peers); nothing is applied locally.
        """
        if not peer_id:
            raise ValueError("peer_id is required")
        direction = direction.lower()
        if direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}")
        push_seq, pull_seq, _, last_pull_ts = self._progress(peer_id)
        pv = SyncPreview(peer_id=peer_id, direction=direction)
        if direction in ("push", "both"):
            pv.push_total = self.store.count_since_seq(push_seq)
            pv.push_keys = [r.key for r in self.store.list_since_seq(
                push_seq, limit=PREVIEW_KEY_CAP)]
        if direction in ("pull", "both"):
            try:
                incoming = peer.fetch_since_seq(pull_seq)
            except NotImplementedError:
                incoming = peer.fetch_since(last_pull_ts)
            pv.pull_total = len(incoming)
            pv.pull_keys = [r.key for r in incoming[:PREVIEW_KEY_CAP]]
            for rec in incoming:
                current = self.store.get_record(rec.key)
                if current is not None and not _records_equal(current, rec):
                    pv.would_conflict.append(rec.key)
        return pv

    # ── phases ─────────────────────────────────────────────────────────

    def _push(self, peer: SyncPeer, peer_id: str,
              on_progress: Callable[..., None] | None,
              chunk_size: int) -> int:
        """Push in chunks, checkpointing the cursor after each chunk."""
        push_seq, pull_seq, last_push_ts, last_pull_ts = self._progress(
            peer_id)
        total = self.store.count_since_seq(push_seq)
        pushed = done = 0
        cursor, cursor_ts = push_seq, last_push_ts
        while True:
            chunk = self.store.list_since_seq(cursor, limit=chunk_size)
            if not chunk:
                break
            n = self._with_retry(
                f"push to {peer_id}",
                lambda: peer.push_records(chunk))
            pushed += n
            done += len(chunk)
            cursor = max(r.seq for r in chunk)
            cursor_ts = max([r.updated_at for r in chunk],
                            default=cursor_ts)
            # Checkpoint: a crash from here resumes after this chunk.
            # Re-push of a chunk is idempotent, so a failure *inside*
            # the chunk only ever repeats one chunk.
            self._save_progress(peer_id, cursor, pull_seq,
                                cursor_ts, last_pull_ts)
            self._progress_event("push", peer_id, done, total)
            if on_progress is not None:
                on_progress("push", done, total, peer_id=peer_id)
        return pushed

    def _pull(self, peer: SyncPeer, peer_id: str,
              on_progress: Callable[..., None] | None,
              chunk_size: int) -> tuple[int, int]:
        """Pull and merge, checkpointing the pull cursor periodically."""
        push_seq, pull_seq, last_push_ts, last_pull_ts = self._progress(
            peer_id)
        try:
            incoming = self._with_retry(
                f"pull from {peer_id}",
                lambda: peer.fetch_since_seq(pull_seq))
            legacy = False
        except NotImplementedError:
            incoming = self._with_retry(
                f"pull from {peer_id}",
                lambda: peer.fetch_since(last_pull_ts))
            legacy = True

        total = len(incoming)
        pulled = conflicts = done = 0
        new_pull_seq, new_pull_ts = pull_seq, last_pull_ts
        for rec in incoming:
            before = self.store.get_record(rec.key)
            if self.store.apply(rec):
                pulled += 1
                if before is not None:
                    conflicts += 1
            if legacy:
                new_pull_ts = max(new_pull_ts, rec.updated_at)
            else:
                new_pull_seq = max(new_pull_seq, rec.seq)
            done += 1
            if done % chunk_size == 0:
                self._save_progress(peer_id, push_seq, new_pull_seq,
                                    last_push_ts, new_pull_ts)
                self._progress_event("pull", peer_id, done, total)
                if on_progress is not None:
                    on_progress("pull", done, total, peer_id=peer_id)
        self._save_progress(peer_id, push_seq, new_pull_seq,
                            last_push_ts, new_pull_ts)
        self._progress_event("pull", peer_id, done, total)
        if on_progress is not None:
            on_progress("pull", done, total, peer_id=peer_id)
        return pulled, conflicts

    # ── history & status ───────────────────────────────────────────────

    def _record_run(self, peer_id: str, direction: str, started_at: float,
                    duration_s: float, pushed: int, pulled: int,
                    conflicts: int, *, ok: bool, error: str = "") -> None:
        try:
            self.db.execute(
                f"""INSERT INTO {HISTORY_TABLE}
                        (peer_id, started_at, duration_s, direction,
                         pushed, pulled, conflicts, ok, error)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (peer_id, started_at, duration_s, direction, pushed,
                 pulled, conflicts, 1 if ok else 0, error),
            )
        except Exception:  # noqa: BLE001 - history is observability, not load-bearing
            _log.debug("sync history write failed", exc_info=True)

    def history(self, peer_id: str = "hub", limit: int = 20) -> list[dict]:
        """Most recent sync runs for a peer, newest first."""
        rows = self.db.query(
            f"""SELECT * FROM {HISTORY_TABLE}
                WHERE peer_id=? ORDER BY id DESC LIMIT ?""",
            (peer_id, max(1, int(limit))),
        )
        out = []
        for r in rows:
            d = dict(r)
            d["ok"] = bool(d.get("ok"))
            out.append(d)
        return out

    def last_run(self, peer_id: str = "hub") -> dict | None:
        """The most recent sync run for a peer, or None if never synced."""
        rows = self.history(peer_id, limit=1)
        return rows[0] if rows else None

    def status(self, peer_id: str = "hub") -> dict:
        push_seq, pull_seq, last_push_ts, last_pull_ts = self._progress(peer_id)
        return {
            "peer_id": peer_id,
            "device_id": self.store.device_id,
            "local_keys": self.store.count(),
            "tombstones": self.store.tombstone_count(),
            "pending_push": len(self.store.list_since_seq(push_seq)),
            "push_seq": push_seq,
            "pull_seq": pull_seq,
            "last_push": last_push_ts,
            "last_pull": last_pull_ts,
            "last_run": self.last_run(peer_id),
            "history_runs": len(self.history(peer_id, limit=1000)),
        }

    def format_status(self, peer_id: str = "hub",
                      style: str = "rich") -> str:
        """Human-readable status card for a peer."""
        t = _theme(style)
        st = self.status(peer_id)
        last = st["last_run"]
        if style == "compact":
            ago = _ago(last["started_at"]) if last else "never"
            mark = t["ok"] if last and last["ok"] else (
                t["fail"] if last else t["warn"])
            return (f"sync {peer_id} {mark} keys={st['local_keys']} "
                    f"pending↑{st['pending_push']} last={ago}")
        lines = [f"sync status {t['arrow']} {peer_id}"]
        lines.append(f"  device {t['sep']} {st['device_id']}")
        keys = f"{st['local_keys']} keys"
        if st["tombstones"]:
            keys += f" (+{st['tombstones']} tombstones)"
        lines.append(f"  local {t['sep']} {keys}")
        lines.append(f"  push {t['sep']} seq {st['push_seq']} "
                     f"({st['pending_push']} pending {t['up']})")
        lines.append(f"  pull {t['sep']} cursor {st['pull_seq']}")
        if last:
            mark = t["ok"] if last["ok"] else t["fail"]
            outcome = (f"{mark} {t['up']}{last['pushed']} "
                       f"{t['down']}{last['pulled']} "
                       f"{t['merge']}{last['conflicts']}")
            if not last["ok"]:
                outcome += f" — {last['error']}"
            lines.append(f"  last sync {t['sep']} {_ago(last['started_at'])} "
                         f"— {outcome} ({last['duration_s']:.2f}s)")
        else:
            lines.append(f"  last sync {t['sep']} never")
        return "\n".join(lines)


class AutoSync:
    """Live/continuous sync with a peer (PouchDB ``live+retry`` style).

    - Re-syncs (debounced) on every local store change via
      :meth:`SyncStore.subscribe` — no blind polling for the hot path.
    - Falls back to interval polling for changes that arrive only on the
      peer side.
    - Transient failures back off exponentially (with jitter) up to
      ``backoff_max_s``; a success resets to the plain interval.

    Usage::

        auto = AutoSync(engine, peer, peer_id="hub", interval_s=300)
        auto.start()
        ...
        auto.stop()
    """

    def __init__(
        self,
        engine: SyncEngine,
        peer: SyncPeer,
        peer_id: str = "hub",
        *,
        interval_s: float = 300.0,
        debounce_s: float = 2.0,
        backoff_base_s: float = 5.0,
        backoff_max_s: float = 900.0,
        direction: str = "both",
        chunk_size: int = PUSH_CHUNK,
        on_result: Callable[[SyncResult], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        if not peer_id:
            raise ValueError("peer_id is required")
        direction = direction.lower()
        if direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}")
        self._engine = engine
        self._peer = peer
        self._peer_id = peer_id
        self.interval_s = max(1.0, float(interval_s))
        self.debounce_s = max(0.0, float(debounce_s))
        self.backoff_base_s = max(0.5, float(backoff_base_s))
        self.backoff_max_s = max(self.backoff_base_s, float(backoff_max_s))
        self.direction = direction
        self.chunk_size = max(1, int(chunk_size))
        self.on_result = on_result
        self.on_error = on_error
        # RLock: stats() calls the locked `running` property while holding it.
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._wakeup = threading.Event()
        self._delay_s = self.interval_s
        self._runs = 0
        self._failures = 0
        self._last_run_at = 0.0
        self._last_error = ""

    # ── lifecycle ──────────────────────────────────────────────────────

    def start(self) -> "AutoSync":
        """Start the background loop (idempotent). Syncs immediately."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                return self
            self._stop.clear()
            self._wakeup.clear()
            self._engine.store.subscribe(self._on_store_change)
            self._thread = threading.Thread(
                target=self._loop, name=f"autosync-{self._peer_id}",
                daemon=True)
            self._thread.start()
        self.trigger()  # first sync right away, not after one interval
        return self

    def stop(self) -> None:
        """Stop the background loop and unsubscribe from the store."""
        with self._lock:
            thread = self._thread
            self._thread = None
        try:
            self._engine.store.unsubscribe(self._on_store_change)
        except Exception:  # noqa: BLE001 - best-effort
            _log.debug("autosync unsubscribe failed", exc_info=True)
        self._stop.set()
        self._wakeup.set()
        if thread and thread is not threading.current_thread():
            thread.join(timeout=10.0)

    @property
    def running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def trigger(self) -> None:
        """Request a sync now (debounced through the loop)."""
        self._wakeup.set()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "running": self.running,
                "peer_id": self._peer_id,
                "direction": self.direction,
                "runs": self._runs,
                "consecutive_failures": self._failures,
                "last_run_at": self._last_run_at,
                "last_run_ago": _ago(self._last_run_at),
                "last_error": self._last_error,
                "interval_s": self.interval_s,
                "current_delay_s": round(self._delay_s, 1),
            }

    # ── internals ──────────────────────────────────────────────────────

    def _on_store_change(self, _rec: SyncRecord) -> None:
        self._wakeup.set()

    def _loop(self) -> None:
        next_due = time.time() + self._delay_s
        while not self._stop.is_set():
            timeout = max(0.0, next_due - time.time())
            woke = self._wakeup.wait(timeout)
            self._wakeup.clear()
            if self._stop.is_set():
                break
            if woke and self.debounce_s > 0:
                # Let a burst of local writes settle into one sync.
                if self._stop.wait(self.debounce_s):
                    break
                self._wakeup.clear()
            self._run_once()
            with self._lock:
                delay = self._delay_s
            next_due = time.time() + delay

    def _run_once(self) -> None:
        try:
            result = self._engine.sync(
                self._peer, peer_id=self._peer_id,
                direction=self.direction, chunk_size=self.chunk_size)
        except Exception as exc:  # noqa: BLE001 - backoff, don't die
            with self._lock:
                self._failures += 1
                self._last_error = str(exc)
                backoff = min(
                    self.backoff_base_s * (2.0 ** (self._failures - 1)),
                    self.backoff_max_s)
                self._delay_s = backoff * (0.8 + 0.4 * random.random())
            _log.warning("autosync with %s failed (%s); retry in %.0fs",
                         self._peer_id, exc, self._delay_s)
            if self.on_error is not None:
                try:
                    self.on_error(exc)
                except Exception:  # noqa: BLE001 - fail-open
                    _log.debug("autosync on_error failed", exc_info=True)
            return
        with self._lock:
            self._runs += 1
            self._failures = 0
            self._last_run_at = time.time()
            self._last_error = ""
            self._delay_s = self.interval_s
        if self.on_result is not None:
            try:
                self.on_result(result)
            except Exception:  # noqa: BLE001 - fail-open
                _log.debug("autosync on_result failed", exc_info=True)
