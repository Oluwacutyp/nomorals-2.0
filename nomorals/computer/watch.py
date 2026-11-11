"""Live watch: tail a desktop session's stream to the owner's chat.

The computer organ's stream is its perceptions: every
:func:`nomorals.computer.loop.capture` lands a frame file under the
session's frames folder plus the window list and pointer that go
with it. Until now the owner only ever saw that stream one frame
per ask (``/computer view`` perceives once). A *watch* is the live
tail: a background job that captures at a steady cadence and
delivers each NEW frame to the owner's chat as it is produced, so
the owner watches her computer work as it happens instead of
polling for the next file.

The job discipline is doc 18's, in the browser runs' idiom
(``nomorals/browser/jobs.py``):

* The live watch objects are in-memory; when the manager is given
  a database every watch also lands in the durable
  ``computer_watch_records`` table (:class:`WatchRecordStore` —
  runtime-created DDL, migrations untouched): a snapshot at start,
  on every status transition, and as a heartbeat at every capture
  tick, so the record's ``updated_at`` is the watch's liveness
  proof. Record writes are evidence, never a gate: a failed write
  is recorded via the organ's failure ledger and the watch
  proceeds unchanged. Writes are direct (no writer thread — the
  browser manager's queue exists because its job threads must not
  block on the database on their way INTO a runner, which is not
  a watch's shape), but snapshot + write are serialized per watch
  under the watch's record lock: a snapshot is always taken at
  write time, so a racing heartbeat write can never land after a
  terminal write carrying a stale pre-cancel state and resurrect
  the watch on its record.
* A runtime restart does not resurrect a watch. The manager
  reconciles on construction (the sessions organ's discipline:
  records are re-read against reality on first touch): a record
  standing live that this process does not hold is landed
  ``done`` with the restart as its reason — it reads ended, never
  live. One manager per runtime (the runtime caches it, the
  browser-job-manager precedent); reconciliation assumes that.
* Cancellation is cooperative and receipted through the ONE
  facade (``core.cancellation``): :meth:`ComputerWatchManager.cancel`
  sets the request; the capture loop polls it every tick (its wait
  is the cancel event itself, so a request lands within one
  cadence tick); the record distinguishes ``requested`` from
  ``confirmed``, and a watch that ends on its own terms despite a
  standing request is recorded ``unhonored`` — except that a
  request standing at the close of a watch whose loop polls (this
  one does, every tick) lands it ``cancelled``/``confirmed``,
  the browser close-out rule. A watch that FAILS keeps its
  failure; the request reads unhonored there.

Delivery discipline (the live-display paging law, applied to a
stream): frames are never dumped in a burst and never backlogged.
The capture thread publishes into a ONE-frame pending slot; a
frame that is overwritten before the delivery thread sends it is
counted, and the next delivered frame's caption carries the count
("skipped N frames to stay current"). One delivery thread owns
every surface send — frames, stall and gate notes, the end note —
so the owner's chat sees one ordered stream. On cancel the
pending frame is dropped: delivery ceases. On a natural end the
final pending frame is delivered before the end note.

Honesty rules:

* A static screen is not a stall: captures that succeed with an
  unchanged frame prove the producer is alive. A stall is the
  producer not producing — no successful capture for
  ``stall_after_s`` — and it is announced ONCE per episode, with
  the last-seen time; recovery is announced with the next frame
  (or on its own when the screen is unchanged).
* The control gate binds hands, never eyes (the sessions organ's
  law): a takeover or the owner's sticky stop does NOT pause the
  watch. The flips are surfaced in words, once per flip, so the
  owner sees the state change on the stream itself.
* A watch is bounded: ``max_seconds`` (default one hour) is
  recorded on its record as its budget; when the budget is spent
  the watch ends and says so. The stop is always available
  sooner.

Layer 4, beside sessions/loop: imports downward only (core
failure idioms arrive via the organ's own failures module; the
database is a duck-typed handle, the browser jobs precedent).
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import failures
from .sessions import SessionError, SessionManager, SessionRecord

__all__ = [
    "STATUS_RUNNING",
    "STATUS_STALLED",
    "STATUS_DONE",
    "STATUS_CANCELLED",
    "STATUS_FAILED",
    "CANCEL_REQUESTED",
    "CANCEL_CONFIRMED",
    "CANCEL_UNHONORED",
    "DEFAULT_INTERVAL_S",
    "DEFAULT_MAX_SECONDS",
    "StreamFrame",
    "WatchRecordStore",
    "ComputerWatch",
    "ComputerWatchManager",
    "session_capture_fn",
    "session_gate_fn",
]

#: Watch statuses. Terminal: done / cancelled / failed.
STATUS_RUNNING = "running"
STATUS_STALLED = "stalled"
STATUS_DONE = "done"
STATUS_CANCELLED = "cancelled"
STATUS_FAILED = "failed"

_ACTIVE = frozenset({STATUS_RUNNING, STATUS_STALLED})
_TERMINAL = frozenset({STATUS_DONE, STATUS_CANCELLED, STATUS_FAILED})

#: Cancel lifecycle on the durable record (browser idiom): ""
#: (never requested), "requested" (asked, not yet honored),
#: "confirmed" (the watch landed cancelled), "unhonored" (the
#: watch ended on its own terms despite the request).
CANCEL_REQUESTED = "requested"
CANCEL_CONFIRMED = "confirmed"
CANCEL_UNHONORED = "unhonored"

#: Capture cadence: the producer's natural beat for a desktop
#: stream — fast enough to feel live, slow enough to never flood.
DEFAULT_INTERVAL_S = 5.0
#: A forgotten watch must not stream forever: one hour, recorded
#: on the watch's record as its budget.
DEFAULT_MAX_SECONDS = 3600.0
#: No successful capture for this long (and never less than six
#: cadence ticks) reads as a stalled producer.
_MIN_STALL_AFTER_S = 30.0

#: Why a record reconciled at manager construction ended.
_RESTART_REASON = (
    "the runtime stopped while this watch was live — watches do "
    "not resume across restarts, so it ended with the process")

_RECORD_DDL = """
CREATE TABLE IF NOT EXISTS computer_watch_records (
  watch_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL DEFAULT '',
  label TEXT NOT NULL DEFAULT '',
  platform TEXT NOT NULL DEFAULT '',
  chat_key TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT '',
  reason TEXT NOT NULL DEFAULT '',
  error TEXT NOT NULL DEFAULT '',
  frames_delivered INTEGER NOT NULL DEFAULT 0,
  frames_skipped INTEGER NOT NULL DEFAULT 0,
  frames_dropped INTEGER NOT NULL DEFAULT 0,
  last_frame_at REAL NOT NULL DEFAULT 0,
  last_seen_at REAL NOT NULL DEFAULT 0,
  budget_s REAL NOT NULL DEFAULT 0,
  cancel_state TEXT NOT NULL DEFAULT '',
  cancel_requested_at REAL NOT NULL DEFAULT 0,
  created_at REAL NOT NULL DEFAULT 0,
  started_at REAL NOT NULL DEFAULT 0,
  finished_at REAL NOT NULL DEFAULT 0,
  updated_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_computer_watch_records_status
  ON computer_watch_records(status, updated_at);
"""


@dataclass
class StreamFrame:
    """One chunk of the desktop stream: a frame + its context.

    The producer's output and the delivery's input — the same
    shape :class:`nomorals.computer.loop.Perception` carries,
    minus the raw bytes (the file on disk is the deliverable).
    """

    session_id: str
    ts: float
    frame_path: str = ""
    frame_hash: str = ""
    width: int = 0
    height: int = 0
    windows: list[dict[str, Any]] = field(default_factory=list)
    focused_id: int = 0
    pointer: dict[str, int] = field(default_factory=dict)

    def focus_line(self) -> str:
        """One line of activity context for a delivery caption."""
        focused = ""
        for win in self.windows:
            if win.get("focused") or win.get("id") == self.focused_id:
                focused = str(win.get("title") or "")
                if focused:
                    break
        count = len(self.windows)
        if focused:
            return f"focused: {focused[:80]} · {count} window(s)"
        return f"{count} window(s) open"


#: ``capture(session_id) -> StreamFrame`` — the stream producer.
#: Raises :class:`SessionError` when the session is gone (the
#: watch's natural end); any other exception is producer trouble
#: (stall bookkeeping, never a silent death).
CaptureFn = Callable[[str], StreamFrame]
#: ``gate(session_id) -> {"takeover": bool, "owner_stop": bool}`` —
#: the session's control-gate flags, read per tick so flips are
#: surfaced on the stream as they happen.
GateFn = Callable[[str], dict[str, Any]]
#: ``send_frame(frame, caption) -> None`` — one frame delivery.
SendFrame = Callable[[StreamFrame, str], None]
#: ``send_note(text) -> None`` — one lifecycle note delivery.
SendNote = Callable[[str], None]


def session_capture_fn(manager: SessionManager) -> CaptureFn:
    """The production producer: perceive the session per tick.

    Perception is ungated (the gate binds hands, never eyes), so
    a takeover or a degraded brain never blinds the watch. A
    session that is no longer open raises SessionError — the
    watch's natural end.
    """
    from . import loop

    def _capture(session_id: str) -> StreamFrame:
        rec = manager.get(session_id)
        if rec is None:
            raise SessionError(f"no session {session_id!r}")
        if rec.state != "open":
            raise SessionError(
                f"session {session_id} is {rec.state}"
                + (f" — {rec.note}" if rec.note else ""))
        seen = loop.capture(manager, rec)
        return StreamFrame(
            session_id=session_id, ts=seen.ts,
            frame_path=seen.frame_path, frame_hash=seen.frame_hash,
            width=seen.width, height=seen.height,
            windows=seen.windows, focused_id=seen.focused_id,
            pointer=seen.pointer)

    return _capture


def session_gate_fn(manager: SessionManager) -> GateFn:
    """The production gate reader: the record's own flags."""

    def _gate(session_id: str) -> dict[str, Any]:
        rec = manager.get(session_id)
        if rec is None:
            return {"takeover": False, "owner_stop": False,
                    "state": "missing"}
        return {"takeover": bool(rec.takeover),
                "owner_stop": bool(rec.owner_stop),
                "state": rec.state}

    return _gate


class WatchRecordStore:
    """Durable per-watch records (doc 18 P1 pattern).

    Runtime-created table over the shared ``Database`` (Batch B
    pattern — DDL in code, migrations untouched). One row per
    watch, upserted at start, on every transition, and as a
    heartbeat at every capture tick.
    """

    def __init__(self, db: Any) -> None:
        self.db = db
        self.db.executescript(_RECORD_DDL)

    def record(self, snapshot: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT INTO computer_watch_records (watch_id,"
            " session_id, label, platform, chat_key, status,"
            " reason, error, frames_delivered, frames_skipped,"
            " frames_dropped, last_frame_at, last_seen_at,"
            " budget_s, cancel_state, cancel_requested_at,"
            " created_at, started_at, finished_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,"
            " ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(watch_id) DO UPDATE SET"
            " session_id = excluded.session_id,"
            " label = excluded.label,"
            " platform = excluded.platform,"
            " chat_key = excluded.chat_key,"
            " status = excluded.status,"
            " reason = excluded.reason,"
            " error = excluded.error,"
            " frames_delivered = excluded.frames_delivered,"
            " frames_skipped = excluded.frames_skipped,"
            " frames_dropped = excluded.frames_dropped,"
            " last_frame_at = excluded.last_frame_at,"
            " last_seen_at = excluded.last_seen_at,"
            " budget_s = excluded.budget_s,"
            " cancel_state = excluded.cancel_state,"
            " cancel_requested_at = excluded.cancel_requested_at,"
            " started_at = excluded.started_at,"
            " finished_at = excluded.finished_at,"
            " updated_at = excluded.updated_at",
            (snapshot["watch_id"], snapshot["session_id"],
             snapshot["label"], snapshot["platform"],
             snapshot["chat_key"], snapshot["status"],
             snapshot["reason"], snapshot["error"],
             snapshot["frames_delivered"],
             snapshot["frames_skipped"],
             snapshot["frames_dropped"],
             snapshot["last_frame_at"], snapshot["last_seen_at"],
             snapshot["budget_s"], snapshot["cancel_state"],
             snapshot["cancel_requested_at"],
             snapshot["created_at"], snapshot["started_at"],
             snapshot["finished_at"], snapshot["updated_at"]))

    def get(self, watch_id: str) -> dict[str, Any] | None:
        return self.db.query_one(
            "SELECT * FROM computer_watch_records"
            " WHERE watch_id = ?", (watch_id,))

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.db.query(
            "SELECT * FROM computer_watch_records"
            " ORDER BY created_at DESC LIMIT ?", (int(limit),))

    def live(self) -> list[dict[str, Any]]:
        return self.db.query(
            "SELECT * FROM computer_watch_records"
            " WHERE status IN (?, ?)",
            (STATUS_RUNNING, STATUS_STALLED))


@dataclass
class ComputerWatch:
    """One live watch: state, the pending slot, and its threads."""

    watch_id: str
    session_id: str
    label: str = ""
    platform: str = ""
    chat_key: str = ""
    status: str = STATUS_RUNNING
    reason: str = ""
    error: str = ""
    frames_delivered: int = 0
    #: Frames overwritten in the pending slot before delivery —
    #: coalesced away to stay current, and reported on the next
    #: delivered frame's caption.
    frames_skipped: int = 0
    #: Frames whose delivery itself failed (counted, ledgered).
    frames_dropped: int = 0
    last_frame_at: float = 0.0
    last_seen_at: float = 0.0
    budget_s: float = 0.0
    cancel_state: str = ""
    cancel_requested_at: float = 0.0
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    finished_at: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock,
                                  repr=False, compare=False)
    #: Serializes snapshot+write for the durable record (see the
    #: module docstring): held across both, so record writes for
    #: one watch land in snapshot order and never regress.
    _record_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False)
    _cancel: threading.Event = field(default_factory=threading.Event,
                                     repr=False, compare=False)
    _cond: threading.Condition = field(
        default_factory=threading.Condition, repr=False,
        compare=False)
    _done: threading.Event = field(default_factory=threading.Event,
                                   repr=False, compare=False)
    _pending: StreamFrame | None = field(
        default=None, repr=False, compare=False)
    _pending_skipped: int = field(default=0, repr=False,
                                  compare=False)
    _pending_note: str = field(default="", repr=False,
                               compare=False)
    _notes: list[str] = field(default_factory=list, repr=False,
                              compare=False)
    _finished: bool = field(default=False, repr=False,
                            compare=False)
    _end_note: str = field(default="", repr=False, compare=False)
    _send_frame: SendFrame | None = field(
        default=None, repr=False, compare=False)
    _send_note: SendNote | None = field(
        default=None, repr=False, compare=False)
    #: Manager-installed snapshot hook (durable record writes).
    _on_change: Callable[[], None] | None = field(
        default=None, repr=False, compare=False)

    # -- cancellation ---------------------------------------------------

    def cancel_requested(self) -> bool:
        """The capture loop polls this: True once cancel is asked."""
        return self._cancel.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """True when the watch has fully landed (delivery done)."""
        return self._done.wait(timeout)

    # -- record ----------------------------------------------------------

    def _changed(self) -> None:
        hook = self._on_change
        if hook is not None:
            hook()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "watch_id": self.watch_id,
                "session_id": self.session_id,
                "label": self.label,
                "platform": self.platform,
                "chat_key": self.chat_key,
                "status": self.status,
                "reason": self.reason,
                "error": self.error,
                "frames_delivered": self.frames_delivered,
                "frames_skipped": self.frames_skipped,
                "frames_dropped": self.frames_dropped,
                "last_frame_at": self.last_frame_at,
                "last_seen_at": self.last_seen_at,
                "budget_s": self.budget_s,
                "cancel_state": self.cancel_state,
                "cancel_requested_at": self.cancel_requested_at,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "updated_at": time.time(),
            }

    def status_text(self) -> str:
        with self._lock:
            lines = [
                f"computer watch {self.watch_id} [{self.status}]",
                f"session: {self.session_id}",
                f"frames: {self.frames_delivered} delivered · "
                f"{self.frames_skipped} skipped to stay current · "
                f"{self.frames_dropped} dropped",
            ]
            if self.reason:
                lines.append(f"reason: {self.reason}")
            if self.error:
                lines.append(f"error: {self.error}")
            if self.cancel_state:
                lines.append(f"cancel: {self.cancel_state}")
        return "\n".join(lines)


class ComputerWatchManager:
    """Registry + threads for computer watches. One per runtime."""

    def __init__(self, *, capture_fn: CaptureFn | None = None,
                 gate_fn: GateFn | None = None, db: Any = None,
                 interval_s: float = DEFAULT_INTERVAL_S,
                 stall_after_s: float | None = None,
                 max_seconds: float = DEFAULT_MAX_SECONDS) -> None:
        self._capture_fn = capture_fn
        self._gate_fn = gate_fn
        self._interval_s = max(0.05, float(interval_s))
        self._stall_after_s = (
            max(_MIN_STALL_AFTER_S, 6.0 * self._interval_s)
            if stall_after_s is None
            else max(0.05, float(stall_after_s)))
        self._max_seconds = max(0.0, float(max_seconds))
        self._lock = threading.Lock()
        self._watches: dict[str, ComputerWatch] = {}
        self._keys: dict[tuple[str, str], str] = {}
        self._records: WatchRecordStore | None = None
        if db is not None:
            try:
                self._records = WatchRecordStore(db)
            except Exception as exc:  # noqa: BLE001 - never a gate
                failures.record_failure("watch record store", exc)
        self.reconcile_stale()

    # -- durable records --------------------------------------------------

    def reconcile_stale(self) -> int:
        """Land records this process does not hold as ended.

        A record standing live whose watch is not among this
        manager's live watches belongs to a process that is gone
        (this manager is the runtime's one watch home): the watch
        died with it and does not resume. The record lands
        ``done`` with the restart as its reason, its finish time
        set to its last heartbeat — the last moment it was proven
        alive. Returns how many records were reconciled.
        """
        if self._records is None:
            return 0
        with self._lock:
            held = set(self._watches)
        landed = 0
        try:
            rows = self._records.live()
        except Exception as exc:  # noqa: BLE001 - fail-open
            failures.record_failure("watch reconcile read", exc)
            return 0
        for row in rows:
            if row.get("watch_id") in held:
                continue
            snapshot = dict(row)
            snapshot["status"] = STATUS_DONE
            snapshot["reason"] = _RESTART_REASON
            snapshot["finished_at"] = (
                float(row.get("updated_at") or 0)
                or float(row.get("created_at") or 0))
            try:
                self._records.record(snapshot)
                landed += 1
            except Exception as exc:  # noqa: BLE001 - fail-open
                failures.record_failure("watch reconcile", exc)
        return landed

    def _record_watch(self, watch: ComputerWatch) -> None:
        """Snapshot one watch for its durable record. Fail-open:
        the record is evidence, not a gate. Snapshot and write
        are atomic per watch (the record lock), so writes land
        in snapshot order and a stale snapshot can never
        overwrite a fresher state."""
        if self._records is None:
            return
        with watch._record_lock:
            try:
                self._records.record(watch.snapshot())
            except Exception as exc:  # noqa: BLE001 - fail-open
                failures.record_failure("watch record", exc)

    def flush_records(self, timeout: float = 5.0) -> bool:
        """Records are written synchronously at each transition —
        there is nothing queued to flush. Kept for the browser
        manager's shape so callers can treat the two alike."""
        return True

    # -- lifecycle ---------------------------------------------------------

    def start(self, session: SessionRecord | str, *,
              label: str = "", platform: str = "", chat_key: str = "",
              send_frame: SendFrame | None = None,
              send_note: SendNote | None = None,
              capture_fn: CaptureFn | None = None,
              gate_fn: GateFn | None = None,
              force: bool = False) -> tuple[ComputerWatch, bool]:
        """Start a watch on a session. Returns ``(watch, dup)``.

        ``dup`` is True when a live watch for the same session +
        chat was returned instead of starting a second tail
        (the browser runs' dedupe idiom; ``force`` starts fresh).
        The producer/senders default to the manager's; per-start
        overrides exist for wiring and tests.
        """
        session_id = session.id if isinstance(session, SessionRecord) \
            else str(session or "")
        if not session_id:
            raise SessionError("a watch needs a session")
        capture = capture_fn or self._capture_fn
        if capture is None:
            raise SessionError(
                "no stream producer is wired into this watch "
                "manager — the runtime supplies the session "
                "capture; a bare manager cannot watch")
        if not label and isinstance(session, SessionRecord):
            label = str(session.mission or "")
        key = (session_id, chat_key)
        with self._lock:
            if not force:
                existing_id = self._keys.get(key)
                existing = self._watches.get(existing_id or "")
                if existing is not None \
                        and existing.status in _ACTIVE:
                    return existing, True
            watch_id = f"cw-{uuid.uuid4().hex[:12]}"
            watch = ComputerWatch(
                watch_id=watch_id, session_id=session_id,
                label=label, platform=platform, chat_key=chat_key,
                budget_s=self._max_seconds,
                _send_frame=send_frame, _send_note=send_note)
            watch._on_change = \
                lambda w=watch: self._record_watch(w)
            self._watches[watch_id] = watch
            self._keys[key] = watch_id
        self._record_watch(watch)
        gate = gate_fn or self._gate_fn
        sender = threading.Thread(
            target=self._deliver_loop, args=(watch,),
            name=f"computer-watch-send-{watch_id}", daemon=True)
        producer = threading.Thread(
            target=self._capture_loop, args=(watch, capture, gate),
            name=f"computer-watch-{watch_id}", daemon=True)
        sender.start()
        producer.start()
        return watch, False

    def get(self, watch_id_or_prefix: str) -> ComputerWatch | None:
        """Exact id, or a unique prefix."""
        needle = (watch_id_or_prefix or "").strip()
        if not needle:
            return None
        with self._lock:
            if needle in self._watches:
                return self._watches[needle]
            matches = [w for wid, w in self._watches.items()
                       if wid.startswith(needle)]
        return matches[0] if len(matches) == 1 else None

    def active_for(self, chat_key: str,
                   session_id: str = "") -> ComputerWatch | None:
        """The live watch delivering to one chat (newest first)."""
        with self._lock:
            watches = sorted(
                self._watches.values(),
                key=lambda w: w.created_at, reverse=True)
        for watch in watches:
            if watch.chat_key != chat_key:
                continue
            if session_id and watch.session_id != session_id:
                continue
            if watch.status in _ACTIVE:
                return watch
        return None

    def list_recent(self, limit: int = 5) -> list[ComputerWatch]:
        with self._lock:
            watches = sorted(self._watches.values(),
                             key=lambda w: w.created_at,
                             reverse=True)
        return watches[:max(1, limit)]

    def status_text(self, watch_id_or_prefix: str) -> str:
        watch = self.get(watch_id_or_prefix)
        if watch is None:
            return (f"no computer watch {watch_id_or_prefix!r} — "
                    "/computer watch starts one on a session.")
        return watch.status_text()

    # -- cancellation -------------------------------------------------------

    def cancel(self, watch_id_or_prefix: str) -> dict[str, Any]:
        """Request cancellation of a watch. Typed result, never a
        bool (browser idiom): ``state`` is one of ``not_found`` /
        ``already_terminal`` / ``requested``. A request is not a
        confirmation — the durable record carries the final word
        (``confirmed`` when the watch landed cancelled,
        ``unhonored`` when it ended on its own terms). A live
        watch polls the request every cadence tick, so it takes
        effect at the next tick at the latest."""
        watch = self.get(watch_id_or_prefix)
        if watch is None:
            return {"ok": False, "state": "not_found",
                    "watch_id": str(watch_id_or_prefix or "")}
        with watch._lock:
            status = watch.status
            if status in _TERMINAL:
                return {"ok": True, "state": "already_terminal",
                        "watch_id": watch.watch_id, "status": status,
                        "cancel_state": watch.cancel_state}
            watch._cancel.set()
            if not watch.cancel_state:
                watch.cancel_state = CANCEL_REQUESTED
                watch.cancel_requested_at = time.time()
        watch._changed()
        return {"ok": True, "state": "requested",
                "watch_id": watch.watch_id, "status": status,
                "cancel_state": watch.cancel_state,
                "takes_effect": "at the watch's next capture tick"}

    # -- the capture loop (producer side) -----------------------------------

    def _capture_loop(self, watch: ComputerWatch,
                      capture_fn: CaptureFn,
                      gate_fn: GateFn | None) -> None:
        started = time.time()
        with watch._lock:
            watch.started_at = started
            watch.status = STATUS_RUNNING
        watch._changed()
        last_seen = 0.0
        last_hash = ""
        stalled = False
        gate_flags: tuple[bool, bool] | None = None
        try:
            while True:
                if watch.cancel_requested():
                    self._land(watch, STATUS_CANCELLED,
                               "cancelled by the owner")
                    return
                now = time.time()
                if watch.budget_s and now - started > watch.budget_s:
                    self._land(
                        watch, STATUS_DONE,
                        f"the watch budget of "
                        f"{watch.budget_s:.0f}s was spent — "
                        f"/computer watch starts a new one")
                    return
                frame: StreamFrame | None = None
                try:
                    frame = capture_fn(watch.session_id)
                except SessionError as exc:
                    self._land(watch, STATUS_DONE,
                               f"the desktop session ended: {exc}")
                    return
                except Exception as exc:  # noqa: BLE001 - stall food
                    failures.record_failure("watch capture", exc)
                if frame is not None:
                    last_seen = frame.ts or time.time()
                    with watch._lock:
                        watch.last_seen_at = last_seen
                    if stalled:
                        stalled = False
                        with watch._lock:
                            watch.status = STATUS_RUNNING
                        self._publish(watch, None,
                                      note="the stream stalled and "
                                           "is flowing again")
                    gate_flags = self._gate_notes(
                        watch, gate_fn, gate_flags)
                    if frame.frame_hash and \
                            frame.frame_hash != last_hash:
                        last_hash = frame.frame_hash
                        self._publish(watch, frame)
                    watch._changed()
                elif not stalled:
                    silent_since = last_seen or started
                    if time.time() - silent_since \
                            > self._stall_after_s:
                        stalled = True
                        with watch._lock:
                            watch.status = STATUS_STALLED
                        watch._changed()
                        if last_seen:
                            stamp = time.strftime(
                                "%H:%M:%S",
                                time.localtime(last_seen))
                            detail = (f"no new frame since {stamp}")
                        else:
                            detail = ("no frame yet from this "
                                      "desktop")
                        self._queue_note(
                            watch,
                            f"🖥 the desktop stream has stalled — "
                            f"{detail}. The watch is still on; it "
                            f"picks up the moment the desktop "
                            f"answers again.")
                if watch._cancel.wait(self._interval_s):
                    self._land(watch, STATUS_CANCELLED,
                               "cancelled by the owner")
                    return
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
            failures.record_failure("watch loop", exc)
            self._land(watch, STATUS_FAILED,
                       f"the watch loop failed: {exc}",
                       error=f"{type(exc).__name__}: {exc}")

    def _gate_notes(self, watch: ComputerWatch,
                    gate_fn: GateFn | None,
                    previous: tuple[bool, bool] | None,
                    ) -> tuple[bool, bool] | None:
        """Surface control-gate flips on the stream, once per flip.

        The organ's law: the gate binds hands, never eyes — a
        takeover or the owner's stop never pauses the watch; the
        flip itself is news the owner should see where they are
        already looking.
        """
        if gate_fn is None:
            return previous
        try:
            flags = gate_fn(watch.session_id)
        except Exception as exc:  # noqa: BLE001 - color, not a gate
            failures.record_failure("watch gate read", exc)
            return previous
        current = (bool(flags.get("takeover")),
                   bool(flags.get("owner_stop")))
        if previous is None or current == previous:
            return current
        was_takeover, was_stop = previous
        takeover, stopped = current
        if takeover and not was_takeover:
            self._queue_note(
                watch,
                "🖥 the owner has taken over this desktop — the "
                "agent's hands are off; the watch keeps streaming "
                "(watching is never gated).")
        if stopped and not was_stop:
            self._queue_note(
                watch,
                "🖥 the owner's stop is on for this computer — "
                "the agent cannot act; the watch keeps streaming "
                "(watching is never gated).")
        if was_takeover and not takeover:
            self._queue_note(
                watch,
                "🖥 takeover handed back — the agent may drive "
                "this desktop again.")
        if was_stop and not stopped:
            self._queue_note(
                watch,
                "🖥 the owner's stop is cleared — this computer "
                "is resumed.")
        return current

    # -- delivery (owner side) ----------------------------------------------

    def _publish(self, watch: ComputerWatch,
                 frame: StreamFrame | None, note: str = "") -> None:
        """Publish into the one-frame pending slot (latest wins).

        A frame overwritten before delivery is counted as
        skipped — coalesced away to stay current, and the count
        rides the next delivered frame's caption.
        """
        with watch._cond:
            if frame is not None:
                if watch._pending is not None:
                    watch._pending_skipped += 1
                    with watch._lock:
                        watch.frames_skipped += 1
                watch._pending = frame
            if note:
                watch._pending_note = note
            watch._cond.notify_all()

    def _queue_note(self, watch: ComputerWatch, text: str) -> None:
        with watch._cond:
            watch._notes.append(text)
            watch._cond.notify_all()

    def _deliver_loop(self, watch: ComputerWatch) -> None:
        """The one sender thread: frames, notes, the end note —
        one ordered stream to the owner's chat."""
        try:
            while True:
                with watch._cond:
                    while watch._pending is None \
                            and not watch._notes \
                            and not watch._finished:
                        watch._cond.wait(timeout=1.0)
                    frame = watch._pending
                    skipped = watch._pending_skipped
                    note = watch._pending_note
                    watch._pending = None
                    watch._pending_skipped = 0
                    watch._pending_note = ""
                    notes = watch._notes[:]
                    watch._notes.clear()
                    finished = watch._finished
                    end_note = watch._end_note
                if note and frame is None:
                    # Recovery with an unchanged screen: nothing
                    # to deliver, but the recovery is still news.
                    notes.insert(0, f"🖥 desktop — {note}")
                for text in notes:
                    self._send_note(watch, text)
                if frame is not None:
                    self._deliver_frame(watch, frame, skipped, note)
                if finished:
                    if end_note:
                        self._send_note(watch, end_note)
                    return
        finally:
            watch._done.set()

    def _deliver_frame(self, watch: ComputerWatch,
                       frame: StreamFrame, skipped: int,
                       note: str) -> None:
        parts = [f"🖥 desktop — {frame.focus_line()}"]
        if note:
            parts.append(note)
        if skipped:
            parts.append(
                f"skipped {skipped} frame(s) to stay current")
        caption = " · ".join(parts)
        send = watch._send_frame
        if send is None:
            with watch._lock:
                watch.frames_dropped += 1
            watch._changed()
            return
        try:
            send(frame, caption)
        except Exception as exc:  # noqa: BLE001 - counted, ledgered
            failures.record_failure("watch deliver", exc)
            with watch._lock:
                watch.frames_dropped += 1
            watch._changed()
            return
        with watch._lock:
            watch.frames_delivered += 1
            watch.last_frame_at = frame.ts
        watch._changed()

    def _send_note(self, watch: ComputerWatch, text: str) -> None:
        send = watch._send_note
        if send is None:
            return
        try:
            send(text)
        except Exception as exc:  # noqa: BLE001 - notes never kill
            failures.record_failure("watch note", exc)

    # -- landing --------------------------------------------------------------

    def _land(self, watch: ComputerWatch, status: str, reason: str,
              error: str = "") -> None:
        """Land a watch at its terminal state, exactly once.

        The browser close-out rule: a cancel request standing at
        the close of a loop that polls it (this one polls every
        tick) lands the watch cancelled/confirmed — never a clean
        completion the owner had already revoked. A FAILED watch
        keeps its failure; the request there reads unhonored.
        """
        with watch._lock:
            if watch.status in _TERMINAL:
                return
            if status != STATUS_FAILED \
                    and watch.cancel_state == CANCEL_REQUESTED:
                prior = reason
                status = STATUS_CANCELLED
                reason = ("cancel request landed before the watch "
                          f"closed (it was ending: {prior})")
            watch.status = status
            watch.reason = reason
            if error:
                watch.error = error
            watch.finished_at = time.time()
            if watch.cancel_state == CANCEL_REQUESTED:
                watch.cancel_state = (
                    CANCEL_CONFIRMED if status == STATUS_CANCELLED
                    else CANCEL_UNHONORED)
            delivered = watch.frames_delivered
        # The terminal record lands BEFORE the delivery thread is
        # released: a waiter that wakes on the watch's end must
        # find the record already standing at its final state,
        # never racing the write.
        watch._changed()
        with watch._cond:
            if status == STATUS_CANCELLED:
                # Delivery ceases: the pending frame is dropped.
                watch._pending = None
                watch._pending_skipped = 0
                watch._pending_note = ""
            watch._finished = True
            if status == STATUS_CANCELLED:
                watch._end_note = (
                    f"✋ computer watch stopped — {delivered} "
                    f"frame(s) delivered.")
            elif status == STATUS_FAILED:
                watch._end_note = (
                    f"⛔ computer watch failed: {error or reason} "
                    f"— {delivered} frame(s) delivered.")
            else:
                watch._end_note = (
                    f"🖥 computer watch ended: {reason} — "
                    f"{delivered} frame(s) delivered.")
            watch._cond.notify_all()
