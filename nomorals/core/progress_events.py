"""One measured progress stream for long work (Play & Downloads P1).

Every long operation — a download, an upload, a search-then-fetch —
reports into ONE event stream.  An event carries the operation id,
the stage, the attempt, a monotonic timestamp, the bytes received so
far, the expected bytes when (and only when) they are known, segment
counts kept strictly separate from bytes, and — on the final event —
the terminal result.

The honesty contract (owner rules, doc 8 Priority 1):

* Unknown content length → bytes + elapsed + rolling speed.  A
  percentage is NEVER derived from an estimate or a guess.
* ETA comes from a rolling window of recent samples, never from the
  instantaneous rate of the last chunk.
* Segmented media counts segments as segments; segment counts are
  never mixed into byte counts (or turned into a byte percentage).
* Retrieval complete is NOT delivered.  ``delivered`` is a terminal
  state set only by the layer that actually handed the result over.
* An upload whose transport reports no progress renders as an
  indeterminate stage — no fabricated bytes, no invented percentage.
* Stalled work reads as waiting, with its last-observed time — never
  as failed, and a stale rate is never presented as a current rate.
* Terminal states render promptly; byte updates coalesce behind a
  cadence so chat adapters are never flooded with edits.

This module is core (layer 1): media, tools, social and agents all
import it; it imports nothing above stdlib.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

__all__ = [
    "SEARCHING", "RESOLVING", "CONNECTING", "DOWNLOADING", "VERIFYING",
    "PROCESSING", "UPLOADING", "DELIVERED", "STAGES",
    "TERMINAL_DELIVERED", "TERMINAL_FAILED", "TERMINAL_CANCELLED",
    "ProgressEvent", "ProgressTracker", "render_progress_text",
    "current_tracker", "bind_tracker", "set_current_tracker",
    "reset_current_tracker", "emit_stage", "emit_bytes", "emit_attempt",
    "emit_source", "emit_segments", "emit_finish",
]

_log = logging.getLogger(__name__)

#: The stage vocabulary, in journey order.
SEARCHING = "searching"
RESOLVING = "resolving"
CONNECTING = "connecting"
DOWNLOADING = "downloading"
VERIFYING = "verifying"
PROCESSING = "processing"
UPLOADING = "uploading"
DELIVERED = "delivered"

STAGES = (SEARCHING, RESOLVING, CONNECTING, DOWNLOADING, VERIFYING,
          PROCESSING, UPLOADING, DELIVERED)

TERMINAL_DELIVERED = "delivered"
TERMINAL_FAILED = "failed"
TERMINAL_CANCELLED = "cancelled"

_STAGE_ICONS = {
    SEARCHING: "🔎",
    RESOLVING: "🧭",
    CONNECTING: "🔌",
    DOWNLOADING: "⬇️",
    VERIFYING: "🔍",
    PROCESSING: "⚙️",
    UPLOADING: "⬆️",
    DELIVERED: "✅",
}

_STAGE_PHRASES = {
    SEARCHING: "searching the sources",
    RESOLVING: "resolving the source",
    CONNECTING: "connecting",
    PROCESSING: "processing",
}


@dataclass(frozen=True)
class ProgressEvent:
    """One measured observation from an operation's life."""

    op_id: str
    stage: str
    attempt: int
    source_index: int
    mono: float
    bytes_received: int = 0
    bytes_expected: int | None = None
    segments_done: int | None = None
    segments_total: int | None = None
    note: str = ""
    terminal: str | None = None
    error: str = ""


def _fmt_bytes(n: float) -> str:
    n = max(0.0, float(n))
    if n < 1024:
        return f"{int(n)} B"
    for unit, div in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= div:
            return f"{n / div:.1f} {unit}"
    return f"{int(n)} B"  # pragma: no cover — the KB arm always matches


def _fmt_dur(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs}s"


def _bar(pct: float, width: int = 10) -> str:
    filled = int(min(100.0, max(0.0, pct)) / 100.0 * width + 0.5)
    return "▓" * filled + "░" * (width - filled)


class ProgressTracker:
    """Thread-safe measured state for one operation.

    Mutators record a :class:`ProgressEvent` (kept on ``events``) and
    invoke ``on_event`` under the render cadence: stage, attempt,
    source and terminal changes render immediately; byte updates
    render at most once per ``min_render_interval_s`` — stale updates
    coalesce instead of flooding the surface.
    """

    def __init__(self, op_id: str, *, title: str = "",
                 on_event: Callable[[ProgressEvent], None] | None = None,
                 clock: Callable[[], float] | None = None,
                 min_render_interval_s: float = 2.0,
                 stall_after_s: float = 20.0,
                 speed_window_s: float = 10.0) -> None:
        self.op_id = str(op_id)
        self.title = str(title or "")
        self.on_event = on_event
        self._clock = clock or time.monotonic
        self.min_render_interval_s = max(0.0, float(min_render_interval_s))
        self.stall_after_s = float(stall_after_s)
        self.speed_window_s = max(1.0, float(speed_window_s))
        self._lock = threading.RLock()
        now = self._clock()
        self._started = now
        self._stage = ""
        self._stage_started = now
        self._last_event_mono = now
        self.attempt = 0
        self.source_index = 0
        self.source_label = ""
        self._note = ""
        self._bytes = 0
        self._expected: int | None = None
        self._seg_done: int | None = None
        self._seg_total: int | None = None
        self._samples: list[tuple[float, int]] = []
        self._terminal: str | None = None
        self._error = ""
        self.events: list[ProgressEvent] = []
        self._last_render_mono = float("-inf")
        self._rendered_key: tuple | None = None

    # ── mutators ─────────────────────────────────────────────────────

    def begin_source(self, label: str = "") -> ProgressEvent:
        """A new source takes over: attempt restarts at 1 and every
        measurement resets — the previous source's percentage never
        pretends to describe this transfer."""
        with self._lock:
            self.source_index += 1
            self.attempt = 1
            self.source_label = str(label or "")
            self._reset_transfer()
            # A fresh source has not reached any stage yet — the old
            # source's stage (and its stage note) must not head the
            # new source's first render.
            self._stage = ""
            self._note = ""
            self._stage_started = self._clock()
            return self._record(force=True)

    def begin_attempt(self, note: str = "") -> ProgressEvent:
        """Another try of the current source (retry / route switch):
        the attempt counter increments and measurements reset."""
        with self._lock:
            self.attempt += 1
            if note:
                self._note = str(note)
            self._reset_transfer()
            return self._record(force=True)

    def set_stage(self, stage: str, note: str = "") -> ProgressEvent:
        with self._lock:
            if stage != self._stage:
                self._stage = stage
                self._stage_started = self._clock()
            if note:
                self._note = str(note)
            return self._record()

    def report_bytes(self, received: int,
                     expected: int | None = None) -> ProgressEvent:
        """Report the transfer's absolute byte position.  ``expected``
        is the known total; pass ``None`` when it is not known (an
        estimate is not knowledge and is never accepted here)."""
        with self._lock:
            now = self._clock()
            self._bytes = max(0, int(received))
            if expected is not None:
                self._expected = int(expected) if expected > 0 else None
            self._samples.append((now, self._bytes))
            cutoff = now - self.speed_window_s
            while len(self._samples) > 2 \
                    and self._samples[0][0] < cutoff:
                self._samples.pop(0)
            if self._stage == CONNECTING:
                self._stage = DOWNLOADING
                self._stage_started = now
            return self._record()

    def report_segments(self, done: int, total: int) -> ProgressEvent:
        """Segment progress for segmented media — a separate axis
        from bytes, always rendered as a count, never a byte share."""
        with self._lock:
            self._seg_done = max(0, int(done))
            self._seg_total = max(0, int(total))
            return self._record()

    def note(self, text: str) -> ProgressEvent:
        with self._lock:
            self._note = str(text)
            return self._record(force=True)

    def finish(self, terminal: str, *, note: str = "",
               error: str = "") -> ProgressEvent:
        """Record the terminal result.  Only the layer that actually
        delivered (or definitively failed) may call this."""
        with self._lock:
            self._terminal = str(terminal)
            if error:
                self._error = str(error)
            if note:
                self._note = str(note)
            if terminal == TERMINAL_DELIVERED:
                self._stage = DELIVERED
                self._stage_started = self._clock()
            return self._record(force=True)

    # ── reads ────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = self._clock()
            ago = max(0.0, now - self._last_event_mono)
            stalled = self._terminal is None and ago > self.stall_after_s
            speed = None if stalled else self._speed()
            pct = None
            eta = None
            if self._expected:
                pct = min(100, int(self._bytes * 100 / self._expected))
                if speed and self._bytes < self._expected:
                    eta = (self._expected - self._bytes) / speed
            return {
                "op_id": self.op_id,
                "title": self.title,
                "stage": self._stage,
                "attempt": self.attempt,
                "source_index": self.source_index,
                "source_label": self.source_label,
                "bytes_received": self._bytes,
                "bytes_expected": self._expected,
                "pct": pct,
                "speed_bps": speed,
                "eta_s": eta,
                "elapsed_s": max(0.0, now - self._started),
                "stage_elapsed_s": max(0.0, now - self._stage_started),
                "last_update_ago_s": ago,
                "stalled": stalled,
                "segments_done": self._seg_done,
                "segments_total": self._seg_total,
                "terminal": self._terminal,
                "error": self._error,
                "note": self._note,
            }

    def render_text(self) -> str:
        return render_progress_text(self.snapshot())

    # ── internals ────────────────────────────────────────────────────

    def _reset_transfer(self) -> None:
        self._bytes = 0
        self._expected = None
        self._seg_done = None
        self._seg_total = None
        self._samples = []

    def _speed(self) -> float | None:
        """Rolling-window transfer rate (bytes/sec) from the samples
        inside the window ending at the most recent sample."""
        if len(self._samples) < 2:
            return None
        t_last, b_last = self._samples[-1]
        window = [s for s in self._samples
                  if s[0] >= t_last - self.speed_window_s]
        if len(window) < 2:
            window = self._samples[-2:]
        t0, b0 = window[0]
        dt = t_last - t0
        if dt <= 0 or b_last <= b0:
            return None
        return (b_last - b0) / dt

    def _record(self, *, force: bool = False) -> ProgressEvent:
        now = self._clock()
        ev = ProgressEvent(
            op_id=self.op_id, stage=self._stage, attempt=self.attempt,
            source_index=self.source_index, mono=now,
            bytes_received=self._bytes, bytes_expected=self._expected,
            segments_done=self._seg_done, segments_total=self._seg_total,
            note=self._note, terminal=self._terminal, error=self._error)
        self.events.append(ev)
        if len(self.events) > 500:
            del self.events[:len(self.events) - 500]
        self._last_event_mono = now
        key = (ev.stage, ev.attempt, ev.source_index, ev.terminal)
        due = force or key != self._rendered_key or (
            now - self._last_render_mono) >= self.min_render_interval_s
        if due and self.on_event is not None:
            self._rendered_key = key
            self._last_render_mono = now
            try:
                self.on_event(ev)
            except Exception as exc:  # a sink never kills the operation
                _log.debug("progress sink failed for %s: %s",
                           self.op_id, exc)
        return ev


def render_progress_text(snapshot: dict[str, Any], *, title: str = "") -> str:
    """Render one snapshot as chat text.  Pure: the same snapshot
    renders identically on every surface (Telegram, WhatsApp, CLI)."""
    title = title or str(snapshot.get("title") or "")
    terminal = snapshot.get("terminal")
    elapsed = _fmt_dur(snapshot.get("elapsed_s", 0.0))
    if terminal == TERMINAL_DELIVERED:
        head = f"✅ {title} — delivered" if title else "✅ delivered"
        return f"{head} in {elapsed}"
    if terminal in (TERMINAL_FAILED, TERMINAL_CANCELLED):
        head = f"⚠️ {title} — {terminal}" if title else f"⚠️ {terminal}"
        error = str(snapshot.get("error") or "")
        return f"{head}: {error}" if error else head

    stage = str(snapshot.get("stage") or "")
    icon = _STAGE_ICONS.get(stage, "⏳")
    word = stage.capitalize() if stage else "Working"
    lines = [f"{icon} {word} — {title}" if title else f"{icon} {word}"]

    body = _stage_body(snapshot)
    if body:
        lines.append(body)
    seg_total = snapshot.get("segments_total")
    if seg_total:
        lines.append(
            f"segments {snapshot.get('segments_done') or 0}/{seg_total}")
    source_bits = []
    if snapshot.get("source_index"):
        source_bits.append(f"source {snapshot['source_index']}")
    if snapshot.get("source_index") or (snapshot.get("attempt") or 0) > 1:
        source_bits.append(f"attempt {snapshot.get('attempt') or 1}")
    if source_bits:
        line = " · ".join(source_bits)
        label = str(snapshot.get("source_label") or "")
        if label:
            line += f" — {label}"
        lines.append(line)
    if snapshot.get("stalled"):
        ago = int(snapshot.get("last_update_ago_s") or 0)
        lines.append(f"waiting — last update {ago}s ago")
    return "\n".join(lines)


def _stage_body(snapshot: dict[str, Any]) -> str:
    stage = str(snapshot.get("stage") or "")
    received = int(snapshot.get("bytes_received") or 0)
    expected = snapshot.get("bytes_expected")
    pct = snapshot.get("pct")
    speed = snapshot.get("speed_bps")
    note = str(snapshot.get("note") or "")

    if stage == UPLOADING and not (expected and received > 0):
        # No progress callback on this route: indeterminate, honest.
        return "sending the file — no live count on this route"
    if stage in (DOWNLOADING, CONNECTING, UPLOADING, PROCESSING) \
            and (received > 0 or expected):
        if pct is not None and expected:
            parts = [f"{_bar(pct)} {pct}%",
                     f"{_fmt_bytes(received)} / {_fmt_bytes(expected)}"]
            if speed:
                parts.append(f"{_fmt_bytes(speed)}/s")
            eta = snapshot.get("eta_s")
            if eta is not None:
                parts.append(f"about {_fmt_dur(eta)} left")
            return " · ".join(parts)
        parts = [f"{_fmt_bytes(received)} so far"]
        if speed:
            parts.append(f"{_fmt_bytes(speed)}/s")
        parts.append(f"{_fmt_dur(snapshot.get('elapsed_s', 0.0))} in")
        return " · ".join(parts)
    if stage == VERIFYING:
        if received > 0:
            return f"{_fmt_bytes(received)} received — checking the file"
        return "checking the landed file"
    if note:
        return note
    return _STAGE_PHRASES.get(stage, "")


# ── context binding (mirrors core.jobs' report_progress contract) ────

_current_tracker: ContextVar[ProgressTracker | None] = ContextVar(
    "nomorals_progress_tracker", default=None)


def current_tracker() -> ProgressTracker | None:
    return _current_tracker.get()


def set_current_tracker(tracker: ProgressTracker | None):
    return _current_tracker.set(tracker)


def reset_current_tracker(token) -> None:
    _current_tracker.reset(token)


@contextmanager
def bind_tracker(tracker: ProgressTracker) -> Iterator[ProgressTracker]:
    token = _current_tracker.set(tracker)
    try:
        yield tracker
    finally:
        _current_tracker.reset(token)


def emit_stage(stage: str, note: str = "") -> bool:
    tracker = current_tracker()
    if tracker is None:
        return False
    tracker.set_stage(stage, note=note)
    return True


def emit_bytes(received: int, expected: int | None = None) -> bool:
    tracker = current_tracker()
    if tracker is None:
        return False
    tracker.report_bytes(received, expected)
    return True


def emit_attempt(note: str = "") -> bool:
    tracker = current_tracker()
    if tracker is None:
        return False
    tracker.begin_attempt(note)
    return True


def emit_source(label: str = "") -> bool:
    tracker = current_tracker()
    if tracker is None:
        return False
    tracker.begin_source(label)
    return True


def emit_segments(done: int, total: int) -> bool:
    tracker = current_tracker()
    if tracker is None:
        return False
    tracker.report_segments(done, total)
    return True


def emit_finish(terminal: str, *, note: str = "", error: str = "") -> bool:
    tracker = current_tracker()
    if tracker is None:
        return False
    tracker.finish(terminal, note=note, error=error)
    return True
