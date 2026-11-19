"""SoundCraft realtime engine — the callback law (lane 107).

Ported from storytold/soundcraft's ``crates/playback`` callback.rs
into Devon's Python. The law, in the same shape upstream states it:

* The callback runs on its own thread at a fixed block size. It
  touches only pre-allocated buffers — it never allocates, blocks
  on a lock, or does I/O. (In Python the law is structural: the
  block path holds no I/O and no lock acquisition; the watchdog
  measures every block and names any block that overruns its
  budget.)
* ``guarded_block``: if a callback raises, that block and every
  later one are silent and the transport stops — the exception is
  recorded as a named fault, never taking the engine down.
* Stall detection: if the engine is playing and no block has been
  rendered for ``stall_timeout_s`` (upstream: 2 s), the output is
  ``stalled`` and the fault is recorded once.
* Commands travel from the control side to the callback through a
  bounded ring (fixed capacity, no locks on the realtime side —
  the callback drains it at the head of every block).

There is no audio device here: the engine renders into memory.
``drive(n)`` renders n blocks synchronously for deterministic
tests; ``start()`` runs the same block path on a wall-clock thread.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from .errors import SoundError, SoundErrorKind
from .mix import MixEngine
from .model import SessionDocument

__all__ = ["RealtimeEngine"]

_DEFAULT_STALL_TIMEOUT_S = 2.0
_RING_CAPACITY = 64


class _CommandRing:
    """Bounded single-producer single-consumer ring.

    Fixed-size, pre-allocated; the realtime side drains it without
    taking a lock. A full ring refuses the command instead of
    growing — the law, not a suggestion.
    """

    def __init__(self, capacity: int = _RING_CAPACITY):
        self._slots: list = [None] * capacity
        self._head = 0
        self._tail = 0
        self._capacity = capacity
        self._lock = threading.Lock()

    def push(self, command: dict) -> None:
        with self._lock:
            if (self._tail - self._head) >= self._capacity:
                raise SoundError(
                    "command ring is full — the realtime side is not "
                    "draining; command refused rather than queued",
                    kind=SoundErrorKind.REALTIME_VIOLATION)
            self._slots[self._tail % self._capacity] = command
            self._tail += 1

    def drain(self) -> list:
        out = []
        while self._head != self._tail:
            out.append(self._slots[self._head % self._capacity])
            self._slots[self._head % self._capacity] = None
            self._head += 1
        return out


class RealtimeEngine:
    """A realtime audio engine rendering a session in fixed blocks."""

    def __init__(self, doc: SessionDocument, block_size: int = 512,
                 callback=None, stall_timeout_s: float = _DEFAULT_STALL_TIMEOUT_S):
        self.doc = doc
        self.sample_rate = doc.sample_rate
        self.block_size = int(block_size)
        self.callback = callback
        self.stall_timeout_s = float(stall_timeout_s)
        self._mix = MixEngine(doc, block_size=block_size)
        self._ring = _CommandRing()
        self._out = np.zeros((2, self.block_size), dtype=np.float64)
        self.beats = 0
        self.underruns = 0
        self.faults: list[dict] = []
        self.position = 0
        self.playing = False
        self.last_block_max = 0.0
        self._last_beat_at: float | None = None
        self._stall_reported = False
        self._failed = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # -- transport ----------------------------------------------------
    def play(self) -> None:
        self.playing = True
        self._last_beat_at = time.monotonic()

    def pause(self) -> None:
        self.playing = False

    def start(self) -> None:
        """Run the block loop on a wall-clock thread."""
        if self._thread is not None and self._thread.is_alive():
            raise SoundError("engine already running",
                             kind=SoundErrorKind.INVALID_ARGUMENT)
        self.play()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.playing = False
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None

    # -- control side --------------------------------------------------
    def send_command(self, command: dict) -> None:
        self._ring.push(command)

    def insert_plugin(self, track_id: int, plugin: object) -> None:
        self._mix.insert_plugin(track_id, plugin)

    # -- the block path -------------------------------------------------
    def _apply_commands(self) -> None:
        for command in self._ring.drain():
            kind = command.get("kind")
            if kind == "locate":
                self.position = int(command.get("sample", 0))
            elif kind == "play":
                self.playing = True
            elif kind == "pause":
                self.playing = False
            elif kind == "stop":
                self.playing = False
                self.position = 0

    def _render_block(self) -> None:
        """One guarded block: the callback law's unit of work."""
        t0 = time.perf_counter()
        self._apply_commands()
        if not self.playing:
            self._out.fill(0.0)
            self.last_block_max = 0.0
            return
        if self._failed:
            self._out.fill(0.0)
            self.last_block_max = 0.0
            self.beats += 1
            self._last_beat_at = time.monotonic()
            return
        try:
            if self.callback is not None:
                self.callback(self.beats, self._out)
            else:
                left, right, _ = self._mix.render(
                    self.position, self.block_size)
                self._out[0] = left
                self._out[1] = right
        except Exception as exc:  # noqa: BLE001 — guarded_block
            self._failed = True
            self.playing = False
            self.faults.append({
                "kind": "callback",
                "detail": f"{type(exc).__name__}: {exc}",
                "block": self.beats,
            })
            self._out.fill(0.0)
            self.last_block_max = 0.0
            self.beats += 1
            self._last_beat_at = time.monotonic()
            return
        self.last_block_max = float(
            np.max(np.abs(self._out))) if self.block_size else 0.0
        self.position += self.block_size
        self.beats += 1
        self._last_beat_at = time.monotonic()
        elapsed = time.perf_counter() - t0
        budget = self.block_size / self.sample_rate
        if elapsed > budget:
            self.underruns += 1

    def drive(self, n_blocks: int) -> None:
        """Render n blocks synchronously (deterministic tests)."""
        if self._thread is not None and self._thread.is_alive():
            raise SoundError("drive() while the wall-clock thread runs",
                             kind=SoundErrorKind.INVALID_ARGUMENT)
        for _ in range(int(n_blocks)):
            self._render_block()

    def _run(self) -> None:
        budget = self.block_size / self.sample_rate
        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            self._render_block()
            spent = time.perf_counter() - t0
            remaining = budget - spent
            if remaining > 0:
                self._stop_event.wait(remaining)

    # -- health ---------------------------------------------------------
    def poll(self) -> None:
        """Check for a stall; record it once, the way upstream's
        ``Fault.trouble`` reports a stoppage once."""
        if self._stall_reported or not self.playing:
            return
        if self._last_beat_at is None:
            return
        if time.monotonic() - self._last_beat_at >= self.stall_timeout_s:
            self._stall_reported = True
            self.faults.append({
                "kind": "stalled",
                "detail": f"no block for {self.stall_timeout_s}s while playing",
                "block": self.beats,
            })

    @property
    def stalled(self) -> bool:
        self.poll()
        return self._stall_reported
