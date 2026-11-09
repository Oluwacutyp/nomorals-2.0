"""General-purpose live progress for chat: message-edit sequences
driven by REAL milestones.

Any long operation — downloads, proxy checking, long tool runs,
boot/start sequences — uses this to show live progress instead of
silence. It is transport-agnostic: the caller binds ``emit`` to
whatever sends an edit (e.g. ``TelegramBotAdapter.edit_progress``)
and this helper decides *what* to show and *when* (throttled through
:mod:`nomorals.social.chat.progress_throttle`).

Honesty contract (owner rule, 2026-10-26): every step reflects a
REAL milestone — a completed check, a finished batch, a real state
change. Percentages and bars come ONLY from real counts supplied by
the caller (``done``/``total``); a step with no counts renders as
plain milestone text, never a bar. If a step cannot be tied to real
progress, it does not exist — this helper never invents percentages
and never renders a decorative bar.

Status grammar (design absorption #1, 2026-11-09 — from the React
Bits Micro agent-status patterns: Status Mark, Lattice Loader,
Call Chip, Thought Line): a reporter constructed with a ``title``
renders its ONE live message in a single grammar everywhere:

* a header line — a status glyph that morphs by phase (a rotating
  arc while working, ✓ done, ✗ failed), the job title, and the
  job's total elapsed time;
* the step trace — completed steps fold to ``✓ <step>`` lines and
  the running step carries the working glyph plus its own elapsed
  readout, so a long tool call shows how long IT has been running;
* on finish the whole trace COLLAPSES into one summary line —
  ``✓ <title> — <summary> · <elapsed> · <n> steps`` — the detail
  stays in the journal/logs, the chat keeps the outcome (Thought
  Line's fold, and the no-meta-narration doctrine).

A reporter WITHOUT a title keeps the original verbatim behaviour
(the caller's text is the message); the grammar is the titled
reporter's contract, and the production live-message flows (proxy
lab, web navigation) are titled.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from .progress_throttle import ProgressThrottler

_log = logging.getLogger(__name__)

__all__ = [
    "ProgressReporter",
    "render_bar",
    "format_elapsed",
    "GLYPH_DONE",
    "GLYPH_FAILED",
    "SPIN_FRAMES",
    "MAX_TRACE_STEPS",
]

#: Terminal glyphs — the grammar's only success/failure marks.
GLYPH_DONE = "✓"
GLYPH_FAILED = "✗"

#: Working-phase glyph cycle: each render advances one frame, so
#: the sequence of edits reads as motion in a surface that cannot
#: animate (the Lattice Loader wave, in text).
SPIN_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

#: Completed steps shown in the running trace before the oldest
#: fold into one elision line (the present layer's "… (+N more)"
#: shape) — one message, never a split dump.
MAX_TRACE_STEPS = 12


def render_bar(done: int, total: int, *, width: int = 10) -> str:
    """``▓▓▓░░`` bar from REAL counts only.

    Returns "" when there is nothing to count (total <= 0) — callers
    must not render a bar in that case. ``done`` is clamped to
    ``[0, total]`` so a miscounted caller cannot print >100%.
    """
    if total <= 0:
        return ""
    done = max(0, min(int(done), int(total)))
    filled = int(done / total * width + 0.5)  # half-up: 25% → 3 cells
    return "▓" * filled + "░" * (width - filled)


def format_elapsed(seconds: float) -> str:
    """Elapsed readout for the grammar: ``7s`` under a minute,
    ``1m07s`` under an hour, ``1h02m`` beyond. Negative clamps to 0
    (a skewed clock must not print a negative duration)."""
    total = max(0, int(seconds))
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m{total % 60:02d}s"
    return f"{total // 3600}h{(total % 3600) // 60:02d}m"


class _Step:
    """One step of the trace: a plain milestone, or a counted step
    whose counts keep updating in place while it is current."""

    __slots__ = ("label", "done", "total", "counted", "since")

    def __init__(self, label: str, since: float, *,
                 done: int = 0, total: int = 0,
                 counted: bool = False) -> None:
        self.label = label
        self.done = done
        self.total = total
        self.counted = counted
        self.since = since

    def text(self) -> str:
        """The step's own line content (no glyph, no elapsed)."""
        if self.counted and self.total > 0:
            bar = render_bar(self.done, self.total)
            return f"{self.label}: {self.done}/{self.total} {bar}"
        return self.label


class ProgressReporter:
    """Throttled progress edits for one long operation.

    ``emit`` is called with each new text that passes the throttle
    (the caller binds it to the real edit call). ``finish`` always
    emits (the final state change is never throttled away). All
    methods are thread-safe: testing/progress callbacks may fire
    from worker threads.

    With ``title`` set, the reporter renders the status grammar
    (see the module docstring): steps accumulate into a trace, the
    running step shows its elapsed time, :meth:`tick` refreshes the
    elapsed readouts (an optional daemon ticker thread calls it
    every ``tick_s`` seconds when ``tick_s > 0``), and
    :meth:`finish` / :meth:`fail` collapse the trace into ONE
    terminal line. Without a title, the historical verbatim
    behaviour is kept exactly.
    """

    def __init__(self, emit: Callable[[str], None], *,
                 throttler: ProgressThrottler | None = None,
                 title: str = "",
                 clock: Callable[[], float] | None = None,
                 tick_s: float = 0.0) -> None:
        self._emit = emit
        self._clock = clock or time.monotonic
        if throttler is None and clock is not None:
            throttler = ProgressThrottler(clock=clock)
        self._throttler = throttler or ProgressThrottler()
        self._title = str(title or "")
        self._tick_s = max(0.0, float(tick_s or 0.0))
        self._lock = threading.Lock()
        self._started = False
        self._finished = False
        # Grammar state (titled mode only).
        self._started_at = 0.0
        self._steps: list[_Step] = []
        self._current: _Step | None = None
        self._spin = 0
        self._ticker: threading.Thread | None = None
        self._ticker_stop = threading.Event()

    # ── shared verbs ────────────────────────────────────────────

    def milestone(self, text: str) -> None:
        """Named real step (e.g. "scraped 3 sources"). Throttled."""
        with self._lock:
            if self._finished:
                return
            self._started = True
            if not self._title:
                if self._throttler.should_edit(text):
                    self._emit(text)
                return
            now = self._clock()
            self._note_start(now)
            current = self._current
            if current is not None and not current.counted \
                    and current.label == text:
                # Same step reported again: a refresh, not a new
                # step — re-render so its elapsed readout moves.
                pass
            else:
                self._graduate(now)
                self._current = _Step(text, now)
            self._render(now)

    def count(self, done: int, total: int, label: str) -> None:
        """Counted progress: ``label: done/total + bar``.

        The bar comes from the caller's real counts. When total <= 0
        the line is just the label (no bar, no percentage).
        """
        if not self._title:
            if total and total > 0:
                bar = render_bar(done, total)
                text = f"{label}: {done}/{total} {bar}"
            else:
                text = label
            self.milestone(text)
            return
        with self._lock:
            if self._finished:
                return
            self._started = True
            now = self._clock()
            self._note_start(now)
            current = self._current
            if current is not None and current.counted \
                    and current.label == label:
                current.done = int(done)
                current.total = int(total)
            else:
                self._graduate(now)
                self._current = _Step(label, now, done=int(done),
                                      total=int(total), counted=True)
            self._render(now)

    def finish(self, text: str) -> None:
        """Final state, success (e.g. "✅ Done — file sent").
        Always emitted."""
        self._terminate(text, ok=True)

    def fail(self, text: str) -> None:
        """Final state, failure. The terminal line carries the ✗
        glyph; always emitted, like :meth:`finish`."""
        self._terminate(text, ok=False)

    def tick(self) -> None:
        """Refresh the running render (spinner + elapsed readouts).

        Throttled like every edit: a tick inside the edit interval
        sends nothing. A no-op without a title, before the first
        step, and after finish. The optional ticker thread calls
        this; tests and embedders may call it directly.
        """
        with self._lock:
            if self._finished or not self._title or not self._started:
                return
            self._render(self._clock())

    # ── grammar internals ───────────────────────────────────────

    def _note_start(self, now: float) -> None:
        if self._started_at == 0.0:
            self._started_at = now
            self._start_ticker()

    def _graduate(self, now: float) -> None:
        """Fold the current step into the completed trace."""
        if self._current is not None:
            self._steps.append(self._current)
            self._current = None

    def _frame(self) -> str:
        frame = SPIN_FRAMES[self._spin % len(SPIN_FRAMES)]
        self._spin += 1
        return frame

    def _render(self, now: float) -> None:
        frame = self._frame()
        elapsed = format_elapsed(now - self._started_at)
        lines = [f"{frame} {self._title} · {elapsed}"]
        steps = self._steps
        hidden = 0
        if len(steps) > MAX_TRACE_STEPS:
            hidden = len(steps) - MAX_TRACE_STEPS
            steps = steps[hidden:]
            lines.append(f"… (+{hidden} earlier steps)")
        for step in steps:
            lines.append(f"{GLYPH_DONE} {step.text()}")
        current = self._current
        if current is not None:
            step_elapsed = format_elapsed(now - current.since)
            lines.append(f"{frame} {current.text()} · {step_elapsed}")
        text = "\n".join(lines)
        if self._throttler.should_edit(text):
            self._emit(text)

    def _summary(self, text: str) -> str:
        """The caller's finish text as the terminal summary: the
        title prefix (callers used to bake it into every line) and
        any leading terminal glyph come off — the grammar owns
        both now."""
        summary = str(text or "").strip()
        title = self._title
        if title and summary.startswith(title):
            rest = summary[len(title):].lstrip()
            if rest.startswith(("—", "-")):
                rest = rest[1:].lstrip()
            if rest:
                summary = rest
        for glyph in ("✅", "❌", GLYPH_DONE, GLYPH_FAILED):
            if summary.startswith(glyph):
                summary = summary[len(glyph):].lstrip()
                break
        return summary

    def _terminate(self, text: str, *, ok: bool) -> None:
        with self._lock:
            if self._finished:
                return
            self._finished = True
            self._ticker_stop.set()
            if not self._title:
                self._emit(text)
                return
            now = self._clock()
            self._note_start(now)
            n_steps = len(self._steps) + (1 if self._current else 0)
            glyph = GLYPH_DONE if ok else GLYPH_FAILED
            summary = self._summary(text)
            elapsed = format_elapsed(now - self._started_at)
            parts = [glyph, self._title]
            head = " ".join(p for p in parts if p)
            if summary:
                head = f"{head} — {summary}"
            tail = [elapsed]
            if n_steps:
                tail.append(f"{n_steps} step" if n_steps == 1
                            else f"{n_steps} steps")
            self._emit(f"{head} · {' · '.join(tail)}")

    def _start_ticker(self) -> None:
        if self._tick_s <= 0 or self._ticker is not None:
            return

        def _loop() -> None:
            while not self._ticker_stop.wait(self._tick_s):
                try:
                    self.tick()
                except Exception:  # noqa: BLE001 - ticker never dies
                    _log.debug("progress ticker tick failed",
                               exc_info=True)

        self._ticker = threading.Thread(
            target=_loop, name="progress-ticker", daemon=True)
        self._ticker.start()

    @property
    def started(self) -> bool:
        return self._started

    @property
    def finished(self) -> bool:
        return self._finished
