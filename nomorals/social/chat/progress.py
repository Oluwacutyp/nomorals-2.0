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
"""

from __future__ import annotations

import threading
from typing import Callable

from .progress_throttle import ProgressThrottler

__all__ = ["ProgressReporter", "render_bar"]


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


class ProgressReporter:
    """Throttled progress edits for one long operation.

    ``emit`` is called with each new text that passes the throttle
    (the caller binds it to the real edit call). ``finish`` always
    emits (the final state change is never throttled away). All
    methods are thread-safe: testing/progress callbacks may fire
    from worker threads.
    """

    def __init__(self, emit: Callable[[str], None], *,
                 throttler: ProgressThrottler | None = None) -> None:
        self._emit = emit
        self._throttler = throttler or ProgressThrottler()
        self._lock = threading.Lock()
        self._started = False
        self._finished = False

    def milestone(self, text: str) -> None:
        """Named real step (e.g. "scraped 3 sources"). Throttled."""
        with self._lock:
            if self._finished:
                return
            self._started = True
            if self._throttler.should_edit(text):
                self._emit(text)

    def count(self, done: int, total: int, label: str) -> None:
        """Counted progress: ``label: done/total + bar``.

        The bar comes from the caller's real counts. When total <= 0
        the line is just the label (no bar, no percentage).
        """
        if total and total > 0:
            bar = render_bar(done, total)
            text = f"{label}: {done}/{total} {bar}"
        else:
            text = label
        self.milestone(text)

    def finish(self, text: str) -> None:
        """Final state (e.g. "✅ Done — file sent"). Always emitted."""
        with self._lock:
            if self._finished:
                return
            self._finished = True
            self._emit(text)

    @property
    def started(self) -> bool:
        return self._started

    @property
    def finished(self) -> bool:
        return self._finished
