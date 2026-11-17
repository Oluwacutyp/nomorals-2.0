"""Local render timing — the record behind the CPU-slow route trigger.

Every successful local render records how long the pipeline took.
The store is a small atomic JSON ring (the last ``MAX_SAMPLES``
durations) beside the checkpoints, so the median survives restarts
without a database. When the local median exceeds
``SLOW_LOCAL_P50_S``, :meth:`Studio.generate <.studio.Studio.generate>`
routes the next single-image job to the attested forwarded GPU
instead of making the owner wait on a slow CPU — the same branch
the not-ready case already takes. A missing, empty, or corrupt
history means "not slow": the box renders locally exactly as
before, and the first renders seed the history.
"""

from __future__ import annotations

import contextlib
import json
import os
import statistics
import tempfile

from . import checkpoint_dir
from .failures import record_failure

__all__ = [
    "MAX_SAMPLES",
    "SLOW_LOCAL_P50_S",
    "RenderTimingStore",
    "default_timing_store",
    "local_is_slow",
    "local_timing_line",
]

#: A local median render slower than this routes to the forwarded GPU.
#: Judgment call, not a measured constant: a forwarded render takes
#: tens of seconds plus a cold model load (``REMOTE_TIMEOUT_S`` is
#: 600 s, the generous bound), so past two minutes the GPU host is
#: plainly faster than waiting on this box — and two minutes is past
#: the point anyone waits in chat for one image.
SLOW_LOCAL_P50_S = 120.0

#: Ring cap — 50 renders is plenty for a stable median and keeps the
#: file a few hundred bytes.
MAX_SAMPLES = 50

_TIMINGS_FILENAME = "render_timings.json"


class RenderTimingStore:
    """Durable ring of recent local render durations (seconds).

    Bound to one JSON file (default: beside the checkpoints, so
    ``DEVON_IMGGEN_DIR`` moves it with everything else). Reads and
    writes never raise: a missing or corrupt file reads empty, and
    a failed write is recorded in the failure ledger and dropped —
    timing must never break a render.
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = path or os.path.join(
            checkpoint_dir(), _TIMINGS_FILENAME)

    def durations(self) -> list[float]:
        """Recorded durations, oldest first. Never raises."""
        try:
            with open(self.path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return []
        if not isinstance(data, list):
            return []
        out: list[float] = []
        for value in data:
            if isinstance(value, bool) or not isinstance(
                    value, (int, float)):
                continue
            out.append(float(value))
        return out

    def record(self, seconds: float) -> None:
        """Append one duration, capped at ``MAX_SAMPLES``. Never raises."""
        try:
            durations = self.durations()
            durations.append(float(seconds))
            durations = durations[-MAX_SAMPLES:]
            directory = os.path.dirname(self.path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                dir=directory or ".", prefix=".render_timings-",
                suffix=".tmp")
            replaced = False
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(durations, handle)
                os.replace(tmp_path, self.path)
                replaced = True
            finally:
                # A failed write must not leave a stray temp file
                # behind; a successful replace already moved it.
                if not replaced:
                    with contextlib.suppress(OSError):
                        os.unlink(tmp_path)
        except Exception as exc:  # noqa: BLE001 - timing never breaks a render
            record_failure("imggen.timing.record", exc)

    @property
    def count(self) -> int:
        """How many durations are recorded."""
        return len(self.durations())

    def p50(self) -> float | None:
        """Median local render time in seconds, or None with no history."""
        durations = self.durations()
        if not durations:
            return None
        return float(statistics.median(durations))


def default_timing_store() -> RenderTimingStore:
    """The store beside the checkpoints (``DEVON_IMGGEN_DIR`` aware)."""
    return RenderTimingStore()


def local_is_slow(store: RenderTimingStore | None = None,
                  ) -> tuple[bool, str]:
    """Is the local median render slower than ``SLOW_LOCAL_P50_S``?

    Returns ``(True, reason)`` when the median clears the threshold,
    else ``(False, "")``. No history means not slow — the box renders
    locally exactly as before.
    """
    store = store or default_timing_store()
    median = store.p50()
    if median is None or median <= SLOW_LOCAL_P50_S:
        return False, ""
    return True, (
        f"local renders too slow to use (median {median:.1f}s over "
        f"{store.count} recent "
        f"{'render' if store.count == 1 else 'renders'}, "
        f"slow above {SLOW_LOCAL_P50_S:.0f}s)")


def local_timing_line(store: RenderTimingStore | None = None,
                      ) -> str | None:
    """One display line for the checkpoints views, or None with no history."""
    store = store or default_timing_store()
    median = store.p50()
    if median is None:
        return None
    noun = "render" if store.count == 1 else "renders"
    return (f"local render median: {median:.1f}s over {store.count} "
            f"{noun} (slow above {SLOW_LOCAL_P50_S:.0f}s)")
