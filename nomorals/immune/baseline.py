"""Adaptive cancer thresholds — percentiles from live samples, not magic numbers.

The first immune batch shipped ``consult_then_enforce`` with fixed
thresholds (spawned >= 100, or >= 20 with growth >= 3). Those numbers
were initial values, never tuned from live data. This module replaces
the fixed numbers with percentiles computed from an
:class:`AdaptiveBaseline` of real samples:

* ``MIN_SAMPLES = 30`` — below that, the baseline is honest about not
  knowing: :meth:`AdaptiveBaseline.thresholds` reports
  ``status="insufficient"`` and :func:`adaptive_consult_then_enforce`
  returns report-only, never enforces. No enforcement on a thin
  baseline.
* At >= 30 samples, thresholds are the 95th percentile (p95) of
  ``processes_spawned`` and of ``process_growth`` across the recorded
  samples. A threshold of 0 on either leg (degenerate baseline) is
  treated as insufficient — a leg that has only ever seen zero
  cannot say what "too much" is for that leg.
* :func:`collect_process_sample` reads the live process count via
  the platform backend — ``/proc`` on POSIX, Win32
  ``EnumProcesses`` (ctypes, stdlib only) on Windows, and optional
  ``psutil`` (when importable) as the cross-platform fallback — and
  derives growth as the ratio of the current count to the median of
  prior counts. The sample shape is identical on every platform
  (:data:`SAMPLE_KEYS`), so the learner cannot tell which OS a
  sample came from. When no backend can count on this platform it
  returns ``None`` — no fabricated sample, no enforcement — and
  :func:`sampling_availability` reports exactly that, with the
  reason, so callers can distinguish "platform cannot sample" from
  "no sample this tick".

The legacy fixed-threshold ``consult_then_enforce`` in ``police.py``
is untouched: existing callers keep their behaviour, and new callers
use the adaptive gate.
"""

from __future__ import annotations

import os
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "MIN_SAMPLES",
    "SAMPLE_KEYS",
    "AdaptiveBaseline",
    "Thresholds",
    "collect_process_sample",
    "sampling_availability",
    "adaptive_consult_then_enforce",
    "sample_over_thresholds",
]

#: Minimum samples before any percentile threshold is trusted.
MIN_SAMPLES = 30

#: Percentile used for adaptive thresholds (95th percentile).
PERCENTILE = 95.0


def _percentile(values: List[float], pct: float) -> float:
    """Linear-interpolation percentile over sorted ``values``."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of an empty sample")
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


@dataclass
class Thresholds:
    """Outcome of :meth:`AdaptiveBaseline.thresholds`."""

    status: str = "insufficient"  # insufficient | ready
    samples: int = 0
    min_samples: int = MIN_SAMPLES
    spawned_p95: Optional[float] = None
    growth_p95: Optional[float] = None
    reason: str = ""


class AdaptiveBaseline:
    """Accumulates live samples; derives percentile thresholds.

    Samples are plain mappings with at least ``processes_spawned`` and
    optionally ``process_growth`` (ratio vs. prior median). Growth is
    computed by :func:`collect_process_sample` when the caller passes
    the baseline's own history; callers may also supply it directly.

    ``window`` bounds the baseline to the most recent ``window``
    samples (a recency window: the baseline tracks current behaviour
    and the sample list cannot grow unbounded). ``None`` (the default)
    keeps every sample, the historical behaviour.
    """

    def __init__(self, window: Optional[int] = None) -> None:
        if window is not None and int(window) < 1:
            raise ValueError("baseline window must be a positive count")
        self._samples: List[Dict[str, float]] = []
        self._window: Optional[int] = int(window) if window else None

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def window(self) -> Optional[int]:
        """The recency window (``None`` = unbounded)."""
        return self._window

    def samples(self) -> List[Dict[str, float]]:
        """A copy of the recorded (cleaned) samples, oldest first."""
        return [dict(s) for s in self._samples]

    def record(self, sample: Dict[str, Any]) -> int:
        """Record one sample. Returns the total sample count.

        ``process_growth`` accepts the legacy ``growth`` alias — the
        same fallback :func:`adaptive_consult_then_enforce` and
        ``police._exceeds_cancer`` apply when reading a sample.
        Dropping the alias here silently recorded growth 0.0 for
        callers using the legacy key.
        """
        clean: Dict[str, float] = {}
        for key in ("processes_spawned", "tokens", "errors"):
            try:
                clean[key] = float(sample.get(key, 0) or 0)
            except (TypeError, ValueError):
                clean[key] = 0.0
        try:
            clean["process_growth"] = float(
                sample.get("process_growth", sample.get("growth", 0)) or 0
            )
        except (TypeError, ValueError):
            clean["process_growth"] = 0.0
        self._samples.append(clean)
        if self._window is not None and len(self._samples) > self._window:
            del self._samples[: len(self._samples) - self._window]
        return len(self._samples)

    def thresholds(self) -> Thresholds:
        """p95 thresholds, or insufficient when the baseline is thin."""
        n = len(self._samples)
        if n < MIN_SAMPLES:
            return Thresholds(
                status="insufficient",
                samples=n,
                reason=(
                    f"baseline from {n} samples, need {MIN_SAMPLES} — "
                    "report-only, never enforce"
                ),
            )
        spawned = [s["processes_spawned"] for s in self._samples]
        growth = [s["process_growth"] for s in self._samples]
        spawned_p95 = _percentile(spawned, PERCENTILE)
        growth_p95 = _percentile(growth, PERCENTILE)
        if spawned_p95 <= 0 and growth_p95 <= 0:
            return Thresholds(
                status="insufficient",
                samples=n,
                spawned_p95=spawned_p95,
                growth_p95=growth_p95,
                reason="degenerate all-zero baseline — cannot say what 'too much' is",
            )
        if spawned_p95 <= 0 or growth_p95 <= 0:
            leg = "processes_spawned" if spawned_p95 <= 0 else "process_growth"
            return Thresholds(
                status="insufficient",
                samples=n,
                spawned_p95=spawned_p95,
                growth_p95=growth_p95,
                reason=(
                    f"degenerate baseline: {leg} has only ever been 0 — "
                    "cannot say what 'too much' is"
                ),
            )
        return Thresholds(
            status="ready",
            samples=n,
            spawned_p95=spawned_p95,
            growth_p95=growth_p95,
            reason=(
                f"p95 from {n} live samples: spawned {spawned_p95:g}, "
                f"growth {growth_p95:g}"
            ),
        )

    def median_spawned(self) -> Optional[float]:
        """Median of recorded ``processes_spawned`` (None when empty)."""
        if not self._samples:
            return None
        return statistics.median(s["processes_spawned"] for s in self._samples)


#: The sample-shape contract. Every platform backend produces a
#: count; :func:`collect_process_sample` shapes it into exactly
#: these keys (floats), so :class:`AdaptiveBaseline` consumes a
#: Windows sample identically to a Linux one — the learner cannot
#: tell which OS a sample came from, and a future backend cannot
#: silently drift the fields.
SAMPLE_KEYS = ("processes_spawned", "process_growth")


# ── platform backends: one count, three routes ────────────────────
#
# The repo carries ZERO mandatory third-party dependencies
# (pyproject: ``dependencies = []``), and psutil is not declared in
# any optional group either — so the primary backends are stdlib:
# ``/proc`` on POSIX (the original route, unchanged) and Win32
# ``EnumProcesses`` via ctypes on Windows. psutil, when importable,
# is the cross-platform fallback, following the optional-accelerator
# convention of ``nomorals/power/monitor.py`` (try-import, injectable
# module, degrade cleanly). Every backend returns the live process
# COUNT or ``None`` — never a fabricated number.

def _count_processes_proc() -> Optional[int]:
    """POSIX: count numeric entries in ``/proc`` (the original)."""
    try:
        entries = os.listdir("/proc")
    except OSError:
        return None
    return sum(1 for name in entries if name.isdigit())


def _count_processes_win32(ctypes_mod: Any = None) -> Optional[int]:
    """Windows: count processes via psapi ``EnumProcesses`` (ctypes).

    PID 0 (the System Idle Process) is excluded: it has no ``/proc``
    counterpart on POSIX, and the learner compares counts across
    time on ONE host — a constant phantom would only shift the
    baseline, but excluding it keeps the semantics "live processes"
    on both platforms. The buffer doubles until the returned byte
    count fits (a full buffer means the list may be truncated).
    """
    ct = ctypes_mod
    if ct is None:
        try:
            import ctypes as ct
        except Exception:  # noqa: BLE001 - no ctypes, no backend
            return None
    try:
        windll = getattr(ct, "windll", None)
        if windll is None:
            return None
        psapi = windll.psapi
        size = 1024
        while size <= (1 << 20):
            buf = (ct.c_ulong * size)()
            returned = ct.c_ulong(0)
            if not psapi.EnumProcesses(
                    ct.byref(buf), ct.sizeof(buf), ct.byref(returned)):
                return None
            n = int(returned.value) // int(ct.sizeof(ct.c_ulong()))
            if n < size:
                return sum(1 for pid in buf[:n] if pid)
            size *= 2
        return None
    except Exception:  # noqa: BLE001 - any ctypes failure: no sample
        return None


def _count_processes_psutil(psutil_mod: Any = None) -> Optional[int]:
    """Optional accelerator: ``len(psutil.pids())`` when available."""
    mod = psutil_mod
    if mod is None:
        try:
            import psutil as mod
        except Exception:  # noqa: BLE001 - not installed: no backend
            return None
    try:
        return len(mod.pids())
    except Exception:  # noqa: BLE001 - a broken psutil must not lie
        return None


def _process_count(*, os_name: Optional[str] = None,
                   proc_count_fn: Any = None,
                   win32_count_fn: Any = None,
                   psutil_count_fn: Any = None
                   ) -> Any:
    """Dispatch to the platform backends, in order.

    Returns ``(count, backend, reason)``: ``count`` is an int and
    ``backend`` one of ``"proc"`` / ``"win32"`` / ``"psutil"`` on
    success; ``(None, "", reason)`` when every applicable backend
    failed, with the attempted backends named in the reason. The
    ``*_count_fn`` parameters are the injection seam (tests, future
    backends) — each is a zero-arg callable returning a count or
    ``None``.
    """
    name = os.name if os_name is None else os_name
    if name == "nt":
        attempts = [
            ("win32", win32_count_fn or _count_processes_win32),
            ("psutil", psutil_count_fn or _count_processes_psutil),
        ]
    else:
        attempts = [
            ("proc", proc_count_fn or _count_processes_proc),
            ("psutil", psutil_count_fn or _count_processes_psutil),
        ]
    tried: List[str] = []
    for backend, fn in attempts:
        tried.append(backend)
        try:
            count = fn()
        except Exception:  # noqa: BLE001 - a backend must not raise
            count = None
        if count is not None:
            return int(count), backend, ""
    return None, "", (
        "no backend could count processes "
        f"(tried: {', '.join(tried)})")


def sampling_availability(count_fn: Any = None) -> Dict[str, Any]:
    """Can this platform produce a live process sample at all?

    Returns ``{"available": bool, "backend": str, "platform": str,
    "reason": str}``. The probe is a real count attempt (cheap and
    side-effect free). ``available=False`` is a standing platform
    fact — distinct from a single tick collecting nothing — and the
    reason names what was tried. ``count_fn`` is the injection seam:
    a zero-arg callable returning ``(count, backend, reason)``.
    """
    if count_fn is not None:
        count, backend, reason = count_fn()
    else:
        count, backend, reason = _process_count()
    if count is None:
        return {"available": False, "backend": "",
                "platform": os.name,
                "reason": reason or "no backend could count processes"}
    return {"available": True, "backend": backend,
            "platform": os.name, "reason": ""}


def collect_process_sample(baseline: Optional[AdaptiveBaseline] = None,
                           *, count_fn: Any = None
                           ) -> Optional[Dict[str, float]]:
    """Read the live process count; derive growth vs. baseline median.

    The count comes from the platform backend (``/proc`` on POSIX,
    Win32 ``EnumProcesses`` on Windows, psutil when importable as a
    fallback — see :func:`_process_count`); the returned sample has
    exactly :data:`SAMPLE_KEYS` on every platform. Returns ``None``
    when no backend can count on this platform — no fabricated
    sample (see :func:`sampling_availability` for the reason).
    Growth is ``current / median(prior)`` when a baseline with prior
    samples is supplied, else 1.0 (no history). ``count_fn`` is the
    injection seam: a zero-arg callable returning
    ``(count, backend, reason)``.
    """
    if count_fn is not None:
        count, _backend, _reason = count_fn()
    else:
        count, _backend, _reason = _process_count()
    if count is None:
        return None
    current = float(count)
    growth = 1.0
    if baseline is not None:
        median = baseline.median_spawned()
        if median and median > 0:
            growth = current / median
    return {"processes_spawned": current, "process_growth": growth}


def sample_over_thresholds(sample: Dict[str, Any],
                           thresholds: Thresholds) -> bool:
    """True when ``sample`` is strictly over BOTH p95 thresholds.

    The one definition of "over" for the adaptive gate: a sample
    sitting exactly AT the p95 of the baseline is behaviour the
    baseline itself has seen, so the comparison is strict on both
    legs. Non-ready thresholds are never "over".
    """
    if thresholds.status != "ready":
        return False
    spawned = float(sample.get("processes_spawned", 0) or 0)
    growth = float(sample.get("process_growth", sample.get("growth", 0)) or 0)
    return (
        thresholds.spawned_p95 is not None and spawned > thresholds.spawned_p95
        and thresholds.growth_p95 is not None and growth > thresholds.growth_p95
    )


def adaptive_consult_then_enforce(
    sample: Dict[str, Any],
    baseline: AdaptiveBaseline,
    *,
    is_system_process: bool = False,
    owner_directed: bool = False,
    enforce_fn: Any = None,
) -> Dict[str, Any]:
    """Adaptive consult-then-enforce gate.

    Returns a plain receipt dict (``action`` / ``reason`` / ``details``)
    so callers need not import :class:`police.Receipt`:

    * owner-directed → report-only, never enforce;
    * baseline insufficient → report-only, never enforce, with the
      honest sample count;
    * system process over BOTH p95 thresholds (spawned AND growth) →
      call ``enforce_fn`` (when wired) and return an enforce receipt;
    * anything else → report-only.
    """
    from .police import Receipt

    target = str(sample.get("process_id", ""))
    if owner_directed:
        receipt = Receipt(
            rung=1, action="report_only", target=target,
            reason="owner-directed: report only, never enforce",
            details={"blocked": False, "enforced": False},
        )
        return {"action": receipt.action, "reason": receipt.reason,
                "details": receipt.details, "receipt": receipt}
    thresholds = baseline.thresholds()
    if thresholds.status != "ready":
        receipt = Receipt(
            rung=1, action="report_only", target=target,
            reason=f"adaptive baseline insufficient: {thresholds.reason}",
            details={"blocked": False, "enforced": False,
                     "baseline_samples": thresholds.samples,
                     "min_samples": thresholds.min_samples},
        )
        return {"action": receipt.action, "reason": receipt.reason,
                "details": receipt.details, "receipt": receipt}
    spawned = float(sample.get("processes_spawned", 0) or 0)
    growth = float(sample.get("process_growth", sample.get("growth", 0)) or 0)
    # Strictly OVER both thresholds (see sample_over_thresholds):
    # with a low-variance baseline (e.g. a constant process count)
    # ``>=`` would enforce on the most normal sample there is.
    over = sample_over_thresholds(sample, thresholds)
    if is_system_process and over:
        if enforce_fn is not None:
            enforce_fn(sample)
        receipt = Receipt(
            rung=5, action="enforce", target=target,
            reason=(
                f"system process over adaptive thresholds "
                f"(spawned {spawned:g} > p95 {thresholds.spawned_p95:g}, "
                f"growth {growth:g} > p95 {thresholds.growth_p95:g}): enforced"
            ),
            reversible=False,
            details={"enforced": True, "sample": dict(sample),
                     "thresholds": {"spawned_p95": thresholds.spawned_p95,
                                    "growth_p95": thresholds.growth_p95}},
        )
        return {"action": receipt.action, "reason": receipt.reason,
                "details": receipt.details, "receipt": receipt}
    receipt = Receipt(
        rung=1, action="report_only", target=target,
        reason="below adaptive enforcement thresholds: report only",
        details={"blocked": False, "enforced": False,
                 "thresholds": {"spawned_p95": thresholds.spawned_p95,
                                "growth_p95": thresholds.growth_p95}},
    )
    return {"action": receipt.action, "reason": receipt.reason,
            "details": receipt.details, "receipt": receipt}
