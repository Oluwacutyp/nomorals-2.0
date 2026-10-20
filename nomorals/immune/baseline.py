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
* :func:`collect_process_sample` reads the live process count from
  ``/proc`` on Linux (stdlib only) and derives growth as the ratio of
  the current count to the median of prior counts. On non-Linux, or
  when ``/proc`` is unreadable, it returns ``None`` — no fabricated
  sample, no enforcement.

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
    "AdaptiveBaseline",
    "Thresholds",
    "collect_process_sample",
    "adaptive_consult_then_enforce",
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
    """

    def __init__(self) -> None:
        self._samples: List[Dict[str, float]] = []

    def __len__(self) -> int:
        return len(self._samples)

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


def collect_process_sample(baseline: Optional[AdaptiveBaseline] = None) -> Optional[Dict[str, float]]:
    """Read the live process count; derive growth vs. baseline median.

    Returns ``None`` on non-Linux or when ``/proc`` is unreadable —
    no fabricated sample. Growth is ``current / median(prior)`` when
    a baseline with prior samples is supplied, else 1.0 (no history).
    """
    try:
        entries = os.listdir("/proc")
    except OSError:
        return None
    current = float(sum(1 for name in entries if name.isdigit()))
    growth = 1.0
    if baseline is not None:
        median = baseline.median_spawned()
        if median and median > 0:
            growth = current / median
    return {"processes_spawned": current, "process_growth": growth}


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
    # Strictly OVER both thresholds: a sample sitting exactly AT the
    # p95 of the baseline is behaviour the baseline itself has seen —
    # with a low-variance baseline (e.g. a constant process count)
    # ``>=`` would enforce on the most normal sample there is.
    over = (
        thresholds.spawned_p95 is not None and spawned > thresholds.spawned_p95
        and thresholds.growth_p95 is not None and growth > thresholds.growth_p95
    )
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
