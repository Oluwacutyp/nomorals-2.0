"""Metrics and tracing without a dependency on Prometheus/OpenTelemetry.

A long-running autonomous system that cannot tell you what it is doing is
undebuggable. This module provides:

* :class:`Metrics` — counters, gauges, histograms, timers, thread-safe.
* :func:`trace` — nested spans with parent linkage, exportable as JSON.
* :class:`HealthReport` — a single snapshot of the whole system.

The wire format of :meth:`Metrics.as_prometheus` is compatible with Prometheus
text exposition, so scraping it later costs nothing.
"""

from __future__ import annotations

import contextvars
import math
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Iterator

__all__ = ["HealthReport", "Metrics", "Span", "Tracer", "global_metrics", "trace"]


class Metrics:
    """Thread-safe metric registry."""

    def __init__(self, *, histogram_buckets: tuple[float, ...] | None = None) -> None:
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, "_Histogram"] = {}
        self._labels: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._buckets = histogram_buckets or (
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0
        )
        self._lock = threading.RLock()
        self._created = time.time()

    # -- counters ------------------------------------------------------------
    def incr(self, name: str, value: float = 1.0, **labels: str) -> None:
        with self._lock:
            self._counters[name] += value
            if labels:
                bucket = self._labels[name]
                key = _label_key(labels)
                bucket[key] += value

    # -- gauges --------------------------------------------------------------
    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def gauge_add(self, name: str, delta: float) -> float:
        with self._lock:
            self._gauges[name] = self._gauges.get(name, 0.0) + delta
            return self._gauges[name]

    # -- histograms / timers -------------------------------------------------
    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._histograms.setdefault(name, _Histogram(self._buckets)).observe(value)

    @contextmanager
    def timer(self, name: str) -> Iterator["_Timer"]:
        timer = _Timer()
        timer.start()
        try:
            yield timer
        finally:
            timer.stop()
            self.observe(name, timer.elapsed)
            self.incr(f"{name}.count")

    def time_call(self, name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        with self.timer(name):
            return fn(*args, **kwargs)

    # -- export --------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "uptime": round(time.time() - self._created, 3),
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {n: h.summary() for n, h in self._histograms.items()},
                "labels": {n: dict(v) for n, v in self._labels.items()},
            }

    def get(self, name: str) -> float:
        with self._lock:
            if name in self._counters:
                return self._counters[name]
            if name in self._gauges:
                return self._gauges[name]
            return 0.0

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._labels.clear()

    def as_prometheus(self, namespace: str = "nomorals") -> str:
        """Prometheus text exposition format."""
        lines: list[str] = []
        with self._lock:
            for name, value in sorted(self._counters.items()):
                metric = _sanitize(f"{namespace}_{name}")
                lines.append(f"# TYPE {metric} counter")
                lines.append(f"{metric} {value}")
            for name, value in sorted(self._gauges.items()):
                metric = _sanitize(f"{namespace}_{name}")
                lines.append(f"# TYPE {metric} gauge")
                lines.append(f"{metric} {value}")
            for name, hist in sorted(self._histograms.items()):
                metric = _sanitize(f"{namespace}_{name}_seconds")
                lines.append(f"# TYPE {metric} histogram")
                lines.extend(hist.prometheus(metric))
        return "\n".join(lines) + "\n"


class _Histogram:
    __slots__ = ("_buckets", "_counts", "_sum", "_count", "_max", "_min", "_recent")

    def __init__(self, buckets: tuple[float, ...]) -> None:
        self._buckets = buckets
        self._counts = [0] * (len(buckets) + 1)
        self._sum = 0.0
        self._count = 0
        self._max = 0.0
        self._min = math.inf
        self._recent: Deque[float] = deque(maxlen=1024)

    def observe(self, value: float) -> None:
        self._sum += value
        self._count += 1
        self._max = max(self._max, value)
        self._min = min(self._min, value)
        self._recent.append(value)
        for i, bound in enumerate(self._buckets):
            if value <= bound:
                self._counts[i] += 1
                return
        self._counts[-1] += 1

    def percentile(self, q: float) -> float:
        if not self._recent:
            return 0.0
        ordered = sorted(self._recent)
        idx = min(len(ordered) - 1, max(0, int(math.ceil(q * len(ordered))) - 1))
        return ordered[idx]

    def summary(self) -> dict[str, float]:
        if self._count == 0:
            return {"count": 0, "sum": 0.0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
        return {
            "count": self._count,
            "sum": round(self._sum, 6),
            "avg": round(self._sum / self._count, 6),
            "min": round(self._min, 6),
            "max": round(self._max, 6),
            "p50": round(self.percentile(0.50), 6),
            "p95": round(self.percentile(0.95), 6),
            "p99": round(self.percentile(0.99), 6),
        }

    def prometheus(self, metric: str) -> list[str]:
        out: list[str] = []
        cumulative = 0
        for bound, count in zip(self._buckets, self._counts[:-1]):
            cumulative += count
            out.append(f'{metric}_bucket{{le="{bound}"}} {cumulative}')
        cumulative += self._counts[-1]
        out.append(f'{metric}_bucket{{le="+Inf"}} {cumulative}')
        out.append(f"{metric}_sum {self._sum}")
        out.append(f"{metric}_count {self._count}")
        return out


class _Timer:
    __slots__ = ("_start", "_end")

    def __init__(self) -> None:
        self._start = 0.0
        self._end = 0.0

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> None:
        self._end = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return (self._end or time.perf_counter()) - self._start


def _label_key(labels: dict[str, str]) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))


def _sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name.replace(".", "_"))


# ── Tracing ────────────────────────────────────────────────────────────────────


@dataclass
class Span:
    """A unit of work with timing and parent linkage."""

    name: str
    trace_id: str
    span_id: str
    parent_id: str = ""
    start: float = 0.0
    end: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    error: str = ""
    children: list["Span"] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return (self.end or time.perf_counter()) - self.start

    def finish(self, status: str = "ok", error: str = "") -> None:
        self.end = time.perf_counter()
        self.status = status
        self.error = error

    def to_dict(self, *, nested: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "duration_ms": round(self.duration * 1000, 3),
            "status": self.status,
            "attributes": self.attributes,
        }
        if self.error:
            payload["error"] = self.error
        if nested and self.children:
            payload["children"] = [c.to_dict(nested=True) for c in self.children]
        return payload


class Tracer:
    """Context-propagating tracer.

    Spans nest automatically through a :mod:`contextvars` token, which means the
    parent linkage is correct across ``await`` points and thread pool submissions
    that copy the context.
    """

    def __init__(self, *, max_traces: int = 512) -> None:
        self._token: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
            "nm_current_span", default=None
        )
        self._traces: Deque[Span] = deque(maxlen=max_traces)
        self._lock = threading.Lock()

    def current(self) -> Span | None:
        return self._token.get()

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        from .ids import new_short_id

        parent = self._token.get()
        span = Span(
            name=name,
            trace_id=parent.trace_id if parent else new_short_id("trc_"),
            span_id=new_short_id("spn_"),
            parent_id=parent.span_id if parent else "",
            start=time.perf_counter(),
            attributes=dict(attributes),
        )
        token = self._token.set(span)
        if parent is not None:
            parent.children.append(span)
        else:
            with self._lock:
                self._traces.append(span)
        try:
            yield span
        except Exception as exc:  # noqa: BLE001 - record then re-raise
            span.finish(status="error", error=f"{type(exc).__name__}: {exc}")
            raise
        else:
            span.finish()
        finally:
            self._token.reset(token)

    def traces(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._traces)[-limit:]
        return [s.to_dict() for s in items]

    def clear(self) -> None:
        with self._lock:
            self._traces.clear()


@dataclass
class HealthReport:
    """Aggregated snapshot used by ``nm health`` and ``GET /health``."""

    healthy: bool
    version: str
    uptime: float
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "healthy": self.healthy,
            "version": self.version,
            "uptime": round(self.uptime, 2),
            "checks": self.checks,
            "metrics": self.metrics,
            "generated_at": self.generated_at,
        }

    def as_text(self) -> str:
        head = "OK" if self.healthy else "DEGRADED"
        lines = [f"status: {head}   version: {self.version}   uptime: {self.uptime:.0f}s", ""]
        for name, check in self.checks.items():
            mark = "ok " if check.get("ok") else "ERR"
            detail = check.get("detail", "")
            lines.append(f"  [{mark}] {name:<20} {detail}")
        return "\n".join(lines)


#: Process-wide metric registry.
global_metrics = Metrics()
#: Process-wide tracer.
global_tracer = Tracer()


def trace(name: str, **attributes: Any):
    """Shorthand for ``global_tracer.span(name, **attributes)``."""
    return global_tracer.span(name, **attributes)
