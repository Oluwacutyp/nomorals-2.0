"""Metrics and tracing without a dependency on Prometheus/OpenTelemetry.

A long-running autonomous system that cannot tell you what it is doing is
undebuggable. This module provides:

* :class:`Metrics` — counters, gauges, histograms, timers, thread-safe,
  with *labeled* Prometheus exposition.
* :class:`MetricsExporter` — background thread that snapshots metrics to a
  JSON file (the operator's window on a headless box) and can push to a
  Prometheus Pushgateway.
* :class:`Tracer` — nested spans with parent linkage via contextvars,
  head sampling with an always-keep-errors tail rule, span events,
  exception recording, attribute redaction, JSONL export, and W3C
  ``traceparent`` propagation.
* :class:`HealthChecker` / :class:`HealthReport` — a real check engine:
  named checks with timeouts, liveness/readiness split, TTL caching,
  transition tracking, and a report that never leaks secrets.

Wire formats stay compatible with Prometheus text exposition and OTLP-shaped
JSON, so adopting the real backends later costs nothing.
"""

from __future__ import annotations

import concurrent.futures as cf
import contextvars
import json
import math
import os
import secrets
import threading
import time
import urllib.request
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Iterator

__all__ = [
    "HealthCheck",
    "HealthChecker",
    "HealthReport",
    "Metrics",
    "MetricsExporter",
    "Span",
    "Tracer",
    "global_metrics",
    "global_tracer",
    "trace",
]


def _redact_attr(value: Any) -> Any:
    """Span/log attributes must never carry secrets. Reuse the logging
    redactor (same patterns) so both pipelines agree on what a secret is."""
    if isinstance(value, str):
        from .logging_setup import redact

        return redact(value)
    if isinstance(value, dict):
        return {k: _redact_attr(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_attr(v) for v in value]
    return value


class Metrics:
    """Thread-safe metric registry.

    Naming discipline (the trash builds taught this): one name, one unit,
    snake_case, ``_total`` implied on counters, ``_seconds`` on timers.
    High-cardinality values (user ids, raw error text, full URLs) do NOT
    belong in label values — put them in logs/traces instead.
    """

    def __init__(self, *, histogram_buckets: tuple[float, ...] | None = None) -> None:
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, "_Histogram"] = {}
        self._labels: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._gauge_labels: dict[str, dict[str, float]] = defaultdict(dict)
        self._buckets = histogram_buckets or (
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0
        )
        self._lock = threading.RLock()
        self._created = time.time()

    # -- counters ------------------------------------------------------------
    def incr(self, name: str, value: float = 1.0, **labels: str) -> None:
        """Increment a counter. ``labels`` are stored per label-set and also
        exported in the Prometheus exposition."""
        with self._lock:
            self._counters[name] += value
            if labels:
                bucket = self._labels[name]
                key = _label_key(labels)
                bucket[key] += value

    # -- gauges --------------------------------------------------------------
    def set_gauge(self, name: str, value: float, **labels: str) -> None:
        with self._lock:
            if labels:
                self._gauge_labels[name][_label_key(labels)] = value
            else:
                self._gauges[name] = value

    def gauge_add(self, name: str, delta: float, **labels: str) -> float:
        with self._lock:
            if labels:
                key = _label_key(labels)
                self._gauge_labels[name][key] = self._gauge_labels[name].get(key, 0.0) + delta
                return self._gauge_labels[name][key]
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
                "gauge_labels": {n: dict(v) for n, v in self._gauge_labels.items()},
                "histograms": {n: h.summary() for n, h in self._histograms.items()},
                "labels": {n: dict(v) for n, v in self._labels.items()},
            }

    def snapshot_to_file(self, path: str | os.PathLike[str]) -> str:
        """Write the snapshot as JSON — the headless-box escape hatch."""
        target = os.fspath(path)
        tmp = f"{target}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.snapshot(), fh, default=str, ensure_ascii=False)
        os.replace(tmp, target)
        return target

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
            self._gauge_labels.clear()
            self._histograms.clear()
            self._labels.clear()

    def as_prometheus(self, namespace: str = "nomorals") -> str:
        """Prometheus text exposition format, labels included."""
        lines: list[str] = []
        with self._lock:
            for name, value in sorted(self._counters.items()):
                metric = _sanitize(f"{namespace}_{name}")
                lines.append(f"# TYPE {metric} counter")
                lines.append(f"{metric} {value}")
                for label_key, labeled_value in sorted(self._labels.get(name, {}).items()):
                    lines.append(
                        f"{metric}{{{_prom_labels(label_key)}}} {labeled_value}"
                    )
            for name, value in sorted(self._gauges.items()):
                metric = _sanitize(f"{namespace}_{name}")
                lines.append(f"# TYPE {metric} gauge")
                if value or name not in self._gauge_labels:
                    lines.append(f"{metric} {value}")
                for label_key, labeled_value in sorted(self._gauge_labels.get(name, {}).items()):
                    lines.append(
                        f"{metric}{{{_prom_labels(label_key)}}} {labeled_value}"
                    )
            for name in sorted(set(self._gauge_labels) - set(self._gauges)):
                metric = _sanitize(f"{namespace}_{name}")
                lines.append(f"# TYPE {metric} gauge")
                for label_key, labeled_value in sorted(self._gauge_labels[name].items()):
                    lines.append(
                        f"{metric}{{{_prom_labels(label_key)}}} {labeled_value}"
                    )
            for name, hist in sorted(self._histograms.items()):
                metric = _sanitize(f"{namespace}_{name}_seconds")
                lines.append(f"# TYPE {metric} histogram")
                lines.extend(hist.prometheus(metric))
        return "\n".join(lines) + "\n"


def _prom_labels(label_key: str) -> str:
    parts = []
    for pair in label_key.split(","):
        if "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        v = v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        parts.append(f'{k.strip()}="{v}"')
    return ",".join(parts)


class MetricsExporter:
    """Background thread that keeps metrics *somewhere useful*.

    Writes a JSON snapshot to ``path`` every ``interval`` seconds (atomic
    replace, so readers never see a torn file) and, if ``pushgateway_url``
    is set, PUTs the Prometheus exposition to the Pushgateway. Failures are
    counted on the registry itself and never raise — a broken telemetry
    backend must not take the agent down.
    """

    def __init__(
        self,
        metrics: Metrics,
        path: str | os.PathLike[str],
        *,
        interval: float = 60.0,
        pushgateway_url: str = "",
        job: str = "nomorals",
    ) -> None:
        self.metrics = metrics
        self.path = os.fspath(path)
        self.interval = max(1.0, interval)
        self.pushgateway_url = pushgateway_url.rstrip("/")
        self.job = job
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._failures = 0

    def start(self) -> "MetricsExporter":
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="nm-metrics-export", daemon=True
        )
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.flush()
            except Exception:  # noqa: BLE001 - telemetry must never crash
                self._failures += 1
                self.metrics.incr("metrics_export_failures")

    def flush(self) -> None:
        """Export once, now."""
        self.metrics.snapshot_to_file(self.path)
        if self.pushgateway_url:
            body = self.metrics.as_prometheus().encode("utf-8")
            url = f"{self.pushgateway_url}/metrics/job/{self.job}"
            req = urllib.request.Request(url, data=body, method="PUT")
            req.add_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status >= 400:
                        raise OSError(f"pushgateway status {resp.status}")
            except Exception:
                self._failures += 1
                self.metrics.incr("metrics_export_failures")
                raise

    @property
    def failures(self) -> int:
        return self._failures


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

#: Span attribute keys that are silently dropped — a second line of defense
#: behind the value redaction in :func:`_redact_attr`.
_DROPPED_ATTR_KEYS = frozenset(
    {"password", "passwd", "secret", "token", "api_key", "authorization", "cookie"}
)


@dataclass
class SpanEvent:
    """A timestamped annotation inside a span (e.g. a retry, a cache miss)."""

    name: str
    ts: float = field(default_factory=time.perf_counter)
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ts": round(self.ts, 6), "attributes": self.attributes}


@dataclass
class Span:
    """A unit of work with timing and parent linkage."""

    name: str
    trace_id: str
    span_id: str
    parent_id: str = ""
    kind: str = "internal"  # server | client | internal | producer | consumer
    start: float = 0.0
    end: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)
    status: str = "ok"  # ok | error
    error: str = ""
    children: list["Span"] = field(default_factory=list)
    sampled: bool = True

    @property
    def duration(self) -> float:
        return (self.end or time.perf_counter()) - self.start

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute; secret keys are dropped, secret values redacted."""
        if key.lower() in _DROPPED_ATTR_KEYS:
            return
        self.attributes[key] = _redact_attr(value)

    def add_event(self, name: str, **attributes: Any) -> None:
        self.events.append(
            SpanEvent(name=name, attributes={k: _redact_attr(v) for k, v in attributes.items()})
        )

    def record_exception(self, exc: BaseException) -> None:
        """Surface the full exception in the span, the way a flamegraph UI
        expects: an ``exception`` event plus an error status."""
        import traceback

        self.add_event(
            "exception",
            **{
                "exception.type": type(exc).__name__,
                "exception.message": _redact_attr(str(exc)),
                "exception.stacktrace": "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
            },
        )
        self.finish(status="error", error=f"{type(exc).__name__}: {exc}")

    def finish(self, status: str = "ok", error: str = "") -> None:
        self.end = time.perf_counter()
        self.status = status
        self.error = _redact_attr(error)

    def traceparent(self) -> str:
        """W3C traceparent header — propagate across subprocess/network hops."""
        flags = "01" if self.sampled else "00"
        return f"00-{self.trace_id}-{self.span_id}-{flags}"

    def inject(self, headers: dict[str, str] | None = None) -> dict[str, str]:
        """Inject W3C trace context into outbound headers (the OTel
        ``inject`` half — pair with :meth:`Tracer.span(traceparent=...)` on
        the receiving side). Mutates and returns ``headers``."""
        headers = {} if headers is None else headers
        headers["traceparent"] = self.traceparent()
        return headers

    def to_dict(self, *, nested: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "kind": self.kind,
            "duration_ms": round(self.duration * 1000, 3),
            "status": self.status,
            "attributes": self.attributes,
        }
        if self.events:
            payload["events"] = [e.to_dict() for e in self.events]
        if self.error:
            payload["error"] = self.error
        if nested and self.children:
            payload["children"] = [c.to_dict(nested=True) for c in self.children]
        return payload

    def to_otlp(self) -> dict[str, Any]:
        """OTLP-shaped span dict — compatible with real OTel backends later."""
        return {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "parentSpanId": self.parent_id,
            "name": self.name,
            "kind": self.kind.upper(),
            "startTimeUnixNano": int(self.start * 1e9),
            "endTimeUnixNano": int((self.end or time.perf_counter()) * 1e9),
            "attributes": [
                {"key": k, "value": {"stringValue": str(v)}}
                for k, v in self.attributes.items()
            ],
            "events": [
                {
                    "name": e.name,
                    "timeUnixNano": int(e.ts * 1e9),
                    "attributes": [
                        {"key": k, "value": {"stringValue": str(v)}}
                        for k, v in e.attributes.items()
                    ],
                }
                for e in self.events
            ],
            "status": {"code": "ERROR" if self.status == "error" else "OK",
                       "message": self.error},
        }


def _new_trace_id() -> str:
    return secrets.token_hex(16)


def _new_span_id() -> str:
    return secrets.token_hex(8)


def _parse_traceparent(header: str) -> tuple[str, str] | None:
    try:
        parts = header.strip().split("-")
        if len(parts) != 4 or parts[0] != "00":
            return None
        trace_id, span_id = parts[1], parts[2]
        if len(trace_id) != 32 or len(span_id) != 16:
            return None
        int(trace_id, 16)
        int(span_id, 16)
        return trace_id, span_id
    except (ValueError, AttributeError):
        return None


class Tracer:
    """Context-propagating tracer with sampling and export.

    Spans nest automatically through a :mod:`contextvars` token, which means the
    parent linkage is correct across ``await`` points and thread pool submissions
    that copy the context.

    Sampling: ``sample_rate`` applies head sampling deterministically on the
    trace id; spans whose trace is *not* sampled are still created (so
    ``current()`` works) but are not retained. Error spans are **always**
    retained — the tail rule.
    """

    def __init__(self, *, max_traces: int = 512, sample_rate: float = 1.0) -> None:
        self._token: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
            "nm_current_span", default=None
        )
        self._traces: Deque[Span] = deque(maxlen=max_traces)
        self._lock = threading.Lock()
        self.sample_rate = min(1.0, max(0.0, sample_rate))
        self.dropped_spans = 0

    def current(self) -> Span | None:
        return self._token.get()

    def current_traceparent(self) -> str:
        """The active span's ``traceparent`` (``""`` when no span is active).

        The log-correlation idiom: attach this to every log record / outbound
        request so traces and logs join without threading span objects
        through the call chain (OTel's log-correlation pattern).
        """
        span = self._token.get()
        return span.traceparent() if span is not None else ""

    def _sampled(self, trace_id: str) -> bool:
        if self.sample_rate >= 1.0:
            return True
        draw = int(trace_id[:8], 16) / 0xFFFFFFFF
        return draw < self.sample_rate

    @contextmanager
    def span(
        self,
        name: str,
        *,
        kind: str = "internal",
        traceparent: str = "",
        **attributes: Any,
    ) -> Iterator[Span]:
        parent = self._token.get()
        if traceparent:
            parsed = _parse_traceparent(traceparent)
            trace_id = parsed[0] if parsed else _new_trace_id()
            parent_id = parsed[1] if parsed else ""
        elif parent is not None:
            trace_id, parent_id = parent.trace_id, parent.span_id
        else:
            trace_id, parent_id = _new_trace_id(), ""
        span = Span(
            name=name,
            trace_id=trace_id,
            span_id=_new_span_id(),
            parent_id=parent_id,
            kind=kind,
            start=time.perf_counter(),
            attributes={
                k: _redact_attr(v)
                for k, v in attributes.items()
                if k and k.lower() not in _DROPPED_ATTR_KEYS
            },
            sampled=self._sampled(trace_id),
        )
        token = self._token.set(span)
        if parent is not None:
            parent.children.append(span)
        try:
            yield span
        except Exception as exc:  # noqa: BLE001 - record then re-raise
            span.record_exception(exc)
            raise
        else:
            if span.status != "error":
                span.finish()
        finally:
            if parent is None:
                self._retain(span)
            self._token.reset(token)

    def _retain(self, span: Span) -> None:
        if span.status == "error" or span.sampled:
            with self._lock:
                self._traces.append(span)
        else:
            self.dropped_spans += 1

    def extract_traceparent(self, header: str) -> "contextmanager":  # type: ignore[valid-type]
        """Start a root span continuing the trace from a ``traceparent`` header."""
        return self.span("__continued__", traceparent=header)

    def traces(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._traces)[-limit:]
        return [s.to_dict() for s in items]

    def drain_to_jsonl(self, path: str | os.PathLike[str], *, clear: bool = True) -> int:
        """Append retained root spans to ``path`` as JSON lines. Returns the
        count written. The headless-box equivalent of a trace backend."""
        with self._lock:
            items = list(self._traces)
            if clear:
                self._traces.clear()
        count = 0
        if items:
            with open(path, "a", encoding="utf-8") as fh:
                for span in items:
                    fh.write(json.dumps(span.to_dict(), default=str, ensure_ascii=False))
                    fh.write("\n")
                    count += 1
        return count

    def clear(self) -> None:
        with self._lock:
            self._traces.clear()


# ── Health ─────────────────────────────────────────────────────────────────────


@dataclass
class HealthCheck:
    """One probe: a callable returning ``{"ok": bool, "detail": str}``.

    ``kind`` is ``"liveness"`` (process-only: is *this process* responsive —
    never touches external dependencies) or ``"readiness"`` (can we serve —
    dependency probes live here). ``ttl`` caches the result so a scraping
    operator doesn't hammer an expensive check.
    """

    name: str
    fn: Callable[[], dict[str, Any]]
    kind: str = "readiness"
    timeout: float = 5.0
    ttl: float = 0.0
    critical: bool = True
    _last_ok: bool = True
    _last_detail: str = ""
    _last_run: float = 0.0
    _consecutive_failures: int = 0
    _transitions: int = 0

    def run(self) -> dict[str, Any]:
        now = time.monotonic()
        if self.ttl > 0 and (now - self._last_run) < self.ttl and self._last_run:
            return self._report(cached=True)
        with cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="nm-health") as pool:
            future = pool.submit(self._guarded)
            try:
                ok, detail = future.result(timeout=self.timeout)
            except cf.TimeoutError:
                ok, detail = False, f"timed out after {self.timeout}s"
        self._last_run = now
        if ok != self._last_ok:
            self._transitions += 1
            self._last_ok = ok
        self._consecutive_failures = 0 if ok else self._consecutive_failures + 1
        self._last_detail = detail
        return self._report(cached=False)

    def _guarded(self) -> tuple[bool, str]:
        try:
            result = self.fn()
        except Exception as exc:  # noqa: BLE001 - a check never raises
            return False, f"{type(exc).__name__}: {exc}"
        if isinstance(result, dict):
            ok = bool(result.get("ok", False))
            return ok, str(result.get("detail", ""))
        return bool(result), ""

    def _report(self, *, cached: bool) -> dict[str, Any]:
        return {
            "ok": self._last_ok,
            "detail": self._last_detail,
            "kind": self.kind,
            "critical": self.critical,
            "cached": cached,
            "consecutive_failures": self._consecutive_failures,
            "transitions": self._transitions,
        }


@dataclass
class HealthReport:
    """Aggregated snapshot used by ``nm health`` and ``GET /health``.

    ``healthy`` is True only when every *critical* check passes. Per-component
    detail is machine-readable so an operator (human or AI) can identify the
    failing component mechanically. Never includes secrets, versions of
    dependencies, or stack traces.
    """

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


class HealthChecker:
    """The check engine behind :class:`HealthReport`.

    Usage::

        checker = HealthChecker(version="2.0")
        checker.register_default_checks(memory_limit_mb=1500)
        checker.register("telegram", probe_telegram, kind="readiness", ttl=30)
        report = checker.readiness()   # or checker.run() for everything
    """

    def __init__(self, *, version: str = "dev") -> None:
        self.version = version
        self._checks: dict[str, HealthCheck] = {}
        self._lock = threading.Lock()
        self._started = time.monotonic()

    def register(
        self,
        name: str,
        fn: Callable[[], dict[str, Any]],
        *,
        kind: str = "readiness",
        timeout: float = 5.0,
        ttl: float = 0.0,
        critical: bool = True,
    ) -> HealthCheck:
        """Register a probe. ``fn`` returns ``{"ok": bool, "detail": str}``
        and must be cheap, bounded, and side-effect free."""
        if kind not in ("liveness", "readiness"):
            raise ValueError(f"check kind must be 'liveness' or 'readiness', got {kind!r}")
        check = HealthCheck(name=name, fn=fn, kind=kind, timeout=timeout,
                            ttl=ttl, critical=critical)
        with self._lock:
            self._checks[name] = check
        return check

    def unregister(self, name: str) -> bool:
        with self._lock:
            return self._checks.pop(name, None) is not None

    def register_default_checks(self, *, memory_limit_mb: float = 0.0) -> None:
        """Process-local liveness probes: event loop responsive (no deadlock),
        RSS memory sane, thread count sane. Never touch the network."""
        self.register("uptime", lambda: {"ok": True, "detail": "process alive"},
                      kind="liveness", ttl=5.0, critical=True)

        def _memory() -> dict[str, Any]:
            rss_mb = self._rss_mb()
            detail = f"rss {rss_mb:.0f}MB" if rss_mb else "rss unknown"
            ok = not memory_limit_mb or rss_mb <= memory_limit_mb
            if not ok:
                detail += f" over limit {memory_limit_mb:.0f}MB"
            return {"ok": ok, "detail": detail}

        self.register("memory", _memory, kind="liveness", ttl=10.0, critical=False)

        def _threads() -> dict[str, Any]:
            count = threading.active_count()
            return {"ok": count < 500, "detail": f"{count} threads"}

        self.register("threads", _threads, kind="liveness", ttl=10.0, critical=False)

    @staticmethod
    def _rss_mb() -> float:
        try:
            import resource

            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:  # pragma: no cover - non-POSIX
            return 0.0

    def run(self, *, kinds: tuple[str, ...] = ("liveness", "readiness")) -> HealthReport:
        with self._lock:
            checks = [c for c in self._checks.values() if c.kind in kinds]
        results = {c.name: c.run() for c in checks}
        critical_failed = any(
            not r["ok"] and r["critical"] for r in results.values()
        )
        return HealthReport(
            healthy=not critical_failed,
            version=self.version,
            uptime=time.monotonic() - self._started,
            checks=results,
        )

    def liveness(self) -> HealthReport:
        """Process-only: safe to wire to a restart decision."""
        return self.run(kinds=("liveness",))

    def readiness(self) -> HealthReport:
        """Includes dependency probes: safe to wire to traffic routing."""
        return self.run(kinds=("readiness",))


#: Process-wide metric registry.
global_metrics = Metrics()
#: Process-wide tracer.
global_tracer = Tracer()


def trace(name: str, **attributes: Any):
    """Shorthand for ``global_tracer.span(name, **attributes)``."""
    return global_tracer.span(name, **attributes)
