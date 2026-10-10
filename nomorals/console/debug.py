"""Debug telemetry hub for the console debug view.

:class:`DebugHub` is a process-wide, thread-safe collector that feeds the
``debug`` watch-mode view (key ``d``):

- **Recent log lines** — a ``logging.Handler`` installed on the root logger
  keeps the last 200 records in a ring buffer, with per-level counters.
- **Slow operations** — extracted from log lines matching
  ``<name> ... took <n>ns`` (e.g. ``telegram: slow get_entity(...) took 1.2s``)
  plus anything recorded via :meth:`DebugHub.timed`.
- **LLM call traces** — recorded through the router's learning hook via
  :meth:`DebugHub.install_llm_hook` (chain-safe: the existing hook keeps
  working).

Everything is best-effort and never raises: the handler and hooks must not
break the code they observe. Install with :meth:`DebugHub.install`
(idempotent); the watch screen installs it automatically on entry.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Any, Iterator

__all__ = ["DebugHub"]

#: Keep the last N log records / LLM calls.
_LOG_CAPACITY = 200
_LLM_CAPACITY = 40
#: Keep the K slowest operations.
_SLOW_KEEP = 10
#: Keep the K most recent exceptions (with tracebacks).
_EXC_CAPACITY = 20
#: Log-derived slow ops below this are noise (the timed() context records all).
_SLOW_MIN_S = 0.5

#: Matches "... <name> ... took 1.23s" — used to mine slow ops from log text.
_SLOW_RE = re.compile(r"\btook\s+(\d+(?:\.\d+)?)s\b")


class _DebugHandler(logging.Handler):
    """Root-logger handler that feeds DebugHub. Never raises, never logs."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            DebugHub._ingest(record)
        except Exception:  # noqa: BLE001 - telemetry must never break logging
            pass


class DebugHub:
    """Process-wide debug telemetry. All methods are class-level."""

    _lock = threading.Lock()
    _handler: _DebugHandler | None = None
    _records: deque[tuple[float, str, str, str]] = deque(maxlen=_LOG_CAPACITY)
    _level_counts: dict[str, int] = {}
    _logger_counts: dict[str, int] = {}
    _llm: deque[dict[str, Any]] = deque(maxlen=_LLM_CAPACITY)
    _slow: list[dict[str, Any]] = []  # top-K slowest, sorted desc
    _exceptions: deque[dict[str, Any]] = deque(maxlen=_EXC_CAPACITY)
    _started_ts = time.time()

    # ── lifecycle ──

    @classmethod
    def install(cls) -> None:
        """Attach the capture handler to the root logger (idempotent)."""
        with cls._lock:
            if cls._handler is not None:
                return
            handler = _DebugHandler()
            handler.setLevel(logging.DEBUG)
            # Marked so the watch-mode output guard (which mutes every other
            # logging handler to keep the dashboard clean) never mutes this
            # one — telemetry keeps flowing while the screen owns the tty.
            handler._devon_debug_capture = True  # noqa: SLF001
            logging.getLogger().addHandler(handler)
            cls._handler = handler

    @classmethod
    def uninstall(cls) -> None:
        """Detach the capture handler (idempotent). Keeps buffered data."""
        with cls._lock:
            if cls._handler is None:
                return
            try:
                logging.getLogger().removeHandler(cls._handler)
            except Exception:  # noqa: BLE001 - teardown is best-effort
                pass
            cls._handler = None

    @classmethod
    def reset(cls) -> None:
        """Clear all buffered telemetry. Used by tests."""
        with cls._lock:
            cls._records.clear()
            cls._level_counts.clear()
            cls._logger_counts.clear()
            cls._llm.clear()
            cls._slow.clear()
            cls._exceptions.clear()
            cls._started_ts = time.time()

    @classmethod
    def installed(cls) -> bool:
        with cls._lock:
            return cls._handler is not None

    # ── ingestion ──

    @classmethod
    def _ingest(cls, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            msg = str(record.msg)
        entry = (record.created, record.levelname, record.name, msg)
        with cls._lock:
            cls._records.append(entry)
            cls._level_counts[record.levelname] = (
                cls._level_counts.get(record.levelname, 0) + 1
            )
            short = record.name.split(".")[-1][:24] or "?"
            cls._logger_counts[short] = cls._logger_counts.get(short, 0) + 1
        if record.exc_info and record.exc_info[0] is not None:
            cls._record_exception(record)
        cls._maybe_slow_from_log(msg, record.created)

    @classmethod
    def _record_exception(cls, record: logging.LogRecord) -> None:
        """Capture a formatted traceback (lnav shows tracebacks inline)."""
        try:
            import traceback

            exc_text = "".join(
                traceback.format_exception(*record.exc_info)  # type: ignore[arg-type]
            )
        except Exception:  # noqa: BLE001
            return
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            msg = str(record.msg)
        with cls._lock:
            cls._exceptions.append(
                {
                    "ts": record.created,
                    "level": record.levelname,
                    "logger": record.name,
                    "message": str(msg)[:200],
                    "traceback": exc_text[-4000:],
                }
            )

    @classmethod
    def record_exception(
        cls,
        exc: BaseException,
        *,
        logger_name: str = "",
        message: str = "",
    ) -> None:
        """Record an exception explicitly (for caught-and-logged errors)."""
        try:
            import traceback

            exc_text = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
        except Exception:  # noqa: BLE001
            exc_text = f"{type(exc).__name__}: {exc}"
        with cls._lock:
            cls._exceptions.append(
                {
                    "ts": time.time(),
                    "level": "ERROR",
                    "logger": str(logger_name or "?"),
                    "message": str(message or f"{type(exc).__name__}: {exc}")[:200],
                    "traceback": exc_text[-4000:],
                }
            )

    @classmethod
    def _maybe_slow_from_log(cls, msg: str, ts: float) -> None:
        match = None
        for match in _SLOW_RE.finditer(msg):
            pass
        if match is None:
            return
        try:
            dur = float(match.group(1))
        except (TypeError, ValueError):
            return
        if dur < _SLOW_MIN_S:
            return
        name = msg[: match.start()].strip()
        # Trim trailing punctuation/parens leftovers, keep it short.
        name = re.sub(r"[\s(:\-–—]+$", "", name)[-48:] or "unknown op"
        cls._record_slow(name, dur, ts)

    @classmethod
    def _record_slow(cls, name: str, duration_s: float, ts: float) -> None:
        with cls._lock:
            cls._slow.append(
                {"name": name, "duration_s": duration_s, "ts": ts}
            )
            cls._slow.sort(key=lambda e: e["duration_s"], reverse=True)
            del cls._slow[_SLOW_KEEP:]

    # ── LLM traces ──

    @classmethod
    def record_llm(
        cls,
        *,
        operation: str = "",
        provider_name: str = "",
        success: bool = True,
        latency_s: float = 0.0,
        error: str = "",
    ) -> None:
        """Record one provider attempt. Called from the learning hook."""
        with cls._lock:
            cls._llm.append(
                {
                    "ts": time.time(),
                    "operation": str(operation),
                    "provider": str(provider_name),
                    "success": bool(success),
                    "latency_s": float(latency_s or 0.0),
                    "error": str(error or "")[:120],
                }
            )

    @classmethod
    def install_llm_hook(cls, router: Any) -> bool:
        """Chain DebugHub into the router's learning hook.

        The router only keeps one hook, so this wraps the existing one
        instead of replacing it. Returns True when a hook is installed.
        Never raises.
        """
        try:
            setter = getattr(router, "set_learning", None)
            getter = getattr(router, "learning", None)
            if not callable(setter):
                return False
            existing = getter() if callable(getter) else None

            def chained(**kw: Any) -> None:
                try:
                    cls.record_llm(
                        operation=kw.get("operation", ""),
                        provider_name=kw.get("provider_name", ""),
                        success=kw.get("success", True),
                        latency_s=kw.get("latency_s", 0.0),
                        error=kw.get("error", ""),
                    )
                except Exception:  # noqa: BLE001
                    pass
                if callable(existing):
                    try:
                        existing(**kw)
                    except Exception:  # noqa: BLE001
                        pass

            setter(chained)
            return True
        except Exception:  # noqa: BLE001 - hookup must never break boot
            return False

    # ── explicit op timing ──

    @classmethod
    @contextmanager
    def timed(cls, name: str) -> Iterator[None]:
        """Time a block and record it as a slow-op candidate."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            try:
                cls._record_slow(str(name), time.perf_counter() - t0,
                                 time.time())
            except Exception:  # noqa: BLE001
                pass

    # ── reads (for the debug view) ──

    @classmethod
    def recent(
        cls,
        n: int = 20,
        *,
        level: str | None = None,
        logger: str | None = None,
        pattern: str | None = None,
    ) -> list[tuple[float, str, str, str]]:
        """Last ``n`` log records: (ts, level, logger, message).

        Optional lnav-style filters: ``level`` (exact, e.g. ``"ERROR"``),
        ``logger`` (substring match), ``pattern`` (regex over the message).
        """
        levels = {level.upper()} if level else None
        rx = None
        if pattern:
            try:
                rx = re.compile(pattern, re.IGNORECASE)
            except re.error:
                rx = None
        with cls._lock:
            recs = list(cls._records)
        out: list[tuple[float, str, str, str]] = []
        for ts, lvl, name, msg in reversed(recs):
            if levels and lvl.upper() not in levels:
                continue
            if logger and logger.lower() not in name.lower():
                continue
            if rx and not rx.search(msg):
                continue
            out.append((ts, lvl, name, msg))
            if len(out) >= max(0, n):
                break
        return list(reversed(out))

    @classmethod
    def search(cls, pattern: str, n: int = 50) -> list[tuple[float, str, str, str]]:
        """Regex search over buffered records, newest first (lnav ``/``)."""
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return []
        with cls._lock:
            recs = list(cls._records)
        out = [
            (ts, lvl, name, msg)
            for ts, lvl, name, msg in reversed(recs)
            if rx.search(msg) or rx.search(name)
        ]
        return out[: max(0, n)]

    @classmethod
    def exceptions(cls, n: int = 5) -> list[dict[str, Any]]:
        """Most recent captured tracebacks, newest first."""
        with cls._lock:
            return [dict(e) for e in reversed(list(cls._exceptions))][: max(0, n)]

    @classmethod
    def errors_since(cls, ts: float) -> int:
        """Count of ERROR/CRITICAL records at/after ``ts`` (jump-to-error)."""
        with cls._lock:
            return sum(
                1
                for rts, lvl, _n, _m in cls._records
                if rts >= ts and lvl in ("ERROR", "CRITICAL")
            )

    @classmethod
    def log_rate(cls, window_s: float = 300.0) -> float:
        """Messages per second over the trailing ``window_s`` (lnav rate)."""
        cutoff = time.time() - max(1.0, window_s)
        with cls._lock:
            n = sum(1 for rts, _l, _n, _m in cls._records if rts >= cutoff)
        return n / max(1.0, window_s)

    @classmethod
    def histogram(
        cls, n: int = 24, bucket_s: float = 60.0
    ) -> dict[str, Any]:
        """Per-bucket log volume, newest bucket last (lnav histogram).

        Returns ``{"buckets": [start_ts, …], "series": {level: [counts]}}``
        for the levels ERROR, WARNING, INFO, DEBUG.
        """
        now = time.time()
        bucket_s = max(1.0, bucket_s)
        n = max(1, min(120, n))
        starts = [now - bucket_s * (n - i) for i in range(n)]
        levels = ("ERROR", "WARNING", "INFO", "DEBUG")
        series: dict[str, list[int]] = {lv: [0] * n for lv in levels}
        with cls._lock:
            recs = list(cls._records)
        lo = starts[0]
        for rts, lvl, _name, _msg in recs:
            if rts < lo:
                continue
            idx = min(n - 1, int((rts - lo) // bucket_s))
            key = "ERROR" if lvl == "CRITICAL" else lvl
            if key in series:
                series[key][idx] += 1
        return {"buckets": starts, "series": series, "bucket_s": bucket_s}

    @classmethod
    def level_counts(cls) -> dict[str, int]:
        with cls._lock:
            return dict(cls._level_counts)

    @classmethod
    def top_loggers(cls, n: int = 6) -> list[tuple[str, int]]:
        """Noisiest loggers, most records first."""
        with cls._lock:
            items = sorted(cls._logger_counts.items(),
                           key=lambda kv: kv[1], reverse=True)
            return items[:max(0, n)]

    @classmethod
    def llm_calls(cls, n: int = 10) -> list[dict[str, Any]]:
        with cls._lock:
            return list(cls._llm)[-max(0, n):]

    @classmethod
    def slow_ops(cls) -> list[dict[str, Any]]:
        """Top slowest operations, slowest first."""
        with cls._lock:
            return [dict(e) for e in cls._slow]

    @classmethod
    def stats(cls) -> dict[str, Any]:
        """Summary for the debug view header."""
        with cls._lock:
            counts = dict(cls._level_counts)
            snap = {
                "captured": len(cls._records),
                "errors": counts.get("ERROR", 0) + counts.get("CRITICAL", 0),
                "warnings": counts.get("WARNING", 0),
                "llm_calls": len(cls._llm),
                "slow_ops": len(cls._slow),
                "exceptions": len(cls._exceptions),
                "uptime_s": time.time() - cls._started_ts,
            }
        # Outside the lock: log_rate() takes it itself (Lock isn't reentrant).
        snap["log_rate_s"] = cls.log_rate()
        return snap
