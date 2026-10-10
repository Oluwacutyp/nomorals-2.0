"""Structured logging with secret redaction.

One call — :func:`setup_logging` — configures the root logger for the whole
process: console handler, rotating file handler, optional JSON lines, optional
non-blocking queue delivery, and a redaction filter that keeps tokens out of
the log file and out of any log that gets shipped to a backup repository.

Correlation travels on :mod:`contextvars`, not through function signatures —
the structlog gold: bind ``trace_id`` / ``mission_id`` / ``task_id`` / ``agent``
once at the entry point (:func:`bind_log_context` or the :func:`log_context`
context manager) and every record emitted during that scope carries them,
across threads and ``await`` points. Clear the context at job boundaries —
stale IDs on a reused worker are the classic correlation bug.

On the box, logs are the only window in: ``as_json=True`` + JSONL file output
is the production wire format.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import logging.handlers
import os
import queue as _queue
import random
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

__all__ = [
    "LogCapture",
    "RedactionFilter",
    "SampleFilter",
    "audit_log",
    "bind_log_context",
    "child_logger",
    "clear_log_context",
    "get_logger",
    "json_formatter",
    "log_context",
    "redact",
    "scrub_secrets",
    "setup_logging",
    "shutdown_logging",
]

_CONFIGURED = False
_LOCK = threading.Lock()
#: Queue listeners started by ``setup_logging(queue=True)`` — kept alive for
#: the process lifetime and stopped by :func:`shutdown_logging`.
_LISTENERS: list[logging.handlers.QueueListener] = []

# ── Correlation context ──────────────────────────────────────────────────────
#: Keys the logging pipeline knows how to inject into every record.
_CONTEXT_KEYS = ("trace_id", "span_id", "mission_id", "task_id", "agent", "request_id")
_log_context: contextvars.ContextVar[dict[str, str]] = contextvars.ContextVar(
    "nm_log_context", default={}
)


def bind_log_context(**fields: str) -> contextvars.Token[dict[str, str]]:
    """Bind correlation fields (trace_id, mission_id, task_id, agent, ...) for
    the current context. Returns the token so callers can reset manually."""
    current = dict(_log_context.get())
    current.update({k: str(v) for k, v in fields.items() if v is not None})
    return _log_context.set(current)


def clear_log_context() -> None:
    """Drop all bound correlation fields. Call at job/request boundaries."""
    _log_context.set({})


def log_context(**fields: str) -> "_LogContext":
    """Context manager: bind correlation fields for the block, clear on exit.

    >>> with log_context(trace_id="trc_123", mission="morning-pulse"):
    ...     log.info("waking up")   # carries trace_id + mission on every record
    """
    return _LogContext(fields)


class _LogContext:
    def __init__(self, fields: dict[str, str]) -> None:
        self._fields = {k: str(v) for k, v in fields.items() if v is not None}
        self._token: contextvars.Token[dict[str, str]] | None = None
        self._previous: dict[str, str] | None = None

    def __enter__(self) -> "_LogContext":
        self._previous = dict(_log_context.get())
        merged = dict(self._previous)
        merged.update(self._fields)
        self._token = _log_context.set(merged)
        return self

    def __exit__(self, *exc: object) -> None:
        _log_context.set(self._previous or {})


class _ContextInjectionFilter(logging.Filter):
    """Stamp every record with the bound correlation context."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            for key, value in _log_context.get().items():
                if not hasattr(record, key):
                    setattr(record, key, value)
        except Exception:  # noqa: E103 - logging must never raise (pragma: no cover)
            pass
        return True


#: Structured-event keys whose *values* are always secrets, whatever the
#: surrounding text looks like. Complements the regex pass, which handles
#: secrets embedded in free text.
_SENSITIVE_KEYS = frozenset(
    {
        "password", "passwd", "secret", "token", "api_key", "apikey",
        "access_token", "refresh_token", "client_secret", "app_password",
        "private_key", "authorization", "cookie", "set_cookie", "session",
        "session_id", "sessionid", "card_number", "cardnumber", "cvv",
        "ssn", "national_id", "iban", "pin", "otp", "totp",
    }
)

#: Keys whose values are safe to *fingerprint* instead of fully dropping —
#: the hash lets an operator correlate events (same session hit two errors)
#: without ever writing the secret.
_FINGERPRINT_KEYS = frozenset({"session_id", "sessionid", "email", "refresh_token"})


def _fingerprint(value: Any) -> str:
    try:
        digest = hashlib.sha256(str(value).encode("utf-8", "replace")).hexdigest()
    except Exception:  # pragma: no cover - defensive
        return "[REDACTED]"
    return f"sha256:{digest[:16]}"


def _scrub_mapping(payload: dict[Any, Any]) -> dict[Any, Any]:
    """Recursively scrub a structured payload by key name."""
    out: dict[Any, Any] = {}
    for key, value in payload.items():
        low = str(key).lower()
        if low in _SENSITIVE_KEYS:
            out[key] = "[REDACTED]"
        elif low in _FINGERPRINT_KEYS:
            out[key] = _fingerprint(value)
        elif isinstance(value, dict):
            out[key] = _scrub_mapping(value)
        elif isinstance(value, (list, tuple)):
            out[key] = type(value)(
                _scrub_mapping(v) if isinstance(v, dict) else v for v in value
            )
        elif isinstance(value, str):
            out[key] = redact(value)
        else:
            out[key] = value
    return out


#: Patterns that reliably identify secrets in free text.
# Order matters. Header-shaped secrets are scrubbed to end-of-line FIRST, because a
# generic ``key=value`` rule would otherwise match ``Authorization: Bearer`` and
# redact the word "Bearer" while leaving the actual token in the log.
_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Header / assignment whose value is a whole credential: take the rest of the line.
    (
        re.compile(r"(?im)^([ \t]*(?:authorization|proxy-authorization|cookie|set-cookie)\s*[:=]).*$"),
        r"\1 [REDACTED]",
    ),
    (
        re.compile(
            r"(?i)\b(authorization|proxy-authorization|cookie|set-cookie)\b\s*[:=]\s*.*"
        ),
        r"\1=[REDACTED]",
    ),
    # Bearer / basic tokens anywhere in a line.
    (re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._\-+/=]{8,}"), "[REDACTED_AUTH]"),
    # Provider key shapes.
    (re.compile(r"\b(sk|hf|pk|ghp|gho|ghs|github_pat)_[A-Za-z0-9_\-]{12,}\b"), "[REDACTED_KEY]"),
    (re.compile(r"\b(xox[baprs]-[A-Za-z0-9\-]{10,})\b"), "[REDACTED_SLACK]"),
    (re.compile(r"\b(AKIA[0-9A-Z]{16})\b"), "[REDACTED_AWS]"),
    (
        re.compile(r"\b(ey[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{5,})\b"),
        "[REDACTED_JWT]",
    ),
    # Generic key=value credentials.
    (
        re.compile(
            r"(?i)\b(api[_-]?key|apikey|access[_-]?token|refresh[_-]?token|token|secret|"
            r"password|passwd|app[_-]?password|client[_-]?secret)\b\s*[:=]\s*(\S{6,})"
        ),
        r"\1=[REDACTED]",
    ),
)


def redact(text: str) -> str:
    """Scrub secrets from an arbitrary string."""
    if not text:
        return text
    out = text
    for pattern, replacement in _SECRET_PATTERNS:
        out = pattern.sub(replacement, out)
    return out


def scrub_secrets(text: str) -> str:
    """Outbound guard: remove secret-shaped values from text leaving the system.

    Uses the same patterns as the log redactor (:data:`_SECRET_PATTERNS`).
    Wire into outbound chat paths (gateway.send) — never into tool inputs
    or stored memory, which may legitimately carry credentials for API use.
    """
    return redact(text)


class RedactionFilter(logging.Filter):
    """Logging filter that redacts the message, known-sensitive args, and any
    structured ``extra=`` payload the record carries.

    Never raises: a broken redactor must not break logging.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if isinstance(record.msg, str):
                record.msg = redact(record.msg)
            if record.args:
                if isinstance(record.args, dict):
                    record.args = _scrub_mapping(record.args)
                elif isinstance(record.args, tuple):
                    record.args = tuple(
                        redact(a) if isinstance(a, str)
                        else _scrub_mapping(a) if isinstance(a, dict)
                        else a
                        for a in record.args
                    )
            payload = getattr(record, "payload", None)
            if isinstance(payload, dict):
                record.payload = _scrub_mapping(payload)
        except Exception:  # noqa: E103 - logging must never raise (pragma: no cover)
            pass
        return True


class SampleFilter(logging.Filter):
    """Deliberate sampling for high-volume log paths.

    ``rates`` maps level name (or ``"default"``) to a keep-rate in [0, 1].
    Errors and criticals are *always* kept — the rare event is the one worth
    having. Sampling is deterministic per (trace_id, message): the same trace
    is sampled the same way across handlers and restarts.
    """

    def __init__(self, rates: dict[str, float] | None = None) -> None:
        super().__init__()
        self.rates = {"default": 1.0, **(rates or {})}

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR:
            return True
        rate = self.rates.get(record.levelname, self.rates.get("default", 1.0))
        if rate >= 1.0:
            return True
        if rate <= 0.0:
            return False
        key = f"{getattr(record, 'trace_id', '')}:{record.getMessage()}".encode(
            "utf-8", "replace"
        )
        draw = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
        if draw >= rate:
            record.msg = f"[sampled {rate:.0%}] {record.msg}"
            return False
        return True


class _ConsoleFormatter(logging.Formatter):
    """Compact, readable console output with colour when the tty supports it."""

    # Palette note: the owner hates red — errors are magenta, never red,
    # and nothing uses a black background.
    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[35m",
        "CRITICAL": "\033[1;35m",
    }
    RESET = "\033[0m"

    def __init__(self, *, color: bool = True, verbose: bool = False) -> None:
        super().__init__()
        self.color = color
        self.verbose = verbose

    def format(self, record: logging.LogRecord) -> str:
        _ContextInjectionFilter().filter(record)
        level = record.levelname
        name = record.name
        if name.startswith("nomorals."):
            name = name[len("nomorals."):]
        stamp = self.formatTime(record, "%H:%M:%S")
        message = record.getMessage()
        ctx = _context_suffix(record)
        if self.color and sys.stderr.isatty():
            paint = self.COLORS.get(level, "")
            head = f"{paint}{stamp} {level[:4]:<4}{self.RESET} {name}"
        else:
            head = f"{stamp} {level[:4]:<4} {name}"
        body = f"{head}{ctx} {message}" if not self.verbose else f"{head}{ctx} {message} [{record.filename}:{record.lineno}]"
        if record.exc_info:
            body = f"{body}\n{self.formatException(record.exc_info)}"
        return body


def _context_suffix(record: logging.LogRecord) -> str:
    parts = []
    for key in ("trace_id", "mission_id", "task_id", "agent"):
        value = getattr(record, key, None)
        if value:
            parts.append(f"{key}={value}")
    return f" [{' '.join(parts)}]" if parts else ""


#: LogRecord attributes that are *not* user extras.
_DEFAULT_ATTRS = frozenset(
    {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "taskName", "message", "asctime",
        *(_CONTEXT_KEYS), "audit",
    }
)


class JsonFormatter(logging.Formatter):
    """One JSON object per line — for shipping logs somewhere that parses them.

    Carries the correlation context (trace_id, span_id, mission_id, task_id,
    agent), any ``extra=`` fields the call site attached, and the audit marker.
    """

    def format(self, record: logging.LogRecord) -> str:
        _ContextInjectionFilter().filter(record)
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        context: dict[str, Any] = {}
        for key in _CONTEXT_KEYS:
            value = getattr(record, key, None)
            if value is not None:
                context[key] = value
        if context:
            payload["context"] = context
        if getattr(record, "audit", False):
            payload["audit"] = True
        for key, value in vars(record).items():
            if key not in _DEFAULT_ATTRS and not key.startswith("_"):
                try:
                    json.dumps(value, default=str)
                    payload[key] = value
                except Exception:  # pragma: no cover - defensive
                    payload[key] = str(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def json_formatter() -> JsonFormatter:
    return JsonFormatter()


def setup_logging(
    level: str = "INFO",
    *,
    file: str | os.PathLike[str] | None = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    as_json: bool = False,
    redact_secrets: bool = True,
    color: bool = True,
    force: bool = False,
    queue: bool = False,
    queue_size: int = 10_000,
    sample_rates: dict[str, float] | None = None,
    bind: dict[str, str] | None = None,
    quiet_libs: Iterable[str] = (
        "urllib3",
        "asyncio",
        "filelock",
        # Third-party protocol libraries: their DEBUG output (full Discord
        # gateway event dicts, Telethon MTProto packet traces) drowns the
        # application log. Devon's own adapters log under ``nomorals.*`` and
        # are unaffected.
        "discord",
        "telethon",
    ),
) -> logging.Logger:
    """Configure root logging for the process. Idempotent unless ``force``.

    ``queue=True`` routes every record through a :class:`QueueHandler` + a
    dedicated listener thread — the hot path never blocks on formatting or
    disk I/O. ``sample_rates`` (e.g. ``{"DEBUG": 0.1, "INFO": 0.5}``) drops a
    deterministic fraction of verbose records; ERROR+ always survive.
    ``bind`` pre-binds correlation fields for the whole process.
    """
    global _CONFIGURED
    with _LOCK:
        root = logging.getLogger()
        if _CONFIGURED and not force:
            root.setLevel(_parse_level(level))
            return root

        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()

        numeric = _parse_level(level)
        root.setLevel(numeric)

        console = logging.StreamHandler(stream=sys.stderr)
        console.setLevel(numeric)
        console.setFormatter(JsonFormatter() if as_json else _ConsoleFormatter(color=color))
        handlers: list[logging.Handler] = [console]

        if file:
            path = Path(os.path.expanduser(str(file)))
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                rotating = logging.handlers.RotatingFileHandler(
                    path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
                )
                rotating.setLevel(numeric)
                rotating.setFormatter(
                    JsonFormatter()
                    if as_json
                    else logging.Formatter(
                        "%(asctime)s %(levelname)-8s %(name)s %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%S",
                    )
                )
                handlers.append(rotating)
            except OSError as exc:  # pragma: no cover - read-only fs
                root.warning("could not open log file %s: %s", path, exc)

        for handler in handlers:
            if redact_secrets:
                handler.addFilter(RedactionFilter())
            handler.addFilter(_ContextInjectionFilter())
            if sample_rates:
                handler.addFilter(SampleFilter(sample_rates))
            root.addHandler(handler)

        if queue:
            # Re-route: the root keeps a single QueueHandler; a listener thread
            # drains into the real handlers off the hot path.
            for handler in handlers:
                root.removeHandler(handler)
            log_queue: _queue.Queue[logging.LogRecord] = _queue.Queue(maxsize=queue_size)
            queue_handler = logging.handlers.QueueHandler(log_queue)
            queue_handler.addFilter(_ContextInjectionFilter())
            root.addHandler(queue_handler)
            listener = logging.handlers.QueueListener(log_queue, *handlers)
            listener.start()
            _LISTENERS.append(listener)

        for lib in quiet_libs:
            logging.getLogger(lib).setLevel(max(numeric, logging.WARNING))

        if bind:
            bind_log_context(**bind)

        logging.captureWarnings(True)
        _CONFIGURED = True
        return root


def shutdown_logging() -> None:
    """Stop queue listeners started by ``setup_logging(queue=True)``."""
    global _CONFIGURED
    with _LOCK:
        for listener in _LISTENERS:
            try:
                listener.stop()
            except Exception:  # pragma: no cover - defensive
                pass
        _LISTENERS.clear()
        _CONFIGURED = False


def _parse_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    resolved = logging.getLevelName(level.upper())
    return resolved if isinstance(resolved, int) else logging.INFO


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under ``nomorals``."""
    if name.startswith("nomorals"):
        return logging.getLogger(name)
    return logging.getLogger(f"nomorals.{name}")


def child_logger(parent: logging.Logger, suffix: str) -> logging.Logger:
    return logging.getLogger(f"{parent.name}.{suffix}")


def audit_log(name: str = "audit") -> logging.Logger:
    """A logger whose records are marked ``audit: true`` in JSON output.

    For security-relevant events: auth decisions, signup attempts, vault
    access, policy overrides. Always emitted at INFO or above and never
    sampled away.
    """
    logger = get_logger(name)
    return _AuditLogger(logger)


class _AuditLogger(logging.LoggerAdapter):
    def __init__(self, logger: logging.Logger) -> None:
        super().__init__(logger, {"audit": True})

    def process(self, msg: Any, kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        extra = dict(kwargs.get("extra") or {})
        extra["audit"] = True
        kwargs["extra"] = extra
        return msg, kwargs


class LogCapture:
    """Capture log records emitted during a block — used by the test suite."""

    def __init__(
        self,
        logger_name: str = "",
        level: int = logging.DEBUG,
        *,
        redact: bool = False,
    ) -> None:
        self.logger = logging.getLogger(logger_name)
        self.level = level
        self.records: list[logging.LogRecord] = []
        self._handler: logging.Handler | None = None
        self._redact = redact

    def __enter__(self) -> "LogCapture":
        capture = self

        class _Handler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                capture.records.append(record)

        self._handler = _Handler(level=self.level)
        if self._redact:
            self._handler.addFilter(RedactionFilter())
        self._handler.addFilter(_ContextInjectionFilter())
        self.logger.addHandler(self._handler)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._handler is not None:
            self.logger.removeHandler(self._handler)

    @property
    def messages(self) -> list[str]:
        return [r.getMessage() for r in self.records]

    def find(self, substring: str) -> list[str]:
        return [m for m in self.messages if substring in m]
