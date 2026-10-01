"""Structured logging with secret redaction.

One call — :func:`setup_logging` — configures the root logger for the whole
process: console handler, rotating file handler, optional JSON lines, and a
redaction filter that keeps tokens out of the log file and out of any log that
gets shipped to a backup repository.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Iterable

__all__ = ["RedactionFilter", "get_logger", "json_formatter", "setup_logging"]

_CONFIGURED = False
_LOCK = threading.Lock()

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


class RedactionFilter(logging.Filter):
    """Logging filter that redacts the message and known-sensitive args."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if isinstance(record.msg, str):
                record.msg = redact(record.msg)
            if record.args:
                if isinstance(record.args, dict):
                    record.args = {
                        k: (redact(v) if isinstance(v, str) else v) for k, v in record.args.items()
                    }
                elif isinstance(record.args, tuple):
                    record.args = tuple(redact(a) if isinstance(a, str) else a for a in record.args)
        except Exception:  # noqa: E103 - logging must never raise (pragma: no cover)
            pass
        return True


class _ConsoleFormatter(logging.Formatter):
    """Compact, readable console output with colour when the tty supports it."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[1;41m",
    }
    RESET = "\033[0m"

    def __init__(self, *, color: bool = True, verbose: bool = False) -> None:
        super().__init__()
        self.color = color
        self.verbose = verbose

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        name = record.name
        if name.startswith("nomorals."):
            name = name[len("nomorals.") :]
        stamp = self.formatTime(record, "%H:%M:%S")
        message = record.getMessage()
        if self.color and sys.stderr.isatty():
            paint = self.COLORS.get(level, "")
            head = f"{paint}{stamp} {level[:4]:<4}{self.RESET} {name}"
        else:
            head = f"{stamp} {level[:4]:<4} {name}"
        body = f"{head} {message}" if not self.verbose else f"{head} {message} [{record.filename}:{record.lineno}]"
        if record.exc_info:
            body = f"{body}\n{self.formatException(record.exc_info)}"
        return body


class JsonFormatter(logging.Formatter):
    """One JSON object per line — for shipping logs somewhere that parses them."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("agent", "task", "mission", "topic", "trace_id"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
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
    quiet_libs: Iterable[str] = ("urllib3", "asyncio", "filelock"),
) -> logging.Logger:
    """Configure root logging for the process. Idempotent unless ``force``."""
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
        if redact_secrets:
            console.addFilter(RedactionFilter())
        root.addHandler(console)

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
                if redact_secrets:
                    rotating.addFilter(RedactionFilter())
                root.addHandler(rotating)
            except OSError as exc:  # pragma: no cover - read-only fs
                root.warning("could not open log file %s: %s", path, exc)

        for lib in quiet_libs:
            logging.getLogger(lib).setLevel(max(numeric, logging.WARNING))

        logging.captureWarnings(True)
        _CONFIGURED = True
        return root


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


class LogCapture:
    """Capture log records emitted during a block — used by the test suite."""

    def __init__(self, logger_name: str = "", level: int = logging.DEBUG) -> None:
        self.logger = logging.getLogger(logger_name)
        self.level = level
        self.records: list[logging.LogRecord] = []
        self._handler: logging.Handler | None = None

    def __enter__(self) -> "LogCapture":
        capture = self

        class _Handler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                capture.records.append(record)

        self._handler = _Handler(level=self.level)
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
