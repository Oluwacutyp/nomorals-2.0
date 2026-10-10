"""Error Intelligence System - catch, classify, diagnose, learn, escalate.

The error catcher is a full pipeline, not a classifier:

1. **Catch** — :func:`catch_and_analyze` / :meth:`ErrorIntelligence.analyze`
   wrap any operation and never lose the exception.
2. **Classify** — exception type + message patterns + the shared
   :func:`~nomorals.core.errors.classify` hierarchy (one error dialect).
3. **Diagnose root cause** — pattern knowledge, live exception chains
   (``__cause__``/``__context__``), and per-fingerprint error groups.
4. **Suggest / attempt repair** — static pattern fixes first, then *learned*
   fixes: every recorded outcome (``record_outcome``) teaches the knowledge
   base which fixes actually work, per fingerprint. Fixes that repeatedly
   fail are demoted automatically.
5. **Learn from history** — errors are fingerprinted Sentry-style (dynamic
   values normalized away), grouped with counts / first-seen / last-seen /
   status, persisted to disk, and re-opened on regression. Sudden surges in
   a fingerprint raise the severity (spike detection).
6. **Escalate honestly** — when there is no pattern, no learned fix, and no
   classification, the analysis says so (``needs_escalation``) with an
   operator-ready summary instead of a confident-sounding guess.

Usage:
    from nomorals.core.error_intelligence import ErrorIntelligence, catch_and_analyze

    ei = ErrorIntelligence()

    try:
        result = some_risky_operation()
    except Exception as e:
        analysis = ei.analyze(e, context={"operation": "email_send"})
        print(analysis.explanation)
        print(analysis.suggested_fix)
        if analysis.needs_escalation:
            page_operator(analysis.escalation_message())

    # Teach it what worked — this is how the knowledge base learns:
    ei.record_outcome(analysis, fixed=True, fix="rotated the API key")

    # Or use the decorator
    @catch_and_analyze(context={"operation": "calendar_sync"})
    def sync_calendar():
        ...
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import inspect
import json
import os
import re
import sys
import threading
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Iterator, Optional, TypeVar

from ..core.logging_setup import get_logger
from .errors import NoMoralsError, classify, retry_after_of

__all__ = [
    "ErrorIntelligence",
    "ErrorAnalysis",
    "ErrorCategory",
    "ErrorFingerprint",
    "ErrorGroup",
    "ErrorSeverity",
    "LearningKnowledgeBase",
    "catch_and_analyze",
    "ErrorKnowledgeBase",
    "analyze_error",
    "add_breadcrumb",
]

_log = get_logger(__name__)


class ErrorCategory(Enum):
    """Classification of error types."""

    NETWORK = "network"  # Connection, timeout, DNS
    AUTH = "authentication"  # Invalid credentials, expired tokens, permissions
    RATE_LIMIT = "rate_limit"  # API rate limits, quota exceeded
    VALIDATION = "validation"  # Invalid input, schema errors
    RESOURCE = "resource"  # File not found, disk full, memory
    PARSING = "parsing"  # JSON/XML/HTML parsing failures
    INTEGRATION = "integration"  # Third-party API errors
    CRYPTO = "cryptography"  # Encryption/decryption failures
    DATABASE = "database"  # SQL errors, connection pool
    CONFIGURATION = "configuration"  # Missing config, invalid settings
    PERMISSION = "permission"  # File permissions, access denied
    TIMEOUT = "timeout"  # Operation exceeded time limit
    UNKNOWN = "unknown"  # Unclassified errors


class ErrorSeverity(Enum):
    """Severity levels for errors."""

    LOW = "low"  # Cosmetic, non-critical
    MEDIUM = "medium"  # Degraded functionality
    HIGH = "high"  # Feature broken, user impacted
    CRITICAL = "critical"  # System failure, data loss risk


_SEVERITY_ORDER = [ErrorSeverity.LOW, ErrorSeverity.MEDIUM,
                   ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]


def _bump_severity(sev: ErrorSeverity) -> ErrorSeverity:
    idx = _SEVERITY_ORDER.index(sev)
    return _SEVERITY_ORDER[min(idx + 1, len(_SEVERITY_ORDER) - 1)]


# ---------------------------------------------------------------------------
# fingerprinting — Sentry-style grouping
# ---------------------------------------------------------------------------

_NORMALIZERS: list[tuple[re.Pattern[str], str]] = [
    # order matters: most specific first
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"), "<uuid>"),
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<addr>"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d{1,5})?\b"), "<ip>"),
    (re.compile(r"https?://\S+"), "<url>"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "<email>"),
    (re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?"), "<ts>"),
    (re.compile(r"\b[0-9a-fA-F]{16,64}\b"), "<hash>"),
    (re.compile(r"(?:[A-Za-z]:\\|/)(?:[\w.~-]+[/\\])+[\w.~-]*"), "<path>"),
    (re.compile(r"'[^']{24,}'"), "'<str>'"),
    (re.compile(r'"[^"]{24,}"'), '"<str>"'),
    (re.compile(r"\b\d+\b"), "<num>"),
]


def _normalize_message(message: str) -> str:
    """Replace dynamic values with placeholders so identical errors group."""
    out = message
    for pattern, placeholder in _NORMALIZERS:
        out = pattern.sub(placeholder, out)
    return out


def _is_user_code(filename: str) -> bool:
    name = filename.replace("\\", "/")
    if "site-packages" in name or "dist-packages" in name:
        return False
    stdlib = sys.base_prefix.replace("\\", "/")
    if stdlib and name.startswith(stdlib):
        rest = name[len(stdlib):]
        if "/lib/python" in rest:
            return False
    return not (name.startswith("<") and name.endswith(">"))


def _stack_signature(exc: BaseException, depth: int = 3) -> str:
    """Stable stack identity: module-ish path + function name of the top
    in-app frames. Line numbers are deliberately excluded — they shift with
    every deploy and would shatter grouping."""
    frames: list[str] = []
    tb = getattr(exc, "__traceback__", None)
    while tb is not None:
        frame = tb.tb_frame
        filename = frame.f_code.co_filename
        if _is_user_code(filename):
            short = filename.replace("\\", "/").split("/")[-2:]
            frames.append(f"{'/'.join(short)}:{frame.f_code.co_name}")
        tb = tb.tb_next
    return "|".join(frames[-depth:]) if frames else "<no-app-frames>"


class ErrorFingerprint:
    """Stable identity for an error: ``sha256(type + normalized message +
    stack signature)``. ``Connection refused to 127.0.0.1:6379`` and
    ``... to 10.0.0.1:6379`` produce the same fingerprint — one issue,
    not a thousand."""

    @staticmethod
    def of(exception: BaseException) -> str:
        exc_type = type(exception).__name__
        normalized = _normalize_message(str(exception) or exc_type)
        stack_sig = _stack_signature(exception)
        digest = hashlib.sha256(
            f"{exc_type}\n{normalized}\n{stack_sig}".encode("utf-8")
        ).hexdigest()
        return digest[:16]

    @staticmethod
    def normalized_message(exception: BaseException) -> str:
        return _normalize_message(str(exception) or type(exception).__name__)


@dataclass
class ErrorGroup:
    """All occurrences of one fingerprint: counts, lifecycle, examples."""

    fingerprint: str
    exc_type: str
    pattern_name: str = ""
    category: ErrorCategory = ErrorCategory.UNKNOWN
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    count: int = 0
    first_seen: str = ""
    last_seen: str = ""
    status: str = "new"  # new | acknowledged | resolved
    resolved_at: str | None = None
    examples: list[str] = field(default_factory=list)  # recent raw messages

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "exc_type": self.exc_type,
            "pattern_name": self.pattern_name,
            "category": self.category.value,
            "severity": self.severity.value,
            "count": self.count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "status": self.status,
            "resolved_at": self.resolved_at,
            "examples": self.examples[-3:],
        }


@dataclass
class LearnedFix:
    """A fix associated with a fingerprint, with a track record."""

    fix: str
    source: str = "operator"  # operator | pattern | outcome
    successes: int = 0
    failures: int = 0
    updated_at: str = ""

    @property
    def confidence(self) -> float:
        # Laplace smoothing: a fresh fix starts at 0.5, evidence moves it.
        return (self.successes + 1) / (self.successes + self.failures + 2)

    @property
    def demoted(self) -> bool:
        return self.failures >= 3 and self.confidence < 0.3

    def to_dict(self) -> dict[str, Any]:
        return {
            "fix": self.fix, "source": self.source,
            "successes": self.successes, "failures": self.failures,
            "confidence": round(self.confidence, 3),
            "updated_at": self.updated_at,
        }


# ---------------------------------------------------------------------------
# static pattern knowledge (kept for bootstrap + backward compat)
# ---------------------------------------------------------------------------

class ErrorKnowledgeBase:
    """Database of known errors and their solutions."""

    # Pattern matchers for error messages
    PATTERNS = {
        # Network errors
        "connection_refused": {
            "patterns": [r"connection refused", r"connection reset", r"ECONNREFUSED"],
            "category": ErrorCategory.NETWORK,
            "severity": ErrorSeverity.HIGH,
            "root_cause": "Service is down or not accepting connections",
            "fix": "Check if the service is running and accessible. Verify the host/port configuration.",
        },
        "dns_failure": {
            "patterns": [r"name resolution failed", r"getaddrinfo failed", r"DNS.*not found"],
            "category": ErrorCategory.NETWORK,
            "severity": ErrorSeverity.HIGH,
            "root_cause": "DNS lookup failed - domain doesn't exist or DNS server unreachable",
            "fix": "Check your internet connection and verify the domain name is correct.",
        },
        "timeout": {
            "patterns": [r"timeout", r"timed out", r"deadline exceeded"],
            "category": ErrorCategory.TIMEOUT,
            "severity": ErrorSeverity.MEDIUM,
            "root_cause": "Operation took too long to complete",
            "fix": "The service is slow or overloaded. Try again later or increase the timeout value.",
            "retryable": True,
        },

        # Authentication errors
        "invalid_credentials": {
            "patterns": [r"invalid.*password", r"authentication failed", r"unauthorized", r"401"],
            "category": ErrorCategory.AUTH,
            "severity": ErrorSeverity.HIGH,
            "root_cause": "Wrong username or password",
            "fix": "Verify credentials are correct. If using OAuth, the token may have expired - refresh it.",
        },
        "expired_token": {
            "patterns": [r"token.*expired", r"invalid.*token", r"access.*denied"],
            "category": ErrorCategory.AUTH,
            "severity": ErrorSeverity.HIGH,
            "root_cause": "OAuth token or API key has expired",
            "fix": "Refresh the OAuth token or generate a new API key. Check token expiry settings.",
        },
        "permission_denied": {
            "patterns": [r"permission denied", r"forbidden", r"403", r"access denied"],
            "category": ErrorCategory.PERMISSION,
            "severity": ErrorSeverity.HIGH,
            "root_cause": "Insufficient permissions for this operation",
            "fix": "Check that the account has the required permissions/scopes. Re-authorize if needed.",
        },

        # Rate limiting
        "rate_limit": {
            "patterns": [r"rate.*limit", r"too many requests", r"429", r"quota.*exceeded"],
            "category": ErrorCategory.RATE_LIMIT,
            "severity": ErrorSeverity.MEDIUM,
            "root_cause": "API rate limit exceeded",
            "fix": "Wait before making more requests. Consider using exponential backoff or reducing request frequency.",
            "retryable": True,
        },

        # Resource errors
        "file_not_found": {
            "patterns": [r"no such file", r"file not found", r"does not exist"],
            "category": ErrorCategory.RESOURCE,
            "severity": ErrorSeverity.MEDIUM,
            "root_cause": "File or directory doesn't exist",
            "fix": "Verify the file path is correct. Check if the file was moved or deleted.",
        },
        "disk_full": {
            "patterns": [r"no space left", r"disk.*full", r"out of space"],
            "category": ErrorCategory.RESOURCE,
            "severity": ErrorSeverity.CRITICAL,
            "root_cause": "Disk is full",
            "fix": "Free up disk space by deleting old files or increasing storage quota.",
        },
        "memory_error": {
            "patterns": [r"out of memory", r"memory allocation", r"killed.*oom"],
            "category": ErrorCategory.RESOURCE,
            "severity": ErrorSeverity.CRITICAL,
            "root_cause": "Out of memory",
            "fix": "Reduce memory usage or increase available RAM. Consider processing data in smaller chunks.",
        },

        # Parsing errors
        "json_parse": {
            "patterns": [r"json.*decode", r"invalid.*json", r"expecting.*value"],
            "category": ErrorCategory.PARSING,
            "severity": ErrorSeverity.MEDIUM,
            "root_cause": "Invalid JSON data",
            "fix": "The API returned malformed JSON. Check the response format or contact the service provider.",
        },
        "html_parse": {
            "patterns": [r"html.*parse", r"selector.*not found", r"element.*not found"],
            "category": ErrorCategory.PARSING,
            "severity": ErrorSeverity.MEDIUM,
            "root_cause": "HTML structure changed or element not found",
            "fix": "The website may have updated their layout. Update the selectors or try a different method.",
        },

        # Integration errors
        "api_error": {
            "patterns": [r"api.*error", r"service.*unavailable", r"500.*internal"],
            "category": ErrorCategory.INTEGRATION,
            "severity": ErrorSeverity.HIGH,
            "root_cause": "Third-party API returned an error",
            "fix": "The external service is experiencing issues. Try again later or check their status page.",
            "retryable": True,
        },

        # Database errors
        "database_locked": {
            "patterns": [r"database.*locked", r"sqlite.*busy"],
            "category": ErrorCategory.DATABASE,
            "severity": ErrorSeverity.MEDIUM,
            "root_cause": "Database is locked by another process",
            "fix": "Another process is using the database. Wait and retry, or check for stuck transactions.",
            "retryable": True,
        },

        # Crypto errors
        "decryption_failed": {
            "patterns": [r"decryption.*failed", r"cipher.*error", r"invalid.*key"],
            "category": ErrorCategory.CRYPTO,
            "severity": ErrorSeverity.HIGH,
            "root_cause": "Decryption failed - wrong key or corrupted data",
            "fix": "Verify the encryption key is correct. The data may be corrupted or encrypted with a different key.",
        },
    }

    @classmethod
    def match(cls, error_message: str) -> Optional[dict[str, Any]]:
        """Match error message against known patterns.

        Args:
            error_message: Error message to match

        Returns:
            Matched pattern info or None
        """
        error_lower = error_message.lower()

        for pattern_name, pattern_info in cls.PATTERNS.items():
            for pattern in pattern_info["patterns"]:
                if re.search(pattern, error_lower, re.IGNORECASE):
                    return {
                        "pattern_name": pattern_name,
                        **pattern_info,
                    }

        return None


# ---------------------------------------------------------------------------
# learning knowledge base — groups + learned fixes + persistence
# ---------------------------------------------------------------------------

def _default_kb_path() -> str:
    override = os.environ.get("DEVON_ERROR_KB")
    if override:
        return override
    return os.path.expanduser("~/.local/share/devon/error_kb.json")


class LearningKnowledgeBase:
    """The knowledge base that actually learns.

    * Groups errors by fingerprint (counts, first/last seen, status
      lifecycle with regression re-open).
    * Associates fixes with fingerprints and tracks their success rate;
      fixes that repeatedly fail are demoted automatically.
    * Detects spikes: a sudden surge in a fingerprint's rate bumps severity.
    * Persists to disk so learning survives restarts.
    """

    def __init__(self, store_path: str | None = None,
                 regression_cooldown_days: float = 7.0,
                 max_groups: int = 5000) -> None:
        self.store_path = store_path if store_path != "" else None
        if store_path is None:
            self.store_path = _default_kb_path()
        self.regression_cooldown = regression_cooldown_days * 86400.0
        self.max_groups = max_groups
        self._groups: dict[str, ErrorGroup] = {}
        self._fixes: dict[str, LearnedFix] = {}
        self._spike_times: dict[str, deque[float]] = {}
        self._lock = threading.RLock()
        self._loaded = False
        self._dirty = False
        self._last_save = 0.0

    # -- persistence -------------------------------------------------------
    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _ensure_loaded(self) -> None:
        if self._loaded or not self.store_path:
            return
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            try:
                if not os.path.exists(self.store_path):
                    return
                with open(self.store_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:  # noqa: BLE001 - corrupt KB never breaks analysis
                _log.warning("error KB at %s unreadable (%s); starting fresh",
                             self.store_path, exc)
                return
            try:
                for fp, g in (data.get("groups") or {}).items():
                    self._groups[fp] = ErrorGroup(
                        fingerprint=fp,
                        exc_type=g.get("exc_type", "?"),
                        pattern_name=g.get("pattern_name", ""),
                        category=ErrorCategory(g.get("category", "unknown")),
                        severity=ErrorSeverity(g.get("severity", "medium")),
                        count=int(g.get("count", 0)),
                        first_seen=g.get("first_seen", ""),
                        last_seen=g.get("last_seen", ""),
                        status=g.get("status", "new"),
                        resolved_at=g.get("resolved_at"),
                        examples=list(g.get("examples") or []),
                    )
                for fp, fx in (data.get("fixes") or {}).items():
                    self._fixes[fp] = LearnedFix(
                        fix=fx.get("fix", ""),
                        source=fx.get("source", "operator"),
                        successes=int(fx.get("successes", 0)),
                        failures=int(fx.get("failures", 0)),
                        updated_at=fx.get("updated_at", ""),
                    )
            except Exception as exc:  # noqa: BLE001
                _log.warning("error KB at %s has bad entries (%s); kept what parsed",
                             self.store_path, exc)

    def _save(self, force: bool = False) -> None:
        if not self.store_path or not self._dirty:
            return
        now = datetime.now(timezone.utc).timestamp()
        if not force and now - self._last_save < 2.0:
            return  # throttle: at most one write per 2s
        with self._lock:
            if not self._dirty:
                return
            payload = {
                "version": 1,
                "groups": {fp: g.to_dict() for fp, g in self._groups.items()},
                "fixes": {fp: fx.to_dict() for fp, fx in self._fixes.items()},
            }
            try:
                os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
                tmp = self.store_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=1)
                os.replace(tmp, self.store_path)
                self._dirty = False
                self._last_save = now
            except Exception as exc:  # noqa: BLE001 - persistence never breaks analysis
                _log.debug("error KB save failed: %s", exc)

    def flush(self) -> None:
        """Force a save. Called on shutdown paths."""
        self._save(force=True)

    # -- groups ------------------------------------------------------------
    def note_event(self, fingerprint: str, exc_type: str, message: str,
                   pattern_name: str, category: ErrorCategory,
                   severity: ErrorSeverity) -> tuple[ErrorGroup, bool, bool]:
        """Record one occurrence. Returns (group, is_new, regression).

        A ``resolved`` group that recurs after the cooldown re-opens as
        ``new`` with ``regression=True``.
        """
        self._ensure_loaded()
        now_iso = self._now_iso()
        now_ts = datetime.now(timezone.utc).timestamp()
        with self._lock:
            group = self._groups.get(fingerprint)
            is_new = group is None
            if is_new:
                group = ErrorGroup(
                    fingerprint=fingerprint, exc_type=exc_type,
                    pattern_name=pattern_name, category=category,
                    severity=severity, first_seen=now_iso,
                )
                self._groups[fingerprint] = group
                if len(self._groups) > self.max_groups:
                    # evict the stalest group
                    oldest = min(self._groups.values(),
                                 key=lambda g: g.last_seen or "")
                    del self._groups[oldest.fingerprint]
            regression = False
            if group.status == "resolved":
                try:
                    resolved_ts = datetime.fromisoformat(
                        group.resolved_at or "").timestamp()
                except ValueError:
                    resolved_ts = 0.0
                if now_ts - resolved_ts >= self.regression_cooldown:
                    group.status = "new"
                    group.resolved_at = None
                    regression = True
            group.count += 1
            group.last_seen = now_iso
            if not group.first_seen:
                group.first_seen = now_iso
            group.examples.append(message[:300])
            del group.examples[:-3]
            # in-memory spike tracking (live signal, not persisted)
            times = self._spike_times.setdefault(fingerprint, deque(maxlen=200))
            times.append(now_ts)
            self._dirty = True
        self._save()
        return group, is_new, regression

    def check_spike(self, fingerprint: str, window: float = 60.0,
                    min_count: int = 5, ratio: float = 3.0) -> bool:
        """True when a fingerprint's recent rate surged vs the previous window."""
        with self._lock:
            times = self._spike_times.get(fingerprint)
            if not times:
                return False
            now = datetime.now(timezone.utc).timestamp()
            recent = sum(1 for t in times if t > now - window)
            previous = sum(1 for t in times
                           if now - 2 * window < t <= now - window)
            return recent >= min_count and recent >= ratio * max(1, previous)

    def _window_times(self, fingerprint: str,
                      window: float) -> list[float]:
        now = datetime.now(timezone.utc).timestamp()
        times = self._spike_times.get(fingerprint) or []
        return sorted(t for t in times if t > now - window)

    def flakiness(self, fingerprint: str,
                  window: float = 3600.0) -> dict[str, Any]:
        """Is this error flaky (intermittent) or steady?

        Two signals: *episodes* (occurrence clusters separated by quiet
        gaps > 5 min or > window/12) and the coefficient of variation of
        the inter-arrival times. Evenly-spaced recurrences are "steady";
        irregular bursts with long quiets between are "flaky". Needs ≥4
        occurrences in the window.
        """
        times = self._window_times(fingerprint, window)
        out: dict[str, Any] = {
            "fingerprint": fingerprint, "window_s": window,
            "occurrences": len(times), "episodes": 0,
            "verdict": "insufficient-data", "score": 0.0,
        }
        if len(times) < 4:
            return out
        gap = max(300.0, window / 12)
        episodes = 1
        gaps: list[float] = []
        for prev, cur in zip(times, times[1:]):
            gaps.append(cur - prev)
        # episode boundary: a quiet much longer than the typical spacing
        # (adapts to the error's own rhythm; floor keeps one-offs apart)
        med = sorted(gaps)[len(gaps) // 2]
        episode_gap = max(gap, 3 * med)
        for g in gaps:
            if g > episode_gap:
                episodes += 1
        mean_gap = sum(gaps) / len(gaps)
        var = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
        cv = (var ** 0.5) / mean_gap if mean_gap else 0.0
        span = times[-1] - times[0]
        out["episodes"] = episodes
        out["gap_cv"] = round(cv, 2)
        out["span_s"] = round(span, 1)
        if episodes >= 2 and cv > 1.2:
            verdict, score = "flaky", min(1.0, 0.4 + cv / 5)
        elif episodes >= 2:
            verdict, score = "intermittent", 0.4
        elif span < window * 0.1:
            verdict, score = "burst", 0.2
        else:
            verdict, score = "steady", 0.1
        out["verdict"] = verdict
        out["score"] = round(score, 2)
        return out

    def trend(self, fingerprint: str, window: float = 86400.0,
              buckets: int = 12) -> dict[str, Any]:
        """Rate trend over the window: rising / falling / stable.

        Buckets the window, fits a least-squares slope to the per-bucket
        counts, and compares the slope against the mean rate.
        """
        times = self._window_times(fingerprint, window)
        now = datetime.now(timezone.utc).timestamp()
        edges = [now - window + i * window / buckets for i in range(buckets + 1)]
        counts = [0] * buckets
        for t in times:
            idx = min(buckets - 1, int((t - edges[0]) / window * buckets))
            if idx >= 0:
                counts[idx] += 1
        n = buckets
        mean_x = (n - 1) / 2
        mean_y = sum(counts) / n if n else 0.0
        denom = sum((i - mean_x) ** 2 for i in range(n))
        slope = (sum((i - mean_x) * (c - mean_y) for i, c in enumerate(counts))
                 / denom) if denom and mean_y else 0.0
        rel = slope / mean_y if mean_y else 0.0
        if rel > 0.15:
            direction = "rising"
        elif rel < -0.15:
            direction = "falling"
        else:
            direction = "stable"
        half = n // 2
        first_half = sum(counts[:half]) or 1
        return {
            "fingerprint": fingerprint,
            "window_s": window,
            "occurrences": len(times),
            "per_bucket": counts,
            "slope_per_bucket": round(slope, 3),
            "direction": direction,
            "second_vs_first_half": round(sum(counts[half:]) / first_half, 2),
        }

    def get_group(self, fingerprint: str) -> ErrorGroup | None:
        self._ensure_loaded()
        with self._lock:
            return self._groups.get(fingerprint)

    def groups(self, status: str | None = None) -> list[ErrorGroup]:
        self._ensure_loaded()
        with self._lock:
            items = list(self._groups.values())
        if status:
            items = [g for g in items if g.status == status]
        return items

    def top_groups(self, limit: int = 10) -> list[ErrorGroup]:
        return sorted(self.groups(), key=lambda g: -g.count)[:limit]

    def acknowledge(self, fingerprint: str) -> bool:
        return self._set_status(fingerprint, "acknowledged")

    def resolve(self, fingerprint: str) -> bool:
        ok = self._set_status(fingerprint, "resolved")
        if ok:
            with self._lock:
                group = self._groups.get(fingerprint)
                if group:
                    group.resolved_at = self._now_iso()
                    self._dirty = True
            self._save(force=True)
        return ok

    def _set_status(self, fingerprint: str, status: str) -> bool:
        self._ensure_loaded()
        with self._lock:
            group = self._groups.get(fingerprint)
            if group is None:
                return False
            group.status = status
            self._dirty = True
        self._save(force=True)
        return True

    # -- learned fixes -----------------------------------------------------
    def learn_fix(self, fingerprint: str, fix: str,
                  source: str = "operator") -> None:
        """Teach the KB a fix for a fingerprint. Operator-taught fixes start
        with one success on trust; outcome-taught fixes earn theirs."""
        self._ensure_loaded()
        with self._lock:
            existing = self._fixes.get(fingerprint)
            if existing is not None and existing.fix == fix:
                return  # already known
            self._fixes[fingerprint] = LearnedFix(
                fix=fix, source=source,
                successes=1 if source == "operator" else 0,
                updated_at=self._now_iso(),
            )
            self._dirty = True
        self._save(force=True)  # learned knowledge is durable immediately

    def record_fix_outcome(self, fingerprint: str, success: bool,
                           fix: str = "") -> None:
        """The learning loop: record whether a fix worked.

        ``fix`` names the fix that was tried; when it differs from the stored
        one (or none is stored) and it succeeded, it becomes the learned fix.
        """
        self._ensure_loaded()
        with self._lock:
            entry = self._fixes.get(fingerprint)
            if entry is None:
                if success and fix:
                    self._fixes[fingerprint] = LearnedFix(
                        fix=fix, source="outcome", successes=1,
                        updated_at=self._now_iso())
                    self._dirty = True
                self._save(force=True)
                return
            if fix and fix != entry.fix:
                if success:
                    # the tried fix beat the stored one — replace it
                    self._fixes[fingerprint] = LearnedFix(
                        fix=fix, source="outcome", successes=1,
                        updated_at=self._now_iso())
                    self._dirty = True
                    self._save(force=True)
                    return
                # a different fix failed: penalize nothing (it wasn't ours),
                # but don't learn it either
                return
            if success:
                entry.successes += 1
            else:
                entry.failures += 1
            entry.updated_at = self._now_iso()
            self._dirty = True
        self._save(force=True)

    def suggest_fix(self, fingerprint: str,
                    min_confidence: float = 0.5) -> LearnedFix | None:
        """Best learned fix for a fingerprint, or None.

        Demoted fixes (repeatedly failed) are never suggested — the KB would
        rather admit ignorance than repeat a known-bad fix.
        """
        self._ensure_loaded()
        with self._lock:
            entry = self._fixes.get(fingerprint)
        if entry is None or entry.demoted:
            return None
        if entry.confidence < min_confidence:
            return None
        return entry


@dataclass
class ErrorAnalysis:
    """Complete analysis of an error."""

    # Basic info
    exception_type: str
    exception_message: str
    category: ErrorCategory
    severity: ErrorSeverity

    # Analysis
    root_cause: str
    explanation: str
    suggested_fix: str

    # Context
    context: dict[str, Any] = field(default_factory=dict)
    stack_trace: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Metadata
    retryable: bool = False
    retry_after: Optional[float] = None  # Seconds to wait before retry
    error_code: str = ""

    # Related
    related_errors: list[str] = field(default_factory=list)
    documentation_url: str = ""

    # Learning / grouping (new)
    fingerprint: str = ""
    normalized_message: str = ""
    group_count: int = 1
    group_status: str = "new"
    is_new_group: bool = True
    regression: bool = False
    spike: bool = False
    learned_fix: str = ""
    fix_confidence: float = 0.0
    fix_source: str = ""  # pattern | learned | none
    chained: list[dict[str, str]] = field(default_factory=list)
    breadcrumbs: list[dict[str, Any]] = field(default_factory=list)
    needs_escalation: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "category": self.category.value,
            "severity": self.severity.value,
            "root_cause": self.root_cause,
            "explanation": self.explanation,
            "suggested_fix": self.suggested_fix,
            "context": self.context,
            "retryable": self.retryable,
            "retry_after": self.retry_after,
            "error_code": self.error_code,
            "timestamp": self.timestamp.isoformat(),
            "documentation_url": self.documentation_url,
            "fingerprint": self.fingerprint,
            "normalized_message": self.normalized_message,
            "group_count": self.group_count,
            "group_status": self.group_status,
            "is_new_group": self.is_new_group,
            "regression": self.regression,
            "spike": self.spike,
            "learned_fix": self.learned_fix,
            "fix_confidence": round(self.fix_confidence, 3),
            "fix_source": self.fix_source,
            "chained": self.chained,
            "breadcrumbs": self.breadcrumbs,
            "needs_escalation": self.needs_escalation,
        }

    def to_user_message(self) -> str:
        """Generate user-friendly error message."""
        msg = f"⚠️ {self.explanation}\n\n"
        if self.suggested_fix:
            msg += f"💡 **Fix:** {self.suggested_fix}\n\n"
            if self.fix_source == "learned" and self.fix_confidence >= 0.7:
                msg += (f"_This fix worked {self.fix_confidence:.0%} of the time "
                        f"on similar errors._\n\n")
        if self.regression:
            msg += "🔁 This error was marked resolved before and came back — regression.\n\n"
        if self.spike:
            msg += "📈 This error is spiking right now — worth a look.\n\n"
        if self.retryable:
            retry_text = "in a moment" if not self.retry_after else f"in {int(self.retry_after)} seconds"
            msg += f"🔄 This is temporary - I'll retry {retry_text}."
        if self.needs_escalation:
            msg += "\n🚨 I don't have a known fix for this — escalating."
        return msg.strip()

    def escalation_message(self) -> str:
        """Operator-ready summary for when the catcher can't fix it."""
        lines = [
            "🚨 ERROR ESCALATION — no known fix",
            f"Type: {self.exception_type}",
            f"Fingerprint: {self.fingerprint or 'n/a'} "
            f"(seen {self.group_count}x, status: {self.group_status})",
            f"Category: {self.category.value} | Severity: {self.severity.value}",
            f"Root cause: {self.root_cause}",
            f"Message: {self.exception_message[:300]}",
        ]
        if self.chained:
            lines.append("Chain: " + " <- ".join(
                f"{c['type']}" for c in self.chained))
        if self.breadcrumbs:
            lines.append("Breadcrumbs:")
            for b in self.breadcrumbs[-8:]:
                lines.append(f"  · [{b.get('category')}] {b.get('message')}")
        op = self.context.get("operation")
        if op:
            lines.append(f"Operation: {op}")
        if self.stack_trace:
            lines.append("Traceback (tail):")
            lines.extend(self.stack_trace.strip().splitlines()[-12:])
        return "\n".join(lines)


class ErrorIntelligence:
    """Intelligent error analysis and root cause detection.

    Every :meth:`analyze` call fingerprints the error, updates its group,
    consults static patterns *and* learned fixes, checks for spikes and
    regressions, and records the outcome hooks the learning loop needs.
    """

    def __init__(self, store_path: str | None = None) -> None:
        self.knowledge_base = ErrorKnowledgeBase()
        self.learning = LearningKnowledgeBase(store_path=store_path)
        self._breadcrumbs: deque[dict[str, Any]] = deque(maxlen=100)
        self._scope_tags: dict[str, Any] = {}
        self._lock = threading.RLock()
        _log.info("Error Intelligence System initialized")

    # -- breadcrumbs & scoped context --------------------------------------
    def add_breadcrumb(self, message: str, category: str = "default",
                       data: dict[str, Any] | None = None) -> None:
        """Leave a trail crumb. The next analyzed error carries the trail —
        the debugging context a bare stack trace never has."""
        with self._lock:
            self._breadcrumbs.append({
                "ts": datetime.now(timezone.utc).isoformat(),
                "category": category,
                "message": message,
                "data": data or {},
            })

    @contextlib.contextmanager
    def scoped_context(self, **tags: Any) -> Iterator[None]:
        """Tags merged into every analysis made inside the block, without
        polluting the caller's context dicts."""
        prev = self._scope_tags
        self._scope_tags = {**prev, **tags}
        try:
            yield
        finally:
            self._scope_tags = prev

    # -- main entry point --------------------------------------------------
    def analyze(
        self,
        exception: Exception,
        *,
        context: dict[str, Any] | None = None,
        include_stack: bool = True,
    ) -> ErrorAnalysis:
        """Analyze an exception and provide insights.

        Args:
            exception: The exception to analyze
            context: Additional context about the operation
            include_stack: Include stack trace in analysis

        Returns:
            ErrorAnalysis with full details
        """
        context = {**self._scope_tags, **(context or {})}

        # Get basic info
        exc_type = type(exception).__name__
        exc_message = str(exception)
        stack_trace = traceback.format_exc() if include_stack else ""

        # Exception chain: the root cause often hides behind "During handling
        # of the above exception..."
        chained = self._exception_chain(exception)

        # Fingerprint + group bookkeeping (Sentry-style)
        fingerprint = ErrorFingerprint.of(exception)
        normalized = ErrorFingerprint.normalized_message(exception)

        # Try to match against knowledge base
        match = self.knowledge_base.match(exc_message)

        if match:
            # Known error pattern
            category = match["category"]
            severity = match["severity"]
            root_cause = match["root_cause"]
            explanation = self._generate_explanation(match, exception, context)
            suggested_fix = match["fix"]
            retryable = match.get("retryable", False)
            error_code = match["pattern_name"]
            fix_source = "pattern"
        else:
            # Unknown error - classify by exception type
            category = self._classify_by_type(exception)
            severity = self._estimate_severity(exception)
            root_cause = self._infer_root_cause(exception, context, chained)
            explanation = self._generate_explanation_for_unknown(exception, context)
            suggested_fix = self._suggest_fix_for_unknown(exception, category)
            retryable = self._is_retryable(exception)
            error_code = f"unknown_{exc_type.lower()}"
            fix_source = "none"

        # Learned fixes override static ones when the track record is good.
        learned = self.learning.suggest_fix(fingerprint)
        fix_confidence = 0.0
        learned_fix = ""
        if learned is not None:
            learned_fix = learned.fix
            fix_confidence = learned.confidence
            suggested_fix = learned.fix
            fix_source = "learned"

        # Group bookkeeping: counts, lifecycle, regression
        group, is_new, regression = self.learning.note_event(
            fingerprint, exc_type, exc_message,
            match["pattern_name"] if match else "",
            category, severity,
        )
        spike = self.learning.check_spike(fingerprint)
        if spike:
            severity = _bump_severity(severity)

        # Retryable-ness from the shared error hierarchy when the pattern
        # didn't already decide it.
        if not match:
            retryable = self._is_retryable(exception)

        # Extract retry-after if available (shared helper: one dialect)
        retry_after = self._extract_retry_after(exception)

        # Honest escalation: no pattern, no learned fix, unknown category —
        # say so instead of a confident-sounding guess.
        needs_escalation = (
            match is None and learned is None
            and category == ErrorCategory.UNKNOWN
        )

        with self._lock:
            breadcrumbs = list(self._breadcrumbs)

        # Build analysis
        analysis = ErrorAnalysis(
            exception_type=exc_type,
            exception_message=exc_message,
            category=category,
            severity=severity,
            root_cause=root_cause,
            explanation=explanation,
            suggested_fix=suggested_fix,
            context=context,
            stack_trace=stack_trace,
            retryable=retryable,
            retry_after=retry_after,
            error_code=error_code,
            fingerprint=fingerprint,
            normalized_message=normalized,
            group_count=group.count,
            group_status=group.status,
            is_new_group=is_new,
            regression=regression,
            spike=spike,
            learned_fix=learned_fix,
            fix_confidence=fix_confidence,
            fix_source=fix_source,
            chained=chained,
            breadcrumbs=breadcrumbs,
            needs_escalation=needs_escalation,
        )

        # Log the analysis
        self._log_analysis(analysis)

        return analysis

    # -- learning loop -----------------------------------------------------
    def record_outcome(self, analysis_or_fingerprint: ErrorAnalysis | str, *,
                       fixed: bool, fix: str = "") -> None:
        """Teach the KB whether a fix worked. This is the learning loop.

        Call it after acting on an analysis: ``fixed=True`` with the fix
        that worked promotes it for the next identical error;
        ``fixed=False`` demotes the tried fix so it stops being suggested.
        """
        fp = (analysis_or_fingerprint.fingerprint
              if isinstance(analysis_or_fingerprint, ErrorAnalysis)
              else str(analysis_or_fingerprint))
        if not fp:
            return
        fix = fix or (analysis_or_fingerprint.suggested_fix
                      if isinstance(analysis_or_fingerprint, ErrorAnalysis)
                      else "")
        self.learning.record_fix_outcome(fp, fixed, fix)

    def learn_fix(self, fingerprint: str, fix: str) -> None:
        """Directly teach a fix for a fingerprint (operator knowledge)."""
        self.learning.learn_fix(fingerprint, fix, source="operator")

    def acknowledge(self, fingerprint: str) -> bool:
        return self.learning.acknowledge(fingerprint)

    def resolve(self, fingerprint: str) -> bool:
        return self.learning.resolve(fingerprint)

    def groups(self, status: str | None = None) -> list[dict[str, Any]]:
        return [g.to_dict() for g in self.learning.groups(status)]

    def top_groups(self, limit: int = 10) -> list[dict[str, Any]]:
        return [g.to_dict() for g in self.learning.top_groups(limit)]

    def flakiness(self, fingerprint: str,
                  window: float = 3600.0) -> dict[str, Any]:
        """Flakiness analysis for one error fingerprint — see
        :meth:`LearningKnowledgeBase.flakiness`."""
        return self.learning.flakiness(fingerprint, window)

    def trend(self, fingerprint: str, window: float = 86400.0,
              buckets: int = 12) -> dict[str, Any]:
        """Rate-trend analysis for one error fingerprint — see
        :meth:`LearningKnowledgeBase.trend`."""
        return self.learning.trend(fingerprint, window, buckets)

    def flush(self) -> None:
        """Persist learned knowledge. Called on shutdown paths."""
        self.learning.flush()

    # -- internals ---------------------------------------------------------
    def _exception_chain(self, exc: BaseException) -> list[dict[str, str]]:
        chain: list[dict[str, str]] = []
        seen: set[int] = set()
        cur: BaseException | None = exc
        while cur is not None and id(cur) not in seen and len(chain) < 5:
            seen.add(id(cur))
            chain.append({
                "type": type(cur).__name__,
                "message": str(cur)[:500],
            })
            nxt = cur.__cause__ if cur.__cause__ is not None else cur.__context__
            cur = nxt if nxt is not cur else None
        return chain

    def _classify_by_type(self, exception: Exception) -> ErrorCategory:
        """Classify error by exception type."""
        # Framework errors already know their category via their code.
        if isinstance(exception, NoMoralsError):
            code = getattr(exception, "code", "")
            prefix = code.split(".")[0]
            return {
                "model": ErrorCategory.INTEGRATION,
                "tool": ErrorCategory.INTEGRATION,
                "task": ErrorCategory.INTEGRATION,
                "storage": ErrorCategory.DATABASE,
                "parse": ErrorCategory.PARSING,
                "media": ErrorCategory.RESOURCE,
                "config": ErrorCategory.CONFIGURATION,
                "validation": ErrorCategory.VALIDATION,
                "policy": ErrorCategory.PERMISSION,
                "budget": ErrorCategory.RESOURCE,
                "network": ErrorCategory.NETWORK,
                "auth": ErrorCategory.AUTH,
                "quota": ErrorCategory.RATE_LIMIT,
                "dependency": ErrorCategory.INTEGRATION,
                "state": ErrorCategory.CONFIGURATION,
                "rate": ErrorCategory.RATE_LIMIT,
                "circuit": ErrorCategory.INTEGRATION,
                "eventbus": ErrorCategory.INTEGRATION,
                "resolve": ErrorCategory.VALIDATION,
            }.get(prefix, ErrorCategory.UNKNOWN)

        exc_name = type(exception).__name__.lower()

        # Network errors
        if any(name in exc_name for name in ["connection", "socket", "http", "urllib"]):
            return ErrorCategory.NETWORK

        # Timeout errors
        if "timeout" in exc_name:
            return ErrorCategory.TIMEOUT

        # Permission errors
        if "permission" in exc_name or "access" in exc_name:
            return ErrorCategory.PERMISSION

        # File/resource errors
        if any(name in exc_name for name in ["file", "io", "os"]):
            return ErrorCategory.RESOURCE

        # Parsing errors
        if any(name in exc_name for name in ["json", "xml", "parse", "decode"]):
            return ErrorCategory.PARSING

        # Validation errors
        if any(name in exc_name for name in ["value", "type", "key", "attribute"]):
            return ErrorCategory.VALIDATION

        # Database errors
        if any(name in exc_name for name in ["sql", "database", "sqlite"]):
            return ErrorCategory.DATABASE

        # Auth errors
        if any(name in exc_name for name in ["auth", "credential", "token"]):
            return ErrorCategory.AUTH

        return ErrorCategory.UNKNOWN

    def _estimate_severity(self, exception: Exception) -> ErrorSeverity:
        """Estimate error severity."""
        exc_name = type(exception).__name__.lower()
        exc_message = str(exception).lower()

        # Critical errors
        if any(term in exc_message for term in ["out of memory", "disk full", "data loss"]):
            return ErrorSeverity.CRITICAL

        # High severity
        if any(term in exc_name for term in ["auth", "permission", "crypto"]):
            return ErrorSeverity.HIGH

        # Medium severity
        if any(term in exc_name for term in ["timeout", "rate", "parse"]):
            return ErrorSeverity.MEDIUM

        # Default to medium
        return ErrorSeverity.MEDIUM

    def _infer_root_cause(self, exception: Exception, context: dict[str, Any],
                          chained: list[dict[str, str]]) -> str:
        """Infer root cause from exception, chain, and context."""
        exc_type = type(exception).__name__
        exc_message = str(exception)

        # The innermost chained exception is usually the true root cause.
        if len(chained) > 1:
            inner = chained[-1]
            cause = (f"{exc_type}: {exc_message} "
                     f"(root: {inner['type']}: {inner['message'][:200]})")
        else:
            cause = f"{exc_type}: {exc_message}"

        if context:
            operation = context.get("operation", "unknown operation")
            cause = f"Failed during {operation}. {cause}"

        return cause

    def _generate_explanation(
        self,
        match: dict[str, Any],
        exception: Exception,
        context: dict[str, Any],
    ) -> str:
        """Generate human-readable explanation for known error."""
        operation = context.get("operation", "operation")
        service = context.get("service", "service")

        explanation = f"The {operation} failed because {match['root_cause'].lower()}"

        if service:
            explanation += f" while connecting to {service}"

        return explanation + "."

    def _generate_explanation_for_unknown(
        self,
        exception: Exception,
        context: dict[str, Any],
    ) -> str:
        """Generate explanation for unknown error."""
        operation = context.get("operation", "operation")
        exc_type = type(exception).__name__

        return f"An unexpected error occurred during {operation}: {exc_type}."

    def _suggest_fix_for_unknown(
        self,
        exception: Exception,
        category: ErrorCategory,
    ) -> str:
        """Suggest fix for unknown error."""
        if category == ErrorCategory.NETWORK:
            return "Check your internet connection and try again."
        elif category == ErrorCategory.AUTH:
            return "Verify your credentials are correct and not expired."
        elif category == ErrorCategory.RESOURCE:
            return "Check available system resources (disk space, memory)."
        elif category == ErrorCategory.TIMEOUT:
            return "The operation took too long. Try again or increase timeout."
        else:
            return "This is an unexpected error. Check the logs for details or try again."

    def _is_retryable(self, exception: Exception) -> bool:
        """Decide retryable-ness from the shared error hierarchy (one dialect).

        Framework errors carry their own ``retryable`` flag; anything else is
        classified into the hierarchy, which decides.
        """
        if isinstance(exception, NoMoralsError):
            return exception.retryable
        try:
            return classify(exception).retryable
        except Exception:  # noqa: BLE001 - classification never breaks analysis
            return False

    def _extract_retry_after(self, exception: Exception) -> Optional[float]:
        """Shared ``retry_after`` extraction (see :mod:`nomorals.core.errors`)."""
        try:
            return retry_after_of(exception)
        except Exception:  # noqa: BLE001
            return None

    def _log_analysis(self, analysis: ErrorAnalysis) -> None:
        """Log error analysis."""
        log_method = {
            ErrorSeverity.LOW: _log.info,
            ErrorSeverity.MEDIUM: _log.warning,
            ErrorSeverity.HIGH: _log.error,
            ErrorSeverity.CRITICAL: _log.critical,
        }.get(analysis.severity, _log.error)

        log_method(
            f"Error Analysis: [{analysis.category.value}] {analysis.exception_type}: "
            f"{analysis.explanation} (fp={analysis.fingerprint or 'n/a'} "
            f"seen={analysis.group_count}x)"
        )

        if analysis.suggested_fix:
            _log.info(f"Suggested fix: {analysis.suggested_fix}")

        if analysis.needs_escalation:
            _log.warning("No known fix for %s — escalation recommended",
                         analysis.exception_type)


# ── Decorator for easy error catching ──────────────────────────────────────────

T = TypeVar("T")


def catch_and_analyze(
    context: dict[str, Any] | None = None,
    *,
    reraise: bool = False,
    default: Any = None,
) -> Callable:
    """Decorator that catches exceptions and provides analysis.

    Args:
        context: Context to include in analysis
        reraise: If True, re-raise exception after analysis
        default: Default return value on error (if not reraising)

    Returns:
        Decorator function

    Example:
        @catch_and_analyze(context={"operation": "send_email"})
        async def send_email(to, subject, body):
            ...
    """
    ei = ErrorIntelligence()

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Add function info to context
                full_context = {
                    "function": func.__name__,
                    "module": func.__module__,
                    **(context or {}),
                }

                analysis = ei.analyze(e, context=full_context)

                # Store analysis on exception for later access
                e.error_analysis = analysis

                if reraise:
                    raise

                return default

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                full_context = {
                    "function": func.__name__,
                    "module": func.__module__,
                    **(context or {}),
                }

                analysis = ei.analyze(e, context=full_context)
                e.error_analysis = analysis

                if reraise:
                    raise

                return default

        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# ── Global instance for convenience ────────────────────────────────────────────

_global_ei = ErrorIntelligence()


def analyze_error(
    exception: Exception,
    *,
    context: dict[str, Any] | None = None,
) -> ErrorAnalysis:
    """Convenience function to analyze an error.

    Args:
        exception: Exception to analyze
        context: Additional context

    Returns:
        ErrorAnalysis object
    """
    return _global_ei.analyze(exception, context=context)


def add_breadcrumb(
    message: str,
    category: str = "default",
    data: dict[str, Any] | None = None,
) -> None:
    """Leave a breadcrumb on the global intelligence instance."""
    _global_ei.add_breadcrumb(message, category=category, data=data)
