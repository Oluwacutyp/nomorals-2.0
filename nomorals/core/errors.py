"""Typed error hierarchy.

Design rule: exceptions signal *programmer error* or *unrecoverable state*.
Expected failure modes (network down, model overloaded, file missing) travel as
:class:`~nomorals.core.result.Err` values so callers cannot accidentally ignore them.

Every error carries:
  * ``code``    — stable machine-readable string, safe to match on across versions
  * ``message`` — human-readable, safe to log
  * ``details`` — structured payload
  * ``retryable`` — whether a retry with backoff could plausibly succeed
"""

from __future__ import annotations

import inspect
import json
import re
from typing import Any


# ── HTTP status helpers ──────────────────────────────────────────────────────
#: Status codes worth matching out of a free-text error message, for errors
#: that did not come through :func:`nomorals.core.http.http_error` (which
#: sets ``details["http_status"]`` directly).
_STATUS_RE = re.compile(r"\b(400|401|403|404|408|425|429|5\d\d)\b")


def http_status_of(exc: BaseException) -> int | None:
    """Best-effort HTTP status for an error, or ``None`` when unknown.

    Prefers ``details["http_status"]`` (set by the HTTP layer); falls back
    to the first status-like code in the message, which covers errors
    built from raw exception strings (tests, third-party clients).
    """
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        status = details.get("http_status")
        if isinstance(status, bool):
            pass  # a bool is not a status; fall through to the message
        elif isinstance(status, int):
            return status
        elif isinstance(status, str) and status.isdigit():
            return int(status)
    text = str(getattr(exc, "message", "") or exc)
    match = _STATUS_RE.search(text)
    return int(match.group(1)) if match else None


def is_auth_error(exc: BaseException) -> bool:
    """401/403 — the key is bad or lacks access.  Terminal: retrying with
    the same key, or swapping models under it, can never succeed."""
    status = http_status_of(exc)
    if status is not None:
        return status in (401, 403)
    lowered = str(getattr(exc, "message", "") or exc).lower()
    return "unauthorized" in lowered or "forbidden" in lowered


def is_not_found_error(exc: BaseException) -> bool:
    """404 — the URL or model id does not exist.  Terminal for this id:
    retrying the same id is pointless, but a *different* model id may work."""
    return http_status_of(exc) == 404


def is_rate_limited_error(exc: BaseException) -> bool:
    """429 — back off (honoring ``retry_after``) and retry."""
    return isinstance(exc, RateLimited) or http_status_of(exc) == 429


def is_server_error(exc: BaseException) -> bool:
    """5xx / 408 / 425 — the server stumbled; back off and retry."""
    status = http_status_of(exc)
    if status is None:
        return isinstance(exc, (ProviderUnavailable, ProviderError)) and getattr(
            exc, "retryable", False
        )
    return status in (408, 425) or 500 <= status <= 599


def retry_after_of(exc: BaseException) -> float | None:
    """Best-effort ``Retry-After`` seconds for an error, or ``None``.

    Single shared helper so the retry loop, the rate limiter, and the error
    intelligence all honor the same signal instead of each inventing their
    own extraction. Prefers an explicit ``retry_after`` attribute, then a
    ``Retry-After`` response header, then ``details["retry_after"]``.
    """
    direct = getattr(exc, "retry_after", None)
    if isinstance(direct, (int, float)) and direct >= 0:
        return float(direct)
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        try:
            raw = headers.get("Retry-After") or headers.get("retry-after")
        except Exception:  # noqa: BLE001 - exotic header mappings
            raw = None
        if raw is not None:
            try:
                return max(0.0, float(raw))
            except (TypeError, ValueError):
                pass
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        raw = details.get("retry_after")
        if isinstance(raw, (int, float)) and raw >= 0:
            return float(raw)
    return None


class NoMoralsError(Exception):
    """Base class for all framework errors."""

    code: str = "error"
    retryable: bool = False

    #: code → error class, populated automatically by __init_subclass__.
    _registry: dict[str, type["NoMoralsError"]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        code = getattr(cls, "code", "")
        if code and code != "error":
            NoMoralsError._registry[code] = cls

    def __init__(
        self,
        message: str = "",
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message or self.__class__.__name__)
        self.message = message or self.__class__.__name__
        if code is not None:
            self.code = code
        if retryable is not None:
            self.retryable = retryable
        self.details: dict[str, Any] = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "retryable": self.retryable,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NoMoralsError":
        """Rebuild an error from :meth:`to_dict` output. Never raises.

        Resolves the concrete class by ``code`` (then by ``type`` name);
        falls back to a plain :class:`NoMoralsError` when the class has a
        custom constructor that the generic path cannot satisfy.
        """
        try:
            code = data.get("code", "") or ""
            type_name = data.get("type", "") or ""
            klass: type[NoMoralsError] = NoMoralsError._registry.get(code) or NoMoralsError
            if klass is NoMoralsError and type_name:
                candidate = globals().get(type_name)
                if isinstance(candidate, type) and issubclass(candidate, NoMoralsError):
                    klass = candidate
            message = data.get("message", "") or klass.__name__
            details = data.get("details") or {}
            retryable = data.get("retryable")
            # Forward structured extras (retry_after, field, role, ...) that the
            # concrete __init__ actually accepts — otherwise reconstruction
            # silently drops them (e.g. RateLimited.retry_after).
            fwd: dict[str, Any] = {}
            try:
                params = set(inspect.signature(klass.__init__).parameters)
            except (TypeError, ValueError):
                params = set()
            for attr in ("field", "capability", "actor", "role", "tool",
                         "reason", "kind", "retry_after", "dependency"):
                if attr in details and attr in params:
                    fwd[attr] = details[attr]
            try:
                err = klass(message, code=code or None,
                            details=dict(details),
                            retryable=bool(retryable) if retryable is not None else None,
                            **fwd)
            except TypeError:
                # custom __init__ (e.g. AmbiguousRef) — generic reconstruction
                err = NoMoralsError(message, code=code or "error",
                                    details=dict(details),
                                    retryable=bool(retryable) if retryable is not None else None)
            # restore structured extras that subclasses setdefault into details
            for attr in ("field", "capability", "actor", "role", "tool",
                         "reason", "kind", "retry_after", "breaker"):
                if attr in details and not hasattr(err, attr):
                    try:
                        setattr(err, attr, details[attr])
                    except Exception:  # noqa: BLE001 - best effort
                        pass
            return err
        except Exception:  # noqa: BLE001 - from_dict never raises
            return NoMoralsError(str(data)[:200], code="error.deser")

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}(code={self.code!r}, message={self.message!r})"


def resolve_error(code: str) -> type[NoMoralsError] | None:
    """Look up the error class registered for ``code`` (or ``None``)."""
    return NoMoralsError._registry.get(code)


def error_codes() -> dict[str, str]:
    """All registered codes → class names. Useful for docs and dashboards."""
    return {code: klass.__name__ for code, klass in sorted(NoMoralsError._registry.items())}


# ── Configuration ──────────────────────────────────────────────────────────────
class ConfigError(NoMoralsError):
    code = "config.invalid"


class ValidationError(NoMoralsError):
    code = "validation.failed"

    def __init__(self, message: str = "", *, field: str | None = None, **kw: Any) -> None:
        super().__init__(message, **kw)
        self.field = field
        if field:
            self.details.setdefault("field", field)


# ── Storage ────────────────────────────────────────────────────────────────────
class StorageError(NoMoralsError):
    code = "storage.error"
    retryable = True


class MigrationError(StorageError):
    code = "storage.migration"
    retryable = False


class NotFound(NoMoralsError):
    code = "storage.not_found"
    retryable = False


class AmbiguousRef(NoMoralsError):
    """A short id/name reference matched more than one entity.

    Raised by id-prefix resolvers instead of guessing: the chat handler
    catches it and asks the user to disambiguate. ``candidates`` is a list
    of ``(id, human label)`` pairs; ``min_prefix_len`` is the smallest id
    prefix length that identifies every candidate uniquely.
    """

    code = "resolve.ambiguous"
    retryable = False

    def __init__(
        self,
        ref: str,
        candidates: list[tuple[str, str]],
        min_prefix_len: int,
        *,
        entity: str = "entity",
        hint: str = "",
    ) -> None:
        self.ref = ref
        self.candidates = list(candidates)
        self.min_prefix_len = min_prefix_len
        self.entity = entity
        self.hint = hint
        super().__init__(self._message())

    def _message(self) -> str:
        lines = [f"{self.ref!r} is ambiguous — matches "
                 f"{len(self.candidates)} {self.entity}s:"]
        for cid, label in self.candidates[:8]:
            lines.append(f"  · {cid} — {label}"[:120])
        if len(self.candidates) > 8:
            lines.append(f"  … +{len(self.candidates) - 8} more")
        lines.append(
            f"use a longer id prefix (at least {self.min_prefix_len} "
            f"characters) to pick one.")
        if self.hint:
            lines.append(self.hint)
        return "\n".join(lines)


class ConstraintViolation(StorageError):
    code = "storage.constraint"
    retryable = False


# ── Models / providers ─────────────────────────────────────────────────────────
class ModelError(NoMoralsError):
    code = "model.error"


class ProviderError(ModelError):
    code = "model.provider"
    retryable = True


class ProviderUnavailable(ProviderError):
    code = "model.provider.unavailable"
    retryable = True


class RateLimited(ProviderError):
    code = "model.provider.rate_limited"
    retryable = True

    def __init__(self, message: str = "rate limited", *, retry_after: float = 1.0, **kw: Any):
        super().__init__(message, **kw)
        self.retry_after = retry_after
        self.details.setdefault("retry_after", retry_after)


class ContextOverflow(ModelError):
    code = "model.context_overflow"
    retryable = False


class DownloadError(ModelError):
    code = "model.download"
    retryable = True


# ── Network / auth / quotas ────────────────────────────────────────────────────
class NetworkError(NoMoralsError):
    """Generic transport failure not tied to a model provider.

    DNS failures, refused connections, TLS errors on non-model traffic.
    Distinct from :class:`ProviderUnavailable`, which is model-provider scoped.
    """

    code = "network.error"
    retryable = True


class AuthError(NoMoralsError):
    """Credentials are bad, expired, or missing.

    Terminal for these credentials: retrying the same secret can never
    succeed — refresh it first, then retry.
    """

    code = "auth.failed"
    retryable = False


class QuotaExceeded(NoMoralsError):
    """A quota (not a rate window) is exhausted — monthly tokens, seats, storage.

    Unlike :class:`RateLimited`, waiting a few seconds will not help.
    ``retry_after`` names the reset time when it is known.
    """

    code = "quota.exceeded"
    retryable = True

    def __init__(self, message: str = "quota exceeded", *, retry_after: float | None = None,
                 **kw: Any) -> None:
        super().__init__(message, **kw)
        self.retry_after = retry_after
        if retry_after is not None:
            self.details.setdefault("retry_after", retry_after)


class DependencyError(NoMoralsError):
    """A downstream service this operation depends on failed."""

    code = "dependency.failed"
    retryable = True

    def __init__(self, message: str = "", *, dependency: str | None = None, **kw: Any) -> None:
        super().__init__(message, **kw)
        self.dependency = dependency
        if dependency:
            self.details.setdefault("dependency", dependency)


# ── Tools / capabilities ───────────────────────────────────────────────────────
class ToolError(NoMoralsError):
    code = "tool.error"


class ToolNotFound(ToolError):
    code = "tool.not_found"
    retryable = False


class CapabilityDenied(NoMoralsError):
    """A tool call was blocked by the capability policy."""

    code = "policy.denied"
    retryable = False

    def __init__(
        self,
        message: str = "capability denied",
        *,
        capability: str | None = None,
        actor: str | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(message, **kw)
        self.capability = capability
        self.actor = actor
        if capability:
            self.details.setdefault("capability", capability)
        if actor:
            self.details.setdefault("actor", actor)


class ToolDenied(ToolError):
    """A tool call was blocked by a role's allowlist.

    Structured so role agents can adapt (the denial names the role, the
    tool, and the reason) and so the failure ledger can categorize it
    distinctly from capability-policy denials.
    """

    code = "tool.denied"
    retryable = False

    def __init__(
        self,
        message: str = "tool denied by role allowlist",
        *,
        role: str | None = None,
        tool: str | None = None,
        reason: str = "",
        **kw: Any,
    ) -> None:
        super().__init__(message, **kw)
        self.role = role
        self.tool = tool
        self.reason = reason
        if role:
            self.details.setdefault("role", role)
        if tool:
            self.details.setdefault("tool", tool)
        if reason:
            self.details.setdefault("reason", reason)


class SandboxError(ToolError):
    code = "tool.sandbox"


class TimeoutError_(ToolError):
    """Named with a trailing underscore to avoid shadowing the builtin."""

    code = "tool.timeout"
    retryable = True


# ── Agents / tasks ─────────────────────────────────────────────────────────────
class TaskCancelled(NoMoralsError):
    code = "task.cancelled"
    retryable = False


class TaskFailed(NoMoralsError):
    code = "task.failed"
    retryable = False


class BudgetExceeded(NoMoralsError):
    code = "budget.exceeded"
    retryable = False

    def __init__(self, message: str = "budget exceeded", *, kind: str = "", **kw: Any) -> None:
        super().__init__(message, **kw)
        self.kind = kind
        if kind:
            self.details.setdefault("kind", kind)


class DeadlineExceeded(TaskCancelled):
    code = "task.deadline"
    retryable = False


class StateError(NoMoralsError):
    """Invalid internal state — a state machine received an event it cannot
    handle in its current state, or an invariant was violated."""

    code = "state.invalid"
    retryable = False


# ── Media / parsing ────────────────────────────────────────────────────────────
class ParseError(NoMoralsError):
    code = "parse.error"
    retryable = False


class UnsupportedFormat(ParseError):
    code = "parse.unsupported"
    retryable = False


class MediaError(NoMoralsError):
    code = "media.error"
    retryable = True


def classify(exc: BaseException) -> NoMoralsError:
    """Wrap an arbitrary exception in the framework hierarchy.

    Idempotent: a :class:`NoMoralsError` passes through unchanged.
    """
    if isinstance(exc, NoMoralsError):
        return exc
    text = str(exc) or type(exc).__name__
    lowered = text.lower()
    if isinstance(exc, (TimeoutError,)):
        return TimeoutError_(text, retryable=True)
    if isinstance(exc, (ConnectionError,)):
        return ProviderUnavailable(text, retryable=True)
    if isinstance(exc, (PermissionError,)):
        return CapabilityDenied(text)
    if isinstance(exc, (FileNotFoundError,)):
        return NotFound(text)
    if isinstance(exc, json.JSONDecodeError):
        return ParseError(text)
    if isinstance(exc, (ValueError, TypeError)):
        return ValidationError(text)
    if "rate limit" in lowered or "429" in lowered:
        return RateLimited(text)
    if "timed out" in lowered or "timeout" in lowered:
        return TimeoutError_(text, retryable=True)
    return NoMoralsError(text, code=f"unhandled.{type(exc).__name__.lower()}")
