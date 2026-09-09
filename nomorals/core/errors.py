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

from typing import Any


class NoMoralsError(Exception):
    """Base class for all framework errors."""

    code: str = "error"
    retryable: bool = False

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

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}(code={self.code!r}, message={self.message!r})"


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
    if isinstance(exc, (ValueError, TypeError)):
        return ValidationError(text)
    if "rate limit" in lowered or "429" in lowered:
        return RateLimited(text)
    if "timed out" in lowered or "timeout" in lowered:
        return TimeoutError_(text, retryable=True)
    return NoMoralsError(text, code=f"unhandled.{type(exc).__name__.lower()}")
