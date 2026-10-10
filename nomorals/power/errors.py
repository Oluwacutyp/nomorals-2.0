"""Power errors: structured, machine-readable failure taxonomy.

Every error carries a stable ``code`` and a ``retryable`` flag (the
gRPC/HTTP-status style contract: retryable errors are worth another
attempt after ``retry_after_s``; terminal ones are not).  Callers can
branch on ``code`` without string-matching messages.
"""

from __future__ import annotations

from typing import Any


class PowerError(Exception):
    """Base for all power errors."""

    code = "power_error"
    retryable = False

    def __init__(
        self,
        message: str = "",
        *,
        retry_after_s: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message or self.code)
        self.retry_after_s = retry_after_s
        self.details: dict[str, Any] = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "retryable": self.retryable,
            "message": str(self),
            "retry_after_s": self.retry_after_s,
            "details": dict(self.details),
        }


class TaskDeferredError(PowerError):
    """A task was deferred (requeued with backoff), not dropped.

    Raised by helpers that need a task *now* and cannot wait — e.g. a
    synchronous run path.  The task still exists in the queue and will
    flow when power allows.
    """

    code = "task_deferred"
    retryable = True

    def __init__(
        self,
        message: str = "",
        *,
        reason: str = "deferred",
        retry_after_s: float | None = None,
        job_id: str | None = None,
    ) -> None:
        super().__init__(
            message or f"task deferred ({reason})",
            retry_after_s=retry_after_s,
            details={"reason": reason, "job_id": job_id},
        )
        self.reason = reason
        self.job_id = job_id


class PowerConstrainedError(PowerError):
    """Power/thermal/degradation state forbids this work right now."""

    code = "power_constrained"
    retryable = True


class PowerBudgetExhaustedError(PowerError):
    """A subsystem's resource budget is exhausted."""

    code = "budget_exhausted"
    retryable = True


class PowerSamplerUnavailableError(PowerError):
    """No usable power sampler (and the caller required a real reading)."""

    code = "sampler_unavailable"
    retryable = True


class InvalidPowerSpecError(PowerError, ValueError):
    """Bad power_class / tier / constraint spec — terminal, don't retry.

    Subclasses :class:`ValueError` too, so existing ``assertRaises(
    ValueError)`` call sites keep working.
    """

    code = "invalid_power_spec"
    retryable = False
