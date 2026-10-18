"""Shared result contract: one shape for every completed operation.

Machine truth stays separate from prose. Every stage — tool call,
mission step, delivery — reports the same fields so callers never have
to guess whether "done" means ran, verified, delivered, or merely
returned without raising:

* ``succeeded`` — ran successfully (not necessarily verified).
* ``verified`` — ran AND read-back / verification passed.
* ``partial`` — some parts completed; see ``recovery`` for the rest.
* ``waiting`` — blocked on something external.
* ``stopped`` — cancelled before completion.
* ``outcome_unknown`` — the operation may still have happened
  (timeout); never blind-replay this.
* ``delivered`` — reached the platform (receipt held).
* ``failed`` — did not complete; see ``error``.

``completeness`` says whether the payload is the whole truth:
``complete`` | ``truncated`` (remainder retrievable via ``remainder``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["ExecutionResult", "ExecutionStatus"]

#: Canonical execution statuses (see module docstring).
ExecutionStatus = (
    "succeeded", "verified", "partial", "waiting", "stopped",
    "outcome_unknown", "delivered", "failed",
)


@dataclass
class ExecutionResult:
    """One operation's verdict, in the shared contract shape."""

    status: str
    summary: str = ""
    data: Any = None
    artifacts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str = ""
    completeness: str = "complete"
    remainder: str = ""
    recovery: list[str] = field(default_factory=list)
    delivery_state: str = ""
    request_id: str = ""
    run_id: str = ""
    tool: str = ""

    # -- constructors ------------------------------------------------------
    @classmethod
    def succeeded(cls, summary: str = "", **kw: Any) -> "ExecutionResult":
        return cls(status="succeeded", summary=summary, **kw)

    @classmethod
    def verified(cls, summary: str = "", **kw: Any) -> "ExecutionResult":
        return cls(status="verified", summary=summary, **kw)

    @classmethod
    def partial(cls, summary: str = "", **kw: Any) -> "ExecutionResult":
        return cls(status="partial", summary=summary, **kw)

    @classmethod
    def unknown(cls, summary: str = "", **kw: Any) -> "ExecutionResult":
        return cls(status="outcome_unknown", summary=summary, **kw)

    @classmethod
    def failed(cls, summary: str = "", **kw: Any) -> "ExecutionResult":
        return cls(status="failed", summary=summary, **kw)

    @classmethod
    def from_outcome(cls, outcome: Any, *, tool: str = "",
                     summary: str = "") -> "ExecutionResult":
        """Convert a registry ``Outcome`` (Ok/Err) to the contract shape."""
        if getattr(outcome, "ok", False):
            return cls(status="succeeded", tool=tool,
                       summary=summary or f"{tool or 'operation'} succeeded",
                       data=getattr(outcome, "value", None))
        err = getattr(outcome, "error", outcome)
        msg = getattr(err, "message", str(err)) if err is not None \
            else "unknown error"
        return cls(status="failed", tool=tool,
                   summary=summary or f"{tool or 'operation'} failed",
                   error=str(msg))

    # -- shape -------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "summary": self.summary,
            "data": self.data,
            "artifacts": list(self.artifacts),
            "warnings": list(self.warnings),
            "error": self.error,
            "completeness": self.completeness,
            "remainder": self.remainder,
            "recovery": list(self.recovery),
            "delivery_state": self.delivery_state,
            "request_id": self.request_id,
            "run_id": self.run_id,
            "tool": self.tool,
        }

    @property
    def ok(self) -> bool:
        return self.status in ("succeeded", "verified", "delivered")
