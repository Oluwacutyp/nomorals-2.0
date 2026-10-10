"""Mesh errors.

Every error carries a machine-readable ``code`` (gRPC-style canonical
names) and a ``retryable`` hint, plus a :meth:`MeshError.to_dict` wire
form in the spirit of RFC 9457 problem details so the hub and its
clients speak the same error language.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "MeshError",
    "NodeUnknown",
    "NodeSuspect",
    "TransportError",
    "HubUnreachable",
    "AuthError",
    "CircuitOpen",
    "TaskNotFound",
    "TaskExpired",
    "TaskCancelled",
    "PayloadTooLarge",
]


class MeshError(Exception):
    """Base for all mesh errors.

    ``code`` is a stable, machine-readable identifier (e.g.
    ``"node_unknown"``). ``retryable`` tells the caller whether retrying
    the same operation later could plausibly succeed — network blips
    yes, bad input no.
    """

    code = "mesh_error"
    retryable = False

    def __init__(self, message: str = "", *, detail: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.detail = dict(detail or {})

    def to_dict(self) -> dict[str, Any]:
        """Wire form (RFC 9457-flavoured): code, message, retry hint."""
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "detail": dict(self.detail),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MeshError":
        """Rebuild the most specific known error for a wire code."""
        code = str(data.get("code") or "")
        err_cls = _CODE_REGISTRY.get(code, MeshError)
        return err_cls(
            str(data.get("message") or ""),
            detail=dict(data.get("detail") or {}),
        )

    def __str__(self) -> str:
        base = self.message or self.code
        return f"[{self.code}] {base}"


class NodeUnknown(MeshError):
    """A node id was referenced that is not registered."""

    code = "node_unknown"


class NodeSuspect(MeshError):
    """A node missed heartbeats and is suspected down, but not yet gone.

    Raised by capability-aware dispatch when every matching node is
    suspect — the caller can choose to wait, retry, or fall back to
    broadcast instead of silently queueing work nobody will pick up.
    """

    code = "node_suspect"
    retryable = True


class TransportError(MeshError):
    """The transport failed to deliver or fetch."""

    code = "transport_error"
    retryable = True


class HubUnreachable(TransportError):
    """The hub could not be reached after all retries."""

    code = "hub_unreachable"
    retryable = True


class AuthError(TransportError):
    """The hub rejected our credentials.

    A TransportError subclass for backward compatibility (existing code
    catches TransportError around hub calls), but ``retryable`` is False:
    retrying with the same token will never succeed — this is a
    configuration problem, not a blip.
    """

    code = "auth_rejected"
    retryable = False


class CircuitOpen(TransportError):
    """The hub circuit breaker is open: calls fail fast without touching
    the network, giving the hub room to recover."""

    code = "circuit_open"
    retryable = True


class TaskNotFound(MeshError):
    """No task exists with that job id."""

    code = "task_not_found"


class TaskExpired(MeshError):
    """The task was never picked up before its schedule-to-start deadline."""

    code = "task_expired"


class TaskCancelled(MeshError):
    """The task was cancelled before it could run."""

    code = "task_cancelled"


class PayloadTooLarge(MeshError, ValueError):
    """A task payload exceeded the envelope size limit.

    Also a ValueError: the historical contract for bad dispatch input
    was ValueError, and callers rely on it.
    """

    code = "payload_too_large"


_CODE_REGISTRY: dict[str, type[MeshError]] = {}


def _register_errors() -> None:
    for _name, _obj in list(globals().items()):
        if (
            isinstance(_obj, type)
            and issubclass(_obj, MeshError)
            and _obj is not MeshError
        ):
            _CODE_REGISTRY[_obj.code] = _obj


_register_errors()
