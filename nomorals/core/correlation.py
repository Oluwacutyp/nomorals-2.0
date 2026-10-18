"""Request/run correlation: one id family across a whole interaction.

A request (inbound message / CLI invocation) starts a run; every tool
call, artifact, delivery, and timeline event inside it carries the same
``request_id`` / ``run_id`` / ``mission_id`` triple so replay, receipts,
and diagnostics can join stages that previously only shared a timestamp.

Implemented with :mod:`contextvars` so concurrent chats on different
threads/tasks never see each other's ids, and so a value set for one
turn does not leak into the next.
"""

from __future__ import annotations

import contextvars
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

__all__ = [
    "correlation_context",
    "current_correlation",
    "new_request_id",
    "new_run_id",
    "set_correlation",
]

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "nomorals_request_id", default="")
_run_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "nomorals_run_id", default="")
_mission_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "nomorals_mission_id", default="")


def new_request_id() -> str:
    return f"req-{uuid.uuid4().hex[:16]}"


def new_run_id() -> str:
    return f"run-{uuid.uuid4().hex[:16]}"


def current_correlation() -> dict[str, str]:
    """The correlation triple for this context (``""`` when unset)."""
    return {
        "request_id": _request_id.get(),
        "run_id": _run_id.get(),
        "mission_id": _mission_id.get(),
    }


def set_correlation(*, request_id: str = "", run_id: str = "",
                    mission_id: str = "") -> dict[str, Any]:
    """Set the triple; returns tokens for :meth:`contextvars.Token` reset."""
    tokens: dict[str, Any] = {}
    if request_id:
        tokens["request_id"] = _request_id.set(request_id)
    if run_id:
        tokens["run_id"] = _run_id.set(run_id)
    if mission_id:
        tokens["mission_id"] = _mission_id.set(mission_id)
    return tokens


@contextmanager
def correlation_context(*, request_id: str = "", run_id: str = "",
                        mission_id: str = "") -> Iterator[dict[str, str]]:
    """Scope a correlation triple; restores the previous values on exit."""
    tokens: list[tuple[contextvars.ContextVar[str],
                       contextvars.Token[str]]] = []
    if request_id:
        tokens.append((_request_id, _request_id.set(request_id)))
    if run_id:
        tokens.append((_run_id, _run_id.set(run_id)))
    if mission_id:
        tokens.append((_mission_id, _mission_id.set(mission_id)))
    try:
        yield current_correlation()
    finally:
        for var, token in reversed(tokens):
            var.reset(token)
