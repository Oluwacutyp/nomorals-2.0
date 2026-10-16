"""Tier 0 promotion registry with rollback.

Every Tier 0 promotion (a wiring change that promotes an existing
subsystem to a new canonical role) is recorded here with a rollback
function that restores the pre-promotion state.  Rollback is
idempotent: calling it twice runs the rollback function once.

This module is stdlib-only so any layer may import it.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

__all__ = [
    "record_promotion",
    "rollback",
    "rollback_all",
    "promotions",
    "promotion_status",
    "clear_registry",
]

_registry: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def record_promotion(
    name: str,
    rollback_fn: Callable[[], None],
    *,
    description: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record (or re-record) a promotion with its rollback function.

    Re-recording a name replaces the rollback function and resets the
    status to ``active`` — a re-promotion after a rollback is a fresh
    promotion, not a resurrection of the old one.
    """
    if not name:
        raise ValueError("promotion name is required")
    if not callable(rollback_fn):
        raise TypeError("rollback_fn must be callable")
    with _lock:
        record = {
            "name": name,
            "description": description,
            "metadata": dict(metadata or {}),
            "status": "active",
            "recorded_at": time.time(),
            "rolled_back_at": None,
            "rollback_error": None,
        }
        _registry[name] = record
        # Store the callable outside the serializable record body.
        _registry[name]["_rollback_fn"] = rollback_fn
        return _public(record)


def rollback(name: str) -> dict[str, Any]:
    """Roll a promotion back. Idempotent; returns the promotion record.

    The rollback function runs exactly once. If it raises, the error is
    recorded on the promotion and re-raised — the caller knows the
    rollback did not complete.
    """
    with _lock:
        record = _registry.get(name)
        if record is None:
            raise KeyError(f"unknown promotion {name!r}")
        if record["status"] == "rolled_back":
            return _public(record)
        fn = record["_rollback_fn"]
    # Run outside the lock: a rollback fn may record another promotion.
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
        with _lock:
            record["rollback_error"] = f"{type(exc).__name__}: {exc}"
        raise
    with _lock:
        record["status"] = "rolled_back"
        record["rolled_back_at"] = time.time()
        return _public(record)


def rollback_all() -> list[dict[str, Any]]:
    """Roll every still-active promotion back, in record order."""
    with _lock:
        names = [n for n, r in _registry.items()
                 if r["status"] == "active"]
    out = []
    for name in names:
        try:
            out.append(rollback(name))
        except Exception:  # noqa: BLE001 - keep rolling the rest back
            out.append(_public(_registry[name]))
    return out


def promotions() -> list[dict[str, Any]]:
    """All recorded promotions, in record order."""
    with _lock:
        return [_public(r) for r in _registry.values()]


def promotion_status(name: str) -> dict[str, Any] | None:
    """One promotion's record, or None when never recorded."""
    with _lock:
        record = _registry.get(name)
        return _public(record) if record is not None else None


def clear_registry() -> None:
    """Drop every record. Test support only — never used in production."""
    with _lock:
        _registry.clear()


def _public(record: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in record.items() if not k.startswith("_")}
