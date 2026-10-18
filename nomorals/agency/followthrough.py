"""Planner follow-through: waiting is a state, not a spin (audit §5).

Plans distinguish ACTIONS from WAITING. "Wait for client reply"
needs a correlation ID, a reply trigger, a next review time, and an
expiry — not a worker spinning in a loop.

Before irreversible external effects, the planner consults standing
authority (injected ``authority`` callable — never widened here:
no new spending authority, no invented qualifications, no commitments
accepted just because a scheduler fired). Where authority or facts
are unresolved, executable prep work proceeds and only the specific
decision surfaces.

Every stop records WHY it stopped (audit §5): diminishing gain,
expired, poor net value, no execution path — not just why it started.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = ["WaitRecord", "PlanTracker", "STOP_REASONS"]

#: Valid stop reasons — the planner must record one, not just stop.
STOP_REASONS = (
    "delivered", "diminishing_gain", "expired", "poor_net_value",
    "no_execution_path", "blocked", "superseded", "owner_cancelled",
)


@dataclass
class WaitRecord:
    correlation_id: str
    trigger: str  # e.g. "client_reply", "price_below:85", "forecast_update"
    next_review_at: float
    expires_at: float
    initiative_id: str = ""
    created_at: float = 0.0

    def is_expired(self, now: float) -> bool:
        return now >= self.expires_at

    def is_due(self, now: float) -> bool:
        return now >= self.next_review_at


class PlanTracker:
    """Waits + stale-plan cancellation + stop reasons + authority gate."""

    def __init__(self, *,
                 authority: Callable[[str, dict[str, Any]], bool] | None = None
                 ) -> None:
        self._waits: dict[str, WaitRecord] = {}
        self._stops: list[dict[str, Any]] = []
        self._authority = authority or (lambda _action, _ctx: False)

    # ── waiting ────────────────────────────────────────────────────
    def wait_for(self, record: WaitRecord, *, now: float | None = None) -> WaitRecord:
        record.created_at = time.time() if now is None else now
        self._waits[record.correlation_id] = record
        return record

    def resolve_wait(self, correlation_id: str, *, outcome: str) -> WaitRecord | None:
        rec = self._waits.pop(correlation_id, None)
        if rec is not None:
            self._stops.append({"wait": correlation_id, "outcome": outcome,
                                "ts": time.time()})
        return rec

    def due_waits(self, now: float) -> list[WaitRecord]:
        return [w for w in self._waits.values() if w.is_due(now)]

    def expired_waits(self, now: float) -> list[WaitRecord]:
        return [w for w in self._waits.values() if w.is_expired(now)]

    def expire_waits(self, now: float) -> list[WaitRecord]:
        expired = self.expired_waits(now)
        for w in expired:
            del self._waits[w.correlation_id]
            self._stops.append({"wait": w.correlation_id,
                                "outcome": "expired", "ts": now})
        return expired

    # ── stopping ───────────────────────────────────────────────────
    def stop(self, initiative_id: str, *, reason: str,
             detail: str = "") -> dict[str, Any]:
        if reason not in STOP_REASONS:
            raise ValueError(f"unknown stop reason {reason!r}; use {STOP_REASONS}")
        entry = {"initiative_id": initiative_id, "reason": reason,
                 "detail": detail, "ts": time.time()}
        self._stops.append(entry)
        return entry

    @property
    def stops(self) -> list[dict[str, Any]]:
        return list(self._stops)

    # ── authority gate ─────────────────────────────────────────────
    def may_act(self, action: str, ctx: dict[str, Any] | None = None) -> bool:
        """Standing authority check. Default-deny: no silent widening."""
        return bool(self._authority(action, ctx or {}))
