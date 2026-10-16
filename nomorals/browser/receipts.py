"""Read-back verification + receipts + checkpoints.

* After every mutating action, assert the state changed as intended
  (cart count, sent-items list, DOM text). Never assume a click worked.
* Consequential acts get 3 independent signals: order-history read-back +
  confirmation email (injected mail check) + receipt (injected).
* One ledger row per web action: task, site, action, before/after hash,
  verified outcome, duration, rung, cost. Append-only.
* Checkpoints via injected durable store (storage/queue.py shape) so a
  long task resumes instead of restarting.
* Honest failure always — a failed verification is a failed action,
  never a fake success.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["ActionLedger", "Checkpoint", "LedgerRow", "ReadBackVerifier",
           "verify_consequential"]


@dataclass
class LedgerRow:
    task: str
    site: str
    action: dict[str, Any]
    before_hash: str
    after_hash: str
    verified: bool
    duration_ms: float
    rung: str = ""
    cost: float = 0.0
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"task": self.task, "site": self.site,
                "action": self.action, "before_hash": self.before_hash,
                "after_hash": self.after_hash, "verified": self.verified,
                "duration_ms": self.duration_ms, "rung": self.rung,
                "cost": self.cost, "at": self.at}


class ActionLedger:
    """Append-only ledger: one row per web action."""

    def __init__(self) -> None:
        self._rows: list[LedgerRow] = []

    def record(self, row: LedgerRow) -> LedgerRow:
        self._rows.append(row)
        return row

    def rows(self) -> list[LedgerRow]:
        return list(self._rows)

    def unverified(self) -> list[LedgerRow]:
        return [r for r in self._rows if not r.verified]

    def to_dict(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self._rows]


class ReadBackVerifier:
    """Assert state changed as intended after a mutating action."""

    def __init__(self, *,
                 read_state: Callable[[str], dict[str, Any]] | None = None) -> None:
        self._read_state = read_state

    def verify(self, action: dict[str, Any], *,
               expect: dict[str, Any], read: Callable[[], dict[str, Any]] | None = None) -> dict[str, Any]:
        """Compare expected state against a fresh read. Returns verdict."""
        reader = read or self._read_state
        if reader is None:
            return {"verified": False, "reason": "no state reader injected"}
        try:
            actual = reader(str(expect.get("key", "")))
        except Exception as exc:  # noqa: BLE001 - verification failure is data
            return {"verified": False, "reason": f"read-back failed: {exc}"}
        mismatches = {k: {"expected": v, "actual": actual.get(k)}
                      for k, v in expect.items()
                      if k != "key" and actual.get(k) != v}
        if mismatches:
            return {"verified": False, "reason": "state mismatch",
                    "mismatches": mismatches}
        return {"verified": True, "actual": actual}


def verify_consequential(*, order_history: Callable[[], bool],
                         confirmation_mail: Callable[[], bool],
                         receipt: Callable[[], bool]) -> dict[str, Any]:
    """3 independent signals for a consequential act. All 3 must pass."""
    signals = {
        "order_history_readback": bool(order_history()),
        "confirmation_email": bool(confirmation_mail()),
        "receipt": bool(receipt()),
    }
    ok = all(signals.values())
    return {"verified": ok, "signals": signals,
            "reason": "" if ok else "consequential act missing a signal"}


@dataclass
class Checkpoint:
    task: str
    step: int
    state: dict[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"task": self.task, "step": self.step,
                "state": dict(self.state), "at": self.at}
