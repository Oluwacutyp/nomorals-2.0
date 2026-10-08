"""Move executor: run a :class:`Move` through the driver behind a safety gate.

The gate is deny-by-default on three axes:

* **Real money** — any URL or label matching payment/checkout/topup patterns
  is blocked, always. Virtual in-game currency (₦ game cash, coins) is fine;
  anything that touches fiat is not.
* **Destruction** — delete/destroy/remove-account labels are blocked, always.
* **Irreversible moves** — anything else flagged ``irreversible=True`` needs
  an explicit confirmation callback; without one the move is declined.

Every attempted move — executed, blocked, or dry-run — is appended to
:attr:`Executor.log` and to the module logger. In ``dry_run`` mode nothing
touches the network beyond state capture; the move is recorded as skipped.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ...core.errors import ToolError
from ...core.logging_setup import get_logger
from .decider import Move
from .driver import BrowserDriver, Page
from .state import GameState, capture_state

_log = get_logger(__name__)

__all__ = [
    "ExecutionResult",
    "Executor",
    "SafetyBlocked",
    "is_forbidden",
]

#: URL patterns that smell like real-money flows. Blocked unconditionally.
FORBIDDEN_URL_RE = re.compile(
    r"(?i)(checkout|/pay/|payment|paystack|flutterwave|stripe|paypal|"
    r"topup|top-up|/wallet/(deposit|fund)|subscribe|billing)"
)

#: Labels that smell like real-money or destruction. Blocked unconditionally.
FORBIDDEN_LABEL_RE = re.compile(
    r"(?i)(real money|\bpay\s+[$₦€£]|\bbuy\s+with|top\s*up|deposit\s+(funds|cash)|"
    r"withdraw|checkout|delete|destroy|remove\s+account|close\s+account)"
)


class SafetyBlocked(ToolError):
    """Raised when the safety gate refuses a move."""

    code = "gameplayer.safety_blocked"


def is_forbidden(move: Move, state: GameState) -> str | None:
    """Return the block reason, or None when the move passes the gate."""
    label = move.action.label or ""
    if FORBIDDEN_LABEL_RE.search(label):
        return f"forbidden label: {label!r}"
    # Resolve the move's target URL when we can, and check that too.
    target_url = ""
    if move.action.kind == "goto":
        target_url = move.action.target
    else:
        for el in state.elements:
            if el.id == move.action.target:
                target_url = el.url or el.form_action
                break
    if target_url and FORBIDDEN_URL_RE.search(target_url):
        return f"forbidden url: {target_url!r}"
    return None


@dataclass
class ExecutionResult:
    ok: bool
    move: Move
    page: Page | None = None
    blocked_reason: str | None = None
    dry_run: bool = False
    at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "action": {
                "kind": self.move.action.kind,
                "target": self.move.action.target,
                "label": self.move.action.label,
            },
            "reason": self.move.reason,
            "blocked_reason": self.blocked_reason,
            "dry_run": self.dry_run,
            "url": self.page.url if self.page else None,
            "at": self.at,
        }


class Executor:
    """Executes moves with the safety gate. Fail-fast, fully logged."""

    def __init__(
        self,
        driver: BrowserDriver,
        *,
        dry_run: bool = True,
        confirm: Callable[[Move, GameState], bool] | None = None,
    ) -> None:
        # dry_run=True is the default: looking is free, touching is not.
        self.driver = driver
        self.dry_run = dry_run
        self.confirm = confirm
        self.log: list[ExecutionResult] = []

    def execute(self, move: Move, state: GameState, page: Page) -> ExecutionResult:
        """Run one move against ``page`` (the page ``state`` was captured from).

        Blocked moves raise :class:`SafetyBlocked`. Every attempt is logged.
        """
        blocked = is_forbidden(move, state)
        if blocked:
            result = ExecutionResult(ok=False, move=move, blocked_reason=blocked)
            self.log.append(result)
            _log.warning("SAFETY BLOCKED move %r: %s", move.action.label, blocked)
            raise SafetyBlocked(blocked)

        if move.action.kind == "freeform":
            # interpret_freeform's escape hatch: the raw player intent is in
            # the payload for the GAME to handle. The browser driver has no
            # element to act on, so this stops here — loudly, not silently.
            reason = ("freeform intent needs game-level handling "
                      f"({move.action.payload.get('text', '')!r}); "
                      "no page element to act on")
            result = ExecutionResult(ok=False, move=move,
                                     blocked_reason=reason)
            self.log.append(result)
            raise SafetyBlocked(reason)

        if move.irreversible and not self._confirmed(move, state):
            result = ExecutionResult(
                ok=False,
                move=move,
                blocked_reason="irreversible move declined: no confirmation",
            )
            self.log.append(result)
            _log.warning(
                "declined irreversible move %r: no confirmation", move.action.label
            )
            raise SafetyBlocked("irreversible move declined: no confirmation")

        if self.dry_run:
            result = ExecutionResult(ok=True, move=move, dry_run=True)
            self.log.append(result)
            _log.info(
                "DRY-RUN would %s %r (%s)",
                move.action.kind,
                move.action.label,
                move.reason,
            )
            return result

        page = self.driver.act(move.action, page)
        result = ExecutionResult(ok=True, move=move, page=page)
        self.log.append(result)
        _log.info(
            "executed %s %r -> %s (%s)",
            move.action.kind,
            move.action.label,
            page.url,
            move.reason,
        )
        return result

    def _confirmed(self, move: Move, state: GameState) -> bool:
        if self.confirm is None:
            return False  # fail-closed: no callback means no.
        try:
            return bool(self.confirm(move, state))
        except Exception:  # noqa: BLE001 - a broken callback must not approve
            _log.exception("confirmation callback failed; treating as declined")
            return False
