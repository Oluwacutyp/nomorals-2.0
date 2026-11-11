"""Owner approval channel for consequential browser steps.

The agent loop's approver used to be a wall for consequential
actions (submit / post / pay / delete / irreversible): benign acts
proceeded, consequential ones were denied and the run DIED at the
step. This module turns that wall into a pause, on the same
pause → owner acts → resume discipline the takeover plumbing
proved for challenge walls:

* the run thread parks inside the approver call;
* the owner is asked in the origin chat — what the step is, on
  which page, for which task — with the exact verbs to answer;
* approve → the approver returns True and the loop performs and
  verifies the step like any other;
* deny or timeout → the approver returns False and the loop stops,
  with the outcome recorded so the run's receipt says WHY (owner
  denied / no answer), never a bare "approval denied".

Two postures exist, chosen by configuration
(``partner.browser_require_approval``):

* DEFAULT — the owner's standing order (2026-11-11): she acts on
  consequential steps herself. ``ActingApprover`` approves and
  records each one in the run's receipt; nothing parks.
* OPT-IN ask-first — this broker: nothing consequential executes
  without the owner's approval. The broker is a mechanism, not a
  rubber stamp — with no notifier attached it cannot ask, so it
  cannot approve.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger
from .agent import CONSEQUENTIAL_KINDS, Decision, Observation

_log = get_logger(__name__)

__all__ = [
    "ActingApprover",
    "ApprovalBroker",
    "ApprovalRequest",
    "RunApprover",
]


@dataclass
class ApprovalRequest:
    """One consequential step waiting on the owner's word."""

    request_id: str
    task_label: str
    owner_chat: str
    kind: str
    label: str
    target: str
    url: str
    reason: str
    created_at: float
    verdict: str = ""  # "" pending | approved | denied | timeout
    answered_by: str = ""
    _event: threading.Event = field(default_factory=threading.Event,
                                    repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {"request_id": self.request_id,
                "task_label": self.task_label,
                "owner_chat": self.owner_chat,
                "kind": self.kind, "label": self.label,
                "target": self.target, "url": self.url,
                "reason": self.reason, "verdict": self.verdict,
                "answered_by": self.answered_by}


class ApprovalBroker:
    """Issues approval requests, delivers the ask, parks the run.

    ``notifier`` is the owner-chat sender, ``(message, url) -> None``
    — the same seam the takeover escalation uses, so the ask lands
    in the owner's chat and nowhere else. ``timeout_s`` bounds the
    park: silence is a no.
    """

    def __init__(self,
                 notifier: Callable[[str, str], Any] | None = None, *,
                 timeout_s: float = 900.0) -> None:
        self._notifier = notifier
        self.timeout_s = max(0.0, float(timeout_s))
        self._counter = itertools.count(1)
        self._lock = threading.Lock()
        self._pending: dict[str, ApprovalRequest] = {}
        #: Every answered/timed-out request this process handled,
        #: oldest first — the receipt trail for run status.
        self.history: list[dict[str, Any]] = []

    # -- the loop-side seam --------------------------------------------------
    def approver(self, *, task_label: str = "",
                 owner_chat: str = "") -> RunApprover:
        """An ``AgentLoop`` approver bound to one run."""
        return RunApprover(self, task_label=task_label,
                           owner_chat=owner_chat)

    # -- the owner-side seam ---------------------------------------------------
    def resolve(self, request_id: str, approved: bool, *,
                by: str = "") -> bool:
        """Record the owner's answer. False when no such request is
        pending (unknown id, or it already timed out / was answered)."""
        with self._lock:
            req = self._pending.get((request_id or "").strip())
        if req is None:
            return False
        req.verdict = "approved" if approved else "denied"
        req.answered_by = by or ""
        req._event.set()
        return True

    def pending(self) -> list[dict[str, Any]]:
        """Requests still waiting on the owner (status displays)."""
        with self._lock:
            return [req.to_dict() for req in self._pending.values()
                    if not req.verdict]

    # -- internals -------------------------------------------------------------
    def _ask(self, *, task_label: str, owner_chat: str,
             decision: Decision, obs: Observation) -> ApprovalRequest:
        with self._lock:
            request_id = f"appr-{next(self._counter)}"
        action = decision.action
        req = ApprovalRequest(
            request_id=request_id, task_label=task_label,
            owner_chat=owner_chat, kind=action.kind,
            label=action.label, target=action.target,
            url=obs.page.url, reason=decision.reason,
            created_at=time.time())
        with self._lock:
            self._pending[request_id] = req
        minutes = max(1, int(self.timeout_s // 60))
        what = action.label or action.target or action.kind
        message = (
            "⏸️ Browser run paused — your approval is needed\n"
            f"Task: {task_label or 'browsing'}\n"
            f"Page: {obs.page.url}\n"
            f"Next step: {action.kind} — {what}\n"
            f"Why: {decision.reason or 'the next step toward the goal'}\n"
            "This step is consequential, so I won't do it without "
            "your word.\n"
            f"/browser approve {request_id} — do it\n"
            f"/browser deny {request_id} — stop the run\n"
            f"No answer in {minutes} min and the run stops here.")
        if self._notifier is not None:
            try:
                self._notifier(message, "")
            except Exception:  # noqa: BLE001 - a failed send is
                # recorded; the wait below still times out honestly
                _log.warning("approval ask failed to send",
                             exc_info=True)
        _log.info("browser approval requested: %s (%s on %s)",
                  request_id, action.kind, obs.page.url)
        return req

    def _wait(self, req: ApprovalRequest) -> str:
        """Park until the owner answers or the timeout lands."""
        answered = req._event.wait(self.timeout_s)
        with self._lock:
            self._pending.pop(req.request_id, None)
        if not answered or not req.verdict:
            req.verdict = "timeout"
        outcome = {"request_id": req.request_id, "kind": req.kind,
                   "label": req.label, "url": req.url,
                   "verdict": req.verdict,
                   "answered_by": req.answered_by}
        with self._lock:
            self.history.append(outcome)
        _log.info("browser approval %s: %s (%s)",
                  req.request_id, req.verdict, req.kind)
        return req.verdict


class RunApprover:
    """The approver one run hands to its ``AgentLoop``.

    Benign acts proceed exactly as the old deny-wall approver let
    them; a consequential (or irreversible) decision goes through
    the broker: ask, park, resume or stop. ``outcomes`` records
    every consequential verdict for the run's receipt.
    """

    def __init__(self, broker: ApprovalBroker, *, task_label: str = "",
                 owner_chat: str = "") -> None:
        self._broker = broker
        self.task_label = task_label
        self.owner_chat = owner_chat
        self.outcomes: list[dict[str, Any]] = []

    def __call__(self, decision: Decision, obs: Observation) -> bool:
        if not decision.irreversible \
                and decision.action.kind not in CONSEQUENTIAL_KINDS:
            return True
        if self._broker._notifier is None:
            # No channel to the owner = no way to ask = no approval.
            self.outcomes.append({
                "request_id": "", "kind": decision.action.kind,
                "label": decision.action.label, "url": obs.page.url,
                "verdict": "no_channel", "answered_by": ""})
            return False
        req = self._broker._ask(task_label=self.task_label,
                                owner_chat=self.owner_chat,
                                decision=decision, obs=obs)
        verdict = self._broker._wait(req)
        self.outcomes.append({
            "request_id": req.request_id, "kind": req.kind,
            "label": req.label, "url": req.url,
            "verdict": verdict, "answered_by": req.answered_by})
        return verdict == "approved"


class ActingApprover:
    """The default approver: Devon acts on consequential steps
    herself (owner's standing order, 2026-11-11).

    Same seam as ``RunApprover`` — benign acts proceed silently;
    a consequential (or irreversible) decision is approved and
    recorded in ``outcomes`` with verdict ``acted``, so the run's
    receipt names every consequential step she took, and the loop
    verifies each by read-back like any other action. The
    ask-first discipline stays available as a configuration
    through ``ApprovalBroker``.
    """

    def __init__(self, *, task_label: str = "",
                 owner_chat: str = "") -> None:
        self.task_label = task_label
        self.owner_chat = owner_chat
        self.outcomes: list[dict[str, Any]] = []
        self._counter = itertools.count(1)

    def __call__(self, decision: Decision, obs: Observation) -> bool:
        if not decision.irreversible \
                and decision.action.kind not in CONSEQUENTIAL_KINDS:
            return True
        self.outcomes.append({
            "request_id": f"act-{next(self._counter)}",
            "kind": decision.action.kind,
            "label": decision.action.label, "url": obs.page.url,
            "verdict": "acted", "answered_by": "devon"})
        _log.info("browser consequential step acted: %s on %s (%s)",
                  decision.action.kind, obs.page.url,
                  self.task_label or "browser run")
        return True
