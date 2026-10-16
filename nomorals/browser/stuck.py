"""Stuck detector: the #1 browser-agent failure is not knowing it is stuck.

WebVoyager failure table (per BROWSER_AGENT doc): navigation stuck 44.4%,
visual grounding 24.8%, hallucination 21.8%. So stuck detection is
first-class, with state delta as a first-class signal:

* same URL + same snapshot hash across N actions
* same action repeated inside a window
* modal / dialog / error undismissed across steps
* network idle + no DOM mutation
* step budget burning with no state delta

Escalation ladder: escalate perception tier -> change strategy (direct
URL, search, API, alternate flow) -> re-plan from goal -> handoff with
diagnosis + screenshot + exact step.
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["StuckDetector", "StuckDiagnosis", "StuckSignal"]

#: Signals that feed the stuck verdict.
StuckSignal = str


@dataclass
class StuckDiagnosis:
    stuck: bool
    signals: list[str] = field(default_factory=list)
    recommended: str = "continue"  # continue | escalate_tier | change_strategy | replan | handoff
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"stuck": self.stuck, "signals": list(self.signals),
                "recommended": self.recommended, "detail": self.detail}


class StuckDetector:
    """Tracks (url, snapshot-hash, action) history and calls stuck."""

    def __init__(self, *,
                 same_state_threshold: int = 3,
                 repeat_action_threshold: int = 3,
                 window: int = 10,
                 modal_threshold: int = 2) -> None:
        self.same_state_threshold = max(2, int(same_state_threshold))
        self.repeat_action_threshold = max(2, int(repeat_action_threshold))
        self.window = max(4, int(window))
        self.modal_threshold = max(1, int(modal_threshold))
        self._history: deque[dict[str, Any]] = deque(maxlen=self.window)
        self._modal_seen = 0

    @staticmethod
    def snapshot_hash(url: str, html: str = "", title: str = "") -> str:
        return hashlib.sha256(
            f"{url}\n{title}\n{html}".encode("utf-8", "replace")
        ).hexdigest()[:16]

    def observe(self, *, url: str, html: str = "", title: str = "",
                action: dict[str, Any] | None = None,
                modal_present: bool = False,
                dom_mutated: bool = True,
                network_idle: bool = False) -> StuckDiagnosis:
        """Record one step; return the stuck verdict for the window."""
        self._history.append({
            "url": url,
            "hash": self.snapshot_hash(url, html, title),
            "action": (action or {}).get("kind", "") + ":" + (action or {}).get("target", ""),
            "modal": bool(modal_present),
            "mutated": bool(dom_mutated),
            "idle": bool(network_idle),
        })
        if modal_present:
            self._modal_seen += 1
        elif self._history and not self._history[-1]["modal"]:
            self._modal_seen = 0
        return self.diagnose()

    def diagnose(self) -> StuckDiagnosis:
        hist = list(self._history)
        signals: list[str] = []
        if len(hist) >= self.same_state_threshold:
            tail = hist[-self.same_state_threshold:]
            if len({(h["url"], h["hash"]) for h in tail}) == 1:
                signals.append(
                    f"same-state x{self.same_state_threshold}: "
                    f"{tail[-1]['url']} hash {tail[-1]['hash']}")
        if len(hist) >= self.repeat_action_threshold:
            tail = hist[-self.repeat_action_threshold:]
            if len({h["action"] for h in tail}) == 1 and tail[-1]["action"]:
                signals.append(
                    f"same-action x{self.repeat_action_threshold}: "
                    f"{tail[-1]['action']}")
        if self._modal_seen >= self.modal_threshold:
            signals.append(f"modal/dialog undismissed x{self._modal_seen}")
        if len(hist) >= 3:
            tail = hist[-3:]
            if all(h["idle"] and not h["mutated"] for h in tail):
                signals.append("network-idle + no-DOM-mutation x3")
        if len(hist) >= 4:
            tail = hist[-4:]
            if all(not h["mutated"] for h in tail):
                signals.append("no state delta x4 (budget burning)")
        if not signals:
            return StuckDiagnosis(stuck=False)
        # Ladder: mild signals -> escalate tier; repeated/compound -> strategy;
        # modal or long burn -> replan; everything at once -> handoff.
        if "modal" in " ".join(signals):
            recommended = "handoff"
            detail = ("A modal/dialog will not dismiss. Owner action needed; "
                      "session stays warm for resume.")
        elif len(signals) >= 3:
            recommended = "handoff"
            detail = "Multiple stuck signals at once — hand off with diagnosis."
        elif len(signals) == 2:
            recommended = "change_strategy"
            detail = "Try direct URL / search / API / alternate flow."
        else:
            recommended = "escalate_tier"
            detail = "Escalate perception tier before changing strategy."
        return StuckDiagnosis(stuck=True, signals=signals,
                              recommended=recommended, detail=detail)

    def reset(self) -> None:
        self._history.clear()
        self._modal_seen = 0
