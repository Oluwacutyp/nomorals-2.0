"""Jev — the cheap decision classifier (build-map #18 extension; Hercules "Jev" pattern).

The *named* primitive for cheap decisions: **classify / route / moderate**.
Explicitly separate from the main brain — **zero LLM calls inside, ever**.
Offline, instant (keyword/regex scans, microseconds), never raises.

Why a separate primitive instead of an anonymous helper: callers should
*reach for it deliberately*.  Instead of spending the big model on "what
kind of request is this, where should it go, is it safe?", the system asks
Jev first and spends the brain only on the work itself::

    from nomorals.agents.jev import jev

    d = jev.decide("build me a csv parser")
    # d.classify.kind == "build", d.route.handler == "coding",
    # d.moderate.level == "safe"

Jev composes the two existing offline layers — ``task_type.classify_task``
(keyword intent: build/investigate/research/chat) and
``complexity.classify_complexity`` (easy/medium/hard) — and adds a cheap
input-moderation screen.  ``use_model=False`` is forced everywhere: Jev
must decide with no context, no provider, no network.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger
from .complexity import COMPLEXITIES, classify_complexity
from .task_type import TASK_KINDS, classify_task

__all__ = [
    "DecisionClassifier",
    "Classification",
    "Route",
    "Moderation",
    "Decision",
    "jev",
    "MODERATION_LEVELS",
    "ROUTES",
]

_log = get_logger(__name__)

#: Moderation verdicts, least → most severe.
MODERATION_LEVELS = ("safe", "flagged", "unsafe")

# ── moderation patterns ──────────────────────────────────────────────────
# Cheap input triage only: injection attempts are "unsafe" (refuse-worthy),
# destructive/credential-adjacent requests are "flagged" (route through the
# confirmation gradient — see core/policy.py), everything else is "safe".
# These are deliberately narrow: Jev must not cry wolf on ordinary chat.

_UNSAFE_RES = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior)\s+instructions",
        r"\breveal\s+(your|the)\s+(system\s+prompt|prompt|instructions)\b",
        r"\byou\s+are\s+now\b",
        r"\bdo\s+anything\s+now\b",
        r"\bdeveloper\s+mode\b",
        r"\bjailbreak\b",
        r"\bpretend\s+(you\s+are|to\s+be)\b",
        r"\boverride\s+your\b",
        r"\bdisregard\s+(all\s+|any\s+|the\s+)?(previous|prior|above)\b",
    )
)

_FLAGGED_RES = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\brm\s+-rf?\b",
        r"\bformat\s+[a-zA-Z]:",
        r"\bdelete\s+(everything|all\s+files)\b",
        r"\bdrop\s+table\b",
        r"\bmkfs\b",
        r"\bwipe\s+(the\s+)?(disk|drive|database)\b",
        r"\bbypass\b.{0,20}\b(auth|authentication|login|permission)\b",
        # "tell me your api key" — request verb + possessive + credential noun.
        # "what is an api key" does NOT match (no request verb).
        r"\b(give|show|tell|send|reveal|share|print|output)\b"
        r".{0,30}\b(your|my|the)\s+"
        r"(password|api[\s_-]?key|secret(\s+key)?|private\s+key)\b",
    )
)

# ── routing table ────────────────────────────────────────────────────────
# kind → (task_type from router_select.TASK_TYPES, owning agent module).
# All modules below exist in nomorals/agents/.

ROUTES: dict[str, tuple[str, str]] = {
    "build": ("coding", "coding"),
    "investigate": ("reasoning", "agent_loop"),
    "research": ("reasoning", "researcher"),
    "chat": ("chat", "chat"),
}


@dataclass
class Classification:
    """What kind of request this is (intent + complexity)."""

    kind: str = "chat"                    # one of TASK_KINDS
    kind_confidence: float = 0.3
    kind_reason: str = ""
    complexity: str = "medium"            # one of COMPLEXITIES
    complexity_confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "kind_confidence": round(self.kind_confidence, 3),
            "kind_reason": self.kind_reason,
            "complexity": self.complexity,
            "complexity_confidence": round(self.complexity_confidence, 3),
        }


@dataclass
class Route:
    """Where the request should go."""

    task_type: str = "chat"               # router_select task type
    handler: str = "chat"                 # owning agent module
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"task_type": self.task_type, "handler": self.handler,
                "reason": self.reason}


@dataclass
class Moderation:
    """Cheap input safety screen."""

    level: str = "safe"                   # one of MODERATION_LEVELS
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"level": self.level, "reasons": list(self.reasons)}


@dataclass
class Decision:
    """Everything Jev decided about one piece of text."""

    classification: Classification = field(default_factory=Classification)
    route: Route = field(default_factory=Route)
    moderation: Moderation = field(default_factory=Moderation)
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification.to_dict(),
            "route": self.route.to_dict(),
            "moderation": self.moderation.to_dict(),
            "elapsed_ms": round(self.elapsed_ms, 3),
        }


class DecisionClassifier:
    """The cheap decision primitive.  No brain, no network, no excuses.

    Every public method is offline, instant, and never raises.  ``decide``
    is the one-call entry point; ``classify`` / ``route`` / ``moderate``
    are the individual primitives for callers that need just one.
    """

    # ── classify ───────────────────────────────────────────────────────
    def classify(self, text: Any) -> Classification:
        """Intent (build/investigate/research/chat) + complexity (easy/medium/hard)."""
        try:
            s = self._text(text)
            tt = classify_task(None, s, use_model=False)
            level, conf = classify_complexity(s)
            kind = tt.kind if tt.kind in TASK_KINDS else "chat"
            return Classification(
                kind=kind,
                kind_confidence=float(tt.confidence or 0.3),
                kind_reason=str(getattr(tt, "reason", "") or ""),
                complexity=level if level in COMPLEXITIES else "medium",
                complexity_confidence=float(conf),
            )
        except Exception:  # noqa: BLE001 — Jev never breaks the caller
            _log.debug("jev.classify failed; defaulting", exc_info=True)
            return Classification()

    # ── route ──────────────────────────────────────────────────────────
    def route(self, text: Any) -> Route:
        """Which agent/tool should handle this?"""
        try:
            kind = self.classify(text).kind
            task_type, handler = ROUTES.get(kind, ("chat", "chat"))
            return Route(
                task_type=task_type,
                handler=handler,
                reason=f"kind={kind} → {handler} ({task_type})",
            )
        except Exception:  # noqa: BLE001
            _log.debug("jev.route failed; defaulting", exc_info=True)
            return Route()

    # ── moderate ───────────────────────────────────────────────────────
    def moderate(self, text: Any) -> Moderation:
        """Cheap input screen: safe | flagged (confirm first) | unsafe (refuse)."""
        try:
            s = self._text(text)
            unsafe_hits = [rx.pattern for rx in _UNSAFE_RES if rx.search(s)]
            if unsafe_hits:
                return Moderation(
                    level="unsafe",
                    reasons=[f"injection marker: {p[:60]}" for p in unsafe_hits[:3]],
                )
            flagged_hits = [rx.pattern for rx in _FLAGGED_RES if rx.search(s)]
            if flagged_hits:
                return Moderation(
                    level="flagged",
                    reasons=[f"needs confirmation: {p[:60]}" for p in flagged_hits[:3]],
                )
            return Moderation(level="safe")
        except Exception:  # noqa: BLE001
            _log.debug("jev.moderate failed; defaulting", exc_info=True)
            return Moderation()

    # ── decide (the deliberate reach) ──────────────────────────────────
    def decide(self, text: Any) -> Decision:
        """Classify + route + moderate in one cheap call.  Never raises."""
        start = time.perf_counter()
        try:
            classification = self.classify(text)
            task_type, handler = ROUTES.get(classification.kind, ("chat", "chat"))
            route = Route(
                task_type=task_type,
                handler=handler,
                reason=f"kind={classification.kind} → {handler} ({task_type})",
            )
            moderation = self.moderate(text)
            return Decision(
                classification=classification,
                route=route,
                moderation=moderation,
                elapsed_ms=(time.perf_counter() - start) * 1000.0,
            )
        except Exception:  # noqa: BLE001
            _log.debug("jev.decide failed; defaulting", exc_info=True)
            return Decision()

    # ── helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _text(value: Any) -> str:
        try:
            if value is None:
                return ""
            if isinstance(value, str):
                return value
            return str(value)
        except Exception:  # noqa: BLE001
            return ""


#: The shared instance — the deliberate reach: ``jev.decide(text)``.
jev = DecisionClassifier()
