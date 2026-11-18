"""Classification, flags, priority and work rank (autocraft port).

The contract is autocraft's docs/pipeline.md (Classify, Priority,
Work rank), carried verbatim where it is pure arithmetic and stated
as an interpretation where the upstream text leaves a gap:

* Types: bugfix, performance, missing feature, small extra, plugin
  candidate, infra. Flags: security, needs decision.
* Settle: the cheap first pass at 0.9 or more settles the type, or the
  model agreeing with it settles it. Disagreements among the scope
  labels (missing feature, small extra, plugin candidate) go to
  Decide. Other disagreements go to a referee; a sure referee
  settles, an unsure referee sets needs-decision.
* Flags follow the same rule on their own: security settles at 0.9,
  clears under 0.3, in between the model counts; needs-decision the
  same with 0.8 as the settle threshold (the upstream text says "the
  same with 0.8" without restating the clear threshold, so 0.3
  stands — recorded here, not hidden).
* Priority P0–P4 per the upstream table; the security flag raises the
  level by one, up to P0. Unrated items count as P2.
* Reach (0–99) breaks ties inside a level: comments, confirmed
  duplicates, the security flag and age. No model.
* Work rank: ``(4 - level) * 10000 + type * 1000 + size * 100 +
  reach``; unsettled type sorts last inside its level; issues count
  as M until a size is estimated.

The cheap first pass (``keyword_hint``) is keyword/regex signal over
title + body — Devon's offline Jev (``agents/jev.py``) is a *task*
classifier, not the Jev model autocraft calls, so it is not used
here; the model rung is the brain, reached through an injected
``suggest`` callable in the engine.
"""
from __future__ import annotations

import re
from typing import Any

__all__ = [
    "TYPE_LABELS",
    "SCOPE_LABELS",
    "PRIORITY_MEANINGS",
    "SIZE_LABELS",
    "keyword_hint",
    "settle_type",
    "settle_flag",
    "priority_for",
    "reach",
    "work_rank",
]

TYPE_LABELS: tuple[str, ...] = (
    "bugfix", "performance", "missing feature", "small extra",
    "plugin candidate", "infra",
)
#: The scope labels — disagreements among these go to Decide.
SCOPE_LABELS: tuple[str, ...] = (
    "missing feature", "small extra", "plugin candidate",
)
PRIORITY_MEANINGS: dict[str, str] = {
    "P0": "critical: crash, data loss, security hole, broken build "
          "or install, unusable app",
    "P1": "high: a core feature is broken, or a regression",
    "P2": "normal: an ordinary bug, feature or change",
    "P3": "low: cosmetic, wording, docs, nice-to-have",
    "P4": "backlog: kept open but not worked on now — vague or "
          "wishlist requests, ideas without a clear ask, very large "
          "asks with little reach; plugin candidates are P4",
}
SIZE_LABELS: tuple[str, ...] = ("S", "M", "L", "XL")

SETTLE_CONFIDENCE = 0.9
FLAG_SET_CONFIDENCE = 0.9
FLAG_CLEAR_CONFIDENCE = 0.3
NEEDS_DECISION_SET_CONFIDENCE = 0.8

_TYPE_INDEX = {label: i for i, label in enumerate(TYPE_LABELS)}
_SIZE_INDEX = {label: i for i, label in enumerate(SIZE_LABELS)}
_LEVEL_INDEX = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "P4": 4}


def _hits(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


_CRASH_RES = (
    r"\bcrash(es|ed|ing)?\b", r"\bsegfault\b", r"\bsegmentation fault\b",
    r"\bdata loss\b", r"\blost (my|all|the) data\b", r"\bhangs?\b",
    r"\bfreezes?\b", r"\bwon't start\b", r"\bdoesn'?t start\b",
    r"\bbroken build\b", r"\bbuild (fails|failing|broken)\b",
    r"\binstall (fails|failing|broken)\b", r"\bunusable\b",
)
_SECURITY_RES = (
    r"\bsecurity\b", r"\bvulnerab", r"\binjection\b", r"\bxss\b",
    r"\bcsrf\b", r"\bauth bypass\b", r"\bprivilege escalation\b",
    r"\bexploit\b", r"\bsecret (leak|expos)", r"\btoken (leak|expos)",
    r"\bremote code execution\b", r"\brce\b",
)
_REGRESSION_RES = (
    r"\bregression\b", r"\bused to work\b", r"\bbroke in\b",
    r"\bstopped working\b", r"\bno longer works\b",
)
_PERFORMANCE_RES = (
    r"\bslow\b", r"\bperformance\b", r"\blatency\b", r"\bmemory leak\b",
    r"\btoo much memory\b", r"\boptim(is|iz)e\b", r"\btimeout\b",
)
_FEATURE_RES = (
    r"\badd (support|a|an|the)\b", r"\bfeature request\b",
    r"\bplease add\b", r"\bwould be (nice|great) (to|if)\b",
    r"\bnew (capability|feature|option)\b", r"\bsupport for\b",
)
_DOCS_RES = (
    r"\btypo\b", r"\bspelling\b", r"\bwording\b", r"\bdocs?\b",
    r"\bdocumentation\b", r"\breadme\b",
)
_PLUGIN_RES = (
    r"\bplugin\b", r"\bextension\b", r"\baddon\b", r"\badd-on\b",
)
_INFRA_RES = (
    r"\bci\b", r"\bgithub actions\b", r"\bdeploy\b", r"\bpipeline\b",
    r"\binfrastructure\b", r"\bdocker\b",
)


def keyword_hint(title: str, body: str) -> dict[str, Any]:
    """Cheap first pass over title + body.

    Returns ``type`` (or None), ``confidence`` (0–1), ``security``
    + ``security_confidence``, ``needs_decision`` +
    ``needs_decision_confidence``, and ``signals`` (crash,
    regression, cosmetic, vague, wishlist booleans for priority).
    Strong signals settle at 0.95; weak ones stay unsure.
    """
    text = f"{title or ''}\n{body or ''}"
    stripped = text.strip()
    security = _hits(text, _SECURITY_RES)
    security_conf = 0.95 if security else 0.1
    signals = {
        "crash": _hits(text, _CRASH_RES),
        "regression": _hits(text, _REGRESSION_RES),
        "cosmetic": _hits(text, _DOCS_RES) and not _hits(text, _CRASH_RES),
        "vague": len(stripped) < 40,
        "wishlist": "wishlist" in text.lower()
                    or "idea:" in text.lower(),
    }
    hint: dict[str, Any] = {
        "type": None,
        "confidence": 0.0,
        "security": security,
        "security_confidence": security_conf,
        "needs_decision": False,
        "needs_decision_confidence": 0.1,
        "signals": signals,
    }
    if security and _hits(text, _CRASH_RES):
        hint.update(type="bugfix", confidence=0.95)
        return hint
    if signals["crash"] or signals["regression"]:
        hint.update(type="bugfix", confidence=0.95)
        return hint
    if _hits(text, _PERFORMANCE_RES):
        hint.update(type="performance", confidence=0.85)
        return hint
    if _hits(text, _PLUGIN_RES):
        hint.update(type="plugin candidate", confidence=0.85)
        return hint
    if _hits(text, _INFRA_RES):
        hint.update(type="infra", confidence=0.85)
        return hint
    if _hits(text, _FEATURE_RES):
        hint.update(type="missing feature", confidence=0.7)
        return hint
    if _hits(text, _DOCS_RES):
        hint.update(type="small extra", confidence=0.85)
        return hint
    if signals["vague"]:
        hint.update(needs_decision=True, needs_decision_confidence=0.6)
    return hint


def settle_type(offline: dict[str, Any],
                model: dict[str, Any] | None,
                referee: dict[str, Any] | None = None) -> dict[str, Any]:
    """Settle a type from the cheap pass, the model, and a referee.

    Returns ``type`` (or None), ``source`` (offline/model/referee/
    unsettled), ``confidence`` and ``needs_decision``.
    """
    off_type = offline.get("type")
    off_conf = float(offline.get("confidence") or 0.0)
    if off_type and off_conf >= SETTLE_CONFIDENCE:
        return {"type": off_type, "source": "offline",
                "confidence": off_conf, "needs_decision": False}
    model_type = (model or {}).get("type")
    if off_type and model_type and off_type == model_type:
        return {"type": off_type, "source": "model",
                "confidence": float(model.get("confidence") or 0.0)
                or off_conf,
                "needs_decision": False}
    if off_type and model_type and off_type != model_type:
        if off_type in SCOPE_LABELS and model_type in SCOPE_LABELS:
            return {"type": None, "source": "unsettled",
                    "confidence": 0.0, "needs_decision": True}
        ref_type = (referee or {}).get("type")
        if ref_type and referee.get("sure"):
            return {"type": ref_type, "source": "referee",
                    "confidence": float(referee.get("confidence") or 0.0),
                    "needs_decision": False}
        return {"type": None, "source": "unsettled",
                "confidence": 0.0, "needs_decision": True}
    if model_type:
        return {"type": model_type, "source": "model",
                "confidence": float(model.get("confidence") or 0.0),
                "needs_decision": False}
    if off_type:
        return {"type": off_type, "source": "offline",
                "confidence": off_conf, "needs_decision": False}
    return {"type": None, "source": "unsettled",
            "confidence": 0.0, "needs_decision": True}


def settle_flag(value: bool, confidence: float,
                model_value: bool | None, *,
                high: float = FLAG_SET_CONFIDENCE,
                low: float = FLAG_CLEAR_CONFIDENCE) -> bool | None:
    """Settle one flag: >= high takes the offline value, < low clears
    it, in between the model's value counts, and with no model it
    stays unsettled (None)."""
    if confidence >= high:
        return bool(value)
    if confidence < low:
        # Below the clear threshold the flag is off: a low-confidence
        # offline verdict never sets it, whatever it said.
        return False
    return model_value


def priority_for(type_: str | None, *,
                 security: bool = False,
                 signals: dict[str, Any] | None = None) -> str:
    """P0–P4 for a settled (or unsettled) type.

    Unrated items count as P2. The security flag raises the level by
    one, up to P0.
    """
    sig = signals or {}
    if sig.get("crash"):
        level = "P0"
    elif sig.get("regression"):
        level = "P1"
    elif type_ == "plugin candidate":
        level = "P4"
    elif type_ == "small extra":
        level = "P3"
    else:
        level = "P2"
    if security and level != "P0":
        order = ["P0", "P1", "P2", "P3", "P4"]
        level = order[max(0, order.index(level) - 1)]
    return level


def reach(item: dict[str, Any], *, duplicate_count: int = 0) -> int:
    """Tie-break reach, 0–99: comments, confirmed duplicates, the
    security flag and age. No model."""
    from datetime import UTC, datetime

    score = 0
    score += min(int(item.get("comments") or 0), 20) * 2
    score += min(int(duplicate_count or 0), 5) * 5
    if item.get("security"):
        score += 15
    created = str(item.get("created_at") or "")
    try:
        when = datetime.fromisoformat(created.replace("Z", "+00:00"))
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        age_days = max(0, (datetime.now(UTC) - when).days)
    except ValueError:
        age_days = 0
    score += min(age_days, 365) // 36
    return max(0, min(99, score))


def work_rank(level: str, type_: str | None, size: str,
              reach_value: int) -> int:
    """One integer key: level first, then type, then size, then reach.

    ``(4 - level) * 10000 + type * 1000 + size * 100 + reach`` —
    where ``type`` and ``size`` are *priority-ordered* values (bugfix
    highest, plugin candidate lowest, unsettled below all; S highest,
    XL lowest), so one descending sort yields the documented order:
    level P0 first, then bugfix → performance → infra → missing
    feature → small extra → plugin candidate → unsettled, then
    smaller first, then higher reach first.
    """
    level_n = _LEVEL_INDEX.get(level, 2)
    type_n = (len(TYPE_LABELS) - 1 - _TYPE_INDEX[type_]
              if type_ in _TYPE_INDEX else -1)
    size_n = len(SIZE_LABELS) - 1 - _SIZE_INDEX.get(size, _SIZE_INDEX["M"])
    return ((4 - level_n) * 10000 + type_n * 1000 + size_n * 100
            + max(0, min(99, int(reach_value))))
