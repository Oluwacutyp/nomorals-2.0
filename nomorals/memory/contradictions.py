"""Contradiction detection as a strategy chain — not a boolean.

When a new FACT or PREFERENCE lands, older records may disagree with it.
The old code had one crude test (negation flip) wired only into the
``memory`` agent tool.  This module runs a *chain of strategies*, each
returning a confidence, so "X contradicts Y" is an evidence-backed claim,
not a hardcoded ``if``.

Strategies (ordered weakest → strongest):
1. ``negation_flip`` — "X" vs "not X" / "never X" on the same topic.
2. ``preference_flip`` — two preferences on the same topic that read
   nothing alike ("prefer dark mode" → "prefer light mode").
3. ``value_change`` — same topic, different number/date ("deadline oct 20"
   → "deadline oct 25", "budget 50k" → "budget 30k").
4. ``llm_adjudicate`` — opt-in native adjudication via ``llm_fn``; fails
   closed (no verdict) on any error.

Resolution is ADDITIVE: the old record is marked
``metadata.superseded_by`` — never deleted, never edited.  The user's
"forget only on explicit command" rule is untouched: contradiction
handling supersedes, it does not forget.
"""

from __future__ import annotations

import difflib
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..core.logging_setup import get_logger
from .base import MemoryKind

_log = get_logger(__name__)

__all__ = [
    "Contradiction",
    "ContradictionStrategy",
    "detect_against",
    "detect_for",
    "resolve",
    "STRATEGIES",
]

#: auto-resolve at or above this confidence; below it the contradiction is
#: reported but the old record is left alone for the owner to judge.
AUTO_RESOLVE_CONFIDENCE = 0.6


@dataclass
class Contradiction:
    """One detected disagreement between two records."""

    older_id: str
    newer_id: str
    strategy: str
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "older_id": self.older_id,
            "newer_id": self.newer_id,
            "strategy": self.strategy,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence,
            "note": self.note,
        }


@dataclass
class ContradictionStrategy:
    """One named detection strategy in the chain."""

    name: str
    detect: Callable[[str, str, Any, Any], float]
    describe: Callable[[str, str, dict[str, Any]], str]


# ── shared text helpers ──────────────────────────────────────────────────

_WORD4 = re.compile(r"[a-z]{4,}")
_NEGATIONS = ("not ", "n't ", "never ", "no longer ", "don't ",
              "doesn't ", "dont ", "dont like", "no more ")

_NUMBER = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?\s*(?:k|m|bn|million|billion|thousand)?\b", re.I)
_DATE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b"  # 2026-10-09
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?\b"  # oct 20
    r"|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b",  # 20 oct
    re.I)
_TIME = re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b", re.I)


def _topic_words(text: str) -> set[str]:
    return set(_WORD4.findall((text or "").lower()))


def _negated(text: str) -> bool:
    low = (text or "").lower()
    return any(neg in low for neg in _NEGATIONS)


def _values(text: str) -> list[str]:
    """Number/date/time tokens — the things that *change* in a contradiction.

    Trailing punctuation is stripped so "oct 20," and "oct 20" compare
    equal — a comma is not a changed deadline.
    """
    low = (text or "").lower()
    raw = _NUMBER.findall(low) + _DATE.findall(low) + _TIME.findall(low)
    # NB: the number pattern eats trailing whitespace ("20, "), so strip
    # spaces BEFORE punctuation — otherwise the comma survives.
    return [re.sub(r"[,\.;:]+$", "", v.strip()).strip()
            for v in raw if v.strip()]


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, (a or "").lower(),
                                   (b or "").lower()).ratio()


# ── strategies ───────────────────────────────────────────────────────────

def _negation_flip(old_text: str, new_text: str,
                   old: Any, new: Any) -> float:
    """'X' vs 'not X' on the same topic."""
    shared = _topic_words(old_text) & _topic_words(new_text)
    if len(shared) < 2:
        return 0.0
    if _negated(old_text) != _negated(new_text):
        return 0.65
    return 0.0


def _describe_negation(old_text: str, new_text: str,
                       evidence: dict[str, Any]) -> str:
    return (f"negation flip on {', '.join(evidence.get('shared', [])[:3])}: "
            f"“{old_text[:80]}” → “{new_text[:80]}”")


def _preference_flip(old_text: str, new_text: str,
                     old: Any, new: Any) -> float:
    """Two preferences about the same thing with different objects.

    Not string similarity — "dark mode" vs "light mode" are 90% similar
    strings and a 100% contradiction.  The strategy: extract the
    preference *target* (what comes after prefer/like/rather), then
    - an antonym swap in the target (dark→light) → 0.75;
    - the same head noun with different modifiers
      ("morning briefings" → "evening briefings") → 0.70.
    Different head nouns ("prefer tea" → "prefer coffee") do NOT fire:
    a person can hold both, so that is not a contradiction.
    """
    if getattr(old, "kind", "") != MemoryKind.PREFERENCE:
        return 0.0
    if getattr(new, "kind", "") != MemoryKind.PREFERENCE:
        return 0.0
    if len(_topic_words(old_text) & _topic_words(new_text)) < 2:
        return 0.0
    old_target = _pref_target(old_text)
    new_target = _pref_target(new_text)
    if not old_target or not new_target:
        return 0.0
    old_words = _content_words(old_target)
    new_words = _content_words(new_target)
    if not old_words or not new_words or old_words == new_words:
        return 0.0
    for a, b in _ANTONYM_PAIRS:
        if (a in old_words and b in new_words) or \
                (b in old_words and a in new_words):
            return 0.75
    # head noun: last content word of the target ("strong coffee" → coffee)
    if old_words[-1] == new_words[-1] and \
            set(old_words) != set(new_words):
        return 0.70
    return 0.0


def _describe_preference(old_text: str, new_text: str,
                         evidence: dict[str, Any]) -> str:
    return (f"preference changed on {', '.join(evidence.get('shared', [])[:3])}: "
            f"“{old_text[:80]}” → “{new_text[:80]}”")


_PREF_VERBS = ("prefer", "rather", "like", "love", "hate", "enjoy")

#: minimal, safe antonym pairs — the mutually-exclusive swaps
_ANTONYM_PAIRS = (
    ("dark", "light"), ("hot", "cold"), ("early", "late"),
    ("loud", "quiet"), ("more", "less"), ("always", "never"),
    ("short", "long"), ("fast", "slow"), ("high", "low"),
    ("on", "off"), ("open", "closed"), ("big", "small"),
    ("cheap", "expensive"), ("simple", "complex"),
)

#: trailing adverbials that are not the head noun
_NON_HEAD = frozenset({
    "always", "never", "usually", "often", "sometimes", "daily",
    "anymore", "again", "still", "now", "today",
})


def _content_words(text: str) -> list[str]:
    """Content words (len ≥ 4, order preserved), adverbials dropped."""
    return [w for w in _WORD4.findall((text or "").lower())
            if w not in _NON_HEAD]


def _pref_target(text: str) -> str:
    """The object of the preference: what comes after prefer/like/…."""
    low = (text or "").lower()
    for verb in _PREF_VERBS:
        m = re.search(rf"\b{verb}\b(.{{1,120}})", low)
        if m:
            target = m.group(1)
            # cut subordinate clauses — the target is the first clause
            target = re.split(r"\b(because|when|unless|if|so that)\b",
                              target)[0]
            target = re.sub(r"[,\.;:]+$", "", target).strip()
            if target:
                return target
    return ""


def _value_change(old_text: str, new_text: str,
                  old: Any, new: Any) -> float:
    """Same topic, different number/date — the deadline/budget pattern."""
    shared = _topic_words(old_text) & _topic_words(new_text)
    if len(shared) < 2:
        return 0.0
    old_vals = {v.strip().lower() for v in _values(old_text)}
    new_vals = {v.strip().lower() for v in _values(new_text)}
    if old_vals and new_vals and old_vals != new_vals:
        return 0.75
    return 0.0


def _describe_value(old_text: str, new_text: str,
                    evidence: dict[str, Any]) -> str:
    return (f"value changed ({evidence.get('old_values')} → "
            f"{evidence.get('new_values')}): “{old_text[:80]}”")


def _llm_adjudicate(old_text: str, new_text: str,
                     old: Any, new: Any) -> float:
    """Placeholder slot for the LLM strategy — the real implementation is
    injected via ``detect_for(..., llm_fn=...)`` / ``detect_against(...,
    llm_fn=...)``.  Without an ``llm_fn`` this strategy abstains (0.0).
    """
    return 0.0


def _describe_llm(old_text: str, new_text: str,
                  evidence: dict[str, Any]) -> str:
    return f"model judged contradiction: {evidence.get('reason', '')}"[:200]


STRATEGIES: list[ContradictionStrategy] = [
    ContradictionStrategy("negation_flip", _negation_flip, _describe_negation),
    ContradictionStrategy("preference_flip", _preference_flip, _describe_preference),
    ContradictionStrategy("value_change", _value_change, _describe_value),
    ContradictionStrategy("llm_adjudicate", _llm_adjudicate, _describe_llm),
]

_LLM_PROMPT = (
    "Two memories are shown. Reply with exactly one line: "
    "CONTRADICT or AGREE or UNRELATED, then a colon, then at most 20 words "
    "of reason. A contradiction means both cannot be true at once."
)


def _llm_score(llm_fn: Callable[[str], str],
               old_text: str, new_text: str) -> tuple[float, str]:
    """Ask the model whether two records contradict. Fails closed."""
    try:
        raw = (llm_fn(
            f"{_LLM_PROMPT}\n\nOLD: {old_text[:500]}\nNEW: {new_text[:500]}"
        ) or "").strip()
    except Exception as exc:  # noqa: BLE001
        _log.debug("contradiction llm adjudication failed: %s", exc)
        return 0.0, ""
    verdict = raw.split(":", 1)[0].strip().upper()
    reason = raw.split(":", 1)[1].strip()[:120] if ":" in raw else ""
    if verdict == "CONTRADICT":
        return 0.85, reason
    return 0.0, reason


# ── detection ────────────────────────────────────────────────────────────

def _eligible(record: Any) -> bool:
    """Only facts and preferences contradict — episodes are history."""
    try:
        if (record.metadata or {}).get("superseded_by"):
            return False  # already superseded: not the current belief
        return record.kind in (MemoryKind.FACT, MemoryKind.PREFERENCE)
    except Exception:  # noqa: BLE001
        return False


def _is_private(record: Any) -> bool:
    try:
        return bool((record.metadata or {}).get("private"))
    except Exception:  # noqa: BLE001
        return False


def detect_against(new_record: Any,
                   candidates: Sequence[Any],
                   *,
                   llm_fn: Callable[[str], str] | None = None,
                   min_confidence: float = 0.5) -> list[Contradiction]:
    """Run the strategy chain of ``new_record`` against ``candidates``.

    Returns contradictions sorted by confidence desc.  Never raises.
    """
    out: list[Contradiction] = []
    try:
        if not _eligible(new_record) or _is_private(new_record):
            return []
        new_text = (new_record.content or "").strip()
        if not new_text:
            return []
        for old in candidates:
            try:
                if old.id == new_record.id:
                    continue
                if not _eligible(old) or _is_private(old):
                    continue
                old_text = (old.content or "").strip()
                if not old_text:
                    continue
                shared = sorted(_topic_words(old_text) & _topic_words(new_text))
                # every strategy votes; the strongest verdict wins the pair
                best: Contradiction | None = None
                for strategy in STRATEGIES:
                    if strategy.name == "llm_adjudicate":
                        if llm_fn is None:
                            continue
                        conf, reason = _llm_score(llm_fn, old_text, new_text)
                        evidence = {"shared": shared, "reason": reason}
                    else:
                        conf = strategy.detect(old_text, new_text, old,
                                               new_record)
                        if conf < min_confidence:
                            continue
                        evidence = {
                            "shared": shared,
                            "similarity": round(_similarity(old_text,
                                                            new_text), 3),
                            "old_values": sorted({v.strip().lower()
                                                  for v in _values(old_text)}),
                            "new_values": sorted({v.strip().lower()
                                                  for v in _values(new_text)}),
                        }
                    if conf < min_confidence:
                        continue
                    if best is None or conf > best.confidence:
                        best = Contradiction(
                            older_id=old.id, newer_id=new_record.id,
                            strategy=strategy.name, confidence=conf,
                            evidence=evidence,
                            note=strategy.describe(old_text, new_text,
                                                   evidence))
                if best is not None:
                    out.append(best)
            except Exception:  # noqa: BLE001 — one bad pair never kills the scan
                continue
    except Exception as exc:  # noqa: BLE001
        _log.debug("contradiction detection failed: %s", exc)
    out.sort(key=lambda c: -c.confidence)
    return out


def detect_for(manager: Any, record_id: str, *,
               llm_fn: Callable[[str], str] | None = None,
               limit: int = 12,
               min_confidence: float = 0.5) -> list[Contradiction]:
    """Detect contradictions for one stored record. Never raises."""
    try:
        new_record = manager.get(record_id)
        if new_record is None:
            return []
        result = manager.recall(new_record.content, limit=limit,
                                kind=new_record.kind)
        candidates = [r for r in result.records if r.id != record_id]
        return detect_against(new_record, candidates, llm_fn=llm_fn,
                              min_confidence=min_confidence)
    except Exception as exc:  # noqa: BLE001
        _log.debug("detect_for failed: %s", exc)
        return []


# ── resolution (additive — supersede, never delete) ──────────────────────

def resolve(manager: Any, contradiction: Contradiction, *,
            dry_run: bool = False) -> dict[str, Any]:
    """Apply a contradiction: mark the old record ``superseded_by`` the new.

    Additive by construction — the old record keeps its content and stays
    queryable via ``get()`` / ``supersession_chain()``; it is simply no
    longer the current belief.  Returns the outcome dict; never raises.
    """
    outcome: dict[str, Any] = {"ok": False,
                               "strategy": contradiction.strategy,
                               "dry_run": dry_run}
    try:
        old = manager.get(contradiction.older_id)
        new = manager.get(contradiction.newer_id)
        if old is None or new is None:
            outcome["error"] = "record not found"
            return outcome
        if (old.metadata or {}).get("superseded_by"):
            outcome["ok"] = True
            outcome["already"] = True
            return outcome
        md = dict(old.metadata or {})
        md["superseded_by"] = new.id
        md["superseded_at"] = time.time()
        md["supersede_reason"] = (
            f"contradiction:{contradiction.strategy}")
        if dry_run:
            outcome["ok"] = True
            outcome["would_supersede"] = old.id
            outcome["note"] = contradiction.note
            return outcome
        manager.update(old.id, metadata=md)
        # back-link on the new record so supersession_chain() walks the
        # full history — the same shape manager.supersede() produces
        new_md = dict(new.metadata or {})
        if not new_md.get("supersedes"):
            new_md["supersedes"] = old.id
            manager.update(new.id, metadata=new_md)
        outcome["ok"] = True
        outcome["superseded"] = old.id
        outcome["by"] = new.id
        outcome["note"] = (
            f"noted: “{old.content.strip()[:80]}” no longer holds — "
            f"updated to “{new.content.strip()[:80]}”. still true?")
    except Exception as exc:  # noqa: BLE001
        outcome["error"] = str(exc)[:200]
    return outcome
