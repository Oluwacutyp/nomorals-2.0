"""Memory extraction: mining durable knowledge out of conversation.

After every turn the extractor reads what he said (and, when useful,
what she replied) and decides whether it contains something that should
outlive the chat history — a fact about him, a preference, a decision
they made together, or something said about the relationship itself.

Two passes:

* **Heuristic** (always on, free): sentence-level patterns for the common
  cases — "I live in …", "I'd rather …", "let's use …", "you're my …".
  Runs on the C++ kernel (``nomorals/native/memextract.cpp``) when it is
  built — same patterns, verified element-for-element — else pure Python.
* **LLM** (opt-in via ``settings.memory.extract_llm``): a single cheap
  completion that returns a JSON array for the long tail the regexes
  miss. Fails closed to the heuristic result.

Every candidate is **deduped against recall** before it is stored: if an
existing memory is near-identical (fuzzy ratio or containment), the new
candidate is dropped and the existing one is re-touched instead.

Wired in as a fire-and-forget thread from :mod:`partner_runtime`, so this
module must never raise into the reply path and must never block it.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..llm.brain import brain_for
from ..core.logging_setup import get_logger
from .base import ALL_KINDS, MemoryKind

_log = get_logger(__name__)

__all__ = ["ExtractedMemory", "MemoryExtractor", "extract_sentences"]

# ── heuristics ───────────────────────────────────────────────────────────────

#: kind -> (pattern, default importance)
_PREFERENCE = re.compile(
    r"\b(?:i prefer|i'?d rather|i would rather|don'?t (?:call|text|message|post|send|ask) me"
    r"|never (?:send|post|message|call)|always (?:send|post|message)|call me "
    r"|address me as|i want you to (?:always|never|stop|keep)|stop (?:sending|asking)"
    r"|i (?:like|love|hate) (?:it when|when) (?:you|you')|keep (?:it|things) (?:short|simple|casual))\b",
    re.IGNORECASE,
)
_DECISION = re.compile(
    r"\b(?:let'?s (?:do|use|try|go|start|make|pick|stick with)"
    r"|i (?:just )?decided (?:to|on|that)|we (?:should|will|'ll|can) (?:do|use|go|start|pick)"
    r"|going with|i'?m going to (?:start|begin|try|use)|final (?:answer|decision):?)\b",
    re.IGNORECASE,
)
_RELATIONSHIP = re.compile(
    r"\b(?:you'?re my|you are my|i (?:really )?(?:like|love|miss|appreciate) (?:you|that|when you)"
    r"|best friend|you make me|i'?m glad (?:you|that you|to have you|we)"
    r"|how (?:do|did) you (?:feel|think) about (?:us|me|yourself|what we)"
    r"|i (?:need|want) (?:you|us) to)\b",
    re.IGNORECASE,
)
_FACT_SELF = re.compile(
    r"\b(?:i (?:live|work|study|lived|worked|studied) (?:in|as|at|for|on)"
    r"|(?:i'?m|i am) (?:from|a|an|about|in \w+)"
    r"|i (?:just |still |already |finally |going to )?(?:got|have|had|bought|sold|named) "
    r"|i (?:just |still |already |finally )?(?:went|started|finished|moved|joined))\b",
    re.IGNORECASE,
)
_FACT_ABOUT_HIM = re.compile(
    r"\b(?:you (?:live|work|study|are from|come from)|your name is"
    r"|you (?:like|love|prefer|don'?t like|hate) (?:it when|when)? )\b",
    re.IGNORECASE,
)

_KIND_IMPORTANCE = {
    MemoryKind.PREFERENCE: 0.8,
    MemoryKind.DECISION: 0.7,
    MemoryKind.RELATIONSHIP: 0.65,
    MemoryKind.FACT: 0.6,
}

#: small auto-tag lexicon — keeps /recall tag filters useful without an LLM call
_TAG_LEXICON: dict[str, str] = {
    "coffee": "coffee", "tea": "coffee", "espresso": "coffee",
    "sleep": "sleep", "insomnia": "sleep", "tired": "sleep", "bedtime": "sleep",
    "gym": "fitness", "workout": "fitness", "running": "fitness", "protein": "fitness",
    "money": "money", "salary": "money", "broke": "money", "budget": "money", "pay": "money",
    "work": "work", "job": "work", "boss": "work", "office": "work", "deadline": "work",
    "family": "family", "mum": "family", "mom": "family", "dad": "family", "sister": "family",
    "brother": "family", "wife": "family", "husband": "family",
    "food": "food", "eat": "food", "eating": "food", "cook": "food", "chef": "food",
    "music": "music", "song": "music", "playlist": "music",
    "travel": "travel", "flight": "travel", "trip": "travel", "holiday": "travel",
    "tech": "tech", "code": "tech", "coding": "tech", "script": "tech", "server": "tech",
    "health": "health", "sick": "health", "illness": "health", "doctor": "health",
    "lagos": "lagos", "nigeria": "nigeria", "abuja": "abuja",
}

_MIN_LEN, _MAX_LEN = 4, 240
_MAX_PER_TURN = 4


def extract_sentences(text: str) -> list[str]:
    """Split into candidate sentences (newlines + . ! ? boundaries)."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    out: list[str] = []
    for part in parts:
        s = " ".join(part.split())
        if _MIN_LEN <= len(s) <= _MAX_LEN and s.rstrip(".!?") != "":
            out.append(s)
    return out


@dataclass
class ExtractedMemory:
    """One candidate from a turn, with its provenance."""

    kind: str
    content: str
    importance: float
    tags: list[str] = field(default_factory=list)
    pass_name: str = "heuristic"


def _clean(text: str) -> str:
    s = " ".join((text or "").split())
    s = re.sub(r"^[,;:\-–\s]+", "", s)
    return s[:_MAX_LEN]


def _tags_for(text: str) -> list[str]:
    low = text.lower()
    return sorted({tag for key, tag in _TAG_LEXICON.items() if key in low})


def _looks_like_question(s: str) -> bool:
    return s.strip().endswith("?") or bool(re.match(r"^\s*(?:what|why|how|when|where|who|do|does|did)\b", s, re.IGNORECASE))


def _heuristic_pass(text: str) -> list[ExtractedMemory]:
    # wave 94: the C++ kernel runs the same pass (sentence split, the five
    # verbatim patterns, clean, tag, dedupe, cap) — verified element-for-
    # element against this Python reference.  It returns None when the
    # kernel is absent or refused the buffer, in which case the Python
    # reference below is the fallback.
    try:
        from ..native import mem_heuristic
        native_result = mem_heuristic(text)
        if native_result is not None:
            return [
                ExtractedMemory(kind=kind, content=content,
                                importance=importance, tags=list(tags))
                for kind, importance, content, tags in native_result
            ]
    except Exception:  # noqa: BLE001 - a kernel hiccup must not kill extraction
        _log.debug("native memory extraction unavailable; using Python pass")
    return _heuristic_pass_python(text)


def _heuristic_pass_python(text: str) -> list[ExtractedMemory]:
    """The reference implementation (also the fallback for the C++ kernel)."""
    found: list[ExtractedMemory] = []
    seen: set[str] = set()
    for sentence in extract_sentences(text):
        low = sentence.lower()
        kind = ""
        if _PREFERENCE.search(low):
            kind = MemoryKind.PREFERENCE
        elif _DECISION.search(low):
            kind = MemoryKind.DECISION
        elif _RELATIONSHIP.search(low):
            kind = MemoryKind.RELATIONSHIP
        elif _FACT_ABOUT_HIM.search(sentence):
            kind = MemoryKind.FACT
        elif _FACT_SELF.search(sentence) and not _looks_like_question(sentence):
            kind = MemoryKind.FACT
        if not kind:
            continue
        content = _clean(sentence)
        key = content.lower().rstrip(".!?")
        if not content or key in seen:
            continue
        seen.add(key)
        found.append(ExtractedMemory(
            kind=kind,
            content=content,
            importance=_KIND_IMPORTANCE.get(kind, 0.6),
            tags=_tags_for(content),
        ))
        if len(found) >= _MAX_PER_TURN:
            break
    return found


_LLM_PROMPT = (
    "You extract durable personal memories from one chat turn. "
    'Reply with a JSON array (or []) of objects: {"kind": "fact"|"preference"|"decision"|"relationship", '
    '"content": "one clean sentence", "importance": 0.0 to 1.0}. '
    "Only include things worth remembering days or weeks later — never small talk, "
    "never questions, never things that only make sense inside this one turn. "
    "Output the JSON array only, no prose."
)


def _llm_pass(context: Any, user_text: str, assistant_text: str, speaker: str) -> list[ExtractedMemory]:
    router = getattr(context, "router", None)
    if router is None:
        return []
    from ..llm.base import Message, SamplingParams

    user_prompt = (
        f"Speaker: {speaker or 'him'}\n"
        f"Him: {user_text[:1200]}\n"
        + (f"Her: {assistant_text[:600]}\n" if assistant_text else "")
    )
    try:
        # NB: module-level function — the context is the *argument*, not
        # ``self``.  A previous revision referenced ``self.context`` here,
        # which raised NameError on every call and silently killed the whole
        # LLM pass (the except below swallowed it).  This is the fix.
        response = brain_for(context).chat(
            [
                Message(role="system", content=_LLM_PROMPT),
                Message(role="user", content=user_prompt),
            ],
            params=SamplingParams(max_tokens=300, temperature=0.0),
        task_kind="extract")
        raw = (getattr(response, "text", "") or "").strip()
    except Exception:  # noqa: BLE001 - LLM pass is best-effort
        _log.debug("memory extraction LLM pass failed; using heuristics only")
        return []

    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        items = json.loads(raw[start : end + 1])
    except (ValueError, TypeError):
        return []
    if not isinstance(items, list):
        return []

    out: list[ExtractedMemory] = []
    for item in items[:_MAX_PER_TURN]:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "").strip().lower()
        content = _clean(str(item.get("content") or ""))
        if kind not in ALL_KINDS or len(content) < _MIN_LEN:
            continue
        try:
            importance = max(0.0, min(1.0, float(item.get("importance", 0.6))))
        except (TypeError, ValueError):
            importance = _KIND_IMPORTANCE.get(kind, 0.6)
        out.append(ExtractedMemory(
            kind=kind, content=content, importance=importance,
            tags=_tags_for(content), pass_name="llm",
        ))
    return out


# ── dedupe ───────────────────────────────────────────────────────────────────

_PUNCT = re.compile(r"[^\w\s]")


def _normalize(s: str) -> str:
    return _PUNCT.sub(" ", (s or "").lower()).strip()


def is_duplicate(candidate: str, existing: str, ratio: float) -> bool:
    """Near-identical? Fuzzy ratio or one containing the other (min 20 chars)."""
    a, b = _normalize(candidate), _normalize(existing)
    if not a or not b:
        return False
    if a == b:
        return True
    if min(len(a), len(b)) >= 20 and (a in b or b in a):
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= ratio


# ── the extractor ────────────────────────────────────────────────────────────


class MemoryExtractor:
    """Mine one conversation turn for durable memories. Never raises."""

    def __init__(self, context: Any) -> None:
        self.context = context

    # settings
    @property
    def _settings(self) -> Any:
        return getattr(getattr(self.context, "settings", None), "memory", None)

    def _enabled(self) -> bool:
        return bool(getattr(self._settings, "extract_enabled", True))

    def _use_llm(self) -> bool:
        return bool(getattr(self._settings, "extract_llm", False))

    def _dedupe_ratio(self) -> float:
        return float(getattr(self._settings, "extract_dedupe_ratio", 0.80))

    def extract_turn(
        self,
        user_text: str,
        *,
        assistant_text: str = "",
        chat_key: str = "",
        speaker: str = "",
        is_owner: bool = True,
    ) -> list[dict[str, Any]]:
        """Run both passes, dedupe against recall, store the rest.

        Returns one action per candidate:
        ``{"action": "stored"|"duplicate"|"skipped:<why>", "kind", "content", "id"}``.
        """
        actions: list[dict[str, Any]] = []
        try:
            return self._extract(user_text, assistant_text, chat_key, speaker, is_owner, actions)
        except Exception:  # noqa: BLE001 - extraction must never break the reply
            _log.exception("memory extraction crashed; turn skipped")
            actions.append({"action": "skipped:crash", "kind": "", "content": user_text[:80]})
            return actions

    def _extract(
        self, user_text: str, assistant_text: str, chat_key: str,
        speaker: str, is_owner: bool, actions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        memory = getattr(self.context, "memory", None)
        if memory is None or not self._enabled():
            return actions
        text = (user_text or "").strip()
        if len(text) < 8 or len(text) > 800:
            return actions

        candidates = _heuristic_pass(text)
        if self._use_llm():
            for cand in _llm_pass(self.context, text, assistant_text, speaker):
                if all(not is_duplicate(cand.content, c.content, self._dedupe_ratio())
                       for c in candidates):
                    candidates.append(cand)
        if not candidates:
            return actions

        # Attribute non-owner speakers so group facts stay about the right person.
        prefix = "" if is_owner else (f"{speaker}: " if speaker else "")

        for cand in candidates[:_MAX_PER_TURN]:
            content = prefix + cand.content
            action = self._store(memory, cand, content, chat_key)
            actions.append(action)
        return actions

    def _store(self, memory: Any, cand: ExtractedMemory, content: str, chat_key: str) -> dict[str, Any]:
        from .base import MemoryRecord as _MemoryRecord

        ratio = self._dedupe_ratio()
        existing = memory.recall(content, limit=4)
        dup_of = None
        for record in existing.records:
            if is_duplicate(content, record.content, ratio):
                dup_of = record
                break
        # Contradiction check on the write path: a new fact/preference that
        # disagrees with an older one supersedes it (additive — the old
        # record is marked, never deleted).  Reuses the dedupe recall, so
        # this costs no extra query.  Crucially it runs BEFORE the duplicate
        # decision: a near-duplicate that contradicts is an *update*, not a
        # dupe ("deadline oct 20" → "deadline oct 25" are ~87% similar —
        # dedupe alone would swallow the update).
        contradictions: list[Any] = []
        if self._wants_contra_check(cand):
            try:
                from .contradictions import (AUTO_RESOLVE_CONFIDENCE,
                                             detect_against)
                probe = _MemoryRecord(id="", kind=cand.kind, content=content)
                found = detect_against(probe, existing.records)
                contradictions = [c for c in found
                                  if c.confidence >= AUTO_RESOLVE_CONFIDENCE]
            except Exception as exc:  # noqa: BLE001 — never break extraction
                _log.debug("extraction contradiction check failed: %s", exc)
        if dup_of is not None and not contradictions:
            memory._touch([dup_of.id])  # reinforce instead of re-storing
            return {"action": "duplicate", "kind": dup_of.kind,
                    "content": content[:120], "id": dup_of.id}
        record_id = memory.remember(
            content,
            kind=cand.kind,
            importance=cand.importance,
            source=f"extraction:{cand.pass_name}",
            tags=cand.tags,
            origin=f"chat:{chat_key}" if chat_key else "",
            agent="memory_extractor",
        )
        action: dict[str, Any] = {"action": "stored", "kind": cand.kind,
                                  "content": content[:120], "id": record_id}
        if record_id and contradictions:
            try:
                from .contradictions import resolve
                notes = []
                for contra in contradictions:
                    contra.newer_id = record_id
                    if resolve(memory, contra).get("ok"):
                        notes.append(contra.note)
                if notes:
                    action["contradictions"] = notes
            except Exception as exc:  # noqa: BLE001
                _log.debug("extraction contradiction resolve failed: %s",
                           exc)
        return action

    def _wants_contra_check(self, cand: ExtractedMemory) -> bool:
        return (bool(getattr(self._settings, "extract_contradictions", True))
                and cand.kind in (MemoryKind.FACT, MemoryKind.PREFERENCE))
