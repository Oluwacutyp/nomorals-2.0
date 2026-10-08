"""Heuristic complexity classification for multi-model routing (build-map #18).

``classify_complexity`` buckets a prompt into ``easy`` / ``medium`` / ``hard``
so the task router can send cheap work to cheap models and hard work to
strong ones.  It is deliberately *offline and instant* (a handful of regex
scans — microseconds, no LLM call): it runs on the hot path before every
routed call, so it must never add latency.

Signals
--------
* **easy** — greetings, clock/date questions, short translations, tiny
  factual lookups ("what time is it", "translate hello to Spanish").
* **medium** — the safe default: multi-step research, drafting, debugging
  help, ordinary code writing, explanations.
* **hard** — architecture/design/strategy language, refactor or multi-file
  work, fenced code blocks, very long or multi-part requests.

The classifier never raises: on any failure it returns ``("medium", 0.5)``,
the safe middle — routing then behaves exactly as it did before
complexity existed.
"""

from __future__ import annotations

import re

__all__ = ["COMPLEXITIES", "classify_complexity"]

#: The complexity tiers, cheapest-first.
COMPLEXITIES = ("easy", "medium", "hard")

# Compiled once at import; the hot path is a few regex scans.
_EASY_RES = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^(hi|hey|hello|yo|good\s?(morning|afternoon|evening|night))\b",
        r"\bwhat time is it\b",
        r"\bwhat('s| is) the (date|day today|time)\b",
        r"\btranslate\b.{0,60}\bto\b",          # "translate hello to Spanish"
        r"^what is (a |an |the )?[\w\s\-']{1,40}\??$",
        r"^who is [\w\s\-']{1,40}\??$",
        r"^define [\w\-]+$",
        r"^summarise this in one sentence",
        r"^summarize this in one sentence",
    )
)

_HARD_RES = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"```",                                 # fenced code blocks
        r"\b(architect|architecture)\b",
        r"\b(refactor|redesign|re-?architect)\b",
        r"\bdesign (a|the|this)\b",
        r"\bstrateg(y|ic)\b",
        r"\btrade-?offs?\b",
        r"\bmulti-?file\b",
        r"\b\d+\s+files?\b",                  # "across 12 files"
        r"\bcodebase\b",
        r"\bdistributed systems?\b",
        r"\bmigration plan\b",
        r"\bnovel\b.{0,20}\b(approach|algorithm|design)\b",
    )
)

_MEDIUM_RES = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bwrite (a|the) (function|script|class|module)\b",
        r"\bdebug(gin)?\b",
        r"\bhow do i\b",
        r"\bwhy does\b",
        r"\bexplain\b",
        r"\bdraft\b",
    )
)

_LIST_ITEM_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+\S", re.MULTILINE)


def _count_hits(text: str, patterns: tuple) -> int:
    return sum(1 for rx in patterns if rx.search(text))


def classify_complexity(
    text: str, *, task_type: str | None = None
) -> tuple[str, float]:
    """Bucket ``text`` into ``("easy"|"medium"|"hard", confidence)``.

    ``task_type`` (e.g. ``"coding"``) is an optional hint that nudges the
    score; ``None`` means "no hint".  Confidence is 0–1.  Never raises.
    """
    try:
        return _classify(str(text or ""), task_type=task_type)
    except Exception:  # noqa: BLE001 — classifier must never break the caller
        return "medium", 0.5


def _classify(text: str, *, task_type: str | None) -> tuple[str, float]:
    stripped = text.strip()
    if not stripped:
        return "medium", 0.5

    easy_hits = _count_hits(stripped, _EASY_RES)
    hard_hits = _count_hits(stripped, _HARD_RES)
    medium_hits = _count_hits(stripped, _MEDIUM_RES)
    length = len(stripped)

    score = 0.0
    score -= min(easy_hits, 2) * 1.0
    score += min(hard_hits, 3) * 1.5
    score += min(medium_hits, 2) * 0.25  # weak pull toward the middle

    # Length bands: tiny asks lean trivial (weakly — a pattern must agree
    # for a confident "easy"); long ones rarely are.
    if length < 80:
        score -= 0.5
    elif length > 700:
        score += 1.0
    if length > 2000:
        score += 1.0

    # Multi-part requests (several questions, numbered lists) lean hard.
    questions = stripped.count("?")
    if questions > 1:
        score += 0.5 * min(questions - 1, 3)
    list_items = len(_LIST_ITEM_RE.findall(stripped))
    if list_items >= 2:
        score += 1.0

    # task_type hint: coding-flavoured hard signals count a bit more;
    # short chat is a bit more likely trivial.
    hint = (task_type or "").strip().lower()
    if hint in {"coding", "reasoning", "planning"} and hard_hits:
        score += 0.5
    if hint == "chat" and length < 120 and not hard_hits:
        score -= 0.5

    if score <= -1.0:
        level = "easy"
    elif score >= 1.5:
        level = "hard"
    else:
        level = "medium"

    total_hits = easy_hits + hard_hits + medium_hits
    confidence = min(0.95, 0.45 + 0.18 * total_hits + 0.08 * abs(score))
    if total_hits == 0 and level == "medium":
        confidence = 0.5  # genuinely unsure: the safe middle
    return level, round(confidence, 3)
