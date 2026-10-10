"""Proactive recall — memories that surface unprompted.

The brain's tool loop can call ``memory_recall`` deliberately. This module
covers the other direction: given an incoming message, find memories
relevant enough to inject into context *without being asked*. The bar is
high — only memories that would genuinely change the reply get surfaced,
so the prompt doesn't drown in trivia.

Also hosts the autonomous distillation pass: after a conversation, decide
what (if anything) is worth keeping. She decides, not just what she's told.
"""

from __future__ import annotations

from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

#: Minimum blended score for a memory to surface unprompted. High on
#: purpose — proactive recall must earn its prompt tokens.
PROACTIVE_THRESHOLD = 0.55

#: Cap on surfaced memories per message.
PROACTIVE_LIMIT = 3


def surface(message_text: str, memory: Any, *,
            threshold: float = PROACTIVE_THRESHOLD,
            limit: int = PROACTIVE_LIMIT,
            origin: str = "") -> list[str]:
    """Return context lines for memories worth surfacing unprompted.

    Never raises — a failure here must never break message handling.
    """
    if not message_text or not memory:
        return []
    try:
        result = memory.recall(message_text, limit=limit * 2,
                               origin=origin or None)
    except Exception as exc:  # noqa: BLE001
        _log.debug("proactive recall failed: %s", exc)
        return []
    records = getattr(result, "records", []) or []
    lines: list[str] = []
    for record in records:
        score = float(getattr(record, "score", 0.0) or 0.0)
        if score < threshold:
            continue
        # MemoryRecord exposes .content; be liberal for duck-typed records.
        text = (getattr(record, "content", "")
                or getattr(record, "text", "") or "").strip()
        if not text:
            continue
        kind = getattr(record, "kind", "") or "memory"
        lines.append(f"💡 [remembered {kind}] {text}")
        if len(lines) >= limit:
            break
    return lines


def distill_candidates(chat_key: str, recent_messages: list[str],
                       memory: Any, *, limit: int = 5) -> list[dict[str, Any]]:
    """Propose memories worth keeping from a recent exchange.

    Returns candidate dicts (text/kind/importance) for the brain to approve
    or refine — the model makes the final call on what gets remembered.
    Pure retrieval + heuristics here; the LLM judges.
    """
    if not recent_messages or not memory:
        return []
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        for message in recent_messages[-6:]:
            text = (message or "").strip()
            if len(text) < 40 or text in seen:
                continue
            seen.add(text)
            # Skip if it's already well-covered in memory.
            result = memory.recall(text, limit=2)
            records = getattr(result, "records", []) or []
            best = max(
                (float(getattr(r, "score", 0.0) or 0.0) for r in records),
                default=0.0,
            )
            if best > 0.75:
                continue  # already remembered
            candidates.append({
                "text": text[:500],
                "kind": "episode",
                "importance": 0.5,
                "source": f"distill:{chat_key}",
            })
            if len(candidates) >= limit:
                break
    except Exception as exc:  # noqa: BLE001
        _log.debug("distillation candidate scan failed: %s", exc)
    return candidates
