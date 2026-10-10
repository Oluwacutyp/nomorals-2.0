"""Character arcs: growth over time.

Characters change. Beliefs get challenged and revised (with confidence
+ revision count — livingfeed gold). Milestones mark development. The
arc is the story of who a character is becoming, not just who they are.
"""
from __future__ import annotations

import time
from typing import Any

from .character import Character


def add_belief(char: Character, text: str,
               confidence: float = 0.6) -> dict[str, Any]:
    """A character forms a belief. Returns the belief record."""
    text = (text or "").strip()
    if not text:
        return {}
    # don't duplicate
    low = text.lower()
    for b in char.beliefs:
        if str(b.get("text", "")).lower() == low:
            return b
    belief = {"text": text[:300], "confidence": max(0.0, min(1.0, confidence)),
              "revisions": 0, "formed_at": time.time()}
    char.beliefs.append(belief)
    return belief


def challenge_belief(char: Character, text: str, new_confidence: float,
                     note: str = "") -> dict[str, Any]:
    """Something challenged a belief — confidence shifts, revision noted.
    Beliefs don't vanish; they evolve. High-revision beliefs are the
    character's scar tissue."""
    low = (text or "").strip().lower()
    for b in char.beliefs:
        if str(b.get("text", "")).lower() == low:
            b["confidence"] = max(0.0, min(1.0, float(new_confidence)))
            b["revisions"] = int(b.get("revisions") or 0) + 1
            if note:
                hist = b.setdefault("history", [])
                hist.append({"ts": time.time(), "note": note[:200]})
            return b
    return add_belief(char, text, new_confidence)


def drop_belief(char: Character, text: str) -> bool:
    """A belief dies — only when confidence already cratered. Returns True
    if removed."""
    low = (text or "").strip().lower()
    for i, b in enumerate(char.beliefs):
        if str(b.get("text", "")).lower() == low:
            if float(b.get("confidence", 0)) > 0.25:
                return False  # too alive to drop — challenge it first
            char.beliefs.pop(i)
            char.remember(f"I stopped believing: {text[:150]}", 0.7)
            return True
    return False


def milestone(char: Character, title: str, description: str = "") -> None:
    """Mark a development milestone — a moment the character changed."""
    char.remember(f"🏛️ MILESTONE — {title}: {description[:250]}", 0.95)


def arc_summary(char: Character) -> str:
    """Who this character is becoming, in their own trajectory."""
    lines = [f"Arc of {char.name}:"]
    # beliefs with movement
    moved = [b for b in char.beliefs if int(b.get("revisions") or 0) > 0]
    if moved:
        lines.append("Evolving beliefs:")
        for b in moved[:4]:
            lines.append(f"  • \"{b.get('text','')[:70]}\" "
                         f"(confidence {float(b.get('confidence',0)):.1f}, "
                         f"revised {b.get('revisions')}×)")
    fresh = [b for b in char.beliefs if int(b.get("revisions") or 0) == 0][:3]
    if fresh:
        lines.append("Held beliefs: " + "; ".join(
            f"\"{b.get('text','')[:50]}\"" for b in fresh))
    # milestones from memory
    stones = [m["text"] for m in char.memory
              if str(m.get("text","")).startswith("🏛️ MILESTONE")][-3:]
    if stones:
        lines.append("Milestones:")
        lines.extend(f"  {s[:120]}" for s in stones)
    if char.core_motive:
        lines.append(f"Core motive: {char.core_motive}")
    if len(lines) == 1:
        lines.append("(no arc yet — early days)")
    return "\n".join(lines)


def arc_story(char: Character) -> str:
    """The character's arc as a STORY — who they were, what's changing,
    where they're headed. This is the god-tier version of arc_summary:
    narrative, not bullets."""
    name = char.name or "They"
    parts = [f"📖 The story of {name} so far."]
    # reflections are the spine of the story (Stanford gold)
    refs = [r.get("text", "") for r in (char.reflections or [])
            if r.get("text")][-3:]
    if refs:
        parts.append("What " + name + " has figured out lately:")
        parts.extend(f"  💭 \"{r[:140]}\"" for r in refs)
    # scar tissue: heavily-revised beliefs
    scarred = sorted(
        (b for b in char.beliefs if int(b.get("revisions") or 0) >= 2),
        key=lambda b: -int(b.get("revisions") or 0))[:3]
    if scarred:
        parts.append("Scar tissue — beliefs life rewrote:")
        for b in scarred:
            parts.append(f"  🩹 \"{b.get('text','')[:90]}\" "
                         f"(revised {b.get('revisions')}×, now "
                         f"{float(b.get('confidence', 0)):.0%} sure)")
    # milestones as chapters
    stones = [str(m.get("text", "")).replace("🏛️ MILESTONE — ", "")
              for m in char.memory
              if str(m.get("text", "")).startswith("🏛️ MILESTONE")][-4:]
    if stones:
        parts.append("Chapters:")
        parts.extend(f"  🏛️ {s[:110]}" for s in stones)
    # goals: the road ahead
    try:
        agenda = char.goal_agenda()
    except Exception:
        agenda = ""
    if agenda:
        parts.append(f"Where {name} is headed: {agenda}.")
    achieved = [g for g, s in (char.goal_states or {}).items()
                if s.get("status") == "achieved"][-3:]
    if achieved:
        parts.append("Already done: " + "; ".join(a[:60] for a in achieved) + ".")
    if len(parts) == 1:
        parts.append(f"{name}'s story is still being written — early days.")
    elif char.core_motive:
        parts.append(f"Through it all, one thing never changed: {char.core_motive}")
    return "\n".join(parts)


def grow_from_interaction(char: Character, other: str, event: str,
                          outcome: str = "") -> None:
    """Lightweight growth hook: notable interactions can seed or shift
    beliefs. Called by the post-interaction processor."""
    notable = {
        "betrayed": ("People I trust can still hurt me.", 0.5),
        "was_helped_by": ("Accepting help doesn't make me weak.", 0.6),
        "deep_conversation": ("Being real with people matters.", 0.7),
        "won_game": ("Preparation beats talent when talent coasts.", 0.5),
        "argument": ("Being right isn't the same as being heard.", 0.55),
    }
    seed = notable.get(event)
    if not seed:
        return
    text, conf = seed
    existing = [b for b in char.beliefs
                if str(b.get("text","")).lower() == text.lower()]
    if existing:
        b = existing[0]
        challenge_belief(char, text,
                         min(1.0, float(b.get("confidence", 0.5)) + 0.1),
                         f"reinforced by {event} with {other}")
    else:
        add_belief(char, text, conf)
        char.remember(f"Learned from {other}: {text}", 0.65)
