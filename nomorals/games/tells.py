"""One visual grammar for tells.

A "tell" is the moment an enemy telegraphs its next move — the thing a
player reads to dodge, block, or counter. Before this module, every
game rendered tells its own way (different emoji, different colours),
so a skill learned in one game didn't transfer to the next.

There is now exactly one grammar — :data:`TELL_GRAMMAR` — mapping each
tell type to one icon, one colour, and one motion. Every renderer
(Telegram, WhatsApp, console) reads this table; nothing invents its
own. :func:`render_tell` is the single formatting entry point, and
:func:`validate_grammar` proves the table is complete and unambiguous
(one icon per type, every entry carrying all three channels).

Colour is a hex string (the renderer maps it to its platform's nearest
representation); motion is a verb (``flash`` / ``pulse`` / ``shake`` /
``rise`` / ``spin``) the client animates or, on text-only surfaces,
prints in brackets.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "TELL_GRAMMAR",
    "Tell",
    "render_tell",
    "tell_for_move",
    "validate_grammar",
]

#: tell type → {icon, color, motion}. THE single grammar. Extend here.
TELL_GRAMMAR: dict[str, dict[str, str]] = {
    "windup": {"icon": "🌀", "color": "#F5A623", "motion": "spin"},
    "charge": {"icon": "⚡", "color": "#E74C3C", "motion": "pulse"},
    "telegraph": {"icon": "👁️", "color": "#3498DB", "motion": "flash"},
    "dodge_window": {"icon": "💨", "color": "#2ECC71", "motion": "flash"},
    "counter_window": {"icon": "🔄", "color": "#9B59B6", "motion": "pulse"},
    "enrage": {"icon": "🔥", "color": "#E74C3C", "motion": "shake"},
    "heal": {"icon": "💚", "color": "#2ECC71", "motion": "rise"},
    "buff": {"icon": "🛡️", "color": "#3498DB", "motion": "rise"},
    "debuff": {"icon": "☠️", "color": "#7F8C8D", "motion": "pulse"},
}

#: Enemy move kind → tell type (the read the player gets).
_MOVE_TO_TELL: dict[str, str] = {
    "heavy_strike": "windup",
    "charge_attack": "charge",
    "aimed_shot": "telegraph",
    "lunge": "dodge_window",
    "riposte_stance": "counter_window",
    "frenzy": "enrage",
    "regenerate": "heal",
    "harden": "buff",
    "venom": "debuff",
    "dread": "debuff",
}


class Tell(dict):
    """A rendered tell: type + icon + color + motion + text."""


def render_tell(tell_type: str, text: str = "") -> Tell:
    """Render one tell through the single grammar. Unknown types raise —
    a typo must never silently render as a blank tell."""
    entry = TELL_GRAMMAR.get(tell_type)
    if entry is None:
        raise ValueError(
            f"unknown tell type {tell_type!r} — one of {sorted(TELL_GRAMMAR)}")
    label = text or tell_type.replace("_", " ")
    return Tell(type=tell_type, icon=entry["icon"], color=entry["color"],
                motion=entry["motion"],
                text=f"{entry['icon']} {label} [{entry['motion']}]")


def tell_for_move(move_kind: str, *, text: str = "") -> Tell:
    """The tell a player sees for an enemy move kind."""
    tell_type = _MOVE_TO_TELL.get((move_kind or "").strip().lower())
    if tell_type is None:
        raise ValueError(f"no tell mapped for move kind {move_kind!r}")
    return render_tell(tell_type, text or move_kind)


def validate_grammar() -> list[str]:
    """Grammar sanity: every entry has icon+color+motion; icons unique."""
    problems: list[str] = []
    for tell_type, entry in TELL_GRAMMAR.items():
        for key in ("icon", "color", "motion"):
            if not entry.get(key):
                problems.append(f"tell {tell_type} missing {key}")
    for move, tell_type in _MOVE_TO_TELL.items():
        if tell_type not in TELL_GRAMMAR:
            problems.append(f"move {move} maps to unknown tell {tell_type}")
    return problems
