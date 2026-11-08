"""Outcome → presented reply: typography card + action buttons.

Anime results are presented surfaces (DESIGN §27): composed as a
typography :class:`Card` (bold title, tabular stat rows, flavor
note, the configured stamp) and rendered in house dialect — the
gateway's presentation layer transcodes per platform at send.
Pick lists ride as real buttons via ``actionable`` so Telegram
gets taps and WhatsApp gets the honest numbered fallback that
actually executes (register #27), authored from the same state
that rendered the list.
"""

from __future__ import annotations

from typing import Any

from ..social.chat.tgbot_buttons import actionable
from ..social.chat.typography import Card, card_text
from .service import Outcome

__all__ = ["present_outcome"]


def present_outcome(outcome: Outcome) -> Any:
    """Render one Outcome to a reply (str, or ActionableMessage)."""
    if not outcome.ok:
        card = Card("announcement", title=outcome.title or "Anime",
                    icon=outcome.icon or "🎌")
        card.line(outcome.error or "that didn't work.")
        return card_text(card)
    card = Card("announcement", title=outcome.title, icon=outcome.icon)
    for line in outcome.lines:
        card.line(line)
    if outcome.stats:
        card.stats(outcome.stats)
    if outcome.note:
        card.note(outcome.note)
    text = card_text(card)
    if outcome.picks:
        # Picks pack two per row (the grid — the house standard
        # for buttons, tgbot_buttons.grid_layout), not one tall
        # single-button column.
        buttons = [list(outcome.picks[i:i + 2])
                   for i in range(0, len(outcome.picks), 2)]
        return actionable(text, buttons)
    return text
