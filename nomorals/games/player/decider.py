"""Decision engine: turn a :class:`GameState` into the next :class:`Move`.

Strategies are plain callables ``(state) -> Move | None`` registered by game
name. :func:`decide` picks the named strategy and falls back to
:func:`heuristic_strategy`, a conservative generic player that prefers
free gains (claim/collect/daily/work) and never touches spend/pay/delete.
Returning ``None`` means "nothing safe to do" — the session stops instead
of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from ...core.logging_setup import get_logger
from .driver import Action
from .state import GameState

_log = get_logger(__name__)

__all__ = [
    "Decider",
    "Move",
    "Strategy",
    "decide",
    "heuristic_strategy",
    "lagos_life_strategy",
    "register_strategy",
    "strategies",
]


@dataclass
class Move:
    """One decided action, with why, and whether it is hard to undo."""

    action: Action
    reason: str
    irreversible: bool = False
    confidence: float = 0.5


class Strategy(Protocol):
    def __call__(self, state: GameState) -> Move | None: ...


strategies: dict[str, Callable[[GameState], Move | None]] = {}


def register_strategy(name: str, fn: Callable[[GameState], Move | None]) -> None:
    """Register a per-game strategy. Later registrations replace earlier ones."""
    strategies[name.lower()] = fn
    _log.debug("registered game strategy %r", name.lower())


#: Labels that are (almost) always free gains, not spends.
_GAIN_KEYWORDS = (
    "claim",
    "collect",
    "daily bonus",
    "daily reward",
    "check in",
    "check-in",
    "work",
    "earn",
    "free",
)

#: Labels we never auto-pick: money out, destruction, or account changes.
_AVOID_KEYWORDS = (
    "delete",
    "destroy",
    "remove account",
    "pay $",
    "pay ₦",
    "real money",
    "top up",
    "topup",
    "deposit",
    "withdraw",
    "checkout",
    "subscribe",
)


def _is_avoid(label: str) -> bool:
    lowered = label.lower()
    return any(k in lowered for k in _AVOID_KEYWORDS)


def _first_gain(state: GameState) -> Move | None:
    for el in state.elements:
        if el.kind not in ("link", "button", "form"):
            continue
        lowered = el.label.lower()
        if _is_avoid(el.label):
            continue
        if any(k in lowered for k in _GAIN_KEYWORDS):
            kind = "submit" if el.kind == "form" else "click"
            return Move(
                action=Action(kind=kind, target=el.id, label=el.label),
                reason=f"free gain: {el.label!r}",
                irreversible=False,
                confidence=0.7,
            )
    return None


def heuristic_strategy(state: GameState) -> Move | None:
    """Conservative generic player: take free gains, otherwise stop."""
    return _first_gain(state)


def lagos_life_strategy(state: GameState) -> Move | None:
    """Lagos Life: keep the hustle loop running.

    Priority: daily bonus / claims first (free naira), then work shifts
    (steady income), then career progression. Property and rent actions are
    left alone — they move large sums and deserve the owner's eyes.
    """
    lowered_labels = [el.label.lower() for el in state.elements]
    _ = lowered_labels  # documented for future per-label tuning

    # 1. Free money first.
    for keywords in (("daily", "bonus"), ("claim",), ("collect",)):
        for el in state.find_elements(*keywords):
            if el.kind in ("link", "button", "form") and not _is_avoid(el.label):
                kind = "submit" if el.kind == "form" else "click"
                return Move(
                    action=Action(kind=kind, target=el.id, label=el.label),
                    reason=f"lagos life free naira: {el.label!r}",
                    irreversible=False,
                    confidence=0.8,
                )
    # 2. Work a shift for steady income.
    for el in state.find_elements("work", "shift", "job"):
        if el.kind in ("link", "button", "form") and not _is_avoid(el.label):
            # "quit job" / "leave job" would be destructive — skip those.
            if any(w in el.label.lower() for w in ("quit", "leave", "resign")):
                continue
            kind = "submit" if el.kind == "form" else "click"
            return Move(
                action=Action(kind=kind, target=el.id, label=el.label),
                reason=f"lagos life income: {el.label!r}",
                irreversible=False,
                confidence=0.6,
            )
    return None


register_strategy("lagos_life", lagos_life_strategy)
register_strategy("lagoslife", lagos_life_strategy)
register_strategy("generic", heuristic_strategy)


class Decider:
    """Picks the next move for a game."""

    def __init__(self, game: str = "generic") -> None:
        self.game = game.lower()

    def decide(self, state: GameState) -> Move | None:
        strategy = strategies.get(self.game) or heuristic_strategy
        move = strategy(state)
        _log.debug(
            "decider(%s): %s",
            self.game,
            f"{move.action.kind} {move.action.label!r} ({move.reason})"
            if move
            else "no move",
        )
        return move


def decide(game: str, state: GameState) -> Move | None:
    """One-shot convenience: decide the next move for ``game``."""
    return Decider(game).decide(state)
