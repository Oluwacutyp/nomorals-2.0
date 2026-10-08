"""Decision engine: turn a :class:`GameState` into the next :class:`Move`.

Strategies are plain callables ``(state) -> Move | None`` registered by game
name. :func:`decide` picks the named strategy and falls back to
:func:`heuristic_strategy`, a conservative generic player that prefers
free gains (claim/collect/daily/work) and never touches spend/pay/delete.
Returning ``None`` means "nothing safe to do" — the session stops instead
of guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

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
    "interpret_freeform",
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


#: Words too common to mean anything when matching freeform intent.
_FREEFORM_STOPWORDS = frozenset(
    "a an the to do it i me my you your we they them this that these those "
    "is are was were be been and or but if then so at on in of for with as "
    "please just now here there what how".split()
)

_WORD_RE = re.compile(r"[a-z0-9']+")


def _freeform_tokens(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower())
            if w not in _FREEFORM_STOPWORDS and len(w) > 1}


def _element_action_kind(el: Any) -> str:
    return "submit" if getattr(el, "kind", "") == "form" else "click"


def interpret_freeform(
    text: str,
    game: str,
    state: GameState,
    *,
    suggest: Callable[[str], str] | None = None,
) -> Move | None:
    """Map "do anything" input ("bribe the guard", "sneak around back")
    onto a real :class:`Move`.

    1. Keyword/intent matching: the text is tokenized and scored against
       every interactive element's label; a strong match becomes a
       click/submit on that element.
    2. With ``suggest`` (an LLM ``(prompt) -> text`` bridge): the model
       picks the closest option from the game's numbered move list, and
       the pick is validated back against the real elements.
    3. Otherwise a generic ``freeform`` Move carrying the raw text, so the
       game itself can interpret it ("sneak around back" has no button).

    Never returns an illegal move: click/submit targets must be real
    element ids from ``state``. Freeform intent is explicit player
    instruction, so avoid-labeled matches are allowed — but flagged
    ``irreversible`` so downstream gates can confirm them.
    """
    text = (text or "").strip()
    if not text:
        return None
    elements = [el for el in (state.elements or [])
                if getattr(el, "kind", "") in ("link", "button", "form")]
    tokens = _freeform_tokens(text)

    def _move_for(el: Any, reason: str, confidence: float) -> Move:
        risky = _is_avoid(getattr(el, "label", ""))
        return Move(
            action=Action(
                kind=_element_action_kind(el),
                target=getattr(el, "id", ""),
                label=getattr(el, "label", ""),
            ),
            reason=reason,
            irreversible=risky,
            confidence=max(0.0, min(1.0, confidence)),
        )

    # 1. keyword/intent matching against the live move list
    if tokens and elements:
        scored: list[tuple[float, Any]] = []
        for el in elements:
            label_tokens = _freeform_tokens(getattr(el, "label", ""))
            if not label_tokens:
                continue
            overlap = len(tokens & label_tokens)
            # score by coverage of the *label*: "bribe the guard" fully
            # covering a "Bribe guard" button is a strong match even with
            # extra words in the intent.
            score = overlap / len(label_tokens)
            scored.append((score, el))
        scored.sort(key=lambda t: t[0], reverse=True)
        if scored and scored[0][0] >= 0.5:
            best_score, best_el = scored[0]
            return _move_for(
                best_el,
                f"freeform {text!r} → {best_el.label!r}",
                0.5 + 0.4 * best_score,
            )

    # 2. LLM pick from the numbered move list, validated back to elements
    if suggest is not None and elements:
        menu = "\n".join(
            f"{i + 1}. {el.label}" for i, el in enumerate(elements[:20])
        )
        try:
            reply = (suggest(
                f"The player typed a freeform command in the game "
                f"'{game}': {text!r}\n"
                f"Which numbered option below matches their intent best?\n"
                f"{menu}\n"
                f"Reply with ONLY the number (1-{min(20, len(elements))}), "
                f"or 0 if none matches."
            ) or "").strip()
        except Exception:  # noqa: BLE001 - model failure → freeform fallback
            _log.debug("interpret_freeform model call failed", exc_info=True)
            reply = ""
        match = re.fullmatch(r"\s*(\d+)\s*\.?", reply or "")
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < min(20, len(elements)):
                el = elements[idx]
                return _move_for(
                    el,
                    f"freeform {text!r} → {el.label!r} (model)",
                    0.6,
                )

    # 3. generic fallback: the game handles the raw intent itself
    return Move(
        action=Action(kind="freeform", target="", label=text,
                      payload={"text": text}),
        reason=f"freeform intent: {text!r}",
        irreversible=False,
        confidence=0.4,
    )
