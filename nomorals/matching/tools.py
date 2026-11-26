"""``matching`` brain tool — one front door over /match (doc 14, Tier 0 #3).

The three documented surfaces (gigs, community, mentoring) already live
in :mod:`nomorals.matching` and are served in chat by ``/match``
(``matching.chat.control_match``). This tool is the same control for the
brain: the model calls it with a verb + surface + arguments and gets
back the same reply a chat user gets, over the same SQLite stores
(``~/.nomorals/matching/batches.db`` / ``questionnaire.db``) — so a
candidate added in chat shows up in the tool's daily batch, and vice
versa. The tool never duplicates the control's logic; it builds the
``/match`` tail and delegates.

Surfaces: ``gig`` (also ``gigs``), ``community``, ``learning`` (also
``mentoring`` / ``mentor`` / ``mentee``). The surface is set on the
shared context for the call and restored afterwards, so a tool call
never leaks its surface into the next chat command.
"""

from __future__ import annotations

from typing import Any

from ..core.policy import Capability
from .chat import _VERBS, _canon_surface, control_match

__all__ = ["register"]


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "matching",
        description=(
            "Curated daily picks + two-sided stable matching over three "
            "surfaces — gigs, community, mentoring (learning). One call "
            "reaches the same matching engine /match uses, over the same "
            "stores. Actions: daily (today's batch for a surface), add "
            "(register a candidate: '<id> | <title> | tag1,tag2 | "
            "attr=val,...'), remove, like / pass (train future batches), "
            "why (why was this picked?), questionnaire (unanswered "
            "preference questions for a surface), answer, acceptable, "
            "dealbreaker, rank (proposer/reviewer preference order), "
            "capacity (a reviewer's seats), run (Gale-Shapley stable "
            "matching, verified no-blocking-pairs), roommates, assign "
            "(max-total-score assignment), ttc (top trading cycles), "
            "draft (serial-dictatorship pick order), stats, help."
        ),
        capability=Capability.FS_WRITE,
        parameters={
            "action": "str — daily | add | remove | like | pass | why | "
                      "questionnaire | answer | acceptable | dealbreaker | "
                      "rank | capacity | run | roommates | assign | ttc | "
                      "draft | stats | help",
            "surface": "str (optional, gig) — gig | gigs | community | "
                       "learning | mentoring | mentor | mentee",
            "args": "str (optional) — the verb's remaining arguments, "
                    "exactly as /match takes them (e.g. '5' for daily, "
                    "'c-1 | Title | python | quality=0.9' for add, "
                    "'proposer alice c1,c2' for rank)",
        },
    )
    def matching(action: str, *, surface: str = "gig",
                 args: str = "") -> dict[str, Any]:
        act = (action or "").strip().lower()
        if act not in _VERBS:
            raise ValueError(
                f"unknown matching action {action!r} — " + " | ".join(_VERBS))
        surf = _canon_surface(surface)
        rest = (args or "").strip()
        if act == "help":
            tail = ""
        elif act == "daily":
            tail = "daily " + surf + (f" {rest}" if rest else "")
        else:
            tail = act + (f" {rest}" if rest else "")
        prev = getattr(context, "matching_surface", None)
        context.matching_surface = surf
        try:
            reply = control_match(tail, context=context)
        finally:
            if prev is None:
                try:
                    delattr(context, "matching_surface")
                except AttributeError as exc:
                    assert isinstance(exc, Exception)
            else:
                context.matching_surface = prev
        return {"ok": True, "reply": reply, "action": act,
                "surface": surf}
