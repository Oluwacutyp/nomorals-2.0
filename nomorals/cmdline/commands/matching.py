"""``nm matching`` — CLI front door over nomorals.matching (Tier 0 wiring).

Delegates to :func:`nomorals.matching.chat.control_match` — the same
control the chat /match command uses — so CLI and chat stay in sync.

Surface-first, like the doc names them (doc 14, Tier 0 #3)::

    nm matching gigs                      — today's gig picks
    nm matching community                 — today's community picks
    nm matching mentoring                 — today's mentoring picks
    nm matching gigs daily 5              — five gig picks
    nm matching community add <id> | <title> | tag1,tag2 | attr=val,...
    nm matching mentoring run             — stable matching for mentors

The surface word is an alias (gigs→gig, mentoring→learning); a bare
verb works too (``nm matching daily gig``) and passes straight through
to /match.
"""

from __future__ import annotations

import sys
from typing import Any

from ...matching.chat import _VERBS, _canon_surface, control_match

_SURFACES = ("gig", "gigs", "community", "learning", "mentoring",
             "mentor", "mentee")

_USAGE = ("usage: nm matching <surface> [verb ...] | <verb ...>\n"
          "       surfaces: gigs | community | mentoring\n"
          "       nm matching gigs | community | mentoring — today's picks\n"
          "       nm matching <surface> daily [n] | add <id> | <title> | tags | attrs\n"
          "       nm matching <surface> like|pass <id> | why <id> | stats\n"
          "       nm matching <surface> questionnaire | answer <qid> <1-5> | dealbreaker <name> <attr> <pred> <value>\n"
          "       nm matching <surface> rank <proposer|reviewer> <who> <ids> | capacity <who> <n> | run\n"
          "       nm matching <surface> roommates <ids> | assign | ttc ... | draft <agents> | stats\n"
          "       nm matching daily [surface] [n] — same verbs without the surface first")


def _cmd_matching(args: Any, context: Any) -> int:
    """Route ``nm matching <surface|verb> ...``."""
    words = list(getattr(args, "task", None) or [])
    if not words:
        print(_USAGE, file=sys.stderr)
        return 2
    first = (words[0] or "").strip().lower()
    surface = None
    tail = ""
    if first in _SURFACES:
        surface = _canon_surface(first)
        rest = words[1:]
        if not rest:
            tail = "daily " + surface
        elif rest[0].strip().isdigit():
            tail = f"daily {surface} {rest[0].strip()}"
        elif rest[0].strip().lower() in _VERBS:
            verb = rest[0].strip().lower()
            if verb == "daily":
                tail = "daily " + surface + (
                    " " + " ".join(rest[1:]) if len(rest) > 1 else "")
            else:
                tail = " ".join(rest)
        else:
            print(_USAGE, file=sys.stderr)
            return 2
    elif first in _VERBS:
        tail = " ".join(words)
    else:
        print(_USAGE, file=sys.stderr)
        return 2
    prev = getattr(context, "matching_surface", None)
    if surface is not None:
        context.matching_surface = surface
    try:
        print(control_match(tail, context=context))
    finally:
        if surface is not None:
            if prev is None:
                try:
                    delattr(context, "matching_surface")
                except AttributeError as exc:
                    assert isinstance(exc, Exception)
            else:
                context.matching_surface = prev
    return 0
