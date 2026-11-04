"""Competence gate for personal-surface conversational replies.

The scaffold guard (:mod:`reply_guard`) asks whether a reply's TEXT
is fit to send. This gate asks the prior question: was the reply
produced by a brain that is currently competent to speak as the
owner at all?

The harm (owner's WhatsApp, 2026-11-04): with no real model
answering — every provider down or unreachable — the responder
serves its static per-mood template bank (``ReplyBundle.fallback``,
``model="fallback"``: a canned in-character line, no model rung
involved). On a bot surface that is a degraded but honest liveness
signal. On the owner's PERSONAL surfaces, where Devon posts AS HIM
in his real groups and DMs, a template brain free-chatting in his
name is an impersonation he never approved — it cost him three
group memberships in one morning. His standing law: fail closed
and state what is missing; fake success is worse than silence.

So: a conversational reply bound for a personal surface that was
produced by the no-model fallback is NOT sent. The inbound turn is
untouched — it was already observed and persisted upstream
(antidelete cache, memory, journals all keep working); only the
outbound reply is held. The hold is journaled once per degraded
episode (partner failure ledger, site ``...:braingate``), the open
episode is visible on the /platforms surface, and the first healthy
generation closes the episode and resumes replies automatically —
no restart, no owner action.

This is a competence gate, not a content gate: it judges HOW the
reply was produced, never what it says. Healthy replies pass
untouched, byte-identical; the scaffold guard still runs after it
(a reply must pass both). Control/command replies (/help, /pair,
the console) never cross the conversational seam and are never
gated — the owner must never lose command of his own bot because
the model is down.

Surface taxonomy (owner's model, partner/gating.py): the gated
public bot is ``telegram-bot`` — out of scope, its template replies
are the bot's own degraded liveness, not impersonation. The
personal surfaces are ``whatsapp`` and ``telegram`` (the Telethon
userbot: his own account) and ``telegram-personal`` (his private
owner-only bot surface, classed with the owner's private surfaces
in gating.py).
"""

from __future__ import annotations

import threading
import time

from .failures import record_failure

__all__ = [
    "PERSONAL_PLATFORMS",
    "is_personal_platform",
    "BrainGate",
    "BrainGateHeld",
    "BrainGateRecovered",
]

#: Ledger site for both episode entries (open + close).
SITE = "runtime.PartnerRuntime._send_reply:braingate"

#: Platforms where Devon speaks AS the owner (his own accounts) or
#: on his private owner surfaces. Bot surfaces (``telegram-bot``,
#: ``discord``, …) are deliberately absent.
PERSONAL_PLATFORMS = frozenset(
    {"whatsapp", "telegram", "telegram-personal"})


def is_personal_platform(platform: object) -> bool:
    """True when *platform* is one of the owner's personal surfaces."""
    return str(platform or "").strip().lower() in PERSONAL_PLATFORMS


class BrainGateHeld(RuntimeError):
    """Journal marker: a degraded episode opened (first reply held)."""


class BrainGateRecovered(RuntimeError):
    """Journal marker: the brain answered again; the episode closed."""


class BrainGate:
    """Episode bookkeeping for degraded-brain holds.

    One episode = the span from the first held personal reply until
    the first healthy conversational send. Holds inside an open
    episode are counted, not journaled — the ledger gets exactly
    one entry when the episode opens and one when it closes, so a
    long outage is two lines, not a flood.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._open = False
        self._opened_at = 0.0
        self._opened_platform = ""
        self._held = 0

    @property
    def episode_open(self) -> bool:
        with self._lock:
            return self._open

    @property
    def held_count(self) -> int:
        with self._lock:
            return self._held

    def note_hold(self, *, platform: str, chat_key: str,
                  note: str = "") -> bool:
        """Record one held personal reply.

        Journals the episode-open entry on the first hold of an
        episode only. Returns True when this hold opened the
        episode. The reply's text is never journaled — only where
        the hold happened and why the brain is degraded.
        """
        with self._lock:
            self._held += 1
            if self._open:
                return False
            self._open = True
            self._opened_at = time.time()
            self._opened_platform = str(platform or "")
        detail = f": {note}" if note else ""
        record_failure(SITE, BrainGateHeld(
            "personal replies held — brain degraded, no model rung "
            f"served (first hold on {platform} in {chat_key}{detail}); "
            "personal surfaces are observe-only until a model answers"))
        return True

    def note_healthy(self) -> bool:
        """Record a healthy conversational generation.

        Closes an open episode (journaled, with the held count) and
        returns True; a no-op returning False when no episode is
        open, so callers can invoke it on every healthy send.
        """
        with self._lock:
            if not self._open:
                return False
            held = self._held
            opened_at = self._opened_at
            platform = self._opened_platform
            self._open = False
            self._opened_at = 0.0
            self._opened_platform = ""
            self._held = 0
        record_failure(SITE, BrainGateRecovered(
            "brain recovered — a model rung answered again; personal "
            f"replies resumed ({held} held during the episode opened "
            f"on {platform}, {max(0.0, time.time() - opened_at):.0f}s)"))
        return True
