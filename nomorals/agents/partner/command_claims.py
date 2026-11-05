"""Cross-surface command claims (owner order 2026-11-05).

One Telegram message reaches Devon once per attached surface:
the bot gets it through the Bot API, the owner's userbot gets
the same message through MTProto (his own sends arrive as
own-echoes). Before this ledger, EVERY surface that parsed a
command handled it — in the owner's claimed group his
``/rmdeleted@Nomorals_bot`` was executed by the bot AND answered
by the as-him surface with the §33 gate text (which read a
different claim record and lied about the room), and one
``/play`` in the bot's DM minted two jobs, one per surface.

The law: a recognized command is claimed by exactly one
surface; every other surface that sees it stands down silently
(persist/observe only, never a reply). The runtime consults
this ledger at the command checkpoints of ``_process_inner``
— the first surface to claim a (platform-family, chat,
message) key owns the command; a claim repeated by the SAME
surface (the dispatch re-checks at several branches of one
turn) is still its own. Claims are keyed on the platform
MESSAGE identity, which is the same on every surface that
delivers the message (Telegram message ids are per-chat), with
the platform family collapsed (``telegram`` / ``telegram-bot``
/ ``telegram-personal`` are one family — one room, one chat).
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict

__all__ = ["CommandClaimLedger", "WINDOW_S", "MAX_KEYS"]

#: The duplicate deliveries of one message land seconds apart;
#: a slow bot poll is the long pole. Ten minutes matches the
#: runtime's bot-claim probe TTL — past it, a delivery is a new
#: event (adapter replays are held by the stale-replay gate).
WINDOW_S = 600.0
MAX_KEYS = 1024


class CommandClaimLedger:
    """Bounded, thread-safe record of which surface claimed
    which command message."""

    def __init__(self, *, window_s: float = WINDOW_S) -> None:
        self._window_s = float(window_s)
        self._lock = threading.Lock()
        # claim key -> (surface, ts)
        self._claims: OrderedDict[str, tuple[str, float]] = \
            OrderedDict()

    def claim(self, key: str, surface: str, *,
              now: float | None = None) -> bool:
        """Try to own ``key`` for ``surface``.

        True when this surface owns the command — a fresh claim,
        or its own still-live claim repeated. False when another
        surface holds a live claim: stand down silently. Never
        raises: claim bookkeeping must not eat a turn."""
        key = str(key or "")
        surface = str(surface or "")
        if not key or not surface:
            return True  # unkeyable: legacy single-surface path
        ts = time.time() if now is None else float(now)
        with self._lock:
            held = self._claims.get(key)
            if held is not None:
                holder, held_ts = held
                if ts - held_ts <= self._window_s:
                    if holder == surface:
                        self._claims.move_to_end(key)
                        return True
                    return False
            self._claims[key] = (surface, ts)
            self._claims.move_to_end(key)
            while len(self._claims) > MAX_KEYS:
                self._claims.popitem(last=False)
            return True

    def claimed_by(self, key: str, *,
                   now: float | None = None) -> str:
        """The surface holding a live claim on ``key``, or ""."""
        key = str(key or "")
        if not key:
            return ""
        ts = time.time() if now is None else float(now)
        with self._lock:
            held = self._claims.get(key)
            if held is None:
                return ""
            holder, held_ts = held
            if ts - held_ts > self._window_s:
                return ""
            return holder
