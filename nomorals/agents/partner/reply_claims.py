"""Exactly-once claims for conversational replies (2026-11-06).

Commands have had cross-surface arbitration since 2026-11-05
(:mod:`command_claims`): one message identity, one handling.
Conversational replies had NO equivalent, and the layers that
look like they cover it each miss a redelivery:

* the adapters' seen-sets are in-memory (Telegram's bounded
  ``_seen_inbound`` dies with the process) or age-based only
  (WhatsApp stamps ``stale_replay`` past a freshness window — a
  redelivery inside the window is indistinguishable from a first
  delivery);
* the runtime's idempotency scope keys TOOL EFFECTS per message,
  never the reply itself;
* the brain's double-reply discipline compares against the
  inbound row's own persist time — and a redelivery persists a
  FRESH row newer than the first reply, so that guard
  structurally cannot fire for the same message twice.

The incident (owner's Telegram, 2026-11-06): a contact's photo
was delivered twice during a flaky reconnect; Devon generated
the same short reply twice and sent "photo didn't come through"
twice in one minute, as the owner, about a photo sitting in the
chat.

The law: one platform message identity gets one conversational
turn. The runtime claims the identity at the conversational seam
of ``_process_inner`` (after observation, control and command
checkpoints, which have their own arbitration); a second
delivery of the same identity inside the window is observed by
everything above the seam and then stands down — no second
generation, no second send. The delayed-reply path is part of
the SAME turn (it fires after the claim, inside the window), so
it never re-claims. Messages with no platform id are unkeyable
and run unclaimed: inventing an identity could collide two
distinct messages into a false hold. An EDIT of a message keys
exactly like the original: an edit of an already-answered
message is the same event, not a new one (the antiedit layer
above the seam still observes it — the seam only gates the
second conversational generation). And a redelivery that arrives
with a FRESH id — a bridge replay that minted a new identity —
is caught by the content key: same chat, same sender, same
words inside a short window is the same event, whatever id it
wears.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from collections import OrderedDict
from typing import Any

__all__ = ["ReplyClaimLedger", "reply_claim_key",
           "reply_content_key", "WINDOW_S", "CONTENT_WINDOW_S",
           "MAX_KEYS"]

#: Same window as the command claims: duplicate deliveries of one
#: message land seconds apart; a slow redelivery after a restart
#: is the long pole. Past the window a delivery is treated as a
#: new event (adapter replays that old are held by the
#: stale-replay gate instead).
WINDOW_S = 600.0
#: The content key's window: a fresh-id redelivery lands seconds
#: after the original. Past it the same words from the same
#: sender are a new event — a genuine repeat, not a replay.
CONTENT_WINDOW_S = 30.0
MAX_KEYS = 1024


def reply_claim_key(message: Any) -> str:
    """The claim key for one inbound message, or "" when unkeyable.

    Mirrors the command-claim key shape exactly (platform family
    collapsed — ``telegram`` / ``telegram-bot`` /
    ``telegram-personal`` are one family, one room), so both
    ledgers name the same message the same way. An edit keys
    exactly like the original message it edits.
    """
    message_id = str(getattr(message, "message_id", "") or "")
    if not message_id:
        return ""
    chat = getattr(message, "chat", None)
    platform = str(getattr(chat, "platform", "") or "")
    chat_id = str(getattr(chat, "chat_id", "") or "")
    if not platform or not chat_id:
        return ""
    family = "telegram" if platform.startswith("telegram") else platform
    return f"{family}:{chat_id}:{message_id}"


def reply_content_key(message: Any) -> str:
    """The content key for one inbound message, or "" when
    unkeyable (no chat, no sender, no text).

    ``family:chat_id:sender:alt:sha1(words)`` — the same words
    from the same sender in the same chat, inside the short
    window, is one event even when the delivery minted a fresh
    message id. Both sender forms ride the key (a WhatsApp
    redelivery carries the same JID/LID pair as the original),
    and the words are folded (case, surrounding space, internal
    runs) so a re-send that only re-wrapped is still caught.
    """
    chat = getattr(message, "chat", None)
    platform = str(getattr(chat, "platform", "") or "")
    chat_id = str(getattr(chat, "chat_id", "") or "")
    sender = str(getattr(message, "sender_id", "") or "")
    meta = getattr(message, "meta", None) or {}
    alt = ""
    if isinstance(meta, dict):
        alt = str(meta.get("sender_alt_id") or "")
    text = str(getattr(message, "text", "") or "")
    words = re.sub(r"\s+", " ", text.strip().casefold())
    if not platform or not chat_id or not sender or not words:
        return ""
    family = "telegram" if platform.startswith("telegram") else platform
    digest = hashlib.sha1(words.encode("utf-8")).hexdigest()[:16]
    return f"{family}:{chat_id}:{sender}:{alt}:{digest}"


class ReplyClaimLedger:
    """Bounded, thread-safe record of claimed message identities.

    One runtime, one conversation pipeline: unlike the command
    ledger there is no surface dimension — a second claim of the
    same key inside the window is a redelivery, whoever brings it.
    """

    def __init__(self, *, window_s: float = WINDOW_S) -> None:
        self._window_s = float(window_s)
        self._lock = threading.Lock()
        # claim key -> claim timestamp
        self._claims: OrderedDict[str, float] = OrderedDict()

    def claim(self, key: str, *, now: float | None = None,
              window_s: float | None = None) -> bool:
        """Try to own the conversational turn for ``key``.

        True when this caller owns the turn — a fresh claim, or
        the previous claim has aged out of the window. False when
        the key was already claimed inside the window: stand down
        silently. An empty key is unkeyable and always passes.
        ``window_s`` overrides the ledger's window for this one
        claim (the content key's short window rides here).
        Never raises: claim bookkeeping must not eat a turn.
        """
        key = str(key or "")
        if not key:
            return True
        ts = time.time() if now is None else float(now)
        try:
            window = float(window_s) if window_s is not None \
                else self._window_s
        except (TypeError, ValueError):
            window = self._window_s
        with self._lock:
            held_ts = self._claims.get(key)
            if held_ts is not None and ts - held_ts <= window:
                return False
            self._claims[key] = ts
            self._claims.move_to_end(key)
            while len(self._claims) > MAX_KEYS:
                self._claims.popitem(last=False)
            return True
