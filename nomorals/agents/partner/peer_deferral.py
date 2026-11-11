"""Defer-and-observe arbitration for commands in a bot's DM (dupfix2).

Why this exists: the shipped command arbitration (``command_claims``)
is a *shared-state* mechanism — a durable claim ledger plus a peer
rule fed by configured companion bot ids. It is exactly-once whenever
the surfaces share a home store or the userbot was built with both bot
tokens, and the probes for this lane prove it in both shapes. It does
NOT hold in the live shape: the owner's 2026-11-11 recording shows the
userbot answering every command in the personal bot's DM, and the
incident record itself documents the live userbot carrying no companion
ids — a second surface that claims in a store the first never sees, and
whose peer rule has no inputs. No shared state, no shared config: the
only channel both surfaces provably share is the chat itself.

The law here is therefore in-band, not in-store:

* A command on a non-bot surface in a DM whose peer is one of Devon's
  own bots is NEVER handled at once. It is *deferred*: registered
  here, and this surface watches the room for the primary's answer.
* The primary's answer is observed through the MTProto stream the
  userbot always has — the bot's post in that DM arrives as a
  ``peer_post`` observation (see :func:`nomorals.social.chat.telegram.
  TelegramAdapter._handle_inbound`), even though it is never a turn.
  An observation resolves the deferral when it replies to the
  deferred command or is the newest owner command's answer in that
  chat — the primary answered, this surface stands down, silently.
* If the window expires with no answer, the deferred command is
  claimed (the shared ledger, when one exists) and handled here —
  failover: a down/absent primary never means silence, and OFF vs
  BROKEN stays distinguishable (the answer arrives late, it never
  vanishes).
* A newer owner command in the same chat supersedes an older pending
  one: the older record drops without emitting, so a failover can
  never re-apply a command the owner already replaced.

This registry is in-memory per runtime and clock-injectable; it holds
no store of its own, because the observation channel is the point —
the mechanism must work when the surfaces share nothing at all.
"""
from __future__ import annotations

import threading
import time
from typing import Any

__all__ = ["DEFAULT_PEER_ANSWER_WINDOW_S", "PeerAnswerDeferral"]

#: How long a deferred command waits for the primary's answer before
#: the deferring surface fails over and handles it itself. Long
#: enough to cover a slow bot poll plus a slow model call; short
#: enough that a genuinely down primary still answers within a
#: breath. The value is visible in the journaled deferral, never
#: silent.
DEFAULT_PEER_ANSWER_WINDOW_S = 20.0


class PeerAnswerDeferral:
    """Pending echo commands waiting on a primary surface's answer."""

    def __init__(self, *, window_s: float = DEFAULT_PEER_ANSWER_WINDOW_S):
        self._window_s = float(window_s)
        self._lock = threading.Lock()
        #: (chat_key, message_id) -> record
        self._pending: dict[tuple[str, str], dict[str, Any]] = {}
        #: chat_key -> (message_id, created_at) of the newest owner
        #: command registered in that chat
        self._latest: dict[str, tuple[str, float]] = {}

    # -- registration -------------------------------------------------

    def register(self, record: dict[str, Any], *,
                 now: float | None = None) -> bool:
        """Register one deferred command. False when an identical
        (chat, message) deferral already lives — a re-delivery of
        the same echo never double-registers."""
        now = time.time() if now is None else float(now)
        chat_key = str(record.get("chat_key") or "")
        message_id = str(record.get("message_id") or "")
        if not chat_key or not message_id:
            return False
        key = (chat_key, message_id)
        with self._lock:
            if key in self._pending:
                return False
            record = dict(record)
            record["created_at"] = now
            record["deadline"] = now + self._window_s
            self._pending[key] = record
            latest = self._latest.get(chat_key)
            if latest is None or now >= latest[1]:
                self._latest[chat_key] = (message_id, now)
            return True

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    # -- resolution by observation ------------------------------------

    def resolve_for_chat(self, chat_key: str, *,
                         reply_to: str = "",
                         now: float | None = None) -> list[dict[str, Any]]:
        """The primary posted in ``chat_key``. Resolve every pending
        record that post answers: one whose message id the post
        replies to, or the chat's newest pending command (a bot
        reply in its own DM answers the owner's latest command
        there). Older pendings stay: they are superseded, and the
        sweep drops them without emitting. Returns the resolved
        records, newest first."""
        now = time.time() if now is None else float(now)
        chat_key = str(chat_key or "")
        reply_to = str(reply_to or "")
        resolved: list[dict[str, Any]] = []
        with self._lock:
            latest = self._latest.get(chat_key)
            for key in list(self._pending):
                if key[0] != chat_key:
                    continue
                if reply_to and key[1] == reply_to:
                    resolved.append(self._pending.pop(key))
                    continue
                if latest is not None and key[1] == latest[0]:
                    resolved.append(self._pending.pop(key))
            if resolved:
                newest = max(r["created_at"] for r in resolved)
                if latest is not None and latest[1] <= newest:
                    self._latest.pop(chat_key, None)
        return sorted(resolved, key=lambda r: r["created_at"],
                      reverse=True)

    def supersede(self, chat_key: str, message_id: str) -> list[dict[str, Any]]:
        """A newer owner command landed in this chat: every older
        pending record in the chat is superseded — dropped without
        emitting, so a failover can never re-apply a replaced
        command. Returns the dropped records."""
        dropped: list[dict[str, Any]] = []
        with self._lock:
            for key in list(self._pending):
                if key[0] == chat_key and key[1] != str(message_id):
                    dropped.append(self._pending.pop(key))
        return dropped

    # -- expiry (failover) --------------------------------------------

    def expire(self, *, now: float | None = None) -> list[dict[str, Any]]:
        """Records past their deadline that are still the chat's
        newest owner command — the primary never answered, so this
        surface fails over and handles them. Superseded records are
        dropped silently with them. Returns the due records, oldest
        first."""
        now = time.time() if now is None else float(now)
        due: list[dict[str, Any]] = []
        with self._lock:
            for key, record in list(self._pending.items()):
                if float(record["deadline"]) > now:
                    continue
                latest = self._latest.get(key[0])
                self._pending.pop(key)
                if latest is not None and latest[0] == key[1]:
                    self._latest.pop(key[0], None)
                    due.append(record)
        return sorted(due, key=lambda r: r["created_at"])
