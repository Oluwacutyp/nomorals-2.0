"""Owner-replied suppression ledger (restored, owner order 2026-11-05).

Register #28: when the owner himself has already replied to a
message in a chat, Devon stays quiet on that message/thread —
he fixed this once and the behavior vanished, because it had lived
outside the repo. This is the repo-native restoration.

The ledger is fed by the ONE seam every personal-surface adapter
passes an owner-sent message through: ``PartnerRuntime`` records
own-echo messages (``meta['own_message']``) here before anything
else looks at them. The conversational brain consults it twice —
in ``reply_plan`` for the message about to be answered, and again
in ``deliver_reply`` for a reply composed earlier and delivered
late — so suppression holds on the immediate path and the delayed
path alike. Control/command handling never consults it: commands
always answer.

Coverage: targeted (an owner reply to message X answers X) and
windowed (an owner reply anywhere in the chat within the window
means the owner is live in this conversation — Devon does not
talk over him). Both are per chat key.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict

__all__ = ["OwnerReplyLedger", "WINDOW_S", "MAX_CHATS", "MAX_TARGETS"]

#: The owner answering by hand and Devon answering the same thread
#: minutes later is exactly the double-talk being suppressed; the
#: window is generous on purpose — the owner's presence in the
#: chat, not a race, is the signal.
WINDOW_S = 120.0
MAX_CHATS = 512
MAX_TARGETS = 256


class OwnerReplyLedger:
    """Bounded, thread-safe record of owner-sent replies per chat."""

    def __init__(self, *, window_s: float = WINDOW_S) -> None:
        self._window_s = float(window_s)
        self._lock = threading.Lock()
        # chat_key -> (last_owner_reply_ts, OrderedDict[target_id, ts])
        self._chats: OrderedDict[
            str, tuple[float, OrderedDict[str, float]]] = OrderedDict()

    def record_owner_reply(self, chat_key: str, *, reply_to: str = "",
                           now: float | None = None) -> None:
        """Note that the owner replied in ``chat_key``.

        ``reply_to`` (the platform message id the owner's reply
        quotes, when the adapter carries it) marks that exact
        message answered. Never raises: suppression bookkeeping
        must not eat a turn.
        """
        key = str(chat_key or "")
        if not key:
            return
        ts = time.time() if now is None else float(now)
        target = str(reply_to or "")
        with self._lock:
            entry = self._chats.get(key)
            if entry is None:
                entry = (ts, OrderedDict())
                self._chats[key] = entry
                while len(self._chats) > MAX_CHATS:
                    self._chats.popitem(last=False)
            else:
                entry = (ts, entry[1])
                self._chats[key] = entry
                self._chats.move_to_end(key)
            if target:
                targets = entry[1]
                targets[target] = ts
                targets.move_to_end(target)
                while len(targets) > MAX_TARGETS:
                    targets.popitem(last=False)

    def owner_answered(self, chat_key: str, message_id: str = "",
                       *, now: float | None = None) -> str:
        """Why Devon should stay quiet on this message, or \"\".

        ``"owner-replied-target"`` — the owner replied to this
        exact message. ``"owner-replied-recent"`` — the owner
        replied anywhere in this chat inside the window. An empty
        message id still gets the window check (most platform
        echoes carry no target).
        """
        key = str(chat_key or "")
        if not key:
            return ""
        ts = time.time() if now is None else float(now)
        with self._lock:
            entry = self._chats.get(key)
        if entry is None:
            return ""
        last_ts, targets = entry
        target = str(message_id or "")
        if target and target in targets:
            return "owner-replied-target"
        if ts - last_ts <= self._window_s:
            return "owner-replied-recent"
        return ""
