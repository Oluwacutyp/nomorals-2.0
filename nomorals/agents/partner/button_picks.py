"""Numbered button-fallback picks (register #27, 2026-11-05).

When a reply's buttons cannot ride the send (the platform takes
none), ``_send_control_reply`` rewrites the copy into a numbered
action list and registers the picks here. A later bare-number
message in the same chat executes the same command a tap would
have run — the fallback is a working control, not a description
of one. Picks are single-use and expire; a new registration for a
chat replaces the old list wholesale (the newest card on screen
owns the numbers).

Living Chat P1 (2026-11-08): the store now rides the shared
operation store (:mod:`nomorals.core.operations`), which adds
the targeting the chat-only store could not express:

* picks bind to a (chat, owner) slot — in a group, the card
  belongs to the user it was offered to, and another user's
  bare number never consumes it;
* every registration is an OPERATION with a stable id, and
  :meth:`ButtonPickStore.take_result` reports STALE for a pick
  against a replaced/expired card, so the runtime answers
  honestly ("that card has been replaced") instead of silently
  flowing on or — worse — running the newer card's action.
"""

from __future__ import annotations

from ...core.operations import OperationStore, PickResult, PickStatus

__all__ = ["ButtonPickStore", "TTL_S", "MAX_CHATS", "MAX_PICKS",
           "STALE_PICK_MESSAGE"]

TTL_S = 600.0
MAX_CHATS = 256
MAX_PICKS = 12

#: The honest answer to a pick against a card that is gone: it
#: was replaced by a newer card or it expired. Never an action.
STALE_PICK_MESSAGE = (
    "that card has been replaced or has expired — the newest "
    "card in this chat owns the numbers now.")


class ButtonPickStore:
    """Numbered picks over the shared operation store."""

    def __init__(self, *, ttl_s: float = TTL_S) -> None:
        self._ops = OperationStore(ttl_s=float(ttl_s),
                                   max_slots=MAX_CHATS)

    def register(self, chat_key: str, commands: list[str], *,
                 owner_key: str = "") -> str:
        """Register a card's picks; returns its operation id.

        "" when nothing was registered (empty chat or list) —
        the legacy contract returned None; callers that only
        registered for effect are unaffected.
        """
        picks = [str(c) for c in (commands or []) if str(c).strip()]
        if not str(chat_key or "") or not picks:
            return ""
        op = self._ops.open(str(chat_key), picks[:MAX_PICKS],
                            owner_key=str(owner_key or ""),
                            kind="button-fallback")
        return op.operation_id if op is not None else ""

    def take(self, chat_key: str, number: int, *,
             owner_key: str = "", now: float | None = None) -> str:
        """The command for ``number`` in ``chat_key``, consuming it.

        Returns "" when there is no live pick list, the entry
        expired or was replaced, or the number is out of range
        (an out-of-range number leaves the list intact — a typo
        must not burn the card's other actions). Callers that
        need to tell "replaced" from "never existed" use
        :meth:`take_result`.
        """
        res = self.take_result(chat_key, number, owner_key=owner_key,
                               now=now)
        return res.command if res.status is PickStatus.OK else ""

    def take_result(self, chat_key: str, number: int, *,
                    owner_key: str = "", operation_id: str = "",
                    now: float | None = None) -> PickResult:
        """Resolve a pick with its full outcome (see PickStatus).

        With ``operation_id`` the pick names its card: a card
        that is no longer the slot's current one answers STALE.
        Without it, the pick resolves against the current card.
        """
        if operation_id:
            return self._ops.resolve_operation(
                str(chat_key or ""), operation_id, number,
                owner_key=str(owner_key or ""), now=now)
        return self._ops.resolve_number(
            str(chat_key or ""), number,
            owner_key=str(owner_key or ""), now=now)

    def has(self, chat_key: str, *, owner_key: str = "") -> bool:
        return self._ops.current(
            str(chat_key or ""),
            owner_key=str(owner_key or "")) is not None
