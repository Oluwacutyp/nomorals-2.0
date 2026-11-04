"""The wager purse: where a game's staked coins actually live.

Wager games (blackjack, roulette, slots, craps, poker) stake REAL
coins.  The engine stays purse-agnostic: it binds one
:class:`BoundPurse` to every room at open/restore time, and games
reach it as ``room.purse``.  A bound purse resolves to a *backend*:

* :class:`StorePurseBackend` — the games wallet (``PlayerStore``
  profile coins).  The engine's default for every room, and the only
  purse a bare engine (tests, tools) ever needs.
* a runtime-installed backend (``engine.purse_source``, a callable
  ``room -> backend | None``) — the partner runtime binds GROUP
  rooms to the group economy estate, the same organ the coin sink
  credits, so a member's /balance is the number that moves when
  they bet in that group.  Returning None keeps the default.

Every movement journals in the owning purse's own ledger (the
profile ledger for the games wallet, the group ledger for the
estate), tagged with the game and the room.  Debits are
all-or-nothing: a debit that returns False moved NOTHING.
"""
from __future__ import annotations

from typing import Any

__all__ = ["BoundPurse", "StorePurseBackend"]


class BoundPurse:
    """A purse handle bound to one room — the games' only view of
    money.  Methods take the :class:`~nomorals.games.players.Player`
    acting; AI seats have no purse (balance 0, debits refuse,
    credits are no-ops) so house-played spectacle tables never move
    money."""

    def __init__(self, backend: Any, room: Any) -> None:
        self._backend = backend
        self._room = room
        #: human label for panels ("game wallet" / "group purse")
        self.name = str(getattr(backend, "name", "purse"))

    # ── tag composition: every journal entry names game + room ──
    def _reason(self, tag: str) -> str:
        return f"game:{self._room.game} {tag}".strip()

    def _ref(self) -> str:
        return f"wager:{self._room.game}:{self._room.id}"

    def balance(self, player: Any) -> int:
        if player is None or getattr(player, "is_ai", False):
            return 0
        try:
            return int(self._backend.balance(player))
        except Exception:  # noqa: BLE001 - a purse read never breaks a table
            return 0

    def debit(self, player: Any, amount: int, tag: str) -> bool:
        """Escrow ``amount`` out of the player's balance.  False =
        insufficient (or AI seat): nothing moved."""
        amount = int(amount)
        if player is None or getattr(player, "is_ai", False) \
                or amount <= 0:
            return False
        try:
            return bool(self._backend.debit(
                player, amount, self._reason(tag), self._ref()))
        except Exception:  # noqa: BLE001 - a failed debit moved nothing
            return False

    def credit(self, player: Any, amount: int, tag: str) -> int:
        """Pay ``amount`` into the player's balance.  Returns what
        was actually credited (0 for AI seats / non-positive amounts)."""
        amount = int(amount)
        if player is None or getattr(player, "is_ai", False) \
                or amount <= 0:
            return 0
        try:
            return int(self._backend.credit(
                player, amount, self._reason(tag), self._ref()))
        except Exception:  # noqa: BLE001
            return 0


class StorePurseBackend:
    """The games wallet: PlayerStore profile coins.

    Debits ride ``spend_coins`` (atomic check-and-take under the
    store lock), credits ride ``add_coins``; both write the
    profile's own ledger with the wager reason — that ledger is
    this purse's journal.
    """

    name = "game wallet"

    def __init__(self, store: Any, db: Any = None) -> None:
        self._store = store
        self._db = db

    def balance(self, player: Any) -> int:
        return int(self._store.get_for(player).coins)

    def debit(self, player: Any, amount: int, reason: str,
              ref: str) -> bool:
        return self._store.spend_coins(
            player, int(amount), reason=reason) is not None

    def credit(self, player: Any, amount: int, reason: str,
               ref: str) -> int:
        amount = int(amount)
        if amount <= 0:
            return 0
        self._store.add_coins(player, amount, reason=reason)
        return amount
