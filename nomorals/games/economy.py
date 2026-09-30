"""The game economy: coins, a persistent shop, and item ownership.

Coins are earned from game outcomes (the engine credits winners and
participation), spent in the shop.  Shop inventory is defined per game
by the games themselves (battle-arena gear, shop-game goods, escape-room
props) and shared across games through one wallet.  Every transaction is
logged to ``game_wallet`` so balances are auditable — a player who
disputes a balance can be shown the receipts.

Deliberately simple in a good way: no inflation engine, no markets
between players.  The economy exists to make wins *feel* like something
and give the ambitious games (shop, battle arena, persistent world)
durable stakes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .players import Player, PlayerStore

__all__ = ["ShopItem", "GameEconomy", "DEFAULT_SHOP"]


@dataclass(frozen=True)
class ShopItem:
    """One purchasable thing. ``effect`` is game-specific metadata."""

    slug: str
    name: str
    cost: int
    effect: str = ""        # e.g. "+10 max_hp", "reveals one clue"
    game: str = "any"       # which game(s) use it (or "any")

    def to_dict(self) -> dict[str, Any]:
        return {"slug": self.slug, "name": self.name, "cost": self.cost,
                "effect": self.effect, "game": self.game}


#: The cross-game base inventory. Games register their own extras at
#: setup; the union is what the shop sells.
DEFAULT_SHOP: tuple[ShopItem, ...] = (
    ShopItem("lucky_coin", "Lucky Coin", 120,
             "next guess battle: one free retry", "numberguess"),
    ShopItem("hint_scroll", "Hint Scroll", 80,
             "reveals one letter / clue", "any"),
    ShopItem("shield", "Shield", 200,
             "absorbs one loss in battle arena", "battle_arena"),
    ShopItem("gear_sword", "Steel Sword", 350,
             "+10 attack", "battle_arena"),
    ShopItem("gear_armor", "Iron Armor", 300,
             "+15 defense", "battle_arena"),
    ShopItem("potion", "Healing Potion", 150,
             "restore 30 HP", "battle_arena"),
    ShopItem("keycard", "Master Keycard", 500,
             "auto-solve one escape room lock", "escape_room"),
    ShopItem("blueprint", "City Blueprint", 400,
             "build a building for free once", "world"),
    ShopItem("veto", "Veto Token", 250,
             "cancel one vote once per game", "political"),
    ShopItem("fast_pass", "Fast Pass", 100,
             "skip one waiting turn", "any"),
)


class GameEconomy:
    """Wallet + shop over the player ledger."""

    def __init__(self, store: PlayerStore,
                 extra_items: tuple[ShopItem, ...] = ()) -> None:
        self.store = store
        self._items: dict[str, ShopItem] = {
            i.slug: i for i in DEFAULT_SHOP
        }
        for item in extra_items:
            self._items[item.slug] = item

    # ── shop ─────────────────────────────────────────────────────────────────
    def register_items(self, items: tuple[ShopItem, ...]) -> None:
        for item in items:
            self._items[item.slug] = item

    def catalog(self, game: str = "any") -> list[ShopItem]:
        """Everything sellable to a given game (its items + shared ones)."""
        out = [i for i in self._items.values()
               if i.game in {"any", game}]
        out.sort(key=lambda i: i.cost)
        return out

    def catalog_text(self, game: str = "any", player: Player | None = None) -> str:
        items = self.catalog(game)
        if not items:
            return "the shop is empty for this game."
        balance = ""
        if player is not None:
            balance = f"  (you have {self.balance(player)} coins)"
        lines = [f"🛒 shop{balance} — /game shop buy <slug>"]
        for item in items:
            lines.append(f"  {item.slug:<12} {item.name} — {item.cost}c"
                         + (f" · {item.effect}" if item.effect else ""))
        return "\n".join(lines)

    # ── wallet ───────────────────────────────────────────────────────────────
    def balance(self, player: Player) -> int:
        return self.store.get(player.key).coins

    def purchase(self, player: Player, slug: str) -> tuple[bool, str]:
        """Buy one item. Returns (ok, message)."""
        item = self._items.get(slug)
        if item is None:
            known = ", ".join(sorted(self._items))
            return False, f"no such item {slug!r}. in stock: {known}"
        prof = self.store.get(player.key)
        if prof.coins < item.cost:
            return False, (f"{item.name} costs {item.cost}c — you have "
                           f"{prof.coins}c. win games to earn more.")
        self.store.add_coins(player, -item.cost, f"buy:{slug}")
        self.store.grant_item(player, slug)
        return True, (f"bought {item.name} for {item.cost}c — "
                      f"{item.effect} (you now have "
                      f"{self.count(player, slug)}).")

    def count(self, player: Player, slug: str) -> int:
        return int(self.store.get(player.key).items.get(slug) or 0)

    def consume(self, player: Player, slug: str) -> bool:
        """Spend one owned item. False if they don't have one."""
        prof = self.store.get(player.key)
        if int(prof.items.get(slug) or 0) <= 0:
            return False
        prof.items[slug] = int(prof.items.get(slug) or 0) - 1
        prof.updated_at = time.time()
        self.store._upsert(prof)
        return True

    # ── rewards ──────────────────────────────────────────────────────────────
    @staticmethod
    def reward_points(won: bool | None, score: int = 0) -> int:
        """Points credited on game finish: winners by score, losers a
        participation crumb so the board stays alive."""
        if won is True:
            return 50 + max(0, score) * 5
        if won is False:
            return 10
        return 20

    @staticmethod
    def reward_coins(won: bool | None) -> int:
        if won is True:
            return 60
        if won is False:
            return 15
        return 25
