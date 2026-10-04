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
                 extra_items: tuple[ShopItem, ...] = (),
                 gear_store: Any = None) -> None:
        self.store = store
        self.gear = gear_store
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
        balance = ""
        if player is not None:
            balance = f"  (you have {self.balance(player)} coins)"
        lines = [f"🛒 shop{balance} — /game shop buy <slug>"]
        for item in items:
            if item.slug in ("gear_sword", "gear_armor"):
                continue  # legacy one-shots: superseded by durable gear below
            lines.append(f"  {item.slug:<12} {item.name} — {item.cost}c"
                         + (f" · {item.effect}" if item.effect else ""))
        gear_lines = self.gear_catalog_text()
        if gear_lines:
            lines.append("")
            lines.append(gear_lines)
        if len(lines) == 1:
            return "the shop is empty for this game."
        return "\n".join(lines)

    def gear_catalog_text(self) -> str:
        """The durable-gear section of the shop."""
        from .gear import GEAR_CATALOG, effective_stats
        if not GEAR_CATALOG:
            return ""
        lines = ["⚔️ arena gear — durable, wears with use, repairable "
                 "(/inventory, /equip):"]
        by_slot: dict[str, list] = {}
        for defn in GEAR_CATALOG.values():
            by_slot.setdefault(defn.slot, []).append(defn)
        for slot in ("weapon", "armor"):
            pieces = sorted(by_slot.get(slot, []),
                            key=lambda d: (d.cost, d.slug))
            if not pieces:
                continue
            lines.append(f"  — {slot}s —")
            for defn in pieces:
                atk, df = effective_stats(defn)
                stats = f"+{atk} atk" if defn.slot == "weapon" else f"+{df} def"
                set_tag = f" · {defn.set_name} set" if defn.set_name else ""
                lines.append(
                    f"  {defn.slug:<20} {defn.name} — {defn.cost}c · "
                    f"{stats}{set_tag} · {defn.max_durability} dur")
        return "\n".join(lines)

    # ── wallet ───────────────────────────────────────────────────────────────
    def balance(self, player: Player) -> int:
        return self.store.get(player.key).coins

    def purchase(self, player: Player, slug: str) -> tuple[bool, str]:
        """Buy one item. Returns (ok, message).

        Durable gear (swords, armor) becomes a persistent piece in the
        player's inventory with its own durability — it is never a
        one-shot consumable.  Legacy ``gear_sword``/``gear_armor`` slugs
        map onto their modern equivalents.
        """
        from .gear import GEAR_CATALOG, LEGACY_GEAR_MAP, effective_stats
        slug = (slug or "").strip().lower()
        # legacy one-shot slugs → real gear
        if slug in LEGACY_GEAR_MAP:
            slug = LEGACY_GEAR_MAP[slug]
        gear_defn = GEAR_CATALOG.get(slug)
        if gear_defn is not None and self.gear is not None:
            new_balance = self.store.spend_coins(
                player, gear_defn.cost, f"buy:{slug}")
            if new_balance is None:
                prof = self.store.get(player.key)
                return False, (f"{gear_defn.name} costs {gear_defn.cost}c — "
                               f"you have {prof.coins}c. win games to earn more.")
            try:
                inst = self.gear.grant(player.key, slug)
            except Exception as exc:  # noqa: BLE001
                # coins already taken — refund rather than lose them
                self.store.add_coins(player, gear_defn.cost,
                                     f"refund:{slug}")
                return False, f"couldn't forge {gear_defn.name}: {exc}"
            atk, df = effective_stats(gear_defn)
            stats = f"+{atk} atk" if gear_defn.slot == "weapon" else f"+{df} def"
            return True, (f"bought {gear_defn.name} for {gear_defn.cost}c — "
                          f"{stats}, {inst.durability} durability. "
                          f"/equip {slug} to wear it in the arena.")
        item = self._items.get(slug)
        if item is None:
            known = ", ".join(sorted(self._items))
            return False, f"no such item {slug!r}. in stock: {known}"
        new_balance = self.store.spend_coins(player, item.cost,
                                             f"buy:{slug}")
        if new_balance is None:
            prof = self.store.get(player.key)
            return False, (f"{item.name} costs {item.cost}c — you have "
                           f"{prof.coins}c. win games to earn more.")
        self.store.grant_item(player, slug)
        return True, (f"bought {item.name} for {item.cost}c — "
                      f"{item.effect} (you now have "
                      f"{self.count(player, slug)}).")

    def count(self, player: Player, slug: str) -> int:
        return int(self.store.get(player.key).items.get(slug) or 0)

    def consume(self, player: Player, slug: str) -> bool:
        """Spend one owned item. False if they don't have one."""
        return self.store.consume_item(player, slug)

    # ── rewards ──────────────────────────────────────────────────────────────
    #: Difficulty → coin multiplier on the win payout. Playing above your
    #: weight pays; farming the easy table doesn't.
    DIFFICULTY_COIN_MULT = {
        "easy": 0.8,
        "normal": 1.0,
        "hard": 1.5,
        "expert": 2.0,
    }

    #: Win streak → bonus coins, tiered milestones: the longer the streak,
    #: the richer each step pays. ``levels`` is streak_after - 1; each tier
    #: is (streak cap, coins per level).  Earlier tiers are kept — higher
    #: streaks stack on top of them, so a 10-streak earns everything a
    #: 5-streak did, plus the richer 6–10 steps.
    STREAK_BONUS_TIERS: tuple[tuple[float, int], ...] = (
        (5, 10),              # streaks 2–5:   +10 per level
        (10, 15),             # streaks 6–10:  +15 per level
        (15, 25),             # streaks 11–15: +25 per level
        (float("inf"), 40),   # streak 16+:   +40 per level, uncapped
    )

    @staticmethod
    def streak_bonus(streak_after: int) -> int:
        """Tiered win-streak coin bonus. No cap — milestones keep paying."""
        levels = max(0, int(streak_after) - 1)
        bonus, prev = 0, 0
        for cap, per in GameEconomy.STREAK_BONUS_TIERS:
            if levels <= prev:
                break
            bonus += (min(levels, cap) - prev) * per
            prev = cap
        return bonus

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
    def coin_breakdown(won: bool | None, score: int = 0,
                       difficulty: str = "normal",
                       streak_after: int = 0) -> tuple[int, str]:
        """Performance-based coin payout.

        Wins pay a base plus a score bonus (up to +80), multiplied by the
        table difficulty, plus a tiered win-streak bonus (milestones at
        5/10/15+ wins pay richer per step, uncapped).  Losses and
        draws keep flat consolation payouts so the board stays alive.
        Returns ``(total_coins, human_readable_breakdown)``.
        """
        if won is True:
            base = 40
            score_bonus = min(max(0, int(score)), 40) * 2
            mult = GameEconomy.DIFFICULTY_COIN_MULT.get(
                (difficulty or "normal").strip().lower(), 1.0)
            subtotal = int(round((base + score_bonus) * mult))
            streak_bonus = GameEconomy.streak_bonus(streak_after)
            total = subtotal + streak_bonus
            parts = [f"win {base}", f"score +{score_bonus}",
                     f"x{mult:g} {difficulty}"]
            if streak_bonus:
                parts.append(f"streak +{streak_bonus}")
            return total, " + ".join(parts)
        if won is False:
            return 15, "participation"
        return 25, "draw"

    @staticmethod
    def reward_coins(won: bool | None, score: int = 0,
                     difficulty: str = "normal",
                     streak_after: int = 0) -> int:
        """Total coins for a finished game.  Extra kwargs are optional so
        old single-argument callers keep working."""
        total, _ = GameEconomy.coin_breakdown(
            won, score=score, difficulty=difficulty, streak_after=streak_after)
        return total
