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

__all__ = ["ShopItem", "GameEconomy", "DEFAULT_SHOP",
           # ledger / health / gems / dynamic pricing
           "record_ledger", "economy_health", "render_economy_health",
           "gem_balance", "grant_gems", "spend_gems", "GEM_SHOP",
           "gem_shop_text", "earn_velocity", "dynamic_price"]


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
    # ── puzzle power-ups: for the thinking games ──
    ShopItem("xray", "X-Ray", 150,
             "sudoku: reveal one cell", "sudoku"),
    ShopItem("pencil", "Pencil Marks", 100,
             "sudoku: auto-fill all pencil marks", "sudoku"),
    ShopItem("letter_reveal", "Letter Reveal", 120,
             "anagram/cryptogram/wordle: reveal one letter", "any"),
    ShopItem("word_bank", "Word Bank", 200,
             "anagram: show 3 candidate words", "anagram"),
    ShopItem("code_crack", "Code Cracker", 180,
             "cryptogram: reveal the most common letter", "cryptogram"),
    # ── board game aids: for the strategy tables ──
    ShopItem("second_chance", "Second Chance", 150,
             "undo your last move", "any"),
    ShopItem("oracle", "Oracle", 250,
             "ttt/gomoku/checkers/chess: the house suggests its best move",
             "any"),
    ShopItem("time_freeze", "Time Freeze", 100,
             "pause the clock for one turn", "any"),
    # ── boosters: for every game ──
    ShopItem("double_xp", "Double XP Charm", 300,
             "next game pays double XP", "any"),
    ShopItem("coin_charm", "Coin Charm", 250,
             "next game pays +50% coins", "any"),
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
        from .gear import GEAR_CATALOG, durability_display, effective_stats
        if not GEAR_CATALOG:
            return ""
        lines = ["⚔️ arena gear — durable, wears with use, repairable "
                 "(/inventory, /equip):"]
        by_slot: dict[str, list] = {}
        for defn in GEAR_CATALOG.values():
            if defn.raid_only:
                continue  # raid drops only — never for sale
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
                dur_tag = ("∞ unbreakable"
                           if defn.unbreakable
                           else f"{defn.max_durability} dur")
                lines.append(
                    f"  {defn.slug:<20} {defn.name} — {defn.cost}c · "
                    f"{stats}{set_tag} · {dur_tag}")
        return "\n".join(lines)

    # ── wallet ───────────────────────────────────────────────────────────────
    def balance(self, player: Player) -> int:
        return self.store.get_for(player).coins

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
            if gear_defn.raid_only:
                return False, (f"{gear_defn.name} isn't for sale — it "
                               f"drops from raid bosses. /raid and get "
                               f"lucky.")
            new_balance = self.store.spend_coins(
                player, gear_defn.cost, f"buy:{slug}")
            if new_balance is None:
                prof = self.store.get_for(player)
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
            dur_note = ("∞ unbreakable — it will never wear or break."
                        if gear_defn.unbreakable
                        else f"{inst.durability} durability.")
            _ledger_sink(self.store, player, gear_defn.cost, f"shop:{slug}")
            return True, (f"bought {gear_defn.name} for {gear_defn.cost}c — "
                          f"{stats}, {dur_note} "
                          f"/equip {slug} to wear it in the arena.")
        item = self._items.get(slug)
        if item is None:
            known = ", ".join(sorted(self._items))
            return False, f"no such item {slug!r}. in stock: {known}"
        new_balance = self.store.spend_coins(player, item.cost,
                                             f"buy:{slug}")
        if new_balance is None:
            prof = self.store.get_for(player)
            return False, (f"{item.name} costs {item.cost}c — you have "
                           f"{prof.coins}c. win games to earn more.")
        self.store.grant_item(player, slug)
        _ledger_sink(self.store, player, item.cost, f"shop:{slug}")
        return True, (f"bought {item.name} for {item.cost}c — "
                      f"{item.effect} (you now have "
                      f"{self.count(player, slug)}).")

    def count(self, player: Player, slug: str) -> int:
        return int(self.store.get_for(player).items.get(slug) or 0)

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


# ── ledger: every coin movement, tagged source/sink ─────────────────────────
#
# F2P economy design starts with enumerating EVERY source (faucet) and
# EVERY sink (drain). Without the ledger you're tuning blind. Every
# coin grant/spend records (kind, category); ``economy_health`` reads
# the faucet/drain ratio and calls out inflation before players notice.

import time as _etime


def _ledger_sink(store: Any, player: Any, amount: int,
                 category: str) -> None:
    """Best-effort sink record for a purchase. Never raises."""
    try:
        db = getattr(store, "db", None)
        if db is None:
            return
        record_ledger(db, getattr(player, "key", ""), "sink", amount,
                      category)
    except Exception:  # noqa: BLE001
        pass


def _ensure_ledger(db: Any) -> None:
    try:
        db.execute(
            "CREATE TABLE IF NOT EXISTS game_ledger ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "player_key TEXT NOT NULL DEFAULT '', "
            "kind TEXT NOT NULL, "          # "source" | "sink"
            "category TEXT NOT NULL, "       # win, shop, repair, streak…
            "amount INTEGER NOT NULL, "
            "at REAL NOT NULL DEFAULT 0)")
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_game_ledger_at "
            "ON game_ledger(at)")
    except Exception:  # noqa: BLE001
        pass


def record_ledger(db: Any, player_key: str, kind: str, amount: int,
                  category: str) -> None:
    """Log a coin movement. ``kind`` is "source" (coins created) or
    "sink" (coins destroyed). Never raises."""
    if kind not in ("source", "sink") or not amount:
        return
    _ensure_ledger(db)
    try:
        db.execute(
            "INSERT INTO game_ledger (player_key, kind, category, amount,"
            " at) VALUES (?, ?, ?, ?, ?)",
            (player_key or "", kind, category or "misc", abs(int(amount)),
             _etime.time()))
    except Exception:  # noqa: BLE001
        pass


def economy_health(db: Any, days: int = 7) -> dict[str, Any]:
    """Faucet/drain diagnosis over the last ``days`` days.

    Returns ``{'sources', 'sinks', 'ratio', 'verdict', 'top_sources',
    'top_sinks'}``. Ratio > 1.3 sustained means inflation — coins are
    being printed faster than destroyed; < 0.7 means the economy is
    starving players.
    """
    _ensure_ledger(db)
    out: dict[str, Any] = {"sources": 0, "sinks": 0, "ratio": 0.0,
                           "verdict": "no data yet", "top_sources": [],
                           "top_sinks": []}
    try:
        since = _etime.time() - days * 86400.0
        rows = db.query(
            "SELECT kind, category, SUM(amount) AS total FROM game_ledger "
            "WHERE at >= ? GROUP BY kind, category ORDER BY total DESC",
            (since,)) or []
    except Exception:  # noqa: BLE001
        return out
    for r in rows:
        total = int(r.get("total") or 0)
        if r.get("kind") == "source":
            out["sources"] += total
            out["top_sources"].append((r.get("category"), total))
        else:
            out["sinks"] += total
            out["top_sinks"].append((r.get("category"), total))
    if out["sinks"] > 0:
        out["ratio"] = round(out["sources"] / out["sinks"], 2)
    elif out["sources"] > 0:
        out["ratio"] = float("inf")
    r = out["ratio"]
    if out["sources"] == 0 and out["sinks"] == 0:
        out["verdict"] = "no data yet"
    elif r == float("inf") or r > 1.5:
        out["verdict"] = ("🔥 INFLATION RISK — faucets dwarf sinks; "
                          "add sinks (repairs, prestige items, taxes) "
                          "or trim win payouts")
    elif r > 1.3:
        out["verdict"] = "⚠️ warming — sources outpace sinks, watch it"
    elif r < 0.7:
        out["verdict"] = ("🧊 STARVED — sinks eat faster than players earn; "
                          "loosen payouts or players churn")
    else:
        out["verdict"] = "✅ balanced — faucets and sinks in equilibrium"
    return out


def render_economy_health(db: Any, days: int = 7) -> str:
    h = economy_health(db, days=days)
    lines = [f"💹 economy health (last {days}d): {h['verdict']}"]
    lines.append(f"  sources (faucets): {h['sources']}c · "
                 f"sinks (drains): {h['sinks']}c · ratio {h['ratio']}")
    if h["top_sources"]:
        lines.append("  top faucets: " + ", ".join(
            f"{c} {t}c" for c, t in h["top_sources"][:4]))
    if h["top_sinks"]:
        lines.append("  top sinks: " + ", ".join(
            f"{c} {t}c" for c, t in h["top_sinks"][:4]))
    return "\n".join(lines)


# ── gems: the hard currency ─────────────────────────────────────────────────
#
# Two-currency design (soft coins + hard gems) is the F2P standard: coins
# are the grind, gems are the scarce prestige layer. Gems are NEVER sold
# here and never drop from regular play — they're earned from
# achievements, season completion, and tournament championships. They buy
# prestige: name cosmetics, exclusive loot rerolls, streak freezes.

def _ensure_gems(db: Any) -> None:
    try:
        db.execute(
            "CREATE TABLE IF NOT EXISTS game_gems ("
            "player_key TEXT PRIMARY KEY, "
            "gems INTEGER NOT NULL DEFAULT 0)")
    except Exception:  # noqa: BLE001
        pass


def gem_balance(db: Any, player_key: str) -> int:
    _ensure_gems(db)
    try:
        row = db.query_one(
            "SELECT gems FROM game_gems WHERE player_key = ?",
            (player_key,))
        return int((row or {}).get("gems") or 0)
    except Exception:  # noqa: BLE001
        return 0


def grant_gems(db: Any, player_key: str, amount: int,
               reason: str = "") -> int:
    """Award gems (achievements, seasons, championships). Returns the new
    balance. Logs to the ledger as a source."""
    _ensure_gems(db)
    amount = max(0, int(amount))
    if not amount:
        return gem_balance(db, player_key)
    try:
        db.execute(
            "INSERT INTO game_gems (player_key, gems) VALUES (?, ?) "
            "ON CONFLICT(player_key) DO UPDATE SET "
            "gems = game_gems.gems + excluded.gems",
            (player_key, amount))
    except Exception:  # noqa: BLE001
        pass
    record_ledger(db, player_key, "source", amount,
                  f"gems:{reason or 'grant'}")
    return gem_balance(db, player_key)


def spend_gems(db: Any, player_key: str, amount: int,
               reason: str = "") -> bool:
    """Spend gems. False when the balance is too low. Logs the sink."""
    _ensure_gems(db)
    amount = max(0, int(amount))
    if gem_balance(db, player_key) < amount:
        return False
    try:
        db.execute("UPDATE game_gems SET gems = gems - ? "
                   "WHERE player_key = ?", (amount, player_key))
    except Exception:  # noqa: BLE001
        return False
    record_ledger(db, player_key, "sink", amount,
                  f"gems:{reason or 'spend'}")
    return True


#: gem shop: prestige only — nothing here touches win rates.
GEM_SHOP: tuple[tuple[str, str, int], ...] = (
    ("loot_reroll", "🎲 loot reroll — reforge your last drop's affixes", 5),
    ("streak_freeze", "🧊 streak freeze — protects one missed day", 8),
    ("name_glow", "✨ glowing name on leaderboards (30 days)", 15),
    ("myth_charm", "💫 myth charm — next raid drop rolls lucky", 25),
)


def gem_shop_text(db: Any, player_key: str) -> str:
    lines = [f"💎 gem shop — you have {gem_balance(db, player_key)} gems "
             f"(/game gems buy <slug>)"]
    for slug, desc, cost in GEM_SHOP:
        lines.append(f"  {slug:<14} {desc} — {cost}💎")
    lines.append("gems are earned from achievements, seasons and "
                 "tournament wins — never sold.")
    return "\n".join(lines)


# ── dynamic pricing: the shop reads the room ────────────────────────────────
#
# Static prices rot: as the player base gets richer, fixed costs become
# trivial. Dynamic pricing nudges shop prices with server-wide earn
# velocity — when everyone's flush, prices drift up (a soft sink);
# when the economy is starved, they drift down.

def earn_velocity(db: Any, days: int = 7) -> float:
    """Average coins earned per active player per day (sources only)."""
    _ensure_ledger(db)
    try:
        since = _etime.time() - days * 86400.0
        rows = db.query(
            "SELECT SUM(amount) AS total, "
            "COUNT(DISTINCT player_key) AS players FROM game_ledger "
            "WHERE kind = 'source' AND at >= ?", (since,)) or []
        total = int((rows[0] or {}).get("total") or 0)
        players = int((rows[0] or {}).get("players") or 0)
        if not players:
            return 0.0
        return total / players / max(1, days)
    except Exception:  # noqa: BLE001
        return 0.0


def dynamic_price(base_cost: int, db: Any = None,
                  baseline_velocity: float = 120.0) -> int:
    """Adjust ``base_cost`` to server earn velocity.

    Baseline: 120 coins/player/day → price ×1.0. Every doubling of
    velocity adds +15% (diminishing), every halving cuts 10%, clamped
    to [0.7×, 2.0×]. With no ledger data the base price stands.
    """
    if db is None:
        return base_cost
    vel = earn_velocity(db)
    if vel <= 0:
        return base_cost
    import math as _m
    steps = _m.log2(max(vel, 1) / baseline_velocity)
    mult = 1.0 + (0.15 * steps if steps >= 0 else 0.10 * steps)
    mult = max(0.7, min(2.0, mult))
    return max(1, int(round(base_cost * mult)))
