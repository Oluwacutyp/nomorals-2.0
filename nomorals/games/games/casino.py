"""Casino games: card games, wheel spins, slot machines.

* **blackjack** — 21 or bust. Hit, stand, double, split. Dealer hits on 16, stands on 17.
* **roulette** — bet on numbers, colors, dozens. Spin the wheel, payouts 35:1 to 1:1.
* **slots** — 3 reels, match symbols. Cherries 2x, bars 5x, sevens 10x, jackpot 100x.

All games use the room's persistent RNG stream, all betting uses the player's coin balance.
"""
from __future__ import annotations

import random
from typing import Any

from ..ai import GameMind
from ..players import Player
from .base import MultiGame, Room

__all__ = ["CASINO_GAMES"]


# ── 1. blackjack ────────────────────────────────────────────────────────────

class BlackjackGame(MultiGame):
    name = "blackjack"
    description = "21 or bust — beat the dealer without going over"
    min_players = 1
    max_players = 1
    ai_seats = 1
    move_timeout = 0
    rules = ("Get as close to 21 as you can without going over. "
             "Face cards = 10, Aces = 1 or 11. Hit (take a card), "
             "Stand (keep your hand), Double (double bet, one more card), "
             "Split (two hands if you have a pair). Dealer hits on 16, "
             "stands on 17. Blackjack pays 3:2, wins pay 1:1.")

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        deck = self._shuffle(rng)
        player_hand = [deck.pop(), deck.pop()]
        dealer_hand = [deck.pop(), deck.pop()]
        return {"deck": deck, "player": player_hand, "dealer": dealer_hand,
                "bet": 10, "done": False, "result": "", "split": None}

    def _shuffle(self, rng: random.Random) -> list[int]:
        """52-card deck: 2–10, J=11, Q=12, K=13, A=14."""
        deck = [v for v in range(2, 15) for _ in range(4)]
        rng.shuffle(deck)
        return deck

    def _card_value(self, card: int) -> int:
        if card == 14:
            return 11  # Ace
        if card >= 10:
            return 10  # 10, J, Q, K
        return card

    def _hand_value(self, hand: list[int]) -> int:
        total = sum(self._card_value(c) for c in hand)
        aces = sum(1 for c in hand if c == 14)
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1
        return total

    def _card_name(self, card: int) -> str:
        names = {11: "J", 12: "Q", 13: "K", 14: "A"}
        return names.get(card, str(card))

    def _render_hand(self, hand: list[int], hide_second: bool = False) -> str:
        if hide_second and len(hand) >= 2:
            return f"[{self._card_name(hand[0])}] [?]"
        return " ".join(f"[{self._card_name(c)}]" for c in hand)

    def setup(self, room, mind):
        s = room.state
        return (f"dealer: {self._render_hand(s['dealer'], hide_second=True)}\n"
                f"you:    {self._render_hand(s['player'])} "
                f"({self._hand_value(s['player'])})\n"
                f"bet: {s['bet']} coins\n"
                "hit / stand / double / split")

    def on_move(self, room, player, text, mind):
        s = room.state
        t = text.strip().lower()
        if s["done"]:
            return ["game over — start a new one with /game blackjack"]
        if t == "hit":
            s["player"].append(s["deck"].pop())
            val = self._hand_value(s["player"])
            out = [f"you draw [{self._card_name(s['player'][-1])}] — "
                   f"{self._render_hand(s['player'])} ({val})"]
            if val > 21:
                s["done"] = True
                s["result"] = "bust"
                out.append("💥 bust — you lose.")
            elif val == 21:
                s["done"] = True
                out.extend(self._dealer_play(room))
            return out
        if t == "stand":
            s["done"] = True
            return self._dealer_play(room)
        if t == "double":
            if len(s["player"]) != 2:
                return ["can only double on the first two cards"]
            s["bet"] *= 2
            s["player"].append(s["deck"].pop())
            out = [f"doubled to {s['bet']} coins, drew "
                   f"[{self._card_name(s['player'][-1])}]"]
            s["done"] = True
            out.extend(self._dealer_play(room))
            return out
        if t == "split":
            if len(s["player"]) != 2 or s["player"][0] != s["player"][1]:
                return ["can only split a pair"]
            s["split"] = [s["player"][1]]
            s["player"] = [s["player"][0]]
            s["player"].append(s["deck"].pop())
            s["split"].append(s["deck"].pop())
            out = ["split into two hands:",
                   f"hand 1: {self._render_hand(s['player'])} "
                   f"({self._hand_value(s['player'])})",
                   f"hand 2: {self._render_hand(s['split'])} "
                   f"({self._hand_value(s['split'])})",
                   "playing hand 1 — hit / stand"]
            return out
        return ["hit / stand / double / split"]

    def _dealer_play(self, room: Room) -> list[str]:
        s = room.state
        out = [f"dealer reveals: {self._render_hand(s['dealer'])} "
               f"({self._hand_value(s['dealer'])})"]
        while self._hand_value(s["dealer"]) < 17:
            s["dealer"].append(s["deck"].pop())
            out.append(f"dealer draws [{self._card_name(s['dealer'][-1])}] — "
                       f"{self._hand_value(s['dealer'])}")
        dealer_val = self._hand_value(s["dealer"])
        player_val = self._hand_value(s["player"])
        if dealer_val > 21:
            s["result"] = "dealer_bust"
            out.append("🏆 dealer busts — you win!")
        elif player_val > dealer_val:
            s["result"] = "win"
            out.append("🏆 you win!")
        elif player_val < dealer_val:
            s["result"] = "lose"
            out.append("dealer wins.")
        else:
            s["result"] = "push"
            out.append("push — bet returned.")
        return out

    def ai_turn(self, room, mind):
        return []

    def is_over(self, room):
        return room.state.get("done", False)

    def winner(self, room):
        r = room.state.get("result", "")
        if r in {"win", "dealer_bust"}:
            return room.players[0] if room.players else None
        if r == "push":
            return "draw"
        return Player(key="ai:dealer", platform="ai", name="The Dealer", is_ai=True)

    def score(self, room, player):
        r = room.state.get("result", "")
        bet = room.state.get("bet", 0)
        if r in {"win", "dealer_bust"}:
            return bet
        if r == "push":
            return 0
        return -bet


# ── 2. roulette ─────────────────────────────────────────────────────────────

class RouletteGame(MultiGame):
    name = "roulette"
    description = "bet on numbers, colors, dozens — spin the wheel"
    min_players = 1
    max_players = 1
    ai_seats = 0
    move_timeout = 0
    rules = ("Bet on a number (0–36), color (red/black), or dozen "
             "(1st/2nd/3rd). The wheel spins, the ball lands, payouts: "
             "number 35:1, color 1:1, dozen 2:1. 0 is green, neither "
             "red nor black.")

    RED = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        return {"bet_type": "", "bet_value": None, "amount": 10,
                "result": None, "done": False}

    def _spin(self, rng: random.Random) -> int:
        return rng.randint(0, 36)

    def _color(self, num: int) -> str:
        if num == 0:
            return "green"
        return "red" if num in self.RED else "black"

    def setup(self, room, mind):
        return ("roulette — place your bet:\n"
                "  bet number <0-36> — pays 35:1\n"
                "  bet red / black — pays 1:1\n"
                "  bet dozen 1 / 2 / 3 — pays 2:1\n"
                "then 'spin' to roll the wheel.")

    def on_move(self, room, player, text, mind):
        s = room.state
        t = text.strip().lower()
        if s["done"]:
            return ["game over — start a new one with /game roulette"]
        if t.startswith("bet "):
            parts = t.split()
            if len(parts) < 2:
                return ["bet number <n> / bet red / bet black / bet dozen <1-3>"]
            kind = parts[1]
            if kind == "number" and len(parts) >= 3 and parts[2].isdigit():
                n = int(parts[2])
                if 0 <= n <= 36:
                    s["bet_type"] = "number"
                    s["bet_value"] = n
                    return [f"bet {s['amount']} on number {n} (pays 35:1)"]
            elif kind in {"red", "black"}:
                s["bet_type"] = "color"
                s["bet_value"] = kind
                return [f"bet {s['amount']} on {kind} (pays 1:1)"]
            elif kind == "dozen" and len(parts) >= 3 and parts[2] in {"1", "2", "3"}:
                s["bet_type"] = "dozen"
                s["bet_value"] = int(parts[2])
                return [f"bet {s['amount']} on dozen {parts[2]} (pays 2:1)"]
            return ["bet number <n> / bet red / bet black / bet dozen <1-3>"]
        if t == "spin":
            if not s["bet_type"]:
                return ["place a bet first"]
            num = self._spin(room.rng())
            color = self._color(num)
            s["result"] = num
            s["done"] = True
            out = [f"the wheel spins… **{num} {color}**"]
            won = False
            payout = 0
            if s["bet_type"] == "number" and s["bet_value"] == num:
                won = True
                payout = s["amount"] * 35
            elif s["bet_type"] == "color" and s["bet_value"] == color:
                won = True
                payout = s["amount"] * 2
            elif s["bet_type"] == "dozen":
                dozen = (num - 1) // 12 + 1 if num > 0 else 0
                if dozen == s["bet_value"]:
                    won = True
                    payout = s["amount"] * 3
            if won:
                out.append(f"🏆 you win {payout} coins!")
                s["payout"] = payout
            else:
                out.append(f"you lose {s['amount']} coins.")
                s["payout"] = -s["amount"]
            return out
        return ["bet <type> then spin"]

    def ai_turn(self, room, mind):
        return []

    def is_over(self, room):
        return room.state.get("done", False)

    def winner(self, room):
        payout = room.state.get("payout", 0)
        if payout > 0:
            return room.players[0] if room.players else None
        return Player(key="ai:roulette", platform="ai", name="The House", is_ai=True)

    def score(self, room, player):
        return room.state.get("payout", 0)


# ── 3. slots ────────────────────────────────────────────────────────────────

class SlotsGame(MultiGame):
    name = "slots"
    description = "3 reels, match symbols — cherries, bars, sevens, jackpot"
    min_players = 1
    max_players = 1
    ai_seats = 0
    move_timeout = 0
    rules = ("Spin the reels. Three matching symbols pays: cherries 2x, "
             "bars 5x, sevens 10x. Three diamonds = JACKPOT 100x. "
             "Two matching symbols pays 1x (break even).")

    SYMBOLS = ("🍒", "🍋", "🍊", "🔔", "⭐", "7️⃣", "💎")
    WEIGHTS = (30, 20, 20, 15, 10, 4, 1)  # diamonds rare, cherries common

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        return {"bet": 10, "reels": [], "done": False, "payout": 0}

    def _spin_reel(self, rng: random.Random) -> str:
        return rng.choices(self.SYMBOLS, weights=self.WEIGHTS, k=1)[0]

    def setup(self, room, mind):
        return ("slots — bet and spin:\n"
                "  bet <amount> — set your wager\n"
                "  spin — pull the lever\n"
                "pays: 🍒🍒🍒 2x, 🔔🔔🔔 5x, 7️⃣7️⃣7️⃣ 10x, 💎💎💎 100x")

    def on_move(self, room, player, text, mind):
        s = room.state
        t = text.strip().lower()
        if s["done"]:
            return ["game over — start a new one with /game slots"]
        if t.startswith("bet "):
            parts = t.split()
            if len(parts) >= 2 and parts[1].isdigit():
                amt = int(parts[1])
                if amt > 0:
                    s["bet"] = amt
                    return [f"bet set to {amt} coins"]
            return ["bet <amount>"]
        if t == "spin":
            reels = [self._spin_reel(room.rng()) for _ in range(3)]
            s["reels"] = reels
            s["done"] = True
            out = [f"[ {reels[0]} ] [ {reels[1]} ] [ {reels[2]} ]"]
            if reels[0] == reels[1] == reels[2]:
                sym = reels[0]
                if sym == "💎":
                    mult = 100
                    out.append("💎💎💎 JACKPOT! 💎💎💎")
                elif sym == "7️⃣":
                    mult = 10
                elif sym == "🔔":
                    mult = 5
                elif sym == "🍒":
                    mult = 2
                else:
                    mult = 3  # other triples
                payout = s["bet"] * mult
                s["payout"] = payout
                out.append(f"🏆 three {sym} — you win {payout} coins!")
            elif reels[0] == reels[1] or reels[1] == reels[2] or reels[0] == reels[2]:
                s["payout"] = 0  # break even
                out.append("two matching — bet returned.")
            else:
                s["payout"] = -s["bet"]
                out.append(f"no match — you lose {s['bet']} coins.")
            return out
        return ["bet <amount> then spin"]

    def ai_turn(self, room, mind):
        return []

    def is_over(self, room):
        return room.state.get("done", False)

    def winner(self, room):
        payout = room.state.get("payout", 0)
        if payout > 0:
            return room.players[0] if room.players else None
        if payout == 0:
            return "draw"
        return Player(key="ai:slots", platform="ai", name="The House", is_ai=True)

    def score(self, room, player):
        return room.state.get("payout", 0)


CASINO_GAMES: tuple[MultiGame, ...] = (
    BlackjackGame(),
    RouletteGame(),
    SlotsGame(),
)
