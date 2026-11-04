"""Casino games: card games, wheel spins, slot machines.

* **blackjack** — 21 or bust. Hit, stand, double, split. Dealer hits on 16, stands on 17.
* **roulette** — bet on numbers, colors, dozens. Spin the wheel, payouts 35:1 to 1:1.
* **slots** — 3 reels, match symbols. Cherries 2x, bars 5x, sevens 10x, jackpot 100x.

All games use the room's persistent RNG stream.  All betting uses
the player's REAL coin balance: a bet debits (escrows) from the
room's bound purse the moment it is placed, and the hand settles
from that escrow at the stated odds — win pays out of the escrow,
push returns it, loss keeps it.  ``score()`` is the player's NET
coin result everywhere in this module.
"""
from __future__ import annotations

import random
from typing import Any

from ..ai import GameMind
from ..fairness import draw_int, draw_weighted, shuffle as fair_shuffle
from ..gamemaster import feed
from ..players import Player
from .base import MultiGame, Room, table_purse

__all__ = ["CASINO_GAMES"]


def _human(room: Room) -> Player | None:
    for p in room.players:
        if not p.is_ai:
            return p
    return None


def _balance_line(room: Room) -> str:
    """The bettor's current purse balance, for the game panel — the
    number shown is the number that moves."""
    purse = table_purse(room)
    human = _human(room)
    if purse is None or human is None:
        return ""
    return f"your balance: {purse.balance(human)} coins ({purse.name})"


# ── 1. blackjack ────────────────────────────────────────────────────────────

class BlackjackGame(MultiGame):
    name = "blackjack"
    description = "21 or bust — beat the dealer without going over"
    min_players = 1
    max_players = 1
    ai_seats = 1
    move_timeout = 0
    takes_wagers = True
    #: provably fair: the deck order is committed before the deal
    fair = True
    rules = ("Get as close to 21 as you can without going over. "
             "Face cards = 10, Aces = 1 or 11. Hit (take a card), "
             "Stand (keep your hand), Double (double bet, one more card), "
             "Split (two hands if you have a pair). Dealer hits on 16, "
             "stands on 17. Blackjack pays 3:2, wins pay 1:1. "
             "🔒 the deck is provably fair — commitment shown at deal.")

    def new_state(self, rng: random.Random,
                  bet: int = 0) -> dict[str, Any]:
        # the deck is ordered here; the engine opens the fair table right
        # after new_state, and setup() fair-shuffles + deals from the
        # committed sequence — nothing is drawn before the commitment.
        # ``bet`` is the one-command stake (/blackjack <amount>): the
        # hand setup() stakes and escrows before the deal; 0/absent
        # keeps the house default.
        deck = [v for v in range(2, 15) for _ in range(4)]
        return {"deck": deck, "player": [], "dealer": [],
                "bet": bet if bet > 0 else 10,
                "done": False, "result": "", "split": None,
                "dealt": False,
                # wager state: coins escrowed from the purse for this
                # hand, the split hand's stake + result, the dealer's
                # original two cards (natural check), the dealt-
                # natural flag, and the settle guard.
                "escrow": 0, "split_bet": 0, "result2": "",
                "dealer_start": [], "natural": False,
                "wager_settled": False, "wager_net": 0}

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
        # fair deal: shuffle + deal from the committed sequence. setup
        # runs after the engine opens the fair table (see GameEngine.start).
        fair = s.get("fair")
        if not s["dealt"]:
            # the stake leaves the purse BEFORE any card is drawn —
            # a player who can't cover it never sits down to a hand.
            human = _human(room)
            purse = table_purse(room)
            if purse is not None and human is not None:
                if not purse.debit(human, int(s["bet"]),
                                   "blackjack bet"):
                    raise ValueError(
                        f"blackjack stakes {s['bet']} coins — "
                        f"{human.name} has {purse.balance(human)} in "
                        f"the {purse.name}. Earn coins in any game "
                        "first, then come back.")
                s["escrow"] = int(s["bet"])
            if fair is not None:
                fair_shuffle(fair, "deck", s["deck"])
            else:  # pragma: no cover - engine always opens the table
                room.rng().shuffle(s["deck"])
            s["player"] = [s["deck"].pop(), s["deck"].pop()]
            s["dealer"] = [s["deck"].pop(), s["deck"].pop()]
            s["dealt"] = True
            s["dealer_start"] = list(s["dealer"])
            if self._hand_value(s["player"]) == 21:
                s["natural"] = True
                who = human.name if human is not None else "the player"
                feed(room, f"{who} is dealt a natural BLACKJACK",
                     big=True)
        bet_line = f"bet: {s['bet']} coins"
        if s.get("escrow"):
            bet_line += " — staked from your balance"
        tail = "hit / stand / double / split"
        bal = _balance_line(room)
        if bal:
            tail += f"\n{bal}"
        return (f"dealer: {self._render_hand(s['dealer'], hide_second=True)}\n"
                f"you:    {self._render_hand(s['player'])} "
                f"({self._hand_value(s['player'])})\n"
                f"{bet_line}\n"
                f"{tail}")

    def on_move(self, room, player, text, mind):
        s = room.state
        t = text.strip().lower()
        if s["done"]:
            return ["game over — start a new one with /game blackjack"]
        if t == "hit":
            s["natural"] = False  # a drawn card ends the dealt natural
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
            purse = table_purse(room)
            if purse is not None and not player.is_ai:
                add = int(s["bet"])
                if not purse.debit(player, add, "blackjack double"):
                    return [f"doubling stakes {add} more coins — "
                            f"your balance is {purse.balance(player)} "
                            f"in the {purse.name}, not enough"]
                s["escrow"] = int(s.get("escrow", 0)) + add
            s["bet"] *= 2
            s["natural"] = False
            s["player"].append(s["deck"].pop())
            out = [f"doubled to {s['bet']} coins, drew "
                   f"[{self._card_name(s['player'][-1])}]"]
            s["done"] = True
            out.extend(self._dealer_play(room))
            return out
        if t == "split":
            if len(s["player"]) != 2 or s["player"][0] != s["player"][1]:
                return ["can only split a pair"]
            purse = table_purse(room)
            if purse is not None and not player.is_ai:
                add = int(s["bet"])
                if not purse.debit(player, add, "blackjack split"):
                    return [f"splitting stakes {add} more coins — "
                            f"your balance is {purse.balance(player)} "
                            f"in the {purse.name}, not enough"]
                s["escrow"] = int(s.get("escrow", 0)) + add
            s["split_bet"] = int(s["bet"])
            s["natural"] = False
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
        if s.get("split") is not None:
            # the second split hand stands as dealt and settles
            # against the same dealer hand
            v2 = self._hand_value(s["split"])
            if dealer_val > 21 or v2 > dealer_val:
                s["result2"] = "win"
                out.append(f"hand 2: {self._render_hand(s['split'])} "
                           f"({v2}) — 🏆 wins.")
            elif v2 < dealer_val:
                s["result2"] = "lose"
                out.append(f"hand 2: {self._render_hand(s['split'])} "
                           f"({v2}) — loses.")
            else:
                s["result2"] = "push"
                out.append(f"hand 2: {self._render_hand(s['split'])} "
                           f"({v2}) — push.")
        return out

    def ai_turn(self, room, mind):
        return []

    def is_over(self, room):
        return room.state.get("done", False)

    def winner(self, room):
        s = room.state
        if s.get("split") is not None and s.get("done"):
            # a split table is won on the combined money result
            net = self._wager_math(room)[1]
            if net > 0:
                return room.players[0] if room.players else None
            if net < 0:
                return Player(key="ai:dealer", platform="ai",
                              name="The Dealer", is_ai=True)
            return "draw"
        r = s.get("result", "")
        if r in {"win", "dealer_bust"}:
            return room.players[0] if room.players else None
        if r == "push":
            return "draw"
        return Player(key="ai:dealer", platform="ai", name="The Dealer", is_ai=True)

    # ── wager math: everything settles from the escrow ──

    @staticmethod
    def _hand_credit(result: str, bet: int, natural: bool = False) -> int:
        """Total credit for one hand on a win line: a plain win pays
        1:1 (stake + equal winnings), blackjack pays 3:2, a push
        returns the stake, anything else returns nothing."""
        if result in {"win", "dealer_bust"}:
            if natural:
                return bet + (3 * bet) // 2
            return 2 * bet
        if result == "push":
            return bet
        return 0

    def _wager_math(self, room: Room) -> tuple[int, int]:
        """(total credit, net) for the finished hand, from state
        alone — pure, so score() and the settle read the same truth."""
        s = room.state
        bet = int(s.get("bet", 0))
        natural = bool(s.get("natural")) \
            and len(s.get("player", [])) == 2 \
            and self._hand_value(s["player"]) == 21
        start = s.get("dealer_start") or []
        dealer_natural = len(start) == 2 \
            and self._hand_value(list(start)) == 21
        if natural and dealer_natural:
            credit = bet  # blackjack vs blackjack — push
        elif natural:
            credit = bet + (3 * bet) // 2
        else:
            credit = self._hand_credit(s.get("result", ""), bet)
        escrow = int(s.get("escrow", 0))
        if s.get("split") is not None:
            split_bet = int(s.get("split_bet", 0)) or bet
            credit += self._hand_credit(s.get("result2", ""), split_bet)
            if escrow <= 0:
                escrow = bet + split_bet
        elif escrow <= 0:
            escrow = bet
        return credit, credit - escrow

    def score(self, room, player):
        return self._wager_math(room)[1]

    def wager_settle(self, room, player):
        s = room.state
        purse = table_purse(room)
        if purse is None or player.is_ai:
            return None
        if s.get("wager_settled"):
            return (int(s.get("wager_net", 0)), "blackjack wager")
        escrow = int(s.get("escrow", 0))
        if not s.get("done"):
            # table closed mid-hand: the stake comes back whole
            if escrow > 0:
                purse.credit(player, escrow, "blackjack bet refund")
            s["wager_settled"] = True
            s["wager_net"] = 0
            return (0, "blackjack bet refunded — table closed")
        credit, net = self._wager_math(room)
        if credit > 0:
            purse.credit(player, credit, "blackjack payout")
        s["wager_settled"] = True
        s["wager_net"] = net
        if net > 0:
            why = f"blackjack paid {credit} on {escrow} staked"
        elif net == 0:
            why = "blackjack push — stake returned"
        else:
            why = "blackjack wager lost"
        return (net, why)

    def wager_leave(self, room, player):
        s = room.state
        purse = table_purse(room)
        if purse is None or player.is_ai or s.get("done"):
            return None
        escrow = int(s.get("escrow", 0))
        if escrow <= 0:
            return None
        purse.credit(player, escrow, "blackjack bet refund")
        s["escrow"] = 0
        s["wager_settled"] = True
        s["wager_net"] = 0
        return (f"💰 {player.name}'s {escrow}-coin blackjack bet "
                "was refunded.")


# ── 2. roulette ─────────────────────────────────────────────────────────────

class RouletteGame(MultiGame):
    name = "roulette"
    description = "bet on numbers, colors, dozens — spin the wheel"
    min_players = 1
    max_players = 1
    ai_seats = 0
    move_timeout = 0
    takes_wagers = True
    #: provably fair: the spin is drawn from the committed sequence
    fair = True
    rules = ("Bet on a number (0–36), color (red/black), or dozen "
             "(1st/2nd/3rd). The wheel spins, the ball lands, payouts: "
             "number 35:1, color 1:1, dozen 2:1. 0 is green, neither "
             "red nor black. 🔒 every spin is provably fair.")

    RED = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        return {"bet_type": "", "bet_value": None, "amount": 10,
                "result": None, "done": False,
                # wager state: the stake escrowed from the purse,
                # the total-credit multiplier on a win, the credit
                # itself, and the settle guard.
                "escrow": 0, "mult": 0, "credit": 0,
                "wager_settled": False, "wager_net": 0}

    def _spin(self, rng: random.Random) -> int:
        return rng.randint(0, 36)

    def _fair_spin(self, room: Room) -> int:
        fair = (room.state or {}).get("fair")
        if fair is not None:
            return draw_int(fair, "spin", 0, 36)
        return room.rng().randint(0, 36)

    def _color(self, num: int) -> str:
        if num == 0:
            return "green"
        return "red" if num in self.RED else "black"

    def setup(self, room, mind):
        text = ("roulette — place your bet:\n"
                "  bet number <0-36> — pays 35:1\n"
                "  bet red / black — pays 1:1\n"
                "  bet dozen 1 / 2 / 3 — pays 2:1\n"
                "then 'spin' to roll the wheel.")
        bal = _balance_line(room)
        return text + (f"\n{bal}" if bal else "")

    def _place_bet(self, room, player, kind, value, label):
        """Set the bet.  The first placement escrows the stake from
        the purse; moving the bet afterwards keeps the same stake."""
        s = room.state
        purse = table_purse(room)
        if purse is None or player.is_ai:
            s["bet_type"] = kind
            s["bet_value"] = value
            return f"bet {s['amount']} on {label}"
        if int(s.get("escrow", 0)) <= 0:
            if not purse.debit(player, int(s["amount"]),
                               "roulette bet"):
                return (f"a bet stakes {s['amount']} coins — your "
                        f"balance is {purse.balance(player)} in the "
                        f"{purse.name}, not enough")
            s["escrow"] = int(s["amount"])
            s["bet_type"] = kind
            s["bet_value"] = value
            return (f"bet {s['amount']} on {label} — taken from "
                    "your balance")
        s["bet_type"] = kind
        s["bet_value"] = value
        return (f"bet moved: {s['amount']} on {label} "
                "(stake already taken)")

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
                    return [self._place_bet(
                        room, player, "number", n,
                        f"number {n} (pays 35:1)")]
            elif kind in {"red", "black"}:
                return [self._place_bet(
                    room, player, "color", kind,
                    f"{kind} (pays 1:1)")]
            elif kind == "dozen" and len(parts) >= 3 and parts[2] in {"1", "2", "3"}:
                return [self._place_bet(
                    room, player, "dozen", int(parts[2]),
                    f"dozen {parts[2]} (pays 2:1)")]
            return ["bet number <n> / bet red / bet black / bet dozen <1-3>"]
        if t == "spin":
            if not s["bet_type"]:
                return ["place a bet first"]
            num = self._fair_spin(room)
            color = self._color(num)
            s["result"] = num
            s["done"] = True
            out = [f"the wheel spins… **{num} {color}**"]
            won = False
            # total-credit multipliers: the stated odds (35:1 / 1:1
            # / 2:1 profit) PLUS the stake back — that is what
            # actually lands in the purse.
            mult_total = 0
            if s["bet_type"] == "number" and s["bet_value"] == num:
                won = True
                mult_total = 36
                feed(room,
                     f"{player.name} calls the number {num} — "
                     f"it lands, {s['amount'] * 35} coins!",
                     big=True)
            elif s["bet_type"] == "color" and s["bet_value"] == color:
                won = True
                mult_total = 2
            elif s["bet_type"] == "dozen":
                dozen = (num - 1) // 12 + 1 if num > 0 else 0
                if dozen == s["bet_value"]:
                    won = True
                    mult_total = 3
            if won:
                credit = s["amount"] * mult_total
                s["mult"] = mult_total
                s["credit"] = credit
                # score is the NET result: profit after the stake
                s["payout"] = credit - s["amount"]
                out.append(f"🏆 you win {credit} coins!")
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

    def wager_settle(self, room, player):
        s = room.state
        purse = table_purse(room)
        if purse is None or player.is_ai:
            return None
        if s.get("wager_settled"):
            return (int(s.get("wager_net", 0)), "roulette wager")
        escrow = int(s.get("escrow", 0))
        if not s.get("done"):
            # table closed before the spin: the stake comes back
            if escrow > 0:
                purse.credit(player, escrow, "roulette bet refund")
            s["wager_settled"] = True
            s["wager_net"] = 0
            return (0, "roulette bet refunded — table closed")
        net = int(s.get("payout", 0))
        credit = escrow + net
        if credit > 0:
            purse.credit(player, credit, "roulette payout")
        s["wager_settled"] = True
        s["wager_net"] = net
        if net > 0:
            why = (f"roulette paid {credit} on a "
                   f"{int(s.get('amount', 0))} bet")
        elif escrow > 0:
            why = "roulette wager lost"
        else:
            why = "roulette — no wager"
        return (net, why)

    def wager_leave(self, room, player):
        s = room.state
        purse = table_purse(room)
        if purse is None or player.is_ai or s.get("done"):
            return None
        escrow = int(s.get("escrow", 0))
        if escrow <= 0:
            return None
        purse.credit(player, escrow, "roulette bet refund")
        s["escrow"] = 0
        s["wager_settled"] = True
        s["wager_net"] = 0
        return (f"💰 {player.name}'s {escrow}-coin roulette bet "
                "was refunded.")


# ── 3. slots ────────────────────────────────────────────────────────────────

class SlotsGame(MultiGame):
    name = "slots"
    description = "3 reels, match symbols — cherries, bars, sevens, jackpot"
    min_players = 1
    max_players = 1
    ai_seats = 0
    move_timeout = 0
    takes_wagers = True
    #: provably fair: the reels are drawn from the committed sequence
    fair = True
    rules = ("Spin the reels. Three matching symbols pays: cherries 2x, "
             "bars 5x, sevens 10x. Three diamonds = JACKPOT 100x. "
             "Two matching symbols pays 1x (break even). "
             "🔒 every spin is provably fair.")

    SYMBOLS = ("🍒", "🍋", "🍊", "🔔", "⭐", "7️⃣", "💎")
    WEIGHTS = (30, 20, 20, 15, 10, 4, 1)  # diamonds rare, cherries common

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        return {"bet": 10, "reels": [], "done": False, "payout": 0,
                # wager state: the stake escrowed from the purse,
                # the winning multiplier, and the settle guard.
                # "payout" is the NET result (credit minus stake).
                "escrow": 0, "mult": 0,
                "wager_settled": False, "wager_net": 0}

    def _spin_reel(self, rng: random.Random) -> str:
        return rng.choices(self.SYMBOLS, weights=self.WEIGHTS, k=1)[0]

    def _fair_reel(self, room: Room, i: int) -> str:
        fair = (room.state or {}).get("fair")
        if fair is not None:
            return self.SYMBOLS[draw_weighted(fair, f"reel:{i}", self.WEIGHTS)]
        return room.rng().choices(self.SYMBOLS, weights=self.WEIGHTS, k=1)[0]

    def setup(self, room, mind):
        text = ("slots — bet and spin:\n"
                "  bet <amount> — set your wager\n"
                "  spin — pull the lever\n"
                "  /slots <amount> — new table: bet + first spin in "
                "one command\n"
                "pays: 🍒🍒🍒 2x, 🔔🔔🔔 5x, 7️⃣7️⃣7️⃣ 10x, 💎💎💎 100x")
        bal = _balance_line(room)
        return text + (f"\n{bal}" if bal else "")

    def _place_bet(self, room, player, amt):
        """Set the wager: the amount is escrowed from the purse
        NOW.  Changing the bet refunds the old stake and takes the
        new one; a bet the balance can't cover changes nothing."""
        s = room.state
        purse = table_purse(room)
        if purse is None or player.is_ai:
            s["bet"] = amt
            return f"bet set to {amt} coins"
        escrow = int(s.get("escrow", 0))
        bal = purse.balance(player)
        if bal + escrow < amt:
            return (f"bet is {amt} coins — your balance is {bal} "
                    f"in the {purse.name}, not enough")
        if escrow:
            purse.credit(player, escrow, "slots bet change")
            s["escrow"] = 0
        if not purse.debit(player, amt, "slots bet"):
            return (f"bet is {amt} coins — your balance is "
                    f"{purse.balance(player)} in the {purse.name}, "
                    "not enough")
        s["bet"] = amt
        s["escrow"] = amt
        return (f"bet set to {amt} coins — taken from your balance "
                f"({purse.balance(player)} left)")

    def _ensure_stake(self, room, player):
        """Spin stakes the standing bet if it isn't escrowed yet.
        Returns True, or the refusal line to show the player."""
        s = room.state
        if int(s.get("escrow", 0)) > 0:
            return True
        purse = table_purse(room)
        if purse is None or player.is_ai:
            return True
        bet = int(s.get("bet", 0))
        if bet <= 0:
            return "place a bet first — bet <amount>"
        if not purse.debit(player, bet, "slots bet"):
            return (f"spin stakes {bet} coins — your balance is "
                    f"{purse.balance(player)} in the {purse.name}, "
                    "not enough")
        s["escrow"] = bet
        return True

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
                    return [self._place_bet(room, player, amt)]
            return ["bet <amount>"]
        if t == "spin":
            stake = self._ensure_stake(room, player)
            if stake is not True:
                return [stake]
            reels = [self._fair_reel(room, i) for i in range(3)]
            s["reels"] = reels
            s["done"] = True
            out = [f"[ {reels[0]} ] [ {reels[1]} ] [ {reels[2]} ]"]
            if reels[0] == reels[1] == reels[2]:
                sym = reels[0]
                if sym == "💎":
                    mult = 100
                    out.append("💎💎💎 JACKPOT! 💎💎💎")
                    feed(room,
                         f"{player.name} hits the diamond JACKPOT — "
                         f"{s['bet'] * mult} coins!",
                         big=True)
                elif sym == "7️⃣":
                    mult = 10
                elif sym == "🔔":
                    mult = 5
                elif sym == "🍒":
                    mult = 2
                else:
                    mult = 3  # other triples
                payout = s["bet"] * mult
                s["mult"] = mult
                # score/settle read the NET: credit minus the stake
                s["payout"] = payout - s["bet"]
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

    def wager_settle(self, room, player):
        s = room.state
        purse = table_purse(room)
        if purse is None or player.is_ai:
            return None
        if s.get("wager_settled"):
            return (int(s.get("wager_net", 0)), "slots wager")
        escrow = int(s.get("escrow", 0))
        if not s.get("done"):
            # table closed mid-wager: the stake comes back whole
            if escrow > 0:
                purse.credit(player, escrow, "slots bet refund")
            s["wager_settled"] = True
            s["wager_net"] = 0
            return (0, "slots bet refunded — table closed")
        net = int(s.get("payout", 0))
        credit = escrow + net
        if credit > 0:
            purse.credit(player, credit, "slots payout")
        s["wager_settled"] = True
        s["wager_net"] = net
        if net > 0:
            why = (f"slots paid {int(s.get('mult', 0))}x on "
                   f"{int(s.get('bet', 0))}")
        elif escrow > 0 and net == 0:
            why = "slots — bet returned"
        elif escrow > 0:
            why = "slots wager lost"
        else:
            why = "slots — no wager"
        return (net, why)

    def wager_leave(self, room, player):
        s = room.state
        purse = table_purse(room)
        if purse is None or player.is_ai or s.get("done"):
            return None
        escrow = int(s.get("escrow", 0))
        if escrow <= 0:
            return None
        purse.credit(player, escrow, "slots bet refund")
        s["escrow"] = 0
        s["wager_settled"] = True
        s["wager_net"] = 0
        return (f"💰 {player.name}'s {escrow}-coin slots bet "
                "was refunded.")


CASINO_GAMES: tuple[MultiGame, ...] = (
    BlackjackGame(),
    RouletteGame(),
    SlotsGame(),
)
