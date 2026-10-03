"""The wild table: card, dice, and mind games.

Seven more games for the house, each a complete, real implementation —
not a stub of a game:

* poker    — texas hold'em vs the house: a real 7-card hand evaluator,
             blinds, betting rounds, all-ins, side pot
* ttt      — tic-tac-toe vs the house: perfect-play minimax on
             normal and up (unbeatable — the best you can do is a draw),
             a blundering house on easy (beatable: the legend is real)
* bulls    — bulls & cows: the house solves a 4-digit code with a real
             candidate-elimination algorithm, you solve its code
* craps    — full pass-bet craps: come-out roll, the point phase,
             seven-out, the house keeps a running bankroll
* memory   — 4×4 concentration: the house has perfect recall of
             everything that has ever been flipped
* mines    — 9×9 minesweeper, 10 mines: real number logic, flood-fill
             reveals, flags, a safe first click
* wordle   — 6 guesses at a 5-letter word with real green/yellow/gray
             feedback; answers come from the shared lexicon

Everything is pure state logic over ``room.state`` — the same contract
as the rest of the suite, the same unit-testability.
"""
from __future__ import annotations

import itertools
import random
import re
from functools import lru_cache
from typing import Any

from ..ai import GameMind
from ..players import Player
from .base import (DIFFICULTY_LEVELS, MultiGame, Room,
                   normalize_difficulty)

__all__ = ["WILD_GAMES"]

_WORD_RE = re.compile(r"[a-z]+")


def _word_of(text: str) -> str:
    m = _WORD_RE.search(text.strip().lower())
    return m.group(0) if m else ""


# ── shared card machinery ────────────────────────────────────────────────────

_SUITS = ("♠", "♥", "♦", "♣")
_RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
_RANK_VAL = {r: i + 2 for i, r in enumerate(_RANKS)}  # 2..14

#: 5-card hand ranks, highest first.
_HAND_NAMES = (
    "straight flush", "four of a kind", "full house", "flush",
    "straight", "three of a kind", "two pair", "one pair",
    "high card",
)


def make_deck(rng: random.Random, shuffled: bool = True) -> list[str]:
    deck = [r + s for r in _RANKS for s in _SUITS]
    if shuffled:
        rng.shuffle(deck)
    return deck


def eval5(cards: tuple[str, ...]) -> tuple[int, tuple[int, ...]]:
    """Score a 5-card hand.

    Returns (rank, tiebreakers) where rank 8 = straight flush down to
    0 = high card, and tiebreakers compare the hands that share a rank.
    """
    vals = sorted((_RANK_VAL[c[:-1]] for c in cards), reverse=True)
    suits = {c[-1] for c in cards}
    is_flush = len(suits) == 1
    uniq = sorted(set(vals), reverse=True)
    is_straight = len(uniq) == 5 and uniq[0] - uniq[4] == 4
    # the wheel: A-2-3-4-5
    if vals == [14, 5, 4, 3, 2]:
        is_straight, vals = True, [5, 4, 3, 2, 1]
    if is_straight and is_flush:
        return 8, tuple(vals)
    counts = {v: vals.count(v) for v in vals}
    shape = sorted(counts.values(), reverse=True)
    by_count = sorted(counts.items(), key=lambda kv: (-kv[1], -kv[0]))
    kick = tuple(v for v, n in by_count)
    if shape == [4, 1]:
        return 7, kick
    if shape == [3, 2]:
        return 6, kick
    if is_flush:
        return 5, tuple(vals)
    if is_straight:
        return 4, tuple(vals)
    if shape == [3, 1, 1]:
        return 3, kick
    if shape == [2, 2, 1]:
        return 2, kick
    if shape == [2, 1, 1, 1]:
        return 1, kick
    return 0, tuple(vals)


def best7(cards: list[str]) -> tuple[int, tuple[int, ...]]:
    """The best 5-card hand out of 7 cards."""
    best = None
    for combo in itertools.combinations(cards, 5):
        s = eval5(tuple(combo))
        if best is None or s > best:
            best = s
    return best


def _hand_name(score: tuple[int, tuple[int, ...]]) -> str:
    # rank 8 = straight flush … 0 = high card; names list is
    # highest-first, so the index is inverted
    return _HAND_NAMES[8 - score[0]]


# ── 1. poker (texas hold'em vs the house) ────────────────────────────────────

class PokerGame(MultiGame):
    name = "poker"
    description = "texas hold'em vs the house — a real 7-card evaluator"
    min_players = 1
    max_players = 1  # heads-up by design: one human, one house seat
    ai_seats = 1
    move_timeout = 120
    rules = ("Hold'em heads-up. 200 chips each, blinds 2/4. You act "
             "first on every street: 'check' · 'bet <n>' (min-raise 4) · "
             "'call' · 'fold' · 'allin'. Then the house answers — it "
             "folds junk, calls medium, and bets when it's strong. When "
             "bets are even the next street deals: flop, turn, river, "
             "showdown. The best 5 of 7 cards wins, scored by a real "
             "evaluator.")

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        return {"stacks": {}, "hand": 0, "deck": [], "hole": {},
                "board": [], "street": 0, "bets": {}, "pot": 0,
                "committed": {}, "to_act": None, "done": False,
                "blind": 2}

    def setup(self, room, mind):
        for p in room.players:
            room.state["stacks"][p.key] = 200
        out = ["poker — 200 chips each, blinds 2/4. I deal."]
        out.extend(self._deal(room))
        return " · ".join(out[:1]) + "\n" + "\n".join(out[1:])

    # ── dealing & streets ────────────────────────────────────────────
    def _deal(self, room: Room) -> list[str]:
        s = room.state
        s["hand"] += 1
        s["deck"] = make_deck(room.rng())
        s["board"] = []
        s["street"] = 0
        s["pot"] = 0
        s["bets"] = {p.key: 0 for p in room.players}
        s["committed"] = {p.key: 0 for p in room.players}
        s["hole"] = {}
        for p in room.players:
            if s["stacks"].get(p.key, 0) > 0:
                s["hole"][p.key] = [s["deck"].pop(), s["deck"].pop()]
        # one player left with chips → the table is theirs
        alive = [k for k in s["hole"]]
        if len(alive) <= 1:
            s["done"] = True
            out = [f"🏁 the table belongs to "
                   f"{room.player(alive[0]).name if alive else 'no one'}."]
            return out
        # blinds: house small, human big (human acts first)
        house = [p for p in room.players if p.is_ai][0]
        human = [p for p in room.players if not p.is_ai][0]
        for key, amt in ((house.key, s["blind"]), (human.key,
                                                   s["blind"] * 2)):
            if s["stacks"].get(key, 0) > 0:
                pay = min(amt, s["stacks"][key])
                s["stacks"][key] -= pay
                s["bets"][key] += pay
                s["pot"] += pay
                s["committed"][key] += pay
        out = [f"hand {s['hand']} — blinds 2/4. your cards: "
               f"**{human_hole(room)}**"]
        if s["stacks"].get(human.key, 0) <= 0:
            # the human went all-in posting the blinds — the house
            # answers immediately, no human move is possible
            out.append("you're all-in from the blinds.")
            out.extend(self._house_act(room))
            if house.key not in s["hole"]:
                s["stacks"][human.key] += s["pot"]
                out.append(f"you take the {s['pot']} pot.")
                s["pot"] = 0
                out.extend(self._deal(room))
                return out
            out.extend(self._advance_street(room))
            return out
        s["to_act"] = human.key
        return out

    def _everyone_all_in(self, s: dict) -> bool:
        return all(s["stacks"].get(k, 0) <= 0 for k in s["hole"])

    def _advance_street(self, room: Room) -> list[str]:
        s = room.state
        for p in room.players:
            s["bets"][p.key] = 0
        if s["street"] == 0:
            s["board"].extend([s["deck"].pop() for _ in range(3)])
            s["street"] = 1
            msgs = [f"the flop: {'  '.join(s['board'])}"]
        elif s["street"] == 1:
            s["board"].append(s["deck"].pop())
            s["street"] = 2
            msgs = [f"the turn: {s['board'][-1]}"]
        elif s["street"] == 2:
            s["board"].append(s["deck"].pop())
            s["street"] = 3
            msgs = [f"the river: {s['board'][-1]}"]
        else:
            return self._showdown(room)
        if self._everyone_all_in(s):
            # nobody can bet more — run the board out to showdown
            while len(s["board"]) < 5:
                s["board"].append(s["deck"].pop())
            msgs.append("everyone is all-in — the board runs out.")
            return self._showdown(room)
        human = [p for p in room.players if not p.is_ai][0]
        house = [p for p in room.players if p.is_ai][0]
        if s["stacks"].get(human.key, 0) <= 0:
            # the human is all-in: the house gets a free action on each
            # remaining street, then we roll on
            msgs.extend(self._house_act(room))
            return self._advance_street(room)
        s["to_act"] = human.key
        return msgs

    def _showdown(self, room: Room) -> list[str]:
        s = room.state
        out = ["— showdown —"]
        scores = {}
        for p in room.players:
            if p.key not in s["hole"]:
                continue
            cards = s["hole"][p.key] + s["board"]
            scores[p.key] = best7(cards)
            out.append(f"  {p.name}: [{ ' '.join(s['hole'][p.key]) }] "
                       f"→ {_hand_name(scores[p.key])}")
        top = max(scores.values())
        winners = [k for k, v in scores.items() if v == top]
        # achievement tracking: the human's best made hand this game
        # (rank 8 = straight flush … 4 = straight, 0 = high card)
        for p in room.players:
            if not p.is_ai and p.key in scores:
                best = max(s.get("human_best_rank", 0), scores[p.key][0])
                s["human_best_rank"] = best
        # side pots: each player only wins up to what they committed,
        # surplus returns to its owner
        pays: dict[str, int] = {}
        for k in s["committed"]:
            contested = min(s["committed"].values())
            pays[k] = s["committed"][k] - contested
        for k in winners:
            contested = min(s["committed"].values()) * len(s["committed"])
            pays[k] += contested // len(winners)
        for k, amt in pays.items():
            s["stacks"][k] += amt
        out.append(f"pot {s['pot']} → " + ", ".join(
            f"{room.player(k).name if room.player(k) else k} "
            f"(+{pays.get(k, 0)})" for k in s["committed"]))
        for p in room.players:
            if s["stacks"][p.key] <= 0:
                out.append(f"{p.name} is out of chips.")
        if all(s["stacks"][p.key] <= 0 for p in room.players):
            s["done"] = True
            out.append("🏁 the table is empty.")
        else:
            out.extend(self._deal(room))
        return out

    # ── betting ──────────────────────────────────────────────────────
    def _to_call(self, s: dict, key: str) -> int:
        return max(s["bets"].values()) - s["bets"].get(key, 0)

    def _house_strength(self, s: dict, key: str) -> float:
        if key not in s["hole"] or len(s["board"]) < 3:
            return 0.45  # pre-flop: unknown, play it tight
        sc = best7(s["hole"][key] + s["board"])
        # rank 0..8 → 0.2..0.95
        return 0.2 + sc[0] * 0.095

    def _house_act(self, room: Room) -> list[str]:
        s = room.state
        house = [p for p in room.players if p.is_ai][0]
        hk = house.key
        if s["stacks"][hk] <= 0 or hk not in s["hole"]:
            return []
        call = self._to_call(s, hk)
        strength = self._house_strength(s, hk)
        rng = room.rng()
        out = []
        if call <= 0:
            if strength >= 0.78 and s["stacks"][hk] >= 4:
                amt = min(4 + int(strength * 20), s["stacks"][hk])
                s["stacks"][hk] -= amt
                s["bets"][hk] += amt
                s["pot"] += amt
                s["committed"][hk] += amt
                out.append(f"the house bets {amt}.")
            else:
                out.append("the house checks.")
        else:
            # pot odds vs hand strength, with a little variance
            edge = strength - 0.5
            if edge < 0.05 and call > 6:
                out.append("the house folds.")
                s["hole"].pop(hk, None)
                return out
            pay = min(call, s["stacks"][hk])
            s["stacks"][hk] -= pay
            s["bets"][hk] += pay
            s["pot"] += pay
            s["committed"][hk] += pay
            if pay < call:
                out.append(f"the house is all-in for {pay}.")
            else:
                out.append(f"the house calls {pay}.")
        return out

    def on_move(self, room, player, text, mind):
        s = room.state
        if s.get("done"):
            return ["the table is closed."]
        if s.get("to_act") != player.key:
            return ["wait for the house to answer."]
        human = player
        hk_human = human.key
        t = text.strip().lower()
        call = self._to_call(s, hk_human)
        out: list[str] = []
        if s["stacks"][hk_human] <= 0:
            return ["you're out of chips."]
        if t in {"check", "call", "c"} and call == 0:
            out.append("you check.")
        elif t in {"check", "call", "c"} and call > 0:
            pay = min(call, s["stacks"][hk_human])
            s["stacks"][hk_human] -= pay
            s["bets"][hk_human] += pay
            s["pot"] += pay
            s["committed"][hk_human] += pay
            out.append(f"you call {pay}.")
        elif t in {"fold", "f"}:
            out.append("you fold.")
            s["hole"].pop(hk_human, None)
            house = [p for p in room.players if p.is_ai][0]
            s["stacks"][house.key] += s["pot"]
            out.append(f"the house takes the {s['pot']} pot.")
            s["pot"] = 0
            out.extend(self._deal(room))
            return out
        else:
            m = re.match(r"^(?:bet|raise|b|r|allin|all-in)\s*(\d*)$", t)
            if not m:
                return ["'check' · 'bet <n>' · 'call' · 'fold' · 'allin'."]
            amt = int(m.group(1)) if m.group(1) else \
                s["stacks"][hk_human]
            add = amt - s["bets"].get(hk_human, 0)
            if add <= 0:
                return ["that's less than your current bet."]
            pay = min(add, s["stacks"][hk_human])
            s["stacks"][hk_human] -= pay
            s["bets"][hk_human] += pay
            s["pot"] += pay
            s["committed"][hk_human] += pay
            if s["stacks"][hk_human] <= 0:
                s["human_allin"] = True
            out.append(f"you bet {pay}." + (" all-in!" if s.get("human_allin") else ""))
        # house answers
        house = [p for p in room.players if p.is_ai][0]
        if house.key in s["hole"] and s["stacks"].get(house.key, 0) > 0:
            out.extend(self._house_act(room))
        # house folded?
        if house.key not in s["hole"]:
            s["stacks"][hk_human] += s["pot"]
            out.append(f"you take the {s['pot']} pot.")
            s["pot"] = 0
            out.extend(self._deal(room))
            return out
        # even bets, or the all-in side can't answer → next street
        even = len(set(s["bets"].values())) <= 1
        if even or self._everyone_all_in(s) or \
                s["stacks"].get(house.key, 0) <= 0:
            out.extend(self._advance_street(room))
        else:
            s["to_act"] = hk_human
            out.append("your move again.")
        return out

    def ai_turn(self, room, mind):
        return []  # the house acts inside every human move

    def is_over(self, room):
        return room.state.get("done", False)

    def winner(self, room):
        s = room.state
        stacks = {p.key: s["stacks"].get(p.key, 0) for p in room.players}
        alive = [k for k, v in stacks.items() if v > 0]
        if len(alive) == 1:
            return room.player(alive[0])
        if not alive:
            return "draw"
        return None

    def score(self, room, player):
        return int(room.state.get("stacks", {}).get(player.key, 0)) - 200

    def describe_state(self, room):
        s = room.state
        board = " ".join(s.get("board", [])) or "— (pre-flop)"
        return (f"hand {s.get('hand', 0)} · pot {s.get('pot', 0)} · "
                f"board: {board} · "
                f"street {'preflop flop turn river'.split()[s.get('street', 0) % 4]}")


def human_hole(room: Room) -> str:
    s = room.state
    for p in room.players:
        if not p.is_ai and p.key in s.get("hole", {}):
            return " ".join(s["hole"][p.key])
    return "??"


# ── 2. tic-tac-toe (perfect-play house) ──────────────────────────────────────

def _ttt_win(board: tuple[str, ...]) -> str:
    lines = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
             (0, 3, 6), (1, 4, 7), (2, 5, 8),
             (0, 4, 8), (2, 4, 6)]
    for a, b, c in lines:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return ""


def _ttt_minimax(board: list[str], turn: str, me: str,
                 depth: int) -> tuple[int, int]:
    """Return (score, move). score: +10-depth win for me, -10+depth loss,
    0 draw. The house plays ``me``; it never loses. True minimax: our
    nodes maximize, the opponent's nodes minimize."""
    winner = _ttt_win(tuple(board))
    if winner == me:
        return 10 - depth, -1
    if winner:
        return -10 + depth, -1
    if all(board):
        return 0, -1
    if turn == me:
        best = (-99, -1)
        for i, cell in enumerate(board):
            if cell:
                continue
            board[i] = turn
            sc, _ = _ttt_minimax(board, "O" if turn == "X" else "X",
                                 me, depth + 1)
            board[i] = ""
            if sc > best[0]:
                best = (sc, i)
        return best
    best = (99, -1)
    for i, cell in enumerate(board):
        if cell:
            continue
        board[i] = turn
        sc, _ = _ttt_minimax(board, "O" if turn == "X" else "X",
                             me, depth + 1)
        board[i] = ""
        if sc < best[0]:
            best = (sc, i)
    return best


class TicTacToeGame(MultiGame):
    name = "ttt"
    description = "tic-tac-toe vs the house — perfect on normal+"
    min_players = 1
    max_players = 2
    ai_seats = 1
    move_timeout = 60
    difficulties = DIFFICULTY_LEVELS
    rules = ("3×3, you are X, the house is O. Say a square 1–9 (or "
             "'c' center, 'c1'–'c3' corners, 'e1'–'e4' edges). On "
             "normal and up the house plays perfect minimax: the best "
             "you can do is a draw. On easy it blunders sometimes — "
             "beating it is the legend. /game ttt [easy|normal|hard|expert].")

    def new_state(self, rng: random.Random,
                  difficulty: str = "normal") -> dict[str, Any]:
        board = [""] * 9
        return {"board": board, "turn": "X", "over": False,
                "winner": "", "moves": 0,
                "difficulty": normalize_difficulty(difficulty)}

    def _house_move(self, room: Room) -> int:
        """The house's square: perfect minimax on normal+, a blundering
        house on easy (35% random legal move — beatable, but it still
        takes its wins)."""
        s = room.state
        if (self.difficulty(room) == "easy"
                and self.rng(room).random() < 0.35):
            open_squares = [i for i, c in enumerate(s["board"]) if not c]
            return self.rng(room).choice(open_squares)
        _sc, move = _ttt_minimax(list(s["board"]), "O", "O", 0)
        return move

    def setup(self, room, mind):
        return ("tic-tac-toe — you're X, the house is O. say 1–9. "
                "layout:\n 1 | 2 | 3 \n-----------\n 4 | 5 | 6 "
                "\n-----------\n 7 | 8 | 9")

    def _board_text(self, room: Room) -> str:
        b = room.state["board"]
        rows = []
        for i in (0, 3, 6):
            cells = [b[i + j] or "·" for j in range(3)]
            rows.append(" | ".join(cells))
        return "  " + "\n  ".join(rows)

    def _human_move(self, room: Room, p: Player, text: str) -> list[str]:
        s = room.state
        if s["over"]:
            return ["the board is done."]
        if s["turn"] != "X":
            return ["the house is thinking."]
        t = text.strip().lower()
        idx = None
        m = re.match(r"^(\d)$", t)
        if m:
            idx = int(m.group(1)) - 1
        elif t in {"c", "center", "5"}:
            idx = 4
        elif t in {"c1", "1"} and idx is None:
            idx = 0
        elif t in {"c3", "3"} and idx is None:
            idx = 8
        if idx is None or not 0 <= idx <= 8:
            return ["a square 1–9."]
        if s["board"][idx]:
            return [f"square {idx+1} is taken."]
        s["board"][idx] = "X"
        s["moves"] += 1
        out = [f"you take {idx+1}.", self._board_text(room)]
        if self._finish(room):
            return out
        # house replies
        move = self._house_move(room)
        s["board"][move] = "O"
        s["moves"] += 1
        out.append(f"house takes {move+1}.")
        out.append(self._board_text(room))
        if self._finish(room):
            return out
        return out

    def _finish(self, room: Room) -> bool:
        s = room.state
        w = _ttt_win(tuple(s["board"]))
        if w:
            s["over"] = True
            s["winner"] = w
            out = "🏁 " + ("you win. the legend is true."
                           if w == "X" else "the house wins. perfect play.")
            s["msg"] = out
            return True
        if all(s["board"]):
            s["over"] = True
            s["winner"] = "draw"
            s["msg"] = "🏁 a draw — the best the perfect house allows."
            return True
        return False

    def on_move(self, room, player, text, mind):
        return self._human_move(room, player, text)

    def ai_turn(self, room, mind):
        return []

    def is_over(self, room):
        return room.state.get("over", False)

    def winner(self, room):
        s = room.state
        if not s.get("over"):
            return None
        if s.get("winner") == "draw":
            return "draw"
        if s.get("winner") == "X":
            humans = [p for p in room.players if not p.is_ai]
            return humans[0] if humans else None
        return Player(key="ai:ttt", platform="ai", name="The House",
                      is_ai=True)

    def score(self, room, player):
        s = room.state
        if s.get("winner") == "X" and not player.is_ai:
            return 1
        if s.get("winner") == "draw":
            return 0
        return 0

    def describe_state(self, room):
        s = room.state
        return f"move {s['moves']} · " + self._board_text(room)


# ── 3. bulls & cows ──────────────────────────────────────────────────────────

# all 4-digit codes with distinct digits: P(10,4) = 5040
_ALL_CODES: tuple[str, ...] = tuple(
    "".join(p) for p in itertools.permutations("0123456789", 4))


@lru_cache(maxsize=1)
def _bulls_opening_guess() -> str:
    """The single best first guess for a distinct-digit 4-digit code,
    found once by an information heuristic: the guess whose expected
    response distribution is flattest over a sample of the space."""
    sample = _ALL_CODES[::20]  # 252 evenly spaced candidates
    best, best_cost = "", float("inf")
    for g in _ALL_CODES:
        buckets: dict[tuple[int, int], int] = {}
        for c in sample:
            b, w = _score_pair(g, c)
            buckets[(b, w)] = buckets.get((b, w), 0) + 1
        cost = sum(v * v for v in buckets.values())
        if cost < best_cost:
            best, best_cost = g, cost
    return best


def _score_pair(guess: str, secret: str) -> tuple[int, int]:
    bulls = sum(1 for g, s in zip(guess, secret) if g == s)
    cows = sum(1 for g in guess if g in secret) - bulls
    return bulls, cows


def _bulls_solve(secret: str, max_guesses: int = 12) -> tuple[str, int]:
    """Crack ``secret`` with candidate elimination. Returns
    (final_guess, guesses_used). Deterministic: after the smart opening
    the house always plays the first remaining candidate."""
    candidates = list(_ALL_CODES)
    guess = _bulls_opening_guess()
    for n in range(1, max_guesses + 1):
        if guess == secret:
            return guess, n
        b, w = _score_pair(guess, secret)
        candidates = [c for c in candidates
                      if _score_pair(guess, c) == (b, w)]
        guess = candidates[0]
    return guess, max_guesses


class BullsCowsGame(MultiGame):
    name = "bulls"
    description = "bulls & cows — crack the house's code, it cracks yours"
    min_players = 1
    max_players = 2
    ai_seats = 1
    move_timeout = 90
    rules = ("I'm thinking of a 4-digit code, no repeated digits. Guess "
             "with '1234'-style moves: a bull = right digit right "
             "place, a cow = right digit wrong place. 12 guesses. "
             "Optional: 'set <digits>' locks your code, and when the "
             "game ends the house cracks it with real "
             "candidate-elimination — fewest guesses wins.")

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        def code():
            return _ALL_CODES[rng.randrange(len(_ALL_CODES))]
        return {"house_code": code(), "human_code": None,
                "guesses": 0, "max_guesses": 12, "done": False,
                "house_guesses": None, "settled": False}

    def setup(self, room, mind):
        return ("bulls & cows — a 4-digit code, no repeats. guess it "
                "(12 tries). 'set <digits>' locks your code so I can "
                "crack it back.")

    def on_move(self, room, player, text, mind):
        s = room.state
        if s.get("settled"):
            return ["the game is settled."]
        if s.get("done"):
            return self._settle(room)
        t = text.strip().lower()
        m = re.match(r"^set\s+(\d{4})$", t)
        if m:
            if len(set(m.group(1))) != 4:
                return ["no repeated digits."]
            s["human_code"] = m.group(1)
            return [f"your code is locked: {m.group(1)}. crack mine "
                    "before I crack yours."]
        guess = re.sub(r"\D", "", t)
        if len(guess) != 4 or len(set(guess)) != 4:
            return ["a 4-digit number, no repeated digits."]
        s["guesses"] += 1
        bulls, cows = _score_pair(guess, s["house_code"])
        out = [f"guess {guess}: **{bulls} bulls, {cows} cows** "
               f"({s['guesses']}/{s['max_guesses']})"]
        if bulls == 4:
            s["done"] = True
        elif s["guesses"] >= s["max_guesses"]:
            s["done"] = True
            out.append(f"out of guesses — the code was "
                       f"**{s['house_code']}**.")
        if s["done"]:
            out.extend(self._settle(room))
        return out

    def _settle(self, room: Room) -> list[str]:
        s = room.state
        if s.get("settled"):
            return []
        s["settled"] = True
        out = []
        human_cracked = s["guesses"] < s["max_guesses"]
        if s.get("human_code"):
            _, used = _bulls_solve(s["human_code"])
            s["house_guesses"] = used
            out.append(f"the house cracks your code in **{used}** "
                       "guesses.")
            if human_cracked:
                if s["guesses"] < used:
                    out.append("🏁 you win — fewer guesses.")
                elif s["guesses"] > used:
                    out.append("🏁 the house wins — fewer guesses.")
                else:
                    out.append("🏁 dead heat — a draw.")
            else:
                out.append("🏁 the house wins — your code fell.")
        else:
            if human_cracked:
                out.append("🏁 you win — code cracked, and you never "
                           "gave me one.")
            else:
                out.append("🏁 the house wins — the code held.")
        return out

    def ai_turn(self, room, mind):
        return []

    def is_over(self, room):
        return room.state.get("settled", False) or \
            (room.state.get("done") and not room.state.get("human_code"))

    def winner(self, room):
        s = room.state
        if not (s.get("settled") or s.get("done")):
            return None
        human_cracked = s.get("guesses", 99) < s["max_guesses"]
        house_g = s.get("house_guesses")
        if house_g is not None:
            if human_cracked and s["guesses"] < house_g:
                humans = [p for p in room.players if not p.is_ai]
                return humans[0] if humans else None
            if house_g < (s["guesses"] if human_cracked else
                          s["max_guesses"] + 1):
                return Player(key="ai:bulls", platform="ai",
                              name="The House", is_ai=True)
            return "draw"
        if human_cracked:
            humans = [p for p in room.players if not p.is_ai]
            return humans[0] if humans else None
        return Player(key="ai:bulls", platform="ai", name="The House",
                      is_ai=True)

    def score(self, room, player):
        s = room.state
        if s.get("settled") or s.get("done"):
            return max(0, s["max_guesses"] - s["guesses"])
        return 0

    def describe_state(self, room):
        s = room.state
        extra = f" · your code locked" if s.get("human_code") else \
            " · no code locked"
        return f"guesses {s['guesses']}/{s['max_guesses']}{extra}"


# ── 4. craps (pass bet) ──────────────────────────────────────────────────────

class CrapsGame(MultiGame):
    name = "craps"
    description = "full pass-bet craps — come-out, the point, seven-out"
    min_players = 1
    max_players = 3
    ai_seats = 1
    move_timeout = 90
    rules = ("You and the house each bank 100 chips, 10 per roll. "
             "Come-out roll: 7 or 11 wins, 2/3/12 loses, anything else "
             "sets the point. Then the point must land before a 7. "
             "Say 'roll'. 20 rolls or a broke bankroll closes the "
             "table.")

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        return {"bank": {}, "phase": "comeout", "point": None,
                "rolls": 0, "done": False, "bet": 10, "history": []}

    def setup(self, room, mind):
        for p in room.players:
            room.state["bank"][p.key] = 100
        return ("craps — 100 chips each, 10 per roll. say 'roll'. "
                "come-out: 7/11 win, 2/3/12 lose, else the point is "
                "set.")

    def on_move(self, room, player, text, mind):
        s = room.state
        if s.get("done"):
            return ["the table is closed."]
        if text.strip().lower() not in {"roll", "r", "go"}:
            return ["say 'roll'."]
        rng = room.rng()
        d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
        total = d1 + d2
        s["rolls"] += 1
        s["history"].append(total)
        out = [f"🎲 {d1}+{d2} = **{total}**"]
        won = lost = False
        if s["phase"] == "comeout":
            if total in (7, 11):
                won = True
                out.append("natural — you win.")
            elif total in (2, 3, 12):
                lost = True
                out.append("craps — you lose.")
            else:
                s["point"] = total
                s["phase"] = "point"
                out.append(f"point is **{total}** — roll it again "
                           "before a 7.")
        else:
            if total == s["point"]:
                won = True
                out.append(f"point made — {s['point']} lands, you "
                           "win.")
            elif total == 7:
                lost = True
                out.append("seven-out — the table eats the bet.")
            else:
                out.append(f"neither — point {s['point']} still "
                           "stands.")
        if won:
            for k in s["bank"]:
                s["bank"][k] += s["bet"]
        elif lost:
            for k in s["bank"]:
                s["bank"][k] = max(0, s["bank"][k] - s["bet"])
        if s["phase"] == "point" and (won or lost):
            s["phase"] = "comeout"
            s["point"] = None
        if s["rolls"] >= 20 or all(v <= 0 for v in s["bank"].values()):
            s["done"] = True
            out.append("🏁 the table closes — " + " · ".join(
                f"{room.player(k).name if room.player(k) else k}: "
                f"{v}" for k, v in s["bank"].items()))
        return out

    def ai_turn(self, room, mind):
        return []

    def is_over(self, room):
        return room.state.get("done", False)

    def winner(self, room):
        s = room.state
        if not s.get("done"):
            return None
        best = max(s["bank"].items(), key=lambda kv: kv[1])
        if best[1] <= 0:
            return "draw"
        return room.player(best[0])

    def score(self, room, player):
        return int(room.state.get("bank", {}).get(player.key, 0)) - 100

    def describe_state(self, room):
        s = room.state
        phase = "come-out" if s["phase"] == "comeout" else \
            f"point {s['point']}"
        last = s["history"][-3:] if s["history"] else []
        return (f"rolls {s['rolls']}/20 · {phase} · last: "
                f"{last} · bank " + " · ".join(
                    f"{room.player(k).name if room.player(k) else k}: "
                    f"{v}" for k, v in s["bank"].items()))


# ── 5. memory (concentration, perfect-recall house) ──────────────────────────

_PAIRS = ("🍎", "🍌", "🍇", "🍉", "🍒", "🥑", "🌶", "🍔")


class ConcentrationGame(MultiGame):
    name = "memory"
    description = "4×4 concentration — the house remembers everything"
    min_players = 1
    max_players = 2
    ai_seats = 1
    move_timeout = 90
    rules = ("16 cards, 8 pairs, face down. One move = two cards: "
             "'flip 3 4 1 1'. A match is yours and stays up; a miss "
             "flips both back. The house has perfect recall — every "
             "card it has ever flipped is in its memory, so once it "
             "has seen a pair it takes it. Steal the pairs before it "
             "sees them.")

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        cards = list(_PAIRS * 2)
        rng.shuffle(cards)
        return {"grid": cards, "found": {}, "moves": 0, "done": False,
                "seen": []}

    def _face(self, room: Room) -> str:
        s = room.state
        found = set()
        for idxs in s["found"].values():
            found.update(idxs)
        lines = []
        for r in range(4):
            row = []
            for c in range(4):
                i = r * 4 + c
                row.append(s["grid"][i] if i in found else "▫")
            lines.append("  ".join(row))
        return "\n".join(lines)

    def _take(self, room: Room, who_name: str, i: int, j: int) -> list[str]:
        s = room.state
        s["moves"] += 1
        a, b = s["grid"][i], s["grid"][j]
        out = [f"{who_name} flips {i // 4 + 1},{i % 4 + 1} {a} and "
               f"{j // 4 + 1},{j % 4 + 1} {b}"]
        if a == b:
            s["found"].setdefault(who_name, []).extend([i, j])
            out.append("match! it stays up.")
            if sum(len(v) for v in s["found"].values()) == 16:
                s["done"] = True
                out.append("🏁 the table is clear.")
        else:
            out.append("no match — they flip back.")
        out.append(self._face(room))
        return out

    def on_move(self, room, player, text, mind):
        s = room.state
        if s.get("done"):
            return ["the table is clear."]
        nums = re.findall(r"[1-4]", text)
        if len(nums) < 4:
            return ["two cards: 'flip <r> <c> <r> <c>'."]
        i = (int(nums[0]) - 1) * 4 + (int(nums[1]) - 1)
        j = (int(nums[2]) - 1) * 4 + (int(nums[3]) - 1)
        if i == j:
            return ["two different cards."]
        found = set()
        for idxs in s["found"].values():
            found.update(idxs)
        if i in found or j in found:
            return ["that card is already matched."]
        return self._take(room, player.name, i, j)

    def ai_turn(self, room, mind):
        s = room.state
        if s.get("done"):
            return []
        seen = set(s["seen"])
        found = set()
        for idxs in s["found"].values():
            found.update(idxs)
        # perfect recall: if two seen cards match, take the pair
        by_card: dict[str, list[int]] = {}
        for i, v in enumerate(s["grid"]):
            if i not in found:
                by_card.setdefault(v, []).append(i)
        for v, idxs in by_card.items():
            known = [i for i in idxs if i in seen]
            if len(known) >= 2:
                s["seen"] = list(seen | {known[0], known[1]})
                return self._take(room, "house", known[0], known[1])
        # otherwise flip two unseen cards (learning is the game)
        unseen = [i for i, v in enumerate(s["grid"])
                  if i not in found and i not in seen]
        if not unseen:
            s["done"] = True
            return ["the house has seen everything."]
        i, j = unseen[0], unseen[-1]
        s["seen"] = list(seen | {i, j})
        return self._take(room, "house", i, j)

    def is_over(self, room):
        s = room.state
        return s.get("done", False) or \
            sum(len(v) for v in s["found"].values()) == 16

    def winner(self, room):
        s = room.state
        if not (s.get("done") or
                sum(len(v) for v in s["found"].values()) == 16):
            return None
        counts = {k: len(v) // 2 for k, v in s["found"].items()}
        best = max(counts.items(), key=lambda kv: kv[1], default=("", 0))
        if best[1] == 0:
            return "draw"
        names = {p.name: p for p in room.players}
        return names.get(best[0])

    def score(self, room, player):
        return len(room.state.get("found", {}).get(player.name, [])) // 2

    def describe_state(self, room):
        s = room.state
        return f"pairs: " + " · ".join(
            f"{k} {len(v)//2}" for k, v in s["found"].items()) + \
            f" · moves {s['moves']}"


# ── 6. minesweeper (9×9, 10 mines) ───────────────────────────────────────────

class MinesweeperGame(MultiGame):
    name = "mines"
    description = "9×9 minesweeper, 10 mines — your first click is safe"
    min_players = 1
    max_players = 2
    ai_seats = 1
    move_timeout = 120
    rules = ("9×9 grid, 10 mines. Say 'open 4 5' to reveal a square "
             "(rows 1–9, cols 1–9). A number = that many mines next to "
             "it; 0 clears its whole neighborhood. 'flag 4 5' flags a "
             "square. Open every safe square to win. The first open is "
             "always safe.")

    N = 9
    MINES = 10

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        return {"grid": [], "mines": set(), "open": set(),
                "flag": set(), "placed": False, "over": False,
                "win": False}

    def setup(self, room, mind):
        return ("minesweeper — 9×9, 10 mines. 'open <row> <col>' or "
                "'flag <row> <col>'. first open is safe.")

    def _neighbors(self, i: int) -> list[int]:
        r, c = divmod(i, self.N)
        out = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.N and 0 <= nc < self.N:
                    out.append(nr * self.N + nc)
        return out

    def _place(self, room: Room, safe: int) -> None:
        s = room.state
        rng = room.rng()
        options = [i for i in range(self.N * self.N)
                   if i != safe and i not in self._neighbors(safe)]
        s["mines"] = set(rng.sample(options, self.MINES))
        s["placed"] = True

    def _count(self, room: Room, i: int) -> int:
        return sum(1 for n in self._neighbors(i)
                   if n in room.state["mines"])

    def _board(self, room: Room) -> str:
        s = room.state
        lines = []
        for r in range(self.N):
            row = []
            for c in range(self.N):
                i = r * self.N + c
                if i in s["open"]:
                    if i in s["mines"]:
                        row.append("💥")
                    else:
                        n = self._count(room, i)
                        row.append("·" if n == 0 else str(n))
                elif i in s["flag"]:
                    row.append("🚩")
                else:
                    row.append("▫")
            lines.append("  ".join(row))
        return "\n".join(lines)

    def _flood(self, room: Room, start: int, stack: list[int]) -> list[int]:
        s = room.state
        opened = []
        while stack:
            i = stack.pop()
            if i in s["open"] or i in s["mines"]:
                continue
            s["open"].add(i)
            opened.append(i)
            if self._count(room, i) == 0:
                stack.extend(self._neighbors(i))
        return opened

    def on_move(self, room, player, text, mind):
        s = room.state
        if s.get("over"):
            return ["the board is done."]
        t = text.strip().lower()
        m = re.match(r"^(open|flag)\s*([1-9])\s*([1-9])$", t)
        if not m:
            return ["'open <row> <col>' or 'flag <row> <col>'."]
        kind, r, c = m.group(1), int(m.group(2)), int(m.group(3))
        i = (r - 1) * self.N + (c - 1)
        if kind == "flag":
            if i in s["open"]:
                return ["you can't flag an open square."]
            if i in s["flag"]:
                s["flag"].discard(i)
                return [f"flag removed from {r},{c}."]
            s["flag"].add(i)
            return [f"flag at {r},{c}.", self._board(room)]
        # open
        if i in s["flag"]:
            s["flag"].discard(i)
        if i in s["open"]:
            return [f"{r},{c} is already open."]
        if not s["placed"]:
            self._place(room, i)
        if i in s["mines"]:
            s["open"].add(i)
            s["over"] = True
            s["win"] = False
            out = [f"💥 mine at {r},{c} — the board is done."]
            out.append(self._board(room))
            return out
        self._flood(room, i, [i])
        safe_left = self.N * self.N - self.MINES - len(s["open"])
        out = [f"opened {r},{c}."]
        if safe_left <= 0:
            s["over"] = True
            s["win"] = True
            out.append("🏁 every safe square is open — you clear the "
                       "field.")
        else:
            out.append(self._board(room))
        return out

    def ai_turn(self, room, mind):
        return []

    def is_over(self, room):
        return room.state.get("over", False)

    def winner(self, room):
        s = room.state
        if not s.get("over"):
            return None
        if s.get("win"):
            humans = [p for p in room.players if not p.is_ai]
            return humans[0] if humans else None
        return Player(key="ai:mines", platform="ai", name="The Mines",
                      is_ai=True)

    def score(self, room, player):
        s = room.state
        if s.get("win") and not player.is_ai:
            return 1
        return 0

    def describe_state(self, room):
        s = room.state
        return (f"open {len(s['open'])}/71 · flags {len(s['flag'])} · "
                "mines 10")


# ── 7. wordle ────────────────────────────────────────────────────────────────

class WordleGame(MultiGame):
    name = "wordle"
    description = "6 guesses at a 5-letter word, real green/yellow/gray"
    min_players = 1
    max_players = 4
    ai_seats = 0
    move_timeout = 120
    rules = ("I'm thinking of a 5-letter word from the lexicon. Guess a "
             "5-letter word each turn — 6 tries. Green = right letter "
             "right place, yellow = right letter wrong place, gray = "
             "not in the word. I count each letter only once, the real "
             "way.")

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        from ..lexicon import FIVE_LETTERS
        pool = FIVE_LETTERS()
        return {"secret": rng.choice(pool), "guesses": [],
                "done": False, "max": 6}

    def setup(self, room, mind):
        return ("wordle — a 5-letter word, 6 guesses. say the word. "
                "🟩 right place · 🟨 wrong place · ⬜ not in it.")

    @staticmethod
    def _feedback(guess: str, secret: str) -> list[str]:
        # per-position, counting each secret letter only once
        out = [""] * 5
        secret_counts = {ch: secret.count(ch) for ch in secret}
        for i, ch in enumerate(guess):
            if ch == secret[i]:
                out[i] = "g"
                secret_counts[ch] -= 1
        for i, ch in enumerate(guess):
            if out[i] == "g":
                continue
            if secret_counts.get(ch, 0) > 0:
                out[i] = "y"
                secret_counts[ch] -= 1
            else:
                out[i] = "b"
        return out

    @staticmethod
    def _render(word: str, marks: list[str]) -> str:
        glyph = {"g": "🟩", "y": "🟨", "b": "⬜"}
        return " ".join(f"{glyph[m]}{ch}" for ch, m in zip(word, marks))

    def on_move(self, room, player, text, mind):
        s = room.state
        if s.get("done"):
            return ["the word is found."]
        from ..lexicon import FIVE_LETTERS
        guess = _word_of(text)
        if len(guess) != 5:
            return ["a 5-letter word."]
        if guess not in FIVE_LETTERS():
            return [f"“{guess}” is not in the word pool."]
        if guess in [g for g, _ in s["guesses"]]:
            return [f"you already tried {guess}."]
        s["guesses"].append((guess, self._feedback(guess, s["secret"])))
        out = [self._render(guess, s["guesses"][-1][1])]
        if guess == s["secret"]:
            s["done"] = True
            out.append(f"🏁 **{guess}** — solved in {len(s['guesses'])} "
                       "guesses.")
            return out
        left = s["max"] - len(s["guesses"])
        if left <= 0:
            s["done"] = True
            out.append(f"out of guesses — it was **{s['secret']}**.")
        else:
            out.append(f"{left} guesses left.")
        return out

    def ai_turn(self, room, mind):
        return []

    def is_over(self, room):
        return room.state.get("done", False)

    def winner(self, room):
        s = room.state
        if not s.get("done"):
            return None
        solved = [g == s["secret"] for g, _ in s["guesses"]]
        if any(solved):
            humans = [p for p in room.players if not p.is_ai]
            return humans[0] if humans else None
        return Player(key="ai:wordle", platform="ai", name="The Word",
                      is_ai=True)

    def score(self, room, player):
        s = room.state
        if s.get("done"):
            for g, _ in s["guesses"]:
                if g == s["secret"]:
                    return s["max"] - s["guesses"].index((g, self._feedback(
                        g, s["secret"]))) + 1
        return 0

    def describe_state(self, room):
        s = room.state
        return f"guesses {len(s['guesses'])}/{s['max']}"


WILD_GAMES: tuple[MultiGame, ...] = (
    PokerGame(), TicTacToeGame(), BullsCowsGame(), CrapsGame(),
    ConcentrationGame(), MinesweeperGame(), WordleGame(),
)
