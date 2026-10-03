"""Puzzle games: solo and race puzzles that fit chat perfectly.

Three new games, each a complete, real implementation — not a stub:

* sudoku  — 9×9 with a real generator (randomized backtracking +
             symmetric digging with a uniqueness check), four
             difficulties by givens, 3-strike mistakes, hint-scroll
             integration, circled-digit rendering.
* anagram — 6-round unscramble race (DM vs the house, or a group
             free-for-all), words from the shared lexicon, escalating
             word lengths, difficulty-graded house opponent.
* cryptogram — 3-round codebreaker race over a built-in quote bank:
             a real substitution cipher (derangement), letter guesses
             (``a=x``), full solves, and a house opponent that does
             honest frequency analysis.

Everything is pure state logic over ``room.state`` — the same contract
as the rest of the suite, the same unit-testability.
"""
from __future__ import annotations

import random
import re
import string
import time
from typing import Any

from ..ai import GameMind
from ..lexicon import WORDSET
from ..players import Player
from .base import (DIFFICULTY_LEVELS, MultiGame, Room,
                   normalize_difficulty)

__all__ = ["PUZZLE_GAMES", "SudokuGame", "AnagramGame", "CryptogramGame"]


# ── 1. sudoku ─────────────────────────────────────────────────────────────────

_SUDOKU_GIVENS = {"easy": 44, "normal": 36, "hard": 30, "expert": 26}
_SUDOKU_BASE_SCORE = {"easy": 400, "normal": 700, "hard": 1000,
                      "expert": 1400}
#: circled digits mark the player's own entries vs the givens
_CIRCLED = ("", "①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨")


def _sudoku_candidates(grid: list[int], i: int) -> list[int]:
    r, c = divmod(i, 9)
    used = set()
    for k in range(9):
        used.add(grid[r * 9 + k])
        used.add(grid[k * 9 + c])
    br, bc = 3 * (r // 3), 3 * (c // 3)
    for dr in range(3):
        for dc in range(3):
            used.add(grid[(br + dr) * 9 + bc + dc])
    return [d for d in range(1, 10) if d not in used]


def _sudoku_count(grid: list[int], cap: int = 2) -> int:
    """Count solutions up to ``cap`` (MRV backtracking)."""
    best, best_cands = -1, None
    for i in range(81):
        if grid[i]:
            continue
        cands = _sudoku_candidates(grid, i)
        if not cands:
            return 0
        if best_cands is None or len(cands) < len(best_cands):
            best, best_cands = i, cands
            if len(best_cands) == 1:
                break
    if best_cands is None:
        return 1
    total = 0
    for d in best_cands:
        grid[best] = d
        total += _sudoku_count(grid, cap - total)
        grid[best] = 0
        if total >= cap:
            break
    return total


def _sudoku_fill(rng: random.Random) -> list[int]:
    """A complete, randomized valid grid."""
    grid = [0] * 81

    def rec() -> bool:
        best, best_cands = -1, None
        for i in range(81):
            if grid[i]:
                continue
            cands = _sudoku_candidates(grid, i)
            if not cands:
                return False
            if best_cands is None or len(cands) < len(best_cands):
                best, best_cands = i, cands
                if len(best_cands) == 1:
                    break
        if best_cands is None:
            return True
        order = list(best_cands)
        rng.shuffle(order)
        for d in order:
            grid[best] = d
            if rec():
                return True
            grid[best] = 0
        return False

    rec()
    return grid


def _sudoku_generate(rng: random.Random,
                     givens: int) -> tuple[list[int], list[int]]:
    """(solution, puzzle): dig symmetric pairs while the solution stays
    unique."""
    solution = _sudoku_fill(rng)
    puzzle = solution[:]
    cells = list(range(81))
    rng.shuffle(cells)
    for i in cells:
        j = 80 - i
        if puzzle[i] == 0:
            continue
        backup_i, backup_j = puzzle[i], puzzle[j]
        puzzle[i] = 0
        puzzle[j] = 0
        if _sudoku_count(puzzle[:]) != 1:
            puzzle[i], puzzle[j] = backup_i, backup_j
        if sum(1 for v in puzzle if v) <= givens:
            break
    return solution, puzzle


_CELL_RE = re.compile(
    r"^(?:r\s?(?P<r1>[1-9])\s?c\s?(?P<c1>[1-9])|"
    r"(?P<r2>[1-9])\s*[ ,]\s*(?P<c2>[1-9]))"
    r"\s*(?:=|\s)\s*(?P<d>[1-9])$")
_ERASE_RE = re.compile(
    r"^(?:erase|clear|del)\s+"
    r"(?:r\s?(?P<r1>[1-9])\s?c\s?(?P<c1>[1-9])|"
    r"(?P<r2>[1-9])\s*[ ,]\s*(?P<c2>[1-9]))$")


class SudokuGame(MultiGame):
    name = "sudoku"
    description = "9×9 sudoku — real generated puzzles, 3 strikes out"
    min_players = 1
    max_players = 1
    ai_seats = 1
    move_timeout = 0            # a puzzle, not a race — no per-turn clock
    idle_ttl = 4 * 3600.0       # …but the table waits four hours
    difficulties = DIFFICULTY_LEVELS
    rules = ("Fill the 9×9 grid: every row, column, and 3×3 box holds "
             "1–9 exactly once. Say e.g. 'r3c5 7' or '3 5 7' to place a "
             "digit, 'erase r3c5' to clear one of your entries. A wrong "
             "digit is a strike — three strikes and the puzzle beats "
             "you. 'hint' spends a Hint Scroll (see /shop) to reveal a "
             "cell. Pick the puzzle: /game sudoku "
             "[easy|normal|hard|expert].")

    def new_state(self, rng: random.Random,
                  difficulty: str = "normal") -> dict[str, Any]:
        difficulty = normalize_difficulty(difficulty)
        solution, puzzle = _sudoku_generate(
            rng, _SUDOKU_GIVENS[difficulty])
        return {
            "solution": solution, "puzzle": puzzle,
            "board": puzzle[:],
            "givens": [v != 0 for v in puzzle],
            "mistakes": 0, "hints_used": 0, "max_mistakes": 3,
            "difficulty": difficulty, "start": time.time(),
            "over": False, "won": False, "consumed": {},
        }

    # ── rendering ────────────────────────────────────────────────────
    def _render(self, room: Room) -> str:
        s = room.state
        board, givens = s["board"], s["givens"]
        lines = ["    1 2 3   4 5 6   7 8 9",
                 "  ┌───────┬───────┬───────┐"]
        for r in range(9):
            cells = []
            for c in range(9):
                v = board[r * 9 + c]
                if v == 0:
                    cells.append("·")
                elif givens[r * 9 + c]:
                    cells.append(str(v))
                else:
                    cells.append(_CIRCLED[v])
            lines.append(f"{r + 1} │ " + " ".join(cells[0:3]) + " │ "
                         + " ".join(cells[3:6]) + " │ "
                         + " ".join(cells[6:9]) + " │")
            if r in (2, 5):
                lines.append("  ├───────┼───────┼───────┤")
        lines.append("  └───────┴───────┴───────┘")
        lines.append(f"strikes: {s['mistakes']}/{s['max_mistakes']} · "
                     f"hints: {s['hints_used']} · {s['difficulty']}")
        return "\n".join(lines)

    def setup(self, room: Room, mind: GameMind) -> str:
        s = room.state
        n = sum(s["givens"])
        return (f"sudoku ({s['difficulty']}, {n} givens) — fill it in.\n"
                + self._render(room)
                + "\n'r3c5 7' places, 'erase r3c5' clears, 'hint' reveals.")

    # ── moves ────────────────────────────────────────────────────────
    def on_move(self, room: Room, player: Player, text: str,
                mind: GameMind) -> list[str]:
        s = room.state
        if s["over"]:
            return ["this puzzle is done — /game rematch for a fresh one."]
        t = (text or "").strip().lower()

        if t == "hint":
            return self._use_hint(room, player)

        m = _ERASE_RE.match(t)
        if m:
            r = int(m.group("r1") or m.group("r2")) - 1
            c = int(m.group("c1") or m.group("c2")) - 1
            i = r * 9 + c
            if s["givens"][i]:
                return ["that's a given — it stays."]
            if s["board"][i] == 0:
                return ["already empty."]
            s["board"][i] = 0
            return ["cleared.", self._render(room)]

        m = _CELL_RE.match(t)
        if m is None:
            return ["say e.g. 'r3c5 7' or '3 5 7' to place, "
                    "'erase r3c5' to clear."]
        r = int(m.group("r1") or m.group("r2")) - 1
        c = int(m.group("c1") or m.group("c2")) - 1
        d = int(m.group("d"))
        i = r * 9 + c
        if s["givens"][i]:
            return ["that's a given — pick an empty cell."]
        if s["board"][i] != 0:
            return ["taken — 'erase' it first to change it."]
        if s["solution"][i] != d:
            s["mistakes"] += 1
            if s["mistakes"] >= s["max_mistakes"]:
                s["over"] = True
                s["won"] = False
                return [f"❌ wrong — strike {s['mistakes']}/"
                        f"{s['max_mistakes']}. the puzzle beats you.\n"
                        + self._render(room)]
            return [f"❌ wrong — strike {s['mistakes']}/"
                    f"{s['max_mistakes']}.", self._render(room)]
        s["board"][i] = d
        if all(s["board"]):
            s["over"] = True
            s["won"] = True
            return ["🏆 solved!", self._render(room)]
        return ["✓", self._render(room)]

    def _use_hint(self, room: Room, player: Player) -> list[str]:
        s = room.state
        inv = (s.get("inventory") or {}).get(player.key, {})
        if int(inv.get("hint_scroll", 0)) <= 0:
            return ["no Hint Scroll — buy one with '/shop buy "
                    "hint_scroll' (80c)."]
        empties = [i for i in range(81)
                   if s["board"][i] == 0 and not s["givens"][i]]
        if not empties:
            return ["nothing left to reveal."]
        i = self.rng(room).choice(empties)
        s["board"][i] = s["solution"][i]
        s["hints_used"] += 1
        used = s.setdefault("consumed", {}).setdefault(player.key, {})
        used["hint_scroll"] = int(used.get("hint_scroll", 0)) + 1
        # the snapshot is spent: keep it consistent for further hints
        inv["hint_scroll"] = int(inv.get("hint_scroll", 0)) - 1
        out = [f"💡 hint reveals r{i // 9 + 1}c{i % 9 + 1} = "
               f"{s['solution'][i]}.", self._render(room)]
        if all(s["board"]):
            s["over"] = True
            s["won"] = True
            out.append("🏆 solved!")
        return out

    def ai_turn(self, room: Room, mind: GameMind) -> list[str]:
        return []

    # ── outcome ──────────────────────────────────────────────────────
    def is_over(self, room: Room) -> bool:
        return bool(room.state.get("over"))

    def winner(self, room: Room) -> Player | str | None:
        s = room.state
        if not s.get("over"):
            return None
        if s.get("won"):
            humans = [p for p in room.players if not p.is_ai]
            return humans[0] if humans else "draw"
        return Player(key="ai:sudoku", platform="ai", name="The Puzzle",
                      is_ai=True)

    def score(self, room: Room, player: Player) -> int:
        s = room.state
        if not (s.get("won") and not player.is_ai):
            return 0
        base = _SUDOKU_BASE_SCORE.get(s.get("difficulty"), 700)
        return max(50, base - 150 * s["mistakes"]
                   - 100 * s["hints_used"])

    def describe_state(self, room: Room) -> str:
        return self._render(room)


# ── 2. anagram ────────────────────────────────────────────────────────────────

_ANAGRAM_ROUNDS = 6
_ANAGRAM_LENGTHS = (5, 5, 6, 6, 7, 8)
#: house answer probability per difficulty
_ANAGRAM_HOUSE_P = {"easy": 0.25, "normal": 0.45, "hard": 0.65,
                    "expert": 0.80}


def _anagram_words(rng: random.Random) -> list[str]:
    """Six lexicon words, escalating lengths, shuffled per match."""
    pool: dict[int, list[str]] = {}
    for w in WORDSET():
        if 5 <= len(w) <= 8 and w.isalpha() and "'" not in w:
            pool.setdefault(len(w), []).append(w)
    words = []
    for ln in _ANAGRAM_LENGTHS:
        words.append(rng.choice(pool[ln]))
    return words


def _scramble(rng: random.Random, word: str) -> str:
    for _ in range(50):
        letters = list(word)
        rng.shuffle(letters)
        s = "".join(letters)
        if s != word:
            return s
    return word[::-1] if word != word[::-1] else word


class AnagramGame(MultiGame):
    name = "anagram"
    description = "unscramble race — 6 rounds, fastest fingers win"
    min_players = 1
    max_players = 8
    ai_seats = 1
    move_timeout = 45
    difficulties = DIFFICULTY_LEVELS
    rules = ("Six rounds of scrambled words, escalating length. Just say "
             "the word — first correct answer takes the round "
             "(longer words and faster solves score more). In a DM you "
             "race the house; in a group, everyone races. Set the house: "
             "/game anagram [easy|normal|hard|expert].")

    def new_state(self, rng: random.Random,
                  difficulty: str = "normal") -> dict[str, Any]:
        words = _anagram_words(rng)
        return {
            "words": words,
            "scrambles": [_scramble(rng, w) for w in words],
            "round": 0, "rounds": _ANAGRAM_ROUNDS,
            "attempts": 0, "scores": {}, "round_wins": {},
            "difficulty": normalize_difficulty(difficulty),
            "over": False,
        }

    def setup(self, room: Room, mind: GameMind) -> str:
        s = room.state
        names = ", ".join(p.name for p in room.players)
        return (f"anagram — {s['rounds']} rounds, racers: {names}.\n"
                f"round 1/{s['rounds']} ({len(s['words'][0])} letters): "
                f"**{s['scrambles'][0]}**")

    def _round_points(self, room: Room) -> int:
        s = room.state
        word = s["words"][s["round"]]
        speed = max(0, 6 - s["attempts"])
        return len(word) * 10 + speed * 10

    def _next_round(self, room: Room) -> str | None:
        s = room.state
        s["round"] += 1
        s["attempts"] = 0
        if s["round"] >= s["rounds"]:
            s["over"] = True
            return None
        word = s["words"][s["round"]]
        return (f"round {s['round'] + 1}/{s['rounds']} "
                f"({len(word)} letters): **{s['scrambles'][s['round']]}**")

    def _take_round(self, room: Room, player: Player) -> list[str]:
        s = room.state
        word = s["words"][s["round"]]
        pts = self._round_points(room)
        s["scores"][player.key] = int(s["scores"].get(player.key, 0)) + pts
        s["round_wins"][player.key] = \
            int(s["round_wins"].get(player.key, 0)) + 1
        out = [f"✅ {player.name} unscrambles **{word}** +{pts}!"]
        nxt = self._next_round(room)
        if nxt is None:
            return out
        out.append(nxt)
        return out

    def on_move(self, room: Room, player: Player, text: str,
                mind: GameMind) -> list[str]:
        s = room.state
        if s["over"]:
            return ["the match is done — /game rematch for another."]
        guess = (text or "").strip().lower()
        if not guess.isalpha():
            return ["just say the word."]
        word = s["words"][s["round"]]
        if guess == word:
            return self._take_round(room, player)
        s["attempts"] += 1
        return ["nope."]

    def ai_turn(self, room: Room, mind: GameMind) -> list[str]:
        s = room.state
        if s["over"]:
            return []
        house = room.current
        p = _ANAGRAM_HOUSE_P.get(self.difficulty(room), 0.45)
        if self.rng(room).random() < p:
            return self._take_round(room, house)
        # a visible wrong guess — the house is thinking out loud
        word = s["words"][s["round"]]
        decoys = [w for w in WORDSET()
                  if len(w) == len(word) and w != word and w.isalpha()]
        guess = self.rng(room).choice(decoys) if decoys else "…"
        s["attempts"] += 1
        return [f"{house.name} guesses '{guess}' — nope."]

    # ── outcome ──────────────────────────────────────────────────────
    def is_over(self, room: Room) -> bool:
        return bool(room.state.get("over"))

    def _standings(self, room: Room) -> list[tuple[str, int, int]]:
        s = room.state
        rows = [(p.key, int(s["round_wins"].get(p.key, 0)),
                 int(s["scores"].get(p.key, 0)))
                for p in room.players]
        rows.sort(key=lambda r: (r[1], r[2]), reverse=True)
        return rows

    def winner(self, room: Room) -> Player | str | None:
        s = room.state
        if not s.get("over"):
            return None
        rows = self._standings(room)
        if not rows:
            return "draw"
        top = [r for r in rows if (r[1], r[2]) == (rows[0][1], rows[0][2])]
        if len(top) > 1:
            return "draw"
        return room.player(top[0][0])

    def score(self, room: Room, player: Player) -> int:
        s = room.state
        return int(s.get("scores", {}).get(player.key, 0))

    def final_message(self, room: Room, mind: GameMind) -> str:
        rows = self._standings(room)
        table = " · ".join(
            f"{(room.player(k).name if room.player(k) else k)}: "
            f"{w} rounds, {pts} pts" for k, w, pts in rows)
        w = self.winner(room)
        head = "🏁 it's a draw." if w == "draw" \
            else f"🏁 {w.name} takes the anagram match!"
        return f"{head}\n{table}"

    def describe_state(self, room: Room) -> str:
        s = room.state
        if s["over"]:
            return "match over."
        return (f"round {s['round'] + 1}/{s['rounds']}: "
                f"**{s['scrambles'][s['round']]}** "
                f"({len(s['words'][s['round']])} letters)")


# ── 3. cryptogram ─────────────────────────────────────────────────────────────

#: quote bank — famous one-liners, short enough to crack in chat
_QUOTES: tuple[str, ...] = (
    "To be or not to be, that is the question.",
    "I think, therefore I am.",
    "The only thing we have to fear is fear itself.",
    "In the middle of difficulty lies opportunity.",
    "What we think, we become.",
    "The journey of a thousand miles begins with a single step.",
    "That which does not kill us makes us stronger.",
    "Knowledge is power.",
    "Time is money.",
    "Actions speak louder than words.",
    "A picture is worth a thousand words.",
    "The pen is mightier than the sword.",
    "When in Rome, do as the Romans do.",
    "No pain, no gain.",
    "Practice makes perfect.",
    "Better late than never.",
    "Two heads are better than one.",
    "The early bird catches the worm.",
    "A friend in need is a friend indeed.",
    "Honesty is the best policy.",
    "Where there is a will, there is a way.",
    "Fortune favors the bold.",
    "All that glitters is not gold.",
    "The grass is always greener on the other side.",
    "Do not count your chickens before they hatch.",
    "A rolling stone gathers no moss.",
    "Birds of a feather flock together.",
    "Do unto others as you would have them do unto you.",
    "Give a man a fish and you feed him for a day.",
    "Imagination is more important than knowledge.",
    "Life is what happens when you are busy making plans.",
    "The best way out is always through.",
    "Whether you think you can or you think you cannot, you are right.",
    "The only way to do great work is to love what you do.",
    "Simplicity is the ultimate sophistication.",
    "We are what we repeatedly do.",
    "The unexamined life is not worth living.",
    "He who has a why to live can bear almost any how.",
    "It is never too late to be what you might have been.",
    "Well begun is half done.",
    "Diligence is the mother of good luck.",
    "Genius is one percent inspiration and ninety-nine percent perspiration.",
    "If you want to go fast, go alone. If you want to go far, go together.",
    "The best time to plant a tree was twenty years ago.",
    "A smooth sea never made a skilled sailor.",
    "Fall seven times and stand up eight.",
    "The darkest hour is just before the dawn.",
    "Hope is the thing with feathers.",
    "We do not remember days, we remember moments.",
)

#: english letter frequency order — the house's frequency analysis
_FREQ_ORDER = "etaoinshrdlcumwfgypbvkjxqz"

_CIPHER_ROUNDS = 3
#: house behavior per difficulty: (reveal_every_n_turns, solve_at_revealed)
_CIPHER_HOUSE = {"easy": (2, 0.95), "normal": (1, 0.90),
                 "hard": (1, 0.75), "expert": (1, 0.60)}
_LETTER_RE = re.compile(r"^([a-z])\s*=\s*([a-z])$")


def _derangement(rng: random.Random) -> dict[str, str]:
    """A substitution cipher with no fixed points (a never maps to a)."""
    letters = list(string.ascii_lowercase)
    while True:
        shuffled = letters[:]
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(letters, shuffled)):
            return dict(zip(letters, shuffled))


def _encipher(quote: str, cipher: dict[str, str]) -> str:
    return "".join(cipher.get(ch, ch) if ch.isalpha()
                   else ch for ch in quote.lower())


def _norm(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalpha())


class CryptogramGame(MultiGame):
    name = "cryptogram"
    description = "crack the substitution cipher — 3-quote codebreaker race"
    min_players = 1
    max_players = 8
    ai_seats = 1
    move_timeout = 60
    difficulties = DIFFICULTY_LEVELS
    rules = ("A famous quote, enciphered with a substitution code. Guess "
             "letters like 'q=e' (cipher Q is plain E) — correct guesses "
             "reveal every occurrence (+5 each). Say 'solve <your decode>' "
             "to crack the whole quote (+50). 'hint' spends a Hint Scroll "
             "to reveal a letter. 3 quotes; most points wins. The house "
             "does real frequency analysis — set it with /game cryptogram "
             "[easy|normal|hard|expert].")

    def new_state(self, rng: random.Random,
                  difficulty: str = "normal") -> dict[str, Any]:
        quotes = rng.sample(list(_QUOTES), _CIPHER_ROUNDS)
        rounds = []
        for q in quotes:
            cipher = _derangement(rng)
            rounds.append({
                "quote": q, "cipher": cipher,
                "enciphered": _encipher(q, cipher),
                "revealed": {},          # cipher letter -> plain letter
                "wrong": {},             # player key -> wrong guesses
                "letters": {},           # player key -> letters cracked
                "solved_by": "",
            })
        return {
            "rounds": rounds, "round": 0, "rounds_n": _CIPHER_ROUNDS,
            "scores": {}, "difficulty": normalize_difficulty(difficulty),
            "over": False, "consumed": {},
        }

    # ── display ──────────────────────────────────────────────────────
    def _progress(self, rnd: dict[str, Any]) -> str:
        revealed = rnd["revealed"]
        return "".join(
            revealed[ch].upper() if ch.isalpha() and ch in revealed
            else (ch if not ch.isalpha() else "·")
            for ch in rnd["enciphered"])

    def _board(self, room: Room) -> str:
        s = room.state
        rnd = s["rounds"][s["round"]]
        guessed = " ".join(f"{c.upper()}→{p.upper()}"
                           for c, p in sorted(rnd["revealed"].items()))
        lines = [f"quote {s['round'] + 1}/{s['rounds_n']}:",
                 rnd["enciphered"], self._progress(rnd)]
        if guessed:
            lines.append(f"cracked: {guessed}")
        lines.append("guess like 'q=e' · 'solve <decode>' · 'hint'")
        return "\n".join(lines)

    def setup(self, room: Room, mind: GameMind) -> str:
        names = ", ".join(p.name for p in room.players)
        return (f"cryptogram — {_CIPHER_ROUNDS} quotes, codebreakers: "
                f"{names}.\n" + self._board(room))

    # ── rounds ───────────────────────────────────────────────────────
    def _next_round(self, room: Room) -> str | None:
        s = room.state
        s["round"] += 1
        if s["round"] >= s["rounds_n"]:
            s["over"] = True
            return None
        return self._board(room)

    def _take_solve(self, room: Room, player: Player) -> list[str]:
        s = room.state
        rnd = s["rounds"][s["round"]]
        s["scores"][player.key] = int(s["scores"].get(player.key, 0)) + 50
        rnd["solved_by"] = player.key
        out = [f"🔓 {player.name} cracks it: \"{rnd['quote']}\" +50!"]
        nxt = self._next_round(room)
        if nxt is not None:
            out.append(nxt)
        return out

    def _check_fully_revealed(self, room: Room) -> list[str]:
        """All letters cracked with no solve: the round ends, top
        codebreaker takes the round points."""
        s = room.state
        rnd = s["rounds"][s["round"]]
        need = {ch for ch in rnd["enciphered"] if ch.isalpha()}
        if not need.issubset(rnd["revealed"]):
            return []
        letters = rnd["letters"]
        top = max(letters.items(), key=lambda kv: kv[1], default=("", 0))
        winner_p = room.player(top[0]) if top[0] else None
        if winner_p is not None:
            s["scores"][winner_p.key] = \
                int(s["scores"].get(winner_p.key, 0)) + 25
        out = [f"every letter is cracked — the quote was "
               f"\"{rnd['quote']}\"."
               + (f" {winner_p.name} takes the round +25."
                  if winner_p else "")]
        nxt = self._next_round(room)
        if nxt is not None:
            out.append(nxt)
        return out

    # ── moves ────────────────────────────────────────────────────────
    def on_move(self, room: Room, player: Player, text: str,
                mind: GameMind) -> list[str]:
        s = room.state
        if s["over"]:
            return ["the match is done — /game rematch for another."]
        t = (text or "").strip().lower()
        rnd = s["rounds"][s["round"]]

        if t == "hint":
            return self._use_hint(room, player)

        if t.startswith("solve "):
            attempt = t[6:].strip()
            if _norm(attempt) == _norm(rnd["quote"]):
                return self._take_solve(room, player)
            rnd["wrong"][player.key] = \
                int(rnd["wrong"].get(player.key, 0)) + 1
            return ["not quite — keep cracking."]

        m = _LETTER_RE.match(t)
        if m is None:
            return ["guess like 'q=e', or 'solve <your decode>'."]
        cipher_ch, plain_ch = m.group(1), m.group(2)
        if cipher_ch in rnd["revealed"]:
            return [f"{cipher_ch.upper()} is already cracked."]
        truth = {v: k for k, v in rnd["cipher"].items()}
        if truth.get(cipher_ch) == plain_ch:
            rnd["revealed"][cipher_ch] = plain_ch
            s["scores"][player.key] = \
                int(s["scores"].get(player.key, 0)) + 5
            rnd["letters"][player.key] = \
                int(rnd["letters"].get(player.key, 0)) + 1
            out = [f"✅ {cipher_ch.upper()} → {plain_ch.upper()} "
                   f"(+5).", self._board(room)]
            out.extend(self._check_fully_revealed(room))
            return out
        rnd["wrong"][player.key] = int(rnd["wrong"].get(player.key, 0)) + 1
        return [f"❌ {cipher_ch.upper()} is not {plain_ch.upper()}."]

    def _use_hint(self, room: Room, player: Player) -> list[str]:
        s = room.state
        rnd = s["rounds"][s["round"]]
        inv = (s.get("inventory") or {}).get(player.key, {})
        if int(inv.get("hint_scroll", 0)) <= 0:
            return ["no Hint Scroll — buy one with '/shop buy "
                    "hint_scroll' (80c)."]
        need = [ch for ch in {c for c in rnd["enciphered"] if c.isalpha()}
                if ch not in rnd["revealed"]]
        if not need:
            return ["nothing left to reveal."]
        truth = {v: k for k, v in rnd["cipher"].items()}
        pick = self.rng(room).choice(sorted(need))
        rnd["revealed"][pick] = truth[pick]
        rnd["letters"][player.key] = \
            int(rnd["letters"].get(player.key, 0)) + 1
        used = s.setdefault("consumed", {}).setdefault(player.key, {})
        used["hint_scroll"] = int(used.get("hint_scroll", 0)) + 1
        inv["hint_scroll"] = int(inv.get("hint_scroll", 0)) - 1
        out = [f"💡 hint: {pick.upper()} → {truth[pick].upper()}.",
               self._board(room)]
        out.extend(self._check_fully_revealed(room))
        return out

    # ── the house: honest frequency analysis ─────────────────────────
    def _house_play(self, room: Room) -> list[str]:
        """One house turn: frequency-analysis letter guesses, and a
        full solve once enough of the quote is cracked. Wrong guesses
        are remembered (``tried``) so the house works through the
        candidate list like a human instead of repeating itself.
        Returns the messages for the turn."""
        s = room.state
        rnd = s["rounds"][s["round"]]
        house = room.current
        every, solve_at = _CIPHER_HOUSE.get(self.difficulty(room),
                                            _CIPHER_HOUSE["normal"])
        s["house_ticks"] = int(s.get("house_ticks", 0)) + 1
        # enough cracked to read the quote? go for the full solve
        need = {ch for ch in rnd["enciphered"] if ch.isalpha()}
        revealed_frac = (len(rnd["revealed"]) / len(need)) if need else 1.0
        if revealed_frac >= solve_at:
            return self._take_solve(room, house)
        if s["house_ticks"] % every != 0:
            return [f"{house.name} studies the cipher…"]
        # candidate pairs: most-frequent-first cipher letters ×
        # english-frequency-order plain letters, skipping pairs already
        # tried — reveal only when the guess is right
        from collections import Counter
        freq = Counter(ch for ch in rnd["enciphered"] if ch.isalpha())
        used_plain = set(rnd["revealed"].values())
        tried = set(rnd.setdefault("tried", []))
        truth = {v: k for k, v in rnd["cipher"].items()}
        order = [ch for ch, _n in freq.most_common()
                 if ch not in rnd["revealed"]]
        plains = [p for p in _FREQ_ORDER if p not in used_plain]
        for ch in order:
            for plain in plains:
                if f"{ch}{plain}" in tried:
                    continue
                tried.add(f"{ch}{plain}")
                rnd["tried"] = sorted(tried)
                if truth.get(ch) == plain:
                    rnd["revealed"][ch] = plain
                    s["scores"][house.key] = \
                        int(s["scores"].get(house.key, 0)) + 5
                    rnd["letters"][house.key] = \
                        int(rnd["letters"].get(house.key, 0)) + 1
                    out = [f"{house.name} cracks {ch.upper()} → "
                           f"{plain.upper()} (+5).", self._board(room)]
                    out.extend(self._check_fully_revealed(room))
                    return out
                # wrong guess — recorded, the house moves on next turn
                return [f"{house.name} tries {ch.upper()} → "
                        f"{plain.upper()} — no.", self._board(room)]
        return [f"{house.name} studies the cipher…"]

    def ai_turn(self, room: Room, mind: GameMind) -> list[str]:
        if room.state.get("over"):
            return []
        return self._house_play(room)

    # ── outcome ──────────────────────────────────────────────────────
    def is_over(self, room: Room) -> bool:
        return bool(room.state.get("over"))

    def winner(self, room: Room) -> Player | str | None:
        s = room.state
        if not s.get("over"):
            return None
        scores = s.get("scores", {})
        if not scores:
            return "draw"
        top = max(scores.values())
        leaders = [k for k, v in scores.items() if v == top]
        if len(leaders) > 1:
            return "draw"
        return room.player(leaders[0])

    def score(self, room: Room, player: Player) -> int:
        return int(room.state.get("scores", {}).get(player.key, 0))

    def final_message(self, room: Room, mind: GameMind) -> str:
        s = room.state
        table = " · ".join(
            f"{(room.player(k).name if room.player(k) else k)}: {v} pts"
            for k, v in sorted(s.get("scores", {}).items(),
                               key=lambda kv: -kv[1]))
        w = self.winner(room)
        head = "🏁 it's a draw." if w == "draw" \
            else f"🏁 {w.name} is the master codebreaker!"
        return f"{head}\n{table}"

    def describe_state(self, room: Room) -> str:
        if room.state.get("over"):
            return "match over."
        return self._board(room)


PUZZLE_GAMES: tuple[MultiGame, ...] = (
    SudokuGame(),
    AnagramGame(),
    CryptogramGame(),
)
