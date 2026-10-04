"""Inbox games: turn-based, async-friendly, one message per move.

These games are built for the inbox, not a live session: every turn is a
single chat command, there is no per-turn clock (``move_timeout = 0``),
and each room survives ``idle_ttl`` (7 days) of silence between moves, so
a duel can stretch across days of normal chatting. They play through the
same engine as everything else — DM vs the house, group tables, and
DM-to-DM relay duels all work — and every rule below is real, not a stub:

* gomoku   — 15×15 five-in-a-row vs the house (win/block/pressure AI)
* reversi  — full othello rules: legal-move flipping, passes, disc count
* checkers — english draughts: forced captures, multi-jump chains, kings

Humans share the first side (black), the house plays the second — the
same contract as ttt/bulls, so group and relay tables just work.
"""
from __future__ import annotations

import random
import re
from typing import Any

from ..ai import GameMind, reversi_move
from ..players import Player
from .base import (DIFFICULTY_LEVELS, MultiGame, Room,
                   normalize_difficulty)

__all__ = ["INBOX_GAMES"]

#: async rooms live this long with no human move before the table closes
INBOX_IDLE_TTL = 7 * 86400.0

_HOUSE = Player(key="ai:house", platform="ai", name="The House", is_ai=True)


def _human(room: Room) -> Player | None:
    for p in room.players:
        if not p.is_ai:
            return p
    return None


# ── 1. gomoku ───────────────────────────────────────────────────────────────

_GOMOKU_SIZE = 15
_GOMOKU_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))
_GOMOKU_CELL = re.compile(r"^([a-o])(1[0-5]|[1-9])$", re.I)


def _gomoku_parse(text: str) -> tuple[int, int] | None:
    m = _GOMOKU_CELL.match((text or "").strip())
    if not m:
        return None
    return int(m.group(2)) - 1, ord(m.group(1).lower()) - ord("a")


def _gomoku_run(grid: list[list[str]], r: int, c: int,
                side: str) -> int:
    """Longest line through (r, c) if it held ``side``."""
    best = 1
    for dr, dc in _GOMOKU_DIRS:
        n = 1
        for sgn in (1, -1):
            rr, cc = r + sgn * dr, c + sgn * dc
            while (0 <= rr < _GOMOKU_SIZE and 0 <= cc < _GOMOKU_SIZE
                   and grid[rr][cc] == side):
                n += 1
                rr += sgn * dr
                cc += sgn * dc
        best = max(best, n)
    return best


def _gomoku_wins(grid: list[list[str]], r: int, c: int, side: str) -> bool:
    grid[r][c] = side
    try:
        return _gomoku_run(grid, r, c, side) >= 5
    finally:
        grid[r][c] = ""


def _gomoku_cell_value(grid: list[list[str]], r: int, c: int,
                       me: str) -> float:
    """Threat value of ``me`` playing the empty cell (r, c): longest
    run through the cell for me (×10) and for the foe (×9), plus a
    centre bias. The grid is restored before returning."""
    foe = "B" if me == "W" else "W"
    grid[r][c] = me
    mine = _gomoku_run(grid, r, c, me)
    grid[r][c] = foe
    theirs = _gomoku_run(grid, r, c, foe)
    grid[r][c] = ""
    centre = 7 - (abs(r - 7) + abs(c - 7)) / 2.0
    return mine * 10.0 + theirs * 9.0 + centre


def _gomoku_house_move(grid: list[list[str]],
                       rng: random.Random,
                       difficulty: str = "normal"
                       ) -> tuple[int, int] | None:
    """The house's move, graded by difficulty.

    easy: usually sound, but ~35% of the time it genuinely misses the
    board — it won't take its win or block yours, and just plays near
    the action.
    normal: takes its wins, blocks yours, otherwise plays the most
    pressuring empty cell (longest threat through it, centre-biased).
    hard: the normal brain plus one-ply lookahead — each candidate is
    scored minus your best reply, so it spots double threats and
    doesn't walk into forks.
    expert: two-ply on a shortlist — your best reply is itself scored
    minus the house's best counter, so it sets traps a move deeper.
    Deterministic given the room's rng."""
    empties = [(r, c) for r in range(_GOMOKU_SIZE)
               for c in range(_GOMOKU_SIZE) if not grid[r][c]]
    if not empties:
        return None
    blunder = difficulty == "easy" and rng.random() < 0.35
    if not blunder:
        for r, c in empties:
            if _gomoku_wins(grid, r, c, "W"):
                return r, c
        for r, c in empties:
            if _gomoku_wins(grid, r, c, "B"):
                return r, c
    stones = [(r, c) for r in range(_GOMOKU_SIZE)
              for c in range(_GOMOKU_SIZE) if grid[r][c]]
    if not stones:
        return 7, 7
    # only cells near the action matter
    near: set[tuple[int, int]] = set()
    for r, c in stones:
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                rr, cc = r + dr, c + dc
                if (0 <= rr < _GOMOKU_SIZE and 0 <= cc < _GOMOKU_SIZE
                        and not grid[rr][cc]):
                    near.add((rr, cc))
    if blunder:
        # the miss: a plausible-looking but tactically blind move
        return rng.choice(sorted(near)) if near else rng.choice(empties)

    def reply_best(g: list[list[str]], cells: set[tuple[int, int]],
                   side: str) -> float:
        return max((_gomoku_cell_value(g, rr, cc, side)
                    for rr, cc in cells if not g[rr][cc]),
                   default=0.0)

    if difficulty in ("hard", "expert"):
        raw: list[tuple[float, int, int]] = []
        for r, c in near:
            val = _gomoku_cell_value(grid, r, c, "W")
            grid[r][c] = "W"
            one_ply = reply_best(grid, near, "B")
            grid[r][c] = ""
            raw.append((val, one_ply, r, c))
        if difficulty == "expert":
            # two-ply: re-score the shortlist, treating your best
            # reply as itself weakened by the house's best counter
            raw.sort(key=lambda t: t[0] - t[1], reverse=True)
            rescored: list[tuple[float, int, int]] = []
            for val, _one_ply, r, c in raw[:8]:
                grid[r][c] = "W"
                best_reply = float("-inf")
                for rr, cc in near:
                    if grid[rr][cc]:
                        continue
                    rv = _gomoku_cell_value(grid, rr, cc, "B")
                    grid[rr][cc] = "B"
                    rv -= reply_best(grid, near, "W")
                    grid[rr][cc] = ""
                    if rv > best_reply:
                        best_reply = rv
                grid[r][c] = ""
                if best_reply == float("-inf"):
                    best_reply = 0.0
                rescored.append((val - best_reply, r, c))
            scored = rescored
        else:
            scored = [(val - one_ply, r, c) for val, one_ply, r, c in raw]
    else:
        scored = [(_gomoku_cell_value(grid, r, c, "W"), r, c)
                  for r, c in near]
    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[0][0]
    tied = [(r, c) for s, r, c in scored if s == top]
    return rng.choice(tied)


class GomokuGame(MultiGame):
    name = "gomoku"
    description = "five-in-a-row on 15×15 — async duel vs the house"
    min_players = 1
    max_players = 2
    ai_seats = 1
    move_timeout = 0          # inbox game: no per-turn clock
    idle_ttl = INBOX_IDLE_TTL  # …the table waits up to a week
    difficulties = DIFFICULTY_LEVELS
    rules = ("15×15 board. You're ● (black, first), the house is ○. "
             "Say a square like h8 (columns a–o, rows 1–15) — one move "
             "per message, whenever you like. Five in a row, any "
             "direction, wins. No clock: the table stays open a week "
             "between moves. The house's brain scales: easy blunders "
             "its wins sometimes, normal plays sound threats, hard "
             "reads one move ahead, expert two. "
             "/game gomoku [easy|normal|hard|expert].")

    def new_state(self, rng: random.Random,
                  difficulty: str = "normal") -> dict[str, Any]:
        return {"grid": [[""] * _GOMOKU_SIZE for _ in range(_GOMOKU_SIZE)],
                "moves": 0, "over": False, "winner": "",
                "difficulty": normalize_difficulty(difficulty)}

    def setup(self, room: Room, mind: GameMind) -> str:
        return ("gomoku — five in a row wins. you're ●, the house is ○. "
                "say a square like h8 (a–o, 1–15). no clock — play "
                "whenever. set the house's strength: /game gomoku "
                "[easy|normal|hard|expert].")

    def _board(self, room: Room) -> str:
        g = room.state["grid"]
        head = "   " + " ".join(chr(ord("a") + c)
                                for c in range(_GOMOKU_SIZE))
        rows = [head]
        for r in range(_GOMOKU_SIZE):
            cells = [g[r][c] if g[r][c] else "·" for c in range(_GOMOKU_SIZE)]
            rows.append(f"{r + 1:>2} " + " ".join(cells))
        return "\n".join(rows)

    def _place(self, room: Room, r: int, c: int, side: str) -> bool:
        """Place a stone; True when it ends the game."""
        s = room.state
        s["grid"][r][c] = side
        s["moves"] += 1
        if _gomoku_run(s["grid"], r, c, side) >= 5:
            s["over"] = True
            s["winner"] = side
            return True
        if s["moves"] >= _GOMOKU_SIZE * _GOMOKU_SIZE:
            s["over"] = True
            s["winner"] = "draw"
            return True
        return False

    def on_move(self, room: Room, player: Player, text: str,
                mind: GameMind) -> list[str]:
        s = room.state
        if s["over"]:
            return ["the board is done — /game rematch for another."]
        sq = _gomoku_parse(text)
        if sq is None:
            return ["a square like h8 (columns a–o, rows 1–15)."]
        r, c = sq
        if s["grid"][r][c]:
            return [f"{text.strip().lower()} is taken."]
        out = [f"● {text.strip().lower()}."]
        if self._place(room, r, c, "B"):
            out.append(self._board(room))
            return out
        hm = _gomoku_house_move(s["grid"], self.rng(room),
                                self.difficulty(room))
        if hm is not None:
            hr, hc = hm
            out.append(f"○ {chr(ord('a') + hc)}{hr + 1}.")
            self._place(room, hr, hc, "W")
        out.append(self._board(room))
        return out

    def ai_turn(self, room: Room, mind: GameMind) -> list[str]:
        return []

    def is_over(self, room: Room) -> bool:
        return bool(room.state.get("over"))

    def winner(self, room: Room) -> Player | str | None:
        s = room.state
        if not s.get("over"):
            return None
        w = s.get("winner")
        if w == "draw":
            return "draw"
        if w == "B":
            return _human(room)
        return _HOUSE

    def score(self, room: Room, player: Player) -> int:
        s = room.state
        if (s.get("winner") == "B" and not player.is_ai
                and s.get("moves")):
            return max(1, _GOMOKU_SIZE * _GOMOKU_SIZE - s["moves"])
        return 0

    def final_message(self, room: Room, mind: GameMind) -> str:
        w = room.state.get("winner")
        if w == "B":
            h = _human(room)
            return f"🏁 {(h.name if h else 'you')} take five — gomoku won."
        if w == "W":
            return "🏁 the house makes five — better luck next board."
        return "🏁 a full board, no five — a draw."

    def describe_state(self, room: Room) -> str:
        return f"move {room.state['moves']} · you ●, house ○\n" + self._board(room)


# ── 2. reversi ──────────────────────────────────────────────────────────────

_REVERSI_SIZE = 8
_REVERSI_DIRS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                 (0, 1), (1, -1), (1, 0), (1, 1))
_REVERSI_CELL = re.compile(r"^([a-h])([1-8])$", re.I)


def _reversi_flips(grid: list[list[str]], r: int, c: int,
                   side: str) -> list[tuple[int, int]]:
    """Every enemy disc this placement would flip (empty = illegal)."""
    if grid[r][c]:
        return []
    foe = "W" if side == "B" else "B"
    out: list[tuple[int, int]] = []
    for dr, dc in _REVERSI_DIRS:
        line: list[tuple[int, int]] = []
        rr, cc = r + dr, c + dc
        while (0 <= rr < _REVERSI_SIZE and 0 <= cc < _REVERSI_SIZE
               and grid[rr][cc] == foe):
            line.append((rr, cc))
            rr += dr
            cc += dc
        if line and 0 <= rr < _REVERSI_SIZE and 0 <= cc < _REVERSI_SIZE \
                and grid[rr][cc] == side:
            out.extend(line)
    return out


def _reversi_legal(grid: list[list[str]],
                   side: str) -> list[tuple[int, int, int]]:
    """(r, c, flips) for every legal move, most flips first."""
    moves = []
    for r in range(_REVERSI_SIZE):
        for c in range(_REVERSI_SIZE):
            flips = _reversi_flips(grid, r, c, side)
            if flips:
                moves.append((r, c, len(flips)))
    moves.sort(key=lambda m: -m[2])
    return moves


def _reversi_count(grid: list[list[str]]) -> tuple[int, int]:
    b = sum(cell == "B" for row in grid for cell in row)
    w = sum(cell == "W" for row in grid for cell in row)
    return b, w


def _reversi_sq(r: int, c: int) -> str:
    return f"{chr(ord('a') + c)}{r + 1}"


class ReversiGame(MultiGame):
    name = "reversi"
    description = "othello — outflank the house, own the board"
    min_players = 1
    max_players = 2
    ai_seats = 1
    move_timeout = 0
    idle_ttl = INBOX_IDLE_TTL
    difficulties = DIFFICULTY_LEVELS
    rules = ("8×8 othello. You're ● (black, first), the house is ○. Say a "
             "square like d3 — it must outflank at least one white disc in "
             "a straight line (every line you close flips to your colour). "
             "No legal move? You pass automatically. Most discs when nobody "
             "can move wins. No clock — the table waits a week. Set the "
             "house's strength: /game reversi [easy|normal|hard|expert].")

    def new_state(self, rng: random.Random,
                  difficulty: str = "normal") -> dict[str, Any]:
        grid = [[""] * _REVERSI_SIZE for _ in range(_REVERSI_SIZE)]
        grid[3][3] = "W"
        grid[4][4] = "W"
        grid[3][4] = "B"
        grid[4][3] = "B"
        return {"grid": grid, "moves": 0, "over": False, "winner": "",
                "difficulty": normalize_difficulty(difficulty)}

    @staticmethod
    def _apply_grid(grid: list[list[str]], r: int, c: int,
                    side: str) -> None:
        """Apply a placement to a bare grid (used by the AI's lookahead)."""
        flips = _reversi_flips(grid, r, c, side)
        grid[r][c] = side
        for fr, fc in flips:
            grid[fr][fc] = side

    def setup(self, room: Room, mind: GameMind) -> str:
        return ("reversi — you're ● (black, first). say a square like d3 "
                "that flips at least one ○. no clock — play whenever.")

    def _board(self, room: Room) -> str:
        g = room.state["grid"]
        rows = ["   " + " ".join("abcdefgh")]
        for r in range(_REVERSI_SIZE):
            cells = [g[r][c] if g[r][c] else "·"
                     for c in range(_REVERSI_SIZE)]
            rows.append(f"{r + 1}  " + " ".join(cells))
        b, w = _reversi_count(g)
        rows.append(f"● {b} — ○ {w}")
        return "\n".join(rows)

    def _apply(self, room: Room, r: int, c: int, side: str) -> int:
        s = room.state
        flips = _reversi_flips(s["grid"], r, c, side)
        s["grid"][r][c] = side
        for fr, fc in flips:
            s["grid"][fr][fc] = side
        s["moves"] += 1
        return len(flips)

    def _close(self, room: Room) -> bool:
        """True when the game is over (nobody can move); sets winner."""
        s = room.state
        if _reversi_legal(s["grid"], "B") or _reversi_legal(s["grid"], "W"):
            return False
        s["over"] = True
        b, w = _reversi_count(s["grid"])
        s["winner"] = "B" if b > w else ("W" if w > b else "draw")
        s["final"] = (b, w)
        return True

    def on_move(self, room: Room, player: Player, text: str,
                mind: GameMind) -> list[str]:
        s = room.state
        if s["over"]:
            return ["the board is done — /game rematch for another."]
        t = (text or "").strip().lower()
        m = _REVERSI_CELL.match(t)
        if m is None:
            return ["a square like d3."]
        r, c = int(m.group(2)) - 1, ord(m.group(1)) - ord("a")
        flips = _reversi_flips(s["grid"], r, c, "B")
        if not flips:
            legal = [_reversi_sq(rr, cc)
                     for rr, cc, _ in _reversi_legal(s["grid"], "B")[:8]]
            hint = (" legal: " + ", ".join(legal)) if legal else ""
            return [f"{t} flips nothing — illegal.{hint}"]
        n = self._apply(room, r, c, "B")
        out = [f"● {t} flips {n}."]
        if self._close(room):
            out.append(self._board(room))
            return out
        # the house replies, handling passes on both sides
        for _ in range(64):
            w_moves = _reversi_legal(s["grid"], "W")
            b_moves = _reversi_legal(s["grid"], "B")
            if not w_moves and not b_moves:
                self._close(room)
                break
            if w_moves:
                hm = reversi_move(s["grid"], "W",
                                  difficulty=self.difficulty(room),
                                  rng=self.rng(room),
                                  legal_fn=_reversi_legal,
                                  apply_fn=self._apply_grid)
                if hm is None:
                    out.append("house passes — your move.")
                    break
                hr, hc = hm
                n = self._apply(room, hr, hc, "W")
                out.append(f"○ {_reversi_sq(hr, hc)} flips {n}.")
                if self._close(room):
                    break
                if _reversi_legal(s["grid"], "B"):
                    break
                out.append("you have no legal move — house goes again.")
                continue
            out.append("house passes — your move.")
            break
        out.append(self._board(room))
        return out

    def ai_turn(self, room: Room, mind: GameMind) -> list[str]:
        return []

    def is_over(self, room: Room) -> bool:
        return bool(room.state.get("over"))

    def winner(self, room: Room) -> Player | str | None:
        s = room.state
        if not s.get("over"):
            return None
        w = s.get("winner")
        if w == "draw":
            return "draw"
        if w == "B":
            return _human(room)
        return _HOUSE

    def score(self, room: Room, player: Player) -> int:
        s = room.state
        if (s.get("over") and s.get("winner") == "B"
                and not player.is_ai):
            b, w = s.get("final", (0, 0))
            return max(1, b - w)
        return 0

    def final_message(self, room: Room, mind: GameMind) -> str:
        s = room.state
        b, w = s.get("final", _reversi_count(s["grid"]))
        if s.get("winner") == "B":
            h = _human(room)
            return (f"🏁 {(h.name if h else 'you')} win reversi "
                    f"● {b} — ○ {w}.")
        if s.get("winner") == "W":
            return f"🏁 the house wins reversi ○ {w} — ● {b}."
        return f"🏁 reversi drawn — ● {b}, ○ {w}."

    def describe_state(self, room: Room) -> str:
        s = room.state
        legal = [_reversi_sq(r, c)
                 for r, c, _ in _reversi_legal(s["grid"], "B")[:10]]
        head = f"move {s['moves']} · you ●\n"
        if legal and not s.get("over"):
            head += "your moves: " + ", ".join(legal) + "\n"
        return head + self._board(room)


# ── 3. checkers ─────────────────────────────────────────────────────────────

_CHECKERS_SIZE = 8
_CK_DIAG = ((1, -1), (1, 1), (-1, -1), (-1, 1))
_CK_SQ = re.compile(r"^([a-h])([1-8])$", re.I)


def _ck_side(piece: str) -> str:
    return "B" if piece.lower() == "b" else "W"


def _ck_dirs(piece: str, capture: bool) -> tuple:
    """Kings and all captures go any diagonal way; men only step forward."""
    if piece in ("B", "W") or capture:
        return _CK_DIAG
    return ((1, -1), (1, 1)) if piece == "b" else ((-1, -1), (-1, 1))


def _ck_steps(grid: list[list[str]], r: int,
              c: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in _ck_dirs(grid[r][c], False):
        lr, lc = r + dr, c + dc
        if 0 <= lr < 8 and 0 <= lc < 8 and not grid[lr][lc]:
            out.append((lr, lc))
    return out


def _ck_jumps(grid: list[list[str]], r: int,
              c: int) -> list[tuple[int, int]]:
    piece, side = grid[r][c], _ck_side(grid[r][c])
    out = []
    for dr, dc in _ck_dirs(piece, True):
        mr, mc, lr, lc = r + dr, c + dc, r + 2 * dr, c + 2 * dc
        if not (0 <= lr < 8 and 0 <= lc < 8):
            continue
        mid = grid[mr][mc]
        if mid and _ck_side(mid) != side and not grid[lr][lc]:
            out.append((lr, lc))
    return out


def _ck_crown(piece: str, r: int) -> str:
    if piece == "b" and r == 7:
        return "B"
    if piece == "w" and r == 0:
        return "W"
    return piece


def _ck_chains(grid: list[list[str]], r: int,
               c: int) -> list[list[tuple[int, int]]]:
    """Every complete capture chain starting at (r, c).

    A chain ends when no further jump exists — or the moment a man is
    crowned mid-chain (english rule: crowning ends the turn)."""
    results: list[list[tuple[int, int]]] = []

    def dfs(g: list[list[str]], rr: int, cc: int,
            path: list[tuple[int, int]]) -> None:
        jumps = _ck_jumps(g, rr, cc)
        if not jumps:
            if len(path) > 1:
                results.append(path)
            return
        for lr, lc in jumps:
            ng = [row[:] for row in g]
            piece = ng[rr][cc]
            ng[(rr + lr) // 2][(cc + lc) // 2] = ""
            ng[rr][cc] = ""
            crowned = _ck_crown(piece, lr) != piece
            ng[lr][lc] = _ck_crown(piece, lr)
            if crowned:
                results.append(path + [(lr, lc)])
            else:
                dfs(ng, lr, lc, path + [(lr, lc)])

    dfs([row[:] for row in grid], r, c, [(r, c)])
    return results


def _ck_all_moves(grid: list[list[str]],
                  side: str) -> list[list[tuple[int, int]]]:
    """All legal moves for ``side``. Captures are forced: when any
    capture exists, only capture chains are returned."""
    caps: list[list[tuple[int, int]]] = []
    for r in range(8):
        for c in range(8):
            if grid[r][c] and _ck_side(grid[r][c]) == side:
                caps.extend(_ck_chains(grid, r, c))
    if caps:
        return caps
    steps: list[list[tuple[int, int]]] = []
    for r in range(8):
        for c in range(8):
            if grid[r][c] and _ck_side(grid[r][c]) == side:
                for landing in _ck_steps(grid, r, c):
                    steps.append([(r, c), landing])
    return steps


def _ck_has_pieces(grid: list[list[str]], side: str) -> bool:
    return any(cell and _ck_side(cell) == side
               for row in grid for cell in row)


def _ck_parse(text: str) -> list[tuple[int, int]] | None:
    parts = [p.strip() for p in (text or "").strip().lower().split("-")]
    if len(parts) < 2 or any(not _CK_SQ.match(p) for p in parts):
        return None
    out = []
    for p in parts:
        m = _CK_SQ.match(p)
        out.append((int(m.group(2)) - 1, ord(m.group(1)) - ord("a")))
    return out


def _ck_sq(r: int, c: int) -> str:
    return f"{chr(ord('a') + c)}{r + 1}"


class CheckersGame(MultiGame):
    name = "checkers"
    description = "english draughts — forced jumps, kings, async"
    min_players = 1
    max_players = 2
    ai_seats = 1
    move_timeout = 0
    idle_ttl = INBOX_IDLE_TTL
    difficulties = DIFFICULTY_LEVELS
    rules = ("English draughts. You're black (b, bottom, moving up), the "
             "house is white (w). Say a move like c3-d4 — captures are "
             "c3-e5, chains c3-e5-g7. Jumps are FORCED: if you can take, "
             "you must, and you must finish the chain (men crown the "
             "moment they reach the far rank, ending the move). Kings "
             "(B/W) step any diagonal way. Take every enemy piece — or "
             "leave it with no legal move — to win. No clock. Set the "
             "house's strength: /game checkers [easy|normal|hard|expert].")

    def new_state(self, rng: random.Random,
                  difficulty: str = "normal") -> dict[str, Any]:
        grid = [[""] * 8 for _ in range(8)]
        for r in range(3):
            for c in range(8):
                if (r + c) % 2 == 1:
                    grid[r][c] = "b"
        for r in range(5, 8):
            for c in range(8):
                if (r + c) % 2 == 1:
                    grid[r][c] = "w"
        return {"grid": grid, "moves": 0, "over": False, "winner": "",
                "caps_b": 0, "caps_w": 0,
                "difficulty": normalize_difficulty(difficulty)}

    def _house_pick(self, room: Room,
                    moves: list[list[tuple[int, int]]]
                    ) -> list[tuple[int, int]]:
        """Difficulty-aware house move: easy wanders, normal takes the
        longest chain, hard+ also guards its back rank and prefers
        crowning."""
        rng = self.rng(room)
        diff = self.difficulty(room)
        caps = [p for p in moves if abs(p[1][0] - p[0][0]) == 2]
        pool = caps or moves
        if diff == "easy":
            return rng.choice(pool)
        best = max(len(p) for p in pool)
        tied = [p for p in pool if len(p) == best]
        if diff == "normal":
            return rng.choice(tied)

        def value(path: list[tuple[int, int]]) -> tuple[int, ...]:
            g = room.state["grid"]
            (sr, sc), (er, ec) = path[0], path[-1]
            piece = g[sr][sc]
            crown = 1 if (piece == "w" and er == 0) else 0
            # don't abandon the back rank with a man
            back = 1 if (piece == "w" and sr == 7 and er != 7) else 0
            # centralize
            centre = -abs(ec - 3.5)
            return (len(path), crown, -back, int(centre * 2))

        top = max(value(p) for p in tied)
        return rng.choice([p for p in tied if value(p) == top])

    def setup(self, room: Room, mind: GameMind) -> str:
        return ("checkers — you're black (b), bottom, moving up. say "
                "c3-d4, captures c3-e5, chains c3-e5-g7. jumps are "
                "forced. no clock — play whenever.")

    def _board(self, room: Room) -> str:
        g = room.state["grid"]
        rows = ["   " + " ".join("abcdefgh")]
        for r in range(8):
            cells = []
            for c in range(8):
                if (r + c) % 2 == 0:
                    cells.append(" ")
                elif g[r][c]:
                    cells.append(g[r][c])
                else:
                    cells.append("·")
            rows.append(f"{r + 1}  " + " ".join(cells))
        s = room.state
        rows.append(f"taken — you {s['caps_b']}, house {s['caps_w']}")
        return "\n".join(rows)

    def _apply(self, room: Room, path: list[tuple[int, int]],
               side: str) -> int:
        """Run a validated path; returns pieces captured."""
        s, g = room.state, room.state["grid"]
        (r, c) = path[0]
        piece = g[r][c]
        g[r][c] = ""
        taken = 0
        for lr, lc in path[1:]:
            if abs(lr - r) == 2:  # a jump
                g[(r + lr) // 2][(c + lc) // 2] = ""
                taken += 1
            r, c = lr, lc
        g[r][c] = _ck_crown(piece, r)
        s["moves"] += 1
        s["caps_b" if side == "B" else "caps_w"] += taken
        return taken

    def _close(self, room: Room, just_moved: str) -> bool:
        """True when ``just_moved`` just won (opponent wiped or stuck)."""
        s = room.state
        foe = "W" if just_moved == "B" else "B"
        if not _ck_has_pieces(s["grid"], foe) \
                or not _ck_all_moves(s["grid"], foe):
            s["over"] = True
            s["winner"] = just_moved
            return True
        return False

    def on_move(self, room: Room, player: Player, text: str,
                mind: GameMind) -> list[str]:
        s = room.state
        if s["over"]:
            return ["the board is done — /game rematch for another."]
        path = _ck_parse(text)
        legal = _ck_all_moves(s["grid"], "B")
        if path is None or not legal:
            return ["a move like c3-d4 (captures c3-e5, chains c3-e5-g7)."]
        if path not in legal:
            # explain WHY, with the forced-capture rule first
            must_jump = any(len(p) > 2 or abs(p[1][0] - p[0][0]) == 2
                            for p in legal)
            if must_jump:
                eg = "-".join(_ck_sq(r, c) for r, c in legal[0])
                return [f"you have a jump — captures are forced. e.g. {eg}"]
            eg = "-".join(_ck_sq(r, c) for r, c in legal[0])
            return [f"illegal move. e.g. {eg}"]
        taken = self._apply(room, path, "B")
        mv = "-".join(_ck_sq(r, c) for r, c in path)
        out = [f"you play {mv}" + (f" — take {taken}." if taken else ".")]
        if self._close(room, "B"):
            out.append(self._board(room))
            return out
        # the house replies (captures forced for it too)
        w_moves = _ck_all_moves(s["grid"], "W")
        if w_moves:
            pick = self._house_pick(room, w_moves)
            taken = self._apply(room, pick, "W")
            hmv = "-".join(_ck_sq(r, c) for r, c in pick)
            out.append(f"house plays {hmv}"
                       + (f" — takes {taken}." if taken else "."))
            self._close(room, "W")
        out.append(self._board(room))
        return out

    def ai_turn(self, room: Room, mind: GameMind) -> list[str]:
        return []

    def is_over(self, room: Room) -> bool:
        return bool(room.state.get("over"))

    def winner(self, room: Room) -> Player | str | None:
        s = room.state
        if not s.get("over"):
            return None
        w = s.get("winner")
        if w == "draw":
            return "draw"
        if w == "B":
            return _human(room)
        return _HOUSE

    def score(self, room: Room, player: Player) -> int:
        s = room.state
        if (s.get("over") and s.get("winner") == "B"
                and not player.is_ai):
            return 10 + 2 * int(s.get("caps_b") or 0)
        return 0

    def final_message(self, room: Room, mind: GameMind) -> str:
        s = room.state
        h = _human(room)
        who = h.name if h else "you"
        if s.get("winner") == "B":
            return (f"🏁 {who} win checkers — {s['caps_b']} taken.")
        return f"🏁 the house wins checkers — {s['caps_w']} taken."

    def describe_state(self, room: Room) -> str:
        s = room.state
        return (f"move {s['moves']} · you black (b/B), house white\n"
                + self._board(room))


INBOX_GAMES: tuple[MultiGame, ...] = (
    GomokuGame(),
    ReversiGame(),
    CheckersGame(),
)
