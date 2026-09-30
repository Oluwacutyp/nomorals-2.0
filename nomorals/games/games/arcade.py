"""Arcade games: grid-based, deterministic or simple AI, high-score oriented.

* **2048** — the sliding tile puzzle. 4×4 grid, merge equal tiles, reach 2048.
* **snake** — eat food, grow, don't hit yourself or the wall. Turn-based.
* **connect4** — drop discs into columns, align four in a row.
* **battleship** — hide your ships, hunt the AI's. Sinking all five wins.

All state is pure dicts in ``room.state``, all moves are scripted-testable.
"""
from __future__ import annotations

import random
from typing import Any

from ..ai import GameMind
from ..players import Player
from .base import MultiGame, Room

__all__ = ["ARCADE_GAMES"]


# ── 1. 2048 ─────────────────────────────────────────────────────────────────

class TwentyFortyEightGame(MultiGame):
    name = "2048"
    description = "slide tiles, merge equals, reach 2048"
    min_players = 1
    max_players = 1
    ai_seats = 0
    move_timeout = 0
    rules = ("Swipe u/d/l/r to slide all tiles. Equal tiles merge. "
             "Reach 2048 to win, or keep going for a high score. "
             "The board is full and no moves left? Game over.")

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        grid = [[0] * 4 for _ in range(4)]
        self._spawn(grid, rng)
        self._spawn(grid, rng)
        return {"grid": grid, "score": 0, "won": False, "lost": False}

    def _spawn(self, grid: list[list[int]], rng: random.Random) -> None:
        empty = [(r, c) for r in range(4) for c in range(4) if grid[r][c] == 0]
        if not empty:
            return
        r, c = rng.choice(empty)
        grid[r][c] = 4 if rng.random() < 0.1 else 2

    def _slide_row(self, row: list[int]) -> tuple[list[int], int]:
        """Slide and merge one row leftward. Returns (new_row, points)."""
        filtered = [x for x in row if x]
        merged = []
        points = 0
        skip = False
        for i, val in enumerate(filtered):
            if skip:
                skip = False
                continue
            if i + 1 < len(filtered) and filtered[i + 1] == val:
                merged.append(val * 2)
                points += val * 2
                skip = True
            else:
                merged.append(val)
        merged += [0] * (4 - len(merged))
        return merged, points

    def _move(self, grid: list[list[int]], direction: str) -> tuple[list[list[int]], int]:
        """Apply one move. Returns (new_grid, points_earned)."""
        g = [row[:] for row in grid]
        points = 0
        if direction == "l":
            for i in range(4):
                g[i], p = self._slide_row(g[i])
                points += p
        elif direction == "r":
            for i in range(4):
                rev, p = self._slide_row(g[i][::-1])
                g[i] = rev[::-1]
                points += p
        elif direction == "u":
            for c in range(4):
                col = [g[r][c] for r in range(4)]
                col, p = self._slide_row(col)
                for r in range(4):
                    g[r][c] = col[r]
                points += p
        elif direction == "d":
            for c in range(4):
                col = [g[r][c] for r in range(4)][::-1]
                col, p = self._slide_row(col)
                col = col[::-1]
                for r in range(4):
                    g[r][c] = col[r]
                points += p
        return g, points

    def _can_move(self, grid: list[list[int]]) -> bool:
        if any(grid[r][c] == 0 for r in range(4) for c in range(4)):
            return True
        for r in range(4):
            for c in range(4):
                val = grid[r][c]
                if c + 1 < 4 and grid[r][c + 1] == val:
                    return True
                if r + 1 < 4 and grid[r + 1][c] == val:
                    return True
        return False

    def _render(self, grid: list[list[int]], score: int) -> str:
        lines = [f"score: {score}", "┌────┬────┬────┬────┐"]
        for row in grid:
            cells = "│".join(f"{x:^4}" if x else "    " for x in row)
            lines.append(f"│{cells}│")
            lines.append("├────┼────┼────┼────┤" if row != grid[-1] else "└────┴────┴────┴────┘")
        lines.append("swipe: u / d / l / r")
        return "\n".join(lines)

    def setup(self, room, mind):
        return self._render(room.state["grid"], room.state["score"])

    def on_move(self, room, player, text, mind):
        s = room.state
        t = text.strip().lower()
        if t not in {"u", "d", "l", "r", "up", "down", "left", "right"}:
            return ["swipe u / d / l / r"]
        direction = t[0]
        new_grid, points = self._move(s["grid"], direction)
        if new_grid == s["grid"]:
            return ["no tiles moved. try another direction."]
        s["grid"] = new_grid
        s["score"] += points
        self._spawn(s["grid"], room.rng())
        if any(s["grid"][r][c] >= 2048 for r in range(4) for c in range(4)):
            s["won"] = True
        if not self._can_move(s["grid"]):
            s["lost"] = True
        out = [self._render(s["grid"], s["score"])]
        if s["won"] and not s.get("won_announced"):
            out.append("🏆 you hit 2048! keep going or start over.")
            s["won_announced"] = True
        if s["lost"]:
            out.append(f"game over — final score {s['score']}.")
        return out

    def ai_turn(self, room, mind):
        return []

    def is_over(self, room):
        return room.state.get("lost", False)

    def score(self, room, player):
        return room.state.get("score", 0)

    def describe_state(self, room):
        return f"score {room.state['score']}, highest tile {max(max(r) for r in room.state['grid'])}"


# ── 2. snake ────────────────────────────────────────────────────────────────

class SnakeGame(MultiGame):
    name = "snake"
    description = "eat food, grow, don't crash"
    min_players = 1
    max_players = 1
    ai_seats = 0
    move_timeout = 0
    rules = ("Move u/d/l/r. Eat food (+10 pts, +1 length). Don't hit "
             "the wall or yourself. The board is 10×10.")

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        snake = [(5, 5)]
        food = self._place_food(snake, rng, 10, 10)
        return {"snake": snake, "food": food, "dir": "r",
                "score": 0, "alive": True, "size": 10}

    def _place_food(self, snake: list[tuple[int, int]], rng: random.Random,
                    w: int, h: int) -> tuple[int, int]:
        occupied = set(snake)
        while True:
            pos = (rng.randint(0, h - 1), rng.randint(0, w - 1))
            if pos not in occupied:
                return pos

    def _render(self, snake: list[tuple[int, int]], food: tuple[int, int],
                score: int, size: int) -> str:
        grid = [["·" for _ in range(size)] for _ in range(size)]
        for r, c in snake[1:]:
            grid[r][c] = "o"
        grid[snake[0][0]][snake[0][1]] = "@"
        grid[food[0]][food[1]] = "*"
        lines = [f"score: {score}"]
        for row in grid:
            lines.append(" ".join(row))
        lines.append("move: u / d / l / r")
        return "\n".join(lines)

    def setup(self, room, mind):
        s = room.state
        return self._render(s["snake"], s["food"], s["score"], s["size"])

    def on_move(self, room, player, text, mind):
        s = room.state
        t = text.strip().lower()
        if t not in {"u", "d", "l", "r", "up", "down", "left", "right"}:
            return ["move u / d / l / r"]
        direction = t[0]
        opposites = {"u": "d", "d": "u", "l": "r", "r": "l"}
        if direction == opposites.get(s["dir"]):
            return ["can't reverse into yourself"]
        s["dir"] = direction
        head_r, head_c = s["snake"][0]
        deltas = {"u": (-1, 0), "d": (1, 0), "l": (0, -1), "r": (0, 1)}
        dr, dc = deltas[direction]
        new_head = (head_r + dr, head_c + dc)
        size = s["size"]
        if not (0 <= new_head[0] < size and 0 <= new_head[1] < size):
            s["alive"] = False
            return ["crashed into the wall. game over.",
                    f"final score: {s['score']}",
                    self._render(s["snake"], s["food"], s["score"], size)]
        if new_head in s["snake"]:
            s["alive"] = False
            return ["you bit yourself. game over.",
                    f"final score: {s['score']}",
                    self._render(s["snake"], s["food"], s["score"], size)]
        s["snake"].insert(0, new_head)
        if new_head == s["food"]:
            s["score"] += 10
            s["food"] = self._place_food(s["snake"], room.rng(), size, size)
        else:
            s["snake"].pop()
        return [self._render(s["snake"], s["food"], s["score"], size)]

    def ai_turn(self, room, mind):
        return []

    def is_over(self, room):
        return not room.state.get("alive", True)

    def score(self, room, player):
        return room.state.get("score", 0)

    def describe_state(self, room):
        s = room.state
        return f"length {len(s['snake'])}, score {s['score']}"


# ── 3. connect four ─────────────────────────────────────────────────────────

class ConnectFourGame(MultiGame):
    name = "connect4"
    description = "drop discs, align four in a row"
    min_players = 2
    max_players = 2
    ai_seats = 1
    move_timeout = 60
    rules = ("Drop a disc into columns 1–7. First to align four "
             "(horizontal, vertical, or diagonal) wins. The board "
             "is 6 rows × 7 columns.")

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        return {"board": [[0] * 7 for _ in range(6)], "turn": 1,
                "winner": 0, "moves": 0}

    def _drop(self, board: list[list[int]], col: int, player: int) -> int:
        """Drop a disc into col. Returns the row it landed on, or -1 if full."""
        for r in range(5, -1, -1):
            if board[r][col] == 0:
                board[r][col] = player
                return r
        return -1

    def _check_win(self, board: list[list[int]], player: int) -> bool:
        for r in range(6):
            for c in range(7):
                if board[r][c] != player:
                    continue
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    count = 1
                    for i in range(1, 4):
                        nr, nc = r + dr * i, c + dc * i
                        if 0 <= nr < 6 and 0 <= nc < 7 and board[nr][nc] == player:
                            count += 1
                        else:
                            break
                    if count >= 4:
                        return True
        return False

    def _render(self, board: list[list[int]]) -> str:
        symbols = {0: "·", 1: "🔴", 2: "🟡"}
        lines = ["  1 2 3 4 5 6 7"]
        for row in board:
            lines.append(" ".join(symbols[x] for x in row))
        return "\n".join(lines)

    def setup(self, room, mind):
        return f"{self._render(room.state['board'])}\n{room.players[0].name} (🔴) vs {room.players[1].name if len(room.players) > 1 else 'AI'} (🟡)\n{room.players[0].name}'s turn — drop in column 1–7."

    def on_move(self, room, player, text, mind):
        s = room.state
        t = text.strip()
        if not t.isdigit() or not (1 <= int(t) <= 7):
            return ["pick a column 1–7"]
        col = int(t) - 1
        row = self._drop(s["board"], col, s["turn"])
        if row < 0:
            return ["that column is full"]
        s["moves"] += 1
        if self._check_win(s["board"], s["turn"]):
            s["winner"] = s["turn"]
            return [self._render(s["board"]),
                    f"🏆 {player.name} wins!"]
        if s["moves"] >= 42:
            return [self._render(s["board"]), "draw — the board is full."]
        s["turn"] = 3 - s["turn"]
        return [self._render(s["board"]),
                f"{room.players[s['turn'] - 1].name}'s turn."]

    def ai_turn(self, room, mind):
        s = room.state
        # simple AI: block wins, take wins, else pick a random column
        for col in range(7):
            if s["board"][0][col] != 0:
                continue
            test_board = [row[:] for row in s["board"]]
            self._drop(test_board, col, s["turn"])
            if self._check_win(test_board, s["turn"]):
                return [f"{room.current.name} drops in column {col + 1}"]
        for col in range(7):
            if s["board"][0][col] != 0:
                continue
            test_board = [row[:] for row in s["board"]]
            self._drop(test_board, col, 3 - s["turn"])
            if self._check_win(test_board, 3 - s["turn"]):
                return [f"{room.current.name} drops in column {col + 1}"]
        available = [c for c in range(7) if s["board"][0][c] == 0]
        if available:
            col = mind.rng.choice(available)
            return [f"{room.current.name} drops in column {col + 1}"]
        return []

    def is_over(self, room):
        return room.state.get("winner", 0) != 0 or room.state.get("moves", 0) >= 42

    def winner(self, room):
        w = room.state.get("winner", 0)
        if w == 0:
            return "draw"
        return room.players[w - 1] if w <= len(room.players) else None

    def score(self, room, player):
        idx = room.players.index(player) + 1 if player in room.players else 0
        return 10 if room.state.get("winner") == idx else 0


# ── 4. battleship ───────────────────────────────────────────────────────────

class BattleshipGame(MultiGame):
    name = "battleship"
    description = "hide your ships, hunt the AI's"
    min_players = 1
    max_players = 1
    ai_seats = 1
    move_timeout = 0
    rules = ("10×10 grid. You have 5 ships (sizes 5,4,3,3,2). The AI "
             "has the same. Take turns firing coordinates (e.g. B5). "
             "Sink all five to win.")

    SHIP_SIZES = (5, 4, 3, 3, 2)

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        player_board = self._place_ships(rng)
        ai_board = self._place_ships(rng)
        return {"player": player_board, "ai": ai_board,
                "player_shots": [[0] * 10 for _ in range(10)],
                "ai_shots": [[0] * 10 for _ in range(10)],
                "turn": "player", "winner": ""}

    def _place_ships(self, rng: random.Random) -> list[list[int]]:
        board = [[0] * 10 for _ in range(10)]
        for size in self.SHIP_SIZES:
            placed = False
            for _ in range(1000):
                r = rng.randint(0, 9)
                c = rng.randint(0, 9)
                horiz = rng.choice([True, False])
                if horiz:
                    if c + size > 10:
                        continue
                    if any(board[r][c + i] for i in range(size)):
                        continue
                    for i in range(size):
                        board[r][c + i] = 1
                else:
                    if r + size > 10:
                        continue
                    if any(board[r + i][c] for i in range(size)):
                        continue
                    for i in range(size):
                        board[r + i][c] = 1
                placed = True
                break
            if not placed:
                pass  # rare, skip this ship
        return board

    def _parse_coord(self, text: str) -> tuple[int, int] | None:
        text = text.strip().upper()
        if len(text) < 2:
            return None
        col = ord(text[0]) - ord("A")
        if not (0 <= col < 10):
            return None
        try:
            row = int(text[1:]) - 1
        except ValueError:
            return None
        if not (0 <= row < 10):
            return None
        return row, col

    def _render_own(self, board: list[list[int]], shots: list[list[int]]) -> str:
        lines = ["  A B C D E F G H I J"]
        for r in range(10):
            cells = []
            for c in range(10):
                if shots[r][c] == 1:
                    cells.append("x")
                elif shots[r][c] == 2:
                    cells.append("o")
                elif board[r][c]:
                    cells.append("#")
                else:
                    cells.append("·")
            lines.append(f"{r + 1:<2}" + " ".join(cells))
        return "\n".join(lines)

    def _render_enemy(self, shots: list[list[int]]) -> str:
        lines = ["  A B C D E F G H I J"]
        for r in range(10):
            cells = []
            for c in range(10):
                if shots[r][c] == 1:
                    cells.append("x")
                elif shots[r][c] == 2:
                    cells.append("o")
                else:
                    cells.append("·")
            lines.append(f"{r + 1:<2}" + " ".join(cells))
        return "\n".join(lines)

    def _count_hits(self, shots: list[list[int]]) -> int:
        return sum(shots[r][c] == 1 for r in range(10) for c in range(10))

    def setup(self, room, mind):
        s = room.state
        return ("battleship — your fleet:\n"
                f"{self._render_own(s['player'], s['player_shots'])}\n\n"
                "enemy waters:\n"
                f"{self._render_enemy(s['ai_shots'])}\n\n"
                "fire at a coordinate (e.g. B5).")

    def on_move(self, room, player, text, mind):
        s = room.state
        coord = self._parse_coord(text)
        if coord is None:
            return ["fire at a coordinate like B5 or E10"]
        r, c = coord
        if s["ai_shots"][r][c]:
            return ["you already fired there"]
        if s["ai"][r][c]:
            s["ai_shots"][r][c] = 1
            result = "HIT!"
        else:
            s["ai_shots"][r][c] = 2
            result = "miss."
        out = [f"you fire at {text.upper()}: {result}"]
        if self._count_hits(s["ai_shots"]) >= sum(self.SHIP_SIZES):
            s["winner"] = "player"
            out.append("🏆 you sank all their ships!")
            return out
        # AI's turn
        ai_coord = self._ai_fire(s["player_shots"], room.rng())
        if s["player"][ai_coord[0]][ai_coord[1]]:
            s["player_shots"][ai_coord[0]][ai_coord[1]] = 1
            ai_result = "HIT!"
        else:
            s["player_shots"][ai_coord[0]][ai_coord[1]] = 2
            ai_result = "miss."
        col_letter = chr(ord("A") + ai_coord[1])
        out.append(f"the AI fires at {col_letter}{ai_coord[0] + 1}: {ai_result}")
        if self._count_hits(s["player_shots"]) >= sum(self.SHIP_SIZES):
            s["winner"] = "ai"
            out.append("the AI sank all your ships. game over.")
            return out
        out.append("\nenemy waters:\n" + self._render_enemy(s["ai_shots"]))
        return out

    def _ai_fire(self, shots: list[list[int]], rng: random.Random) -> tuple[int, int]:
        # simple: pick a random unshot coordinate
        available = [(r, c) for r in range(10) for c in range(10) if shots[r][c] == 0]
        return rng.choice(available)

    def ai_turn(self, room, mind):
        return []

    def is_over(self, room):
        return bool(room.state.get("winner"))

    def winner(self, room):
        w = room.state.get("winner")
        if w == "player":
            return room.players[0] if room.players else None
        return Player(key="ai:battleship", platform="ai", name="The AI", is_ai=True)

    def score(self, room, player):
        s = room.state
        if s.get("winner") == "player":
            return 100
        return self._count_hits(s.get("ai_shots", []))


ARCADE_GAMES: tuple[MultiGame, ...] = (
    TwentyFortyEightGame(),
    SnakeGame(),
    ConnectFourGame(),
    BattleshipGame(),
)
