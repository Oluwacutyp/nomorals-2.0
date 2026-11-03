"""Emoji board renderers (group build §12): games that look like games.

A small registry maps a game name to a renderer over that game's
state dict. A game asks :func:`render_board` for its board text and
falls back to its own legacy rendering when the registry has no
renderer for it — the registry is open by design: new games bring new
renderers, and games whose boards don't suit an emoji grid (gomoku's
13–19-wide wars, sudoku's numbered grid) simply never register.

Rendering rules:
- One board = one block of text, meant to sit inside the game's
  single message (setup/move/finish all reuse the same render).
- Open cells teach the input: tic-tac-toe shows keycap digits on the
  squares still free, so the board itself says "say 1–9".
- Pure functions of state — no room, no I/O — so every renderer is
  unit-testable without the engine.
"""
from __future__ import annotations

from typing import Any, Callable

#: Keycap digit emoji for 1..9 (index 0 = "1").
KEYCAPS: tuple[str, ...] = (
    "1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣",
    "6️⃣", "7️⃣", "8️⃣", "9️⃣",
)


def keycap(n: int) -> str:
    """Keycap emoji for 1–9; the plain number outside that range."""
    return KEYCAPS[n - 1] if 1 <= n <= 9 else str(n)


def _grid(rows: list[list[str]]) -> str:
    return "\n".join("".join(row) for row in rows)


# ── renderers ────────────────────────────────────────────────────────

def _ttt(state: dict[str, Any]) -> str:
    board = state.get("board") or []
    cells = []
    for i in range(9):
        v = board[i] if i < len(board) else ""
        if v == "X":
            cells.append("❌")
        elif v == "O":
            cells.append("⭕")
        else:
            cells.append(keycap(i + 1))
    return _grid([cells[0:3], cells[3:6], cells[6:9]])


def _connect4(state: dict[str, Any]) -> str:
    board = state.get("board") or []
    symbols = {0: "⚪", 1: "🔴", 2: "🟡"}
    rows = [[symbols.get(v, "⚪") for v in row] for row in board]
    header = [keycap(c + 1) for c in range(7)]
    return _grid([header, *rows])


def _reversi(state: dict[str, Any]) -> str:
    grid = state.get("grid") or []
    symbols = {"B": "⚫", "W": "⚪", "": "🟩"}
    rows = [[symbols.get(v, "🟩") for v in row] for row in grid]
    return _grid(rows)


def _checkers(state: dict[str, Any]) -> str:
    grid = state.get("grid") or []
    # b/w are men, B/W kings; light squares ⬜, dark squares 🟫.
    symbols = {"b": "⚫", "w": "⚪", "B": "♚", "W": "♔"}
    rows = []
    for r, row in enumerate(grid):
        cells = []
        for c, v in enumerate(row):
            if v in symbols:
                cells.append(symbols[v])
            elif (r + c) % 2 == 0:
                cells.append("⬜")
            else:
                cells.append("🟫")
        rows.append(cells)
    return _grid(rows)


#: game name → renderer over the game's state dict. A renderer may
#: return None to decline (the game then uses its own rendering).
BOARD_RENDERERS: dict[str, Callable[[dict[str, Any]], str | None]] = {
    "ttt": _ttt,
    "connect4": _connect4,
    "reversi": _reversi,
    "checkers": _checkers,
}


def render_board(name: str, state: dict[str, Any]) -> str | None:
    """The emoji board for ``name`` over ``state``, or None when the
    registry has no renderer (or the renderer declines) — callers
    fall back to their legacy board text."""
    fn = BOARD_RENDERERS.get(name)
    if fn is None:
        return None
    try:
        return fn(state)
    except Exception:  # noqa: BLE001 - a board must never kill a game
        return None
