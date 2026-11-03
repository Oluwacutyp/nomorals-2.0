"""Emoji game boards (group build §12, owner order 2026-11-02:
"Devon's games, tic-tac-toe and similar games, rendered with emoji").

The registry in nomorals/games/boards.py renders boards from state;
games fall back to their legacy text when no renderer claims them.
These pins hold: the registry's renders, the fallback contract, and
the wired games (ttt / connect4 / reversi / checkers) actually
showing emoji boards through their own board paths.
"""
from __future__ import annotations

from nomorals.games.boards import BOARD_RENDERERS, keycap, render_board


def test_registry_claims_the_converted_games():
    for name in ("ttt", "connect4", "reversi", "checkers"):
        assert name in BOARD_RENDERERS


def test_unknown_game_falls_back():
    assert render_board("no-such-game", {}) is None
    assert render_board("sudoku", {"board": [0] * 81}) is None


def test_keycap_range():
    assert keycap(1) == "1️⃣" and keycap(9) == "9️⃣"
    assert keycap(10) == "10"


def test_ttt_empty_board_teaches_the_numbers():
    out = render_board("ttt", {"board": [""] * 9})
    assert out == "1️⃣2️⃣3️⃣\n4️⃣5️⃣6️⃣\n7️⃣8️⃣9️⃣"


def test_ttt_marks_replace_their_keycaps():
    board = ["X", "", "O", "", "X", "", "", "", "O"]
    out = render_board("ttt", {"board": board})
    lines = out.split("\n")
    assert lines[0] == "❌2️⃣⭕"
    assert lines[1] == "4️⃣❌6️⃣"
    assert lines[2] == "7️⃣8️⃣⭕"


def test_connect4_discs_and_keycap_header():
    board = [[0] * 7 for _ in range(6)]
    board[5][3] = 1
    board[5][4] = 2
    out = render_board("connect4", {"board": board})
    lines = out.split("\n")
    assert lines[0] == "1️⃣2️⃣3️⃣4️⃣5️⃣6️⃣7️⃣"
    assert lines[6] == "⚪⚪⚪🔴🟡⚪⚪"
    assert "·" not in out


def test_reversi_opening_position():
    grid = [[""] * 8 for _ in range(8)]
    grid[3][3] = "W"
    grid[4][4] = "W"
    grid[3][4] = "B"
    grid[4][3] = "B"
    out = render_board("reversi", {"grid": grid})
    lines = out.split("\n")
    assert len(lines) == 8
    assert lines[3] == "🟩🟩🟩⚪⚫🟩🟩🟩"
    assert lines[4] == "🟩🟩🟩⚫⚪🟩🟩🟩"
    assert "B" not in out and "W" not in out


def test_checkers_opening_position():
    grid = [[""] * 8 for _ in range(8)]
    for r in range(3):
        for c in range(8):
            if (r + c) % 2 == 1:
                grid[r][c] = "b"
    for r in range(5, 8):
        for c in range(8):
            if (r + c) % 2 == 1:
                grid[r][c] = "w"
    out = render_board("checkers", {"grid": grid})
    lines = out.split("\n")
    assert lines[0] == "⬜⚫⬜⚫⬜⚫⬜⚫"
    assert lines[3] == "🟫⬜🟫⬜🟫⬜🟫⬜"
    assert lines[7] == "⚪⬜⚪⬜⚪⬜⚪⬜"


def test_ttt_game_shows_the_emoji_board():
    from nomorals.games.games.wild import TicTacToeGame

    game = TicTacToeGame()

    class _Room:
        state = {"board": ["X"] + [""] * 8}

    text = game._board_text(_Room())
    assert text.split("\n")[0] == "❌2️⃣3️⃣"
    assert " | " not in text


def test_connect4_game_render_is_emoji():
    from nomorals.games.games.arcade import ConnectFourGame

    game = ConnectFourGame()
    board = [[0] * 7 for _ in range(6)]
    out = game._render(board)
    assert out.split("\n")[0] == "1️⃣2️⃣3️⃣4️⃣5️⃣6️⃣7️⃣"
    assert "·" not in out
