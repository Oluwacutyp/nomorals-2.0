"""GameMind: the AI's seat at every table.

The house can be three things at once, and it is all of them:

* **Referee** — the pure rule logic lives in each game module; the mind
  renders the referee's voice around it (flavor, pressure, commentary)
  and falls back to neutral wording when no model is connected, so
  games stay fully playable offline.
* **Host** — for lobby-driven games (mafia, escape room, political) the
  mind narrates the world: the case, the night, the session.
* **Player** — every game that fills AI seats needs an actual brain:
  word chains, hangman guesses, duel answers, auction bids, arena
  combat, votes. Each game module implements ``ai_move`` with a
  deterministic built-in brain (word banks, heuristics, minimax-lite);
  the mind wraps it with an optional model call, and the built-in brain
  is the guarantee that a move always exists.

``suggest`` is the same pattern the legacy games agent uses: a
``(prompt) -> text`` function backed by the model router when one is
connected, and ``None`` otherwise.  Nothing here raises for a missing
model — games must never die because the model did.
"""
from __future__ import annotations

import random
import re
import time
from typing import Any, Callable, Sequence

from ..core.logging_setup import get_logger

__all__ = ["SuggestFn", "GameMind", "pick", "one_of",
           "connect4_move", "battleship_shot", "reversi_move"]

_log = get_logger(__name__)

SuggestFn = Callable[[str], str]  # (prompt) -> model text, may return ""

_WORD_RE = re.compile(r"[a-z']+")


def one_of(rng: random.Random, options: tuple | list) -> Any:
    return rng.choice(options)


def pick(rng: random.Random, options: tuple | list) -> Any:
    return rng.choice(options)


class GameMind:
    """The house brain: flavor + model upgrade + built-in fallbacks."""

    def __init__(self, suggest: SuggestFn | None = None,
                 seed: int | None = None) -> None:
        self._suggest = suggest
        self.rng = random.Random(seed)
        self._cache: dict[str, tuple[float, str]] = {}

    # ── model bridge ─────────────────────────────────────────────────────────
    @property
    def model_on(self) -> bool:
        return self._suggest is not None

    def ask(self, prompt: str, *, cache_key: str = "",
            max_len: int = 220) -> str:
        """Ask the model, with a 60 s freshness cache per cache_key so a
        single turn never double-spends on the same question.  Returns
        "" when no model is connected or it fails — the caller always
        has a built-in fallback to fall through to."""
        if self._suggest is None:
            return ""
        if cache_key:
            hit = self._cache.get(cache_key)
            if hit is not None and time.time() - hit[0] < 60:
                return hit[1]
        try:
            raw = self._suggest(prompt) or ""
        except Exception:  # noqa: BLE001 - the model must never kill a game
            _log.debug("game mind model call failed", exc_info=True)
            return ""
        raw = raw.strip().strip('"').strip()
        if len(raw) > max_len:
            raw = raw[:max_len].rstrip() + "…"
        if cache_key:
            now = time.time()
            # the mind lives as long as the engine — evict stale entries
            # and cap the size so this cache can't grow without bound
            stale = [k for k, (ts, _v) in self._cache.items()
                     if now - ts >= 60]
            for k in stale:
                del self._cache[k]
            while len(self._cache) >= 512:
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = (now, raw)
        return raw

    # ── referee / host voice ─────────────────────────────────────────────────
    def say(self, base: str, *, flavor: str = "", cache_key: str = "") -> str:
        """Deliver a message with optional model flavor.

        ``base`` is always the truth (the rules said this); ``flavor`` is
        a prompt asking for a one-line in-character garnish.  When the
        model is off, the base ships as-is — pure, fast, deterministic.
        """
        if not flavor:
            return base
        garnish = self.ask(flavor, cache_key=cache_key or f"fl:{base[:40]}",
                           max_len=110)
        if not garnish:
            return base
        return f"{base}\n_{garnish}_"

    def intro(self, game: str, description: str, players: list[str],
              rules: str) -> str:
        names = ", ".join(players)
        base = (f"🎮 {game} is live — {description}.\n"
                f"players: {names}\n{rules}")
        return self.say(base, flavor=(
            f"You are the host of a chat game called {game}. "
            f"Say one short line of in-character welcome to the players "
            f"({names}). No emoji. Under 20 words."
        ))

    # ── player brains ────────────────────────────────────────────────────────
    def word_starting_with(self, letter: str,
                           bank: tuple[str, ...] = ()) -> str:
        """A real word starting with ``letter``: model first, bank second,
        constructed third. Returns "" only if truly nothing exists."""
        letter = letter.lower()
        reply = self.ask(
            f"Word game. Reply with ONE common English word starting with "
            f"'{letter}', 3-12 letters, lowercase, no punctuation, no "
            "explanation."
        )
        if reply:
            w = _WORD_RE.fullmatch(reply.lower().strip())
            if w and w.group(0)[0] == letter and 3 <= len(w.group(0)) <= 15:
                return w.group(0)
        bank = [w for w in bank if w.startswith(letter)]
        if bank:
            return self.rng.choice(bank)
        # constructed fallbacks that always start with the right letter
        for body in ("ame", "old", "est", "ack", "ing"):
            if len(letter) == 1 and letter.isalpha():
                return letter + body
        return ""

    def letter_guess(self, revealed: set[str], word_length: int,
                     category: str = "",
                     priority: Sequence[str] = ()) -> str:
        """Hangman: pick the most valuable letter not yet revealed.

        ``priority`` is the game's deck-aware ordering for this
        (category, length) — the word list IS the deck, so guessing the
        letter in the most remaining deck cards is optimal play.  Falls
        back to plain English frequency when the game gives none.
        """
        order = tuple(priority) if priority else ("e", "t", "a", "o", "i", "n",
                                                  "s", "h", "r", "d", "l", "c",
                                                  "u", "m", "w", "f", "g", "y",
                                                  "p", "b", "v", "k", "j", "x",
                                                  "q", "z")
        for letter in order:
            if letter not in revealed:
                return letter
        for letter in "abcdefghijklmnopqrstuvwxyz":
            if letter not in revealed:
                return letter
        return "z"

    def number(self, low: int, high: int) -> int:
        """Number-guess battle: bisection of the live window the game
        tracks from high/low feedback — the strongest honest strategy,
        so the AI is a real opponent."""
        low, high = max(1, low), max(low, high)
        if low > high:
            return low
        return (low + high) // 2

    def yes_no(self, truth: bool, question: str = "") -> str:
        """Answer honestly for host-side yes/no games (20q-style, mafia
        day questions) — model may color it, truth never changes."""
        return "yes" if truth else "no"

    def choice(self, options: list[str], context: str = "") -> str:
        """Pick an option: model if connected and it names one, else the
        rng. Used for would-you-rather as AI, auction opening bids,
        story chain prompts, mafia votes."""
        if options:
            letters = "ABCD"
            if self.model_on and context:
                menu = "\n".join(f"{letters[i]}. {o}"
                                 for i, o in enumerate(options[:4]))
                reply = self.ask(
                    f"Game context: {context}\nChoose one option.\n{menu}\n"
                    "Reply with ONLY the letter."
                )
                m = re.fullmatch(r"\s*([A-D])\s*\.?", reply or "", re.I)
                if m:
                    idx = ord(m.group(1).upper()) - 65
                    if idx < len(options):
                        return options[idx]
            return self.rng.choice(options)
        return ""

    def bid(self, floor: int, ceiling: int,
            value_estimate: float) -> int:
        """Auction bidding: bid toward perceived value with noise,
        never past the ceiling, at least the floor."""
        if value_estimate <= floor:
            return floor
        target = min(ceiling, max(floor, int(value_estimate * 0.9)))
        jitter = self.rng.randint(-1, 2)
        return max(floor, min(ceiling, target + jitter))

    def story_sentence(self, prompt: str) -> str:
        """One sentence continuing the story — model, or a built-in
        connector if offline (the game stays playable, just plainer)."""
        reply = self.ask(
            f"Story chain. Continue this story with exactly ONE sentence "
            f"(max 20 words), in the same style:\n{prompt}"
        )
        if reply and 10 <= len(reply) <= 260:
            return reply
        return self.rng.choice((
            "But no one in the room noticed the door had opened.",
            "By morning, everything would look different.",
            "That was the moment the plan stopped being a plan.",
            "The rain, when it came, washed nothing away.",
        ))

    def case_clue(self, case: str) -> str:
        """Investigation host: one new clue from the case, in order
        (deterministic so the case is always solvable)."""
        return self.ask(
            f"Crime story: {case}\nGive the investigators ONE new short "
            "clue (max 25 words) that helps but does not reveal the "
            "culprit."
        )

    def combat_move(self, self_stats: dict[str, int],
                    foe_stats: dict[str, int],
                    skill: int = 0) -> dict[str, Any]:
        """Battle arena AI: attack/focus/fury/defend/potion by math.

        Stats use the arena's key names (``atk``/``def``); the old
        ``attack``/``defense`` names are tolerated so external callers
        keep working.

        ``skill`` 0–5 is the hunter rank index (E→S). Higher ranks potion
        earlier, fury more freely, and guard less panicky — the same brain,
        but the S-rank version fights like it means it.
        """
        skill = max(0, min(5, int(skill or 0)))
        hp = self_stats.get("hp", 1)
        max_hp = self_stats.get("max_hp", max(hp, 1))
        my_atk = self_stats.get("atk", self_stats.get("attack", 0))
        my_def = self_stats.get("def", self_stats.get("defense", 0))
        foe_hp = foe_stats.get("hp", 1)
        foe_atk = foe_stats.get("atk", foe_stats.get("attack", 0))
        foe_def = foe_stats.get("def", foe_stats.get("defense", 0))
        if hp <= 0 or foe_hp <= 0:
            return {"action": "attack"}
        # bleeding and holding a potion: drink — veterans drink earlier
        if hp < max_hp * (0.30 + 0.03 * skill) \
                and self_stats.get("potions", 0) > 0:
            return {"action": "potion"}
        # a focused hit that finishes the fight: set it up (or spend it)
        focused_raw = max(1, int((my_atk - foe_def // 2) * 1.5))
        if focused_raw >= foe_hp:
            if self_stats.get("focused"):
                return {"action": "attack"}
            return {"action": "focus"}
        # the foe out-hits us badly: guard instead of trading — veterans
        # hold their nerve a little longer before turtling
        if foe_atk > my_def + 8 - skill:
            return {"action": "defend"}
        # healthy and off cooldown: fury swings are worth it — veterans
        # commit even when slightly chipped
        if self_stats.get("fury_cd", 0) <= 0 \
                and hp >= max_hp * (0.70 - 0.02 * skill):
            return {"action": "fury"}
        return {"action": "attack"}

    def vote(self, players: list[str], suspicion: dict[str, float],
             context: str = "") -> str:
        """Mafia/political voting: pick the most suspicious player,
        model may override with a named player, otherwise the heuristic
        wins. ``suspicion`` is the game's own scoring of each seat."""
        if not players:
            return ""
        ranked = sorted(suspicion.items(), key=lambda kv: kv[1], reverse=True)
        if self.model_on and context:
            menu = "\n".join(f"- {p} (suspicion {s:.0f})"
                             for p, s in ranked)
            reply = self.ask(
                f"{context}\n{menu}\nWho do you vote for? Reply with ONLY "
                "the player's name."
            )
            for p in players:
                if p.lower() in (reply or "").lower():
                    return p
        return ranked[0][0] if ranked else self.rng.choice(players)


# ── connect four: alpha-beta brain ───────────────────────────────────────────

_C4_ROWS, _C4_COLS = 6, 7
_C4_ORDER = (3, 2, 4, 1, 5, 0, 6)  # centre-first move ordering


def _c4_all_windows() -> tuple[tuple[tuple[int, int], ...], ...]:
    wins = []
    for r in range(_C4_ROWS):
        for c in range(_C4_COLS):
            if c + 3 < _C4_COLS:
                wins.append(((r, c), (r, c + 1), (r, c + 2), (r, c + 3)))
            if r + 3 < _C4_ROWS:
                wins.append(((r, c), (r + 1, c), (r + 2, c), (r + 3, c)))
            if r + 3 < _C4_ROWS and c + 3 < _C4_COLS:
                wins.append(((r, c), (r + 1, c + 1),
                             (r + 2, c + 2), (r + 3, c + 3)))
            if r + 3 < _C4_ROWS and c - 3 >= 0:
                wins.append(((r, c), (r + 1, c - 1),
                             (r + 2, c - 2), (r + 3, c - 3)))
    return tuple(wins)


#: every 4-cell window on the board, computed once
_C4_WINDOWS = _c4_all_windows()

#: windows through each cell — the incremental win check only looks
#: at these instead of re-scanning the whole board per node
_C4_THROUGH: dict[tuple[int, int], tuple] = {}
for _w in _C4_WINDOWS:
    for _cell in _w:
        _C4_THROUGH.setdefault(_cell, []).append(_w)
_C4_THROUGH = {k: tuple(v) for k, v in _C4_THROUGH.items()}


def _c4_score_window(a: int, b: int, c: int, d: int, me: int) -> int:
    foe = 3 - me
    mine = (a == me) + (b == me) + (c == me) + (d == me)
    theirs = (a == foe) + (b == foe) + (c == foe) + (d == foe)
    if mine == 4:
        return 100000
    if theirs == 4:
        return -100000
    if mine == 3 and theirs == 0:
        return 120
    if mine == 2 and theirs == 0:
        return 12
    if theirs == 3 and mine == 0:
        return -150  # blocking an enemy threat outranks our own build
    if theirs == 2 and mine == 0:
        return -10
    return 0


def _c4_evaluate(board: list[list[int]], me: int) -> int:
    score = 0
    for r in range(_C4_ROWS):
        if board[r][3] == me:
            score += 6
    for (a, b, c, d) in _C4_WINDOWS:
        score += _c4_score_window(board[a[0]][a[1]], board[b[0]][b[1]],
                                 board[c[0]][c[1]], board[d[0]][d[1]], me)
    return score


def _c4_valid(board: list[list[int]]) -> list[int]:
    return [c for c in range(_C4_COLS) if board[0][c] == 0]


def _c4_drop_row(board: list[list[int]], col: int) -> int:
    for r in range(_C4_ROWS - 1, -1, -1):
        if board[r][col] == 0:
            return r
    return -1


def _c4_won_at(board: list[list[int]], r: int, c: int,
               player: int) -> bool:
    """Did the disc just dropped at (r, c) complete a four? Only the
    windows through that cell can have changed."""
    for (a, b, d, e) in _C4_THROUGH[(r, c)]:
        if (board[a[0]][a[1]] == player and board[b[0]][b[1]] == player
                and board[d[0]][d[1]] == player
                and board[e[0]][e[1]] == player):
            return True
    return False


class _C4Timeout(Exception):
    """Raised inside the search when the time budget runs out."""


def _c4_alphabeta(board: list[list[int]], depth: int, alpha: float,
                  beta: float, maximizing: bool, me: int,
                  deadline: list[float], nodes: list[int],
                  last: tuple[int, int, int] | None) -> float:
    """``last`` is the (row, col, player) of the move that led to this
    node (None at the root) — the terminal check only inspects the
    windows through that cell."""
    nodes[0] += 1
    if nodes[0] & 1023 == 0 and time.monotonic() > deadline[0]:
        raise _C4Timeout()
    if last is not None and _c4_won_at(board, last[0], last[1], last[2]):
        return (1000000 + depth) if last[2] == me else (-1000000 - depth)
    if depth == 0:
        return _c4_evaluate(board, me)
    valid = [c for c in _C4_ORDER if board[0][c] == 0]
    if not valid:
        return 0
    if maximizing:
        value = float("-inf")
        for col in valid:
            r = _c4_drop_row(board, col)
            board[r][col] = me
            value = max(value, _c4_alphabeta(board, depth - 1, alpha, beta,
                                            False, me, deadline, nodes,
                                            (r, col, me)))
            board[r][col] = 0
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    value = float("inf")
    for col in valid:
        r = _c4_drop_row(board, col)
        board[r][col] = 3 - me
        value = min(value, _c4_alphabeta(board, depth - 1, alpha, beta,
                                        True, me, deadline, nodes,
                                        (r, col, 3 - me)))
        board[r][col] = 0
        beta = min(beta, value)
        if alpha >= beta:
            break
    return value


#: (max depth, seconds) per difficulty for the connect-four brain.
#: iterative deepening keeps the best move of the last fully searched
#: depth, so the house is always responsive no matter the position.
_C4_SEARCH = {"easy": (1, 0.05), "normal": (4, 0.4), "hard": (6, 1.0),
              "expert": (8, 2.0)}


def connect4_move(board: list[list[int]], me: int, *,
                  difficulty: str = "normal",
                  rng: random.Random | None = None) -> int:
    """Pick a column for ``me`` (1 or 2).

    easy: usually plays random columns, and sometimes doesn't even see
    an immediate win or block. normal and up: iterative-deepening
    alpha-beta with centre-first ordering —
    each difficulty gets a deeper ceiling and a bigger time budget, and
    the best move of the last fully searched depth is always kept, so
    the house never stalls the chat. Always returns a legal column (or
    -1 on a full board)."""
    rng = rng or random.Random()
    valid = _c4_valid(board)
    if not valid:
        return -1
    if difficulty == "easy" and rng.random() < 0.35:
        # the easy house genuinely misses it sometimes — the win or
        # the block sits there and it plays elsewhere
        return rng.choice(valid)
    # take the win / block the loss — every difficulty does this first
    for col in valid:
        r = _c4_drop_row(board, col)
        board[r][col] = me
        won = _c4_won_at(board, r, col, me)
        board[r][col] = 0
        if won:
            return col
    for col in valid:
        r = _c4_drop_row(board, col)
        board[r][col] = 3 - me
        won = _c4_won_at(board, r, col, 3 - me)
        board[r][col] = 0
        if won:
            return col
    if difficulty == "easy":
        return rng.choice(valid)
    max_depth, budget = _C4_SEARCH.get(difficulty,
                                       _C4_SEARCH["normal"])
    deadline = [time.monotonic() + budget]
    best = valid[0]
    try:
        for depth in range(1, max_depth + 1):
            best_score = float("-inf")
            best_cols: list[int] = []
            nodes = [0]
            for col in _C4_ORDER:
                if col not in valid:
                    continue
                r = _c4_drop_row(board, col)
                board[r][col] = me
                score = _c4_alphabeta(board, depth - 1, float("-inf"),
                                     float("inf"), False, me,
                                     deadline, nodes, (r, col, me))
                board[r][col] = 0
                if score > best_score:
                    best_score, best_cols = score, [col]
                elif score == best_score:
                    best_cols.append(col)
            # a forced win found: stop searching, play it
            if best_score >= 1000000:
                return best_cols[0]
            best = rng.choice(best_cols) if best_cols else best
    except _C4Timeout:
        # the budget ran out mid-depth: keep the best move from the
        # last fully searched depth (already sitting in `best`)
        _log.debug("connect4 search hit its time budget")
    return best


# ── battleship: probability-density hunter ───────────────────────────────────

def _bs_unresolved_hits(shots: list[list[int]]) -> list[tuple[int, int]]:
    """Hit cells that still have an unshot 4-neighbor — i.e. ships that
    are not provably sunk. A sunk ship is fully enclosed by misses or
    the board edge, so it drops out of targeting on its own."""
    n = len(shots)
    hits = []
    for r in range(n):
        for c in range(n):
            if shots[r][c] != 1:
                continue
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and shots[nr][nc] == 0:
                    hits.append((r, c))
                    break
    return hits


def battleship_shot(shots: list[list[int]], ship_sizes: tuple[int, ...], *,
                    difficulty: str = "normal",
                    rng: random.Random | None = None) -> tuple[int, int]:
    """Pick the next cell to fire at.

    easy: checkerboard guesses; when it hits something it pokes a
    random neighbor (no line tracking — ships escape).
    normal: checkerboard hunt, but a smart target mode — an aligned
    pair of hits extends the line, a lone hit probes the neighbors.
    hard: probability-density hunt (every cell scored by how many
    legal placements of the remaining ship sizes cover it, filtered
    by known misses) plus the smart target mode.
    expert: the hard brain, but it never guesses among equals — the
    single best-density cell wins ties deterministically, and a lone
    hit probes the highest-density neighbor first.
    """
    rng = rng or random.Random()
    n = len(shots)
    hits = _bs_unresolved_hits(shots)

    def unshot(r: int, c: int) -> bool:
        return 0 <= r < n and 0 <= c < n and shots[r][c] == 0

    def centre_key(rc: tuple[int, int]) -> float:
        return (abs(rc[0] - (n - 1) / 2) + abs(rc[1] - (n - 1) / 2))

    def density_grid() -> list[list[int]]:
        density = [[0] * n for _ in range(n)]
        for size in ship_sizes:
            for r in range(n):
                for c in range(n):
                    # horizontal
                    if c + size <= n and all(shots[r][c + i] != 2
                                             for i in range(size)):
                        for i in range(size):
                            if shots[r][c + i] == 0:
                                density[r][c + i] += 1
                    # vertical
                    if r + size <= n and all(shots[r + i][c] != 2
                                             for i in range(size)):
                        for i in range(size):
                            if shots[r + i][c] == 0:
                                density[r + i][c] += 1
        return density

    def parity_hunt() -> tuple[int, int]:
        parity = [(r, c) for r in range(n) for c in range(n)
                  if shots[r][c] == 0 and (r + c) % 2 == 0]
        pool = parity or [(r, c) for r in range(n) for c in range(n)
                          if shots[r][c] == 0]
        return rng.choice(pool)

    def density_hunt(deterministic: bool) -> tuple[int, int]:
        density = density_grid()
        best = max(density[r][c] for r in range(n) for c in range(n)
                   if shots[r][c] == 0)
        top = [(r, c) for r in range(n) for c in range(n)
               if shots[r][c] == 0 and density[r][c] == best]
        # centre bias breaks ties toward the most likely waters
        top.sort(key=centre_key)
        if deterministic:
            return top[0]
        peak = centre_key(top[0])
        tied = [rc for rc in top if centre_key(rc) == peak]
        return rng.choice(tied)

    if hits and difficulty == "easy":
        # dumb target mode: poke a random neighbor of a random hit —
        # no line tracking, so damaged ships regularly escape
        nbrs = [(r + dr, c + dc) for r, c in hits
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))]
        live = [(r, c) for r, c in nbrs if unshot(r, c)]
        if live:
            return rng.choice(live)
        # no live neighbor (shouldn't happen): fall to the hunt

    if hits and difficulty != "easy":
        # smart target mode: finish the ship. An aligned pair of hits
        # reveals the axis — extend the line; a lone hit (or a
        # non-aligned cluster) probes the neighbors.
        density = density_grid() if difficulty == "expert" else None
        if len(hits) >= 2:
            rows = {r for r, _ in hits}
            cols = {c for _, c in hits}
            cands: list[tuple[int, int]] = []
            if len(rows) == 1:
                r = next(iter(rows))
                lo = min(c for _, c in hits) - 1
                hi = max(c for _, c in hits) + 1
                cands = [(r, lo), (r, hi)]
            elif len(cols) == 1:
                c = next(iter(cols))
                lo = min(r for r, _ in hits) - 1
                hi = max(r for r, _ in hits) + 1
                cands = [(lo, c), (hi, c)]
            live = [(r, c) for r, c in cands if unshot(r, c)]
            if live:
                if density is not None:
                    return max(live, key=lambda rc: density[rc[0]][rc[1]])
                return rng.choice(live)
        nbrs = [(r + dr, c + dc) for r, c in hits
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))]
        live = [(r, c) for r, c in nbrs if unshot(r, c)]
        if live:
            if density is not None:
                return max(live, key=lambda rc: density[rc[0]][rc[1]])
            return rng.choice(live)

    if difficulty in ("easy", "normal"):
        return parity_hunt()
    return density_hunt(deterministic=difficulty == "expert")


# ── reversi: positional brain ────────────────────────────────────────────────

#: classic square weights: corners are gold, the squares next to a
#: corner are poison (they hand the corner over).
_REVERSI_WEIGHTS = (
    (120, -20, 20, 5, 5, 20, -20, 120),
    (-20, -40, -5, -5, -5, -5, -40, -20),
    (20, -5, 15, 3, 3, 15, -5, 20),
    (5, -5, 3, 3, 3, 3, -5, 5),
    (5, -5, 3, 3, 3, 3, -5, 5),
    (20, -5, 15, 3, 3, 15, -5, 20),
    (-20, -40, -5, -5, -5, -5, -40, -20),
    (120, -20, 20, 5, 5, 20, -20, 120),
)


def reversi_move(grid: list[list[str]], side: str, *,
                difficulty: str = "normal",
                rng: random.Random | None = None,
                legal_fn=None, apply_fn=None) -> tuple[int, int] | None:
    """Pick a reversi move as (r, c).

    ``legal_fn(grid, side)`` returns ``[(r, c, flips)]`` and
    ``apply_fn(grid, r, c, side)`` mutates a grid copy — injected so the
    brain stays decoupled from any one game's helpers.
    easy: random legal. normal: most flips. hard/expert: positional
    weights (+ a shallow reply search on expert)."""
    rng = rng or random.Random()
    moves = legal_fn(grid, side)
    if not moves:
        return None
    if difficulty == "easy":
        return rng.choice([(r, c) for r, c, _ in moves])
    if difficulty == "normal":
        best = max(m[2] for m in moves)
        tied = [(r, c) for r, c, f in moves if f == best]
        return rng.choice(tied)

    def static_eval(g: list[list[str]]) -> int:
        foe = "W" if side == "B" else "B"
        total = 0
        mobility = 0
        for r in range(8):
            for c in range(8):
                if g[r][c] == side:
                    total += _REVERSI_WEIGHTS[r][c]
                elif g[r][c] == foe:
                    total -= _REVERSI_WEIGHTS[r][c]
        mobility = len(legal_fn(g, side)) - len(legal_fn(g, foe))
        return total + mobility * 5

    scored: list[tuple[float, int, int]] = []
    for r, c, _flips in moves:
        g2 = [row[:] for row in grid]
        apply_fn(g2, r, c, side)
        val = static_eval(g2)
        if difficulty == "expert":
            foe = "W" if side == "B" else "B"
            reply = legal_fn(g2, foe)
            if reply:
                worst = float("inf")
                for rr, cc, _ in reply[:12]:
                    g3 = [row[:] for row in g2]
                    apply_fn(g3, rr, cc, foe)
                    worst = min(worst, static_eval(g3))
                val = worst
        scored.append((val, r, c))
    top = max(s[0] for s in scored)
    tied = [(r, c) for v, r, c in scored if v == top]
    return rng.choice(tied)
