"""The live game table: every active game as ONE living card.

Doc 8 Priority 4: the board, whose turn it is, the legal actions,
the deadline and the stakes stay together on a single card that is
REVISED in place on every move — never a fresh message flood. The
card rides the Living Chat machinery (durable ``Card`` records,
revision-fenced ``cardtap`` buttons, per-platform render forms);
its actions are cut at the room's table revision and route back
through ``/game move <revision> …`` → ``GameEngine.move_fenced``,
so a stale tap is refused twice over (card revision at the card
store, room revision at the engine) and a move by a non-player or
out of turn is refused by identity at the engine.

Per-family identity: game families (strategy, casino, arcade,
puzzles, party, duel, inbox…) carry distinct banners through the
typography theme system — same machinery, different face.

This module is the games-side tenant of the card system: the
engine knows nothing about cards; the runtime installs
``engine.table_sink`` and this module's :func:`sync_table_card`
keeps the room's card current on every table event.
"""
from __future__ import annotations

import contextlib
import logging
import threading
import time
from dataclasses import replace
from typing import Any

from ..social.chat.action_schema import Action
from ..social.chat.living_cards import FAMILY_GAME, RENDERERS, Card
from ..social.chat.typography import Card as ThemeCard
from ..social.chat.typography import card_text
from .boards import render_board
from .engine import IDLE_ROOM_TTL
from .games.base import MultiGame, Room

__all__ = [
    "GAME_FAMILY_THEMES", "game_family", "timeout_policy",
    "stakes_label", "build_table_card", "sync_table_card",
    "table_card_id", "TABLE_TICK_SECONDS", "TableCardTicker",
]

_log = logging.getLogger("nomorals.games.table")

#: family → (icon, banner). The banner is the family's face on the
#: table; the machinery (card, fencing, rows) is shared.
GAME_FAMILY_THEMES: dict[str, tuple[str, str]] = {
    "strategy": ("♟", "strategy table — every move on the record"),
    "casino": ("🎰", "casino table — the purse holds the stakes"),
    "arcade": ("🕹", "arcade table — fast hands, real scores"),
    "puzzles": ("🧩", "puzzle table — think it through"),
    "party": ("🎭", "party table — trust nobody"),
    "duel": ("⚔️", "duel table — one winner takes it"),
    "narrative": ("📖", "story table — the tale is the board"),
    "inbox": ("✉️", "postal table — one move per message"),
    "table": ("🎲", "game table"),
}

#: game name → family, where the module grouping is a difficulty
#: tier rather than a family (easy/medium/wild/ambitious games).
_FAMILY_BY_GAME: dict[str, str] = {
    "ttt": "strategy", "connect4": "strategy", "reversi": "strategy",
    "checkers": "strategy", "gomoku": "strategy", "chess": "strategy",
    "mafia": "party", "spy": "party", "trivia": "party",
    "auction": "party", "twotruths": "party", "wyrr": "party",
    "hangman": "puzzles", "wordchain": "puzzles", "sudoku": "puzzles",
    "cryptogram": "puzzles", "numberguess": "puzzles",
    "arena": "duel", "raid": "duel",
}

#: games module → family (the codebase's own organization).
_FAMILY_BY_MODULE: dict[str, str] = {
    "casino": "casino", "arcade": "arcade", "puzzles": "puzzles",
    "inbox": "inbox", "pvp": "duel",
}


def game_family(game: MultiGame) -> str:
    """The presentation family of one game: its own declaration
    first, then the per-game map, then its module's family."""
    declared = str(getattr(game, "family", "") or "")
    if declared and declared != "table":
        return declared
    if game.name in _FAMILY_BY_GAME:
        return _FAMILY_BY_GAME[game.name]
    module = type(game).__module__.rsplit(".", 1)[-1]
    return _FAMILY_BY_MODULE.get(module, "table")


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds >= 86400:
        return f"{seconds // 86400}d"
    if seconds >= 3600:
        return f"{seconds // 3600}h"
    if seconds >= 60:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def timeout_policy(game: MultiGame) -> str:
    """The declared clock law for one game, in surface words:
    the per-turn clock, what silence costs, and when an abandoned
    table closes. Disconnects are not a separate case — a silent
    player IS a stalled player, and the engine clock enforces it.
    """
    parts: list[str] = []
    if game.move_timeout and game.move_timeout > 0:
        parts.append(f"⏳ {int(game.move_timeout)}s per turn")
        if type(game).on_timeout is MultiGame.on_timeout:
            limit = int(getattr(game, "stall_limit", 0) or 0)
            if limit > 0:
                parts.append(
                    f"{limit} stalls in a row = forfeit")
        else:
            parts.append("the game runs its own timeout rule")
    else:
        parts.append("no turn clock")
    idle = game.idle_ttl if game.idle_ttl else IDLE_ROOM_TTL
    parts.append(f"a silent table closes after {_duration(idle)}")
    return " · ".join(parts)


def stakes_label(game: MultiGame, room: Room) -> str:
    """What rides on this table, in surface words."""
    if getattr(game, "takes_wagers", False):
        return "💰 real wagers — stakes escrow in the purse"
    return "🪙 coins + points on the result"


def table_card_id(room: Room) -> str:
    """The card identity for one room's table (stable per room)."""
    return f"g{room.id}"


def build_table_card(engine: Any, room: Room, *,
                     viewer_key: str | None = None) -> Card:
    """Build the live-table card for one room.

    The shared card (``viewer_key=None``) renders from the game's
    redacted shared view — no player's secret is on it. Actions
    are the CURRENT player's legal moves, cut at the room's
    current revision; each routes back as a fenced engine move.
    """
    game = engine.games.get(room.game)
    view = engine.table_view(room, viewer_key)
    chat_id = room.chat_key.split(":", 1)[1] \
        if ":" in room.chat_key else room.chat_key

    lines: list[str] = []
    board = render_board(room.game, view["state"])
    if board:
        lines.append(board)
    elif game is not None:
        try:
            shim = replace(room, state=view["state"])
            extra = game.describe_state(shim)
            if extra:
                lines.append(extra)
        except Exception:  # noqa: BLE001 - a bad describe never
            pass  # breaks the table; the rows still carry the state
    cur = room.current
    if room.status == "active":
        if cur is not None and not cur.is_ai:
            lines.append(f"turn: {cur.name}")
        elif cur is not None:
            lines.append("turn: the house")
    players_line = " · ".join(
        ("🤖 " if p["is_ai"] else "") + p["name"]
        for p in view["players"])
    if players_line:
        lines.append(f"at the table: {players_line}")

    rows: list[tuple[str, str]] = []
    if cur is not None and room.status == "active":
        rows.append(("turn", cur.name if not cur.is_ai else "the house"))
    if view["deadline_at"]:
        left = max(0, int(view["deadline_at"] - time.time()))
        rows.append(("deadline", f"{left}s left"))
    rows.append(("clock", timeout_policy(game) if game else ""))
    rows.append(("stakes", view["stakes"]))

    actions: list[Action] = []
    if game is not None and room.status == "active" \
            and cur is not None and not cur.is_ai:
        options: list[str] = []
        try:
            options = list(game.describe_options(room) or [])
        except Exception:  # noqa: BLE001 - options are a courtesy;
            options = []  # the fence, not the list, is the guard
        for opt in options[:6]:
            actions.append(Action(
                label=str(opt),
                data=f"game move {room.revision} {opt}",
                target=room.id, operation_id=room.id,
                revision=room.revision))
        if not any(o.strip().lower() in ("pass", "/pass")
                   for o in options):
            actions.append(Action(
                label="Pass",
                data=f"game move {room.revision} /pass",
                target=room.id, operation_id=room.id,
                revision=room.revision))
        actions.append(Action(
            label="Resign",
            data="game resign",
            target=room.id, operation_id=room.id,
            revision=room.revision))

    return Card(
        card_id=table_card_id(room), family=FAMILY_GAME,
        chat_key=room.chat_key, owner_key="",
        platform=room.platform, chat_id=chat_id,
        title=room.game, body="\n".join(lines), rows=tuple(rows),
        status=room.status, actions=tuple(actions),
        revision=max(1, room.revision),
        terminal=room.status == "finished",
        state={"room_id": room.id, "game": room.game,
               "family": view["family"], "revision": room.revision},
    )


def sync_table_card(host: Any, engine: Any, room: Room,
                    chat: Any) -> Card:
    """Publish the room's table card, or revise it in place.

    Called by the runtime's table sink on every table event; the
    host owns delivery (edit in place where the platform supports
    it, a superseding send where it does not).
    """
    card = build_table_card(engine, room)
    existing = host.store.get(card.card_id)
    if existing is None:
        return host.publish(card, chat)
    return host.revise(
        existing, chat, title=card.title, body=card.body,
        rows=card.rows, status=card.status, actions=card.actions,
        state=card.state, terminal=card.terminal)


def _render_game_table(card: Card) -> str:
    """The game-family renderer: the typography theme system
    renders the card; the family's banner/icon give the table its
    identity."""
    family = str(card.state.get("family") or "table")
    icon, banner = GAME_FAMILY_THEMES.get(
        family, GAME_FAMILY_THEMES["table"])
    theme = ThemeCard("game", title=f"{icon} {card.title}")
    theme.line(banner)
    if card.status:
        theme.line(f"status: {card.status}")
    for line in (card.body or "").split("\n"):
        if line.strip():
            theme.line(line)
    if card.rows:
        theme.stats(list(card.rows))
    return card_text(theme)


# the rich renderer owns the game family once this organ is loaded
RENDERERS[FAMILY_GAME] = _render_game_table


#: How often a live table's card re-renders on the clock (seconds).
#: The countdown ("59s left") otherwise only moves on game events,
#: so the card's clock is frozen between moves.  Five seconds is
#: the sane cadence: a 1s ticker would hammer the adapter's edit
#: rate limit and flood the edit log for a display that only
#: resolves to whole seconds.
TABLE_TICK_SECONDS = 5.0


def _card_fingerprint(card: Card) -> tuple:
    """The rendered content of a table card, for the tick's
    identical-content skip: a tick that changes nothing must not
    reach the adapter at all (an identical edit is refused and
    falls back to a duplicate send)."""
    rows = tuple((str(key), str(value))
                 for key, value in (card.rows or ()))
    actions = tuple((action.label, action.data, action.url,
                     action.intent)
                    for action in (card.actions or ()))
    return (card.title, card.body, rows, card.status, actions)


class TableCardTicker:
    """Revises one chat's live table card on a clock.

    The table sink only fires on game events — a live table's
    countdown is otherwise frozen between moves.  The ticker
    re-runs the EXISTING sink on a slow clock (``TABLE_TICK_SECONDS``)
    while the room is live, and stops the moment the room closes,
    times out, or the engine can no longer find it — no second card
    path, no edit when the card is unchanged, no daemon left
    running after the table is gone.
    """

    def __init__(self, engine: Any, sink: Any, host: Any,
                 chat_key: str, *,
                 interval: float = TABLE_TICK_SECONDS,
                 on_stop: Any = None) -> None:
        self._engine = engine
        self._sink = sink
        self._host = host
        self._chat_key = str(chat_key)
        self._interval = (float(interval) if interval
                          else TABLE_TICK_SECONDS)
        self._on_stop = on_stop
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def tick(self) -> bool:
        """One clock pass.  Returns False when the table is gone
        (stop the thread); True when the ticker should keep going."""
        room = self._engine.live(self._chat_key)
        if room is None:
            return False
        try:
            card = build_table_card(self._engine, room)
        except Exception as exc:  # noqa: BLE001 - a bad tick never kills the clock
            _log.debug("table tick build failed for %s: %s",
                       self._chat_key, exc)
            return True
        stored = None
        if self._host is not None:
            stored = self._host.store.get(card.card_id)
        if stored is not None and \
                _card_fingerprint(stored) == _card_fingerprint(card):
            return True
        try:
            self._sink(room, "tick")
        except Exception as exc:  # noqa: BLE001 - a bad tick never kills the clock
            _log.debug("table tick sink failed for %s: %s",
                       self._chat_key, exc)
        return True

    def start(self) -> threading.Thread:
        if self._thread is not None and self._thread.is_alive():
            return self._thread
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"table-tick-{self._chat_key}",
            daemon=True)
        self._thread.start()
        return self._thread

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                if not self.tick():
                    break
            except Exception as exc:  # noqa: BLE001 - the clock never dies on a tick
                _log.debug("table ticker failed for %s: %s",
                           self._chat_key, exc)
        if self._on_stop is not None:
            with contextlib.suppress(Exception):
                self._on_stop(self)

    def stop(self) -> None:
        self._stop.set()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
