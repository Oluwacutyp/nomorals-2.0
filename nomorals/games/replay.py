"""Replays from the real record.

A finished game's replay is assembled ONLY from what the game
actually recorded: the engine's event journal (moves, passes,
stalls, timeouts, forfeits, resigns — persisted inside the room's
state), the settlement receipts (the result and the exact money),
the final board, and the fairness fragment when the table was
provably fair. Nothing is narrated into existence: a game with no
journal replays as its result and its settle, plainly.
"""
from __future__ import annotations

import json
from typing import Any

from .boards import render_board
from .games.base import Room

__all__ = ["build_replay"]

#: clock-event kind → surface glyph
_CLOCK_GLYPHS = {
    "timeout": "⏰", "stall": "⏳", "pass": "⏭",
    "forfeit": "🏳", "resign": "🏳",
}

#: journal-event kind → surface glyph
_EVENT_GLYPHS = {
    "move": "▸", "pass": "⏭", "resign": "🏳", "forfeit": "🏳",
}

_MAX_EVENTS = 24


def _names(room: Room) -> dict[str, str]:
    return {p.key: p.name for p in room.players}


def build_replay(row: dict[str, Any],
                 settlement: dict[str, Any] | None) -> str:
    """Render one finished room row (+ its settlement record when
    the ledger has one) as a replay/highlight summary."""
    room = Room.from_row(row, {})
    state = room.state or {}
    names = _names(room)
    lines = [f"🎬 replay — {room.game}"]
    seated = " · ".join(p.name for p in room.players)
    if seated:
        lines.append(f"at the table: {seated}")

    receipts = (settlement or {}).get("receipts") or {}
    players_rec = receipts.get("players") or {}

    # the result, from the records that decided it
    decided = state.get("decided") or {}
    winners = [str(rec.get("name") or names.get(key, key))
               for key, rec in players_rec.items()
               if rec.get("won") is True]
    if winners:
        lines.append(f"🏁 {', '.join(winners)} won")
    elif decided.get("how"):
        loser = names.get(str(decided.get("loser") or ""), "")
        lines.append(
            f"🏁 decided by {decided['how']}"
            + (f" — {loser} lost" if loser else ""))
    elif players_rec:
        lines.append("🏁 settled — no winner (draw)")

    # the events, in the order they happened
    events: list[tuple[float, str]] = []
    for entry in state.get("_events") or []:
        if not isinstance(entry, dict):
            continue
        kind = str(entry.get("kind") or "")
        glyph = _EVENT_GLYPHS.get(kind, "▸")
        who = str(entry.get("name") or "")
        text = str(entry.get("text") or "")
        if kind == "move":
            line = f"{glyph} {who}: {text}"
        elif kind in ("pass", "resign"):
            line = f"{glyph} {who} {kind}s"
        else:
            line = f"{glyph} {who} {kind} {text}".strip()
        events.append((float(entry.get("ts") or 0), line))
    for entry in state.get("_clock_events") or []:
        if not isinstance(entry, dict):
            continue
        kind = str(entry.get("event") or "")
        glyph = _CLOCK_GLYPHS.get(kind, "⏰")
        who = names.get(str(entry.get("player") or ""),
                        str(entry.get("player") or ""))
        n = entry.get("n")
        extra = f" ({n})" if kind == "stall" and n else ""
        events.append((float(entry.get("ts") or 0),
                       f"{glyph} {who} — {kind}{extra}"))
    events.sort(key=lambda e: e[0])
    if events:
        lines.append("highlights:")
        shown = events[:_MAX_EVENTS]
        lines.extend(f"  {line}" for _, line in shown)
        if len(events) > len(shown):
            lines.append(f"  … (+{len(events) - len(shown)} more)")

    board = render_board(room.game, state)
    if board:
        lines.append("final board:")
        lines.append(board)

    # the settle, from the receipts — the exact money that moved
    for key, rec in players_rec.items():
        coins = int(rec.get("coins") or 0)
        lines.append(
            f"🪙 {rec.get('name') or names.get(key, key)}: "
            f"{coins:+} coins ({rec.get('why') or 'settled'})")

    if state.get("fair") is not None:
        lines.append(
            "🎲 provably fair table — /game fair reveals the seed")
    return "\n".join(lines)


def replay_from_json(state_json: str, settlement_json: str = "") -> str:
    """Convenience for tools: replay from raw stored JSON blobs."""
    try:
        state = json.loads(state_json or "{}")
    except (TypeError, ValueError):
        state = {}
    rec = None
    if settlement_json:
        try:
            rec = {"receipts": json.loads(settlement_json)}
        except (TypeError, ValueError):
            rec = None
    row = {"id": "", "game": "", "chat_key": "", "platform": "local",
           "kind": "dm", "players": "[]", "turn": 0,
           "state": json.dumps(state), "status": "finished"}
    return build_replay(row, rec)
