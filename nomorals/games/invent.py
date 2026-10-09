"""Game invention: Devon designs new games on the fly.

An invented game is a real MultiGame — it goes through the same engine,
the same rooms, the same turns as every built-in game. The difference is
the referee: instead of hardcoded rules, an InventedGame carries a design
(the LLM wrote it) and the model adjudicates each move against that design.

Design schema (JSON):
    {
        "name": "slug",
        "title": "Display Title",
        "blurb": "one-line pitch",
        "min_players": 2, "max_players": 6,
        "setup": "what the opening state looks like",
        "turn": "what a player does on their turn",
        "win": "how someone wins",
        "examples": ["example move -> example outcome", ...]
    }

The design is validated (all fields present, sane player counts) and the
game is registered on the engine so it plays like a first-class citizen.
Designs persist to JSON so invented games survive restarts and can be
replayed, refined, and shared.
"""
from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .games.base import MultiGame, Room

_log = get_logger(__name__)

__all__ = [
    "DESIGN_REQUIRED",
    "InventedGame",
    "validate_design",
    "invention_dir",
    "save_design",
    "load_designs",
    "design_game",
]

DESIGN_REQUIRED = (
    "name", "title", "blurb", "min_players", "max_players",
    "setup", "turn", "win",
)

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_]{1,30}$")


def invention_dir() -> Path:
    p = Path(__file__).resolve().parent / "data" / "invented"
    p.mkdir(parents=True, exist_ok=True)
    return p


def validate_design(design: dict[str, Any]) -> list[str]:
    """Return a list of problems (empty = valid)."""
    problems: list[str] = []
    for field in DESIGN_REQUIRED:
        if not design.get(field):
            problems.append(f"missing {field!r}")
    name = str(design.get("name", ""))
    if name and not _SLUG_RE.match(name):
        problems.append("name must be a lowercase slug (letters, digits, _)")
    try:
        lo = int(design.get("min_players", 0))
        hi = int(design.get("max_players", 0))
        if lo < 1:
            problems.append("min_players must be >= 1")
        if hi < lo:
            problems.append("max_players must be >= min_players")
        if hi > 12:
            problems.append("max_players capped at 12")
    except (TypeError, ValueError):
        problems.append("player counts must be integers")
    return problems


def save_design(design: dict[str, Any]) -> Path:
    design = dict(design)
    design["invented_at"] = time.time()
    p = invention_dir() / f"{design['name']}.json"
    p.write_text(json.dumps(design, indent=1, ensure_ascii=False), encoding="utf-8")
    return p


def load_designs() -> list[dict[str, Any]]:
    out = []
    for p in sorted(invention_dir().glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception as exc:  # noqa: BLE001 - one bad file can't kill the list
            _log.warning("invent: bad design file %s: %s", p.name, exc)
    return out


class InventedGame(MultiGame):
    """A game designed at runtime. The model is the referee.

    State tracks: round number, per-player scores/notes, and a free-form
    ``situation`` the referee updates each turn. Move adjudication asks the
    model: given the design, the situation, and the player's move text, what
    happens? The answer is parsed as JSON {outcome, situation, scores, winner}.
    When no model is connected, a seeded fallback keeps the game moving with
    honest, design-shaped narration instead of stalling.
    """

    name = "invented"
    title = "Invented Game"

    def __init__(self, design: dict[str, Any], suggest: Any = None):
        problems = validate_design(design)
        if problems:
            raise ValueError("bad design: " + "; ".join(problems))
        self.design = dict(design)
        self.name = self.design["name"]
        self.title = self.design["title"]
        self._suggest = suggest

    # ── MultiGame surface ──────────────────────────────────────────
    @property
    def min_players(self) -> int:
        return int(self.design["min_players"])

    @property
    def max_players(self) -> int:
        return int(self.design["max_players"])

    def new_state(self, rng: random.Random, **kw: Any) -> dict[str, Any]:
        return {
            "round": 1,
            "situation": self.design["setup"],
            "scores": {},
            "log": [],
            "winner": None,
        }

    def setup(self, room: Room, mind: Any) -> str:
        d = self.design
        lines = [
            f"🎲 **{d['title']}** — invented just now",
            f"_{d['blurb']}_",
            "",
            f"👥 {d['min_players']}–{d['max_players']} players",
            f"🏁 Win: {d['win']}",
            "",
            f"📜 On your turn: {d['turn']}",
            "",
            f"🌍 {room.state['situation']}",
        ]
        return "\n".join(lines)

    def describe_options(self, room: Room) -> list[str]:
        return [f"any move fitting: {self.design['turn']}"]

    def on_move(self, room: Room, player: Any, text: str, mind: Any) -> list[str]:
        d = self.design
        st = room.state
        outcome = self._referee(room, player, text)
        st["log"].append(f"{player.name}: {text} → {outcome['outcome']}")
        st["situation"] = outcome.get("situation", st["situation"])
        for k, v in (outcome.get("scores") or {}).items():
            st["scores"][k] = v
        msgs = [f"🎲 {outcome['outcome']}"]
        if outcome.get("situation"):
            msgs.append(f"🌍 {outcome['situation']}")
        winner = outcome.get("winner")
        if winner:
            st["winner"] = winner
            room.status = "finished"
            msgs.append(f"🏆 **{winner} wins {d['title']}!**")
        else:
            st["round"] += 1
            room.advance_turn()
        return msgs

    # ── refereeing ─────────────────────────────────────────────────
    def _referee(self, room: Room, player: Any, text: str) -> dict[str, Any]:
        if self._suggest is not None:
            try:
                return self._llm_referee(room, player, text)
            except Exception as exc:  # noqa: BLE001
                _log.warning("invent: llm referee failed: %s", exc)
        return self._fallback_referee(room, player, text)

    def _llm_referee(self, room: Room, player: Any, text: str) -> dict[str, Any]:
        d = self.design
        st = room.state
        prompt = (
            "You are the referee of an invented party game. Adjudicate fairly "
            "and stay in the game's spirit.\n"
            f"Game: {d['title']} — {d['blurb']}\n"
            f"Turn rule: {d['turn']}\nWin rule: {d['win']}\n"
            f"Round {st['round']}. Situation: {st['situation']}\n"
            f"Scores: {json.dumps(st['scores'])}\n"
            f"Players: {', '.join(p.name for p in room.players)}\n"
            f"{player.name} plays: {text}\n"
            "Reply ONLY as JSON: {\"outcome\": \"what happened (1-2 sentences)\", "
            "\"situation\": \"new situation\", \"scores\": {\"name\": n}, "
            "\"winner\": \"name or null\"}"
        )
        raw = self._suggest(prompt) if callable(self._suggest) else ""
        m = re.search(r"\{.*\}", str(raw), re.S)
        if not m:
            raise ValueError("referee returned no JSON")
        data = json.loads(m.group(0))
        return {
            "outcome": str(data.get("outcome", "the move stands.")),
            "situation": str(data.get("situation", st["situation"])),
            "scores": dict(data.get("scores") or {}),
            "winner": data.get("winner"),
        }

    def _fallback_referee(self, room: Room, player: Any, text: str) -> dict[str, Any]:
        # Honest seeded fallback: acknowledges the move, nudges the
        # situation forward, never pretends the LLM judged it.
        rng = random.Random(hash((self.name, room.id, text)) & 0xFFFFFFFF)
        st = room.state
        scores = dict(st["scores"])
        scores[player.name] = scores.get(player.name, 0) + rng.randint(1, 3)
        return {
            "outcome": f"{player.name}'s move lands (+{scores[player.name]}).",
            "situation": st["situation"],
            "scores": scores,
            "winner": None,
        }


def design_game(theme: str, suggest: Any, n_players: int = 4) -> dict[str, Any]:
    """Ask the model to design a game around a theme. Returns the design.

    Raises RuntimeError when no model is available — invention needs a brain.
    """
    if suggest is None:
        raise RuntimeError("game invention needs a connected model")
    prompt = (
        "Invent a fun chat-playable party game. "
        f"Theme: {theme}. Players: about {n_players}.\n"
        "Reply ONLY as JSON with keys: name (lowercase slug), title, blurb "
        "(one line), min_players, max_players, setup (opening situation), "
        "turn (what a player does on their turn), win (how someone wins), "
        "examples (2 short 'move -> outcome' strings)."
    )
    raw = suggest(prompt) if callable(suggest) else suggest
    m = re.search(r"\{.*\}", str(raw), re.S)
    if not m:
        raise RuntimeError("the model didn't return a game design")
    design = json.loads(m.group(0))
    problems = validate_design(design)
    if problems:
        raise RuntimeError("bad design from model: " + "; ".join(problems))
    return design
