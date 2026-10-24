"""Agent matches: the brain and characters as real game players.

Seats character agents (and Devon herself) at a real game table and
plays through the actual engine — real state, real turns, real winner.
No fake turns: every move goes through the game's on_move, the same
path human moves take.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

from .character import Character, SuggestFn
from .failures import record_failure

CHAR_KEY_PREFIX = "char:"
BRAIN_KEY = "brain:devon"

__all__ = [
    "CHAR_KEY_PREFIX", "BRAIN_KEY", "AgentSeat", "MatchResult",
    "run_agent_match", "match_intro",
]


def match_intro(game_name: str, seats: list[AgentSeat],
                suggest: SuggestFn | None,
                graph: Any = None,
                rng: Any = None) -> list[str]:
    """Game-night intro: every seat gets a one-liner in character, and
    real rivalries get called out. Game night has culture — the table
    should feel like one before the first move.

    Rivalry comes from the relationship graph (kind == "rival" or high
    friction), not from vibes — callouts are earned.
    """
    r = rng or random.Random()
    lines = [f"🎮 {game_name} — seats taken:"]
    chars = [s.character for s in seats
             if s.kind == "character" and s.character]
    for s in seats:
        if s.kind == "brain":
            lines.append("  Devon cracks her knuckles. \"Let's play.\"")
        elif s.character:
            c = s.character
            try:
                one = c.speak(
                    f"You're sitting down to play {game_name} with "
                    f"{', '.join(x.display for x in seats if x.display != c.name)}. "
                    f"Say ONE cocky line as you take your seat. One sentence.",
                    suggest)
            except Exception as e:
                record_failure("match.match_intro:speak", e)
                one = c._fallback_line(f"ready for {game_name}")
            lines.append(f"  {c.name}: \"{one[:140]}\"")
    # rivalry callouts — earned from the graph
    if graph is not None and len(chars) >= 2:
        for i, a in enumerate(chars):
            for b in chars[i + 1:]:
                try:
                    e = graph.edge(a.id, b.id)
                    kind = getattr(e, "kind", "")
                    fric = float((getattr(e, "dims", None) or {})
                                     .get("friction", 0.0))
                    if kind == "rival" or fric > 0.6:
                        pick = r.choice([
                            f"⚔️ Grudge match: {a.name} vs {b.name} — "
                            f"the table goes quiet.",
                            f"⚔️ {a.name} and {b.name} have history. "
                            f"This one's personal.",
                            f"⚔️ All eyes on {a.name} vs {b.name}.",
                        ])
                        lines.append("  " + pick)
                except Exception as e:
                    record_failure("match.match_intro:edge", e)
                    continue
    return lines


@dataclass
class AgentSeat:
    kind: str            # "brain" | "character"
    character: Character | None = None
    name: str = ""

    @property
    def key(self) -> str:
        if self.kind == "brain":
            return BRAIN_KEY
        return f"{CHAR_KEY_PREFIX}{self.character.id}" if self.character else "char:?"

    @property
    def display(self) -> str:
        if self.kind == "brain":
            return "Devon"
        return self.character.name if self.character else "?"


@dataclass
class MatchResult:
    game: str
    winner: str
    transcript: list[str] = field(default_factory=list)
    moves: int = 0
    duration_s: float = 0.0


def _brain_decide(game_name: str, rules: str, state_text: str,
                  options: list[str], suggest: SuggestFn | None,
                  rng: random.Random) -> str:
    prompt = (
        "You are Devon, playing a game as yourself.\n"
        f"Game: {game_name}\nRules: {rules}\n"
        f"Current state: {state_text}\n"
        f"Your legal options:\n" + "\n".join(f"- {o}" for o in options) + "\n"
        "Reply with ONLY your chosen option, exactly as written. "
        "Play to win.")
    choice = ""
    if suggest is not None:
        try:
            choice = (suggest(prompt) or "").strip().strip('"').strip()
        except Exception as e:
            record_failure("match._brain_decide:strip", e)
            choice = ""
    if choice:
        low = choice.lower()
        for o in options:
            if o.lower() == low or o.lower() in low or low in o.lower():
                return o
    return rng.choice(options) if options else ""


def _grade_match_decides(seats: list[AgentSeat], winner_obj: Any,
                         game_name: str) -> None:
    """Grade character seats' journaled decide moves from the match
    outcome. The winner OBJECT decides, never the winner NAME:
    Player keys are exact (``char:{id}`` / ``brain:devon``), names
    are not (two seats can share one).

    - winner Player with a ``char:`` key → that seat won, the other
      character seats lost; ``brain:devon`` key → every character
      seat lost; string ``"draw"`` → every character seat drew.
    - anything else (None, keyless object, non-draw string, a
      ``char:`` key with no matching seat) → SKIP grading entirely.
      Entries stay ungraded; never fake a grade from an ambiguous
      winner. Grading failure per character is recorded, never
      raised — it must not eat the MatchResult."""
    from .learn import get_handle

    chars = [s.character for s in seats
             if s.kind == "character" and s.character is not None]
    if not chars:
        return
    if isinstance(winner_obj, str):
        if winner_obj != "draw":
            return
        outcomes = {id(c): "draw" for c in chars}
    else:
        key = getattr(winner_obj, "key", None)
        if not key:
            return
        if key == BRAIN_KEY:
            outcomes = {id(c): "lost" for c in chars}
        elif key.startswith(CHAR_KEY_PREFIX):
            won_id = key[len(CHAR_KEY_PREFIX):]
            won = [c for c in chars if c.id == won_id]
            if len(won) != 1:
                return
            outcomes = {id(c): ("won" if c is won[0] else "lost")
                        for c in chars}
        else:
            return
    for c in chars:
        try:
            handle = get_handle(c)
            if handle is None:
                continue
            for eid in handle.drain_decide_ids():
                try:
                    handle.grade_decide(
                        eid, outcomes[id(c)], source="system",
                        note=f"match:{game_name}:{outcomes[id(c)]}")
                except Exception as e:
                    record_failure("match.run_agent_match:grade_decide", e)
        except Exception as e:
            record_failure("match.run_agent_match:grade_decide", e)


def run_agent_match(game_name: str, seats: list[AgentSeat],
                    suggest: SuggestFn | None,
                    seed: int | None = None,
                    store: Any = None,
                    max_turns: int = 200,
                    context: Any = None,
                    graph: Any = None) -> MatchResult:
    """Play a full game with agent seats. Returns the result + transcript."""
    from ..games import engine as engine_mod
    from ..games.players import Player

    t0 = time.time()
    rng = random.Random(seed)

    class _Ctx:
        db = None

    eng = engine_mod.GameEngine(context or _Ctx(), suggest=suggest)
    game = eng.games.get(game_name)
    if game is None:
        raise ValueError(f"unknown game: {game_name}")

    chat_key = f"local:agent-match-{rng.randrange(10**9)}"
    players = [Player(key=s.key, platform="ai", name=s.display, is_ai=True)
               for s in seats]
    from ..games.games.base import Room
    import uuid as _uuid
    room = Room(id=_uuid.uuid4().hex[:12], game=game.name,
                chat_key=chat_key, platform="local", kind="dm",
                players=players, seed=seed or 0)
    # bypass human requirements — all seats are agents
    room.status = "active"
    room.state = game.new_state(room.rng())
    try:
        transcript: list[str] = match_intro(
            game.name, seats, suggest, graph=graph, rng=rng)
    except Exception as e:
        record_failure("match.run_agent_match:match_intro", e)
        transcript = []
    transcript.append(game.setup(room, eng._mind))
    seat_by_key = {s.key: s for s in seats}
    moves = 0

    for _ in range(max_turns):
        if eng.is_over(room):
            break
        cur = room.current
        if cur is None:
            break
        seat = seat_by_key.get(cur.key)
        options = game.describe_options(room)
        state_text = ""
        try:
            state_text = game.describe_state(room) or ""
        except Exception as e:
            record_failure("match.run_agent_match:describe_state", e)
        if seat is None or not options:
            # house / unstructured seat: built-in brain
            out = game.ai_turn(room, eng._mind)
        elif seat.kind == "brain":
            move = _brain_decide(game.name, game.rules, state_text,
                                 options, suggest, rng)
            out = game.on_move(room, cur, move, eng._mind)
            out = [f"Devon plays: {move}"] + list(out or [])
        else:
            char = seat.character
            move = char.decide(game.name, game.rules, state_text,
                               options, suggest, room.rng())
            out = game.on_move(room, cur, move, eng._mind)
            out = [f"{char.name} plays: {move}"] + list(out or [])
            char.remember(
                f"Played {game.name}: I played '{move}' (state: {state_text[:150]})",
                0.4)
            if store is not None:
                try:
                    store.save(char)
                except Exception as e:
                    record_failure("match.run_agent_match:save", e)
        transcript.extend(out or [])
        moves += 1
        room.advance_turn()

    winner = eng.winner(room) if hasattr(eng, "winner") else None
    winner_obj: Any = None
    try:
        w = game.winner(room)
        winner_obj = w
        winner_name = w.name if hasattr(w, "name") else str(w)
    except Exception as e:
        record_failure("match.run_agent_match:winner", e)
        winner_name = str(winner) if winner else "?"
    transcript.append(f"🏁 winner: {winner_name}")
    # decide journaling: grade the match's moves from the outcome
    # (never from the name — see _grade_match_decides).
    _grade_match_decides(seats, winner_obj, game_name)
    return MatchResult(game=game_name, winner=winner_name,
                       transcript=transcript, moves=moves,
                       duration_s=time.time() - t0)


def character_seat_decider(store: Any, suggest: SuggestFn | None):
    """Build an engine seat-decider for character/brain seats.

    Register on ``engine.seat_deciders`` under the ``char:`` and
    ``brain:`` prefixes so character agents play through the live
    engine in chat games, not just standalone matches.
    """
    def decider(room: Any, player: Any, game: Any,
                mind: Any) -> list[str] | None:
        key = player.key or ""
        options = game.describe_options(room)
        try:
            state_text = game.describe_state(room) or ""
        except Exception as e:
            record_failure("match.decider:describe_state", e)
            state_text = ""
        if key.startswith(BRAIN_KEY):
            if not options:
                return None
            move = _brain_decide(game.name, game.rules, state_text,
                                 options, suggest, room.rng())
            return [f"Devon plays: {move}"] + list(
                game.on_move(room, player, move, mind) or [])
        if key.startswith(CHAR_KEY_PREFIX):
            char_id = key[len(CHAR_KEY_PREFIX):]
            char = store.get(char_id) if store else None
            if char is None or not options:
                return None
            move = char.decide(game.name, game.rules, state_text,
                               options, suggest, room.rng())
            out = [f"{char.name} plays: {move}"] + list(
                game.on_move(room, player, move, mind) or [])
            char.remember(
                f"Played {game.name}: I played '{move}'", 0.4)
            try:
                store.save(char)
            except Exception as e:
                record_failure("match.decider:save", e)
            return out
        return None
    return decider


def register_with_engine(engine: Any, store: Any,
                         suggest: SuggestFn | None) -> None:
    """Wire character/brain seats into a live GameEngine."""
    decider = character_seat_decider(store, suggest)
    engine.seat_deciders[CHAR_KEY_PREFIX] = decider
    engine.seat_deciders[BRAIN_KEY] = decider
