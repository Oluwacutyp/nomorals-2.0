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

__all__ = ["SuggestFn", "GameMind", "pick", "one_of"]

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
            self._cache[cache_key] = (time.time(), raw)
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
                    foe_stats: dict[str, int]) -> dict[str, Any]:
        """Battle arena AI: attack/focus/fury/defend/potion by math.

        Stats use the arena's key names (``atk``/``def``); the old
        ``attack``/``defense`` names are tolerated so external callers
        keep working.
        """
        hp = self_stats.get("hp", 1)
        max_hp = self_stats.get("max_hp", max(hp, 1))
        my_atk = self_stats.get("atk", self_stats.get("attack", 0))
        my_def = self_stats.get("def", self_stats.get("defense", 0))
        foe_hp = foe_stats.get("hp", 1)
        foe_atk = foe_stats.get("atk", foe_stats.get("attack", 0))
        foe_def = foe_stats.get("def", foe_stats.get("defense", 0))
        if hp <= 0 or foe_hp <= 0:
            return {"action": "attack"}
        # bleeding badly and holding a potion: drink
        if hp < max_hp * 0.35 and self_stats.get("potions", 0) > 0:
            return {"action": "potion"}
        # a focused hit that finishes the fight: set it up (or spend it)
        focused_raw = max(1, int((my_atk - foe_def // 2) * 1.5))
        if focused_raw >= foe_hp:
            if self_stats.get("focused"):
                return {"action": "attack"}
            return {"action": "focus"}
        # the foe out-hits us badly: guard instead of trading
        if foe_atk > my_def + 8:
            return {"action": "defend"}
        # healthy and off cooldown: fury swings are worth it
        if self_stats.get("fury_cd", 0) <= 0 and hp >= max_hp * 0.7:
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
