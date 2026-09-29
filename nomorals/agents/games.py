"""Games module: social games the companion plays IN the chat.

Each game is a self-contained sub-agent (a ``Game``) with pure state logic —
no I/O inside a move, so every game is unit-testable with a fixed RNG. The
``GamesAgent`` owns persistence (``game_sessions``) and routing: while a game
is live in a chat, that chat's messages ARE game moves (``/game quit`` or a
finished round hands the chat back to her).

Games ship: trivia, 20 questions, word chain, rock-paper-scissors,
would-you-rather, number memory. Model-driven moves fall back to built-in
banks when no live model is answering, so games stay playable offline.
"""

from __future__ import annotations

import json
import random
import time
from typing import Any, Callable

from ..core.ids import new_id

__all__ = ["GAMES", "GamesAgent", "Game"]

SuggestFn = Callable[[str], str]  # (prompt) -> model text, may return ""


# ── individual games (sub-agents) ────────────────────────────────────────────


class Game:
    name: str = "base"
    description: str = ""

    def new_state(self) -> dict[str, Any]:
        return {}

    def intro(self, state: dict[str, Any], rng: random.Random) -> str:
        return "let's play."

    def move(
        self,
        state: dict[str, Any],
        text: str,
        rng: random.Random,
        suggest: SuggestFn | None,
    ) -> tuple[str, dict[str, Any], str]:
        """One move. Returns (reply, new_state, status)."""
        return "…", state, "active"


class TriviaGame(Game):
    name = "trivia"
    description = "I ask, you answer — 5 rounds, 3 lives"

    QUESTIONS: tuple[tuple[str, str], ...] = (
        ("What year did the first iPhone launch?", "2007"),
        ("Which planet has the most moons (2026 count, ~165)?", "Saturn"),
        ("What does 'HTTP' stand for?", "HyperText Transfer Protocol"),
        ("In which country is the city of Enugu?", "Nigeria"),
        ("What gas do plants absorb from the atmosphere?", "Carbon dioxide"),
        ("Which language runs natively in most web browsers?", "JavaScript"),
        ("How many bits are in a byte?", "8"),
        ("What is the capital of Canada?", "Ottawa"),
        ("Which element has the chemical symbol 'Au'?", "Gold"),
        ("Who wrote '1984'?", "George Orwell"),
        ("What does 'CPU' stand for?", "Central Processing Unit"),
        ("Largest ocean on Earth?", "Pacific"),
        ("What year did World War II end?", "1945"),
        ("Which protocol is port 443 for?", "HTTPS"),
        ("How many time zones does China officially use?", "1"),
        ("What is the speed of light, roughly?", "300,000 km/s"),
    )

    def new_state(self) -> dict[str, Any]:
        return {"round": 0, "score": 0, "lives": 3, "question": None, "answer": None}

    def intro(self, state: dict[str, Any], rng: random.Random) -> str:
        q, a = self._next(state, rng)
        return (
            "trivia — 5 rounds, 3 lives. answer like a human, not a search bar.\n"
            f"round 1/5: {q}"
        )

    def _next(self, state: dict[str, Any], rng: random.Random) -> tuple[str, str]:
        q, a = rng.choice(self.QUESTIONS)
        state["question"], state["answer"] = q, a
        return q, a

    def move(self, state, text, rng, suggest):
        if state.get("question") is None:
            q, _ = self._next(state, rng)
            return f"round {state['round']+1}/5: {q}", state, "active"
        guess = text.strip().lower()
        answer = str(state["answer"]).lower()
        right = answer in guess or guess in answer
        state["round"] += 1
        if right:
            state["score"] += 1
        else:
            state["lives"] -= 1
        if state["round"] >= 5 or state["lives"] <= 0:
            status = "won" if state["score"] >= 3 else "ended"
            return (
                f"done. you scored {state['score']}/5 "
                + ("— sharp work." if state["score"] >= 3 else "— i'm keeping this one in."),
                state, status,
            )
        q, _ = self._next(state, rng)
        extra = "" if right else f" (it was {state['answer']})"
        lives = "" if state["lives"] == 3 else f"  ·  {state['lives']} lives left"
        return f"{'correct.' if right else 'nope.'}{extra}  round {state['round']+1}/5: {q}{lives}", state, "active"


class TwentyQuestionsGame(Game):
    name = "20q"
    description = "i think of something, you ask yes/no questions (20)"

    BANK: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("a houseplant", ("plant", "leaf", "pot", "water", "green")),
        ("a mechanical keyboard", ("key", "click", "desk", "switch", "clack")),
        ("a thunderstorm", ("rain", "lightning", "loud", "sky", "weather")),
        ("a cat", ("fur", "meow", "pet", "paw", "milk")),
        ("a submarine", ("water", "deep", "metal", "ocean", "silent")),
        ("a library", ("book", "quiet", "shelf", "read", "card")),
    )

    def new_state(self) -> dict[str, Any]:
        return {"target": None, "hints": (), "asked": 0, "max": 20}

    def intro(self, state: dict[str, Any], rng: random.Random) -> str:
        state["target"], state["hints"] = rng.choice(self.BANK)
        return (
            f"20 questions — i'm thinking of something. you have {state['max']} "
            "yes/no questions. start anywhere."
        )

    def move(self, state, text, rng, suggest):
        if state.get("target") is None:
            return self.intro(state, rng), state, "active"
        state["asked"] += 1
        word = text.strip().lower().rstrip("?")
        hint = self._answer(word, state, suggest, rng)
        left = state["max"] - state["asked"]
        if "guess:" in word or word.startswith(("it's ", "its ", "its ")):
            guess = word.split(":", 1)[-1] if ":" in word else word[4:]
            if state["target"] in guess:
                return f"guessed it in {state['asked']} questions — {state['target']}. well played.", state, "won"
            return f"not that one. {left} questions left.", state, "active" if left else "lost"
        if left <= 0:
            return f"out of questions. it was {state['target']}. rematch: /game 20q", state, "lost"
        return f"{hint}  ({left} left)", state, "active"

    def _answer(self, word: str, state: dict, suggest: SuggestFn | None, rng: random.Random) -> str:
        for hint in state["hints"]:
            if hint in word:
                return "yes"
        for hint in state["hints"]:
            if word in hint:
                return "yes"
        if suggest is not None:
            try:
                reply = suggest(
                    f"You are thinking of: {state['target']} (hints: {', '.join(state['hints'])}). "
                    f"The player asked: '{word}'. Reply with only 'yes' or 'no'."
                ).strip().lower()
                if reply.startswith("yes"):
                    return "yes"
                if reply.startswith("no"):
                    return "no"
            except Exception:  # noqa: BLE001
                pass
        return "no"


class WordChainGame(Game):
    name = "wordchain"
    description = "each word must start with the last letter of the previous one"

    BANK: tuple[str, ...] = (
        "apple", "elephant", "tiger", "rabbit", "tiger", "rain", "night",
        "train", "ink", "kite", "echo", "orange", "egg", "ghost", "house",
        "engine", "elephant", "piano", "owl", "wave", "egg", "garden", "noodle",
    )

    def new_state(self) -> dict[str, Any]:
        return {"last": "", "score": 0, "stuck": 0}

    def intro(self, state: dict[str, Any], rng: random.Random) -> str:
        state["last"] = "city"
        return "word chain — my word is CITY. end with a Y… go."

    def _my_word(self, letter: str, suggest: SuggestFn | None, rng: random.Random) -> str:
        if suggest is not None:
            try:
                reply = suggest(
                    f"Word chain game. The word must start with the letter '{letter}'. "
                    "Reply with ONE common English word only, no punctuation."
                ).strip().lower().strip(" .!?\n")
                if reply.isalpha() and reply.startswith(letter) and 3 <= len(reply) <= 15:
                    return reply
            except Exception:  # noqa: BLE001
                pass
        matches = [w for w in self.BANK if w.startswith(letter)]
        if matches:
            return rng.choice(matches)
        return ""

    def move(self, state, text, rng, suggest):
        word = "".join(ch for ch in text.strip().lower() if ch.isalpha())
        if not word:
            return "one word at a time.", state, "active"
        if not state.get("last"):
            state["last"] = word
        elif not word.startswith(state["last"][-1]):
            state["stuck"] += 1
            if state["stuck"] >= 3:
                return f"three misses — you lose. i win {state['score']}-0 by forfeit.", state, "lost"
            return f"has to start with “{state['last'][-1]}” — the last letter of {state['last']}.", state, "active"
        state["score"] += 1
        state["stuck"] = 0
        mine = self._my_word(word[-1], suggest, rng)
        if not mine:
            state["last"] = word
            return (
                f"ok… {word}. i'm genuinely stuck on “{word[-1]}” — "
                "you win this round by default.", state, "won"
            )
        state["last"] = mine
        return f"good. mine: {mine}. your turn — “{mine[-1]}”.", state, "active"


class RpsGame(Game):
    name = "rps"
    description = "rock paper scissors — first to 3"

    MOVES = ("rock", "paper", "scissors")
    BEATS = {"rock": "scissors", "scissors": "paper", "paper": "rock"}

    def new_state(self) -> dict[str, Any]:
        return {"me": 0, "you": 0}

    def intro(self, state: dict[str, Any], rng: random.Random) -> str:
        return "rock paper scissors, first to 3. type rock, paper or scissors."

    def move(self, state, text, rng, suggest):
        guess = text.strip().lower()
        if guess not in self.MOVES:
            return "rock, paper, or scissors — that's the whole menu.", state, "active"
        mine = rng.choice(self.MOVES)
        if mine == guess:
            reply = f"both {mine}. again."
        elif self.BEATS[mine] == guess:
            state["me"] += 1
            reply = f"i throw {mine}. mine!" if state["me"] < 3 else f"i throw {mine}. three-three, i take it."
        else:
            state["you"] += 1
            reply = f"i throw {mine}. yours."
        if state["me"] >= 3:
            return reply, state, "lost"
        if state["you"] >= 3:
            return f"{reply} you won 3-{state['me']} — respect.", state, "won"
        return reply, state, "active"


class WouldYouRatherGame(Game):
    name = "wyrr"
    description = "would you rather — i pose, you pick, i roast"

    PAIRS: tuple[tuple[str, str, str], ...] = (
        ("Never use the internet again", "Never use a phone again",
         "the internet one. a phone without internet is a very expensive brick with good camera."),
        ("Speak your thoughts out loud always", "Only be able to whisper",
         "whisper. at least nobody can record you complaining in traffic."),
        ("Run every errand yourself", "Someone else runs every errand but narrates it",
         "brutal. the narration one — you'll be asking them to stop by Tuesday."),
        ("Know when someone is lying to you", "Know when you are lying to yourself",
         "the second one. that's the one that actually changes a life."),
        ("One perfect meal every day, same dish", "Different meals, but 30% worse quality",
         "variety. the same perfect meal becomes background noise by week three."),
        ("Answer every DM you send ever gets", "Never answer a single one again",
         "never answer. consider it a boundary you didn't have to learn."),
    )

    def new_state(self) -> dict[str, Any]:
        return {"pair": None, "idx": 0}

    def intro(self, state: dict[str, Any], rng: random.Random) -> str:
        self._next(state, rng)
        a, b, _ = state["pair"]
        return f"would you rather: {a} — OR — {b}?  (answer 1 or 2, or /game more)"

    def _next(self, state: dict[str, Any], rng: random.Random) -> None:
        state["idx"] += 1
        state["pair"] = self.PAIRS[(state["idx"] - 1) % len(self.PAIRS)]

    def move(self, state, text, rng, suggest):
        choice = text.strip().lower()
        if choice in {"more", "next", "/game more"}:
            self._next(state, rng)
            a, b, _ = state["pair"]
            return f"ok fine, next: {a} — OR — {b}?  (1 or 2)", state, "active"
        if state["pair"] is None:
            self._next(state, rng)
            a, b, _ = state["pair"]
            return f"would you rather: {a} — OR — {b}?  (1 or 2)", state, "active"
        a, b, roast = state["pair"]
        if choice in {"1", "one", "a", a[:6]}:
            reply = f"one. {roast}"
        elif choice in {"2", "two", "b", b[:6]}:
            reply = f"two. {roast}"
        else:
            return "pick 1 or 2 — this isn't a debate club.", state, "active"
        self._next(state, rng)
        a2, b2, _ = state["pair"]
        return f"{reply}\n\nnext: {a2} — OR — {b2}?", state, "active"


class NumberMemoryGame(Game):
    name = "memory"
    description = "i read digits, you repeat them — gets longer each round"

    def new_state(self) -> dict[str, Any]:
        return {"round": 0, "number": None, "best": 0}

    def intro(self, state: dict[str, Any], rng: random.Random) -> str:
        state["round"] = 1
        state["number"] = "".join(rng.choice("0123456789") for _ in range(3))
        return f"memory — repeat back: {state['number']}  (3 digits to start, grows each round)"

    def move(self, state, text, rng, suggest):
        guess = "".join(ch for ch in text if ch.isdigit())
        target = str(state.get("number") or "")
        if guess == target:
            state["round"] += 1
            state["best"] = max(state["best"], state["round"] - 1)
            state["number"] = "".join(rng.choice("0123456789") for _ in range(2 + state["round"]))
            if state["round"] > 8:
                return f"eight rounds?! best was {state['best']} — that's a brain, not a phone.", state, "won"
            return f"correct. now: {state['number']}  ({len(state['number'])} digits)"
        return (
            f"that's not it — it was {target}. best round: {state['best']}."
            " rematch: /game memory", state, "ended"
        )


GAMES: dict[str, Game] = {
    g.name: g for g in (
        TriviaGame(), TwentyQuestionsGame(), WordChainGame(),
        RpsGame(), WouldYouRatherGame(), NumberMemoryGame(),
    )
}


# ── the games agent (top-level) ──────────────────────────────────────────────


class GamesAgent:
    """Owns live game sessions and routes chat moves into the right game."""

    def __init__(self, context: Any) -> None:
        self.context = context
        self.db = getattr(context, "db", None)
        self.router = getattr(context, "router", None)
        self.rng = random.Random()

    def _suggest(self) -> SuggestFn | None:
        if self.router is None:
            return None

        def _call(prompt: str) -> str:
            from ..llm.base import Message

            response = self.router.chat([Message(role="user", content=prompt)])
            return getattr(response, "text", "") or ""

        return _call

    # ── sessions ─────────────────────────────────────────────────────────────
    def active(self, chat_key: str) -> dict[str, Any] | None:
        if self.db is None:
            return None
        try:
            row = self.db.query_one(
                "SELECT * FROM game_sessions WHERE chat_key = ? AND status = 'active' "
                "ORDER BY updated_at DESC LIMIT 1", (chat_key,)
            )
        except Exception:  # noqa: BLE001
            return None
        if row is None:
            return None
        try:
            row["state"] = json.loads(row.get("state") or "{}")
        except Exception:  # noqa: BLE001
            row["state"] = {}
        return row

    def _persist(self, session: dict[str, Any], state: dict[str, Any], status: str) -> None:
        if self.db is None:
            return
        try:
            with self.db.transaction():
                self.db.execute(
                    "INSERT INTO game_sessions (id, game, chat_key, state, status, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET state = excluded.state, status = excluded.status, "
                    "updated_at = excluded.updated_at",
                    (session["id"], session["game"], session["chat_key"],
                     json.dumps(state), status, session["created_at"], time.time()),
                )
        except Exception:  # noqa: BLE001
            pass

    # ── API used by the chat dispatch ────────────────────────────────────────
    def list_games(self) -> str:
        lines = ["games — start one with /game <name>:"]
        for name, game in GAMES.items():
            lines.append(f"  /game {name:<11} {game.description}")
        lines.append("  /game quit        leave the current game")
        return "\n".join(lines)

    def begin(self, chat_key: str, game_name: str) -> str:
        game = GAMES.get(game_name.strip().lower())
        if game is None:
            return self.list_games()
        state = game.new_state()
        rng = self.rng
        intro = game.intro(state, rng)
        session_id = new_id()
        self._persist(
            {"id": session_id, "game": game.name, "chat_key": chat_key, "created_at": time.time()},
            state, "active",
        )
        return f"🎮 {game.name} — {intro}\n(your next messages go to the game until it ends or you /game quit)"

    def play(self, chat_key: str, text: str) -> str | None:
        """Route one chat message into the live game. None = no game here."""
        session = self.active(chat_key)
        if session is None:
            return None
        game = GAMES.get(session["game"])
        if game is None:
            return None
        reply, state, status = game.move(session["state"], text, self.rng, self._suggest())
        self._persist(
            {"id": session["id"], "game": session["game"], "chat_key": chat_key,
             "created_at": session["created_at"]},
            state, status,
        )
        if status != "active":
            return f"{reply}\n(game over — the chat is yours again)"
        return reply

    def quit(self, chat_key: str) -> str:
        session = self.active(chat_key)
        if session is None:
            return "no game is live in this chat."
        self._persist(
            {"id": session["id"], "game": session["game"], "chat_key": chat_key,
             "created_at": session["created_at"]},
            session["state"], "ended",
        )
        return "game over. the chat is yours again."
