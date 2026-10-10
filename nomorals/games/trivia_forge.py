"""TriviaForge: dynamic, personalized trivia question generation.

The old trivia bank was 143 static Q&A pairs — the same questions every
game, forever. This module replaces that with:

* **LLM generation** — fresh questions on any topic, at any difficulty,
  via the ``suggest`` bridge (same pattern as GameMaster). The model
  gets the topic, difficulty, and anti-repeat list; it returns JSON.
* **Template forge fallback** — seeded composable questions from
  structured facts when no model is available. Never stalls.
* **Interest-profile topics** — question topics are sampled from the
  player's interest profile (arena activity), so trivia feels personal.
* **Persisted anti-repeat** — served question hashes live in KVStore;
  nothing repeats within the window (default 50).

Usage::

    forge = TriviaForge(suggest=model_suggest_fn, db=db)
    questions = forge.deal(count=8, difficulty="normal",
                           profile=interest_profile(db))
    # → [(question, answer), ...]
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
from typing import Any, Callable

from ..core.logging_setup import get_logger
from ..storage.kv import KVStore

_log = get_logger(__name__)

__all__ = ["TriviaForge", "TRIVIA_TOPICS", "ANTI_REPEAT_DEFAULT",
           "TriviaLadder", "LADDER_PRIZES", "LADDER_DIFFICULTIES",
           "LADDER_SAFE_HAVENS", "LIFELINES", "build_options"]

#: question topic pools, weighted by generality. The interest profile
#: re-weights these per player.
TRIVIA_TOPICS: dict[str, list[str]] = {
    "general": ["science", "history", "geography", "sports", "arts",
                "technology", "nature", "food"],
    "nigeria": ["nigerian history", "nollywood", "afrobeats",
                "nigerian geography", "nigerian politics"],
    "tech": ["programming", "ai", "gadgets", "internet culture",
             "video games"],
    "pop": ["movies", "music", "celebrities", "tv shows", "memes"],
    "brain": ["riddles", "logic puzzles", "wordplay", "math trivia"],
}

ANTI_REPEAT_DEFAULT = 50

_JSON_RE = re.compile(r"\[.*\]", re.S)


def _hash_q(question: str) -> str:
    return hashlib.sha256(question.strip().lower().encode()).hexdigest()[:16]


class TriviaForge:
    """Deal fresh trivia questions. LLM-first, template fallback."""

    def __init__(
        self,
        suggest: Callable[[str], str] | None = None,
        db: Any = None,
        seed: int | None = None,
        anti_repeat: int = ANTI_REPEAT_DEFAULT,
    ) -> None:
        self._suggest = suggest
        self._kv = KVStore(db) if db is not None else None
        self.rng = random.Random(seed)
        self.anti_repeat = anti_repeat
        self._lock = threading.RLock()
        self._seen: list[str] = []
        self._load_seen()

    # ── anti-repeat ──────────────────────────────────────────────────
    def _load_seen(self) -> None:
        if self._kv is None:
            return
        try:
            data = self._kv.get("trivia_forge.seen") or {}
            self._seen = list(data.get("hashes", []))[-self.anti_repeat:]
        except Exception:  # noqa: BLE001
            self._seen = []

    def _save_seen(self) -> None:
        if self._kv is None:
            return
        try:
            self._kv.set("trivia_forge.seen",
                         {"hashes": self._seen[-self.anti_repeat:],
                          "updated": time.time()})
        except Exception:  # noqa: BLE001
            pass

    def _is_fresh(self, question: str) -> bool:
        return _hash_q(question) not in self._seen

    def _mark_seen(self, question: str) -> None:
        with self._lock:
            self._seen.append(_hash_q(question))
            self._seen = self._seen[-self.anti_repeat:]
            self._save_seen()

    # ── topic sampling ───────────────────────────────────────────────
    def sample_topics(self, count: int,
                      profile: dict[str, float] | None = None) -> list[str]:
        """Pick topics, weighted by the player's interest profile."""
        all_topics = [t for sub in TRIVIA_TOPICS.values() for t in sub]
        if not profile:
            return [self.rng.choice(all_topics) for _ in range(count)]
        # map profile categories onto trivia topics loosely
        weights: list[float] = []
        for t in all_topics:
            w = 1.0
            tl = t.lower()
            for cat, score in profile.items():
                if cat.lower() in tl or tl in cat.lower():
                    w += float(score)
            weights.append(w)
        return self.rng.choices(all_topics, weights=weights, k=count)

    # ── LLM generation ───────────────────────────────────────────────
    def _ask_model(self, topic: str, difficulty: str,
                   count: int) -> list[tuple[str, str]]:
        if self._suggest is None:
            return []
        prompt = (
            f"Generate {count} {difficulty} trivia questions about {topic}. "
            "Return ONLY a JSON array of [question, answer] pairs, e.g. "
            '[["What is the capital of France?", "Paris"]]. '
            "Short answers (1-3 words). No duplicates. No explanations."
        )
        try:
            raw = (self._suggest(prompt) or "").strip()
        except Exception:  # noqa: BLE001
            return []
        m = _JSON_RE.search(raw)
        if not m:
            return []
        try:
            pairs = json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            return []
        out = []
        batch_hashes: set[str] = set()
        for p in pairs:
            if isinstance(p, (list, tuple)) and len(p) == 2:
                q, a = str(p[0]).strip(), str(p[1]).strip()
                h = _hash_q(q)
                if q and a and h not in batch_hashes and self._is_fresh(q):
                    batch_hashes.add(h)
                    out.append((q, a))
        return out

    # ── template forge (offline fallback) ────────────────────────────
    #: (template, answer_fn_key) — the answer is computed from the seed data
    _TEMPLATES: tuple[tuple[str, str], ...] = (
        ("What is {n1} + {n2}?", "add"),
        ("What is {n1} × {n2}?", "mul"),
        ("How many days are in {n3} weeks?", "weeks"),
        ("What is the {ord} planet from the sun?", "planet"),
        ("How many sides does a {shape} have?", "sides"),
        ("What year is {n4} years after 2000?", "year"),
        ("Spell the word '{word}' backwards.", "reverse"),
        ("How many letters are in the word '{word}'?", "letters"),
    )

    _PLANETS = ("Mercury", "Venus", "Earth", "Mars", "Jupiter",
                "Saturn", "Uranus", "Neptune")
    _ORDS = ("first", "second", "third", "fourth", "fifth",
             "sixth", "seventh", "eighth")
    _SHAPES = {"triangle": 3, "square": 4, "pentagon": 5,
               "hexagon": 6, "octagon": 8, "decagon": 10}
    _WORDS = ("python", "rocket", "garden", "thunder", "piano",
              "bridge", "candle", "forest", "anchor", "comet")

    def _forge_one(self, difficulty: str) -> tuple[str, str]:
        tpl, kind = self.rng.choice(self._TEMPLATES)
        if kind == "add":
            n1 = self.rng.randint(11, 99)
            n2 = self.rng.randint(11, 99)
            return tpl.format(n1=n1, n2=n2), str(n1 + n2)
        if kind == "mul":
            hi = 12 if difficulty in ("easy", "normal") else 20
            n1 = self.rng.randint(3, hi)
            n2 = self.rng.randint(3, hi)
            return tpl.format(n1=n1, n2=n2), str(n1 * n2)
        if kind == "weeks":
            n3 = self.rng.randint(2, 12)
            return tpl.format(n3=n3), str(n3 * 7)
        if kind == "planet":
            i = self.rng.randrange(8)
            return tpl.format(ord=self._ORDS[i]), self._PLANETS[i]
        if kind == "sides":
            shape = self.rng.choice(list(self._SHAPES))
            return tpl.format(shape=shape), str(self._SHAPES[shape])
        if kind == "year":
            n4 = self.rng.randint(1, 30)
            return tpl.format(n4=n4), str(2000 + n4)
        if kind == "reverse":
            word = self.rng.choice(self._WORDS)
            return tpl.format(word=word), word[::-1]
        if kind == "letters":
            word = self.rng.choice(self._WORDS)
            return tpl.format(word=word), str(len(word))
        return "What is 2 + 2?", "4"

    def _forge(self, count: int, difficulty: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        batch_hashes: set[str] = set()
        attempts = 0
        while len(out) < count and attempts < count * 10:
            attempts += 1
            q, a = self._forge_one(difficulty)
            h = _hash_q(q)
            if h not in batch_hashes and self._is_fresh(q):
                batch_hashes.add(h)
                out.append((q, a))
        return out

    def _interest_profile(self) -> dict[str, float] | None:
        """Build the player's interest profile from arena activity."""
        if self._kv is None:
            return None
        try:
            db = self._kv._db  # internal but same package family
            from ..agents.arena.activity import interest_profile
            return interest_profile(db)
        except Exception:  # noqa: BLE001
            return None

    # ── public API ───────────────────────────────────────────────────
    def deal(self, count: int = 8, difficulty: str = "normal",
             profile: dict[str, float] | None = None) -> list[tuple[str, str]]:
        """Deal ``count`` fresh questions. LLM first, forge fallback."""
        if profile is None:
            profile = self._interest_profile()
        topics = self.sample_topics(count, profile)
        out: list[tuple[str, str]] = []
        # try the model per topic (one call can cover several)
        if self._suggest is not None:
            for topic in dict.fromkeys(topics):  # dedup, keep order
                need = count - len(out)
                if need <= 0:
                    break
                got = self._ask_model(topic, difficulty, need)
                out.extend(got)
        # fill the rest from the forge
        if len(out) < count:
            out.extend(self._forge(count - len(out), difficulty))
        # mark everything seen
        for q, _ in out:
            self._mark_seen(q)
        return out[:count]


# ── TriviaLadder: Millionaire-style match structure ──────────────────────────
#
# Dealt questions are raw material; a *match* needs stakes. The ladder
# climbs difficulty (and prize) each round: answer to bank the round's
# coins, miss and fall to the last safe haven. Three lifelines —
# 50:50, skip, audience — one use each per match. Answer streaks
# multiply the prize. Wrong answers never end the night empty-handed:
# safe havens guarantee a floor.

LADDER_PRIZES: tuple[int, ...] = (10, 25, 50, 100, 200, 350, 600, 1000)
LADDER_DIFFICULTIES: tuple[str, ...] = (
    "easy", "easy", "normal", "normal", "normal", "hard", "hard", "expert",
)
#: rounds that are safe havens — falling later still banks this prize.
LADDER_SAFE_HAVENS: tuple[int, ...] = (2, 5)

LIFELINES: tuple[tuple[str, str], ...] = (
    ("fifty", "5️⃣0️⃣ 50:50 — remove two wrong options"),
    ("skip", "⏭️ skip — swap this question for a fresh one"),
    ("audience", "🗳️ audience — the crowd votes (usually right)"),
)


class TriviaLadder:
    """One ladder match: state, lifelines, streaks, prizes.

    ``questions`` is ``[(question, answer, options)]`` — options a list
    of 4 with the answer included (the forge deals Q&A; the ladder
    builds options via the model or distractor forge). ``ask()`` renders
    the current question card; ``answer(text)`` scores it; lifelines via
    :meth:`use_lifeline`.
    """

    def __init__(self, questions: list[tuple[str, str, list[str]]],
                 seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.questions = [(q, a, list(o)) for q, a, o in questions]
        self.round = 0
        self.streak = 0
        self.banked = 0
        self.lifelines = {slug: True for slug, _ in LIFELINES}
        self.over = False
        self._fifty: list[str] | None = None
        self._log: list[str] = []

    # ── state ────────────────────────────────────────────────────────────
    @property
    def prize(self) -> int:
        idx = min(self.round, len(LADDER_PRIZES) - 1)
        return LADDER_PRIZES[idx]

    @property
    def difficulty(self) -> str:
        idx = min(self.round, len(LADDER_DIFFICULTIES) - 1)
        return LADDER_DIFFICULTIES[idx]

    def current(self) -> tuple[str, str, list[str]] | None:
        if self.over or self.round >= len(self.questions):
            return None
        return self.questions[self.round]

    def ask(self) -> str:
        """The question card."""
        cur = self.current()
        if cur is None:
            return self._final()
        q, _a, options = cur
        opts = list(options)
        if self._fifty is not None:
            opts = [o for o in opts if o in self._fifty]
        self.rng.shuffle(opts)
        letters = "ABCD"
        lines = [f"❓ round {self.round + 1}/{len(self.questions)} — "
                 f"**{self.prize}c** ({self.difficulty})"]
        if self.streak >= 2:
            lines[0] += f" · 🔥 streak ×{self.streak}"
        lines.append(q)
        for i, o in enumerate(opts[:4]):
            lines.append(f"  {letters[i]}. {o}")
        avail = [slug for slug, _desc in LIFELINES
                 if self.lifelines[slug]]
        if avail:
            lines.append("lifelines: " + " ".join(
                f"/lifeline {s}" for s in avail))
        haven = max([h for h in LADDER_SAFE_HAVENS if h <= self.round],
                    default=None)
        if haven is not None:
            lines.append(f"🛟 safe haven: {LADDER_PRIZES[haven]}c banked")
        return "\n".join(lines)

    # ── lifelines ────────────────────────────────────────────────────────
    def use_lifeline(self, slug: str) -> str:
        slug = (slug or "").strip().lower()
        if self.over:
            return "the match is over."
        if slug not in self.lifelines:
            return f"unknown lifeline. try: {', '.join(self.lifelines)}"
        if not self.lifelines[slug]:
            return "that lifeline is spent."
        cur = self.current()
        if cur is None:
            return "no active question."
        _q, answer, options = cur
        self.lifelines[slug] = False
        if slug == "fifty":
            wrong = [o for o in options if o != answer]
            keep = self.rng.sample(wrong, min(2, len(wrong)))
            self._fifty = [answer] + [o for o in wrong if o not in keep]
            return "5️⃣0️⃣ two wrong answers removed."
        if slug == "skip":
            self._fifty = None
            self._log.append(f"round {self.round + 1} skipped")
            self.round += 1
            if self.round >= len(self.questions):
                self.over = True
                return "⏭️ skipped — and that was the last question.\n" \
                    + self._final()
            return "⏭️ skipped — fresh question:\n" + self.ask()
        # audience: weighted vote, usually (not always) right
        weights = []
        for o in options:
            weights.append(55 if o == answer else 15)
        total = sum(weights)
        pick = self.rng.choices(options,
                                weights=[w / total for w in weights])[0]
        conf = self.rng.randint(52, 88)
        return (f"🗳️ the audience votes **{pick}** ({conf}% sure). "
                f"trust them?")

    # ── answering ────────────────────────────────────────────────────────
    def answer(self, text: str) -> tuple[bool, str]:
        """Score an answer. Returns (match_over, message)."""
        cur = self.current()
        if cur is None or self.over:
            return True, self._final()
        _q, answer, _options = cur
        guess = (text or "").strip().lower()
        # accept letter or full text
        letters = "abcd"
        if len(guess) == 1 and guess in letters:
            opts = [o for o in _options
                    if self._fifty is None or o in self._fifty]
            guess = opts[letters.index(guess)].lower() \
                if letters.index(guess) < len(opts) else guess
        correct = guess == answer.strip().lower()
        self._fifty = None
        if correct:
            self.streak += 1
            mult = 1 + 0.25 * (self.streak - 1)
            won = int(round(self.prize * mult))
            self.banked += won
            self._log.append(f"round {self.round + 1}: +{won}c")
            self.round += 1
            if self.round >= len(self.questions):
                self.over = True
                return True, (f"✅ correct! +{won}c\n\n{self._final()}")
            return False, (f"✅ correct! +{won}c (banked {self.banked}c)\n"
                           + self.ask())
        # miss: fall to the last safe haven — banked keeps only what
        # was actually won up to the haven floor
        haven = max([h for h in LADDER_SAFE_HAVENS if h <= self.round],
                    default=None)
        self.banked = LADDER_PRIZES[haven] if haven is not None else 0
        self.over = True
        self.streak = 0
        return True, (f"❌ wrong — the answer was **{answer}**.\n"
                      f"🛟 you fall to the safe haven: {self.banked}c banked.\n\n"
                      + self._final())

    def walk_away(self) -> str:
        """Bank current winnings and end the match."""
        if self.over:
            return self._final()
        self.over = True
        return (f"🚶 you walk away with **{self.banked}c** — "
                f"smart money.\n\n" + self._final())

    def _final(self) -> str:
        self.over = True
        lines = [f"🏁 ladder complete — banked **{self.banked}c** "
                 f"in {len(self._log)} rounds"]
        if self._log:
            lines.append("  " + " · ".join(self._log[-4:]))
        spent = [s for s, v in self.lifelines.items() if not v]
        if spent:
            lines.append(f"  lifelines used: {', '.join(spent)}")
        return "\n".join(lines)


def build_options(answer: str, distractors: list[str],
                  rng: random.Random | None = None) -> list[str]:
    """4 options with the answer shuffled in (deduped, padded)."""
    rng = rng or random.Random()
    opts = [answer]
    for d in distractors:
        if d and d != answer and d not in opts:
            opts.append(d)
        if len(opts) == 4:
            break
    while len(opts) < 4:
        opts.append(f"none of these ({len(opts)})")
    rng.shuffle(opts)
    return opts
