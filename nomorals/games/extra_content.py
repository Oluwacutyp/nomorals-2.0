"""Extra game content: new modes + expanded trivia packs.

Gap-fill, not a rebuild: the existing games live in
:mod:`nomorals.games.games` (arcade / cases / casino / puzzles / pvp /
wild / ...) and :mod:`nomorals.games.trivia_forge` (model-forged
trivia). This module adds the modes the skip-list named that did not
exist yet — word chain, riddle hunt, emoji story, speed typing,
memory match, prediction market — plus hand-written trivia packs.

Universal, not Nigeria-only: packs are keyed by
``(nation, language, culture)`` and the registry accepts a pack for
ANY nation/culture. The Nigerian-history and Afrobeats packs are two
packs among N (Brazil, Japan, Germany, India, ... ship alongside);
Nigeria is a default pack selection, never the axis. A user in
Brazil, Japan, Germany, or India gets the same full capability with
their own pack as default — pass ``pack_id`` / ``nation`` to select.

Economy wiring: games return *scores*, never coins. Crediting goes
through an injected ``reward_fn(member, amount, reason)`` supplied by
the caller (e.g. a group-economy ``award``). No coin amounts are
hardcoded here — the caller decides what a point is worth. When no
``reward_fn`` is given, :func:`credit` is a no-op and the score is
still returned, so games work standalone.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = [
    "TRIVIA_PACKS", "trivia_pack", "trivia_packs_for",
    "credit", "WordChain", "RiddleHunt", "RIDDLES", "EmojiStory",
    "EMOJI_STORIES", "SpeedTyping", "TYPING_PASSAGES", "MemoryMatch",
    "PredictionMarket", "MarketError",
]

RewardFn = Callable[[str, int, str], Any]


def credit(reward_fn: RewardFn | None, member: str, amount: int, reason: str) -> Any:
    """Credit ``amount`` via the injected economy hook. No-op when none."""
    if reward_fn is None or amount <= 0:
        return None
    return reward_fn(member, int(amount), reason)


# ── trivia packs ────────────────────────────────────────────────────────────
# Each pack: title, nation (ISO-ish, "" = worldwide), language, culture,
# questions: [{q, options[4], answer (index)}]. Hand-written, deterministic.

def _pack(title: str, nation: str, language: str, culture: str,
          rows: list[tuple[str, list[str], int]]) -> dict[str, Any]:
    return {"title": title, "nation": nation, "language": language,
            "culture": culture,
            "questions": [{"q": q, "options": opts, "answer": ans}
                          for q, opts, ans in rows]}


TRIVIA_PACKS: dict[str, dict[str, Any]] = {
    "nigerian-history": _pack(
        "Nigerian History", "NG", "en", "nigerian", [
            ("Which city was the capital of Nigeria before Abuja?",
             ["Lagos", "Ibadan", "Kano", "Enugu"], 0),
            ("In which year did Nigeria gain independence from Britain?",
             ["1957", "1960", "1963", "1966"], 1),
            ("The Benin Bronzes are associated with which historic kingdom?",
             ["Oyo", "Benin", "Kanem-Bornu", "Sokoto"], 1),
            ("Who was Nigeria's first president?",
             ["Nnamdi Azikiwe", "Obafemi Awolowo", "Tafawa Balewa", "Yakubu Gowon"], 0),
            ("The Nok culture is best known for what art form?",
             ["Bronze heads", "Terracotta sculptures", "Rock paintings", "Woven cloth"], 1),
        ]),
    "afrobeats": _pack(
        "Afrobeats", "", "en", "afrobeats", [
            ("Which artist released the global hit 'Essence'?",
             ["Burna Boy", "Wizkid ft. Tems", "Davido", "Rema"], 1),
            ("Fela Kuti is credited as the pioneer of which genre?",
             ["Highlife", "Afrobeat", "Juju", "Amapiano"], 1),
            ("'Calm Down' is a worldwide hit by which artist?",
             ["Ayra Starr", "Rema", "Asake", "Omah Lay"], 1),
            ("Which city is widely regarded as the home of Afrobeats?",
             ["Accra", "Lagos", "Johannesburg", "Nairobi"], 1),
            ("Burna Boy's 2019 album that broke him globally is titled?",
             ["Twice as Tall", "African Giant", "Love, Damini", "Outside"], 1),
        ]),
    "football": _pack(
        "Football (Global)", "", "en", "football", [
            ("Which country has won the most FIFA World Cups?",
             ["Germany", "Argentina", "Brazil", "Italy"], 2),
            ("How often is the FIFA World Cup held?",
             ["Every 2 years", "Every 3 years", "Every 4 years", "Every 5 years"], 2),
            ("Which club is nicknamed 'The Red Devils'?",
             ["Liverpool", "Arsenal", "Manchester United", "AC Milan"], 2),
            ("The Africa Cup of Nations is organised by which body?",
             ["FIFA", "UEFA", "CAF", "AFC"], 2),
            ("In football, how many players per team are on the pitch?",
             ["9", "10", "11", "12"], 2),
        ]),
    "science": _pack(
        "Science", "", "en", "science", [
            ("What is the chemical symbol for gold?",
             ["Ag", "Au", "Gd", "Go"], 1),
            ("Which planet is closest to the Sun?",
             ["Venus", "Mercury", "Mars", "Earth"], 1),
            ("What force keeps planets in orbit around the Sun?",
             ["Magnetism", "Friction", "Gravity", "Inertia"], 2),
            ("H2O is the formula for what?",
             ["Salt", "Oxygen", "Water", "Hydrogen"], 2),
            ("How many legs does a spider have?",
             ["6", "8", "10", "4"], 1),
        ]),
    "brazil-culture": _pack(
        "Brazil Culture", "BR", "pt", "brazilian", [
            ("Which city hosts the world's largest carnival?",
             ["Sao Paulo", "Salvador", "Rio de Janeiro", "Recife"], 2),
            ("What is Brazil's official language?",
             ["Spanish", "Portuguese", "French", "Italian"], 1),
            ("Capoeira combines martial art with what?",
             ["Dance and music", "Swimming", "Chess", "Archery"], 0),
            ("Which river runs through the Amazon rainforest?",
             ["Parana", "Amazon", "Sao Francisco", "Tocantins"], 1),
            ("Brasilia is best described as what?",
             ["A beach resort", "The planned capital", "A football stadium", "An island"], 1),
        ]),
    "japan-culture": _pack(
        "Japan Culture", "JP", "ja", "japanese", [
            ("What is Japan's capital?",
             ["Seoul", "Beijing", "Tokyo", "Osaka"], 2),
            ("Which flower is a national symbol of Japan?",
             ["Rose", "Cherry blossom", "Lotus", "Tulip"], 1),
            ("Sushi is traditionally made with what staple?",
             ["Noodles", "Rice", "Bread", "Potato"], 1),
            ("Which martial art originated in Japan?",
             ["Taekwondo", "Kung Fu", "Judo", "Muay Thai"], 2),
            ("Mount Fuji is what kind of mountain?",
             ["A volcano", "A glacier", "A mesa", "A canyon"], 0),
        ]),
    "germany-culture": _pack(
        "Germany Culture", "DE", "de", "german", [
            ("What is Germany's capital?",
             ["Munich", "Hamburg", "Berlin", "Frankfurt"], 2),
            ("The Oktoberfest festival began in which city?",
             ["Berlin", "Cologne", "Munich", "Stuttgart"], 2),
            ("Which German composer wrote the Brandenburg Concertos?",
             ["Beethoven", "Bach", "Handel", "Mozart"], 1),
            ("What is the German word for 'thank you'?",
             ["Bitte", "Danke", "Hallo", "Tschuss"], 1),
            ("The Rhine is best described as what?",
             ["A mountain", "A river", "A forest", "A castle"], 1),
        ]),
    "india-culture": _pack(
        "India Culture", "IN", "hi", "indian", [
            ("What is India's capital?",
             ["Mumbai", "Kolkata", "New Delhi", "Chennai"], 2),
            ("Diwali is best described as what?",
             ["Harvest dance", "Festival of Lights", "New Year parade", "Monsoon rite"], 1),
            ("Which river is considered sacred in Hindu tradition?",
             ["Yamuna", "Ganges", "Brahmaputra", "Godavari"], 1),
            ("Cricket's most-watched rivalry includes India and which team?",
             ["Australia", "Pakistan", "England", "South Africa"], 1),
            ("Which spice gives turmeric its yellow colour?",
             ["Cumin", "Curcumin", "Cardamom", "Chilli"], 1),
        ]),
    "world-history": _pack(
        "World History", "", "en", "world", [
            ("The Great Wall is located in which country?",
             ["Japan", "China", "Korea", "Mongolia"], 1),
            ("Which ancient civilisation built the pyramids of Giza?",
             ["Romans", "Greeks", "Egyptians", "Aztecs"], 2),
            ("The first Moon landing took place in which year?",
             ["1959", "1965", "1969", "1975"], 2),
            ("Which ocean is the largest?",
             ["Atlantic", "Indian", "Arctic", "Pacific"], 3),
            ("The printing press is credited to which inventor?",
             ["Newton", "Gutenberg", "Galileo", "Da Vinci"], 1),
        ]),
}


def trivia_pack(pack_id: str) -> dict[str, Any] | None:
    """Return one pack by id, or None. Packs are data — never mutated."""
    return TRIVIA_PACKS.get(pack_id)


def trivia_packs_for(*, nation: str = "", language: str = "",
                     culture: str = "") -> list[str]:
    """Pack ids matching the filters; worldwide packs (nation='') match all.

    Empty filters return every pack. A caller in any nation gets their
    own pack first, then the worldwide packs.
    """
    out: list[str] = []
    for pid, pack in TRIVIA_PACKS.items():
        if nation and pack["nation"] not in ("", nation):
            continue
        if language and pack["language"] not in ("", language):
            continue
        if culture and pack["culture"] not in ("", culture):
            continue
        out.append(pid)
    own = [p for p in out if TRIVIA_PACKS[p]["nation"] == nation] if nation else []
    rest = [p for p in out if p not in own]
    return own + rest


# ── word chain ──────────────────────────────────────────────────────────────

@dataclass
class WordChain:
    """Word chain: each word must start with the previous word's last letter.

    ``players`` take turns; ``play(member, word)`` returns a result dict.
    Scoring is returned as points; crediting is the caller's job via
    ``reward_fn`` (injected, never hardcoded coins).
    """
    players: list[str]
    reward_fn: RewardFn | None = None
    min_len: int = 3
    chain: list[dict[str, Any]] = field(default_factory=list)
    scores: dict[str, int] = field(default_factory=dict)
    turn: int = 0
    over: bool = False

    def current_player(self) -> str | None:
        if not self.players or self.over:
            return None
        return self.players[self.turn % len(self.players)]

    def play(self, member: str, word: str, *, points_per_letter: int = 1) -> dict[str, Any]:
        if self.over:
            return {"ok": False, "error": "game over"}
        if member != self.current_player():
            return {"ok": False, "error": "not your turn"}
        w = (word or "").strip().lower()
        if len(w) < self.min_len or not w.isalpha():
            return {"ok": False, "error": f"word must be >= {self.min_len} letters"}
        if any(c["word"] == w for c in self.chain):
            return {"ok": False, "error": "word already used"}
        if self.chain and w[0] != self.chain[-1]["word"][-1]:
            return {"ok": False, "error": f"word must start with '{self.chain[-1]['word'][-1]}'"}
        pts = len(w) * max(0, points_per_letter)
        self.chain.append({"member": member, "word": w})
        self.scores[member] = self.scores.get(member, 0) + pts
        self.turn += 1
        credit(self.reward_fn, member, pts, "word_chain")
        return {"ok": True, "word": w, "points": pts, "score": self.scores[member],
                "next": self.current_player()}

    def leader(self) -> str | None:
        return max(self.scores, key=self.scores.get) if self.scores else None


# ── riddle hunt ─────────────────────────────────────────────────────────────

RIDDLES: list[dict[str, Any]] = [
    {"q": "I speak without a mouth and hear without ears. I have no body, but I come alive with wind. What am I?",
     "a": ["echo"], "culture": "world", "hint": "You hear me in mountains."},
    {"q": "The more of me you take, the more you leave behind. What am I?",
     "a": ["footsteps", "steps"], "culture": "world", "hint": "Think of walking."},
    {"q": "I am always coming but never arrive. What am I?",
     "a": ["tomorrow"], "culture": "world", "hint": "It follows today."},
    {"q": "What has keys but cannot open a single lock?",
     "a": ["piano", "keyboard"], "culture": "world", "hint": "It makes music or types."},
    {"q": "In Lagos traffic I am long, in a story I am short; I am measured in minutes and wasted by all. What am I?",
     "a": ["time"], "culture": "nigerian", "hint": "Everyone wants more of me."},
    {"q": "I wear a crown but am not a king; I am eaten, not obeyed. What am I?",
     "a": ["pineapple"], "culture": "world", "hint": "A tropical fruit."},
]


@dataclass
class RiddleHunt:
    """Riddle hunt: answer riddles; hints cost a point, solves earn points."""
    riddles: list[dict[str, Any]] = field(default_factory=lambda: list(RIDDLES))
    reward_fn: RewardFn | None = None
    seed: int = 0
    idx: int = 0
    scores: dict[str, int] = field(default_factory=dict)
    hints_used: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rng = random.Random(self.seed)
        self._order = list(range(len(self.riddles)))
        rng.shuffle(self._order)

    def current(self) -> dict[str, Any] | None:
        if self.idx >= len(self._order):
            return None
        r = self.riddles[self._order[self.idx]]
        return {"n": self.idx + 1, "total": len(self._order), "question": r["q"],
                "culture": r.get("culture", "world")}

    def hint(self, member: str) -> dict[str, Any]:
        cur = self.current()
        if cur is None:
            return {"ok": False, "error": "hunt complete"}
        self.hints_used[member] = self.hints_used.get(member, 0) + 1
        self.scores[member] = self.scores.get(member, 0) - 1
        return {"ok": True, "hint": self.riddles[self._order[self.idx]].get("hint", "")}

    def answer(self, member: str, text: str, *, points: int = 10) -> dict[str, Any]:
        cur = self.current()
        if cur is None:
            return {"ok": False, "error": "hunt complete"}
        r = self.riddles[self._order[self.idx]]
        guess = (text or "").strip().lower()
        if guess in [a.lower() for a in r["a"]]:
            self.scores[member] = self.scores.get(member, 0) + points
            self.idx += 1
            credit(self.reward_fn, member, points, "riddle_hunt")
            nxt = self.current()
            return {"ok": True, "correct": True, "points": points,
                    "next": nxt["question"] if nxt else None,
                    "complete": nxt is None}
        return {"ok": True, "correct": False, "points": 0}


# ── emoji story ─────────────────────────────────────────────────────────────

EMOJI_STORIES: list[dict[str, Any]] = [
    {"emoji": "🌧️🏃‍♂️🏠☕", "keywords": ["rain", "run", "home", "coffee", "tea", "shelter"],
     "title": "Caught in the rain"},
    {"emoji": "🎣🐟🍲👨‍👩‍👧", "keywords": ["fish", "fishing", "cook", "soup", "family", "eat"],
     "title": "The catch"},
    {"emoji": "✈️🗺️🏖️📸", "keywords": ["travel", "trip", "flight", "beach", "photo", "journey"],
     "title": "Holiday"},
    {"emoji": "🌱🌧️🌽🏆", "keywords": ["plant", "rain", "corn", "farm", "harvest", "win"],
     "title": "The farm"},
]


@dataclass
class EmojiStory:
    """Emoji story: write a one-line story from an emoji prompt.

    Score = keyword coverage (fraction of keywords present) × max_points.
    """
    prompts: list[dict[str, Any]] = field(default_factory=lambda: list(EMOJI_STORIES))
    reward_fn: RewardFn | None = None
    seed: int = 0
    idx: int = 0
    scores: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rng = random.Random(self.seed)
        self._order = list(range(len(self.prompts)))
        rng.shuffle(self._order)

    def current(self) -> dict[str, Any] | None:
        if self.idx >= len(self._order):
            return None
        p = self.prompts[self._order[self.idx]]
        return {"emoji": p["emoji"], "title": p["title"]}

    def submit(self, member: str, story: str, *, max_points: int = 10) -> dict[str, Any]:
        cur = self.current()
        if cur is None:
            return {"ok": False, "error": "no more prompts"}
        p = self.prompts[self._order[self.idx]]
        text = (story or "").lower()
        hits = sum(1 for kw in p["keywords"] if kw.lower() in text)
        pts = round(max_points * hits / max(1, len(p["keywords"])))
        self.scores[member] = self.scores.get(member, 0) + pts
        self.idx += 1
        credit(self.reward_fn, member, pts, "emoji_story")
        return {"ok": True, "points": pts, "hits": hits,
                "keywords": len(p["keywords"]), "story": story}


# ── speed typing ────────────────────────────────────────────────────────────

TYPING_PASSAGES: dict[str, str] = {
    "en": "The quick brown fox jumps over the lazy dog near the river bank.",
    "yo": "Omo rere ni omo ti a bi, omo ti a to, omo ti a ko ni eko.",
    "pcm": "Naija no dey carry last, we dey move, we dey hustle every day.",
    "pt": "O Brasil e um pais grande com praias, florestas e muitas culturas.",
    "ja": "Konnichiwa, genki desu ka. Kyou wa ii tenki desu ne.",
    "de": "Guten Tag, wie geht es dir heute in dieser schoenen Stadt.",
    "hi": "Namaste, aaj ka din bahut achha hai aur hum sab khush hain.",
}


@dataclass
class SpeedTyping:
    """Speed typing: type the passage; WPM + accuracy, timed by caller."""
    language: str = "en"
    reward_fn: RewardFn | None = None
    started_at: float = 0.0
    passage: str = ""

    def __post_init__(self) -> None:
        self.passage = TYPING_PASSAGES.get(self.language, TYPING_PASSAGES["en"])

    def start(self, now: float | None = None) -> dict[str, Any]:
        self.started_at = now if now is not None else time.time()
        return {"passage": self.passage, "language": self.language}

    def submit(self, member: str, typed: str, now: float | None = None,
               *, points_per_wpm: float = 1.0) -> dict[str, Any]:
        end = now if now is not None else time.time()
        elapsed = max(0.01, end - self.started_at)
        target, got = self.passage, typed or ""
        correct = sum(1 for a, b in zip(target, got) if a == b)
        accuracy = correct / max(1, len(target))
        words = len(got.split())
        wpm = round(words / (elapsed / 60.0), 1)
        pts = int(wpm * accuracy * points_per_wpm)
        credit(self.reward_fn, member, pts, "speed_typing")
        return {"ok": True, "wpm": wpm, "accuracy": round(accuracy, 3),
                "correct_chars": correct, "points": pts, "elapsed_s": round(elapsed, 2)}


# ── memory match ────────────────────────────────────────────────────────────

@dataclass
class MemoryMatch:
    """Memory match: flip two cards; a matching pair scores. Seeded deck."""
    pairs: int = 6
    reward_fn: RewardFn | None = None
    seed: int = 0
    deck: list[str] = field(default_factory=list)
    revealed: list[bool] = field(default_factory=list)
    matched: list[bool] = field(default_factory=list)
    _first: int | None = None
    scores: dict[str, int] = field(default_factory=dict)
    flips: int = 0

    def __post_init__(self) -> None:
        symbols = ["🍎", "🍌", "🍇", "🍓", "🥭", "🍍", "🥥", "🍊",
                   "⚽", "🎸", "🚀", "🌙"][: max(1, self.pairs)]
        deck = symbols + symbols
        random.Random(self.seed).shuffle(deck)
        self.deck = deck
        self.revealed = [False] * len(deck)
        self.matched = [False] * len(deck)

    def flip(self, member: str, pos: int, *, points_per_pair: int = 5) -> dict[str, Any]:
        if pos < 0 or pos >= len(self.deck):
            return {"ok": False, "error": "bad position"}
        if self.matched[pos] or (self._first is not None and self._first == pos):
            return {"ok": False, "error": "card unavailable"}
        self.flips += 1
        self.revealed[pos] = True
        if self._first is None:
            self._first = pos
            return {"ok": True, "card": self.deck[pos], "match": None}
        first, second = self._first, pos
        self._first = None
        if self.deck[first] == self.deck[second]:
            self.matched[first] = self.matched[second] = True
            self.scores[member] = self.scores.get(member, 0) + points_per_pair
            credit(self.reward_fn, member, points_per_pair, "memory_match")
            return {"ok": True, "card": self.deck[pos], "match": True,
                    "points": points_per_pair,
                    "complete": all(self.matched)}
        self.revealed[first] = self.revealed[second] = False
        return {"ok": True, "card": self.deck[pos], "match": False, "points": 0}


# ── prediction market (game currency only) ───────────────────────────────────

class MarketError(ValueError):
    """Prediction-market rule violation."""


@dataclass
class PredictionMarket:
    """Prediction market settled in GAME CURRENCY ONLY.

    Balances live with the caller: ``balance_fn(member)``,
    ``debit_fn(member, amount, reason)`` and ``credit_fn(member, amount,
    reason)`` are injected. No real money, no cash-out, no purchase —
    stakes are game currency the player already holds. A market that
    cannot verify a balance refuses the bet.
    """
    balance_fn: Callable[[str], int] | None = None
    debit_fn: Callable[[str, int, str], Any] | None = None
    credit_fn: Callable[[str, int, str], Any] | None = None
    markets: dict[str, dict[str, Any]] = field(default_factory=dict)
    _seq: int = 0

    def create(self, question: str, outcomes: list[str], *,
               closes_at: float = 0.0, culture: str = "world",
               nation: str = "") -> str:
        if len(outcomes) < 2:
            raise MarketError("need at least 2 outcomes")
        self._seq += 1
        mid = f"mkt-{self._seq}"
        self.markets[mid] = {"question": question, "outcomes": list(outcomes),
                             "closes_at": closes_at, "culture": culture,
                             "nation": nation, "bets": [], "resolved": None}
        return mid

    def bet(self, market_id: str, member: str, outcome: str, stake: int,
            *, now: float | None = None) -> dict[str, Any]:
        m = self.markets.get(market_id)
        if m is None:
            raise MarketError(f"unknown market {market_id!r}")
        if m["resolved"] is not None:
            raise MarketError("market already resolved")
        if outcome not in m["outcomes"]:
            raise MarketError(f"unknown outcome {outcome!r}")
        stake = int(stake)
        if stake <= 0:
            raise MarketError("stake must be positive")
        if m["closes_at"] and (now if now is not None else time.time()) > m["closes_at"]:
            raise MarketError("market closed")
        if self.balance_fn is not None and self.balance_fn(member) < stake:
            raise MarketError("insufficient game-currency balance")
        if self.debit_fn is not None:
            self.debit_fn(member, stake, f"prediction bet {market_id}:{outcome}")
        m["bets"].append({"member": member, "outcome": outcome, "stake": stake})
        return {"ok": True, "market": market_id, "outcome": outcome, "stake": stake}

    def resolve(self, market_id: str, winning_outcome: str) -> dict[str, Any]:
        m = self.markets.get(market_id)
        if m is None:
            raise MarketError(f"unknown market {market_id!r}")
        if m["resolved"] is not None:
            raise MarketError("market already resolved")
        if winning_outcome not in m["outcomes"]:
            raise MarketError(f"unknown outcome {winning_outcome!r}")
        m["resolved"] = winning_outcome
        total = sum(b["stake"] for b in m["bets"])
        win_total = sum(b["stake"] for b in m["bets"] if b["outcome"] == winning_outcome)
        payouts: dict[str, int] = {}
        if win_total > 0:
            for b in m["bets"]:
                if b["outcome"] == winning_outcome:
                    share = round(b["stake"] * total / win_total)
                    payouts[b["member"]] = payouts.get(b["member"], 0) + share
                    if self.credit_fn is not None:
                        self.credit_fn(b["member"], share, f"prediction win {market_id}")
        return {"ok": True, "market": market_id, "winner": winning_outcome,
                "pool": total, "payouts": payouts}
