"""Easy / high-engagement games: quick to start, quick to finish,
playable solo against the house or in a full group.

Every game here is pure state logic over ``room.state`` — no I/O, no
clocks, fixed-seed rng — so each one is unit-testable by scripting a
sequence of moves.  The house (AI) is a real opponent in every one:
word banks and bisection for the number battle, honest-but-handicapped
answers for trivia, value-tracking bids for the auction.
"""
from __future__ import annotations

import random
import re
from typing import Any

from ..ai import GameMind
from ..players import Player
from .base import MultiGame, Room

__all__ = ["EASY_GAMES"]

_WORD_RE = re.compile(r"[a-z']+")

WORD_BANK: tuple[str, ...] = (
    "apple", "elephant", "tiger", "rabbit", "train", "night", "kite",
    "echo", "orange", "ghost", "house", "engine", "piano", "owl", "wave",
    "garden", "noodle", "letter", "trail", "ink", "key", "yarn", "nest",
    "stone", "cloud", "drum", "maple", "loop", "pumpkin", "silver",
    "river", "window", "guitar", "candle", "forest", "mountain", "desert",
    "ocean", "island", "bridge", "market", "temple", "valley", "bottle",
    "lantern", "sparrow", "willow", "cobweb", "hollow", "meadow", "thunder",
    "falcon", "copper", "willow", "anchor", "breeze", "crater", "dune",
)

HANGMAN_WORDS: tuple[tuple[str, str], ...] = (
    ("elephant", "animal"), ("giraffe", "animal"), ("penguin", "animal"),
    ("caterpillar", "animal"), ("dolphin", "animal"), ("octopus", "animal"),
    ("chameleon", "animal"), ("hedgehog", "animal"), ("flamingo", "animal"),
    ("rhinoceros", "animal"), ("kangaroo", "animal"), ("porcupine", "animal"),
    ("squirrel", "animal"), ("hamster", "animal"), ("panther", "animal"),
    ("sparrow", "animal"), ("crocodyl", "animal"), ("llama", "animal"),
    ("otter", "animal"), ("walrus", "animal"), ("volcano", "nature"),
    ("glacier", "nature"), ("thunder", "weather"), ("moonlight", "nature"),
    ("earthquake", "nature"), ("tornado", "weather"), ("hurricane", "weather"),
    ("avalanche", "nature"), ("waterfall", "nature"), ("canyon", "nature"),
    ("meadow", "nature"), ("wildflower", "nature"), ("flood", "weather"),
    ("drought", "weather"), ("breeze", "weather"), ("rainbow", "nature"),
    ("sunflower", "nature"), ("seashell", "nature"), ("coral", "nature"),
    ("mushroom", "nature"), ("keyboard", "object"), ("umbrella", "object"),
    ("telescope", "object"), ("bicycle", "object"), ("microscope", "object"),
    ("stethoscope", "object"), ("typewriter", "object"), ("lighthouse", "place"),
    ("sandwich", "food"), ("chandelier", "object"), ("backpack", "object"),
    ("screwdriver", "object"), ("thermometer", "object"), ("parachute", "object"),
    ("laptop", "object"), ("calculator", "object"), ("microphone", "object"),
    ("headphones", "object"), ("kettle", "object"), ("teapot", "object"),
    ("lantern", "object"), ("suitcase", "object"), ("snowflake", "nature"),
    ("fireworks", "event"), ("chocolate", "food"), ("pancake", "food"),
    ("watermelon", "food"), ("strawberry", "food"), ("blueberry", "food"),
    ("pineapple", "food"), ("broccoli", "food"), ("avocado", "food"),
    ("cheeseburger", "food"), ("spaghetti", "food"), ("croissant", "food"),
    ("dumpling", "food"), ("popcorn", "food"), ("cinnamon", "food"),
    ("lemonade", "food"), ("cappuccino", "food"), ("macaroon", "food"),
    ("pretzel", "food"), ("icecream", "food"), ("caramel", "food"),
    ("library", "place"), ("airport", "place"), ("cemetery", "place"),
    ("greenhouse", "place"), ("playground", "place"), ("observatory", "place"),
    ("laboratory", "place"), ("pharmacy", "place"), ("kindergarten", "place"),
    ("warehouse", "place"), ("cathedral", "place"), ("amusement", "place"),
    ("submarine", "vehicle"), ("helicopter", "vehicle"), ("battleship", "vehicle"),
    ("motorcycle", "vehicle"), ("canoe", "vehicle"), ("skyscraper", "place"),
    ("windmill", "object"), ("fountain", "place"), ("drummer", "people"),
    ("astronaut", "people"), ("pharmacist", "people"), ("firefighter", "people"),
    ("plumber", "people"), ("dentist", "people"), ("surgeon", "people"),
    ("sculptor", "people"), ("cartographer", "people"), ("architect", "people"),
    ("detective", "people"), ("pilot", "people"), ("magician", "people"),
    ("photographer", "people"), ("conductor", "people"), ("veterinarian", "people"),
    ("journalist", "people"), ("mechanic", "people"), ("gardener", "people"),
    ("saxophone", "music"), ("trumpet", "music"), ("harmonica", "music"),
    ("trombone", "music"), ("xylophone", "music"), ("accordion", "music"),
    ("maracas", "music"), ("timpani", "music"), ("bagpipes", "music"),
    ("mandolin", "music"), ("clarinet", "music"), ("recorder", "object"),
    ("vampire", "fiction"), ("mermaid", "fiction"), ("labyrinth", "mystery"),
    ("treasure", "adventure"), ("skeleton", "mystery"), ("mummia", "fiction"),
    ("pharaoh", "fiction"), ("werewolf", "fiction"), ("dragon", "fiction"),
    ("ghost", "fiction"), ("unicorn", "fiction"), ("knight", "fiction"),
    ("pirate", "fiction"), ("zombie", "fiction"), ("griffin", "fiction"),
    ("centaur", "fiction"), ("fingernail", "body"), ("eyebrow", "body"),
    ("knuckle", "body"), ("fingertip", "body"), ("elbow", "body"),
    ("shoulder", "body"), ("ankle", "body"), ("kneecap", "body"),
    ("wrist", "body"), ("palm", "body"), ("throat", "body"),
    ("eyelash", "body"), ("tournament", "event"), ("carnival", "event"),
    ("birthday", "event"), ("wedding", "event"), ("graduation", "event"),
    ("competition", "event"), ("celebration", "event"), ("rehearsal", "event"),
    ("masquerade", "event"), ("adventure", "adventure"), ("journey", "adventure"),
    ("expedition", "adventure"), ("mystery", "mystery"), ("secret", "mystery"),
    ("clue", "mystery"), ("puzzle", "mystery"), ("enigma", "mystery"),
    ("riddle", "mystery"), ("horizon", "nature"), ("constellation", "nature"),
    ("meteor", "nature"), ("aurora", "nature")
)

_GLOBAL_FREQ = ("e", "t", "a", "o", "i", "n", "s", "h", "r", "d",
               "l", "c", "u", "m", "w", "f", "g", "y", "p", "b",
               "v", "k", "j", "x", "q", "z")


def hangman_daily_word(day: str | None = None) -> str:
    """The Word of the Day: the same word for every player, every chat.

    Deterministic per calendar day (date string -> sha256 -> index), so a
    3 a.m. game in Lagos and a 9 p.m. game in Denver pick the same word —
    the shared word is the point of a daily.
    """
    import hashlib
    from datetime import date

    key = day or date.today().isoformat()
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    word, _cat = HANGMAN_WORDS[int.from_bytes(digest[:4], "big") % len(HANGMAN_WORDS)]
    return word


def _build_letter_priority() -> dict[tuple[str, int], tuple[str, ...]]:
    """Per (category, word length) letter ordering for the house AI.

    The word list IS the deck, so optimal play for the house is to guess
    the letter that appears in the most remaining deck cards.  Computed
    once at import: count occurrences of each letter in the words of that
    category at that length, rank by (count desc, global frequency).
    """
    buckets: dict[tuple[str, int], dict[str, int]] = {}
    for word, category in HANGMAN_WORDS:
        b = buckets.setdefault((category, len(word)), {})
        for ch in set(word.lower()):
            b[ch] = b.get(ch, 0) + 1
    out: dict[tuple[str, int], tuple[str, ...]] = {}
    for key, counts in buckets.items():
        ranked = sorted(
            set("".join(counts)) | set(_GLOBAL_FREQ),
            key=lambda ch: (-counts.get(ch, 0), _GLOBAL_FREQ.index(ch)),
        )
        out[key] = tuple(ranked)
    return out


#: house letter order by (category, length) — deck-aware, beats the flat
#: frequency list when the category narrows the field.
HANGMAN_LETTER_PRIORITY: dict[tuple[str, int], tuple[str, ...]] = _build_letter_priority()

TRIVIA: tuple[tuple[str, str], ...] = (
    ("What year did the first iPhone launch?", "2007"), ("Which planet has the most moons?", "Saturn"),
    ("What does 'HTTP' stand for? (first word)", "HyperText"), ("In which country is the city of Enugu?", "Nigeria"),
    ("What gas do plants absorb from the atmosphere?", "Carbon dioxide"), ("Which language runs natively in most web browsers?", "JavaScript"),
    ("How many bits are in a byte?", "8"), ("What is the capital of Canada?", "Ottawa"),
    ("Which element has the chemical symbol 'Au'?", "Gold"), ("Who wrote '1984'?", "Orwell"),
    ("What does 'CPU' stand for? (first word)", "Central"), ("Largest ocean on Earth?", "Pacific"),
    ("What year did World War II end?", "1945"), ("Which protocol commonly uses port 443?", "HTTPS"),
    ("How many officially recognized time zones does China use?", "1"), ("The speed of light is roughly how many km/s?", "300000"),
    ("Which desert is the largest hot desert?", "Sahara"), ("How many players are on a soccer team on the pitch?", "11"),
    ("What is the tallest mountain above sea level?", "Everest"), ("Which planet is known as the Red Planet?", "Mars"),
    ("How many sides does a hexagon have?", "6"), ("The Great Wall is in which country?", "China"),
    ("What metal is liquid at room temperature?", "Mercury"), ("Which ocean is the smallest?", "Arctic"),
    ("What is the chemical symbol for iron?", "Fe"), ("How many strings does a standard guitar have?", "6"),
    ("Which country invented paper?", "China"), ("What is the largest mammal on Earth?", "Blue whale"),
    ("In which year did the Titanic sink?", "1912"), ("What is the currency of Japan?", "Yen"),
    ("Which bone is the longest in the human body?", "Femur"), ("How many continents are there on Earth?", "7"),
    ("What is the capital of Australia?", "Canberra"), ("Which gas makes up most of the atmosphere?", "Nitrogen"),
    ("What year was the World Wide Web invented?", "1989"), ("Which planet is closest to the Sun?", "Mercury"),
    ("How many colors are in a rainbow?", "7"), ("What is the smallest prime number?", "2"),
    ("Which country has the longest coastline?", "Canada"), ("What is the hardest natural substance?", "Diamond"),
    ("How many teeth does an adult human typically have?", "32"), ("Which artist painted the Mona Lisa?", "Da Vinci"),
    ("What is the boiling point of water in Celsius?", "100"), ("How many wings does a bee have?", "4"),
    ("Which element has the atomic number 1?", "Hydrogen"), ("What is the capital of Egypt?", "Cairo"),
    ("How many minutes are in a full day?", "1440"), ("Which desert is known for sand dunes and camels?", "Sahara"),
    ("How many players are on a basketball team on court?", "5"),
    ("What year did humans first land on the Moon?", "1969"), ("Which animal is known as the 'King of the Jungle'?", "Lion"),
    ("What is the national currency of Brazil?", "Real"), ("How many strings does a standard piano have?", "88"),
    ("Which planet do we call home?", "Earth"), ("What is the most spoken language in the world by total speakers?", "Mandarin"),
    ("How many bones are in the adult human body?", "206"), ("What is the speed of sound in air roughly, in m/s?", "343"),
    ("Which country is home to the kangaroo?", "Australia"), ("What is the largest planet in our solar system?", "Jupiter"),
    ("How many hours are in a week?", "168"), ("Which metal is used in pennies in the US?", "Zinc"),
    ("What is the capital of France?", "Paris"), ("How many legs does a spider have?", "8"),
    ("What year was the Berlin Wall torn down?", "1989"), ("Which element has the chemical symbol 'Ag'?", "Silver"),
    ("What is the largest land animal?", "African elephant"), ("How many periods are in the periodic table?", "7"),
    ("What is the name of the galaxy we live in?", "Milky Way"), ("Which instrument has 88 keys?", "Piano"),
    ("What is the freezing point of water in Fahrenheit?", "32"), ("How many states are in the United States?", "50"),
    ("What is the capital of Germany?", "Berlin"), ("Which planet is famous for its rings?", "Saturn"),
    ("How many sides does a triangle have?", "3"), ("What is the only metal that is liquid at room temperature?", "Mercury"),
    ("What year did the first Olympic Games take place in modern history?", "1896"), ("Which country is famous for sushi?", "Japan"),
    ("What is the largest bird that cannot fly?", "Ostrich"), ("How many hearts does an octopus have?", "3"),
    ("What is the capital of South Africa's legislative branch?", "Cape Town"), ("Which is the longest river in the world?", "Nile"),
    ("What is the chemical formula for water?", "H2O"), ("How many lines are on a standard music staff?", "5"),
    ("What year did the first Super Bowl take place?", "1967"), ("Which planet rotates on its side?", "Uranus"),
    ("What is the most abundant element in the universe?", "Hydrogen"), ("How many time zones does France span, including territories?", "12"),
    ("What is the smallest country in the world?", "Vatican City"), ("Which animal can change its color?", "Chameleon"),
    ("How many strings does a harp typically have?", "47"), ("What is the capital of New Zealand?", "Wellington"),
    ("What year did the internet become widely used by the public?", "1991"), ("Which is the largest reptile?", "Komodo dragon"),
    ("How many cubes make a 3x3x3 Rubik's cube?", "27"), ("What is the only continent that spans all four hemispheres?", "Africa"),
    ("What is the national sport of Japan?", "Sumo"), ("How many degrees are in a straight line?", "180"),
    ("Which gas do humans exhale in large quantities?", "Carbon dioxide"), ("What is the capital of Turkey?", "Ankara"),
    ("How many sides does a dodecagon have?", "12"), ("What year did the first human walk on the Moon?", "1969"),
    ("What is the fastest land animal?", "Cheetah"), ("How many continents does the polar bear live on?", "1")
)

WYRR_PAIRS: tuple[tuple[str, str], ...] = (
    ("Never use the internet again", "Never use a phone again"),
    ("Speak your thoughts out loud always", "Only be able to whisper"),
    ("Run every errand yourself", "Someone narrates every errand they run for you"),
    ("Know when someone is lying to you", "Know when you are lying to yourself"),
    ("One perfect meal every day, same dish", "Different meals, 30% worse quality"),
    ("Answer every DM you send ever gets", "Never answer a single one again"),
    ("Always know the time without a clock", "Always know the date without a calendar"),
    ("Free flights for life, economy only", "Free food for life, fast food only"),
    ("A pet dragon", "A pet phoenix"),
    ("Live in a castle with no wifi", "Live in a loft with unlimited wifi"),
)

AUCTION_ITEMS: tuple[tuple[str, int, int], ...] = (
    ("a 1977 vinyl of your favorite album", 120, 300),
    ("a signed jersey from a rival team", 80, 260),
    ("a vintage mechanical keyboard", 150, 420),
    ("a hand-thrown ceramic dinner set", 100, 320),
    ("a first-edition sci-fi paperback", 200, 500),
    ("a copper desk lamp from 1968", 90, 280),
    ("a framed map of a city you've never visited", 60, 240),
    ("a brass telescope (unverified)", 180, 480),
)

SPY_WORDS: tuple[tuple[str, str], ...] = (
    ("lighthouse", "light"), ("volcano", "fire"), ("submarine", "water"),
    ("library", "book"), ("airplane", "sky"), ("chocolate", "sweet"),
    ("desert", "sand"), ("penguin", "cold"), ("guitar", "string"),
    ("thunderstorm", "storm"), ("labyrinth", "maze"), ("treasure", "gold"),
    ("candle", "wax"), ("ocean", "salt"), ("elephant", "tusks"),
    ("pizza", "cheese"), ("snowman", "winter"), ("beach", "sand"),
    ("castle", "knight"), ("drum", "beat"), ("rainbow", "colors"),
    ("spider", "web"), ("rocket", "space"), ("cactus", "thorn"),
    ("pumpkin", "jack"), ("sailboat", "wind"), ("firefighter", "truck"),
    ("cheese", "mouse"), ("mountain", "peak"), ("piano", "keys"),
    ("pencil", "lead"), ("dolphin", "jelly"), ("knight", "armor"),
    ("cherry", "pit"), ("volleyball", "net"), ("pirate", "plank"),
    ("caterpillar", "butterfly"), ("lightning", "flash")
)

TRUTHS: tuple[str, ...] = (
    "I once ran a marathon and stopped for a hot dog.", "I can name every prime number under 30.",
    "I have never broken a phone screen.", "I once paid for coffee with coins I'd saved for a year.",
    "I can whistle with my eyes closed and stay in tune.", "I have read a whole book in one sitting.",
    "I once won a local chess tournament.", "I can boil water using only a microwave.",
    "I have slept in an airport.", "I once typed with one hand for a month without noticing.",
    "I can identify a song after one bar.", "I have never been lost in a big city.",
    "I once set an alarm for 3 a.m. on purpose and actually woke up.", "I have typed the word 'the' with two fingers for years without retraining.",
    "I once won a staring contest by blinking less than once a minute.", "I can recite the alphabet backwards starting from the G.",
    "I have eaten a meal with every utensil at the same time.", "I once memorized a 10-digit number just to prove a point.",
    "I have given a full presentation in a dead serious voice at 2 a.m. to an empty room.", "I once counted every step in a building and hit exactly 300.",
    "I can whistle a melody with just my nose pressed closed.", "I once finished a crossword in under ten minutes.",
    "I have slept through a thunderstorm and not once moved.", "I once typed a whole grocery list from memory and got it all right.",
    "I can name the first seven prime numbers without pausing.", "I once held my breath for a full minute and half.",
    "I have read an entire novel in a single airport layover.", "I once won a spelling bee by luck and preparation.",
    "I can fold a fitted sheet into a neat rectangle on the first try.", "I once cycled a whole loop around a park without stopping.",
    "I have memorized the route of a bus I barely ever took.", "I once played a full round of chess in under twenty minutes.",
    "I can identify a coin by its sound from across a quiet room.", "I once kept a journal for a whole year without missing a single day.",
    "I have eaten a whole pizza by myself and felt no guilt.", "I once stayed up all night to watch a meteor shower and saw one.",
    "I can do a full push-up without touching the floor.", "I once solved a Rubik's cube in under five minutes.",
    "I have walked a full city block counting every green light.", "I once wrote a poem in ten minutes on the spot."
)

LIES: tuple[str, ...] = (
    "I once argued with a vending machine and won.", "I can juggle three tomatoes without dropping one.",
    "I once won a hot dog eating contest in Lagos.", "I have met a penguin that learned my name.",
    "I can solve a Rubik's cube in under two minutes.", "I once high-fived a robot.",
    "I have eaten a whole watermelon in one sitting.", "I can recite the alphabet backwards in my sleep.",
    "I once raced a goat and placed second.", "I have spoken to a parrot that only said yes.",
    "I once rode a horse in a parade.", "I have met a famous actor at a grocery store and they shook my hand.",
    "I once won a dance-off in a shopping mall.", "I can speak four languages fluently.",
    "I once held a live lizard as a pet for a week.", "I have seen a shooting star land in a neighbor's yard.",
    "I once performed stand-up at a local open mic and got a standing ovation.", "I can touch my nose with my tongue.",
    "I once beat a professional at chess in one game.", "I have survived a volcanic eruption as a child.",
    "I once won a national spelling bee.", "I can name all fifty US states in under a minute.",
    "I have swum across a lake in one continuous effort.", "I once taught a dog to open doors.",
    "I once built a treehouse entirely from driftwood.", "I can do a backflip without practicing.",
    "I once won a karaoke contest for 'Best Voice'.", "I have held the world record for slowest 100-meter sprint.",
    "I once wrote a novel before turning twenty.", "I can balance a spoon on my nose for a full minute.",
    "I have been on a reality TV show without anyone knowing.", "I once tamed a stray cat in under an hour."
)


def _word_of(text: str) -> str:
    m = _WORD_RE.fullmatch(text.strip().lower().strip(".!?"))
    return m.group(0) if m else ""


# ── 1. word chain ────────────────────────────────────────────────────────────

class WordChainGame(MultiGame):
    name = "wordchain"
    description = "each word starts with the last letter of the last one"
    min_players = 1
    max_players = 8
    ai_seats = 1
    move_timeout = 90
    rules = ("Say a word that starts with the last letter of the previous "
             "word. No repeats. Three misses and you're out — last player "
             "standing wins.")

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        return {"last": "city", "used": ["city"],
                "misses": {}, "eliminated": []}

    def setup(self, room, mind):
        return ("word chain — the table starts with **city**. "
                "your word must start with Y. go.")

    def _miss(self, room: Room, player: Player, why: str) -> list[str]:
        room.state["misses"][player.key] = \
            int(room.state["misses"].get(player.key, 0)) + 1
        left = 3 - room.state["misses"][player.key]
        if left <= 0:
            room.state["eliminated"].append(player.name)
            room.players = [p for p in room.players if p.key != player.key]
            room.state["misses"].pop(player.key, None)
            if not room.humans:
                room.state["done"] = True
                return [f"{player.name} got stuck three times and is "
                        "out. the house takes the chain. 🏁"]
            return [f"{player.name} is out — three misses. "
                    f"last letter: {room.state['last'][-1]}."]
        return [f"{why} ({left} misses left)."]

    def on_move(self, room, player, text, mind):
        word = _word_of(text)
        if not word:
            return ["one word at a time — letters only."]
        if word in room.state["used"]:
            return [f"{word} was already said. pick a fresh one."]
        if not word.startswith(room.state["last"][-1]):
            return self._miss(
                room, player,
                f"it has to start with “{room.state['last'][-1]}”")
        if len(word) < 2:
            return self._miss(room, player, f"“{word}” is too short")
        from ..lexicon import is_real_word
        if not is_real_word(word):
            return self._miss(room, player, f"“{word}” is not a real word")
        room.state["used"].append(word)
        room.state["last"] = word
        return [f"good — **{word}**. next letter: {word[-1]}."]

    def ai_turn(self, room, mind):
        from ..lexicon import WORDSET, words_starting_with
        me = room.current
        last = room.state["last"]
        used = set(room.state["used"])
        # model first, bank second — but the model's pick must be a
        # real, fresh word or we fall back to the seeded index
        word = ""
        reply = mind.word_starting_with(last[-1], ())
        if (reply and reply not in used and reply in WORDSET()
                and reply.startswith(last[-1])):
            word = reply
        if not word:
            options = [w for w in words_starting_with(last[-1])
                       if w not in used]
            if options:
                word = mind.rng.choice(options)
        if not word:
            room.state["eliminated"].append(me.name)
            room.players = [p for p in room.players if p.key != me.key]
            if not any(not p.is_ai for p in room.players):
                room.state["done"] = True
                return ["the house is stuck — and so was everyone. "
                        "🏁 the house takes the chain."]
            return [f"the house is stuck on “{last[-1]}” and drops out."]
        room.state["used"].append(word)
        room.state["last"] = word
        return [f"house: **{word}** — your letter: {word[-1]}."]

    def is_over(self, room):
        return room.status == "finished" or room.state.get("done", False)

    def winner(self, room):
        if not room.state.get("done"):
            return None
        winners = [p for p in room.players if not p.is_ai]
        if len(winners) == 1:
            return winners[0]
        if len(winners) > 1:
            return "draw"
        return Player(key="ai:wordchain", platform="ai", name="The House",
                      is_ai=True)

    def describe_state(self, room):
        return (f"last word: {room.state['last']} · "
                f"out: {', '.join(room.state['eliminated']) or 'nobody'}")


# ── 2. hangman ───────────────────────────────────────────────────────────────

class HangmanGame(MultiGame):
    name = "hangman"
    description = "guess the word letter by letter — 6 wrongs ends it"
    min_players = 1
    max_players = 6
    ai_seats = 1
    move_timeout = 60
    rules = ("One letter per turn, or 'word <guess>' to try the whole word. "
             "Six wrong letters and the word is revealed — if the table "
             "cracks it first, the table wins.")

    def new_state(self, rng: random.Random, **kw: Any) -> dict[str, Any]:
        daily = bool(kw.get("daily"))
        if daily:
            word = hangman_daily_word()
            category = next(c for w, c in HANGMAN_WORDS if w == word)
        else:
            word, category = rng.choice(HANGMAN_WORDS)
        return {"word": word, "category": category, "revealed": [],
                "wrong": 0, "max_wrong": 6, "found_by": [], "daily": daily}

    def setup(self, room, mind):
        s = self._board(room)
        daily = " — today's word, same for everyone" if room.state.get("daily") else ""
        return (f"hangman — a {len(room.state['word'])}-letter word, "
                f"category: {room.state['category']}{daily}.\n{s}\n"
                "letters, or 'word <guess>'. the house guesses too.")

    @staticmethod
    def _board(room: Room) -> str:
        s = room.state
        shown = " ".join(c if c in s["revealed"] else "·"
                         for c in s["word"])
        wrongs = "✗" * s["wrong"] + " " * (s["max_wrong"] - s["wrong"])
        return f"{shown}\n[{wrongs}]  ({s['max_wrong'] - s['wrong']} left)"

    def _check_done(self, room: Room) -> str | None:
        s = room.state
        if all(c in s["revealed"] for c in s["word"]):
            return f"🏁 cracked — **{s['word']}**. the table wins."
        if s["wrong"] >= s["max_wrong"]:
            return f"the gallows is full. it was **{s['word']}**."
        return None

    def on_move(self, room, player, text, mind):
        s = room.state
        t = text.strip().lower()
        if t.startswith("word "):
            guess = _word_of(t[5:])
            if guess == s["word"]:
                room.state["done"] = "table"
                return [f"🏁 **{s['word']}** — {player.name} nailed it. "
                        "table wins."]
            s["wrong"] += 1
            out = [f"not it — that costs a letter. {self._board(room)}"]
            fin = self._check_done(room)
            if fin:
                room.state["done"] = "house"
                out.append(fin)
            return out
        letter = t if (len(t) == 1 and t.isalpha()) else ""
        if not letter:
            return ["one letter (a–z), or 'word <guess>'."]
        if letter in s["revealed"]:
            return [f"“{letter}” is already on the board."]
        s["revealed"].append(letter)
        if letter in s["word"]:
            s["found_by"].append(f"{letter}←{player.name}")
            out = [f"yes, {letter}! {self._board(room)}"]
        else:
            s["wrong"] += 1
            out = [f"no {letter}. {self._board(room)}"]
        fin = self._check_done(room)
        if fin:
            room.state["done"] = "table" if "cracked" in fin else "house"
            out.append(fin)
        return out

    def ai_turn(self, room, mind):
        s = room.state
        letter = mind.letter_guess(
            set(s["revealed"]), len(s["word"]), s["category"],
            priority=HANGMAN_LETTER_PRIORITY.get((s["category"], len(s["word"]))))
        if letter in s["revealed"]:
            return []
        s["revealed"].append(letter)
        if letter in s["word"]:
            out = [f"house finds {letter}. {self._board(room)}"]
            fin = self._check_done(room)
            if fin:
                # the board is complete — by the rules a cracked word is
                # the table's win, even when the house's letter was the last
                room.state["done"] = "table"
                out.append(fin)
            return out
        s["wrong"] += 1
        out = [f"house misses {letter}. {self._board(room)}"]
        fin = self._check_done(room)
        if fin:
            room.state["done"] = "house"
            out.append(fin)
        return out

    def is_over(self, room):
        return "done" in room.state

    def winner(self, room):
        if room.state.get("done") == "table":
            if not room.humans:
                return None
            if len(room.humans) == 1:
                return room.humans[0]
            # group table: credit the player who cracked the most letters
            counts: dict[str, int] = {}
            for entry in room.state.get("found_by", []):
                name = entry.split("←", 1)[-1]
                counts[name] = counts.get(name, 0) + 1
            top = max(counts.items(), key=lambda kv: kv[1], default=None)
            for h in room.humans:
                if top is not None and h.name == top[0]:
                    return h
            return "draw"
        if room.state.get("done") == "house":
            return Player(key="ai:hangman", platform="ai",
                          name="The House", is_ai=True)
        return None


# ── 3. number guess battle ───────────────────────────────────────────────────

class NumberGuessGame(MultiGame):
    name = "numberguess"
    description = "both of you pick a secret 1–100 — first to crack the other's wins"
    min_players = 1
    max_players = 1
    ai_seats = 1
    move_timeout = 120
    rules = ("You both secretly pick a number 1–100. Then you trade "
             "guesses: you guess mine, I guess yours, with high/low "
             "feedback. First to hit the other's exact number wins. "
             "The house plays perfect bisection — bring it.")

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        return {"phase": "pick", "p_secret": None, "a_secret": None,
                "p_lo": 1, "p_hi": 100,   # house's window on your number
                "a_lo": 1, "a_hi": 100,   # your window on the house number
                "rounds": 0, "max_rounds": 14}

    def setup(self, room, mind):
        return ("number battle — pick your secret number, 1 to 100. "
                "(i'm picking mine right now.)")

    def on_move(self, room, player, text, mind):
        s = room.state
        digits = re.sub(r"\D", "", text)
        n = int(digits) if digits else None
        if s["phase"] == "pick":
            if n is None or not 1 <= n <= 100:
                return ["a number from 1 to 100 — that's the whole game."]
            s["p_secret"] = n
            s["a_secret"] = mind.rng.randint(1, 100)
            s["phase"] = "battle"
            room.state["rounds"] = 0
            return [f"locked: your {n} vs my ?? — round 1. guess my number "
                    "(i'll guess yours right after)."]
        if n is None or not 1 <= n <= 100:
            return ["a number from 1 to 100."]
        s["rounds"] += 1
        out: list[str] = []
        # the player guesses the house number
        if n == s["a_secret"]:
            room.state["done"] = "player"
            out.append(f"🏁 {n} — you cracked my number in "
                       f"{s['rounds']} rounds. you win.")
            return out
        if n < s["a_secret"]:
            s["a_lo"] = max(s["a_lo"], n + 1)
            fb = "too low"
        else:
            s["a_hi"] = min(s["a_hi"], n - 1)
            fb = "too high"
        out.append(f"your {n}: {fb}.")
        # the house guesses the player's number (perfect bisection)
        guess = mind.number(s["p_lo"], s["p_hi"])
        if guess == s["p_secret"]:
            room.state["done"] = "house"
            out.append(f"house guessed **{guess}** — that's your number. "
                       "house wins.")
            return out
        if guess < s["p_secret"]:
            s["p_lo"] = max(s["p_lo"], guess + 1)
            hfb = "too low"
        else:
            s["p_hi"] = min(s["p_hi"], guess - 1)
            hfb = "too high"
        out.append(f"house guesses {guess}: {hfb}. "
                   f"window: {s['a_lo']}–{s['a_hi']}. next guess.")
        if s["rounds"] >= s["max_rounds"]:
            room.state["done"] = "draw"
            out.append("twelve rounds and nobody bled — draw.")
        return out

    def ai_turn(self, room, mind):
        return []  # the house guesses inside the player's move

    def is_over(self, room):
        return "done" in room.state

    def winner(self, room):
        done = room.state.get("done")
        if done == "player":
            return room.humans[0] if room.humans else None
        if done == "house":
            return Player(key="ai:numberguess", platform="ai",
                          name="The House", is_ai=True)
        return "draw"

    def describe_state(self, room):
        s = room.state
        if s["phase"] == "pick":
            return "waiting for your secret number."
        return (f"round {s['rounds']}/{s['max_rounds']} · your window: "
                f"{s['a_lo']}–{s['a_hi']}")


# ── 4. two truths and a lie ──────────────────────────────────────────────────

class TwoTruthsGame(MultiGame):
    name = "two_truths"
    description = "3 statements, 1 lie — the table votes which one"
    min_players = 1
    max_players = 8
    ai_seats = 1
    move_timeout = 180
    rules = ("Each seat in turn posts 3 statements (one per line, or "
             "separated by ;), one of them a lie. Every other seat votes "
             "1–3 for the lie, then the poster reveals it. Voters who "
             "caught the lie +1; a lie that escapes the vote +2 to the "
             "liar. 3 rounds, most points wins.")

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        return {"round": 1, "rounds": 3, "phase": "posting",
                "poster_idx": 0, "order": [], "posts": {}, "votes": {},
                "points": {}, "lie": None}

    def setup(self, room, mind):
        s = room.state
        s["order"] = [p.key for p in room.players]
        cur = room.players[0]
        who = "the house" if cur.is_ai else f"{cur.name}"
        return (f"two truths & a lie — 3 rounds. {who}, post 3 statements "
                "(one per line), one of them a lie.")

    def _poster(self, room: Room) -> Player | None:
        s = room.state
        if not s["order"]:
            return None
        return room.player(s["order"][s["poster_idx"] % len(s["order"])])

    def _start_voting(self, room: Room) -> None:
        s = room.state
        s["phase"] = "voting"
        s["votes"] = {}

    def _advance_round(self, room: Room) -> None:
        s = room.state
        s["phase"] = "posting"
        s["posts"] = {}
        s["votes"] = {}
        s["lie"] = None
        s["poster_idx"] += 1

    def _resolve(self, room: Room) -> list[str]:
        s = room.state
        poster = self._poster(room)
        post = s["posts"].get(poster.key if poster else "", {})
        items = post.get("items", [])
        lie = s.get("lie")
        votes = list(s["votes"].values())
        lines = []
        if items and lie:
            lines.append(f"reveal: statement {lie} was the lie — "
                         f"“{items[lie - 1]}”")
            top = max(set(votes), key=votes.count) if votes else None
            if top == lie:
                for v, n in s["votes"].items():
                    if n == lie:
                        p = room.player(v)
                        if p and not p.is_ai:
                            s["points"][v] = int(s["points"].get(v, 0)) + 1
                lines.append("the table caught it — voters +1.")
            else:
                if poster is not None and not poster.is_ai:
                    s["points"][poster.key] = \
                        int(s["points"].get(poster.key, 0)) + 2
                lines.append("the lie got away — poster +2.")
        if s["round"] >= s["rounds"]:
            s["done"] = True
            best = max(s["points"].items(), key=lambda kv: kv[1],
                       default=("", 0))
            w = room.player(best[0]) if best[0] else None
            lines.append("🏁 final — " +
                         (f"{w.name} wins with {best[1]} pts."
                          if (w and not w.is_ai and best[1] > 0)
                          else "an even table."))
        else:
            s["round"] += 1
            self._advance_round(room)
            nxt = self._poster(room)
            if nxt is not None:
                lines.append(f"{nxt.name}, your 3 statements (one is a "
                             "lie).")
        return lines

    def on_move(self, room, player, text, mind):
        s = room.state
        poster = self._poster(room)
        if s["phase"] == "posting":
            parts = [x.strip() for x in re.split(r";|\n", text) if x.strip()]
            if len(parts) != 3:
                return ["exactly 3 statements — one per line or "
                        "separated by ;."]
            s["posts"][player.key] = {"items": parts, "name": player.name}
            self._start_voting(room)
            return [f"{player.name} posts:\n" +
                    "\n".join(f"{i+1}. {x}" for i, x in enumerate(parts)) +
                    "\nthe table votes 1–3 for the lie, in turn."]
        if s["phase"] == "voting":
            if poster is not None and player.key == poster.key:
                return []  # the poster watches the votes
            n = re.sub(r"\D", "", text)
            if not (n and 1 <= int(n) <= 3):
                return ["vote 1, 2, or 3 — which statement is the lie?"]
            s["votes"][player.key] = int(n)
            out = [f"{player.name} votes {n}."]
            if all(p.key in s["votes"] or
                   (poster is not None and p.key == poster.key)
                   for p in room.players):
                # everyone in: resolve now if the lie is known (AI poster)
                if s.get("lie") is not None:
                    out.extend(self._resolve(room))
                else:
                    s["phase"] = "reveal"
                    out.append(f"{poster.name if poster else 'poster'} — "
                               "which one was the lie? just the number "
                               "(votes are already in).")
            return out
        if s["phase"] == "reveal":
            n = re.sub(r"\D", "", text)
            if not (n and 1 <= int(n) <= 3):
                return ["which one was the lie — 1, 2, or 3?"]
            s["lie"] = int(n)
            return self._resolve(room)
        return []

    def ai_turn(self, room, mind):
        s = room.state
        me = room.current
        poster = self._poster(room)
        if s["phase"] == "posting" and poster is not None and \
                poster.key == me.key:
            truths = mind.rng.sample(TRUTHS, 2)
            lie = mind.rng.choice(LIES)
            items = truths + [lie]
            mind.rng.shuffle(items)
            s["posts"][me.key] = {"items": items, "name": me.name,
                                  "lie": items.index(lie) + 1}
            s["lie"] = s["posts"][me.key]["lie"]
            out = [f"{me.name} posts:\n" +
                   "\n".join(f"{i+1}. {x}" for i, x in enumerate(items))]
            self._start_voting(room)
            out.append("the table votes 1–3 for the lie, in turn.")
            return out
        if s["phase"] == "voting" and me.key not in s["votes"] and \
                poster is not None and me.key != poster.key:
            post = s["posts"].get(poster.key, {})
            items = post.get("items", [])
            pick = 1
            if s.get("lie") is not None and \
                    s["posts"].get(me.key, {}).get("items"):
                pass  # the AI never votes on its own post
            elif s.get("lie") is not None:
                pick = s["lie"]  # an AI seat voting on another AI's post
            elif items and mind.model_on:
                menu = "\n".join(f"{i+1}. {x}"
                                  for i, x in enumerate(items))
                reply = mind.ask(f"Two truths and a lie — these 3 "
                                 f"statements, one is a lie:\n{menu}\n"
                                 "Which number is the lie? Reply with ONLY "
                                 "the number.")
                if reply.strip().isdigit() and 1 <= int(reply) <= 3:
                    pick = int(reply)
            s["votes"][me.key] = pick
            out = [f"{me.name} votes {pick}."]
            if all(p.key in s["votes"] or
                   (poster is not None and p.key == poster.key)
                   for p in room.players):
                if s.get("lie") is not None:
                    out.extend(self._resolve(room))
                else:
                    s["phase"] = "reveal"
                    out.append(f"{poster.name} — which one was the lie? "
                               "just the number.")
            return out
        if s["phase"] == "reveal" and s.get("lie") is not None:
            return self._resolve(room)
        return []

    def is_over(self, room):
        return room.state.get("done", False)

    def winner(self, room):
        best = max(room.state.get("points", {}).items(),
                   key=lambda kv: kv[1], default=("", 0))
        w = room.player(best[0]) if best[0] else None
        return w if (w and not w.is_ai and best[1] > 0) else "draw"

    def score(self, room, player):
        return int(room.state.get("points", {}).get(player.key, 0))

    def describe_state(self, room):
        s = room.state
        return (f"round {s['round']}/{s['rounds']} · phase: {s['phase']} · "
                f"points: {s['points'] or 'none yet'}")


# ── 5. would you rather (table edition) ─────────────────────────────────────

class WyrrGame(MultiGame):
    name = "wyrr"
    description = "the house poses, the table picks, everyone sees who's who"
    min_players = 1
    max_players = 8
    ai_seats = 1
    move_timeout = 60
    rules = ("Would you rather — each player in turn answers 1 or 2. "
             "When the table has all answered, the picks are revealed "
             "and the next pair lands. 5 rounds, participation pays.")

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        pairs = rng.sample(list(WYRR_PAIRS), 5)
        return {"pairs": pairs, "idx": 0, "answers": {},
                "rounds": 5}

    def setup(self, room, mind):
        a, b = room.state["pairs"][0]
        names = ", ".join(p.name for p in room.players)
        return (f"would you rather, {len(room.players)} at the table "
                f"({names}).\n1: {a}\n2: {b}\n{room.players[0].name}, "
                "your pick?")

    def on_move(self, room, player, text, mind):
        s = room.state
        t = text.strip().lower()
        if t not in {"1", "2", "one", "two", "a", "b"}:
            return ["1 or 2 — the whole menu."]
        s["answers"][player.key] = 1 if t in {"1", "one", "a"} else 2
        out = [f"{player.name}: {s['answers'][player.key]}"]
        pending = [p for p in room.players
                   if p.key not in s["answers"] and not p.is_ai]
        if pending:
            out.append(f"{pending[0].name}, your pick?")
        elif self._all_answered(room):
            out.extend(self._next_round(room, mind))
        return out

    def _all_answered(self, room: Room) -> bool:
        return all(p.key in room.state["answers"] for p in room.players)

    def _next_round(self, room: Room, mind: GameMind) -> list[str]:
        s = room.state
        a, b = s["pairs"][s["idx"]]
        lines = ["the table says:"]
        for p in room.players:
            pick = s["answers"].get(p.key)
            if pick == 1:
                lines.append(f"  {p.name} → {a[:40]}")
            elif pick == 2:
                lines.append(f"  {p.name} → {b[:40]}")
        s["answers"] = {}
        s["idx"] += 1
        if s["idx"] >= s["rounds"]:
            room.state["done"] = True
            lines.append("five rounds of character revealed. 🏁")
            return lines
        a2, b2 = s["pairs"][s["idx"]]
        lines.append(f"\nnext — 1: {a2}\n2: {b2}\n{room.players[0].name}, "
                     "your pick?")
        return lines

    def ai_turn(self, room, mind):
        s = room.state
        if s.get("done") or s["answers"].get(room.current.key) is not None:
            return []
        a, b = s["pairs"][s["idx"]]
        pick = 1 if mind.choice([a, b], "would you rather") == a else 2
        s["answers"][room.current.key] = pick
        out = [f"house: {pick}"]
        if self._all_answered(room):
            out.extend(self._next_round(room, mind))
        return out

    def is_over(self, room):
        return room.state.get("done", False)

    def winner(self, room):
        return "draw"  # participation game — the ledger gets the points

    def score(self, room, player):
        return room.state["rounds"]


# ── 6. spy (the house holds a word you never see) ────────────────────────────

class SpyGame(MultiGame):
    name = "spy"
    description = "the house knows a word you never see — bluff your way out"
    min_players = 1
    max_players = 1
    ai_seats = 1
    move_timeout = 120
    rules = ("I'm holding a word. I'll describe it in one word — you never "
             "see mine. You describe *your* guess of it in one word. If "
             "yours matches the real word (or its family), you pass as an "
             "innocent. Three rounds — two clean rounds and you walk free.")

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        picks = rng.sample(list(SPY_WORDS), 3)
        return {"rounds": picks, "round": 0, "wins": 0, "needed": 2}

    def setup(self, room, mind):
        word, assoc = room.state["rounds"][0]
        room.state["current_word"], room.state["current_assoc"] = word, assoc
        return (f"spy round 1/3 — my word: **{assoc}** "
                f"(that's my one-word description of it).\n"
                f"you don't see the word itself. describe your guess in "
                f"ONE word.")

    def _round_line(self, room: Room) -> str:
        word, assoc = room.state["rounds"][room.state["round"]]
        room.state["current_word"], room.state["current_assoc"] = word, assoc
        return (f"round {room.state['round']+1}/3 — my description: "
                f"**{assoc}** — your one word?")

    def on_move(self, room, player, text, mind):
        s = room.state
        word = s["current_word"]
        assoc = s["current_assoc"]
        guess = _word_of(text)
        if not guess:
            return ["one word — that's the spy's whole craft."]
        hit = guess == word or guess == assoc or word in guess or \
            guess in word
        if mind.model_on:
            reply = mind.ask(
                f"A spy game. The hidden word is '{word}' (described as "
                f"'{assoc}'). The spy said '{guess}'. Did the spy clearly "
                "know the word? Reply ONLY 'yes' or 'no'.")
            if reply.strip().lower().startswith("no"):
                hit = False
            elif reply.strip().lower().startswith("yes"):
                hit = True
        if hit:
            s["wins"] += 1
            s["round"] += 1
            out = [f"{player.name} says **{guess}** — clean. you walk "
                   f"through that door. ({s['wins']}/{s['needed']})"]
            if s["wins"] >= s["needed"]:
                s["done"] = "player"
                out.append("🏁 the house never caught you — you're not "
                           "the spy. (or you're a very good one.)")
            elif s["round"] >= len(s["rounds"]):
                s["done"] = "house"
                out.append("the rounds ran out — not enough clean "
                           "rounds. the house calls you the spy. 🏁")
            else:
                out.append(self._round_line(room))
            return out
        s["round"] += 1
        out = [f"**{guess}** — the house frowns. the real word was "
              f"**{word}**. that round is on the record."]
        if s["round"] >= len(s["rounds"]):
            s["done"] = "house"
            out.append("🏁 three strikes — the house exposes you as the "
                       "spy.")
        else:
            out.append(self._round_line(room))
        return out

    def ai_turn(self, room, mind):
        return []

    def is_over(self, room):
        return "done" in room.state

    def winner(self, room):
        if room.state.get("done") == "player":
            return room.humans[0] if room.humans else None
        if room.state.get("done") == "house":
            return Player(key="ai:spy", platform="ai", name="The House",
                          is_ai=True)
        return None


# ── 7. auction ───────────────────────────────────────────────────────────────

class AuctionGame(MultiGame):
    name = "auction"
    description = "3 hidden-value lots — bid, outbid, or eat the loss"
    min_players = 1
    max_players = 6
    ai_seats = 2
    move_timeout = 90
    rules = ("Each round I auction a lot with a hidden value. Bid 'bid "
             "<coins>' or 'pass'. When everyone but the top bidder has "
             "passed, the lot is theirs — value above the bid is profit, "
             "value below is loss. Most profit after 3 lots wins. The "
             "house knows what it's buying.")

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        lots = rng.sample(list(AUCTION_ITEMS), 3)
        return {"lots": [(n, rng.randint(lo, hi)) for n, lo, hi in lots],
                "round": 0, "bids": {}, "passes": {}, "profit": {},
                "leader_key": None, "leader": 0}

    def setup(self, room, mind):
        name, _ = room.state["lots"][0]
        return (f"auction — lot 1/3: **{name}**.\n"
                f"bid 'bid <coins>' or 'pass'. the house is bidding too.")

    def _open_lot(self, room: Room) -> str:
        s = room.state
        name, _ = s["lots"][s["round"]]
        s["bids"], s["passes"] = {}, {}
        s["leader_key"], s["leader"] = None, 0
        return f"lot {s['round']+1}/{len(s['lots'])}: **{name}** — open " \
               "the bidding."

    def _all_passed(self, room: Room) -> bool:
        s = room.state
        for p in room.players:
            if s["leader_key"] is not None and p.key == s["leader_key"]:
                continue
            if p.key not in s["passes"]:
                return False
        return True

    def _resolve_lot(self, room: Room) -> list[str]:
        s = room.state
        name, value = s["lots"][s["round"]]
        out: list[str] = []
        if s["leader_key"] is None:
            out.append(f"everyone passed — {name} goes unsold.")
        else:
            leader_name = room.player(s["leader_key"])
            profit = value - s["leader"]
            s["profit"][s["leader_key"]] = \
                int(s["profit"].get(s["leader_key"], 0)) + profit
            out.append(
                f"{name} sold to {leader_name.name if leader_name else '?'} "
                f"for {s['leader']} — value was {value}. "
                f"{'profit +' if profit >= 0 else 'loss -'}{abs(profit)}.")
        s["round"] += 1
        if s["round"] >= len(s["lots"]):
            s["done"] = True
            best = max(s["profit"].items(), key=lambda kv: kv[1],
                       default=("", 0))
            w = room.player(best[0]) if best[0] else None
            out.append("🏁 final — " +
                       (f"{w.name} closes the auction with {best[1]} "
                        f"coins." if w and best[1] > 0
                        else "a quiet auction."))
        else:
            out.append(self._open_lot(room))
        return out

    def _act(self, room: Room, player: Player, text: str,
             mind: GameMind) -> list[str]:
        s = room.state
        t = text.strip().lower()
        if t == "pass":
            s["passes"][player.key] = True
            out = [f"{player.name} passes."]
            if self._all_passed(room):
                out.extend(self._resolve_lot(room))
            return out
        m = re.match(r"bid\s*(\d+)$", t)
        if not m:
            return ["'bid <coins>' or 'pass'."]
        amount = int(m.group(1))
        if amount <= s["leader"]:
            return [f"you have to clear {s['leader']} to be in it."]
        s["bids"][player.key] = amount
        s["leader"], s["leader_key"] = amount, player.key
        out = [f"{player.name} bids {amount}."]
        if self._all_passed(room):
            out.extend(self._resolve_lot(room))
        return out

    def on_move(self, room, player, text, mind):
        return self._act(room, player, text, mind)

    def ai_turn(self, room, mind):
        s = room.state
        me = room.current
        if me.key in s["passes"] or me.key == s["leader_key"]:
            return []
        name, value = s["lots"][s["round"]]
        est = value * (0.85 + 0.2 * mind.rng.random())
        if s["leader"] >= est or mind.rng.random() < 0.25:
            s["passes"][me.key] = True
            out = [f"{me.name} passes."]
        else:
            bid = mind.bid(s["leader"] + 10, int(value * 1.15), est)
            s["bids"][me.key] = bid
            s["leader"], s["leader_key"] = bid, me.key
            out = [f"{me.name} bids {bid}."]
        if self._all_passed(room):
            out.extend(self._resolve_lot(room))
        return out

    def is_over(self, room):
        return room.state.get("done", False)

    def winner(self, room):
        best = max(room.state.get("profit", {}).items(),
                   key=lambda kv: kv[1], default=("", 0))
        w = room.player(best[0]) if best[0] else None
        if w is None or best[1] <= 0:
            return "draw"
        return w

    def score(self, room, player):
        return int(room.state.get("profit", {}).get(player.key, 0))

    def describe_state(self, room):
        s = room.state
        name, _ = s["lots"][s["round"]]
        lead = ""
        if s["leader"] and s["leader_key"]:
            lp = room.player(s["leader_key"])
            lead = f" — top bid {s['leader']}" + (
                f" by {lp.name}" if lp else "")
        return f"lot {s['round']+1}/{len(s['lots'])}: {name}{lead}"


# ── 8. trivia royale ─────────────────────────────────────────────────────────

class TriviaRoyaleGame(MultiGame):
    name = "trivia"
    description = "8 rounds, 3 lives, streaks pay double — last brain standing"
    min_players = 1
    max_players = 8
    ai_seats = 1
    move_timeout = 60
    rules = ("8 questions, everyone in turn. Correct: 10 pts + 5×streak. "
             "Wrong: lose a life (3 total). Out of lives = eliminated. "
             "Most points at the end wins — the house answers at 80% "
             "confidence, so it bleeds sometimes. that's your window.")

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        qs = rng.sample(list(TRIVIA), 8)
        return {"questions": qs, "round": 0, "points": {}, "lives": {},
                "streak": {}, "q": None, "a": None, "total": 8}

    def setup(self, room, mind):
        for p in room.players:
            room.state["lives"][p.key] = 3
        q, a = room.state["questions"][0]
        room.state["q"], room.state["a"] = q, a
        return (f"trivia royale — {len(room.players)} seated, 8 questions, "
                f"3 lives each.\nQ1: {q}")

    def _next_question(self, room: Room) -> str:
        s = room.state
        s["round"] += 1
        if s["round"] >= s["total"]:
            s["done"] = True
            return "the questions are done."
        q, a = s["questions"][s["round"]]
        s["q"], s["a"] = q, a
        return f"Q{s['round']+1}: {q}"

    def _check_done(self, room: Room) -> str | None:
        s = room.state
        alive = [k for k, v in s["lives"].items() if v > 0
                 and room.player(k) is not None]
        if s["round"] >= s["total"]:
            return None  # handled by _next_question
        if len([k for k in alive if not k.startswith("ai:")]) <= 1:
            s["done"] = True
            return "only one seat left at the table."
        return None

    def on_move(self, room, player, text, mind):
        s = room.state
        if s["lives"].get(player.key, 0) <= 0:
            return [f"{player.name} is already out of this one."]
        answer = s["a"].lower()
        guess = text.strip().lower()
        right = answer in guess or guess in answer
        out: list[str] = []
        if right:
            s["streak"][player.key] = int(s["streak"].get(player.key, 0)) + 1
            pts = 10 + 5 * (s["streak"][player.key] - 1)
            s["points"][player.key] = int(s["points"].get(player.key, 0)) + pts
            out.append(f"correct — +{pts} (streak {s['streak'][player.key]})")
        else:
            s["streak"][player.key] = 0
            s["lives"][player.key] -= 1
            left = s["lives"][player.key]
            out.append(f"nope — it was **{s['a']}**. "
                       f"{'out of lives!' if left <= 0 else f'{left} lives left.'}")
        if s["lives"][player.key] <= 0:
            out.append(f"{player.name} is eliminated.")
            fin = self._check_done(room)
            if fin:
                out.append(fin)
        line = self._next_question(room)
        if s.get("done"):
            best = max(s["points"].items(), key=lambda kv: kv[1],
                       default=("", 0))
            w = room.player(best[0]) if best[0] else None
            out.append("🏁 final scores — " +
                       (f"{w.name} wins with {best[1]} pts."
                        if w and best[1] else "an even table."))
        elif not line.startswith("the questions"):
            out.append(line)
        return out

    def ai_turn(self, room, mind):
        s = room.state
        me = room.current
        if s["lives"].get(me.key, 0) <= 0:
            return []
        # the house answers from the bank at 80% confidence — a real,
        # seeded handicap so the table has a window.
        correct = mind.rng.random() < 0.8
        if correct:
            s["streak"][me.key] = int(s["streak"].get(me.key, 0)) + 1
            pts = 10 + 5 * (s["streak"][me.key] - 1)
            s["points"][me.key] = int(s["points"].get(me.key, 0)) + pts
            out = [f"house: {s['a']} — +{pts}."]
        else:
            s["streak"][me.key] = 0
            s["lives"][me.key] -= 1
            out = [f"house fumbles it. {s['lives'][me.key]} lives left."]
        line = self._next_question(room)
        if s.get("done"):
            best = max(s["points"].items(), key=lambda kv: kv[1],
                       default=("", 0))
            w = room.player(best[0]) if best[0] else None
            out.append("🏁 final — " +
                       (f"{w.name} wins with {best[1]} pts."
                        if w and best[1] else "an even table."))
        elif not line.startswith("the questions"):
            out.append(line)
        return out

    def is_over(self, room):
        return room.state.get("done", False)

    def winner(self, room):
        best = max(room.state.get("points", {}).items(),
                   key=lambda kv: kv[1], default=("", 0))
        w = room.player(best[0]) if best[0] else None
        return w if (w and best[1] > 0) else "draw"

    def score(self, room, player):
        return int(room.state.get("points", {}).get(player.key, 0))

    def describe_state(self, room):
        s = room.state
        return (f"Q{s['round']+1}/{s['total']} · lives: " +
                " · ".join(f"{room.player(k).name if room.player(k) else k}: {v}"
                           for k, v in s["lives"].items()))


class TwentyQuestionsGame(MultiGame):
    """Ported from the legacy solo GamesAgent: the house thinks of
    something, the table asks yes/no questions (20 max)."""
    name = "20q"
    description = "i think of something, you ask yes/no questions (20)"
    min_players = 1
    max_players = 4
    ai_seats = 0
    move_timeout = 120
    rules = ("I'm thinking of something. Ask yes/no questions — 20 max. "
             "Guess with “guess: <thing>”. The table wins together.")

    BANK: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("a houseplant", ("plant", "leaf", "pot", "water", "green")),
        ("a mechanical keyboard", ("key", "click", "desk", "switch", "clack")),
        ("a thunderstorm", ("rain", "lightning", "loud", "sky", "weather")),
        ("a cat", ("fur", "meow", "pet", "paw", "milk")),
        ("a submarine", ("water", "deep", "metal", "ocean", "silent")),
        ("a library", ("book", "quiet", "shelf", "read", "card")),
        ("a lighthouse", ("light", "beam", "tower", "coast", "ship")),
        ("a vending machine", ("coin", "snack", "button", "drink", "slot")),
    )

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        return {"target": None, "hints": [], "asked": 0, "max": 20}

    def setup(self, room, mind):
        target, hints = room.rng().choice(self.BANK)
        room.state["target"] = target
        room.state["hints"] = list(hints)
        return (f"20 questions — i'm thinking of something. the table gets "
                f"{room.state['max']} yes/no questions. start anywhere "
                f"(“guess: …” to guess).")

    def on_move(self, room, player, text, mind):
        s = room.state
        word = text.strip().lower().rstrip("?")
        if "guess:" in word or word.startswith(("it's ", "its ")):
            guess = word.split(":", 1)[-1] if ":" in word else word[4:]
            s["asked"] += 1
            if s["target"] in guess:
                s["done"] = "table"
                return [f"guessed it in {s['asked']} questions — "
                        f"{s['target']}. well played."]
            left = s["max"] - s["asked"]
            if left <= 0:
                s["done"] = "house"
                return [f"out of questions. it was {s['target']}."]
            return [f"not that one. {left} questions left."]
        s["asked"] += 1
        truth = any(h in word for h in s["hints"]) or any(
            word in h for h in s["hints"])
        answer = mind.yes_no(truth, word)
        left = s["max"] - s["asked"]
        if left <= 0:
            s["done"] = "house"
            return [f"{answer}. out of questions — it was {s['target']}."]
        return [f"{answer}  ({left} left)"]

    def is_over(self, room):
        return room.status == "finished" or bool(room.state.get("done"))

    def winner(self, room):
        done = room.state.get("done")
        if done == "table":
            return room.humans[0] if room.humans else "draw"
        if done == "house":
            return "house"
        return None

    def score(self, room, player):
        return max(0, room.state.get("max", 20) - room.state.get("asked", 0))


class RpsGame(MultiGame):
    """Ported from the legacy solo GamesAgent: rock-paper-scissors,
    first to 3 against the house."""
    name = "rps"
    description = "rock paper scissors — first to 3"
    min_players = 1
    max_players = 1
    ai_seats = 0
    move_timeout = 60
    rules = "Type rock, paper, or scissors. First to 3 takes it."

    MOVES = ("rock", "paper", "scissors")
    BEATS = {"rock": "scissors", "scissors": "paper", "paper": "rock"}

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        return {"me": 0, "you": 0}

    def setup(self, room, mind):
        return "rock paper scissors, first to 3. type rock, paper or scissors."

    def on_move(self, room, player, text, mind):
        s = room.state
        guess = text.strip().lower()
        if guess not in self.MOVES:
            return ["rock, paper, or scissors — that's the whole menu."]
        mine = room.rng().choice(self.MOVES)
        if mine == guess:
            reply = f"both {mine}. again."
        elif self.BEATS[mine] == guess:
            s["me"] += 1
            reply = (f"i throw {mine}. mine! ({s['me']}-{s['you']})"
                     if s["me"] < 3 else
                     f"i throw {mine}. three — i take it.")
        else:
            s["you"] += 1
            reply = f"i throw {mine}. yours. ({s['you']}-{s['me']})"
        if s["me"] >= 3:
            s["done"] = "house"
            return [reply]
        if s["you"] >= 3:
            s["done"] = "table"
            return [f"{reply} you won 3-{s['me']} — respect."]
        return [reply]

    def is_over(self, room):
        return room.status == "finished" or bool(room.state.get("done"))

    def winner(self, room):
        done = room.state.get("done")
        if done == "table":
            return room.humans[0] if room.humans else "draw"
        if done == "house":
            return "house"
        return None

    def score(self, room, player):
        return int(room.state.get("you", 0))


class DigitMemoryGame(MultiGame):
    """Ported from the legacy solo GamesAgent's number-memory (named
    “digits” here — the engine already has a concentration game called
    “memory”)."""
    name = "digits"
    description = "i show digits, you repeat them — grows each round"
    min_players = 1
    max_players = 1
    ai_seats = 0
    move_timeout = 120
    rules = ("I show you digits, you type them back. Each round adds a "
             "digit. Survive 8 rounds.")

    def new_state(self, rng: random.Random) -> dict[str, Any]:
        return {"round": 0, "number": "", "best": 0}

    def setup(self, room, mind):
        room.state["round"] = 1
        room.state["number"] = "".join(
            room.rng().choice("0123456789") for _ in range(3))
        return (f"digits — repeat back: {room.state['number']} "
                f"(3 digits to start, grows each round)")

    def on_move(self, room, player, text, mind):
        s = room.state
        guess = "".join(ch for ch in text if ch.isdigit())
        target = str(s.get("number") or "")
        if guess == target and guess:
            s["round"] += 1
            s["best"] = max(s["best"], s["round"] - 1)
            s["number"] = "".join(room.rng().choice("0123456789")
                                  for _ in range(2 + s["round"]))
            if s["round"] > 8:
                s["done"] = "table"
                return [f"eight rounds?! best was {s['best']} — that's a "
                        f"brain, not a phone."]
            return [f"correct. now: {s['number']} "
                    f"({len(s['number'])} digits)"]
        s["done"] = "house"
        return [f"that's not it — it was {target}. "
                f"best round: {s['best']}."]

    def is_over(self, room):
        return room.status == "finished" or bool(room.state.get("done"))

    def winner(self, room):
        if room.state.get("done") == "table":
            return room.humans[0] if room.humans else "draw"
        if room.state.get("done") == "house":
            return "house"
        return None

    def score(self, room, player):
        return int(room.state.get("best", 0))


EASY_GAMES: tuple[MultiGame, ...] = (
    WordChainGame(), HangmanGame(), NumberGuessGame(), TwoTruthsGame(),
    WyrrGame(), SpyGame(), AuctionGame(), TriviaRoyaleGame(),
    TwentyQuestionsGame(), RpsGame(), DigitMemoryGame(),
)
