"""The games' shared lexicon.

A real, frequency-ordered English word list (google-10000-english,
cleaned) is shipped at ``nomorals/games/data/words.txt`` — 9,892
words.  Every word game in the suite validates against it:

* wordchain — a move only counts if it is a real word
* wordle    — answers and guesses are 5-letter entries from the set
* hangman/spy — word pools are drawn from the same vocabulary family

The loader is lazy (first use) and case-insensitive.  A small
explicit blocklist is stripped so the list stays chat-safe for a
group game.  Games that need a deterministic AI pick pull from the
per-letter index rather than the model, so the house plays identically
with or without an API key.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

_DATA = Path(__file__).parent / "data" / "words.txt"

#: Explicit / adult words present in the raw frequency list that would
#: be tone-deaf in a friendly group chat.  Only words that actually
#: occur in ``words.txt`` matter here.
_BLOCKLIST: frozenset[str] = frozenset({
    "babe", "babes", "butts", "breast", "fetish", "pantyhose",
    "peeing", "spanking", "sucking", "titten", "swingers",
    "voyeurweb", "nudist", "naked", "sperm", "erotica", "sexually",
    "sexuality", "escort", "escorts", "thong", "thongs",
    "transexual", "transexuales", "transsexual", "travesti",
    "gay", "gays", "lesbian", "lesbians", "interracial",
    "poison", "slave", "suicide", "torture", "violence", "violent",
})

_WORD_RE = re.compile(r"^[a-z']+$")


def _load() -> frozenset[str]:
    text = _DATA.read_text(encoding="utf-8")
    out = set()
    for line in text.splitlines():
        w = line.strip().lower()
        if not w or not _WORD_RE.fullmatch(w):
            continue
        if not 1 <= len(w) <= 25:
            continue
        if w in _BLOCKLIST:
            continue
        out.add(w)
    return frozenset(out)


@lru_cache(maxsize=1)
def WORDSET() -> frozenset[str]:
    return _load()


@lru_cache(maxsize=1)
def WORDS_BY_FIRST() -> dict[str, tuple[str, ...]]:
    """Per-letter index for deterministic AI picks.

    Game words only: 3+ letters, so the house never plays 'aa' or 'ab'
    in a chain.
    """
    by_first: dict[str, list[str]] = {}
    for w in sorted(WORDSET()):
        if len(w) < 3:
            continue
        by_first.setdefault(w[0], []).append(w)
    return {k: tuple(v) for k, v in by_first.items()}


#: Everyday 5-letter words the web-frequency list under-represents.
#: Unioned into the wordle pool so the answer space feels like a real
#: word game, not a corpus artifact.
_EXTRA_5: tuple[str, ...] = (
    "about", "above", "actor", "acute", "admit", "adopt", "after",
    "again", "agent", "agree", "ahead", "alarm", "album", "alert",
    "alike", "alive", "allow", "alone", "along", "alter", "among",
    "anger", "angle", "angry", "anime", "apart", "apple", "apply",
    "arrow", "audio", "avoid", "awake", "bacon", "badge", "bagel",
    "baker", "basic", "batch", "beach", "beads", "beast", "begin",
    "being", "below", "bench", "berry", "bible", "birth", "black",
    "blame", "blank", "blast", "blaze", "blend", "bless", "blind",
    "block", "bloom", "blown", "board", "boast", "bloat", "blurb",
    "board", "bread", "break", "brews", "bring", "broad", "broke",
    "brown", "brush", "build", "built", "bunch", "burnt", "burst",
    "cabin", "cable", "candy", "cargo", "carry", "catch", "cause",
    "cedar", "chain", "chair", "charm", "chart", "chase", "cheap",
    "check", "cheek", "chest", "chief", "child", "chill", "china",
    "choir", "chunk", "cider", "cigar", "cling", "clock", "clone",
    "close", "cloth", "cloud", "clump", "coach", "coast", "cobra",
    "cocoa", "colon", "color", "comic", "comet", "coral", "corps",
    "count", "court", "cover", "crack", "craft", "crane", "crash",
    "crazy", "cream", "creek", "crest", "crime", "crisp", "cross",
    "crowd", "crown", "crush", "curve", "civic", "civil", "claim",
    "climb", "couch", "cough", "court", "coyot", "crank", "crape",
    "cubit", "daisy", "dance", "dandy", "dealt", "death", "debit",
    "delay", "delta", "dense", "depth", "diary", "dodge", "doing",
    "donor", "doubt", "dough", "dozen", "draft", "drank", "dread",
    "dream", "dress", "drill", "drink", "drive", "drove", "dwell",
    "dwelt", "eager", "early", "earth", "eaten", "eight", "elbow",
    "elfin", "empty", "enjoy", "enter", "entry", "equal", "erase",
    "error", "essay", "event", "every", "exact", "exile", "exist",
    "extra", "fable", "facet", "faint", "fancy", "fault", "feast",
    "feign", "fence", "fever", "fiber", "field", "fifth", "fifty",
    "fight", "final", "first", "flash", "fleet", "flesh", "fling",
    "flint", "float", "flock", "flood", "floor", "flour", "fluid",
    "flush", "flute", "focal", "foggy", "folds", "folio", "force",
    "forgo", "forum", "found", "frame", "frank", "frank", "fraud",
    "fresh", "front", "frost", "froze", "fruit", "fuzzy", "gauge",
    "ghost", "giant", "given", "glide", "globe", "gloom", "glory",
    "gloss", "glove", "going", "grace", "grade", "grain", "grand",
    "grant", "grape", "graph", "grasp", "grass", "grate", "grave",
    "great", "green", "greet", "grief", "grill", "grime", "grind",
    "groan", "groom", "gross", "group", "grove", "grown", "guard",
    "guess", "guest", "guide", "guild", "guilt", "happy", "harsh",
    "haste", "hatch", "haunt", "haven", "hazel", "heady", "heart",
    "heavy", "hedge", "heist", "honey", "honor", "horse", "hotel",
    "house", "human", "humor", "hunch", "hurry", "ideal", "image",
    "imply", "inbox", "index", "inner", "input", "irish", "issue",
    "jelly", "jewel", "joint", "jolly", "judge", "juice", "jumpy",
    "keeps", "ketch", "kneel", "knife", "knock", "known", "label",
    "labor", "large", "laser", "laugh", "layer", "learn", "lease",
    "lemon", "level", "lever", "light", "limit", "liver", "local",
    "lodge", "logic", "loose", "lower", "lucky", "lunar", "mango",
    "marry", "match", "maybe", "meant", "meaty", "melee", "melon",
    "merge", "merit", "metal", "meter", "midst", "might", "minor",
    "minus", "mirth", "mixed", "model", "molar", "money", "month",
    "moral", "motor", "motto", "mourn", "mouse", "mouth", "movie",
    "music", "naive", "nerdy", "noble", "noise", "north", "noted",
    "novel", "nudge", "oasis", "ocean", "offer", "often", "olive",
    "onset", "opera", "orbit", "order", "organ", "other", "ought",
    "outer", "owned", "oxide", "ozone", "paint", "panel", "panic",
    "paper", "parka", "party", "peace", "pearl", "pedal", "penny",
    "perch", "peril", "petal", "phase", "phone", "photo", "piano",
    "piece", "pilot", "pinch", "pitch", "pixel", "pizza", "place",
    "plain", "plane", "plant", "plate", "plead", "plot", "point",
    "pound", "power", "press", "price", "pride", "prime", "print",
    "prior", "prism", "prize", "probe", "proof", "proud", "prove",
    "prowl", "pulse", "punch", "pupil", "purer", "pursue", "queen",
    "query", "quest", "quick", "quiet", "quite", "quota", "quote",
    "radar", "radio", "raise", "rally", "ranch", "range", "rapid",
    "ratio", "reach", "ready", "realm", "rebel", "refer", "reign",
    "relax", "relay", "repay", "reply", "rifle", "right", "rigid",
    "risky", "rival", "river", "roast", "robot", "rocky", "roger",
    "roman", "roofy", "rouge", "rough", "round", "route", "royal",
    "rubin", "rupee", "rural", "sadly", "saint", "salad", "sandy",
    "sauce", "scale", "scene", "scope", "score", "scout", "scrap",
    "screw", "seize", "sense", "serve", "seven", "shade", "shaky",
    "shape", "share", "sharp", "shark", "shave", "shelf", "shell",
    "shift", "shine", "shiny", "shirt", "shock", "shore", "short",
    "shout", "shown", "shred", "shrug", "sight", "since", "sings",
    "siren", "sixth", "skirt", "skull", "slant", "slate", "slept",
    "slice", "slide", "slope", "small", "smart", "smell", "smile",
    "smoke", "snake", "snore", "solar", "solid", "solve", "sorry",
    "sound", "south", "space", "spare", "spark", "speak", "speed",
    "spend", "spent", "spice", "spine", "spoke", "spoon", "sport",
    "spray", "spree", "squad", "stack", "staff", "stage", "stain",
    "stair", "stake", "stale", "stalk", "stamp", "stand", "stare",
    "stark", "start", "state", "stays", "steak", "steam", "steer",
    "steep", "steer", "steel", "steep", "stern", "stick", "stiff",
    "still", "sting", "stock", "stole", "stone", "stood", "stool",
    "stoop", "store", "storm", "story", "stove", "straw", "strip",
    "stuck", "study", "stuff", "style", "sugar", "suite", "sunny",
    "super", "surge", "swamp", "swamp", "swarm", "swear", "sweet",
    "swell", "sweep", "swung", "swept", "swing", "syrup", "table",
    "taken", "tally", "taper", "taste", "teach", "teeth", "tempo",
    "tenor", "tense", "thank", "theft", "their", "theme", "thick",
    "thing", "think", "third", "those", "three", "throw",
    "throb", "thumb", "tight", "timer", "tired", "title", "toast",
    "today", "token", "tonic", "tooth", "topic", "torch", "total",
    "touch", "tough", "tower", "toxic", "trace", "track", "trade",
    "train", "trait", "trait", "trash", "treat", "trend", "trial",
    "tribe", "trick", "trio", "trout", "truck", "truly", "trunk",
    "tweak", "twill", "twist", "type", "union", "unity", "urban",
    "usage", "usual", "utter", "valid", "value", "video", "viral",
    "virus", "visit", "vital", "vivid", "vocal", "voice", "wafer",
    "wagon", "waste", "watch", "water", "weary", "where", "which",
    "while", "white", "whole", "whose", "woman", "women", "wooden",
    "wooly", "world", "worry", "worse", "worst", "worth", "would",
    "wound", "wrath", "write", "wrong", "wrote", "yacht", "yield",
    "young", "zebra",
)


@lru_cache(maxsize=1)
def FIVE_LETTERS() -> tuple[str, ...]:
    """5-letter words — wordle answers and the human's legal guesses.

    The lexicon's 5-letter words unioned with a curated set of
    everyday 5-letter words the web-frequency list under-represents.
    """
    pool = {w for w in WORDSET() if len(w) == 5}
    for w in _EXTRA_5:
        if len(w) == 5 and re.fullmatch(r"[a-z]{5}", w):
            pool.add(w)
    return tuple(sorted(pool))


def is_real_word(word: str) -> bool:
    return word.lower().strip() in WORDSET()


def words_starting_with(letter: str) -> tuple[str, ...]:
    return WORDS_BY_FIRST().get(letter.lower(), ())


def size() -> int:
    return len(WORDSET())
