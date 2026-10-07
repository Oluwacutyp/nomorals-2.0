"""Case-bank enrichment + services (split from cases.py, Wave H3)."""
from __future__ import annotations

from typing import Any

from .bank_meta import _BANK_META
from .constants import HINT_COST, HISTORY_VERSION, TIER_BASE_SCORE, TIERS, _SEEN_CAP
from .new_cases import _NEW_CASES
from .raw_cases import _RAW_CASES

# ══════════════════════════════════════════════════════════════════════
# Bank enrichment + god-tier case services.
#
# _RAW_CASES (the 30 originals) get their god-tier metadata from
# _BANK_META; _NEW_CASES already carry the full schema.  Every case in
# CASES therefore has: id, title, tier, briefing, story, suspects,
# motives, culprit, clues (ordered), red_herrings, herring_suspect,
# solution, statements, difficulty (back-compat), generated=False.
# ══════════════════════════════════════════════════════════════════════

def _sig_words(name: str) -> list[str]:
    """Significant words of a suspect name (for fair-play matching)."""
    words = []
    for w in (name or "").split():
        w = w.strip(".,'\"").lower()
        if w != "the" and len(w) >= 3:
            words.append(w)
    return words


def _full_name(name: str) -> str:
    """Suspect name without the leading 'the' — for implication checks."""
    n = (name or "").strip().lower()
    if n.startswith("the "):
        n = n[4:]
    return n


def _herring_suspect(case: dict[str, Any]) -> str:
    """Which innocent the red-herring clue (clues[3]) points at."""
    blob = (case.get("clues") or [""] * 4)[3].lower()
    for s in case.get("suspects") or ():
        if s == case.get("culprit"):
            continue
        if any(w in blob for w in _sig_words(s)):
            return s
    return ""


def _enrich() -> tuple[dict[str, Any], ...]:
    out: list[dict[str, Any]] = []
    raws = list(_RAW_CASES) + list(_NEW_CASES)
    for i, raw in enumerate(raws):
        case = dict(raw)
        if i < len(_BANK_META):
            for key in ("title", "tier", "briefing", "motives",
                        "red_herrings", "solution"):
                case[key] = _BANK_META[i][key]
        tier = case.get("tier") or "medium"
        case["tier"] = tier
        case["id"] = f"bank:{i}"
        # back-compat grade for older scoring paths (expert → hard)
        case["difficulty"] = tier if tier in ("easy", "medium", "hard") \
            else "hard"
        case["herring_suspect"] = _herring_suspect(case)
        case["generated"] = False
        case["suspects"] = list(case.get("suspects") or ())
        case["clues"] = list(case.get("clues") or ())
        case["statements"] = dict(case.get("statements") or {})
        case["motives"] = dict(case.get("motives") or {})
        case["red_herrings"] = list(case.get("red_herrings") or ())
        out.append(case)
    return tuple(out)


CASES = _enrich()

# ── how many hand-written cases the bank holds (the generator adds more) ──
BANK_SIZE = len(CASES)


def _copy_case(case: dict[str, Any]) -> dict[str, Any]:
    """A mutable deep-enough copy: the game adapts clues per player."""
    c = dict(case)
    c["suspects"] = list(case.get("suspects") or ())
    c["clues"] = list(case.get("clues") or ())
    c["statements"] = dict(case.get("statements") or {})
    c["motives"] = dict(case.get("motives") or {})
    c["red_herrings"] = list(case.get("red_herrings") or ())
    return c


# ── fair-play solvability ─────────────────────────────────────────────────

_CONFESS_MARKERS = ("i did it", "i'm guilty", "i confess", "it was me",
                    "my fault", "i'll admit")


def verify_solvability(case: dict[str, Any]) -> list[str]:
    """Fair-play check: can the culprit be deduced from the clues?

    Returns a list of problems; empty means the case is solvable and
    fair: the smoking gun names the culprit, at least two innocents
    are cleared by alibis, the red herring points at an innocent (never
    the culprit), everyone has a motive and a statement, and the
    solution walks the deduction.
    """
    problems: list[str] = []
    suspects = list(case.get("suspects") or ())
    culprit = case.get("culprit")
    if len(suspects) != 4 or len(set(suspects)) != 4:
        problems.append("need exactly 4 unique suspects")
    if culprit not in suspects:
        problems.append("culprit must be one of the suspects")
    for key in ("title", "tier", "briefing", "story", "motives", "clues",
                "red_herrings", "solution", "statements"):
        if not case.get(key):
            problems.append(f"missing/empty {key}")
    if case.get("tier") not in TIERS:
        problems.append("tier must be one of easy/medium/hard/expert")
    motives = case.get("motives") or {}
    for s in suspects:
        if not (motives.get(s) or "").strip():
            problems.append(f"no motive for {s}")
    clues = list(case.get("clues") or ())
    if len(clues) < 5:
        problems.append("need at least 5 ordered clues")
    cwords = _sig_words(culprit or "")
    cname = _full_name(culprit or "")
    if clues and not any(w in clues[-1].lower() for w in cwords):
        problems.append("smoking gun (last clue) must name the culprit")
    # adapted cases splice extra clues in before the smoking gun, so
    # the classic five keep their order and the gun still closes the file
    innocents = [s for s in suspects if s != culprit]
    if len(clues) >= 5:
        for idx in (1, 2):
            blob = clues[idx].lower()
            cleared = [s for s in innocents
                       if any(w in blob for w in _sig_words(s))]
            if not cleared:
                problems.append(f"clue {idx + 1} (clears an innocent) "
                                "names no innocent")
        herring_blob = clues[3].lower()
        # full-name phrase: incidental shared words ("cage camera" vs
        # "cage cashier") must not count as implicating the culprit
        if cname and cname in herring_blob:
            problems.append("red-herring clue must not name the culprit")
        if not any(any(w in herring_blob for w in _sig_words(s))
                   for s in innocents):
            problems.append("red-herring clue should point at an innocent")
    for i, blob in enumerate([case.get("story") or ""] + clues):
        if "{" in blob or "}" in blob:
            problems.append(f"unfilled template slot in story/clue {i}")
        if not blob.strip():
            problems.append(f"empty story/clue {i}")
    statements = case.get("statements") or {}
    if set(statements) != set(suspects):
        problems.append("every suspect needs exactly one statement")
    cline = (statements.get(culprit) or "").lower()
    if any(m in cline for m in _CONFESS_MARKERS):
        problems.append("culprit must not confess in their statement")
    for name, line in statements.items():
        if "{" in line or "}" in line or not line.strip():
            problems.append(f"bad statement for {name}")
    for h in case.get("red_herrings") or ():
        hl = h.lower()
        if not any(any(w in hl for w in _sig_words(s)) for s in innocents):
            problems.append(f"red herring names no innocent: {h[:48]}")
        if cname and cname in hl:
            problems.append(f"red herring implicates the culprit: {h[:48]}")
    sol = case.get("solution") or ""
    if len(sol) < 40 or not any(w in sol.lower() for w in cwords):
        problems.append("solution must walk the deduction to the culprit")
    return problems


# ── per-player history: anti-repeat, skill, streaks ────────────────────────

def blank_history() -> dict[str, Any]:
    """A fresh per-player case record (JSON-safe, stored in the player
    profile's per_game['case'])."""
    return {
        "v": HISTORY_VERSION,
        "seen": {},            # tier -> [case ids already served]
        "attempts": {},        # tier -> cases played
        "solved": {},          # tier -> cases closed by this player
        "streak": 0,           # current consecutive solves
        "best_streak": 0,
        "reshuffles": 0,       # times a tier pool was exhausted
        "hints_taken": 0,
        "timed_solved": 0,
    }


def solve_rate(history: dict[str, Any], tier: str | None = None) -> float:
    """Rolling solve rate: solved / attempted. 0.5 (neutral) until the
    player has data, so new players get unmodified cases."""
    att = history.get("attempts") or {}
    sol = history.get("solved") or {}
    if tier is not None:
        a = int(att.get(tier) or 0)
        s = int(sol.get(tier) or 0)
    else:
        a = sum(int(v) for v in att.values())
        s = sum(int(v) for v in sol.values())
    return (s / a) if a else 0.5


def eligible_tiers(history: dict[str, Any]) -> tuple[str, ...]:
    """Tiers a player may draw from. New players start on easy/medium;
    hard unlocks at a 50%+ solve rate over 3+ cases, expert at 65%+
    over 6+."""
    attempts = sum(int(v) for v in (history.get("attempts") or {}).values())
    rate = solve_rate(history)
    tiers = ["easy", "medium"]
    if attempts >= 3 and rate >= 0.50:
        tiers.append("hard")
    if attempts >= 6 and rate >= 0.65:
        tiers.append("expert")
    return tuple(tiers)


def unseen_bank_cases(tier: str, seen_ids: Any) -> list[dict[str, Any]]:
    """Bank cases of a tier the player hasn't seen yet."""
    seen = set(seen_ids or ())
    return [c for c in CASES
            if c.get("tier") == tier and c.get("id") not in seen]


def deal_case(rng: Any, history: dict[str, Any]
              ) -> tuple[dict[str, Any], str, bool]:
    """Serve one case with per-player anti-repeat.

    Picks a tier from the player's eligible tiers, then an unseen bank
    case of that tier (with a ~25% sprinkle of fresh generated cases).
    When the tier's bank pool is exhausted, the pool reshuffles
    (``reshuffled=True``) and bank cases become eligible again.
    Applies skill-adaptive clue tuning before returning.
    """
    import random as _random

    if not isinstance(rng, _random.Random):
        rng = _random.Random(rng)
    tiers = eligible_tiers(history)
    tier = rng.choice(list(tiers))
    seen = history.setdefault("seen", {}).setdefault(tier, [])
    pool = unseen_bank_cases(tier, seen)
    reshuffled = False
    if not pool:
        pool = unseen_bank_cases(tier, [])
        reshuffled = True
    if pool and rng.random() < 0.75:
        case = _copy_case(rng.choice(pool))
    else:
        case = generate_case(rng, tier)
    return adapt_case(case, rng, history), tier, reshuffled


def record_played(history: dict[str, Any], case_id: str, tier: str,
                  solved: bool, reshuffled: bool = False) -> dict[str, Any]:
    """Fold one finished case into the player's history. Returns it."""
    attempts = history.setdefault("attempts", {})
    solved_map = history.setdefault("solved", {})
    attempts[tier] = int(attempts.get(tier) or 0) + 1
    seen = history.setdefault("seen", {}).setdefault(tier, [])
    if reshuffled:
        del seen[:]
        history["reshuffles"] = int(history.get("reshuffles") or 0) + 1
    if case_id and case_id not in seen:
        seen.append(case_id)
        if len(seen) > _SEEN_CAP:
            del seen[:len(seen) - _SEEN_CAP]
    if solved:
        solved_map[tier] = int(solved_map.get(tier) or 0) + 1
        history["streak"] = int(history.get("streak") or 0) + 1
        history["best_streak"] = max(int(history.get("best_streak") or 0),
                                     history["streak"])
    else:
        history["streak"] = 0
    history["v"] = HISTORY_VERSION
    return history


# ── skill-adaptive clue counts ─────────────────────────────────────────────

_ADAPT_HERRINGS = (
    "{s} was seen arguing with the victim's associate that week — over "
    "an old debt, settled since.",
    "A witness puts {s} near the scene that night — but the witness "
    "wore no watch and the hour is off.",
    "{s} called the victim that morning — an 11-second wrong number; "
    "the carrier log shows it.",
    "{s} asked about the case unprompted — curiosity, not guilt: every "
    "question was about public facts.",
)


def adapt_case(case: dict[str, Any], rng: Any,
               history: dict[str, Any]) -> dict[str, Any]:
    """Tune clue counts to the player's solve rate.

    Struggling players (sub-40% over 3+ cases) get one extra genuine
    clue — a background check on everyone's motives, inserted before
    the smoking gun. Strong players (75%+ over 5+ cases) get extra red
    herrings (two at 90%+) targeting innocents. Everyone else plays
    the case as written.
    """
    import random as _random

    if not isinstance(rng, _random.Random):
        rng = _random.Random(rng)
    case = _copy_case(case)
    rate = solve_rate(history)
    attempts = sum(int(v) for v in (history.get("attempts") or {}).values())
    innocents = [s for s in case["suspects"] if s != case["culprit"]]
    if attempts >= 3 and rate < 0.40:
        motives = case.get("motives") or {}
        bit = "; ".join(f"{s}: {motives.get(s) or 'no known motive'}"
                        for s in case["suspects"])
        assist = ("background check — everyone had a reason: " + bit +
                  ". means and opportunity still decide it.")
        case["clues"] = case["clues"][:-1] + [assist] + case["clues"][-1:]
        case["assisted"] = True
    elif attempts >= 5 and rate > 0.75:
        n = 2 if rate > 0.90 else 1
        pool = [s for s in innocents
                if s != case.get("herring_suspect")] or innocents
        for _ in range(n):
            text = rng.choice(_ADAPT_HERRINGS).format(s=rng.choice(pool))
            case["clues"] = case["clues"][:-1] + [text] + case["clues"][-1:]
            case["red_herrings"] = list(case.get("red_herrings") or []) + [text]
        case["sharpened"] = True
    return case


# ── scoring ───────────────────────────────────────────────────────────────

def score_solve(tier: str, strikes: int = 0, hints_used: int = 0,
                streak_in: int = 0, time_bonus: int = 0) -> int:
    """Points for closing a case.

    Base by tier, minus a point per wrong accusation and HINT_COST per
    hint, multiplied by the incoming solve streak (10% per streak step,
    capped at 2x), plus any timed-mode bonus. Always at least 1.
    """
    base = TIER_BASE_SCORE.get(tier, 5)
    mult = 1.0 + 0.1 * max(0, min(int(streak_in), 10))
    raw = (base - int(strikes) - HINT_COST * int(hints_used)) * mult
    return max(1, int(round(raw))) + max(0, int(time_bonus))


# ══════════════════════════════════════════════════════════════════════
# Dynamic case generator — seeded, combinatorial, airtight by construction.
#
# Every generated case follows the same 5-clue contract as the bank:
#   clue 1 = the mechanism (how it was done)
#   clue 2 = an alibi clearing innocent A
#   clue 3 = an alibi clearing innocent B
#   clue 4 = a red herring (suspicious-looking, provably innocent)
#   clue 5 = the smoking gun, naming the culprit by name
# The culprit's statement denies the smoking gun; every innocent's
# statement is consistent with their own clue. Because the pieces are
# assembled from matched (clue, statement) pairs, a generated case
# cannot contradict itself. Generated cases also carry the full
# god-tier schema (title, tier, briefing, motives, red_herrings,
# herring_suspect, solution).
#
# Difficulty: easy → blunt herring + blunt gun; hard/expert → subtle
# herring + subtle gun; medium → mixed. ~12 scenes × C(24,4) suspect
# sets × templates = millions of distinct, solvable cases.
# ══════════════════════════════════════════════════════════════════════

_GEN_ROLES = (
    "the night guard", "the curator", "the janitor", "the sous-chef",
    "the florist", "the valet", "the electrician", "the intern",
    "the archivist", "the stagehand", "the sommelier", "the locksmith",
    "the chauffeur", "the housekeeper", "the gardener", "the bartender",
    "the dispatcher", "the lab tech", "the docent", "the projectionist",
    "the bookkeeper", "the courier", "the radio operator",
    "the pastry chef",
    # wave 2 — double the suspect pool
    "the beekeeper", "the tattoo artist", "the lighthouse keeper",
    "the taxidermist", "the clockmaker", "the falconer",
    "the perfumer", "the blacksmith", "the cartographer",
    "the puppeteer", "the glazier", "the horologist",
    "the vintner", "the stonemason", "the milliner",
    "the apiarist", "the luthier", "the farrier",
    "the chandler", "the cutler", "the glover",
    "the haberdasher", "the ironmonger", "the lapidary",
    "the mason", "the needlemaker",
)

_GEN_SCENES = (
    {"item": "first-edition map", "place": "the sealed vault",
     "lock": "keycard reader"},
    {"item": "championship trophy", "place": "the trophy room",
     "lock": "code lock"},
    {"item": "sapphire necklace", "place": "the gallery case",
     "lock": "case key"},
    {"item": "leather-bound ledger", "place": "the corner office",
     "lock": "master key"},
    {"item": "concert violin", "place": "the green room",
     "lock": "spare key"},
    {"item": "reserve wine barrel", "place": "the cellar",
     "lock": "cellar key"},
    {"item": "illuminated manuscript", "place": "the reading room",
     "lock": "reading-room key"},
    {"item": "studio camera", "place": "the equipment cage",
     "lock": "studio fob"},
    {"item": "pocket watch", "place": "the wall safe",
     "lock": "combination dial"},
    {"item": "oil painting", "place": "the storage wing",
     "lock": "badge reader"},
    {"item": "field laptop", "place": "the server room",
     "lock": "biometric pad"},
    {"item": "signet ring", "place": "the bridal suite",
     "lock": "door code"},
    # wave 2 — double the scene pool
    {"item": "meteorite fragment", "place": "the observatory dome",
     "lock": "dome crank"},
    {"item": "ivory chess set", "place": "the smoking lounge",
     "lock": "lounge key"},
    {"item": "vintage synthesizer", "place": "the recording booth",
     "lock": "booth code"},
    {"item": "samurai sword", "place": "the dojo armory",
     "lock": "armory seal"},
    {"item": "dinosaur fossil", "place": "the dig tent",
     "lock": "tent zipper"},
    {"item": "golden compass", "place": "the captain's cabin",
     "lock": "cabin latch"},
    {"item": "crystal skull", "place": "the reliquary",
     "lock": "reliquary ward"},
    {"item": "antique telescope", "place": "the crow's nest",
     "lock": "nest hatch"},
    {"item": "emerald scarab", "place": "the sarcophagus chamber",
     "lock": "stone seal"},
    {"item": "steam engine model", "place": "the workshop loft",
     "lock": "loft padlock"},
    {"item": "pearl-handled revolver", "place": "the gun cabinet",
     "lock": "cabinet key"},
    {"item": "tapestry of kings", "place": "the great hall",
     "lock": "hall bar"},
    {"item": "jade burial mask", "place": "the tomb antechamber",
     "lock": "tomb door"},
    {"item": "brass astrolabe", "place": "the map room",
     "lock": "map-room key"},
    {"item": "velvet crown", "place": "the throne vault",
     "lock": "vault wheel"},
    {"item": "obsidian dagger", "place": "the ritual circle",
     "lock": "circle binding"},
    {"item": "silver tea service", "place": "the conservatory",
     "lock": "glass door"},
    {"item": "ruby-encrusted bible", "place": "the chapel sacristy",
     "lock": "sacristy key"},
)

_GEN_PLACE2 = (
    "the lobby", "the east wing", "the loading dock", "the courtyard",
    "the far stairwell", "the service corridor",
)

_GEN_TIMES = (
    ("19:40", "20:15"), ("20:52", "21:30"), ("21:41", "22:20"),
    ("22:15", "23:00"), ("01:47", "02:30"), ("02:14", "03:00"),
    ("05:20", "06:00"),
)

_GEN_TOOLS = ("wrench", "torch", "screwdriver", "multitool")

# (clue template, statement template) — the statement always agrees
# with the clue, so innocents can never contradict themselves.
_GEN_ALIBIS = (
    ("{s} was on the {place2} camera from {t1} to {t2}; the loop is unbroken.",
     "I was on the camera the whole time. Check the loop."),
    ("{s}'s badge logged the far wing at {t1} — the reader log is timestamped.",
     "My badge never left the far wing. The reader log proves it."),
    ("{s} was on a live call from {t1} to {t2}; the call record is unbroken.",
     "I was on a live call the whole time. The record is unbroken."),
    ("Two witnesses place {s} at {place2} at {t1}; both statements agree.",
     "Two people saw me at {place2}. Ask them."),
    ("{s}'s van GPS pinged {place2} until {t1}; the tracker log is intact.",
     "My van never moved. The tracker log is intact."),
    ("{s} clocked the {place2} register at {t1} — the tape agrees to the minute.",
     "I clocked the register at {t1}. The tape agrees."),
)

# (clue, statement, subtle) — subtle herrings look worse before the
# exonerating tail lands.
_GEN_HERRINGS = (
    ("{s} was seen near {place} at {t1} — but the {place2} camera catches "
     "{s} buying coffee at that exact minute.",
     "I was buying coffee. The camera has me.", False),
    ("A {tool} belonging to {s} was found at {place} — reported missing "
     "from {s}'s kit three days earlier.",
     "My {tool} went missing days ago. I filed the report.", False),
    ("{s} asked about the {item}'s value that afternoon — as part of an "
     "insurance inventory, filed at {t1}.",
     "It was for the insurance inventory. The filing is timestamped.", False),
    ("{s}'s fingerprints are on the {lock} — from the morning opening "
     "shift, logged at {t1}.",
     "I open up every morning. The log has my shift.", True),
    ("{s} had {item} dust on their cuffs — but the lab confirms it is from "
     "{s}'s legitimate bench work at {t1}.",
     "That is from my bench work. The lab confirmed it.", True),
    ("{s} left {place} in a hurry at {t1} — to catch the last ferry; "
     "the manifest lists {s}.",
     "I was catching the ferry. The manifest lists me.", True),
)

# (evidence, incriminating detail, evidence noun)
_GEN_EVIDENCE = (
    ("a duplicate key, freshly cut",
     "the {item}'s serial scratched into the tang", "duplicate key"),
    ("a pawn ticket dated last night",
     "the {item}'s description written on the stub", "pawn ticket"),
    ("photos of the {item}",
     "timestamps from {t1}, before anyone reported it gone", "photos"),
    ("a shipping label",
     "made out to a private buyer, in {s}'s own hand", "shipping label"),
    ("a cloth with {item} fibers",
     "monogrammed with {s}'s initials", "cloth"),
    ("a burner phone",
     "one {t1} call to a known fence, logged", "burner phone"),
)

# motives: the culprit's is pointed; everyone else gets a plausible
# but milder reason.
_GEN_MOTIVES = (
    "owes money all over town",
    "was passed over for promotion twice",
    "holds a grudge after a public demotion",
    "needs cash to cover an old debt",
    "was promised a cut by a fence",
    "is desperate to keep their job",
    "resents the victim's success",
    "has expensive tastes and an empty wallet",
)

_GEN_CULPRIT_MOTIVES = (
    "was offered good money for the {item} by a private buyer",
    "needed the {item} to cover a debt coming due",
    "had been planning to take the {item} for months",
)

# (clue template, subtle)
_GEN_GUNS = (
    ("In {s}'s locker: {ev} — {det}.", False),
    ("On {s}'s workbench: {ev}, and {det}.", False),
    ("Tucked inside {s}'s bag: {ev}; {det}.", False),
    ("{s}'s phone location pinged {place} at {t1} — and {det}.", True),
    ("The {item}'s packing straw turned up in {s}'s locker: {ev} — {det}.", True),
    ("A receipt in {s}'s name surfaced at the pawn shop: {ev}, {det}.", True),
)

_GEN_DENIALS = (
    "\"I've never touched the {item}.\"",
    "\"I was nowhere near {place} that night.\"",
    "\"That {noun} isn't mine — someone planted it.\"",
    "\"Ask anyone — I left well before {t1}.\"",
    "\"I don't even know what the {item} looks like.\"",
    "\"Check the cameras — you'll see I'm innocent.\"",
)

_RECENT_CASE_IDS: Any = None  # lazy deque, avoids import cost at module load


def _recent() -> Any:
    global _RECENT_CASE_IDS
    if _RECENT_CASE_IDS is None:
        from collections import deque

        _RECENT_CASE_IDS = deque(maxlen=12)
    return _RECENT_CASE_IDS


def generate_case(rng: Any, difficulty: str = "medium") -> dict[str, Any]:
    """Compose a fresh, solvable case from the template pools.

    ``rng`` is any ``random.Random``-like; the same seed always builds
    the same case. ``difficulty`` is ``easy`` | ``medium`` | ``hard`` |
    ``expert`` (expert uses the subtle templates, like hard).  Emits
    the full god-tier schema: id, title, tier, briefing, motives,
    red_herrings, herring_suspect and solution alongside the classic
    story/suspects/culprit/clues/statements.
    """
    import random as _random

    if not isinstance(rng, _random.Random):
        rng = _random.Random(rng)
    difficulty = difficulty if difficulty in TIERS else "medium"
    tier = difficulty

    scene = rng.choice(_GEN_SCENES)
    item, place, lock = scene["item"], scene["place"], scene["lock"]
    suspects = list(rng.sample(_GEN_ROLES, 4))
    culprit = rng.choice(suspects)
    others = [s for s in suspects]
    rng.shuffle(others)
    others = [s for s in others if s != culprit]
    cleared_a, cleared_b, herring_suspect = others[0], others[1], others[2]

    t1, t2 = rng.choice(_GEN_TIMES)
    place2 = rng.choice(_GEN_PLACE2)
    tool = rng.choice(_GEN_TOOLS)

    def fill(tpl: str, s: str = "", **kw: str) -> str:
        return tpl.format(s=s, item=item, place=place, lock=lock,
                          place2=place2, t1=t1, t2=t2, tool=tool, **kw)

    story = (f"The {item} vanished from {place} overnight. "
             f"Four people had access.")
    title = f"The {item.title()} Case"
    briefing = (story + " Four people had access — and every one of "
                "them had a motive.")
    clues = [fill(f"The {lock} shows one clean entry at {t1} — "
                   "no force, no alarm.")]

    statements: dict[str, str] = {}
    for sus, (ctpl, stpl) in zip(
            (cleared_a, cleared_b), rng.sample(_GEN_ALIBIS, 2)):
        clues.append(fill(ctpl, sus))
        statements[sus] = fill(stpl, sus)

    # Red herring: difficulty picks the subtlety.
    herring_pool = [h for h in _GEN_HERRINGS
                    if (h[2] == (difficulty in ("hard", "expert")))
                    or difficulty == "medium"]
    hclue, hstmt, _ = rng.choice(herring_pool or _GEN_HERRINGS)
    hclue_f = fill(hclue, herring_suspect)
    clues.append(hclue_f)
    statements[herring_suspect] = fill(hstmt, herring_suspect)

    # Smoking gun: always names the culprit.
    gun_pool = [g for g in _GEN_GUNS
                if (g[1] == (difficulty in ("hard", "expert")))
                or difficulty == "medium"]
    gtpl, _ = rng.choice(gun_pool or _GEN_GUNS)
    ev, det, noun = rng.choice(_GEN_EVIDENCE)
    ev_f = fill(ev, culprit)
    det_f = fill(det, culprit)
    gun = fill(gtpl, culprit, ev=ev_f, det=det_f)
    clues.append(gun)

    denial = rng.choice(_GEN_DENIALS)
    statements[culprit] = fill(denial, culprit, noun=noun)

    motives = {s: rng.choice(_GEN_MOTIVES) for s in suspects}
    motives[culprit] = fill(rng.choice(_GEN_CULPRIT_MOTIVES), culprit)

    solution = (
        f"The {lock} showed one clean entry at {t1}: an insider did "
        f"it. {cleared_a} and {cleared_b} are cleared by the camera "
        f"and call logs. {herring_suspect} looked guilty at first "
        f"glance, but the exonerating detail in the evidence cleared "
        f"them. The smoking gun: {det_f} — in {culprit}'s possession. "
        f"Only {culprit} fits all five clues."
    )

    return {
        "id": f"gen-{rng.getrandbits(32):08x}",
        "title": title,
        "tier": tier,
        "difficulty": tier if tier != "expert" else "hard",
        "briefing": briefing,
        "story": story,
        "suspects": suspects,
        "motives": motives,
        "culprit": culprit,
        "clues": clues,
        "red_herrings": [hclue_f],
        "herring_suspect": herring_suspect,
        "solution": solution,
        "statements": statements,
        "generated": True,
    }


def random_case(rng: Any, difficulty: str | None = None) -> dict[str, Any]:
    """A case for a new game: ~55% from the hand-written bank, the rest
    freshly generated. Never repeats one of the last 12 served cases,
    and always carries a ``difficulty`` grade for scoring."""
    import random as _random

    if not isinstance(rng, _random.Random):
        rng = _random.Random(rng)
    difficulty = difficulty or rng.choice(("easy", "medium", "hard"))
    recent = _recent()

    if rng.random() < 0.55:
        candidates = [(f"bank:{i}", c) for i, c in enumerate(CASES)
                      if f"bank:{i}" not in recent]
        if not candidates:
            candidates = [(f"bank:{i}", c) for i, c in enumerate(CASES)]
        cid, picked = rng.choice(candidates)
        case = dict(picked)
        case["id"] = cid
        case["statements"] = dict(picked.get("statements") or {})
        case["generated"] = False
    else:
        case = generate_case(rng, difficulty)
        cid = case["id"]

    case = dict(case)
    case["difficulty"] = difficulty
    recent.append(cid)
    return case
