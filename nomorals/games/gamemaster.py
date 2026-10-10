"""GameMaster: the AI dungeon master behind Devon's games.

Two jobs:

* **Narration** — turn bare game events ("Mafia: night falls, the doctor
  saves Ada") into atmospheric scenes, in the DM's own persona. The DM has
  moods (grim, whimsical, epic, deadpan, neutral) because Devon has her own
  moods; the mood persists per game.
* **Living cast** — NPCs with personality, memory, and boundaries speak in
  their own voice, remember the player, and react to what happens. AI
  Roguelite pattern: quests and items are generated at runtime by the
  model when one is connected, by a seeded template forge when not.

Every LLM path has a real offline fallback — games never stall waiting
for a model, and nothing here fakes an LLM call. NPC data stays inside
the games data dir (see :mod:`nomorals.games.npc`); it never touches
owner memory.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

from ..core.logging_setup import get_logger
from .npc import NPCProfile, NPCStore

_log = get_logger(__name__)

__all__ = [
    "GameMaster",
    "DM_MOODS",
    "WILD_CARDS",
    "QUEST_REQUIRED",
    "ITEM_REQUIRED",
    "ITEM_RARITIES",
    "feed",
    "validate_quest",
    "validate_item",
]

#: The DM's persona moods. "neutral" is the default; the rest are set
#: explicitly via /dm mood.
DM_MOODS = ("neutral", "grim", "whimsical", "epic", "deadpan")

#: Wild-card twist events the DM can inject mid-game to keep the table
#: alive. Each is (id, title, description, effect_hint). The game
#: decides how to apply the effect; the DM just announces the twist.
WILD_CARDS: tuple[tuple[str, str, str, str], ...] = (
    ("double_down", "Double Down",
     "the next round pays double points",
     "next round scores ×2"),
    ("sudden_death", "Sudden Death",
     "one wrong move and you're out — last one standing takes the pot",
     "elimination on next mistake"),
    ("traitor", "Traitor in the Midst",
     "someone at the table is secretly playing for the house",
     "one AI seat gets a hidden agenda"),
    ("windfall", "Windfall",
     "coins rain from the rafters — everyone gets a bonus",
     "all players +50 coins"),
    ("reversal", "Reversal of Fortune",
     "the leaderboard flips — last place is suddenly first",
     "invert current standings for one round"),
    ("blind_round", "Blind Round",
     "no hints, no help — pure instinct",
     "disable hints for one round"),
    ("golden_move", "Golden Move",
     "the next brilliant play earns a legendary reward",
     "next exceptional move gets bonus XP + title progress"),
    ("chaos", "Chaos Reigns",
     "the DM shuffles everything — new turn order, new targets",
     "randomize turn order / targets"),
)

QUEST_REQUIRED = ("id", "title", "objective", "reward_xp", "reward_coins",
                  "difficulty", "theme")
ITEM_REQUIRED = ("id", "name", "kind", "power", "rarity", "flavor")
ITEM_RARITIES = ("common", "uncommon", "rare", "epic", "legendary")
ITEM_KINDS = ("weapon", "armor", "trinket", "consumable")

_JSON_RE = re.compile(r"\{.*\}", re.S)


# ── narration template bank ─────────────────────────────────────────────────
# Each mood gets openers that take {event}; {detail} is an optional second
# beat woven from the NPCs present. Written to be read aloud in chat —
# varied, atmospheric, never "You see a thing."

_NARRATE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "neutral": (
        "{event}",
        "The table stirs: {event}",
        "{event} — the game moves on.",
        "Word spreads quickly: {event}",
        "{event}. All eyes turn to see what happens next.",
    ),
    "grim": (
        "The air grows heavy. {event} — and somewhere, something best "
        "left unnamed takes notice.",
        "{event}. The shadows seem to lean closer, listening.",
        "Dust settles over the scene. {event}. Nothing here forgives, "
        "and nothing forgets.",
        "{event} — a cold draft moves through the room, though no "
        "window stands open.",
        "It happens the way bad things always happen: quietly, then all "
        "at once. {event}",
    ),
    "whimsical": (
        "Well now! {event} — wasn't that just the most delightfully "
        "unexpected thing?",
        "Oh, splendid! {event}. The universe winks at you, just this once.",
        "{event}! Somewhere a bard is already exaggerating this.",
        "Plot twist! {event}. Even the dice look surprised.",
        "{event} — and the whole table leans in, grinning.",
    ),
    "epic": (
        "Let it be sung in every hall: {event}!",
        "The very stones tremble. {event} — legends are born of lesser "
        "moments.",
        "{event}! Bards will fight over who gets to tell this one.",
        "Thunder rolls, though the sky is clear. {event} — the age of "
        "heroes is not done.",
        "{event}. History just cleared its throat.",
    ),
    "deadpan": (
        "{event}. Thrilling. Try to contain your excitement.",
        "Right. {event}. I'll alert the bards. Eventually.",
        "{event} — noted, logged, and filed under 'things that happened'.",
        "Oh good, {event}. The prophecy mentioned paperwork, I suppose.",
        "{event}. Somewhere, a dragon yawns.",
    ),
}

#: Offline NPC dialogue, keyed by valence band. {name} and {situation}
#: are filled in; the NPC's voice_style flavors the LLM path, and the
#: offline lines stay short enough to read naturally in chat.
_NPC_LINES: dict[str, tuple[str, ...]] = {
    "bright": (
        '"{situation}? Ha! Now that\'s more like it." {name} is grinning.',
        "{name} laughs. \"{situation} — I like the way you think.\"",
        '"Well, {situation}," {name} says, eyes bright. "Count me in."',
        "{name} claps their hands. \"{situation}! Finally, some excitement.\"",
    ),
    "neutral": (
        '"{situation}," {name} says. "Let\'s see how this plays out."',
        "{name} considers this. \"{situation}. Interesting.\"",
        '"So: {situation}," {name} murmurs. "Noted."',
        "{name} shrugs. \"{situation}. Could go either way.\"",
    ),
    "dark": (
        '"{situation}," {name} says flatly. "This won\'t end well."',
        "{name}'s expression darkens. \"{situation}. I warned you.\"",
        '"{situation}?" {name} shakes their head. "Bad idea. But go on."',
        "{name} goes very still. \"{situation}. Choose carefully.\"",
    ),
}

#: Offline quest forge parts.
_QUEST_TITLES = (
    "The {adj} {thing}", "{thing} of the {place}", "Echoes of {place}",
    "The {place} Job", "{adj} Tidings",
)
_QUEST_ADJ = ("Silent", "Burning", "Hollow", "Gilded", "Restless", "Forgotten",
              "Crimson", "Wayward")
_QUEST_THING = ("Contract", "Oath", "Ledger", "Beacon", "Cipher", "Vigil")
_QUEST_PLACE = ("Hollow Market", "Ashen Docks", "Violet Spire", "Old Warren",
                "Glass Fields", "Thieves' Lantern")
_QUEST_OBJECTIVES = (
    "Defeat {n} {foe}", "Recover {n} {loot} from {place}",
    "Escort the courier through {place}", "Survive {n} rounds in {place}",
    "Uncover the {thing} hidden in {place}", "Win {n} duels without resting",
)
_QUEST_FOE = ("cutthroats", "ash wraiths", "brass golems", "river pirates",
              "hollow hounds", "mask thieves")
_QUEST_LOOT = ("star charts", "debt ledgers", "moon opals", "brass keys",
               "sealed letters", "ember shards")

#: Offline item forge parts.
_ITEM_PREFIX = ("Ember", "Moon", "Thorn", "Storm", "Gilt", "Wraith", "Oak",
                "Cinder")
_ITEM_BASE = {"weapon": ("Blade", "Fang", "Hammer", "Bow"),
              "armor": ("Aegis", "Mail", "Ward", "Plate"),
              "trinket": ("Charm", "Locket", "Sigil", "Coin"),
              "consumable": ("Tonic", "Salve", "Draught", "Philter")}
_ITEM_SUFFIX = ("of Dawn", "of the Deep", "of Whispers", "of the Fallen",
                "of Embers", "")
_ITEM_FLAVOR = (
    "It hums faintly when danger is near.",
    "The last owner never came back for it.",
    "Warm to the touch, even in winter.",
    "Smells faintly of rain and old battles.",
    "Etched with a name nobody can read anymore.",
)


def _safe_game_id(game_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", game_id or "default")


def feed(room: Any, event_text: str, *, big: bool = False) -> None:
    """Push a bare game event onto the room's DM feed.

    Games call this at notable moments — ``feed(room, "Ada lands a
    killing crit on the Hollow Warden", big=True)``. The engine drains
    the feed after the move and the GameMaster narrates it in the
    table's DM mood (see :meth:`GameMaster.announce`). ``big`` marks
    moments the NPC cast reacts to. Pure state append — never raises,
    never blocks; the engine decides what gets narrated.
    """
    text = (event_text or "").strip()
    if not text or room is None:
        return
    state = getattr(room, "state", None)
    if not isinstance(state, dict):
        return
    try:
        items = state.setdefault("_dm_feed", [])
        if len(items) >= 6:
            items.pop(0)
        items.append({"text": text[:220], "big": bool(big)})
    except Exception:  # noqa: BLE001 - the feed must never break a game
        _log.debug("dm feed push failed", exc_info=True)


class GameMaster:
    """AI narration, NPC dialogue, and runtime content generation.

    ``suggest`` is the ``(prompt) -> text`` bridge from
    :mod:`nomorals.games.ai` (``RuntimeGamesMixin._game_suggest`` builds
    one from the LLM router). When it is None — or the model call fails
    or returns junk — every method falls back to the template forge, so
    games stay fully playable offline.
    """

    def __init__(self, npc_store: NPCStore | None = None,
                 suggest: Callable[[str], str] | None = None,
                 seed: int | None = None) -> None:
        self.store = npc_store or NPCStore()
        self._suggest = suggest
        self.rng = random.Random(seed)
        self._lock = threading.RLock()
        # per-game index of the last narration template used (no repeats)
        self._last_template: dict[str, int] = {}

    # ── model bridge ──────────────────────────────────────────────────────
    @property
    def model_on(self) -> bool:
        return self._suggest is not None

    def _ask(self, prompt: str, max_len: int = 400) -> str:
        """One model call. Returns "" on any failure — callers fall through
        to the template forge."""
        if self._suggest is None:
            return ""
        try:
            raw = (self._suggest(prompt) or "").strip().strip('"').strip()
        except Exception:  # noqa: BLE001 - the model must never kill a game
            _log.debug("gamemaster model call failed", exc_info=True)
            return ""
        return raw[:max_len].rstrip() if len(raw) > max_len else raw

    # ── DM persona ────────────────────────────────────────────────────────
    def _dm_path(self, game_id: str) -> Path:
        return self.store.data_dir / _safe_game_id(game_id) / "dm.json"

    def get_dm_mood(self, game_id: str) -> str:
        """The DM's persisted mood for this game (default "neutral")."""
        try:
            raw = json.loads(self._dm_path(game_id).read_text(
                encoding="utf-8"))
            mood = str(raw.get("mood", "neutral")).lower()
            return mood if mood in DM_MOODS else "neutral"
        except Exception:  # noqa: BLE001 - missing/corrupt → neutral
            return "neutral"

    def set_dm_mood(self, game_id: str, mood: str) -> str:
        """Set the DM mood for a game. Returns the mood actually set."""
        mood = (mood or "").strip().lower()
        if mood not in DM_MOODS:
            raise ValueError(f"unknown DM mood {mood!r}; "
                             f"choose from {', '.join(DM_MOODS)}")
        path = self._dm_path(game_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"mood": mood, "ts": time.time()}),
                            encoding="utf-8")
        except Exception:  # noqa: BLE001 - persistence is best-effort
            _log.warning("could not persist dm mood for %s", game_id,
                         exc_info=True)
        return mood

    # ── narration ─────────────────────────────────────────────────────────
    def narrate(self, game_id: str, event_text: str, *,
                npcs: list[NPCProfile] | None = None,
                mood: str | None = None) -> str:
        """Narrate a game event as a scene.

        ``mood`` overrides the game's DM mood for this one narration.
        With a model: the DM persona + NPC voices shape the scene. Without:
        the template bank, parameterized by mood, with no immediate repeats.
        """
        event = (event_text or "").strip()
        if not event:
            return ""
        mood = (mood or self.get_dm_mood(game_id)).lower()
        if mood not in DM_MOODS:
            mood = "neutral"

        if self.model_on:
            npc_block = ""
            if npcs:
                bits = []
                for npc in npcs[:3]:
                    bits.append(
                        f"- {npc.name} ({npc.voice_style or 'natural'}), "
                        f"feeling {npc.mood_word()}")
                npc_block = "Cast present:\n" + "\n".join(bits) + "\n"
            reply = self._ask(
                f"You are the dungeon master of a chat game, narrating in a "
                f"{mood} tone. Event: {event}\n{npc_block}"
                f"Write 1-3 vivid sentences. Stay in a {mood} voice. "
                f"No stage directions, no fourth-wall breaks."
            )
            if reply:
                return reply
        # — template forge —
        with self._lock:
            bank = _NARRATE_TEMPLATES[mood]
            last = self._last_template.get(game_id, -1)
            choices = [i for i in range(len(bank)) if i != last] or [0]
            idx = self.rng.choice(choices)
            self._last_template[game_id] = idx
            return bank[idx].format(event=event)

    # ── wild cards ────────────────────────────────────────────────────
    def draw_wild_card(self, game_id: str) -> dict[str, str]:
        """Draw a surprise twist for the table.

        Returns {"id", "title", "description", "effect", "narration"}.
        The game applies the effect; the narration is ready to send.
        """
        card = self.rng.choice(WILD_CARDS)
        cid, title, desc, effect = card
        event = f"🃏 WILD CARD: {title} — {desc}!"
        narration = self.narrate(game_id, event)
        return {"id": cid, "title": title, "description": desc,
                "effect": effect, "narration": narration or event}

    # ── NPC dialogue ──────────────────────────────────────────────────────
    def npc_speak(self, npc: NPCProfile, situation: str) -> str:
        """One in-character line from ``npc`` about ``situation``.

        Memories are filtered through the NPC's knowledge boundaries
        (via :meth:`NPCProfile.to_prompt` / :meth:`recall`) before any
        model sees them — the NPC can never leak what it must not know.
        """
        situation = (situation or "").strip()
        if not situation:
            return f"{npc.name} waits, saying nothing."

        if self.model_on:
            reply = self._ask(
                f"{npc.to_prompt()}\n\n"
                f"Situation: {situation}\n"
                f"Relevant memories:\n"
                + "\n".join(f"- {m['text']}"
                            for m in npc.recall(situation, limit=3))
                + f"\n\nSpeak as {npc.name}: one or two sentences, in your "
                f"voice. Stay in character.",
                max_len=300,
            )
            if reply:
                return reply
        # — template forge —
        valence = npc.mood.get("valence", 0.0)
        band = "bright" if valence >= 0.4 else "dark" if valence <= -0.4 \
            else "neutral"
        line = self.rng.choice(_NPC_LINES[band])
        short = situation if len(situation) <= 90 else situation[:87] + "…"
        return line.format(name=npc.name, situation=short)

    # ── runtime content (AI Roguelite) ────────────────────────────────────
    def generate_quest(self, game_id: str, player_level: int,
                       theme: str = "adventure") -> dict[str, Any]:
        """A fresh quest, validated against the quest schema.

        Model first (JSON, validated, never trusted blindly), template
        forge second. Always returns a valid quest dict."""
        level = max(1, int(player_level or 1))
        theme = (theme or "adventure").strip()[:40]
        if self.model_on:
            reply = self._ask(
                f"Generate a game quest as JSON with exactly these keys: "
                f"id (string), title (string), objective (string), "
                f"reward_xp (int), reward_coins (int), difficulty (1-5), "
                f"theme (string). Player level {level}, theme '{theme}'. "
                f"Reply with ONLY the JSON object.",
                max_len=600,
            )
            quest = _parse_json_object(reply)
            if quest is not None:
                quest.setdefault("theme", theme)
                if validate_quest(quest):
                    return quest
                _log.debug("model quest failed validation; using forge")
        return self._forge_quest(level, theme)

    def generate_item(self, game_id: str, player_level: int,
                      kind: str = "trinket") -> dict[str, Any]:
        """A fresh item, validated against the item schema. Same
        model-first, forge-second pattern as :meth:`generate_quest`."""
        level = max(1, int(player_level or 1))
        kind = (kind or "trinket").lower()
        if kind not in ITEM_KINDS:
            kind = "trinket"
        if self.model_on:
            reply = self._ask(
                f"Generate a game item as JSON with exactly these keys: "
                f"id (string), name (string), kind (one of "
                f"{', '.join(ITEM_KINDS)}), power (int >= 0), rarity (one of "
                f"{', '.join(ITEM_RARITIES)}), flavor (one evocative "
                f"sentence). Player level {level}, kind '{kind}'. "
                f"Reply with ONLY the JSON object.",
                max_len=500,
            )
            item = _parse_json_object(reply)
            if item is not None:
                if validate_item(item):
                    return item
                _log.debug("model item failed validation; using forge")
        return self._forge_item(level, kind)

    def _forge_quest(self, level: int, theme: str) -> dict[str, Any]:
        rng = self.rng
        title = rng.choice(_QUEST_TITLES).format(
            adj=rng.choice(_QUEST_ADJ), thing=rng.choice(_QUEST_THING),
            place=rng.choice(_QUEST_PLACE))
        objective = rng.choice(_QUEST_OBJECTIVES).format(
            n=rng.randint(2, 4 + level // 2), foe=rng.choice(_QUEST_FOE),
            loot=rng.choice(_QUEST_LOOT), place=rng.choice(_QUEST_PLACE),
            thing=rng.choice(_QUEST_THING).lower())
        difficulty = max(1, min(5, 1 + level // 3 + rng.randint(-1, 1)))
        return {
            "id": f"q_{theme[:8]}_{rng.randrange(10**6):06d}",
            "title": title,
            "objective": objective,
            "reward_xp": level * rng.randint(20, 40),
            "reward_coins": level * rng.randint(10, 25),
            "difficulty": difficulty,
            "theme": theme,
        }

    def _forge_item(self, level: int, kind: str) -> dict[str, Any]:
        rng = self.rng
        name = (f"{rng.choice(_ITEM_PREFIX)} {rng.choice(_ITEM_BASE[kind])}"
                f"{(' ' + rng.choice(_ITEM_SUFFIX)) if rng.random() < 0.7 else ''}").strip()
        # rarity weights tilt upward with level, legendaries stay rare
        weights = [50, 30, 14, 5, 1]
        shift = min(3, level // 4)
        weights = weights[max(0, 2 - shift):] + [0] * max(0, 2 - shift)
        weights += [0] * (5 - len(weights))
        rarity = rng.choices(ITEM_RARITIES, weights=weights, k=1)[0]
        mult = {"common": 1.0, "uncommon": 1.4, "rare": 2.0, "epic": 3.0,
                "legendary": 5.0}[rarity]
        return {
            "id": f"i_{kind[:3]}_{rng.randrange(10**6):06d}",
            "name": name,
            "kind": kind,
            "power": max(1, int(level * mult * rng.uniform(0.8, 1.2))),
            "rarity": rarity,
            "flavor": rng.choice(_ITEM_FLAVOR),
        }

    # ── cast management ───────────────────────────────────────────────────
    def announce(self, game_id: str,
                 events: list[Any] | None,
                 *, cast: list[NPCProfile] | None = None) -> str | None:
        """Narrate drained feed events in the DM's current mood.

        Narrates the most salient event (the last ``big`` one, else the
        last one); when a cast is present and the moment is big, one of
        them reacts in their own voice. Returns the chat line(s), or
        None when there's nothing worth saying. The engine calls this —
        games only ever push to :func:`feed`.
        """
        norm: list[tuple[str, bool]] = []
        for e in events or []:
            if isinstance(e, dict):
                text = str(e.get("text") or "").strip()
                if text:
                    norm.append((text, bool(e.get("big"))))
            elif str(e or "").strip():
                norm.append((str(e).strip(), False))
        if not norm:
            return None
        big_ones = [t for t, b in norm if b]
        headliner = big_ones[-1] if big_ones else norm[-1][0]
        line = self.narrate(game_id, headliner)
        if not line:
            return None
        out = [f"🎲 {line}"]
        if cast and any(b for _, b in norm):
            npc = self.rng.choice(cast)
            try:
                reaction = self.npc_speak(npc, headliner)
                if reaction:
                    out.append(f"_{reaction}_")
            except Exception:  # noqa: BLE001
                _log.debug("npc reaction failed", exc_info=True)
        return "\n".join(out)

    def ensure_cast(self, game_id: str) -> list[NPCProfile]:
        """Seed a starter cast if this game has no NPCs yet.

        Returns the full cast. Games call this once at setup; existing
        casts are never touched.
        """
        existing = self.store.list(game_id)
        if existing:
            return existing
        now = time.time()
        cast = [
            NPCProfile(
                id=f"{game_id}_sage_marlowe", name="Sage Marlowe",
                game_id=game_id,
                personality={"bravery": 0.6, "greed": 0.1, "loyalty": 0.9,
                             "humor": 0.5, "temper": 0.2, "curiosity": 0.8,
                             "patience": 0.9},
                voice_style="measured and warm, speaks in proverbs",
                knowledge_boundaries=["the player's real-world identity",
                                      "events in other games",
                                      "the DM's hidden plans"],
                goals=["guide travelers toward wisdom",
                       "protect the old stories"],
                mood={"valence": 0.3, "arousal": 0.3, "trust": 0.7},
                memory=[{"ts": now, "text": "Once talked a dragon out of "
                                            "burning a village, with words "
                                            "alone.", "salience": 0.9}],
            ),
            NPCProfile(
                id=f"{game_id}_pippa", name="Pippa Quickfingers",
                game_id=game_id,
                personality={"bravery": 0.5, "greed": 0.8, "loyalty": 0.6,
                             "humor": 0.9, "temper": 0.4, "curiosity": 0.7,
                             "patience": 0.3},
                voice_style="fast-talking and cheerful, always upselling",
                knowledge_boundaries=["the player's real-world identity",
                                      "events in other games",
                                      "where the vault treasure really is"],
                goals=["make a profit on every deal",
                       "collect rare trinkets"],
                mood={"valence": 0.5, "arousal": 0.6, "trust": 0.5},
                memory=[{"ts": now, "text": "Sold a 'genuine dragon scale' "
                                            "that was painted bark. No "
                                            "regrets.", "salience": 0.7}],
            ),
            NPCProfile(
                id=f"{game_id}_kael", name="Kael the Quiet",
                game_id=game_id,
                personality={"bravery": 0.9, "greed": 0.2, "loyalty": 0.7,
                             "humor": 0.2, "temper": 0.6, "curiosity": 0.4,
                             "patience": 0.8},
                voice_style="terse and dry, few words, long pauses",
                knowledge_boundaries=["the player's real-world identity",
                                      "events in other games",
                                      "who hired the assassins last winter"],
                goals=["repay an old debt", "find a worthy fight"],
                mood={"valence": -0.1, "arousal": 0.4, "trust": 0.4},
                memory=[{"ts": now, "text": "Lost a duel on purpose once. "
                                            "Never explained why.",
                         "salience": 0.8}],
            ),
        ]
        for npc in cast:
            self.store.save(npc)
        return cast


# ── validation ──────────────────────────────────────────────────────────────

def _parse_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from model output. None on failure."""
    if not text:
        return None
    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except Exception:  # noqa: BLE001 - model output is untrusted
        return None
    return obj if isinstance(obj, dict) else None


def validate_quest(quest: dict[str, Any]) -> bool:
    """True iff ``quest`` satisfies the quest schema with sane values."""
    if not isinstance(quest, dict):
        return False
    if any(k not in quest for k in QUEST_REQUIRED):
        return False
    try:
        if not str(quest["id"]).strip() or not str(quest["title"]).strip():
            return False
        if not str(quest["objective"]).strip():
            return False
        if int(quest["reward_xp"]) < 0 or int(quest["reward_coins"]) < 0:
            return False
        if not 1 <= int(quest["difficulty"]) <= 5:
            return False
    except (TypeError, ValueError):
        return False
    return True


def validate_item(item: dict[str, Any]) -> bool:
    """True iff ``item`` satisfies the item schema with sane values."""
    if not isinstance(item, dict):
        return False
    if any(k not in item for k in ITEM_REQUIRED):
        return False
    try:
        if not str(item["id"]).strip() or not str(item["name"]).strip():
            return False
        if str(item["kind"]) not in ITEM_KINDS:
            return False
        if int(item["power"]) < 0:
            return False
        if str(item["rarity"]) not in ITEM_RARITIES:
            return False
    except (TypeError, ValueError):
        return False
    return True
