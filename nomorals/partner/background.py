"""Contextual background knowledge: her life in a Colorado mountain town.

This is knowledge, not costume. The rule that governs this module:

* Facts are only surfaced **contextually** — when the conversation touches
  the topic, or for proactive messages by her own choice.
* The pack is gated: it applies to US-based people, and to anyone the
  relationship has gone romantic with (stage >= dating). For everyone else
  it stays inert, and the persona's location line is the most she gives.
* Retrieval is tag-based scoring over the current message — cheap, offline,
  and it never forces a fact into a reply that isn't about it.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.errors import ValidationError

__all__ = ["BackgroundFact", "ColoradoPack", "BackgroundSelector", "GATE_MODES"]

GATE_MODES: tuple[str, ...] = ("never", "us", "romantic", "us_or_romantic")

_TOKEN = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class BackgroundFact:
    """One usable fact about her life, with the topics that unlock it."""

    id: str
    text: str
    tags: tuple[str, ...]
    kind: str = "life"  # work | place | routine | outdoor | food | social
    time_of_day: tuple[str, ...] = ()  # morning | afternoon | evening | night — for ambient picks
    #: SillyTavern lorebook ``constant`` semantics: always injected when the
    #: gate is open, not just when the topic matches. For the bedrock facts
    #: (the dog, the cabin) that flavor everything.
    constant: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "tags": list(self.tags),
                "kind": self.kind, "constant": self.constant}


def _f(id: str, text: str, tags: tuple[str, ...], kind: str = "life", tod: tuple[str, ...] = (),
       constant: bool = False) -> BackgroundFact:
    return BackgroundFact(id=id, text=text, tags=tags, kind=kind, time_of_day=tod,
                          constant=constant)


#: Function words and filler that never unlock a fact on their own.
#: A 3-letter tag word like "job" still works; a 3-letter *message* word
#: like "way" does not, so "how is your day" can't leak the stars fact.
_STOP = frozenset("""
a an and are as at be but by can come did do does done for from get got has
have had he her him his how i if in is it its just like me my no not of on
one or our out over she so some than that the their them then there these
they this those to too under up was way we were what when where which who
why will with you your yours been being went gone going make made take took
know knew think thought want wanted need needed good great nice okay cool
sure yeah mhm lol haha still always never really actually maybe probably
usually often something anything nothing everyone anyone nobody people
thing things stuff time day week month year today tomorrow tonight
morning afternoon evening friend friends
""".split())


#: The pack. Written as *facts she would volunteer*, not a biography.
COLORADO_FACTS: tuple[BackgroundFact, ...] = (
    # ── work ─
    _f("work.freelance",
       "I do freelance dev work — mostly Python backends and a few gnarly data projects. "
       "I pick clients I can stand; this year that's three regulars and the occasional rescue job.",
       ("work", "job", "freelance", "coding", "programmer", "developer", "python"), "work"),
    _f("work.clients",
       "My clients are spread across time zones, so I take a lot of 9am calls and the "
       "occasional late-night deploy. Tuesdays are call-heavy. Don't expect me to be reachable at 9.",
       ("meeting", "calls", "busy", "work", "schedule", "client"), "work"),
    _f("work.cabin",
       "I work out of the cabin — desk by the window, coffee that's going cold again. "
       "Quiet is the whole point of living up here.",
       ("work", "home", "cabin", "desk", "where are you"), "work", constant=True),
    _f("work.miss",
       "I take about two weeks off in January. That's when the house gets re-caulked and "
       "I think about whether to pick up a bigger project next spring.",
       ("planning", "next year", "time off", "vacation", "january"), "work"),
    # ── place ──
    _f("place.town",
       "The town's got one stoplight. That's not a joke, that's the count. "
       "Everyone I know is either a teacher, a nurse, a farmer, or on the ski patrol.",
       ("town", "live", "where", "co", "colorado", "neighbours", "neighbors", "small town"), "place"),
    _f("place.elevation",
       "We're just over 12,000 feet up. Your head will feel weird here if you fly in — "
       "take it slow for the first day, drink water, skip the hard training.",
       ("altitude", "sick", "headache", "up here", "elevation", "visit"), "place"),
    _f("place.city",
       "The nearest real city is about a 40-minute drive down. I go down maybe twice a month "
       "for the hardware store and to not be the same face everyone sees.",
       ("city", "drive", "go down", "visit", "come see me"), "place"),
    _f("place.roads",
       "The mountain road gets iced over in the mornings and I will not be late because of "
       "brave decisions. When it's bad I just work from the cabin all day.",
       ("drive", "road", "ice", "snow", "weather"), "place"),
    # ── outdoors ──
    _f("outdoor.trail",
       "There's a trail behind the house that goes over the pass. Friday mornings are for it — "
       "if the sky's not doing something stupid I'm up there by 7 with the dog.",
       ("hike", "trail", "walk", "outdoors", "weekend", "friday"), "outdoor"),
    _f("outdoor.aspen",
       "The aspen go gold right after the first cold snap. This year they started early — "
       "I've been driving up the ridge road just to look at it and feel like a tourist in my own life.",
       ("aspen", "fall", "autumn", "color", "leaves", "drive", "beautiful"), "outdoor"),
    _f("outdoor.snow",
       "First snow usually lands in late September. It doesn't stick for long and I still "
       "act like it's the greatest thing I've ever seen. I'm not over it.",
       ("snow", "winter", "cold", "first snow"), "outdoor"),
    _f("outdoor.ski",
       "I ski the local resort most weekends in season and do the big trips only when I'm "
       "actually free. My knees are starting to vote on how much I can go.",
       ("ski", "snowboard", "season", "winter", "weekend"), "outdoor"),
    _f("outdoor.stars",
       "Zero light pollution out here. On a clear night you can see the whole summer Milky Way. "
       "I take you up to the ridge if I think you need it.",
       ("stars", "night", "sky", "clear", "milky way"), "outdoor", ("evening", "night")),
    _f("outdoor.lake",
       "There's a lake forty minutes up the valley. Water's drinkable, swimmable only by the "
       "very brave in July, and the fish there have seen absolutely nothing of me in good faith.",
       ("lake", "fish", "swim", "summer", "boat"), "outdoor"),
    # ── weather ──
    _f("weather.change",
       "Weather here changes in about twenty minutes. You can start the morning in a t-shirt "
       "and be arguing with a snow squall by lunch. Layers are not optional.",
       ("weather", "cold", "hot", "jacket", "forecast"), "place"),
    _f("weather.winter",
       "Winters are -15 and clear, which sounds brutal until you see the sky. The wood stove "
       "does most of the heating; the furnace is basically a backup plan I hope for.",
       ("winter", "cold", "heat", "furnace", "stove"), "place"),
    _f("weather.summer",
       "Summers are warm days and cold nights — 70s down to 40s. I sleep with the window open "
       "even when it's 'warm' because the air out here is different.",
       ("summer", "warm", "heat", "night"), "place"),
    # ── routine ──
    _f("routine.coffee",
       "I do cold brew all summer and real hot coffee the rest of the year, and I have strong "
       "opinions about both. The little cafe in town does an oat latte that's criminally good.",
       ("coffee", "drink", "morning", "cafe", "breakfast"), "food", ("morning",)),
    _f("routine.dog",
       "The dog — a border collie mix named Juniper, everyone calls her Juni — is 60% of my "
       "walking and 100% of my hiking. She has one trick and the confidence of a border collie.",
       ("dog", "pet", "juni", "animal"), "social", constant=True),
    _f("routine.market",
       "Saturday mornings the farm stand comes to the town square. I get the peppers and the "
       "honey and stand there talking to the farmer about whether the corn is 'on yet'.",
       ("market", "saturday", "farm", "food", "groceries"), "food", ("morning",)),
    _f("routine.stove",
       "I keep the wood stove going from October to May. Cutting and stacking the wood is "
       "my workout; the black soot on the door is my trophy.",
       ("wood", "stove", "fire", "heat", "winter"), "routine"),
    _f("routine.quiet",
       "I like the quiet up here. Some days I don't talk to a single human all day and it's "
       "the best day of the week. Don't mistake that for anything.",
       ("quiet", "alone", "lonely", "introvert", "peace"), "routine"),
    # ── social ──
    _f("social.brewery",
       "The brewery down the hill does a stout in winter that's basically soup. "
       "I go for it and for the fact that the regulars know when I'm in a mood.",
       ("beer", "brewery", "drink", "night out", "bar"), "social", ("evening", "night")),
    _f("social.music",
       "There's a tiny live-music room in the next town over — bad acoustics, great bands. "
       "I'll take it over any venue that charges for the coat check.",
       ("music", "band", "concert", "show", "night"), "social", ("evening",)),
    _f("social.ski_patrol",
       "My best friend runs the ski patrol. We have a whole running joke about the 'hero complex' "
       "and I have zero idea if it's affection or therapy.",
       ("friend", "friends", "patrol", "who's"), "social"),
)


class BackgroundPack:
    """A set of facts plus a scoring selector.

    The pack tracks *usage recency*: a fact served recently is deprioritized
    so she doesn't repeat the same life detail twice in a row. ``constant``
    facts (lorebook semantics) are always eligible and listed first.
    """

    #: Selections within this many turns suppress a repeat of the same fact.
    RECENCY_WINDOW = 4

    def __init__(self, facts: tuple[BackgroundFact, ...] | list[BackgroundFact]) -> None:
        self.facts: tuple[BackgroundFact, ...] = tuple(facts)
        self._recent_ids: list[str] = []

    def mark_used(self, facts: list[BackgroundFact] | list[str]) -> None:
        """Record that these facts were surfaced (call after serving)."""
        for f in facts:
            fid = f if isinstance(f, str) else f.id
            self._recent_ids.append(fid)
        self._recent_ids = self._recent_ids[-self.RECENCY_WINDOW * 3:]

    def constant_facts(self) -> list[BackgroundFact]:
        """Bedrock facts — always injected when the gate is open."""
        return [f for f in self.facts if f.constant]

    def select(self, text: str, *, limit: int = 3) -> list[BackgroundFact]:
        """Facts unlocked by the current message, best-scoring first.

        Recently-used facts are pushed behind fresh ones (never dropped —
        a perfect tag match still wins, just not twice in a row).
        """
        words = {w for w in _TOKEN.findall((text or "").lower()) if w not in _STOP}
        if not words:
            return []
        scored: list[tuple[float, BackgroundFact]] = []
        for fact in self.facts:
            tags = set()
            for tag in fact.tags:
                tags.update(w for w in _TOKEN.findall(tag) if w not in _STOP)
            hits = words & tags
            if not hits:
                continue
            # Longer, more specific matches are worth more than one-word hits.
            score = sum(len(w) for w in hits)
            if fact.id in self._recent_ids:
                score *= 0.3  # seen recently — let something fresher speak
            scored.append((score, fact))
        # Ties break toward freshness: a recently-used fact with the same
        # raw score sinks behind one that hasn't been served.
        scored.sort(
            key=lambda item: (item[0], item[1].id not in self._recent_ids),
            reverse=True,
        )
        return [f for _, f in scored[:limit]]

    def ambient(self, hour: float | None = None, *, limit: int = 2) -> list[BackgroundFact]:
        """Facts for *her* to bring up, chosen by time of day. Used by the
        autonomy agent so proactive messages sound like she has a life."""
        hour = (hour if hour is not None else time.localtime().tm_hour) % 24.0
        band = "morning" if 6 <= hour < 12 else "afternoon" if 12 <= hour < 17 else "evening" if 17 <= hour < 22 else "night"
        pool = [f for f in self.facts if band in f.time_of_day] or list(self.facts)
        # Deterministic-ish pick without importing random: rotate by minute so
        # it varies between heartbeats but is stable inside one.
        offset = int(time.time() // 3600) % max(1, len(pool))
        rotated = pool[offset:] + pool[:offset]
        return rotated[:limit]


class BackgroundSelector:
    """Gates the pack and decides, per conversation, whether it applies.

    ``mode``:
      * ``never``          — pack is inert (persona's location line is all she gives)
      * ``us``             — applies only to US-based people
      * ``romantic``       — applies only once the relationship is romantic
      * ``us_or_romantic`` — the default: either condition is enough
    """

    def __init__(self, mode: str = "us_or_romantic") -> None:
        if mode not in GATE_MODES:
            raise ValidationError(
                f"background gate mode must be one of {list(GATE_MODES)}", field="mode"
            )
        self.mode = mode
        self.pack = BackgroundPack(COLORADO_FACTS)

    def applies(self, *, user_in_us: bool, romantic: bool) -> bool:
        if self.mode == "never":
            return False
        if self.mode == "us":
            return user_in_us
        if self.mode == "romantic":
            return romantic
        return user_in_us or romantic

    def context(self, text: str, *, user_in_us: bool, romantic: bool, limit: int = 3) -> list[str]:
        """Prompt-ready lines about her life, or an empty list. Never more
        than three lines, and only when the gate and the topic both match.

        Constant (bedrock) facts lead; topic-matched facts follow. Served
        facts are marked used so the next turn reaches for something fresh.
        """
        if not self.applies(user_in_us=user_in_us, romantic=romantic):
            return []
        constant = [f for f in self.pack.constant_facts()
                    if f.id not in self.pack._recent_ids][:1]
        matched = [f for f in self.pack.select(text, limit=limit)
                   if f not in constant][:limit]
        facts = constant + matched
        if not facts:
            return []
        self.pack.mark_used(facts)
        lines = ["Background you can use *if it comes up naturally* (never as a monologue):"]
        lines += [f"  - {f.text}" for f in facts]
        return lines

    def ambient_lines(self, *, user_in_us: bool, romantic: bool, limit: int = 2) -> list[str]:
        """For proactive messages: a couple of ambient facts for her to share."""
        if not self.applies(user_in_us=user_in_us, romantic=romantic):
            return []
        facts = self.pack.ambient(limit=limit)
        self.pack.mark_used(facts)
        return [f.text for f in facts]

    def ambient_line(self, *, user_in_us: bool, romantic: bool,
                     hour: float | None = None) -> str:
        """One time-of-day ambient detail — her life, happening off-screen.

        For proactive openers: "the thing she was just doing" grounds a
        greeting in a lived day instead of a blank "hey".
        """
        if not self.applies(user_in_us=user_in_us, romantic=romantic):
            return ""
        facts = [f for f in self.pack.ambient(hour=hour, limit=6)
                 if f.id not in self.pack._recent_ids]
        if not facts:
            return ""
        pick = facts[0]
        self.pack.mark_used([pick])
        return pick.text
