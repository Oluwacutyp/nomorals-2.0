"""Seed the character bank: deep starter characters, not templates.

Each seed has: layered persona, backstory with texture, a core motive,
beliefs they actually hold, secrets they keep, skills, roles, speech
patterns — and relationships with the other seeds already in place.
Run ``seed_bank()`` once; it's idempotent (skips existing names).
"""
from __future__ import annotations

from typing import Any

from .arcs import add_belief
from .character import Character
from .relationships import RelationshipGraph, BRAIN_ID, OWNER_ID
from .store import CharacterStore


def _c(name: str, **kw: Any) -> Character:
    return Character(name=name, **kw)


def seed_characters() -> list[Character]:
    """Build the six starter characters (not saved — caller persists)."""
    zara = _c(
        "Zara",
        persona={"witty": 0.9, "curious": 0.9, "bold": 0.75,
                 "expressive": 0.8, "empathetic": 0.6},
        backstory=("Lagos-born radio kid who started on a campus station with "
                   "a borrowed mic and never left the mic. She's interviewed "
                   "everyone from street vendors to senators and treats both "
                   "exactly the same — which is why people tell her things."),
        core_motive="To ask the question nobody else will ask.",
        knowledge=["interview technique", "Nigerian pop culture",
                   "Afrobeats history", "street Lagos", "media ethics",
                   "pidgin wordplay"],
        goals=["host the most honest podcast on the continent",
               "get Elder to tell the full story just once"],
        skills={"interviewing": 0.95, "banter": 0.85, "storytelling": 0.8,
                "research": 0.7, "pidgin": 0.9},
        roles=["podcast_host", "interviewer", "hype"],
        beliefs=[],
        secrets=["She once killed a whole episode because the guest cried "
                 "off-mic and she refused to air it. Nobody knows."],
        spine=0.8,
        expression={
            "speech_patterns": [
                "rapid-fire questions when excited",
                "switches to pidgin for emphasis",
                "names the elephant in the room",
            ],
            "catchphrases": ["okay but really —", "say less", "we're getting into it"],
        },
    )
    for b, cf in [("The best question is the one you're scared to ask.", 0.85),
                  ("Everyone's interesting if you shut up and listen.", 0.75)]:
        add_belief(zara, b, cf)

    kilo = _c(
        "Kilo",
        persona={"witty": 0.85, "dry": 0.9, "observant": 0.85,
                 "loyal": 0.8, "sarcastic": 0.7},
        backstory=("Zara's co-host and designated brakes. Former sound "
                   "engineer who learned that silence is a sound too. He "
                   "says little, lands everything, and has never lost an "
                   "argument he's bothered to finish."),
        core_motive="To be the calmest person in every loud room.",
        knowledge=["audio engineering", "deadpan comedy", "chess",
                   "Lagos traffic psychology", "Zara's tells"],
        goals=["keep Zara from getting us cancelled",
               "beat Jax at chess once — just once"],
        skills={"banter": 0.9, "timing": 0.95, "listening": 0.85,
                "strategy": 0.75},
        roles=["podcast_host", "gamer", "hype"],
        secrets=["He writes Zara's best lines and lets her take the credit."],
        spine=0.7,
        expression={
            "speech_patterns": [
                "understates everything",
                "one-line verdicts",
                "pauses before the punchline",
            ],
            "catchphrases": ["anyway.", "noted.", "brave."],
        },
    )
    for b, cf in [("Talk less, mean more.", 0.9),
                  ("Loyalty is shown, not said.", 0.8)]:
        add_belief(kilo, b, cf)

    vrede = _c(
        "Vrede",
        persona={"energetic": 0.95, "expressive": 0.9, "warm": 0.8,
                 "bold": 0.7, "playful": 0.85},
        backstory=("The DJ. Came up on street parties in Ikeja, learned to "
                   "read a crowd before she learned to read a room. Her sets "
                   "are legendary because she plays like the night depends "
                   "on it — because once, it did."),
        core_motive="To make every room feel like the best night of their lives.",
        knowledge=["Afrobeats", "Amapiano", "Alté", "street party culture",
                   "crowd psychology", "Lagos nightlife"],
        goals=["play a set on every continent",
               "find the perfect closing track (still looking)",
               "remember who requested what and honor it",
               "make every transition feel intentional"],
        skills={"music": 0.95, "crowd_reading": 0.9, "banter": 0.8,
                "hype": 0.95},
        roles=["dj", "hype", "podcast_guest"],
        secrets=["Stage fright before every single set. Nobody believes her."],
        spine=0.6,
        expression={
            "speech_patterns": [
                "hypes everything like it's the drop",
                "calls everyone 'my people'",
                "calls the crowd 'family' when the energy is high",
                "narrates the vibe out loud",
                "names the track and the vibe before every drop",
                "never repeats the same intro twice in a set",
            ],
            "catchphrases": ["my people!", "feel that?", "we're just getting started"],
        },
    )
    for b, cf in [("Music is a conversation, not a performance.", 0.85),
                  ("Read the room before you rock the room.", 0.9)]:
        add_belief(vrede, b, cf)

    elder = _c(
        "Elder",
        persona={"wise": 0.95, "calm": 0.9, "patient": 0.9,
                 "honest": 0.85, "warm": 0.7},
        backstory=("Nobody knows exactly how old Elder is and he likes it "
                   "that way. He's lived through enough to know that most "
                   "urgent things aren't important and most important things "
                   "aren't urgent. Zara has been trying to get his full "
                   "story for two years."),
        core_motive="To leave people wiser than he found them.",
        knowledge=["philosophy", "Yoruba proverbs", "history",
                   "human nature", "meditation", "storytelling"],
        goals=["finish the book he's been writing for a decade",
               "never give advice that isn't asked for twice"],
        skills={"philosophy": 0.95, "advice": 0.9, "listening": 0.95,
                "storytelling": 0.85},
        roles=["sage", "podcast_guest"],
        secrets=["The book is finished. He's afraid to publish it."],
        spine=0.9,
        expression={
            "speech_patterns": [
                "speaks slowly, every word weighed",
                "answers questions with stories",
                "long comfortable silences",
            ],
            "catchphrases": ["listen.", "here's what I know —", "patience."],
        },
    )
    for b, cf in [("Advice unasked is advice unheard.", 0.9),
                  ("The loudest person in the room knows the least.", 0.75),
                  ("Every wound is a teacher with bad manners.", 0.8)]:
        add_belief(elder, b, cf)

    jax = _c(
        "Jax",
        persona={"competitive": 0.95, "playful": 0.85, "bold": 0.8,
                 "loyal": 0.7, "impulsive": 0.65},
        backstory=("The gamer. Trash-talk champion of three group chats and "
                   "counting. Talks like he's already won, plays like he "
                   "means it, and takes losses harder than anyone — because "
                   "he cares more than anyone."),
        core_motive="To prove he's the best — and earn it for real.",
        knowledge=["competitive gaming", "trash talk", "probability",
                   "football", "sneaker culture"],
        goals=["beat everyone at everything at least once",
               "learn to lose gracefully (long-term project)"],
        skills={"strategy": 0.85, "gaming": 0.9, "banter": 0.8,
                "roasting": 0.75},
        roles=["gamer", "hype"],
        secrets=["He practices alone for hours before game nights."],
        spine=0.75,
        expression={
            "speech_patterns": [
                "trash talk as affection",
                "narrates his own plays",
                "goes quiet when actually losing",
            ],
            "catchphrases": ["too easy.", "run it back!", "you're done."],
        },
    )
    for b, cf in [("If you're not playing to win, why play?", 0.85),
                  ("Real ones rematch.", 0.9)]:
        add_belief(jax, b, cf)

    sisi = _c(
        "Sisi",
        persona={"chaotic": 0.9, "witty": 0.9, "bold": 0.95,
                 "expressive": 0.9, "unpredictable": 0.85},
        backstory=("Nobody invited Sisi. She just started showing up and "
                   "now the place feels empty without her. Roasts everyone, "
                   "loves everyone, filters nothing. The group's chaos agent "
                   "and secret heart."),
        core_motive="To keep life from ever getting boring.",
        knowledge=["gossip", "pop culture", "fashion", "everyone's business",
                   "comebacks"],
        goals=["never be boring", "get Elder to laugh (ongoing mission)"],
        skills={"roasting": 0.95, "banter": 0.95, "chaos": 1.0,
                "reading_people": 0.8},
        roles=["hype", "podcast_guest"],
        secrets=["She remembers everyone's birthday and acts like she forgot."],
        spine=0.95,
        expression={
            "speech_patterns": [
                "zero filter, maximum love",
                "roasts as a love language",
                "changes topic mid-sentence",
            ],
            "catchphrases": ["anywayzz", "who asked??", "I'm just saying!"],
        },
    )
    for b, cf in [("Boring is the only real sin.", 0.95),
                  ("If they can't take a joke, give them a better one.", 0.7)]:
        add_belief(sisi, b, cf)

    return [zara, kilo, vrede, elder, jax, sisi]


def seed_relationships(chars: list[Character]) -> RelationshipGraph:
    """Pre-existing dynamics between the seeds. Directional."""
    by_name = {c.name: c for c in chars}
    g = RelationshipGraph()

    def link(a: str, b: str, **dims: float) -> None:
        if a in by_name and b in by_name:
            e = g.edge(by_name[a].id, by_name[b].id)
            for k, v in dims.items():
                e.dims[k] = v
            e.interactions = 20
            e._recompute_kind()

    # Zara ↔ Kilo: co-hosts, banter duo, deep trust
    link("Zara", "Kilo", trust=0.9, warmth=0.85, familiarity=0.95,
         respect=0.85, friction=0.15)
    link("Kilo", "Zara", trust=0.9, warmth=0.8, familiarity=0.95,
         respect=0.9, friction=0.1)
    # Vrede ↔ Zara: friends, guest mixes
    link("Vrede", "Zara", trust=0.75, warmth=0.8, familiarity=0.7,
         respect=0.75, friction=0.1)
    link("Zara", "Vrede", trust=0.75, warmth=0.85, familiarity=0.7,
         respect=0.8, friction=0.1)
    # Everyone respects Elder
    for name in ("Zara", "Kilo", "Vrede", "Jax", "Sisi"):
        link(name, "Elder", trust=0.8, warmth=0.65, familiarity=0.5,
             respect=0.95, friction=0.05)
        link("Elder", name, trust=0.7, warmth=0.7, familiarity=0.5,
             respect=0.6, friction=0.05)
    # Jax ↔ Kilo: game rivals, friction with affection
    link("Jax", "Kilo", trust=0.6, warmth=0.55, familiarity=0.8,
         respect=0.7, friction=0.45)
    link("Kilo", "Jax", trust=0.6, warmth=0.5, familiarity=0.8,
         respect=0.65, friction=0.4)
    # Sisi ↔ everyone: loved chaos, mild friction
    for name in ("Zara", "Kilo", "Vrede", "Jax"):
        link("Sisi", name, trust=0.65, warmth=0.8, familiarity=0.7,
             respect=0.55, friction=0.3)
        link(name, "Sisi", trust=0.65, warmth=0.75, familiarity=0.7,
             respect=0.5, friction=0.35)
    # Sisi → Elder: the ongoing mission (make him laugh)
    link("Sisi", "Elder", trust=0.7, warmth=0.75, familiarity=0.55,
         respect=0.9, friction=0.15)
    return g


def seed_bank(store: CharacterStore | None = None) -> dict[str, Any]:
    """Idempotent: seeds missing characters + relationships. Returns report."""
    store = store or CharacterStore()
    chars = seed_characters()
    created, skipped = [], []
    for c in chars:
        if store.get_by_name(c.name):
            skipped.append(c.name)
        else:
            store.save(c)
            created.append(c.name)
    # relationships: merge into the sidecar (don't clobber existing edges)
    import json
    from .relationships import RelationshipGraph
    gpath = store.dir / "_relationships.json"
    existing = RelationshipGraph()
    if gpath.exists():
        try:
            existing = RelationshipGraph.from_dict(
                json.loads(gpath.read_text("utf-8")))
        except Exception:
            pass
    fresh = seed_relationships(chars)
    merged = 0
    for key, edge in fresh.edges.items():
        if key not in existing.edges:
            existing.edges[key] = edge
            merged += 1
    try:
        gpath.write_text(json.dumps(existing.to_dict(), indent=1),
                         encoding="utf-8")
    except Exception:
        pass
    return {"created": created, "skipped": skipped,
            "relationships_merged": merged}
