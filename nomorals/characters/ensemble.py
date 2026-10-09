"""Ensemble scenes: multiple characters in one living conversation.

Podcast episodes, room discussions, roundtables. Characters talk to
each other (not just the brain), react, disagree, build on each other's
points. The brain can host, participate, or just direct.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

from .casting import chemistry
from .character import Character, SuggestFn
from .dialogue import Dialogue, DialogueTurn
from .relationships import RelationshipGraph


@dataclass
class Scene:
    """A running multi-character scene."""
    title: str
    participants: list[str] = field(default_factory=list)  # names
    turns: list[DialogueTurn] = field(default_factory=list)
    topic: str = ""
    started_at: float = field(default_factory=time.time)

    def add(self, speaker: str, text: str) -> None:
        self.turns.append(DialogueTurn(speaker=speaker, text=text))

    def transcript(self, limit: int = 14) -> str:
        return "\n".join(f"{t.speaker}: {t.text}"
                         for t in self.turns[-limit:])


def _speak_in_scene(char: Character, scene: Scene,
                    others: list[Character],
                    suggest: SuggestFn | None,
                    graph: RelationshipGraph | None,
                    rng: random.Random) -> str:
    """One character speaks in a group scene — aware of who else is here
    and how they feel about them."""
    others_block = ""
    if others:
        bits = []
        for o in others:
            rel = ""
            if graph is not None:
                e = graph.edge(char.id, o.id)
                rel = f" ({e.kind})"
            bits.append(f"{o.name}{rel}")
        others_block = f"Also here: {', '.join(bits)}.\n"
    ctx = (f"{scene.title} — topic: {scene.topic}\n"
           f"{others_block}"
           f"Conversation so far:\n{scene.transcript()}\n"
           f"Respond naturally as yourself. React to what's been said — "
           f"agree, disagree, build, joke. Don't repeat others.")
    mem = char.recall(scene.transcript(4))
    line = char.speak(ctx, suggest,
                      memories="\n".join(mem) if mem else "")
    char.remember(f"At '{scene.title}': I said: {line[:180]}", 0.55)
    return line


def run_scene(chars: list[Character], title: str, topic: str,
              suggest: SuggestFn | None,
              rounds: int = 6,
              host_line: str = "",
              graph: RelationshipGraph | None = None,
              seed: int | None = None) -> Scene:
    """Run a multi-character scene. Characters take turns speaking;
    order shuffles each round so nobody dominates. Returns the scene."""
    rng = random.Random(seed)
    sc = Scene(title=title, topic=topic,
               participants=[c.name for c in chars])
    if host_line:
        sc.add("Devon", host_line)
    order = list(chars)
    for _ in range(rounds):
        rng.shuffle(order)
        for char in order:
            others = [c for c in chars if c.id != char.id]
            try:
                line = _speak_in_scene(char, sc, others, suggest,
                                       graph, rng)
            except Exception:
                continue
            sc.add(char.name, line)
            # relationships breathe: speaking in a scene is a light
            # positive interaction with everyone present
            if graph is not None:
                for o in others:
                    chem = chemistry(char, o, graph)
                    if chem > 0.55:
                        graph.interact(char.id, o.id, "good_conversation",
                                       mirror="good_conversation")
    return sc


def podcast_episode(host: Character, guests: list[Character], topic: str,
                    suggest: SuggestFn | None,
                    graph: RelationshipGraph | None = None,
                    rounds: int = 8,
                    seed: int | None = None) -> Scene:
    """A podcast episode: host opens, guests discuss, host steers.

    Returns the Scene (transcript). Audio rendering is the podcast
    module's job — this is the living conversation underneath.
    """
    all_cast = [host] + guests
    sc = Scene(title=f"Podcast: {topic}", topic=topic,
               participants=[c.name for c in all_cast])
    opener = host.speak(
        f"You are opening your podcast episode about: {topic}. "
        f"Your guests: {', '.join(g.name for g in guests)}. "
        f"Welcome the listeners and your guests, set up the topic. "
        f"Be yourself — energetic, curious.",
        suggest)
    sc.add(host.name, opener)
    host.remember(f"Hosted podcast on '{topic}' with "
                  f"{', '.join(g.name for g in guests)}", 0.7)

    rng = random.Random(seed)
    # discussion: guests mostly, host interjects every few turns
    order = list(guests)
    for i in range(rounds):
        rng.shuffle(order)
        for g in order:
            others = [c for c in all_cast if c.id != g.id]
            line = _speak_in_scene(g, sc, others, suggest, graph, rng)
            sc.add(g.name, line)
        if i % 3 == 2:  # host steers
            steer = host.speak(
                f"You're hosting. Topic: {topic}. Conversation so far:\n"
                f"{sc.transcript()}\nSteer it — ask a sharp question, "
                f"challenge someone gently, or move to the next angle.",
                suggest)
            sc.add(host.name, steer)
    closer = host.speak(
        f"Close out the episode on '{topic}'. Thank your guests "
        f"({', '.join(g.name for g in guests)}) by name, give the "
        f"listeners one thing to think about. Sign off as yourself.",
        suggest)
    sc.add(host.name, closer)
    if graph is not None:
        for g in guests:
            graph.interact(host.id, g.id, "good_conversation",
                           mirror="good_conversation")
    return sc
