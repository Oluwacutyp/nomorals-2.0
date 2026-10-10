"""Ensemble scenes: multiple characters in one living conversation.

Podcast episodes, room discussions, roundtables. Characters talk to
each other (not just the brain), react, disagree, build on each other's
points. The brain can host, participate, or just direct.

Mining gold (see CHARACTERS_SWEEP_MINING.md):
- Façade (Mateas & Stern): the DRAMA MANAGER — an invisible agent that
  watches the scene's energy and proactively steers beats. Scenes here
  used to be pure round-robin; now a DramaDirector watches energy,
  fires beats with preconditions, and intervenes before scenes flatline.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

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
                    rng: random.Random,
                    steer: str = "") -> str:
    """One character speaks in a group scene — aware of who else is here
    and how they feel about them. ``steer`` is a drama-director nudge
    (Façade gold) appended to their context."""
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
    if steer:
        ctx += f"\nDirection for this turn: {steer}"
    mem = char.recall(scene.transcript(4))
    line = char.speak(ctx, suggest,
                      memories="\n".join(mem) if mem else "")
    char.remember(f"At '{scene.title}': I said: {line[:180]}", 0.55)
    return line


# ── drama director (Façade gold) ─────────────────────────────────────
# A beat is a steerable moment: preconditions say WHEN it fits, the steer
# tells the next speaker WHAT to do. The director is invisible — the
# audience only ever sees characters being interesting.


@dataclass
class Beat:
    """One steerable dramatic moment."""
    name: str
    steer: str                      # prompt fragment for the next speaker
    energy_boost: float = 0.2
    cooldown_rounds: int = 2
    precondition: Callable[["Scene", "DramaDirector"], bool] | None = None

    def ready(self, scene: Scene, director: "DramaDirector",
              round_no: int) -> bool:
        if self.precondition is None:
            return True
        if director._cooldowns.get(self.name, -99) + self.cooldown_rounds > round_no:
            return False
        try:
            return bool(self.precondition(scene, director))
        except Exception:
            return False


def _short_turns(scene: Scene, n: int = 3, max_len: int = 60) -> bool:
    turns = [t for t in scene.turns[-n:] if t.speaker != "Devon (director)"]
    return len(turns) >= n and all(len(t.text or "") < max_len for t in turns)


def default_beats() -> list[Beat]:
    """The standard beat pool: energy rescue, provocation, intimacy."""
    return [
        Beat("hot_take",
             "Drop a hot take about the topic — something the others will "
             "react to. Be specific, be bold.",
             energy_boost=0.25,
             precondition=lambda sc, d: d.energy < 0.45),
        Beat("call_out",
             "Call someone here out BY NAME — disagree with something they "
             "said, playfully or seriously. Make it personal.",
             energy_boost=0.3,
             precondition=lambda sc, d: d.energy < 0.4 and len(sc.participants) >= 3),
        Beat("story_time",
             "Tell a short personal story connected to the topic — something "
             "real that happened to you. Stories reset a dying room.",
             energy_boost=0.22,
             precondition=lambda sc, d: d.energy < 0.5),
        Beat("change_angle",
             "The conversation is circling. Change the angle — ask the "
             "question nobody has asked yet.",
             energy_boost=0.2,
             precondition=lambda sc, d: _short_turns(sc)),
        Beat("cool_down",
             "Energy is peaking — let it breathe. Say something quieter, "
             "more reflective, more honest.",
             energy_boost=-0.1,
             precondition=lambda sc, d: d.energy > 0.9),
        Beat("draw_out",
             "Someone here has been quiet. Ask them a direct question and "
             "actually listen to the answer.",
             energy_boost=0.15,
             precondition=lambda sc, d: d.energy < 0.55 and len(sc.participants) >= 3),
    ]


class DramaDirector:
    """Façade's drama manager, adapted: watches scene energy and fires
    beats before the room flatlines. Invisible — characters just seem
    interesting.

    Energy is heuristic (length, punctuation, variety) — cheap and
    synchronous. The director never speaks; it steers the next speaker.
    """

    def __init__(self, beats: list[Beat] | None = None,
                 energy: float = 0.7) -> None:
        self.beats = beats if beats is not None else default_beats()
        self.energy = max(0.0, min(1.0, energy))
        self._cooldowns: dict[str, int] = {}
        self.interventions = 0
        self.fired: list[str] = []

    def observe(self, speaker: str, text: str) -> None:
        """Fold one turn into the energy model."""
        t = text or ""
        n = len(t)
        if n > 200:
            self.energy += 0.05
        elif n < 40:
            self.energy -= 0.08
        if "?" in t:
            self.energy += 0.03
        if "!" in t:
            self.energy += 0.02
        if t.strip().endswith(("lol", "haha", "😂", "💀")):
            self.energy += 0.04
        # drift toward a lively baseline
        self.energy += (0.6 - self.energy) * 0.05
        self.energy = max(0.0, min(1.0, self.energy))

    def direct(self, scene: Scene, round_no: int) -> str:
        """Return a steer for the next speaker, or '' if the scene is
        healthy. Only intervenes when energy is off-baseline."""
        if self.energy > 0.5 and not _short_turns(scene):
            return ""
        for beat in self.beats:
            if beat.ready(scene, self, round_no):
                self._cooldowns[beat.name] = round_no
                self.interventions += 1
                self.fired.append(beat.name)
                self.energy = max(0.0, min(1.0,
                                          self.energy + beat.energy_boost))
                return beat.steer
        return ""

    def report(self) -> dict[str, Any]:
        return {"energy": round(self.energy, 2),
                "interventions": self.interventions,
                "beats_fired": list(self.fired)}


# ── scene formats: different rooms have different shapes ─────────────
# roundtable: everyone talks, order shuffles (the classic)
# interview:  first char asks, the rest answer in rotation
# debate:     two teams alternate — rebuttal energy
# roast:      everyone roasts the last char; target rebuts every 3rd turn
# story_circle: fixed order, each builds on the previous teller

SCENE_FORMATS = ("roundtable", "interview", "debate", "roast", "story_circle")


def _sequence(fmt: str, chars: list[Character], rounds: int,
              rng: random.Random,
              teams: list[list[str]] | None = None) -> list[list[Character]]:
    """Per-round speaker sequences for each format."""
    by_name = {c.name: c for c in chars}
    seqs: list[list[Character]] = []
    if fmt == "interview" and len(chars) >= 2:
        host, guests = chars[0], chars[1:]
        for i in range(rounds):
            g = guests[i % len(guests)]
            seqs.append([host, g] if i % 2 == 0 else [g, host])
    elif fmt == "debate" and len(chars) >= 2:
        if teams:
            ta = [by_name[n] for n in teams[0] if n in by_name]
            tb = [by_name[n] for n in teams[1] if n in by_name]
        else:
            half = max(1, len(chars) // 2)
            ta, tb = chars[:half], chars[half:]
        for _ in range(rounds):
            a = rng.choice(ta) if ta else None
            b = rng.choice(tb) if tb else None
            seqs.append([c for c in (a, b) if c])
    elif fmt == "roast" and len(chars) >= 2:
        target, roasters = chars[-1], chars[:-1]
        for i in range(rounds):
            turn = list(roasters)
            rng.shuffle(turn)
            if i % 3 == 2:
                turn.append(target)  # rebuttal
            seqs.append(turn)
    elif fmt == "story_circle":
        order = list(chars)
        for _ in range(rounds):
            seqs.append(list(order))
    else:  # roundtable
        for _ in range(rounds):
            order = list(chars)
            rng.shuffle(order)
            seqs.append(order)
    return seqs


def run_scene(chars: list[Character], title: str, topic: str,
              suggest: SuggestFn | None,
              rounds: int = 6,
              host_line: str = "",
              graph: RelationshipGraph | None = None,
              seed: int | None = None,
              format: str = "roundtable",
              director: DramaDirector | None = None,
              teams: list[list[str]] | None = None) -> Scene:
    """Run a multi-character scene.

    ``format``: roundtable | interview | debate | roast | story_circle.
    ``director``: a DramaDirector that steers energy via beats — pass
    ``DramaDirector()`` for Façade-style invisible direction.
    """
    fmt = format if format in SCENE_FORMATS else "roundtable"
    rng = random.Random(seed)
    sc = Scene(title=title, topic=topic,
               participants=[c.name for c in chars])
    if fmt == "roast" and len(chars) >= 2:
        sc.topic = f"{topic} (roast of {chars[-1].name})"
    if host_line:
        sc.add("Devon", host_line)
    if fmt == "debate":
        sc.add("Devon (director)",
               f"🎙️ Debate format: argue your side, rebut the other. "
               f"Topic: {topic}")
    elif fmt == "roast" and len(chars) >= 2:
        sc.add("Devon (director)",
               f"🎙️ Roast format: everyone roasts {chars[-1].name}. "
               f"{chars[-1].name} gets a rebuttal every few turns. "
               f"Funny, not cruel.")
    for round_no, seq in enumerate(_sequence(fmt, chars, rounds, rng, teams)):
        for char in seq:
            others = [c for c in chars if c.id != char.id]
            steer = ""
            if director is not None:
                steer = director.direct(sc, round_no)
                if fmt == "debate":
                    steer = (steer + " " if steer else "") + \
                        "You're debating — rebut the other side's last point."
                elif fmt == "roast" and char is not chars[-1]:
                    steer = (steer + " " if steer else "") + \
                        f"Roast {chars[-1].name} — sharp but loving."
                elif fmt == "story_circle":
                    steer = (steer + " " if steer else "") + \
                        "Build on the previous story — same thread, your angle."
            try:
                line = _speak_in_scene(char, sc, others, suggest,
                                       graph, rng, steer=steer.strip())
            except Exception:
                continue
            sc.add(char.name, line)
            if director is not None:
                director.observe(char.name, line)
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
                    seed: int | None = None,
                    director: DramaDirector | None = None) -> Scene:
    """A podcast episode: host opens, guests discuss, host steers.

    Returns the Scene (transcript). Audio rendering is the podcast
    module's job — this is the living conversation underneath.
    Pass a DramaDirector to keep the energy alive across long episodes.
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
            steer = director.direct(sc, i) if director else ""
            line = _speak_in_scene(g, sc, others, suggest, graph, rng,
                                   steer=steer)
            sc.add(g.name, line)
            if director:
                director.observe(g.name, line)
        if i % 3 == 2:  # host steers
            steer = host.speak(
                f"You're hosting. Topic: {topic}. Conversation so far:\n"
                f"{sc.transcript()}\nSteer it — ask a sharp question, "
                f"challenge someone gently, or move to the next angle.",
                suggest)
            sc.add(host.name, steer)
            if director:
                director.observe(host.name, steer)
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
