"""Brain<->character interaction: conversation, collaboration, initiation.

The brain (Devon's LLM) and characters are fellow agents. Either side
can speak; either side can start. No dialogue trees — every utterance
is generated fresh from persona + context + memory.

Mining gold: Inworld/Stanford characters act PROACTIVELY — they reach
out driven by motives, mood, and time apart, not just when invoked.
``proactive_pulse`` is the restraint-first version: it usually returns
None (silence is a feature), and only yields an initiation when the
character genuinely has a reason.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .character import Character, SuggestFn

__all__ = ["DialogueTurn", "Dialogue", "converse", "character_initiate",
           "proactive_pulse", "icebreakers"]


def _clamp01(v: Any) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.5


@dataclass
class DialogueTurn:
    speaker: str          # "brain" | character name | "owner"
    text: str
    ts: float = field(default_factory=time.time)


@dataclass
class Dialogue:
    """A running conversation between the brain and one or more characters."""
    participants: list[str] = field(default_factory=list)
    turns: list[DialogueTurn] = field(default_factory=list)
    topic: str = ""

    def add(self, speaker: str, text: str) -> None:
        self.turns.append(DialogueTurn(speaker=speaker, text=text))

    def transcript(self, limit: int = 10) -> str:
        return "\n".join(f"{t.speaker}: {t.text}"
                         for t in self.turns[-limit:])


def _brain_say(brain_persona: str, context: str,
               suggest: SuggestFn | None) -> str:
    prompt = (f"{brain_persona}\nSituation: {context}\n"
              f"Reply naturally, first person, in character as Devon. "
              f"No narration.")
    if suggest is not None:
        try:
            out = (suggest(prompt) or "").strip().strip('"').strip()
            if out:
                return out[:600]
        except Exception:
            pass
    return "[Devon] " + context[:120]


def converse(brain_persona: str, char: Character, opener: str,
             suggest: SuggestFn | None, rounds: int = 3,
             memories: str = "") -> Dialogue:
    """Brain and character talk. ``opener`` is who speaks first's line —
    prefix with 'brain:' or '<name>:' or neither (brain opens)."""
    dlg = Dialogue(participants=["Devon", char.name])
    if opener.startswith("brain:"):
        dlg.add("Devon", opener[len("brain:"):].strip())
        char_turn = True
    elif char.name and opener.lower().startswith(char.name.lower() + ":"):
        text = opener.split(":", 1)[1].strip()
        dlg.add(char.name, text)
        char.remember(f"Devon and I talked: {text[:200]}", 0.6)
        char_turn = False
    else:
        dlg.add("Devon", opener)
        char_turn = True

    for _ in range(rounds):
        ctx = (f"Conversation so far:\n{dlg.transcript()}\n"
               f"Continue naturally.")
        if char_turn:
            mem = char.recall(dlg.transcript(4))
            line = char.speak(ctx, suggest,
                              memories="\n".join(mem) if mem else memories)
            dlg.add(char.name, line)
            char.remember(f"I said to Devon: {line[:200]}", 0.5)
        else:
            line = _brain_say(brain_persona, ctx, suggest)
            dlg.add("Devon", line)
        char_turn = not char_turn

    # background pass: relationships + mood shift from what happened
    try:
        from .processing import SessionEvent, process_session
        from .relationships import RelationshipGraph, BRAIN_ID
        graph = RelationshipGraph()
        deep = any(len(t.text or "") > 200 for t in dlg.turns)
        kind = "deep_conversation" if deep else "good_conversation"
        process_session(
            [SessionEvent(char.id, BRAIN_ID, kind, salience=0.6,
                          note=dlg.transcript(3))],
            {char.id: char}, graph)
    except Exception:
        pass
    return dlg


def character_initiate(char: Character, suggest: SuggestFn | None,
                       reason: str = "") -> str:
    """A character starts something on its own — a thought, a question,
    a game invitation. Driven by persona + goals + mood, not a script."""
    goals = "; ".join(char.goals[:3]) if char.goals else "nothing in particular"
    ctx = (f"You feel like reaching out to Devon (your friend and the "
           f"owner's AI). Your goals: {goals}. "
           + (f"Reason: {reason}. " if reason else "")
           + "Say what comes to mind — a thought, question, or invitation. "
           "One or two sentences.")
    return char.speak(ctx, suggest)


# ── proactivity: should this character reach out right now? ──────────
# Restraint-first: most pulses return None. A character that pings every
# hour is a notification, not a person.

def icebreakers(char: Character) -> list[str]:
    """Mood-flavored openers for when the character reaches out. These
    seed the model's initiation — they're sparks, not scripts."""
    v = char.mood.get("valence", 0.0)
    a = char.mood.get("arousal", 0.3)
    dom = max(char.persona.items(), key=lambda kv: kv[1],
              default=("neutral", 0.5))[0]
    out = []
    if v > 0.4:
        out.append(f"something good happened — {char.name} wants to share it")
    elif v < -0.4:
        out.append(f"{char.name} is feeling low and wants company, not advice")
    if a > 0.7:
        out.append("high energy — a game, a debate, something loud")
    agenda = ""
    try:
        agenda = char.goal_agenda()
    except Exception:
        pass
    if agenda:
        out.append(f"an update on: {agenda.split(';')[0].strip()}")
    if char.core_motive:
        out.append(f"something about: {char.core_motive.lower()}")
    out.append(f"just checking in, {dom} as ever")
    return out[:4]


def proactive_pulse(char: Character, suggest: SuggestFn | None,
                    now: float | None = None,
                    rng: Any = None) -> str | None:
    """One heartbeat of character autonomy. Returns an initiation line,
    or None (the common case).

    A character reaches out when: it's been a while AND (mood is
    extreme, a goal itches, or caprice strikes). Extraversion raises the
    odds; low trust in the owner lowers them.
    """
    now = now or time.time()
    r = rng or random.Random()
    idle_h = (now - char.last_active) / 3600.0
    if idle_h < 6:
        return None  # talked recently — leave them alone
    try:
        extra = float(char.ocean.get("extraversion", 0.5))
    except Exception:
        extra = 0.5
    trust = _clamp01(char.mood.get("trust", 0.5))
    v = abs(float(char.mood.get("valence", 0.0)))
    # base urge grows with silence, capped; mood extremes add fuel
    urge = min(0.5, idle_h / 72.0) + v * 0.2 + extra * 0.15 - (1 - trust) * 0.2
    if r.random() > max(0.0, min(0.6, urge)):
        return None
    sparks = icebreakers(char)
    reason = r.choice(sparks) if sparks else ""
    line = character_initiate(char, suggest, reason=reason)
    char.remember(f"I reached out to Devon: {line[:150]}", 0.55)
    return line
