"""Brain<->character interaction: conversation, collaboration, initiation.

The brain (Devon's LLM) and characters are fellow agents. Either side
can speak; either side can start. No dialogue trees — every utterance
is generated fresh from persona + context + memory.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .character import Character, SuggestFn

__all__ = ["DialogueTurn", "Dialogue", "converse", "character_initiate"]


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
