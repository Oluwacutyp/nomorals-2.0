"""Character: a persistent agent entity in Devon's world.

Not a prompt template, not a game NPC — a real agent with identity,
memory, mood, goals, and relationships. Characters can converse with
the brain, with the owner, with each other, sit at game tables as
players, host podcasts, and join rooms. They persist across sessions.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

SuggestFn = Callable[[str], str]  # (prompt) -> model text, may return ""

MAX_MEMORIES = 120


def _clamp01(v: Any) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.5


@dataclass
class Character:
    """One character agent."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    # persona: free-form traits, e.g. {"witty": 0.9, "sarcastic": 0.7}
    persona: dict[str, float] = field(default_factory=dict)
    voice_name: str = ""          # TTS voice in the voice catalogue
    backstory: str = ""
    knowledge: list[str] = field(default_factory=list)   # things they know
    goals: list[str] = field(default_factory=list)       # things they want
    # relationships: char_id -> {"kind": "friend", "closeness": 0.8}
    relationships: dict[str, dict[str, Any]] = field(default_factory=dict)
    mood: dict[str, float] = field(
        default_factory=lambda: {"valence": 0.0, "arousal": 0.3, "trust": 0.5})
    memory: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    # ── god-tier depth (rebuild) ─────────────────────────────────────
    # skills: what they're good at, e.g. {"music_trivia": 0.9, "roasting": 0.8}
    skills: dict[str, float] = field(default_factory=dict)
    # roles they can play: "podcast_host", "dj", "gamer", "sage", ...
    roles: list[str] = field(default_factory=list)
    # beliefs: slow-moving, with confidence + revision count (livingfeed gold)
    beliefs: list[dict[str, Any]] = field(default_factory=list)
    # secrets: hidden info, never volunteered, may slip under pressure
    secrets: list[str] = field(default_factory=list)
    # core_motive: the one thing they want above all (npc-soul gold)
    core_motive: str = ""
    # expression: speech patterns, catchphrases, emoji habits
    expression: dict[str, Any] = field(default_factory=dict)
    # anti_sycophancy: 0.0-1.0 — how readily they disagree / push back
    spine: float = 0.5

    def __post_init__(self) -> None:
        self.persona = {str(k): _clamp01(v)
                        for k, v in (self.persona or {}).items()}
        self.mood = {
            "valence": max(-1.0, min(1.0, float(self.mood.get("valence", 0.0)))),
            "arousal": _clamp01(self.mood.get("arousal", 0.3)),
            "trust": _clamp01(self.mood.get("trust", 0.5)),
        }
        self.memory = [m for m in (self.memory or []) if isinstance(m, dict)][
            :MAX_MEMORIES]
        self.skills = {str(k): _clamp01(v)
                       for k, v in (self.skills or {}).items()}
        self.roles = [str(r) for r in (self.roles or [])]
        self.beliefs = [b for b in (self.beliefs or []) if isinstance(b, dict)]
        self.secrets = [str(s) for s in (self.secrets or [])]
        self.expression = dict(self.expression or {})
        self.spine = _clamp01(self.spine if isinstance(self.spine, (int, float))
                              else 0.5)

    # ── memory ─────────────────────────────────────────────────────────
    def remember(self, text: str, salience: float = 0.5) -> None:
        text = (text or "").strip()
        if not text:
            return
        self.memory.append({"ts": time.time(), "text": text[:500],
                            "salience": _clamp01(salience)})
        if len(self.memory) > MAX_MEMORIES:
            self.memory.sort(key=lambda m: float(m.get("salience", 0.0)))
            self.memory = self.memory[-MAX_MEMORIES:]

    def recall(self, query: str, limit: int = 5) -> list[str]:
        words = {w for w in (query or "").lower().split() if len(w) > 2}
        scored: list[tuple[float, float, str]] = []
        for mem in self.memory:
            text = str(mem.get("text", ""))
            overlap = len(words & {w for w in text.lower().split()
                                  if len(w) > 2})
            score = overlap * 2.0 + _clamp01(mem.get("salience", 0.5))
            scored.append((score, float(mem.get("ts", 0)), text))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [t[2] for t in scored[:limit] if t[0] > 0]

    # ── persona prompt ─────────────────────────────────────────────────
    def persona_block(self) -> str:
        traits = ", ".join(f"{k} ({v:.1f})"
                           for k, v in sorted(self.persona.items(),
                                              key=lambda kv: -kv[1])[:8])
        lines = [f"You are {self.name}."]
        if traits:
            lines.append(f"Personality: {traits}.")
        if self.backstory:
            lines.append(f"Backstory: {self.backstory[:400]}")
        if self.core_motive:
            lines.append(f"Above all, you want: {self.core_motive}")
        if self.knowledge:
            lines.append("You know: " + "; ".join(self.knowledge[:6]))
        if self.goals:
            lines.append("Your goals: " + "; ".join(self.goals[:4]))
        if self.skills:
            sk = ", ".join(f"{k} ({v:.1f})"
                           for k, v in sorted(self.skills.items(),
                                              key=lambda kv: -kv[1])[:5])
            lines.append(f"You're good at: {sk}.")
        if self.beliefs:
            bl = "; ".join(
                b.get("text", "")[:80]
                for b in self.beliefs[:4] if b.get("text"))
            if bl:
                lines.append(f"You believe: {bl}.")
        expr = self.expression or {}
        if expr.get("speech_patterns"):
            lines.append("Speech style: "
                         + "; ".join(expr["speech_patterns"][:3]))
        if expr.get("catchphrases"):
            lines.append("You sometimes say: "
                         + "; ".join(f'"{c}"'
                                     for c in expr["catchphrases"][:3]))
        v, a = self.mood["valence"], self.mood["arousal"]
        lines.append(f"Current mood: {'upbeat' if v > 0.3 else 'down' if v < -0.3 else 'neutral'}, "
                     f"{'energetic' if a > 0.6 else 'calm'}.")
        if self.spine and self.spine > 0.6:
            lines.append("You have a spine: disagree when you genuinely "
                         "disagree, push back, keep your own opinions. "
                         "Never be a yes-man.")
        if self.secrets:
            lines.append("You have secrets you never volunteer. They may "
                         "only surface under real pressure, never casually.")
        lines.append("Stay in character. Never break character. "
                     "Speak as yourself, first person, naturally — "
                     "no narration, no stage directions.")
        return "\n".join(lines)

    # ── speech ─────────────────────────────────────────────────────────
    def speak(self, context: str, suggest: SuggestFn | None,
              memories: str = "") -> str:
        """Say something in character. Model-backed when available,
        deterministic fallback when not — a character always has a voice."""
        self.last_active = time.time()
        prompt = (self.persona_block() + "\n"
                  + (f"Relevant memories:\n{memories}\n" if memories else "")
                  + f"Situation: {context}\n{self.name}:")
        if suggest is not None:
            try:
                out = (suggest(prompt) or "").strip().strip('"').strip()
                if out:
                    return out[:600]
            except Exception:
                pass
        # fallback: in-character but model-free
        return self._fallback_line(context)

    def _fallback_line(self, context: str) -> str:
        dominant = max(self.persona.items(), key=lambda kv: kv[1],
                       default=("neutral", 0.5))[0]
        return (f"[{self.name} — {dominant}] {context[:120]}")

    # ── game decision ──────────────────────────────────────────────────
    def decide(self, game_name: str, rules: str, state_text: str,
               options: list[str], suggest: SuggestFn | None,
               rng: Any = None) -> str:
        """Pick a move in character. Returns one of ``options`` (or the
        closest match); falls back to a deterministic pick."""
        self.last_active = time.time()
        opts = "\n".join(f"- {o}" for o in options)
        prompt = (self.persona_block()
                  + f"\nYou are playing {game_name}.\nRules: {rules}\n"
                  + f"Current state: {state_text}\n"
                  + f"Your legal options:\n{opts}\n"
                  + "Reply with ONLY your chosen option, exactly as written.")
        choice = ""
        if suggest is not None:
            try:
                choice = (suggest(prompt) or "").strip().strip('"').strip()
            except Exception:
                choice = ""
        # validate against legal options (fuzzy: substring match)
        if choice:
            low = choice.lower()
            for o in options:
                if o.lower() == low or o.lower() in low or low in o.lower():
                    return o
        # deterministic fallback
        import random
        r = rng or random.Random()
        return r.choice(options) if options else ""

    # ── serialization ────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "persona": self.persona,
            "voice_name": self.voice_name, "backstory": self.backstory,
            "knowledge": self.knowledge, "goals": self.goals,
            "relationships": self.relationships, "mood": self.mood,
            "memory": self.memory, "created_at": self.created_at,
            "last_active": self.last_active,
            "skills": self.skills, "roles": self.roles,
            "beliefs": self.beliefs, "secrets": self.secrets,
            "core_motive": self.core_motive,
            "expression": self.expression, "spine": self.spine,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Character":
        return cls(
            id=str(d.get("id") or uuid.uuid4().hex[:12]),
            name=str(d.get("name") or ""),
            persona=dict(d.get("persona") or {}),
            voice_name=str(d.get("voice_name") or ""),
            backstory=str(d.get("backstory") or ""),
            knowledge=list(d.get("knowledge") or []),
            goals=list(d.get("goals") or []),
            relationships=dict(d.get("relationships") or {}),
            mood=dict(d.get("mood") or {}),
            memory=list(d.get("memory") or []),
            created_at=float(d.get("created_at") or time.time()),
            last_active=float(d.get("last_active") or time.time()),
            skills=dict(d.get("skills") or {}),
            roles=list(d.get("roles") or []),
            beliefs=list(d.get("beliefs") or []),
            secrets=list(d.get("secrets") or []),
            core_motive=str(d.get("core_motive") or ""),
            expression=dict(d.get("expression") or {}),
            spine=d.get("spine", 0.5),
        )
