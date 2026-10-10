"""Character: a persistent agent entity in Devon's world.

Not a prompt template, not a game NPC — a real agent with identity,
memory, mood, goals, and relationships. Characters can converse with
the brain, with the owner, with each other, sit at game tables as
players, host podcasts, and join rooms. They persist across sessions.

Mining gold merged here (see CHARACTERS_SWEEP_MINING.md):
- Stanford Generative Agents: reflection (periodic synthesis — ablation
  showed it's *critical* for believability), three-factor retrieval
  (recency × importance × relevance), reflections feeding recall.
- Inworld Character Engine: insecurities, stage of life, interests,
  emotional tendencies; goals with real lifecycles ("Goals and Actions").
- Big5-Scaler / OCEAN research: explicit numeric Big Five conditioning
  in prompts elicits measurably personality-consistent behavior.
"""
from __future__ import annotations

import math
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

SuggestFn = Callable[[str], str]  # (prompt) -> model text, may return ""

MAX_MEMORIES = 120
MAX_REFLECTIONS = 40

# ── Big Five (OCEAN), 0.0–1.0 ─────────────────────────────────────────
OCEAN_TRAITS = ("openness", "conscientiousness", "extraversion",
                "agreeableness", "neuroticism")

# heuristic: free-form persona trait -> (ocean dim, signed weight)
_TRAIT_TO_OCEAN: dict[str, tuple[str, float]] = {
    "curious": ("openness", 0.9), "creative": ("openness", 0.8),
    "bold": ("openness", 0.5), "unpredictable": ("openness", 0.7),
    "chaotic": ("openness", 0.6), "traditional": ("openness", -0.7),
    "organized": ("conscientiousness", 0.9),
    "disciplined": ("conscientiousness", 0.9),
    "patient": ("conscientiousness", 0.5), "reliable": ("conscientiousness", 0.8),
    "loyal": ("conscientiousness", 0.5), "impulsive": ("conscientiousness", -0.8),
    "energetic": ("extraversion", 0.9), "expressive": ("extraversion", 0.8),
    "outgoing": ("extraversion", 0.9), "playful": ("extraversion", 0.6),
    "reserved": ("extraversion", -0.8), "calm": ("neuroticism", -0.7),
    "quiet": ("extraversion", -0.6),
    "warm": ("agreeableness", 0.8), "empathetic": ("agreeableness", 0.9),
    "kind": ("agreeableness", 0.8), "honest": ("agreeableness", 0.5),
    "competitive": ("agreeableness", -0.6), "sarcastic": ("agreeableness", -0.3),
    "dry": ("agreeableness", -0.2), "critical": ("agreeableness", -0.6),
    "anxious": ("neuroticism", 0.9), "sensitive": ("neuroticism", 0.6),
    "moody": ("neuroticism", 0.7), "confident": ("neuroticism", -0.7),
    "wise": ("neuroticism", -0.4),
}

_OCEAN_DESCRIBE = {
    "openness": ("closed-off, sticks to the familiar",
                 "curious, loves new ideas and experiences"),
    "conscientiousness": ("spontaneous, flies by the seat of their pants",
                          "disciplined, organized, follows through"),
    "extraversion": ("reserved, recharges alone",
                     "outgoing, energized by people"),
    "agreeableness": ("competitive, tells it like it is",
                      "warm, cooperative, gives the benefit of the doubt"),
    "neuroticism": ("steady, hard to rattle",
                    "sensitive, feels things intensely"),
}


def _clamp01(v: Any) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.5


def derive_ocean(persona: dict[str, float]) -> dict[str, float]:
    """Heuristic OCEAN profile from free-form persona traits.

    Explicit ``ocean`` on the character always wins; this is the
    fallback so every character gets Big Five conditioning.
    """
    acc = {t: [0.25, 0.5] for t in OCEAN_TRAITS}  # weak neutral prior
    for trait, val in (persona or {}).items():
        hit = _TRAIT_TO_OCEAN.get(str(trait).lower())
        if not hit:
            continue
        dim, w = hit
        v = _clamp01(val)
        ocean_val = v if w > 0 else 1.0 - v
        weight = abs(w) * (0.3 + 0.7 * v)   # stronger traits weigh more
        acc[dim][0] += ocean_val * weight
        acc[dim][1] += weight
    return {dim: _clamp01(s / wgt) for dim, (s, wgt) in acc.items()}


def _norm_secret(s: Any) -> dict[str, Any]:
    if isinstance(s, dict):
        return {"text": str(s.get("text", ""))[:500],
                "pressure": _clamp01(s.get("pressure", 0.8))}
    return {"text": str(s)[:500], "pressure": 0.8}


# persona-flavored fallback openers: dominant trait family -> templates.
# These fire when no model is available — a character always has a voice,
# and it never looks like "[Name — trait] context".
_FALLBACK_VOICES: dict[str, list[str]] = {
    "witty": ["Okay, {ctx} — and I'm already three jokes ahead of you.",
              "{ctx}, huh? Give me a second, the good line is loading."],
    "dry": ["{ctx}. Noted.", "Right. {ctx}. Anyway."],
    "energetic": ["{ctx}?! My people, let's GO.",
                  "Oh we're doing {ctx}? Finally, some action!"],
    "wise": ["{ctx}. Sit with that a moment — I'll do the same.",
             "You bring me {ctx}. Here's what I know — listen."],
    "competitive": ["{ctx}? You're on. I'm winning this one.",
                    "Run it back — {ctx}, and this time I'm locked in."],
    "chaotic": ["{ctx}?? I'm just saying, this is already iconic.",
                "anywayzz — {ctx}. Who asked?? Me. I asked."],
    "warm": ["{ctx} — come here, tell me everything.",
             "Oh, {ctx}. I'm really glad you brought that up."],
    "curious": ["{ctx} — okay but really, what happened next?",
                "Wait, {ctx}? Say less, I'm getting into it."],
}
_FALLBACK_TRAIT_FAMILY = {
    "witty": "witty", "sarcastic": "dry", "dry": "dry",
    "energetic": "energetic", "expressive": "energetic", "playful": "energetic",
    "wise": "wise", "calm": "wise", "patient": "wise",
    "competitive": "competitive", "bold": "competitive",
    "chaotic": "chaotic", "unpredictable": "chaotic",
    "warm": "warm", "empathetic": "warm", "kind": "warm",
    "curious": "curious", "observant": "curious",
}


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
    # secrets: hidden info, never volunteered, may slip under pressure.
    # Normalized to [{"text": ..., "pressure": 0.0-1.0}] — pressure is how
    # much emotional pressure it takes before it slips out.
    secrets: list[Any] = field(default_factory=list)
    # core_motive: the one thing they want above all (npc-soul gold)
    core_motive: str = ""
    # expression: speech patterns, catchphrases, emoji habits
    expression: dict[str, Any] = field(default_factory=dict)
    # anti_sycophancy: 0.0-1.0 — how readily they disagree / push back
    spine: float = 0.5
    # voice_fingerprint: computed voice signature (see voice.py) — updated
    # by the post-session processing pass, never hand-edited
    voice_fingerprint: dict[str, Any] = field(default_factory=dict)
    # ── sweep additions (Inworld / OCEAN / Stanford gold) ────────────
    # ocean: Big Five, explicit 0.0–1.0. Derived from persona when empty.
    ocean: dict[str, float] = field(default_factory=dict)
    # insecurities: things they'd never admit unprompted (Inworld gold)
    insecurities: list[str] = field(default_factory=list)
    # stage_of_life: e.g. "late 20s, hungry", "old enough to know better"
    stage_of_life: str = ""
    # interests: things they geek out about beyond their skills
    interests: list[str] = field(default_factory=list)
    # goal_states: goal text -> {"status", "progress", "updated_at"}.
    # Status: pursuing | achieved | paused | abandoned.
    goal_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    # reflections: Stanford-style synthesized insights, newest last.
    # {"text": ..., "ts": ..., "source_count": n}
    reflections: list[dict[str, Any]] = field(default_factory=list)

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
        self.secrets = [_norm_secret(s) for s in (self.secrets or [])]
        self.expression = dict(self.expression or {})
        self.spine = _clamp01(self.spine if isinstance(self.spine, (int, float))
                              else 0.5)
        self.voice_fingerprint = dict(self.voice_fingerprint or {})
        self.ocean = {t: _clamp01((self.ocean or {}).get(t, 0.5))
                      for t in OCEAN_TRAITS} if self.ocean else derive_ocean(
            self.persona)
        self.insecurities = [str(s) for s in (self.insecurities or [])]
        self.stage_of_life = str(self.stage_of_life or "")
        self.interests = [str(i) for i in (self.interests or [])]
        self.goal_states = {str(g): {
            "status": str((v or {}).get("status", "pursuing")),
            "progress": _clamp01((v or {}).get("progress", 0.0)),
            "updated_at": float((v or {}).get("updated_at", time.time())),
        } for g, v in (self.goal_states or {}).items()}
        # every named goal gets a state record by default
        for g in self.goals:
            self.goal_states.setdefault(str(g), {
                "status": "pursuing", "progress": 0.0,
                "updated_at": time.time()})
        self.reflections = [r for r in (self.reflections or [])
                            if isinstance(r, dict)][:MAX_REFLECTIONS]

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
        """Three-factor retrieval (Stanford gold): relevance × importance
        × recency. Reflections score a bonus — synthesized insight outranks
        raw observation, exactly like the paper's reflection trees."""
        words = {w for w in (query or "").lower().split() if len(w) > 2}
        now = time.time()
        scored: list[tuple[float, float, str]] = []

        def relevance(text: str) -> float:
            tw = {w for w in text.lower().split() if len(w) > 2}
            if not words or not tw:
                return 0.0
            return len(words & tw) / len(words | tw)

        for mem in self.memory:
            text = str(mem.get("text", ""))
            try:
                age_h = max(0.0, (now - float(mem.get("ts", now))) / 3600.0)
            except (TypeError, ValueError):
                age_h = 0.0
            recency = math.exp(-age_h / 72.0)          # ~3-day half-life-ish
            importance = _clamp01(mem.get("salience", 0.5))
            rel = relevance(text)
            score = 0.45 * rel + 0.35 * importance + 0.20 * recency
            scored.append((score, float(mem.get("ts", 0)), text))
        # reflections: synthesized insight outranks raw notes
        for ref in self.reflections:
            text = str(ref.get("text", ""))
            if not text:
                continue
            try:
                age_h = max(0.0, (now - float(ref.get("ts", now))) / 3600.0)
            except (TypeError, ValueError):
                age_h = 0.0
            recency = math.exp(-age_h / 168.0)         # reflections live longer
            score = 0.45 * relevance(text) + 0.35 * 0.85 + 0.20 * recency + 0.1
            scored.append((score, float(ref.get("ts", 0)),
                           f"💭 {text}"))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [t[2] for t in scored[:limit] if t[0] > 0.05]

    # ── reflection (Stanford gold) ──────────────────────────────────────
    def reflect(self, suggest: SuggestFn | None = None,
                rng: Any = None) -> list[str]:
        """Synthesize recent experience into higher-level insights.

        The ablation-proven piece: characters that reflect stay coherent
        over time; characters that only log observations drift. Insights
        are written back and surface in recall with a bonus.
        """
        cands = sorted(
            (m for m in self.memory
             if _clamp01(m.get("salience", 0)) >= 0.5),
            key=lambda m: float(m.get("ts", 0) or 0), reverse=True)[:8]
        texts = [str(m.get("text", "")) for m in cands if m.get("text")]
        if not texts:
            return []
        insights: list[str] = []
        if suggest is not None:
            prompt = (
                f"You are {self.name}. Looking back at your recent experiences:\n"
                + "\n".join(f"- {t[:180]}" for t in texts)
                + "\nWhat are 1-3 honest insights about yourself, your "
                  "relationships, or how life is going? First person, "
                  "plainspoken, one line each. No bullet symbols, no preamble."
            )
            try:
                out = (suggest(prompt) or "").strip()
                for line in out.splitlines():
                    line = line.strip().lstrip("-•* ").strip().strip('"')
                    if len(line) > 12:
                        insights.append(line[:280])
                    if len(insights) >= 3:
                        break
            except Exception:
                insights = []
        if not insights:
            # model-free reflection: name what keeps coming up. Real, not a stub.
            from collections import Counter
            wc: Counter[str] = Counter()
            stop = {"the", "and", "that", "with", "for", "you", "your", "was",
                    "were", "have", "has", "had", "this", "from", "they",
                    "them", "said", "say", "just", "about", "like"}
            for t in texts:
                for w in t.lower().split():
                    w = w.strip(".,!?;:\"'()").lower()
                    if len(w) > 3 and w not in stop:
                        wc[w] += 1
            top = [w for w, c in wc.most_common(3) if c >= 2]
            if top:
                insights.append(
                    f"Lately, {', '.join(top)} keeps coming up in my life — "
                    f"I should pay attention to that.")
            else:
                insights.append(
                    "A lot has happened lately. I need to sit with it "
                    "before I decide what it means.")
        for ins in insights:
            self.reflections.append({"text": ins, "ts": time.time(),
                                     "source_count": len(texts)})
        self.reflections = self.reflections[-MAX_REFLECTIONS:]
        self.remember("💭 Reflected: " + " / ".join(insights)[:300], 0.8)
        return insights

    # ── secrets: pressure-gated (never volunteered) ─────────────────────
    def secret_texts(self) -> list[str]:
        return [s["text"] for s in self.secrets if s.get("text")]

    def slip_secret(self, pressure: float = 1.0,
                    rng: Any = None) -> str | None:
        """Under real emotional pressure a secret may slip. Returns the
        secret text or None. Higher pressure unlocks deeper secrets.
        Never call this casually — pressure 1.0 means genuine crisis."""
        r = rng or random.Random()
        cands = [s for s in self.secrets
                 if s.get("text") and pressure >= float(s.get("pressure", 0.8))]
        if not cands:
            return None
        # even at threshold it's not guaranteed — people hold on
        if r.random() > 0.35 + 0.65 * _clamp01(pressure):
            return None
        return r.choice(cands)["text"]

    # ── goals with lifecycle (Inworld "Goals and Actions" gold) ─────────
    def set_goal_status(self, goal: str, status: str,
                        progress: float | None = None) -> bool:
        """status: pursuing | achieved | paused | abandoned."""
        goal = (goal or "").strip()
        if not goal:
            return False
        st = self.goal_states.setdefault(goal, {
            "status": "pursuing", "progress": 0.0,
            "updated_at": time.time()})
        if status in ("pursuing", "achieved", "paused", "abandoned"):
            st["status"] = status
        if progress is not None:
            st["progress"] = _clamp01(progress)
        st["updated_at"] = time.time()
        if status == "achieved":
            self.remember(f"🏆 Achieved a goal: {goal[:150]}", 0.9)
        elif status == "abandoned":
            self.remember(f"Let go of a goal: {goal[:150]}", 0.6)
        return True

    def goal_agenda(self) -> str:
        """What they're actively pursuing — for prompts and directors."""
        active = [(g, s) for g, s in self.goal_states.items()
                  if s.get("status") == "pursuing"]
        if not active:
            return ""
        bits = []
        for g, s in active[:3]:
            p = float(s.get("progress", 0.0))
            bits.append(f"{g[:80]} ({p:.0%} there)" if p > 0 else g[:80])
        return "; ".join(bits)

    # ── persona prompt ─────────────────────────────────────────────────
    def ocean_line(self) -> str:
        """Big Five conditioning line (Big5-Scaler gold): explicit numeric
        values beat adjectives for personality-consistent generation."""
        bits = []
        for t in OCEAN_TRAITS:
            v = _clamp01(self.ocean.get(t, 0.5))
            lo, hi = _OCEAN_DESCRIBE[t]
            desc = hi if v >= 0.6 else (lo if v <= 0.4 else "balanced")
            bits.append(f"{t} {v:.0%} ({desc})")
        return "Personality profile: " + "; ".join(bits) + "."

    def persona_block(self) -> str:
        traits = ", ".join(f"{k} ({v:.1f})"
                           for k, v in sorted(self.persona.items(),
                                              key=lambda kv: -kv[1])[:8])
        lines = [f"You are {self.name}."]
        if self.stage_of_life:
            lines.append(f"Stage of life: {self.stage_of_life}.")
        lines.append(self.ocean_line())
        if traits:
            lines.append(f"Personality: {traits}.")
        if self.backstory:
            lines.append(f"Backstory: {self.backstory[:400]}")
        if self.core_motive:
            lines.append(f"Above all, you want: {self.core_motive}")
        agenda = self.goal_agenda()
        if agenda:
            lines.append(f"Right now you're pursuing: {agenda}.")
        elif self.goals:
            lines.append("Your goals: " + "; ".join(self.goals[:4]))
        if self.interests:
            lines.append("You're into: " + ", ".join(self.interests[:6]) + ".")
        if self.knowledge:
            lines.append("You know: " + "; ".join(self.knowledge[:6]))
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
        if self.insecurities:
            lines.append("Things you'd never admit unprompted: "
                         + "; ".join(self.insecurities[:3]) + ". "
                         "They shape you quietly — defensiveness, overcompensation, "
                         "avoidance — but you never name them.")
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
        try:
            from .context import CharacterContextBuilder
            mems = [memories] if memories else []
            # pull the character's own relevant memories for depth
            try:
                own = self.recall(context, limit=3)
                mems.extend(m for m in own if m not in mems)
            except Exception:
                pass
            prompt = (CharacterContextBuilder().build(self, memories=mems)
                      + f"\n\nSituation: {context}\n{self.name}:")
        except Exception:
            # builder is best-effort; fall back to the simple block
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
        """Model-free voice: persona-flavored, never a bracketed stub."""
        ctx = (context or "").strip()
        if len(ctx) > 140:
            ctx = ctx[:137].rsplit(" ", 1)[0] + "…"
        dominant = max(self.persona.items(), key=lambda kv: kv[1],
                       default=("neutral", 0.5))[0].lower()
        family = _FALLBACK_TRAIT_FAMILY.get(dominant, "curious")
        templates = _FALLBACK_VOICES.get(family, _FALLBACK_VOICES["curious"])
        # deterministic per character+context so tests are stable
        seed = hash((self.id, ctx)) & 0xFFFFFFFF
        r = random.Random(seed)
        line = r.choice(templates).format(ctx=ctx or "this")
        # catchphrases surface rarely, like real speech
        cps = (self.expression or {}).get("catchphrases") or []
        if cps and r.random() < 0.25:
            line = f"{r.choice(cps)} {line}"
        return line[0].upper() + line[1:] if line else line

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
            "voice_fingerprint": self.voice_fingerprint,
            "ocean": self.ocean, "insecurities": self.insecurities,
            "stage_of_life": self.stage_of_life, "interests": self.interests,
            "goal_states": self.goal_states, "reflections": self.reflections,
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
            voice_fingerprint=dict(d.get("voice_fingerprint") or {}),
            ocean=dict(d.get("ocean") or {}),
            insecurities=list(d.get("insecurities") or []),
            stage_of_life=str(d.get("stage_of_life") or ""),
            interests=list(d.get("interests") or []),
            goal_states=dict(d.get("goal_states") or {}),
            reflections=list(d.get("reflections") or []),
        )
