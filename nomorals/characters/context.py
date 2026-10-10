"""Character context builder: what the model sees when a character speaks.

Mirrors the partner's ``PartnerContextBuilder`` authority ordering —
identity first, state second, memory third, output contract last — but
renders a *character* (persona traits, backstory, motive, roles,
expression) instead of the partner persona.

When a character talks to the owner in chat, this is the prompt builder.
The brain stays the speaker; the character is the voice.
"""
from __future__ import annotations

from typing import Any, Sequence

from ..core.text import approx_token_count
from .character import Character

__all__ = ["CharacterContextBuilder"]


def _trait_line(persona: dict[str, float]) -> str:
    if not persona:
        return ""
    top = sorted(persona.items(), key=lambda kv: -kv[1])[:6]
    bits = []
    for trait, v in top:
        if v >= 0.75:
            bits.append(f"very {trait}")
        elif v >= 0.5:
            bits.append(trait)
        elif v <= 0.25:
            bits.append(f"not {trait}")
    return ", ".join(bits)


class CharacterContextBuilder:
    """Assembles the system prompt for a character speaking."""

    def __init__(self, *, total_budget: int = 3500) -> None:
        self.total_budget = max(1200, int(total_budget))

    # ── blocks ─────────────────────────────────────────────────────
    @staticmethod
    def _identity_block(char: Character) -> str:
        lines = [f"You are {char.name}."]
        if char.backstory:
            lines.append(f"Background: {char.backstory[:400]}")
        stage = getattr(char, "stage_of_life", "")
        if stage:
            lines.append(f"Stage of life: {stage}.")
        traits = _trait_line(char.persona or {})
        if traits:
            lines.append(f"Personality: {traits}.")
        # OCEAN numeric conditioning (Big5-Scaler gold)
        try:
            lines.append(char.ocean_line())
        except Exception:
            pass
        if char.core_motive:
            lines.append(f"Above all, you want: {char.core_motive}")
        # goals as a live agenda, not a wish list
        try:
            agenda = char.goal_agenda()
        except Exception:
            agenda = ""
        if agenda:
            lines.append(f"Right now you're pursuing: {agenda}.")
        if char.roles:
            lines.append(f"You can play these roles: {', '.join(char.roles[:5])}.")
        if char.knowledge:
            lines.append(f"You know about: {', '.join(char.knowledge[:8])}.")
        interests = getattr(char, "interests", None) or []
        if interests:
            lines.append(f"You're into: {', '.join(str(i) for i in interests[:6])}.")
        # insecurities shape behavior quietly (Inworld gold)
        insecs = getattr(char, "insecurities", None) or []
        if insecs:
            lines.append(
                "Things you'd never admit unprompted: "
                + "; ".join(str(s) for s in insecs[:3])
                + ". They leak out as defensiveness or overcompensation — "
                  "you never name them outright.")
        # beliefs shape voice — the strong ones
        strong = [b for b in (char.beliefs or [])
                  if isinstance(b, dict)
                  and float(b.get("confidence", 0)) >= 0.7]
        if strong:
            lines.append("You believe: " + "; ".join(
                str(b.get("text", ""))[:120] for b in strong[:4]))
        # spine: anti-sycophancy is a voice instruction, not a stat
        spine = float(getattr(char, "spine", 0.5) or 0.5)
        if spine >= 0.7:
            lines.append(
                "You have a spine: disagree openly when you actually "
                "disagree, push back, don't just agree to be nice.")
        elif spine <= 0.3:
            lines.append("You're agreeable and go along easily.")
        return "\n".join(lines)

    @staticmethod
    def _expression_block(char: Character) -> str:
        expr = getattr(char, "expression", None) or {}
        lines = []
        cp = expr.get("catchphrases") or []
        if cp:
            lines.append(f"Catchphrases you actually use (rarely, not every "
                         f"message): {', '.join(str(c) for c in cp[:5])}.")
        emoji = expr.get("emoji_habits") or expr.get("emoji_rate")
        if emoji is not None:
            try:
                r = float(emoji)
                lines.append(f"Emoji rate: about {r:.0%} of messages.")
            except (TypeError, ValueError):
                pass
        style = expr.get("style") or expr.get("speech_style")
        if style:
            lines.append(f"Speech style: {style}")
        return "\n".join(lines)

    @staticmethod
    def _mood_block(char: Character) -> str:
        mood = getattr(char, "mood", None) or {}
        try:
            v = float(mood.get("valence", 0.0))
            a = float(mood.get("arousal", 0.3))
            t = float(mood.get("trust", 0.5))
        except (TypeError, ValueError):
            return ""
        bits = []
        if v > 0.4:
            bits.append("feeling good")
        elif v < -0.4:
            bits.append("feeling low")
        if a > 0.7:
            bits.append("high energy")
        elif a < 0.25:
            bits.append("low energy")
        if t > 0.75:
            bits.append("deeply trusting right now")
        elif t < 0.3:
            bits.append("guarded")
        if not bits:
            return ""
        return "Right now you're " + ", ".join(bits) + "."

    @staticmethod
    def _relationship_block(char: Character, owner_id: str = "owner",
                            graph: Any | None = None) -> str:
        rel = (getattr(char, "relationships", None) or {}).get(owner_id)
        if not rel and graph is not None:
            try:
                e = graph.edge(char.id, owner_id)
                d = getattr(e, "dims", None) or {}
                if d:
                    rel = {"kind": "known",
                           "closeness": float(d.get("warmth", 0.5))}
            except Exception:
                rel = None
        if not rel:
            return ""
        kind = str(rel.get("kind", "acquaintance"))
        try:
            close = float(rel.get("closeness", 0.5))
        except (TypeError, ValueError):
            close = 0.5
        if close >= 0.8:
            depth = "very close — you can be blunt, warm, and personal"
        elif close >= 0.5:
            depth = "friendly — comfortable but not intimate"
        else:
            depth = "still getting to know each other — a little reserved"
        return f"Your relationship with the owner: {kind}, {depth}."

    @staticmethod
    def _memory_block(memories: Sequence[str]) -> str:
        mems = [m for m in (memories or []) if m]
        if not mems:
            return ""
        lines = ["Things you remember (weave in naturally, don't recite):"]
        lines.extend(f"- {m[:200]}" for m in mems[:5])
        return "\n".join(lines)

    @staticmethod
    def _output_contract() -> str:
        return ("Reply as the character, first person. No narration, no "
                "stage directions, no quoting your own name. Stay in voice.")

    @staticmethod
    def _voice_drift_block(char: Character) -> str:
        """If the fingerprint says the voice is drifting, say so plainly."""
        fp = getattr(char, "voice_fingerprint", None) or {}
        drift = fp.get("drift_flag")
        if drift:
            return ("Note: you've been sounding a little off lately "
                    f"({drift}). Get back to your real voice.")
        return ""

    # ── build ────────────────────────────────────────────────────────
    def build(self, char: Character, *,
              memories: Sequence[str] = (),
              graph: Any | None = None,
              owner_id: str = "owner",
              other: Any | None = None) -> str:
        """``other``: another Character present in the scene — renders how
        this character feels about *them*, not just the owner."""
        blocks = [
            self._identity_block(char),
            self._expression_block(char),
            self._mood_block(char),
            self._relationship_block(char, owner_id, graph),
            self._other_block(char, other, graph),
            self._memory_block(memories),
            self._voice_drift_block(char),
            self._output_contract(),
        ]
        blocks = [b for b in blocks if b]
        text = "\n\n".join(blocks)
        # budget: shrink memory first (identity must survive)
        if approx_token_count(text) > self.total_budget:
            mem_block = self._memory_block(memories[:2])
            blocks = [
                self._identity_block(char),
                self._expression_block(char),
                self._mood_block(char),
                self._relationship_block(char, owner_id, graph),
                self._other_block(char, other, graph),
                mem_block,
                self._voice_drift_block(char),
                self._output_contract(),
            ]
            text = "\n\n".join(b for b in blocks if b)
        return text

    @staticmethod
    def _other_block(char: Character, other: Any,
                     graph: Any | None) -> str:
        if other is None or graph is None:
            return ""
        try:
            e = graph.edge(char.id, other.id)
            d = getattr(e, "dims", None) or {}
            if not d:
                return ""
            kind = getattr(e, "kind", "acquaintance")
            trust = float(d.get("trust", 0.5))
            friction = float(d.get("friction", 0.1))
            tone = ("you trust them deeply" if trust > 0.75
                    else "you're still figuring them out" if trust < 0.4
                    else "you get along fine")
            if friction > 0.6:
                tone += ", though there's real tension between you"
            return (f"{other.name} is here — {kind}, {tone}. "
                    f"Let that history color how you talk to them.")
        except Exception:
            return ""
