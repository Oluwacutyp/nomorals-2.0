"""BookForge collaborative writing — owner + brain + characters as co-authors.

The spec is the floor. This goes beyond: real collaborative fiction where
the owner directs, the brain orchestrates, and character agents write scenes
from their own POV. Plus a writer-critic loop that reviews every chapter
before it ships.

Mined gold:
- Writer-Critic (botlab/swarms): writer drafts, critic reviews, iterate
- AI Writing Studio: specialized agents (continuity, character, timeline)
- HNPX: human guides at each level, LLM as responsive collaborator
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Critique:
    """A critic's review of a chapter draft."""
    score: float  # 0-1 overall
    strengths: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    verdict: str = ""  # "ship" | "revise" | "rewrite"

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "strengths": self.strengths,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "verdict": self.verdict,
        }


class CriticAgent:
    """Reviews chapter drafts like a brutal-but-fair editor.

    Checks: pacing, character voice consistency, plot mechanics,
    show-don't-tell, dialogue quality, continuity with story bible.
    """

    def __init__(self, context: Any = None) -> None:
        self.context = context

    def review(self, text: str, brief: dict[str, Any],
               bible: dict[str, Any] | None = None,
               suggest: Callable | None = None) -> Critique:
        """Review a draft chapter. Returns structured critique."""
        if suggest is None:
            return self._heuristic_review(text, brief, bible)

        prompt = self._build_prompt(text, brief, bible)
        try:
            raw = suggest(prompt, max_tokens=800)
            return self._parse_critique(raw)
        except Exception:
            return self._heuristic_review(text, brief, bible)

    def _build_prompt(self, text: str, brief: dict[str, Any],
                      bible: dict[str, Any] | None) -> str:
        bible_txt = ""
        if bible:
            cast = ", ".join(
                c.get("name", "?") for c in bible.get("cast", [])[:8])
            bible_txt = f"\nCAST: {cast}\n"
        return (
            "You are a brutal-but-fair fiction editor. Review this chapter draft.\n"
            f"BRIEF: {brief.get('summary', '')}\n{bible_txt}\n"
            f"DRAFT:\n{text[:6000]}\n\n"
            "Respond in this exact format:\n"
            "SCORE: <0-100>\n"
            "STRENGTHS: <comma-separated>\n"
            "ISSUES: <comma-separated>\n"
            "SUGGESTIONS: <comma-separated>\n"
            "VERDICT: <ship|revise|rewrite>"
        )

    def _parse_critique(self, raw: str) -> Critique:
        lines = {}
        for line in raw.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                lines[k.strip().upper()] = v.strip()
        score = 70.0
        try:
            score = float(lines.get("SCORE", "70")) / 100.0
        except ValueError:
            pass
        verdict = lines.get("VERDICT", "revise").lower()
        if verdict not in ("ship", "revise", "rewrite"):
            verdict = "revise"
        return Critique(
            score=max(0.0, min(1.0, score)),
            strengths=[s.strip() for s in lines.get("STRENGTHS", "").split(",") if s.strip()],
            issues=[s.strip() for s in lines.get("ISSUES", "").split(",") if s.strip()],
            suggestions=[s.strip() for s in lines.get("SUGGESTIONS", "").split(",") if s.strip()],
            verdict=verdict,
        )

    def _heuristic_review(self, text: str, brief: dict[str, Any],
                          bible: dict[str, Any] | None) -> Critique:
        """Fallback when no model: structural checks, never fake praise."""
        words = len(text.split())
        issues, strengths, suggestions = [], [], []

        if words < 300:
            issues.append("chapter too short — underdeveloped")
            suggestions.append("expand key scenes with sensory detail")
        elif words > 5000:
            issues.append("chapter bloated — pacing risk")
            suggestions.append("cut or split; every scene must earn its place")

        # Dialogue ratio check
        quotes = text.count('"') + text.count('"') + text.count('"')
        dialogue_ratio = quotes / max(len(text), 1)
        if dialogue_ratio < 0.02 and words > 800:
            issues.append("dialogue-starved — characters feel distant")
            suggestions.append("let characters speak; break up exposition")
        elif dialogue_ratio > 0.15:
            strengths.append("strong dialogue presence")

        # Show-don't-tell signals
        telling = sum(text.lower().count(w) for w in
                      ["felt sad", "was happy", "was angry", "felt scared"])
        if telling > 3:
            issues.append("telling emotions instead of showing")
            suggestions.append("replace emotion labels with physical/sensory beats")

        # Repetition check
        sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        if len(sentences) != len(set(sentences)) and sentences:
            issues.append("repeated sentences detected")

        if not issues:
            strengths.append("structurally sound draft")

        score = 0.85 if not issues else max(0.4, 0.75 - len(issues) * 0.08)
        verdict = "ship" if score >= 0.8 and not issues else "revise"
        return Critique(score=score, strengths=strengths, issues=issues,
                        suggestions=suggestions, verdict=verdict)


class CollaborativeSession:
    """A writing session: owner directs, brain orchestrates, characters write.

    Characters can author scenes from their POV — a villain writes their own
    scheming scene, a lover writes the reunion. The brain weaves it together.
    The critic reviews before anything ships.
    """

    def __init__(self, story_slug: str, context: Any = None) -> None:
        self.story_slug = story_slug
        self.context = context
        self.critic = CriticAgent(context)
        self._characters: dict[str, Any] = {}
        self._log: list[dict[str, Any]] = []

    def cast_character(self, character: Any) -> None:
        """Add a character agent as a co-author."""
        self._characters[character.name] = character

    def write_scene(self, character_name: str, prompt: str,
                    suggest: Callable | None = None) -> dict[str, Any]:
        """A character writes a scene from their POV."""
        char = self._characters.get(character_name)
        if char is None:
            return {"ok": False, "reason": f"character '{character_name}' not in session"}

        if suggest is not None:
            try:
                persona = char.persona_block() if hasattr(char, "persona_block") else str(char)
                full_prompt = (
                    f"{persona}\n\n"
                    f"You are writing a scene from your own point of view. "
                    f"Stay in character — your voice, your biases, your blind spots.\n\n"
                    f"SCENE PROMPT: {prompt}\n\n"
                    f"Write the scene (300-800 words):"
                )
                text = suggest(full_prompt, max_tokens=1200)
            except Exception as exc:
                return {"ok": False, "reason": f"generation failed: {exc}"}
        else:
            text = char._fallback_line(prompt) if hasattr(char, "_fallback_line") else ""

        entry = {
            "character": character_name,
            "prompt": prompt,
            "text": text,
            "words": len(text.split()),
        }
        self._log.append(entry)

        # Character remembers writing this
        if hasattr(char, "remember"):
            char.remember(f"I wrote a scene: {prompt[:100]}", salience=0.7)

        return {"ok": True, **entry}

    def critique(self, text: str, brief: dict[str, Any],
                 bible: dict[str, Any] | None = None,
                 suggest: Callable | None = None) -> dict[str, Any]:
        """Run the critic on a draft."""
        c = self.critic.review(text, brief, bible, suggest)
        return {"ok": True, "critique": c.to_dict()}

    def revise_with_critique(self, text: str, critique: dict[str, Any],
                             suggest: Callable | None = None) -> dict[str, Any]:
        """Apply critic feedback to produce a revised draft."""
        if suggest is None:
            return {"ok": False, "reason": "no model available for revision"}
        prompt = (
            "Revise this chapter draft addressing the critic's feedback.\n\n"
            f"ISSUES: {', '.join(critique.get('issues', []))}\n"
            f"SUGGESTIONS: {', '.join(critique.get('suggestions', []))}\n\n"
            f"DRAFT:\n{text[:6000]}\n\n"
            "Revised chapter:"
        )
        try:
            revised = suggest(prompt, max_tokens=2000)
            return {"ok": True, "text": revised,
                    "words": len(revised.split())}
        except Exception as exc:
            return {"ok": False, "reason": str(exc)}

    def session_log(self) -> list[dict[str, Any]]:
        return list(self._log)
