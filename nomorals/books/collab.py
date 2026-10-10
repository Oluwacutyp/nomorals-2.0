"""BookForge collaborative writing — owner + brain + characters as co-authors.

The spec is the floor. This goes beyond: real collaborative fiction where
the owner directs, the brain orchestrates, and character agents write scenes
from their own POV. Plus a writer-critic loop that reviews every chapter
before it ships.

Mined gold:
- Writer-Critic (botlab/swarms): writer drafts, critic reviews, iterate
- AI Writing Studio: specialized agents (continuity, character, timeline)
- HNPX: human guides at each level, LLM as responsive collaborator
- Novelcrafter Smart Highlighting: overused words, crutch phrases,
  dialogue-tag audit, AI-pattern flagging
- Sudowrite feedback personas: the encouraging buddy, the brutal critic —
  distinct voices, distinct standards
"""

from __future__ import annotations

import re
from collections import Counter
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


# ── style analysis (Novelcrafter Smart Highlighting, pure Python) ────────────


class StyleAnalyzer:
    """Sentence-level prose diagnostics, no model needed.

    Mined from Novelcrafter's Smart Highlighting: overused words, crutch
    phrases, repetitive metaphors, dialogue-tag audit, AI-pattern
    flagging.  Plus the classic filter-word scan (show-don't-tell).
    """

    #: filter words — telling the reader about perception instead of
    #: rendering the thing perceived
    FILTER_WORDS = (
        "saw", "heard", "felt", "noticed", "realized", "seemed",
        "appeared", "watched", "observed", "knew", "thought", "wondered",
        "decided", "remembered",
    )
    #: crutch phrases that signal first-draft prose
    CRUTCH = (
        "a moment", "for a moment", "suddenly", "all of a sudden",
        "in that moment", "little did", "little did they know",
        "as if", "as though", "kind of", "sort of", "very",
        "really", "just then", "at that moment", "needless to say",
        "it goes without saying", "deep down", "in the end",
    )
    #: AI-pattern markers (overly polished, low-information phrasing)
    AI_PATTERNS = (
        "delve", "tapestry", "intricate", "vibrant", "testament to",
        "in today's", "furthermore", "moreover", "it's important to note",
        "as an ai",
    )
    #: weak dialogue tags worth auditing per character voice
    DIALOGUE_TAGS = (
        "said", "asked", "replied", "shouted", "whispered", "muttered",
        "growled", "laughed", "sighed", "exclaimed", "demanded",
        "answered", "snarled", "hissed", "yelled", "murmured",
        "declared", "warned", "stammered", "scoffed",
    )

    def analyze(self, text: str) -> dict[str, Any]:
        """Full style scan → findings, tips, score_penalty (0..~0.5)."""
        text = text or ""
        low = text.lower()
        words = re.findall(r"[a-z']+", low)
        findings: list[str] = []
        tips: list[str] = []
        penalty = 0.0

        # 1. filter words
        fhits = {w: len(re.findall(r"\b" + w + r"\b", low))
                 for w in self.FILTER_WORDS}
        fhits = {w: c for w, c in fhits.items() if c}
        if fhits:
            top = sorted(fhits.items(), key=lambda x: -x[1])[:3]
            density = sum(fhits.values()) / max(len(words), 1)
            if density > 0.015:
                findings.append(
                    "filter-word heavy (" + ", ".join(f"{w}×{c}" for w, c in top)
                    + ") — render the perception, not the perceiving")
                tips.append("cut filter words; let the image do the work")
                penalty += 0.06

        # 2. crutch phrases
        chits = [(p, low.count(p)) for p in self.CRUTCH]
        chits = [(p, c) for p, c in chits if c >= 2]
        if chits:
            top = sorted(chits, key=lambda x: -x[1])[:3]
            findings.append("crutch phrases: " +
                            ", ".join(f"“{p}”×{c}" for p, c in top))
            tips.append("rewrite crutch phrases with specific images")
            penalty += 0.05

        # 3. repeated words (excluding stopwords)
        stop = set("the a an and or of to in on at for with is was were be "
                   "been are had has have it its they them he she his her we "
                   "you your i me my as by from that this these those not no "
                   "but so if then than too".split())
        counts = Counter(w for w in words if w not in stop and len(w) > 3)
        repeated = [(w, c) for w, c in counts.most_common(6)
                    if c >= max(6, len(words) // 120)]
        if repeated:
            findings.append("overused words: " +
                            ", ".join(f"{w}×{c}" for w, c in repeated[:4]))
            tips.append("vary diction for the flagged repeats")
            penalty += 0.04

        # 4. sentence-length variance (monotone rhythm)
        sents = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(sents) >= 8:
            lens = [len(s.split()) for s in sents]
            avg = sum(lens) / len(lens)
            var = sum((x - avg) ** 2 for x in lens) / len(lens)
            if avg < 8:
                findings.append("staccato rhythm — very short sentences "
                                "throughout; vary length for flow")
                penalty += 0.03
            elif var < 25:
                findings.append("monotone sentence length — the rhythm "
                                "never breathes")
                tips.append("mix short punches with long rolling sentences")
                penalty += 0.03

        # 5. dialogue-tag audit
        tags = Counter()
        for m in re.finditer(
                r"[\"”]\s*,?\s*([a-z]+)\b", text):
            tag = m.group(1).lower()
            if tag in self.DIALOGUE_TAGS:
                tags[tag] += 1
        if tags:
            said_ratio = tags.get("said", 0) / sum(tags.values())
            exotic = [t for t in tags if t not in ("said", "asked")]
            if exotic and said_ratio < 0.4:
                findings.append(
                    "dialogue tags showing off (" +
                    ", ".join(f"{t}×{c}" for t, c in tags.most_common(3)
                              if t not in ("said", "asked"))
                    + ") — 'said' is invisible, the rest aren't")
                tips.append("default to 'said'; earn the exotic tags")
                penalty += 0.03

        # 6. repeated sentence starts
        starts = Counter(s.split()[0].lower().strip("\"“'(") for s in sents
                         if s.split())
        rep_starts = [(w, c) for w, c in starts.most_common(4)
                      if c >= 4 and w not in stop]
        if rep_starts:
            findings.append("repeated sentence starts: " +
                            ", ".join(f"“{w}”×{c}" for w, c in rep_starts))
            penalty += 0.03

        # 7. AI-pattern markers
        ai = [p for p in self.AI_PATTERNS if p in low]
        if ai:
            findings.append("AI-tell phrasing: " + ", ".join(f"“{p}”" for p in ai))
            tips.append("replace AI-tell phrasing with the author's own voice")
            penalty += 0.06

        # 8. telling emotion labels
        telling = sum(low.count(w) for w in
                      ("felt sad", "was happy", "was angry", "felt scared",
                       "was sad", "felt happy", "was afraid", "felt angry"))
        if telling >= 2:
            findings.append("telling emotions instead of showing "
                            f"({telling} labels)")
            tips.append("replace emotion labels with physical/sensory beats")
            penalty += 0.04

        return {"words": len(words), "sentences": len(sents),
                "findings": findings, "tips": tips,
                "score_penalty": round(min(0.5, penalty), 3),
                "dialogue_tags": dict(tags.most_common(8)),
                "top_words": [w for w, _ in counts.most_common(10)]}


# ── character sheets (Truby/Weiland psychology + voice) ──────────────────────


@dataclass
class CharacterSheet:
    """A character as a co-author: psychology + voice profile.

    The five-point core (want/need/wound/lie/ghost) drives the arc; the
    voice profile (tics, vocab band, rhythm, taboo topics) keeps every
    line they speak sounding like THEM.
    """
    name: str
    role: str = "supporting"
    want: str = ""
    need: str = ""
    wound: str = ""
    lie: str = ""
    ghost: str = ""
    archetype: str = ""
    traits: list[str] = field(default_factory=list)
    # voice profile
    speech_tics: list[str] = field(default_factory=list)
    vocab_band: str = ""        # e.g. "formal, latin-heavy" / "street, clipped"
    rhythm: str = ""            # e.g. "short bursts" / "long rolling periods"
    taboo_topics: list[str] = field(default_factory=list)
    description: str = ""

    def persona_block(self) -> str:
        """Prompt-ready persona for scene writing."""
        lines = [f"CHARACTER: {self.name} ({self.role})"]
        if self.description:
            lines.append(self.description[:300])
        if self.want:
            lines.append(f"Want: {self.want}. Need: {self.need or '—'}.")
        if self.wound:
            lines.append(f"Wound: {self.wound}. Believes the lie: "
                         f"{self.lie or '—'}.")
        if self.ghost:
            lines.append(f"Ghost (resurfaces under pressure): {self.ghost}.")
        voice_bits = []
        if self.speech_tics:
            voice_bits.append("tics: " + ", ".join(self.speech_tics))
        if self.vocab_band:
            voice_bits.append(f"vocabulary: {self.vocab_band}")
        if self.rhythm:
            voice_bits.append(f"rhythm: {self.rhythm}")
        if self.taboo_topics:
            voice_bits.append("never mentions: " +
                              ", ".join(self.taboo_topics))
        if voice_bits:
            lines.append("VOICE — " + "; ".join(voice_bits))
        return "\n".join(lines)

    def voice_check(self, dialogue: str) -> dict[str, Any]:
        """Score whether a line sounds like this character (0..1)."""
        line = (dialogue or "").strip()
        if not line:
            return {"score": 0.0, "notes": ["empty line"]}
        score = 0.5
        notes: list[str] = []
        low = line.lower()
        for tic in self.speech_tics:
            if tic.lower() in low:
                score += 0.15
                notes.append(f"tic present: {tic!r}")
                break
        else:
            if self.speech_tics:
                notes.append("no signature tic — could be anyone")
        for taboo in self.taboo_topics:
            if taboo.lower() in low:
                score -= 0.3
                notes.append(f"taboo topic breached: {taboo!r}")
        words = line.split()
        if self.rhythm.startswith("short") and len(words) > 40:
            score -= 0.1
            notes.append("too long for a short-burst voice")
        if self.rhythm.startswith("long") and len(words) < 12:
            score -= 0.1
            notes.append("too clipped for a rolling voice")
        return {"score": round(max(0.0, min(1.0, score)), 2), "notes": notes}

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CharacterSheet":
        return cls(
            name=str(d.get("name", "")),
            role=str(d.get("role", "supporting")),
            want=str(d.get("want", "")), need=str(d.get("need", "")),
            wound=str(d.get("wound", "")), lie=str(d.get("lie", "")),
            ghost=str(d.get("ghost", "")),
            archetype=str(d.get("archetype", "")),
            traits=[str(t) for t in d.get("traits", [])],
            speech_tics=[str(t) for t in d.get("speech_tics", [])],
            vocab_band=str(d.get("vocab_band", "")),
            rhythm=str(d.get("rhythm", "")),
            taboo_topics=[str(t) for t in d.get("taboo_topics", [])],
            description=str(d.get("description", "")),
        )


class CriticAgent:
    """Reviews chapter drafts like a brutal-but-fair editor.

    Checks: pacing, character voice consistency, plot mechanics,
    show-don't-tell, dialogue quality, continuity with story bible.

    Personas (Sudowrite feedback-plugin style): ``editor`` (structure),
    ``line`` (prose), ``beta`` (reader feel), ``brutal`` (no mercy).
    Every review opens with a StyleAnalyzer pass — pure Python, no model
    needed — then the persona lens, then bible continuity.
    """

    PERSONAS = ("editor", "line", "beta", "brutal")

    _PERSONA_BRIEFS = {
        "editor": ("You are a developmental editor. Judge structure: does "
                   "every scene earn its place? Is the arc moving? Flag "
                   "sagging middles and unearned turns."),
        "line": ("You are a line editor. Judge the sentence level: rhythm, "
                 "word choice, clichés, filter words. Be specific — quote "
                 "the offending lines."),
        "beta": ("You are a beta reader, not a writer. Report how the "
                 "chapter FELT: where you skimmed, where you leaned in, "
                 "what confused or bored you. No craft jargon."),
        "brutal": ("You are the harshest critic alive. No praise unless "
                   "genuinely earned. Tear apart every weak choice and say "
                   "exactly what would make it undeniable."),
    }

    def __init__(self, context: Any = None) -> None:
        self.context = context
        self.style = StyleAnalyzer()

    def review(self, text: str, brief: dict[str, Any],
               bible: dict[str, Any] | None = None,
               suggest: Callable | None = None,
               persona: str = "editor") -> Critique:
        """Review a draft chapter. Returns structured critique."""
        persona = persona if persona in self.PERSONAS else "editor"
        style_report = self.style.analyze(text)
        continuity = self._continuity_check(text, bible) if bible else []
        if suggest is None:
            return self._heuristic_review(text, brief, bible, persona,
                                          style_report, continuity)

        prompt = self._build_prompt(text, brief, bible, persona,
                                    style_report, continuity)
        try:
            raw = suggest(prompt, max_tokens=800)
            return self._parse_critique(raw)
        except Exception:
            return self._heuristic_review(text, brief, bible, persona,
                                          style_report, continuity)

    def persona_brief(self, persona: str) -> str:
        return self._PERSONA_BRIEFS.get(persona, self._PERSONA_BRIEFS["editor"])

    def _continuity_check(self, text: str,
                          bible: dict[str, Any]) -> list[str]:
        """Bible continuity: named characters behaving off-sheet, dead
        characters walking, rule violations."""
        flags: list[str] = []
        low = text.lower()
        chars = bible.get("characters", []) if isinstance(bible, dict) else []
        for c in chars:
            name = (c.get("name") if isinstance(c, dict)
                    else getattr(c, "name", "")) or ""
            if not name or len(name) < 3:
                continue
            desc = (c.get("description") if isinstance(c, dict)
                    else getattr(c, "description", "")) or ""
            if "dead" in desc.lower() or "died" in desc.lower():
                if re.search(r"\b" + re.escape(name) + r"\b", text,
                             re.IGNORECASE):
                    flags.append(f"{name} is marked dead in the bible but "
                                 f"appears alive in this chapter")
        return flags

    def _build_prompt(self, text: str, brief: dict[str, Any],
                      bible: dict[str, Any] | None,
                      persona: str = "editor",
                      style_report: dict[str, Any] | None = None,
                      continuity: list[str] | None = None) -> str:
        bible_txt = ""
        if bible:
            cast = ", ".join(
                c.get("name", "?") for c in bible.get("cast", [])[:8])
            bible_txt = f"\nCAST: {cast}\n"
        style_txt = ""
        if style_report:
            style_txt = ("\nSTYLE SCAN (already measured — don't re-litigate, "
                         "use it):\n" +
                         "\n".join(f"- {k}: {v}"
                                   for k, v in style_report.items()
                                   if k != "score_penalty") + "\n")
        cont_txt = ""
        if continuity:
            cont_txt = "\nCONTINUITY FLAGS:\n- " + "\n- ".join(continuity) + "\n"
        return (
            f"{self.persona_brief(persona)}\n"
            f"BRIEF: {brief.get('summary', '')}\n{bible_txt}"
            f"{style_txt}{cont_txt}\n"
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
                          bible: dict[str, Any] | None,
                          persona: str = "editor",
                          style_report: dict[str, Any] | None = None,
                          continuity: list[str] | None = None) -> Critique:
        """Fallback when no model: structural checks, never fake praise.

        The style scan + continuity flags feed the same issue list the
        model path would produce, so the floor is honest.
        """
        words = len(text.split())
        issues, strengths, suggestions = [], [], []
        style_report = style_report or {}

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

        # style scan findings become issues/suggestions
        for finding in style_report.get("findings", []):
            issues.append(finding)
        for tip in style_report.get("tips", []):
            suggestions.append(tip)
        for flag in continuity or []:
            issues.append(f"continuity: {flag}")

        # persona lens: what each persona weighs
        penalty = style_report.get("score_penalty", 0.0)
        if persona == "brutal":
            penalty += 0.05 * len(issues)
            if not issues:
                issues.append("brutal pass: nothing earned praise yet — "
                              "raise the stakes or cut the chapter")
        elif persona == "beta" and dialogue_ratio < 0.02 and words > 800:
            issues.append("beta note: I skimmed the long expository stretches")

        if not issues:
            strengths.append("structurally sound draft")

        score = 0.85 if not issues else max(0.3, 0.75 - len(issues) * 0.08 - penalty)
        verdict = "ship" if score >= 0.8 and not issues else (
            "rewrite" if score < 0.45 else "revise")
        return Critique(score=round(score, 2), strengths=strengths,
                        issues=issues, suggestions=suggestions,
                        verdict=verdict)


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
        self._sheets: dict[str, CharacterSheet] = {}
        self._log: list[dict[str, Any]] = []

    def cast_character(self, character: Any) -> None:
        """Add a character agent as a co-author.

        Accepts a CharacterSheet (preferred — full psychology + voice) or
        any object with a ``name`` attribute.
        """
        name = getattr(character, "name", None) or str(character)
        self._characters[name] = character
        if isinstance(character, CharacterSheet):
            self._sheets[name] = character
        elif isinstance(character, dict):
            try:
                self._sheets[name] = CharacterSheet.from_dict(character)
            except Exception:  # noqa: BLE001
                pass

    def character_sheet(self, name: str) -> CharacterSheet | None:
        return self._sheets.get(name)

    def write_scene(self, character_name: str, prompt: str,
                    suggest: Callable | None = None) -> dict[str, Any]:
        """A character writes a scene from their POV — voice-locked."""
        char = self._characters.get(character_name)
        if char is None:
            return {"ok": False, "reason": f"character '{character_name}' not in session"}
        sheet = self._sheets.get(character_name)

        if suggest is not None:
            try:
                if sheet is not None:
                    persona = sheet.persona_block()
                elif hasattr(char, "persona_block"):
                    persona = char.persona_block()
                else:
                    persona = str(char)
                full_prompt = (
                    f"{persona}\n\n"
                    f"You are writing a scene from your own point of view. "
                    f"Stay in character — your voice, your biases, your blind spots.\n"
                    + (f"Your wound ({sheet.wound}) and the lie you believe "
                       f"({sheet.lie}) color everything you do — show them, "
                       f"never name them.\n" if sheet and sheet.wound else "")
                    + f"\nSCENE PROMPT: {prompt}\n\n"
                    f"Write the scene (300-800 words):"
                )
                text = suggest(full_prompt, max_tokens=1200)
            except Exception as exc:
                return {"ok": False, "reason": f"generation failed: {exc}"}
        else:
            text = char._fallback_line(prompt) if hasattr(char, "_fallback_line") else ""

        # voice-check the result against the sheet when we have one
        voice_report = None
        if sheet is not None and text:
            quoted = re.findall(r"[\"“]([^\"”]{8,200})[\"”]", text)
            if quoted:
                voice_report = {
                    "lines_checked": len(quoted),
                    "avg_score": round(sum(
                        sheet.voice_check(q)["score"] for q in quoted
                    ) / len(quoted), 2),
                }

        entry = {
            "character": character_name,
            "prompt": prompt,
            "text": text,
            "words": len(text.split()),
            "voice_report": voice_report,
        }
        self._log.append(entry)

        # Character remembers writing this
        if hasattr(char, "remember"):
            char.remember(f"I wrote a scene: {prompt[:100]}", salience=0.7)

        return {"ok": True, **entry}

    def critique(self, text: str, brief: dict[str, Any],
                 bible: dict[str, Any] | None = None,
                 suggest: Callable | None = None,
                 persona: str = "editor") -> dict[str, Any]:
        """Run the critic on a draft (persona: editor|line|beta|brutal)."""
        c = self.critic.review(text, brief, bible, suggest, persona=persona)
        return {"ok": True, "persona": persona, "critique": c.to_dict()}

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
