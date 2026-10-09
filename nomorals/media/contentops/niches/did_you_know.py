"""Niche: did_you_know — paradox hooks, mechanism reveals, twist endings."""

from __future__ import annotations

from .base import NichePlugin, VisualPlan, VoiceSpec
from .registry import register


class DidYouKnowNiche(NichePlugin):
    name = "did_you_know"
    thesis = "Facts that sound fake until the 20-second mechanism reveal makes them obvious."
    cadence = 2.0
    target_seconds = (25, 45)

    def script_prompt(self, topic: str) -> str:
        lo, hi = self.word_budget()
        return f"""You are the writer for a "wait, WHAT?" facts channel. Your facts win because you don't just state them — you make the viewer feel smart for understanding WHY they're true.

TOPIC: {topic}

Write a spoken-word script, {lo}–{hi} words, for a vertical short.

VOICE: playful professor — delighted by the fact, slightly smug that you know the mechanism. Build it like a magic trick: show the impossible, then reveal the method. Short punchy sentences with one "but wait" pivot.

RETENTION STRUCTURE (mandatory):
1. HOOK (first line): the fact stated as a paradox — "X sounds impossible." Make the viewer say "no way" out loud.
2. THE DOUBLE-DOWN (15%): one more impossible-sounding detail that makes it worse. Now they're committed.
3. THE MECHANISM (50%): the reveal — WHY it's true, explained so simply a kid gets it. This is the "ohhh" moment. Use an analogy.
4. THE TWIST (80%): one related fact or implication that reframes the first one — the thing they'll repeat at dinner tonight.

RULES:
- Plain spoken text only. No stage directions, no [brackets], no emojis, no "hey guys".
- Facts must be real and checkable. If you're unsure about a number, round it and say "roughly" — never state a guess as fact.
- One fact per script. Depth beats breadth.
- The analogy in the mechanism reveal is mandatory — it's what makes it stick.

Return ONLY the script text, paragraphs separated by blank lines."""

    _STYLE = (
        "vibrant 3D animated render, Pixar-adjacent stylization, bright "
        "saturated colors, dramatic scale contrasts, playful lighting, "
        "ultra-detailed, cinematic composition"
    )

    _SCENES = (
        "giant glowing question mark floating over a miniature world, dramatic lighting, wonder",
        "impossible-scale comparison: tiny human figure next to a colossal object, sense of awe",
        "x-ray cutaway view revealing the hidden mechanism inside something ordinary, glowing details",
        "mind-blown moment: lightbulb explosion of ideas, particles, vibrant energy burst",
    )

    def visual_strategy(self, script: str) -> VisualPlan:
        return self._beats_to_scenes(
            script,
            self._SCENES,
            style_lock=self._STYLE,
            motions=("zoom-in-slow", "zoom-out-slow", "pan-right-slow", "zoom-in-fast"),
            notes="Bright wonder-aesthetic. Every frame should look like a thumbnail.",
        )

    title_template = "Did You Know? {topic} 🤯"
    description_template = (
        "Did you know: {topic}\n\n"
        "The mechanism is even crazier than the fact. 🤯\n\n"
        "{script}"
    )
    hashtags = {
        "tiktok": ["#didyouknow", "#facts", "#mindblown", "#learnontiktok", "#science"],
        "youtube": ["#didyouknow", "#facts", "#shorts"],
        "instagram": ["#didyouknow", "#factsdaily", "#reels"],
        "default": ["#didyouknow", "#facts"],
    }
    voice_spec = VoiceSpec(
        preferred_voice="devon-curious",
        mood="playful",
        intensity=3,
        pace_wpm=160,
        pause_style="natural",
    )

    ypp_rationale = (
        "Original educational content: AI-researched and rewritten facts "
        "with original explanations and analogies, AI-generated 3D visuals, "
        "Devon's own voice. Not a slideshow of stolen infographics — "
        "transformative educational commentary, YPP-safe."
    )


register(DidYouKnowNiche())
