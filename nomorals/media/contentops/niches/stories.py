"""Niche: stories — one complete original story in 40 seconds."""

from __future__ import annotations

from .base import NichePlugin, VisualPlan, VoiceSpec
from .registry import register


class StoriesNiche(NichePlugin):
    name = "stories"
    default_style = "minimal"
    thesis = "A complete original story in 40 seconds — a hook, a turn, and a landing that sticks."
    cadence = 2.0
    target_seconds = (25, 50)

    def script_prompt(self, topic: str) -> str:
        lo, hi = self.word_budget()
        return f"""You are the storyteller for an original-fiction channel. Your stories win because they are complete — a beginning, a turn, and an ending — inside one short, told like you're leaning in across a fire.

TOPIC: {topic}

Write a spoken-word script, {lo}–{hi} words, for a vertical short.

VOICE: warm storyteller — unhurried, a little mischievous, every sentence earning the next. Short sentences for tension, one long one for the landing. The listener should feel told-to, not lectured.

RETENTION STRUCTURE (mandatory):
1. HOOK (first line): the story's most gripping image as one sentence — the moment everything changes. Make them NEED to know how it started.
2. THE SETUP (fast, 20%): who, where, what they wanted — three sentences max. No backstory that doesn't pay off.
3. THE TURN (50%): the twist — the thing nobody in the story saw coming, told in two beats. The second beat must be worse (or stranger) than the first.
4. THE LANDING (last line): the ending that sticks — one sentence the viewer repeats to someone else later. Resolve the story; the feeling is the cliffhanger, not a missing ending.

RULES:
- Plain spoken text only. No stage directions, no [brackets], no emojis, no "hey guys".
- Original fiction only: invent every name and place. Never retell a real person's story, never claim it happened.
- One story per script. A part number in the topic ("part 2") means this episode is the next chapter — carry the same characters forward.
- Keep it PG-13: imply the dark parts, don't narrate explicit detail.

Return ONLY the script text, paragraphs separated by blank lines."""

    _STYLE = (
        "cinematic storybook illustration, warm painterly light, expressive "
        "original characters, rich color, film-still composition, gentle "
        "depth of field"
    )

    _SCENES = (
        "the story's opening image: a lone character in a vast, beautiful place, warm light, wonder",
        "the character up close, determined expression, the world of the story around them, rich detail",
        "the twist moment: everything tilts, dramatic light shift, the character realising something huge",
        "the landing image: quiet aftermath, golden light, the character changed, cinematic still",
    )

    def visual_strategy(self, script: str) -> VisualPlan:
        return self._beats_to_scenes(
            script,
            self._SCENES,
            style_lock=self._STYLE,
            motions=("zoom-in-slow", "pan-right-slow", "zoom-in-fast", "zoom-out-slow"),
            notes="Storybook cinematic look — one continuous world across the four beats.",
        )

    title_template = "{topic} — A Story in 40 Seconds"
    description_template = (
        "{topic}\n\n"
        "An original story, told in forty seconds. 📖\n\n"
        "{script}"
    )
    hashtags = {
        "tiktok": ["#storytime", "#stories", "#story", "#shorts", "#fiction"],
        "youtube": ["#storytime", "#stories", "#shorts"],
        "instagram": ["#storytime", "#stories", "#reels"],
        "default": ["#storytime", "#stories"],
    }
    voice_spec = VoiceSpec(
        preferred_voice="devon-storyteller",
        mood="warm",
        intensity=3,
        pace_wpm=150,
        pause_style="natural",
    )

    ypp_rationale = (
        "Original fiction: stories are invented, retold in Devon's own "
        "voice, with original AI-generated storybook visuals. No real "
        "person's story, no copyrighted text, no stolen footage — "
        "transformative original content, YPP-safe."
    )


register(StoriesNiche())
