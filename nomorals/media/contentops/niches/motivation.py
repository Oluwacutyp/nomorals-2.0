"""Niche: motivation — one idea, cold-open, shorts."""

from __future__ import annotations

from .base import NichePlugin, VisualPlan, VoiceSpec
from .registry import register


class MotivationNiche(NichePlugin):
    name = "motivation"
    thesis = "One truth about discipline, delivered in 40 seconds."
    cadence = 2.0
    target_seconds = (25, 45)

    def script_prompt(self, topic: str) -> str:
        lo, hi = self.word_budget()
        return f"""You are a ghostwriter for a faceless motivation channel with 2M+ subs. Your scripts get saved and shared because every line feels like it was written about the viewer personally.

TOPIC: {topic}

Write a spoken-word script, {lo}–{hi} words, for a vertical short.

VOICE: human, always. Tone is decided by mood + audience + context, not preset. Short sentences. Specific over abstract: name the 2 AM scroll, the snoozed alarm, the "I'll start Monday".

RETENTION STRUCTURE (mandatory):
1. HOOK (first line, under 3 seconds spoken): a mirror — "You're not tired. You're bored of your own excuses." style. Make it land.
2. ESCALATION: 2–3 truths, each building on the last. No filler between them.
3. THE TURN (~70% in): one reframe — a twist that makes the pain actionable — the one thing they can do in the next 24 hours.
4. ENDING: a single line that loops back to the hook. Design it so the short loops seamlessly on replay.

RULES:
- Plain spoken text only. No stage directions, no [brackets], no emojis, no "hey guys".
- Every line must earn its place. If a line could be cut without losing punch, cut it.
- End on the closing line, not on a CTA. No "follow for more".

Return ONLY the script text, paragraphs separated by blank lines."""

    _STYLE = (
        "cinematic film still, dramatic chiaroscuro lighting, solitary figure "
        "against vast dark space, teal-and-amber grade, 35mm, shallow depth of field"
    )

    _SCENES = (
        "lone figure sitting on the edge of a bed at 3am, face lit by cold phone glow, dark room",
        "silhouette running up concrete stairs at dawn, rain, motion blur, low angle",
        "close-up of hands chalking up before a heavy lift, dust in a beam of light",
        "figure standing on a rooftop at sunrise, arms open, city below, epic scale",
    )

    def visual_strategy(self, script: str) -> VisualPlan:
        return self._beats_to_scenes(
            script,
            self._SCENES,
            style_lock=self._STYLE,
            motions=("zoom-in-slow", "pan-up-slow", "zoom-in-fast", "zoom-out-slow"),
            transitions=("cut", "cut", "whip-pan", "fade"),
            notes="Dark cinematic arc: night → struggle → sunrise. One continuous grade.",
        )

    title_template = "{topic} — Watch This When You Want to Quit"
    description_template = (
        "{topic}\n\n"
        "One truth, forty seconds. Save this for the days you want to quit.\n\n"
        "{script}"
    )
    hashtags = {
        "tiktok": ["#motivation", "#discipline", "#mindset", "#selfimprovement", "#gymtok"],
        "youtube": ["#motivation", "#discipline", "#selfimprovement", "#shorts"],
        "instagram": ["#motivation", "#discipline", "#mindsetmatters", "#reels"],
        "default": ["#motivation", "#discipline"],
    }
    voice_spec = VoiceSpec(
        preferred_voice="devon-deep",
        mood="neutral",
        intensity=3,
        pace_wpm=140,
        pause_style="natural",
    )

    ypp_rationale = (
        "Fully original: AI-written script, AI-generated cinematic visuals, "
        "Devon's own cloned voice. No reused footage, no text-to-speech "
        "slideshow of someone else's content — original commentary and "
        "transformative visuals throughout, YPP-safe."
    )


register(MotivationNiche())
