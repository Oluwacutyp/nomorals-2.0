"""Niche: horror_stories — second-person dread, delayed payoff."""

from __future__ import annotations

from .base import NichePlugin, VisualPlan, VoiceSpec
from .registry import register


class HorrorStoriesNiche(NichePlugin):
    name = "horror_stories"
    thesis = "You are the protagonist. The dread builds for 40 seconds, then the last line ruins your night."
    cadence = 1.0
    target_seconds = (30, 55)

    def script_prompt(self, topic: str) -> str:
        lo, hi = self.word_budget()
        return f"""You are a horror writer for a faceless channel whose stories get millions of views because listeners genuinely can't sleep after. You never use jump scares — you use inevitability.

TOPIC: {topic}

Write a spoken-word horror script, {lo}–{hi} words, for a vertical short.

VOICE: second person, present tense. "You" are living it. Sensory and specific — the exact sound the floorboard makes, the exact wrongness of the silence. Slow dread, not gore. The scariest line is always the quietest one.

RETENTION STRUCTURE (mandatory):
1. HOOK (first line): drop the listener INTO the scene mid-moment — "The knocking stopped the second you answered." No setup, no "this happened to my friend". It is happening NOW.
2. THE RULE OF THREE: three escalating wrong details. Each one should make the listener's stomach drop a little further. Withhold the explanation every time.
3. THE FALSE CALM (~75%): one moment that almost feels safe — then pull it away.
4. THE FINAL LINE: a single quiet sentence that recontextualizes everything before it. The listener should need to replay to catch what they missed. No screaming, no twist-for-twist's-sake.

RULES:
- Plain spoken text only. No stage directions, no [brackets], no emojis, no "hey guys".
- Psychological dread over gore. Nothing graphic — the horror is in what is implied, never shown.
- No real tragedies, no real victims, no real locations. Pure fiction.
- Short sentences. Let silence do the work — the voice will pause where you put periods.

Return ONLY the script text, paragraphs separated by blank lines."""

    _STYLE = (
        "dark atmospheric horror film still, found-footage grain, deep shadows, "
        "single cold light source, desaturated blue-black palette, volumetric fog, "
        "unsettling negative space, 35mm"
    )

    _SCENES = (
        "empty hallway at 3am lit by a single flickering bulb, long shadows, door slightly ajar at the far end",
        "close-up of a hand on a doorknob, knuckles white, darkness beyond the cracked door",
        "bedroom mirror reflecting an empty room, but the bed in the reflection is not empty",
        "wide shot of a house at night, one window lit, a silhouette standing in it that was not there before",
    )

    def visual_strategy(self, script: str) -> VisualPlan:
        return self._beats_to_scenes(
            script,
            self._SCENES,
            style_lock=self._STYLE,
            motions=("zoom-in-slow", "zoom-in-slow", "zoom-in-slow", "zoom-out-slow"),
            transitions=("cut", "fade", "cut", "cut"),
            notes="Never show the monster. Slow push-ins only — the camera is the dread.",
        )

    title_template = "{topic} — A Horror Story in 40 Seconds"
    description_template = (
        "{topic}\n\n"
        "Headphones on. Lights off. You were warned.\n\n"
        "{script}"
    )
    hashtags = {
        "tiktok": ["#horror", "#horrorstories", "#scarystories", "#creepypasta", "#spooky"],
        "youtube": ["#horror", "#scarystories", "#shorts"],
        "instagram": ["#horrorstories", "#creepy", "#reels"],
        "default": ["#horror", "#scarystories"],
    }
    voice_spec = VoiceSpec(
        preferred_voice="devon-noir",
        mood="ominous",
        intensity=4,
        pace_wpm=125,
        pause_style="dramatic",
    )

    ypp_rationale = (
        "Wholly original fiction: AI-written story, AI-generated horror "
        "stills, Devon's own voice performance. Nothing reused, no real "
        "events depicted. Original narrative content — YPP-safe."
    )


register(HorrorStoriesNiche())
