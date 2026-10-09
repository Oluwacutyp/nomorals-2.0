"""Niche: anime_edits — AI anime-style visuals with on-beat cut grammar.

Original characters and scenes only — anime-STYLE, not anime footage.
The edit grammar is the product: cuts land on beats, speed ramps on
drops, Japanese-text overlays for flavor.
"""

from __future__ import annotations

from .base import NichePlugin, VisualPlan, VoiceSpec
from .registry import register


class AnimeEditsNiche(NichePlugin):
    name = "anime_edits"
    thesis = "Original anime-style AMVs: a 40-second hero moment cut exactly on the beat."
    cadence = 1.0
    target_seconds = (25, 45)

    def script_prompt(self, topic: str) -> str:
        lo, hi = self.word_budget()
        return f"""You are the writer for an anime-edit channel. Your edits slap because the NARRATION is a shonen episode compressed to 40 seconds: the vow, the fall, the power-up — and every line is timed to be CUT ON.

TOPIC: {topic}

Write a spoken-word script, {lo}–{hi} words, for a vertical anime edit.

VOICE: dramatic anime narrator — mythic, a little theatrical. Short lines with hard consonants that an editor can slice between. Think episode-preview voiceover: "Next time — everything burns." Leave space for the music to breathe.

RETENTION STRUCTURE (mandatory):
1. HOOK (first line): the vow or the threat — "He promised he'd never lose again." One line, total commitment.
2. THE FALL (30%): the defeat montage in words — each line a hammer blow, getting shorter, until the lowest point.
3. THE POWER-UP LINE (~60%): ONE line the edit explodes on — "Then the sky split open." Shortest paragraph. This is the drop.
4. THE CLIMAX (last 30%): the hero moment — rising, unstoppable, mythic. End on a line that feels like a final attack name.

RULES:
- Plain spoken text only. No stage directions, no [brackets], no emojis, no "hey guys".
- ORIGINAL characters only. Never name a real anime, character, or studio. The hero is an archetype: the vow, the blade, the storm.
- Write for the cut: short lines, pause points, stressed final words. The editor cuts ON your line breaks.
- One arc per script: vow → fall → power-up → climax.

Return ONLY the script text, paragraphs separated by blank lines. Make the power-up line its own paragraph."""

    _STYLE = (
        "dynamic anime key visual, original character design, dramatic "
        "cel shading, speed lines, glowing energy effects, high-contrast "
        "color palette, cinematic composition, studio-quality keyframe art"
    )

    _SCENES = (
        "original anime swordsman standing in rain at night, glowing eyes, dramatic low angle, city lights bokeh",
        "original anime hero kneeling defeated in rubble, cracked mask, embers rising, desaturated palette",
        "original anime warrior mid-transformation, energy aura exploding outward, sky splitting with light",
        "original anime hero unleashing a massive energy slash, dynamic pose, full-page-spread composition",
    )

    _MOTIONS = ("zoom-in-slow", "zoom-in-slow", "zoom-in-fast", "pan-up-fast")
    _TRANSITIONS = ("cut", "fade", "whip-pan", "cut")

    def visual_strategy(self, script: str) -> VisualPlan:
        return self._beats_to_scenes(
            script,
            self._SCENES,
            style_lock=self._STYLE,
            motions=self._MOTIONS,
            transitions=self._TRANSITIONS,
            notes=(
                "AMV grammar: hold the fall shot long, hard-cut on the power-up "
                "line, speed-ramp the climax. Japanese-text overlay flavor on the "
                "hook and power-up frames. All characters original."
            ),
        )

    title_template = "{topic} — Anime Edit ⚔️"
    description_template = (
        "{topic}\n\n"
        "The vow. The fall. The power-up. Sound on. 🔊\n\n"
        "{script}"
    )
    hashtags = {
        "tiktok": ["#animeedit", "#amv", "#anime", "#edit", "#animetiktok"],
        "youtube": ["#animeedit", "#amv", "#shorts"],
        "instagram": ["#animeedit", "#amv", "#reels"],
        "default": ["#animeedit", "#amv"],
    }
    voice_spec = VoiceSpec(
        preferred_voice="devon-epic",
        mood="dramatic",
        intensity=4,
        pace_wpm=140,
        pause_style="dramatic",
    )

    ypp_rationale = (
        "Anime-STYLE, not anime footage: every visual is AI-generated with "
        "original character designs — no frames from real anime, no studio "
        "assets, no copyrighted characters. Original narration written for "
        "the edit. Fully original and transformative — YPP-safe."
    )


register(AnimeEditsNiche())
