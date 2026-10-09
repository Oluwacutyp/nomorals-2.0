"""Niche: sports_edits — cinematic AI athlete scenes with beat-edit grammar.

HARD RULE: every visual is AI-generated and depicts FICTIONAL athletes.
No real players, no real teams, no broadcast footage, no logos. The edit
sells on cinematography and cutting rhythm, not on someone else's highlights.
"""

from __future__ import annotations

from .base import NichePlugin, VisualPlan, VoiceSpec
from .registry import register


class SportsEditsNiche(NichePlugin):
    name = "sports_edits"
    thesis = "Underdog sports cinema with fictional athletes — beat drops, slow-mo, and a 40-second hero arc."
    cadence = 1.0
    target_seconds = (25, 45)

    def script_prompt(self, topic: str) -> str:
        lo, hi = self.word_budget()
        return f"""You are the writer for a sports-edit channel. Your edits hit because the NARRATION is a 40-second sports movie: doubt, grind, glory — timed so the beat drop lands on the payoff.

TOPIC: {topic}

Write a spoken-word script, {lo}–{hi} words, for a vertical sports edit.

VOICE: cinematic sports narrator — quiet at first, then swelling. Think documentary trailer, not commentator. Short declarative lines that leave room for the music. The voice and the beat are dance partners: write lines that a beat drop can land on.

RETENTION STRUCTURE (mandatory):
1. HOOK (first line, under 3s): the doubt — "They said he was finished." Establish the underdog instantly.
2. THE GRIND (30%): training-montage narration — 4am runs, empty gyms, failure. Each line shorter than the last, building pressure.
3. THE BEAT-DROP LINE (~60%): ONE sentence the whole edit pivots on — "And then the lights came on." This line must work as a standalone punch. Mark it by making it the shortest paragraph.
4. THE GLORY (last 30%): the payoff in rising intensity — crowd, lights, the moment. End on a line that feels like a trophy lift.

RULES:
- Plain spoken text only. No stage directions, no [brackets], no emojis, no "hey guys".
- FICTIONAL athletes only. Never name a real player, team, or league. "He" is an archetype, not a person.
- Leave breathing room: short lines, natural pause points. The edit cuts ON the words.
- One arc per script: doubt → grind → glory. No subplots.

Return ONLY the script text, paragraphs separated by blank lines. Make the beat-drop line its own paragraph."""

    _STYLE = (
        "cinematic sports photography, fictional athlete in unbranded kit, "
        "dramatic stadium lighting, sweat and motion frozen mid-action, "
        "high contrast, shallow depth of field, epic scale, film grain"
    )

    _SCENES = (
        "fictional boxer sitting alone in a dark locker room, head down, single overhead light, unbranded gear",
        "fictional sprinter training at dawn on an empty track, motion blur, low sun, determination",
        "fictional basketball player mid-dunk frozen in dramatic slow motion, arena lights blazing, no logos",
        "fictional athlete raising arms in victory under falling confetti, roaring crowd bokeh, epic wide shot",
    )

    _MOTIONS = ("zoom-in-slow", "pan-right-fast", "zoom-in-fast", "zoom-out-slow")
    _TRANSITIONS = ("cut", "whip-pan", "cut", "fade")

    def visual_strategy(self, script: str) -> VisualPlan:
        plan = self._beats_to_scenes(
            script,
            self._SCENES,
            style_lock=self._STYLE,
            motions=self._MOTIONS,
            transitions=self._TRANSITIONS,
            notes=(
                "Beat-edit grammar: cuts on the narration's stressed words, "
                "whip-pan into the training montage, slow push on the glory shot. "
                "All athletes fictional, all kits unbranded."
            ),
        )
        return plan

    title_template = "{topic} — The Comeback Edit 🏆"
    description_template = (
        "{topic}\n\n"
        "Doubt. Grind. Glory. Sound on. 🔊\n\n"
        "{script}"
    )
    hashtags = {
        "tiktok": ["#sportsedit", "#motivation", "#sports", "#comeback", "#edit"],
        "youtube": ["#sportsedit", "#sports", "#shorts"],
        "instagram": ["#sportsedit", "#athlete", "#reels"],
        "default": ["#sportsedit", "#sports"],
    }
    voice_spec = VoiceSpec(
        preferred_voice="devon-epic",
        mood="triumphant",
        intensity=4,
        pace_wpm=135,
        pause_style="dramatic",
    )

    ypp_rationale = (
        "Original visuals only: every frame is AI-generated and depicts "
        "fictional athletes in unbranded kit — no broadcast footage, no "
        "real likenesses, no logos, no reused highlights. Original narration "
        "arc written for the edit. Fully transformative and original — YPP-safe."
    )


register(SportsEditsNiche())
