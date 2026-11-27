"""Niche: cooking — one dish in 40 seconds, three steps, one payoff plate."""

from __future__ import annotations

from .base import NichePlugin, VisualPlan, VoiceSpec
from .registry import register


class CookingNiche(NichePlugin):
    name = "cooking"
    default_style = "vlog"
    thesis = "One dish in 40 seconds — three steps, one finished plate as the payoff."
    cadence = 2.0
    target_seconds = (25, 50)

    def script_prompt(self, topic: str) -> str:
        lo, hi = self.word_budget()
        return f"""You are the cook for a fast-recipe channel. Your recipes win because anyone watching can actually make the dish — three steps, spoken plainly, with the finished plate as the reward at the end.

TOPIC: {topic}

Write a spoken-word script, {lo}–{hi} words, for a vertical short.

VOICE: warm and hungry — a friend in your kitchen who has made this a hundred times. Short punchy sentences, one "trust me" moment. The listener should be able to cook along from the words alone.

RETENTION STRUCTURE (mandatory):
1. HOOK (first line): the finished dish + the one-line promise — "crispy outside, soft inside, one pan." Make them NEED the recipe.
2. THE STEPS (60%): exactly three steps, in order, each one sentence of action plus one sentence of the detail that makes it work (the heat, the timing, the pinch). No step may need equipment a normal kitchen doesn't have.
3. THE PAYOFF (last lines): the reveal — the finished plate, the first bite, the one thing to serve it with. End on the taste, not the technique.

RULES:
- Plain spoken text only. No stage directions, no [brackets], no emojis, no "hey guys".
- Quantities in plain words ("a handful", "two spoons") — never a string of numbers nobody can follow by ear.
- Keep every step cookable: real heat, real timing, nothing impossible. If a step needs a rest or a chill, say so in the step.
- One dish per script. An angle in the topic ("the secret technique", "the twist") means this episode is that angle — lead with it.

Return ONLY the script text, paragraphs separated by blank lines."""

    _STYLE = (
        "warm overhead food photography, steam rising, rich natural color, "
        "cozy kitchen light, appetizing close-ups, professional food "
        "styling, shallow depth of field"
    )

    _SCENES = (
        "the ingredients laid out on a wooden board, warm kitchen light, inviting and organised",
        "the key cooking step: hands at work, sizzling pan, steam and motion, appetizing detail",
        "the finishing moment: garnish, drizzle, or the last flip, glossy and tempting",
        "the plated dish: the payoff shot, steam rising, beautiful plate, first-bite energy",
    )

    def visual_strategy(self, script: str) -> VisualPlan:
        return self._beats_to_scenes(
            script,
            self._SCENES,
            style_lock=self._STYLE,
            motions=("zoom-in-slow", "pan-right-slow", "zoom-in-fast", "zoom-in-slow"),
            notes="Warm food look — overhead and close-up, steam and gloss, the plate is the star.",
        )

    title_template = "{topic} — Cook It in 40 Seconds"
    description_template = (
        "{topic}\n\n"
        "Three steps, one pan, one very good plate. 🍳\n\n"
        "{script}"
    )
    hashtags = {
        "tiktok": ["#cooking", "#recipe", "#foodtok", "#shorts", "#easyrecipes"],
        "youtube": ["#cooking", "#recipe", "#shorts"],
        "instagram": ["#cooking", "#recipe", "#food", "#reels"],
        "default": ["#cooking", "#recipe"],
    }
    voice_spec = VoiceSpec(
        preferred_voice="devon-cook",
        mood="warm",
        intensity=3,
        pace_wpm=155,
        pause_style="natural",
    )

    ypp_rationale = (
        "Original recipe content: dishes are written out in Devon's own "
        "words with original AI-generated food visuals. No copied "
        "recipes, no stolen food photography — transformative original "
        "content, YPP-safe."
    )


register(CookingNiche())
