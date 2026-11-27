"""Niche: news — one news event in 40 seconds, explained plainly."""

from __future__ import annotations

from .base import NichePlugin, VisualPlan, VoiceSpec
from .registry import register


class NewsNiche(NichePlugin):
    name = "news"
    default_style = "documentary"
    thesis = "One news event in 40 seconds — what happened, why it matters, what happens next."
    cadence = 3.0
    target_seconds = (25, 50)

    def script_prompt(self, topic: str) -> str:
        lo, hi = self.word_budget()
        return f"""You are the anchor for a plain-language news channel. Your explainers win because a viewer who knew nothing walks away knowing what happened, why it matters, and what to watch next — in forty seconds, with zero spin.

TOPIC: {topic}

Write a spoken-word script, {lo}–{hi} words, for a vertical short.

VOICE: clear newsreader — steady, unhurried, no hype. Short declarative sentences. Numbers spoken plainly ("roughly two hundred", never a string of digits). The listener should feel informed, not alarmed.

RETENTION STRUCTURE (mandatory):
1. HOOK (first line): the headline as one plain sentence — the event, nothing else. "Gold just hit a record high." Make them NEED the context.
2. WHAT HAPPENED (30%): the facts in order — who, what, when. Two or three sentences, no adjectives doing the work of facts.
3. WHY IT MATTERS (40%): the stakes in one plain paragraph — who it affects and how. Then THE REVEAL: the detail most reports bury, the one that changes how the story reads.
4. WHAT HAPPENS NEXT (last line): the next thing to watch — the date, the decision, the number. End on the future, not the past.

RULES:
- Plain spoken text only. No stage directions, no [brackets], no emojis, no "hey guys".
- Facts must be real and checkable. Never invent a quote, a number, or a name. If the topic gives no source, say what is known and what is not yet confirmed — never fill a gap with a guess.
- No opinion, no spin, no taking sides. Report the event; let the facts carry it.
- One event per script. An angle in the topic ("why it matters", "what happens next") means this episode is that angle — lead with it.

Return ONLY the script text, paragraphs separated by blank lines."""

    _STYLE = (
        "clean broadcast documentary still, professional news photography, "
        "sharp focus, neutral color, uncluttered composition, calm "
        "authoritative framing"
    )

    _SCENES = (
        "the scene of the event: wide documentary shot, real-world detail, natural light",
        "the people affected: candid documentary portrait, honest expression, everyday setting",
        "the key detail: close-up of the document, number, or object that changes the story",
        "the road ahead: open horizon, forward motion, calm documentary framing",
    )

    def visual_strategy(self, script: str) -> VisualPlan:
        return self._beats_to_scenes(
            script,
            self._SCENES,
            style_lock=self._STYLE,
            motions=("zoom-in-slow", "pan-left-slow", "zoom-in-slow", "zoom-out-slow"),
            notes="Documentary news look — calm, factual frames; no hype, no clutter.",
        )

    title_template = "{topic}"
    description_template = (
        "{topic}\n\n"
        "What happened, why it matters, what happens next. 📰\n\n"
        "{script}"
    )
    hashtags = {
        "tiktok": ["#news", "#breaking", "#newstok", "#shorts", "#explained"],
        "youtube": ["#news", "#breaking", "#shorts"],
        "instagram": ["#news", "#breakingnews", "#reels"],
        "default": ["#news", "#shorts"],
    }
    voice_spec = VoiceSpec(
        preferred_voice="devon-news",
        mood="neutral",
        intensity=3,
        pace_wpm=160,
        pause_style="natural",
    )

    ypp_rationale = (
        "Original editorial explainer: news events are summarised and "
        "explained in Devon's own words with original AI-generated "
        "documentary-style visuals. No rebroadcast footage, no quoted "
        "audio, no invented quotes or numbers — transformative "
        "commentary, YPP-safe."
    )


register(NewsNiche())
