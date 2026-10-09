"""Niche: reddit_stories — condensed storytime with comment-bait endings."""

from __future__ import annotations

from .base import NichePlugin, VisualPlan, VoiceSpec
from .registry import register


class RedditStoriesNiche(NichePlugin):
    name = "reddit_stories"
    thesis = "The wildest thread of the day, condensed to 45 seconds with a cliffhanger that fills the comments."
    cadence = 3.0
    target_seconds = (25, 50)

    def script_prompt(self, topic: str) -> str:
        lo, hi = self.word_budget()
        return f"""You are the narrator for a storytime channel that turns long forum threads into shorts people argue about in the comments. You condense ruthlessly and you always know exactly where to cut.

TOPIC: {topic}

Write a spoken-word script, {lo}–{hi} words, for a vertical short.

VOICE: conversational, a little messy — like a friend retelling insane drama they just read. First names only, real stakes, zero corporate polish. React to the story as you tell it ("and THEN—"). The audience should feel like they're hearing gossip, not a summary.

RETENTION STRUCTURE (mandatory):
1. HOOK (first line): the thread title as a bomb — the most unhinged one-sentence version of the story. "My mother-in-law moved into our house while we were on honeymoon." Make them NEED the context.
2. THE SETUP (fast, 20%): who, what, where — in three sentences max. No backstory that doesn't pay off.
3. THE ESCALATION: the 3 wildest beats of the thread, in order, each worse than the last. Cut right BEFORE the resolution — "and that's when I found the camera—"
4. THE CLIFFHANGER ENDING: stop at the most debated moment of the thread and ask the audience the exact question the comments fought over. Do NOT resolve it. The comments are the second half of the video.

RULES:
- Plain spoken text only. No stage directions, no [brackets], no emojis, no "hey guys".
- Retell in your own words — never quote the thread verbatim. Change names.
- Nothing defamatory: no real full names, no real usernames, no identifiable details. Treat every story as "allegedly".
- Keep it PG-13: imply the wild parts, don't narrate explicit detail.
- One story per script.

Return ONLY the script text, paragraphs separated by blank lines."""

    _STYLE = (
        "stylized flat illustration, bold outlines, expressive cartoon characters "
        "in dramatic poses, vibrant saturated colors, comic-book energy, clean "
        "vector look, dramatic lighting"
    )

    _SCENES = (
        "shocked cartoon character dropping their phone, exaggerated expression, dramatic background burst",
        "two cartoon characters arguing across a dinner table, plates flying, motion lines",
        "detective-style evidence board with red string connecting photos, magnifying glass, dramatic",
        "crowd of cartoon commenters with speech bubbles, one giant question mark in the center, vibrant",
    )

    def visual_strategy(self, script: str) -> VisualPlan:
        return self._beats_to_scenes(
            script,
            self._SCENES,
            style_lock=self._STYLE,
            motions=("zoom-in-fast", "pan-left-fast", "zoom-in-slow", "zoom-in-slow"),
            transitions=("cut", "whip-pan", "cut", "cut"),
            notes="Cartoon storytime look — expressive, fast, meme-adjacent but original art.",
        )

    title_template = "{topic} — Reddit's Wildest Thread Today"
    description_template = (
        "{topic}\n\n"
        "Full story in the comments — drop your verdict below. 👇\n\n"
        "{script}"
    )
    hashtags = {
        "tiktok": ["#redditstories", "#storytime", "#reddit", "#drama", "#aitah"],
        "youtube": ["#redditstories", "#storytime", "#shorts"],
        "instagram": ["#storytime", "#reddit", "#drama", "#reels"],
        "default": ["#redditstories", "#storytime"],
    }
    voice_spec = VoiceSpec(
        preferred_voice="devon-gossip",
        mood="animated",
        intensity=4,
        pace_wpm=165,
        pause_style="natural",
    )

    ypp_rationale = (
        "Transformative retelling: stories are condensed, reworded, and "
        "re-narrated with original commentary and original cartoon visuals — "
        "never verbatim reading of threads over gameplay footage. The "
        "comment-bait framing is original editorial. YPP-safe."
    )


register(RedditStoriesNiche())
