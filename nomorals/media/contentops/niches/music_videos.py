"""Niche: music_videos — a song theme as a 40-second visual piece, cut on the beat."""

from __future__ import annotations

from .base import NichePlugin, VisualPlan, VoiceSpec
from .registry import register


class MusicVideosNiche(NichePlugin):
    name = "music_videos"
    default_style = "phonk"
    thesis = "A song theme as a 40-second visual piece — a spoken hook, imagery cut exactly on the beat, the chorus line as the climax."
    cadence = 1.0
    target_seconds = (25, 50)

    def script_prompt(self, topic: str) -> str:
        lo, hi = self.word_budget()
        return f"""You are the director for an original-songs channel. Your music videos win because the words are few and the images land on the beat — a short spoken hook frames the song, then the chorus line hits like the drop.

TOPIC: {topic}

Write a spoken-word script, {lo}–{hi} words, for a vertical short.

VOICE: low-key and rhythmic — short lines that land on the beat, space between them, like a vocalist talking between verses. The music carries the feeling; your words carry the story.

RETENTION STRUCTURE (mandatory):
1. HOOK (first line): one spoken line that frames the song — the feeling, the place, the person. Make them NEED the chorus.
2. THE VERSE (40%): the story of the song in two beats — who is singing, what they lost or want. Short lines, one image per line.
3. THE CLIMAX (last lines): the chorus line — the emotional peak, repeated once, slower the second time. This is the payoff the whole short builds to.

RULES:
- Plain spoken text only. No stage directions, no [brackets], no emojis, no "hey guys".
- Keep it short — this is a music video, not a monologue. The spoken lines are the frame; the beat is the picture.
- Original song only: never present the song as a real artist's work, never quote real lyrics. The theme in the topic is the song; the words are new.
- An angle in the topic ("the chorus", "the bridge") means this episode is that part of the song — lead with it.

Return ONLY the script text, paragraphs separated by blank lines."""

    _STYLE = (
        "cinematic music video still, dramatic stage lighting, neon "
        "accents, film grain, high contrast, emotional close-ups, "
        "atmospheric haze"
    )

    _SCENES = (
        "the opening frame: a lone figure in dramatic light, the mood of the song in one shot",
        "the verse image: the story of the song, cinematic movement, neon and shadow",
        "the beat-drop moment: motion blur, light burst, the energy spiking",
        "the chorus climax frame: the emotional peak, face in dramatic light, unforgettable still",
    )

    def visual_strategy(self, script: str) -> VisualPlan:
        return self._beats_to_scenes(
            script,
            self._SCENES,
            style_lock=self._STYLE,
            motions=("zoom-in-slow", "pan-left-fast", "zoom-in-fast", "zoom-in-slow"),
            transitions=("cut", "cut", "whip-pan", "cut"),
            notes="Music-video look — beat-cut energy, neon and grain, the climax frame is the thumbnail.",
        )

    title_template = "{topic} — A Music Video"
    description_template = (
        "{topic}\n\n"
        "An original song, told in forty seconds. 🎵\n\n"
        "{script}"
    )
    hashtags = {
        "tiktok": ["#music", "#musicvideo", "#song", "#shorts", "#originalsong"],
        "youtube": ["#music", "#musicvideo", "#shorts"],
        "instagram": ["#music", "#musicvideo", "#reels"],
        "default": ["#music", "#musicvideo"],
    }
    voice_spec = VoiceSpec(
        preferred_voice="devon-singer",
        mood="emotional",
        intensity=4,
        pace_wpm=140,
        pause_style="natural",
    )

    ypp_rationale = (
        "Original song themes: lyrics are written fresh for the theme, "
        "visuals are original AI-generated music-video frames, Devon's "
        "own voice. No copyrighted lyrics, no real artist's work, no "
        "impersonation — transformative original content, YPP-safe."
    )


register(MusicVideosNiche())
