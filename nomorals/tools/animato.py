"""Animato Studio tool — spark → build → share, one call.

``animato_spark``: an idea in, a finished video out — rendered through
the shorts factory (brain script, Edge TTS voiceover, generated stills
or stock, beat-synced edit, captions, music bed) inside a persistent
StudioProject, then posted straight to YouTube when asked or handed
back as the exported file. Every rung reports honestly in the returned
dict; a failed publish is reported, never faked, and the video is
still returned.
"""

from __future__ import annotations

from typing import Any

from ..media.contentops.studio import StudioRunner, slugify


def animato_spark(idea: str, project: str = "", style: str = "",
                  voice: str = "", publish: str = "") -> dict[str, Any]:
    """Run one Animato Studio episode from a spark.

    Args:
        idea: the spark — an idea in plain words (required).
        project: studio project slug ("" = ad-hoc project from the idea).
        style: edit style preset ("" = project default).
        voice: voice for the narration ("" = project default, "edge" =
            free Edge TTS first).
        publish: "" (export only — the file path is the result) or
            "youtube" (post straight to the channel; OAuth-configured
            only; a failed post is reported, never faked).
    Returns:
        {"ok", "project", "episode_id", "job_id", "run_id",
        "video_path", "published", "rungs", "notes"} — every rung
        named honestly (plan, preflight, render, publish).
    """
    runner = StudioRunner()
    if project:
        proj = runner.store.load(project)
    else:
        slug = slugify(idea)
        try:
            proj = runner.store.load(slug)
        except FileNotFoundError:
            proj = runner.store.create(idea, niche="did_you_know",
                                       style=style or "",
                                       voice=voice or "edge")
    if style:
        proj.style = style
        runner.store.save(proj)
    if voice:
        proj.voice = voice
        runner.store.save(proj)
    return runner.spark(proj, idea, publish=publish or "")


def register(registry: Any) -> None:
    """Register the Animato Studio tool."""
    registry.register(
        "animato_spark",
        animato_spark,
        description=(
            "Run one Animato Studio episode from a spark: an idea in, "
            "a finished video out — rendered through the shorts "
            "factory inside a persistent studio project (cast, style, "
            "voice, channel, schedule), then posted straight to "
            "YouTube when asked or handed back as the exported file. "
            "Returns ok/project/episode_id/job_id/run_id/video_path/"
            "published/rungs/notes with every rung named honestly."),
        capability="media",
        examples=[
            'animato_spark("why the moon looks bigger on the horizon")',
            'animato_spark("a 60 second story about a lighthouse keeper", '
            'project="stories", publish="youtube")',
        ],
    )
