"""Animato Studio tool — spark → build → share, one call.

``animato_spark``: an idea in, a finished video out — rendered through
the shorts factory (brain script, Edge TTS voiceover, generated stills
or stock, beat-synced edit, captions, music bed) inside a persistent
StudioProject, then posted straight to YouTube when asked or handed
back as the exported file. Every rung reports honestly in the returned
dict; a failed publish is reported, never faked, and the video is
still returned.

``animato_autopost``: the scheduled mission (port step 2) — the
studio's projects run their own schedules and post themselves.
action=run sweeps due projects (one episode each, YouTube publish,
every run journaled); action=start/stop flips the mission switch;
action=status reports the switch, per-project slots and recent runs.
"""

from __future__ import annotations

from typing import Any

from ..media.contentops.autopost import (
    AutopostStore,
    autopost_status,
    run_autopost,
)
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


def animato_autopost(action: str = "status", project: str = "",
                     force: bool = False) -> dict[str, Any]:
    """Run or inspect the Animato auto-post mission.

    Args:
        action: run | status | start | stop. ``run`` sweeps the
            due projects (one episode each, posted to YouTube,
            every run journaled) — it does nothing while the
            mission is stopped, unless ``force`` is set (the manual
            run). ``start``/``stop`` flip the mission switch.
            ``status`` reports the switch, per-project slots and
            recent runs.
        project: narrow a run to one project slug ("" = all due).
        force: run even while stopped / even when the slot is not
            open (manual override; the durable job never forces).
    Returns:
        run → {"ok", "enabled", "ran", "posted", "failed",
        "skipped", "outcomes"}; status → {"ok", "enabled",
        "projects", "recent"}; start/stop → {"ok", "enabled"}.
    """
    action = (action or "status").strip().lower()
    runner = StudioRunner()
    store = AutopostStore(runner.store.root)
    if action == "start":
        return {"ok": True, "enabled": store.set_enabled(True)}
    if action == "stop":
        return {"ok": True, "enabled": store.set_enabled(False)}
    if action == "status":
        return autopost_status(runner=runner, store=store)
    if action == "run":
        return run_autopost(runner=runner, store=store,
                            project=(project or "").strip(),
                            force=bool(force))
    return {"ok": False,
            "error": f"unknown action {action!r} "
                     "(run | status | start | stop)"}


def register(registry: Any) -> None:
    """Register the Animato Studio tools."""
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
    registry.register(
        "animato_autopost",
        animato_autopost,
        description=(
            "Run or inspect the Animato auto-post mission: the studio's "
            "projects run their own schedules and post themselves to "
            "YouTube — action=run sweeps due projects (one episode each, "
            "every run journaled with what posted and what failed and "
            "why); action=start/stop flips the mission switch; "
            "action=status reports the switch, per-project slots and "
            "recent runs. The durable 'animato autopost' job fires "
            "action=run every 30 minutes; call it directly only when "
            "the owner asks for due episodes to go out now."),
        capability="media",
        examples=[
            'animato_autopost("status")',
            'animato_autopost("start")',
            'animato_autopost("run", project="stories")',
        ],
    )
