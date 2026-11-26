"""Short-video factory tool — MoneyPrinterTurbo shape, Devon organs.

``make_short``: topic in, finished narrated vertical short out. The
brain writes the script (no LLM key), Edge TTS speaks it (free, no
key; catalogue voice when edge is down), stock footage fills the
scenes when Pexels/Pixabay keys are set (generated stills otherwise),
Whisper re-times the captions when faster-whisper is installed. The
publish rung posts the finished short through the factory's own
publisher organ (``agents.shorts_publisher.publish_video`` — official
APIs, OAuth-configured only). Every rung reports honestly in the
returned dict.
"""

from __future__ import annotations

from typing import Any

from ..media.contentops.factory import make_short as _make_short


def make_short(topic: str, niche: str = "did_you_know",
               style: str = "", footage: str = "auto",
               voice: str = "edge", publish: str = "") -> dict[str, Any]:
    """Make a finished narrated vertical short from a topic.

    Args:
        topic: what the short is about (required).
        niche: script/visual voice — did_you_know, motivation,
            finance_facts, horror_stories, reddit_stories, sports_edits,
            anime_edits.
        style: edit style preset ("" = niche default).
        footage: "auto" (stock when configured, stills otherwise),
            "stock" (stock first, stills on miss), "stills" (generated
            stills only).
        voice: "edge" (free Edge TTS first, catalogue on failure) or
            "catalogue" (niche voice catalogue only).
        publish: "" (no post) or one of youtube, tiktok, x, instagram,
            facebook — official APIs, OAuth-configured only; a failed
            post is reported, never faked.
    Returns:
        {"ok", "path", "run_id", "job_id", "rungs", "notes"} — every
        rung named honestly (script, voiceover, visuals, captions,
        music, publish).
    """
    result = _make_short(topic, niche=niche, style=style, footage=footage,
                         voice=voice)
    rungs = result.get("rungs") or {}
    notes = result.get("notes") or []
    if not result.get("ok"):
        rungs["publish"] = "skipped (render failed)"
        result["rungs"] = rungs
        result["notes"] = notes
        return result
    if publish:
        # The publish step lives here, not in the L4 factory: this
        # tool surface is pinned L5, same layer as the publisher
        # organ. The import is lazy — publishing is the exception,
        # not the render path.
        from ..agents.shorts_publisher import publish_video

        pub = publish_video(result.get("path", ""), publish)
        rungs["publish"] = ("posted" if pub.get("ok")
                            else f"not posted: {pub.get('reason', '')}")
        if pub.get("ok"):
            notes.append(f"posted to {publish}")
        else:
            notes.append(f"publish skipped: {pub.get('reason', '')}")
    else:
        rungs["publish"] = "skipped (no platform asked)"
    result["rungs"] = rungs
    result["notes"] = notes
    return result


def register(registry: Any) -> None:
    """Register the shorts factory tool."""
    registry.register(
        "make_short",
        make_short,
        description=(
            "Make a finished narrated vertical short from a topic: the "
            "brain writes the script, Edge TTS (free, no key) speaks "
            "it, stock footage fills the scenes when Pexels/Pixabay "
            "keys are set (generated stills otherwise), Whisper "
            "re-times the captions, and the publish rung posts it when "
            "a platform's OAuth is configured. Returns ok/path/run_id/"
            "job_id/rungs/notes with every rung named honestly."),
        capability="media",
        examples=[
            'make_short("why the brain deletes most of what you see")',
            'make_short("compound interest", niche="finance_facts", '
            'footage="stills")',
            'make_short("a ghost story", niche="horror_stories", '
            'publish="tiktok")',
        ],
    )
