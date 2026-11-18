"""Video-brief tool: ``video_brief`` — a long video as a sectioned brief.

Devon hands over a YouTube link (or a local video path) and gets the
video back as a brief: timestamped sections with headings, each
section's summary and key points, an overall TLDR, every section
deep-linking to its moment, and the brief naming which transcript it
was built from (caption track vs ASR). The tool is a thin surface over
:mod:`nomorals.media.video_brief`, which composes the caption fetcher,
the download/transcribe rungs and the document summariser — one
pipeline, no second corpus. A video with neither a caption track nor
a transcribable audio track is refused honestly, never invented.
"""

from __future__ import annotations

from typing import Any

from ..core.policy import Capability

__all__ = ["register"]


def _home_from_context(context: Any) -> str | None:
    settings = getattr(context, "settings", None) if context else None
    if settings is None:
        return None
    home = getattr(settings, "home_path", None) or getattr(
        settings, "home", None)
    return str(home) if home else None


def register(registry: Any) -> None:
    """Attach the ``video_brief`` tool to a registry."""
    context = registry.context

    @registry.register(
        "video_brief",
        description=(
            "Video brief — a long YouTube video (or local video file) as "
            "a sectioned brief. Give it a link and get timestamped "
            "sections with headings, each section's summary and key "
            "points, an overall TLDR, every section deep-linking to its "
            "moment, and the brief naming which transcript it was built "
            "from (caption track vs ASR transcription). Foreign-language "
            "videos arrive translated into English, original title kept. "
            "action=<brief|list|show>; brief takes ref=<YouTube URL or "
            "local path>; show takes brief_id=<id from a previous brief>. "
            "A video with no caption track and no transcribable audio is "
            "refused honestly — no section, fact or timestamp is ever "
            "invented. Returns ok, brief, reply, or error."
        ),
        capability=Capability.NET_OUT,
    )
    def video_brief_tool(
        ref: str = "",
        action: str = "brief",
        brief_id: str = "",
        window_s: float = 300.0,
        abstractive: bool = False,
    ) -> dict[str, Any]:
        from ..media.video_brief import (
            VideoBriefStore,
            brief_video,
            format_brief_reply,
        )

        home = _home_from_context(context)
        store = VideoBriefStore(home=home)
        act = (action or "brief").strip().lower()
        if act == "list":
            return {"ok": True, "briefs": store.list()}
        if act == "show":
            wanted = (brief_id or ref or "").strip()
            record = store.get(wanted)
            if record is None:
                return {"ok": False,
                        "error": f"no stored brief '{wanted}'"}
            from ..media.video_brief import VideoBrief
            brief = VideoBrief.from_dict(record)
            return {"ok": True, "brief": brief.to_dict(),
                    "reply": format_brief_reply(brief)}
        target = (ref or "").strip()
        if not target:
            return {"ok": False,
                    "error": "brief needs a YouTube URL or local video path"}
        brief = brief_video(
            target, home=home, context=context,
            window_s=float(window_s or 300.0),
            abstractive=bool(abstractive), store=store)
        if not brief.ok:
            return {"ok": False, "error": brief.reason}
        return {"ok": True, "brief": brief.to_dict(),
                "reply": format_brief_reply(brief)}
