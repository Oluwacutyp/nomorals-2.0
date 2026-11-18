"""Paste-link media fetch (Cobalt native build, 52-websites #1).

The brain tool over :func:`nomorals.media.downloader.fetch_media`:
paste a link in any chat, name the format (video, audio, image, gif)
and the quality, and Devon returns the file — or the honest reason
the service refuses.  The tool never invents a file, a format, or a
duration: every figure on the result is measured from the landed
file or read from the source's own metadata.
"""

from __future__ import annotations

from typing import Any

from ..core.policy import Capability
from ..media.downloader import fetch_media, formats_for

__all__ = ["format_fetch_reply", "register"]


def _human_size(size_bytes: int) -> str:
    size = float(size_bytes or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def format_fetch_reply(result: dict[str, Any]) -> str:
    """One plain line for the chat reply."""
    if not result.get("ok"):
        cause = str(result.get("likely_cause") or "the fetch failed")
        hint = str(result.get("hint") or "")
        return cause + (f" — {hint}" if hint else "")
    title = str(result.get("title") or result.get("service") or "media")
    kind = str(result.get("kind") or "")
    fmt = str(result.get("format") or "")
    size = _human_size(int(result.get("size_bytes") or 0))
    duration = result.get("duration_s")
    dur = f", {float(duration):.0f}s" if isinstance(
        duration, (int, float)) and duration else ""
    service = str(result.get("service") or "")
    return (f"got it — {title} ({kind}, {fmt}, {size}{dur})"
            + (f" from {service}" if service else ""))


def register(registry: Any) -> None:
    """Attach the ``fetch_media`` tool to a registry."""
    context = registry.context

    @registry.register(
        "fetch_media",
        description=(
            "Fetch a pasted media link as the requested format. Ask it "
            "as 'grab me this video', 'download this as audio', 'make "
            "a gif of this', or 'save this photo'. url=<the pasted "
            "http(s) link>; want=<video | audio | image | gif, default "
            "video>; quality=<best | 1080p | 720p | 480p | 360p, video "
            "only, default best>; action=<fetch | formats — formats "
            "lists what the link's service offers, with its licence, "
            "health and whether it serves downloads>. Returns ok, "
            "service, licence, service_health, kind, format, path, "
            "size_bytes, duration_s, title, failure, likely_cause, "
            "hint, stages, offered_formats, gif_source and a plain "
            "reply — or ok false with the service and the reason "
            "named, never a silent failure."
        ),
        capability=Capability.NET_OUT,
    )
    def fetch_media_tool(url: str = "", action: str = "fetch",
                         want: str = "video",
                         quality: str = "best") -> dict[str, Any]:
        url = str(url or "")
        action = str(action or "fetch").strip().lower()
        if action == "formats":
            info = formats_for(url, context=context)
            rows = ", ".join(
                f"{row['label']}" for row in info.get("formats") or [])
            reply = (
                f"{info.get('service')}: {rows}"
                + ("" if info.get("downloadable")
                   else " — this service does not serve downloads")
                + f" (health: {info.get('service_health')})")
            return {**info, "reply": reply}
        result = fetch_media(
            url, want=str(want or "video"),
            quality=str(quality or "best"), context=context)
        out = result.to_dict()
        out["reply"] = format_fetch_reply(out)
        return out
