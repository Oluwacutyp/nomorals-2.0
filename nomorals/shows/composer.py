"""Show composer — assemble episodes from caller-supplied segments.

A *show* is a recurring programme (schedule, voices, bed track, RSS
feed). An *episode* is composed from segments the caller supplies —
scripts from the morning pulse / research, voice turns, bed music,
ads. The composer does NOT generate scripts: it joins what it is
given, estimates duration from word count, and renders an RSS item.

Publishing delegates to :mod:`nomorals.publicface.publish` with
``pack="show"``: an episode stays a draft unless the owner approves,
and private-data markers keep it a draft even with approval (the
publicface gate's own rule). Audio / DJ / pulse rendering live in
their own modules — this composer never rebuilds them.
"""

from __future__ import annotations

import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from email.utils import formatdate
from typing import Any

__all__ = ["Show", "Episode", "compose_episode", "publish_episode", "WORDS_PER_MINUTE"]

#: Speaking pace used for the duration estimate (words per minute).
WORDS_PER_MINUTE = 150

#: Segment kinds that carry spoken words (counted in script + duration).
_SPOKEN_KINDS = ("script", "voice", "ad")


def _segment_text(segment: dict[str, Any]) -> str:
    if not isinstance(segment, dict):
        return ""
    text = segment.get("text", "")
    return str(text) if text is not None else ""


def _segment_ref(segment: dict[str, Any]) -> str:
    if not isinstance(segment, dict):
        return ""
    ref = segment.get("ref", "")
    return str(ref) if ref is not None else ""


@dataclass
class Show:
    """A recurring show: schedule, voices, bed track, RSS feed."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = ""
    schedule: str = ""  # cron-ish string, e.g. "daily 23:00" — free text
    voices: list[str] = field(default_factory=list)
    bed_track: str = ""
    rss_url: str = ""

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("title is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "title": self.title, "schedule": self.schedule,
            "voices": list(self.voices), "bed_track": self.bed_track,
            "rss_url": self.rss_url,
        }


@dataclass
class Episode:
    """A composed episode: joined script, duration estimate, RSS item."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    show_id: str = ""
    title: str = ""
    script_text: str = ""
    duration_seconds: float = 0.0
    word_count: int = 0
    segments: list[dict[str, Any]] = field(default_factory=list)
    rss_item: str = ""
    created_ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "show_id": self.show_id, "title": self.title,
            "script_text": self.script_text,
            "duration_seconds": self.duration_seconds,
            "word_count": self.word_count,
            "segments": [dict(s) for s in self.segments],
            "rss_item": self.rss_item, "created_ts": self.created_ts,
        }


def _build_rss_item(episode: Episode) -> str:
    """Render the episode as an RSS ``<item>`` (stdlib XML)."""
    item = ET.Element("item")
    title_el = ET.SubElement(item, "title")
    title_el.text = episode.title or f"Episode {episode.id}"
    desc_el = ET.SubElement(item, "description")
    desc_el.text = episode.script_text[:2000]
    guid_el = ET.SubElement(item, "guid")
    guid_el.text = episode.id
    guid_el.set("isPermaLink", "false")
    pub_el = ET.SubElement(item, "pubDate")
    pub_el.text = formatdate(episode.created_ts, usegmt=True)
    dur_el = ET.SubElement(item, "duration")
    dur_el.text = str(int(round(episode.duration_seconds)))
    return ET.tostring(item, encoding="unicode")


def compose_episode(show: Show, segments: list[dict[str, Any]],
                    title: str = "", episode_id: str = "",
                    created_ts: float | None = None) -> Episode:
    """Compose an episode from caller-supplied segments.

    ``segments`` is a list of ``{kind: script|voice|bed|ad, text/ref}``
    dicts. Spoken kinds (``script``, ``voice``, ``ad``) contribute
    their ``text`` to ``script_text`` in order; ``bed`` segments
    contribute no spoken words (their ``ref`` stays on the episode's
    segment list for the audio stage). Duration is estimated from the
    spoken word count at :data:`WORDS_PER_MINUTE`.
    """
    import time as _time

    if not isinstance(show, Show):
        raise TypeError("compose_episode() expects a Show")
    segments = list(segments or [])
    for seg in segments:
        if not isinstance(seg, dict):
            raise TypeError("each segment must be a dict")
        if seg.get("kind", "") not in ("script", "voice", "bed", "ad"):
            raise ValueError(f"unknown segment kind {seg.get('kind')!r}")
    spoken = [_segment_text(s) for s in segments if s.get("kind") in _SPOKEN_KINDS]
    spoken = [t for t in spoken if t.strip()]
    script_text = "\n\n".join(spoken)
    word_count = len(script_text.split())
    duration = round(word_count / WORDS_PER_MINUTE * 60.0, 1)
    episode = Episode(
        id=episode_id or uuid.uuid4().hex[:12],
        show_id=show.id,
        title=title or f"{show.title} — episode",
        script_text=script_text,
        duration_seconds=duration,
        word_count=word_count,
        segments=[dict(s) for s in segments],
        created_ts=float(created_ts if created_ts is not None else _time.time()),
    )
    episode.rss_item = _build_rss_item(episode)
    return episode


def publish_episode(episode: Episode, owner_approved: bool = False):
    """Publish an episode via publicface (``pack="show"``).

    Delegates to :func:`nomorals.publicface.publish.publish`: stays a
    draft unless ``owner_approved`` is true AND the script has no
    private-data markers. Returns the publicface
    :class:`PublishResult` unchanged.
    """
    from ..publicface.publish import PublicPage, publish

    if not isinstance(episode, Episode):
        raise TypeError("publish_episode() expects an Episode")
    if not episode.script_text.strip():
        raise ValueError("cannot publish an episode with an empty script")
    page = PublicPage(
        title=episode.title, content=episode.script_text,
        pack="show", slug=f"show-{episode.id}", status="draft",
    )
    return publish(page, owner_approved=bool(owner_approved))
