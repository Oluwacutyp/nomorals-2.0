"""TrendClip topic configuration — generalized from the source's
``config/topics/gta6.json`` shape.

The source ships one topic file (GTA 6) as its example. Devon generalizes
it: a topic is any subject the owner names — a game, a creator, a subject,
a moment — and the config is the same shape the source uses, minus the
graphical ``scenes`` block. That block fed the source's original-graphic
renderer; Devon already owns that surface (qcard / canvas / makers), so it
is not ported here — the topic only carries what the clip pipeline needs:
what to search for, what to exclude, and how long a clip may run.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "STRATEGIES",
    "TopicConfig",
    "load_topic",
    "topic_from_dict",
    "topic_from_query",
]

#: The moment-selection strategies the deterministic scorer knows, ported
#: from the source's ``highlight_strategies.STRATEGIES``.
STRATEGIES = ("gaming", "conversation", "educational", "comedy", "craft")

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(text: str) -> str:
    slug = _SLUG_RE.sub("-", (text or "").strip().lower()).strip("-")
    return slug or "topic"


@dataclass
class TopicConfig:
    """A trendclip topic: what to look for, what to skip, how to cut."""

    slug: str
    topic_name: str
    language: str = "en"
    keywords: list[str] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)
    strategy: str = "conversation"
    min_clip_seconds: float = 15.0
    max_clip_seconds: float = 120.0
    target_clip_seconds: float = 60.0
    channels: list[str] = field(default_factory=list)

    def validate(self) -> list[str]:
        """Return a list of problems; empty means the topic is usable."""
        problems: list[str] = []
        if not self.slug:
            problems.append("slug is empty")
        elif self.slug != _slugify(self.slug):
            problems.append(f"slug {self.slug!r} is not a valid slug")
        if not self.topic_name.strip():
            problems.append("topic_name is empty")
        if not self.keywords:
            problems.append("keywords is empty — nothing to search for")
        if self.strategy not in STRATEGIES:
            problems.append(
                f"strategy {self.strategy!r} unknown; "
                f"use one of {', '.join(STRATEGIES)}")
        if self.min_clip_seconds <= 0:
            problems.append("min_clip_seconds must be positive")
        if self.max_clip_seconds < self.min_clip_seconds:
            problems.append("max_clip_seconds is below min_clip_seconds")
        if not (self.min_clip_seconds
                <= self.target_clip_seconds <= self.max_clip_seconds):
            problems.append("target_clip_seconds is outside [min, max]")
        return problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "topic_name": self.topic_name,
            "language": self.language,
            "keywords": list(self.keywords),
            "exclusions": list(self.exclusions),
            "strategy": self.strategy,
            "min_clip_seconds": self.min_clip_seconds,
            "max_clip_seconds": self.max_clip_seconds,
            "target_clip_seconds": self.target_clip_seconds,
            "channels": list(self.channels),
        }


def topic_from_dict(data: dict[str, Any]) -> TopicConfig:
    """Build a topic from a dict in the source's ``gta6.json`` shape.

    Unknown keys (notably the source's graphical ``scenes`` block and its
    ``metadata``) are ignored, never rejected — the shape is the contract,
    not the extra keys.
    """
    raw_keywords = data.get("keywords") or []
    keywords = [str(k).strip() for k in raw_keywords if str(k).strip()]
    exclusions = [str(e).strip()
                  for e in (data.get("exclusions") or []) if str(e).strip()]
    channels = [str(c).strip()
                for c in (data.get("channels") or []) if str(c).strip()]
    name = str(data.get("topic_name") or data.get("name") or "").strip()
    slug = str(data.get("slug") or _slugify(name))
    return TopicConfig(
        slug=slug,
        topic_name=name or slug,
        language=str(data.get("language") or "en"),
        keywords=keywords,
        exclusions=exclusions,
        strategy=str(data.get("strategy") or "conversation"),
        min_clip_seconds=float(data.get("min_clip_seconds", 15.0)),
        max_clip_seconds=float(data.get("max_clip_seconds", 120.0)),
        target_clip_seconds=float(data.get("target_clip_seconds", 60.0)),
        channels=channels,
    )


def load_topic(path: str | Path) -> TopicConfig:
    """Load a topic from a JSON file on disk."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"topic file {path} does not contain an object")
    topic = topic_from_dict(data)
    problems = topic.validate()
    if problems:
        raise ValueError(
            f"topic file {path} is invalid: {'; '.join(problems)}")
    return topic


def topic_from_query(query: str, *, strategy: str = "conversation",
                     language: str = "en") -> TopicConfig:
    """Build a topic from a free-text query — the chat-first entry point.

    The query's words become the keywords (stopwords dropped); the slug
    derives from the query. No baked-in topic list, no GTA 6 default: the
    owner names the subject, the topic follows.
    """
    words = [w for w in re.split(r"[^A-Za-z0-9']+", query or "")
             if w and w.lower() not in _STOPWORDS]
    name = (query or "").strip()
    return TopicConfig(
        slug=_slugify(name),
        topic_name=name or "trending",
        language=language,
        keywords=words or ([name] if name else ["trending"]),
        strategy=strategy if strategy in STRATEGIES else "conversation",
    )


_STOPWORDS = frozenset(
    ["a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "about", "what", "who", "when", "where", "why", "how", "is", "are", "was", "were", "be", "been", "it", "its", "this", "that", "these", "those", "from", "by", "at", "as", "i", "you", "he", "she", "we", "they", "my", "your", "his", "her", "our", "their", "me", "him", "us", "them", "do", "does", "did", "have", "has", "had", "will", "would", "can", "could", "should", "may", "might", "show", "me", "find", "get", "give", "please"])
