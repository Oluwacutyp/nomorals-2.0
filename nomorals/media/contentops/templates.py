"""Animato Studio series templates (port step 3) — the four automated genres.

A template is a genre as data: the niche that renders it (brief-shape +
scene-shape + voice preset), the edit style, the voice, and the music-bed
preset. A template turns a *seed* — a story idea, a news hook, a song
theme, a recipe — into queued topics in a :class:`StudioProject`'s
``topic_queue``, shaped so the step-2 auto-post mission picks them up
and renders them through the existing spark → build → share flow.

Seed shaping, honestly:

* A seed with several items (comma/semicolon/newline separated) queues
  one episode per item, each shaped with the genre's base part.
* A single item with ``count > 1`` queues the genre's part arc: each
  part is a different angle on the same seed (setup/turn/landing,
  what-happened/why-it-matters/what-next, ...), so the episodes form
  a series instead of repeating the same topic.
* Nothing is invented beyond the seed: every queued topic names the
  seed (or one of its items) verbatim, plus the angle.

Applying a template to a project sets the project's niche/style/voice/
music to the genre's presets (reported as ``changes``) and appends the
shaped topics to the queue. The niche is validated at apply time —
an unknown niche fails closed here, never as a generic render later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .niches import get as _get_niche
from .studio import StudioStore

__all__ = [
    "TEMPLATES",
    "SeriesTemplate",
    "apply_template",
    "get_template",
    "list_templates",
]

#: The most episodes one apply queues from a single seed. A series arc
#: has at most three parts; beyond that the caller is queueing a list.
MAX_TOPICS_PER_APPLY = 10


@dataclass(frozen=True)
class SeriesTemplate:
    """One automated genre: render presets + seed shaping.

    ``parts`` are the episode angles for a single-item seed: each is a
    ``str.format`` template taking ``{seed}`` (and ``{n}``, the part
    number). ``parts[0]`` is always the plain seed — it is also the
    shape used for every item of a multi-item seed.
    """

    id: str
    name: str
    description: str
    niche: str
    style: str
    voice: str = "edge"
    music: str = ""
    parts: tuple[str, ...] = ("{seed}",)

    def topics(self, seed: str, count: int = 1) -> list[str]:
        """Shape ``seed`` into episode topics (see module docstring).

        Raises ValueError for an empty seed or a count outside 1–10.
        """
        items = [s.strip() for s in re.split(r"[,;\n]+", seed or "")
                 if s.strip()]
        if not items:
            raise ValueError(
                "give a seed — a story idea, a news hook, a song theme, "
                "or a recipe (comma-separated = one episode each)")
        try:
            n = int(count)
        except (TypeError, ValueError):
            raise ValueError(
                f"count must be a whole number, got {count!r}") from None
        if n < 1 or n > MAX_TOPICS_PER_APPLY:
            raise ValueError(
                f"count must be 1–{MAX_TOPICS_PER_APPLY}, got {n}")
        out: list[str] = []
        if len(items) > 1:
            for item in items[:n]:
                out.append(self.parts[0].format(seed=item, n=1))
        else:
            item = items[0]
            for i in range(n):
                if i < len(self.parts):
                    out.append(self.parts[i].format(seed=item, n=i + 1))
                else:
                    out.append(f"{item} (part {i + 1})")
        seen: set[str] = set()
        deduped = []
        for topic in out:
            if topic not in seen:
                seen.add(topic)
                deduped.append(topic)
        return deduped

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe (no callables — the parts are plain strings)."""
        return {"id": self.id, "name": self.name,
                "description": self.description, "niche": self.niche,
                "style": self.style, "voice": self.voice,
                "music": self.music, "parts": list(self.parts)}


TEMPLATES: dict[str, SeriesTemplate] = {
    "stories": SeriesTemplate(
        id="stories",
        name="Stories",
        description="Original short fiction — a complete story in 40 "
                    "seconds: hook, turn, landing.",
        niche="stories",
        style="minimal",
        voice="edge",
        music="cinematic",
        parts=("{seed}",
               "{seed} (part 2: the twist)",
               "{seed} (part 3: the ending)"),
    ),
    "news": SeriesTemplate(
        id="news",
        name="News",
        description="One news event explained plainly — what happened, "
                    "why it matters, what happens next.",
        niche="news",
        style="documentary",
        voice="edge",
        music="ambient",
        parts=("{seed}",
               "{seed} — why it matters",
               "{seed} — what happens next"),
    ),
    "music": SeriesTemplate(
        id="music",
        name="Music videos",
        description="A song theme as a 40-second visual piece, cut on "
                    "the beat — the chorus line is the climax.",
        niche="music_videos",
        style="phonk",
        voice="edge",
        music="phonk",
        parts=("{seed}",
               "{seed} (the chorus)",
               "{seed} (the bridge)"),
    ),
    "cooking": SeriesTemplate(
        id="cooking",
        name="Cooking",
        description="One dish in 40 seconds — three steps, one finished "
                    "plate as the payoff.",
        niche="cooking",
        style="vlog",
        voice="edge",
        music="upbeat",
        parts=("{seed}",
               "{seed} — the secret technique",
               "{seed} — the twist"),
    ),
}


def list_templates() -> list[SeriesTemplate]:
    """All four templates, in the Animato genre order."""
    return [TEMPLATES[k] for k in ("stories", "news", "music", "cooking")]


def get_template(template_id: str) -> SeriesTemplate:
    """Return a template by id; ValueError names the available ones."""
    key = (template_id or "").strip().lower()
    try:
        return TEMPLATES[key]
    except KeyError:
        raise ValueError(
            f"unknown template {template_id!r} — use: "
            f"{', '.join(t.id for t in list_templates())}") from None


def apply_template(store: StudioStore, template_id: str, slug: str,
                   seed: str, count: int = 1) -> dict[str, Any]:
    """Apply a template to a project: genre presets + queued topics.

    Sets the project's niche/style/voice/music to the template's
    presets, shapes the seed into episode topics, appends them to the
    project's ``topic_queue`` and saves. Returns ``{"ok", "template",
    "project", "niche", "style", "voice", "music", "topics",
    "queued", "changes"}`` — ``changes`` names every preset that
    moved, ``"old → new"``. Raises ValueError for an unknown template
    or a bad seed/count, FileNotFoundError for an unknown project.
    """
    template = get_template(template_id)
    topics = template.topics(seed, count)
    # Fail closed here when the template's niche does not resolve —
    # the render would otherwise fall back to the generic shim.
    _get_niche(template.niche)
    project = store.load(slug)
    changes: dict[str, str] = {}
    for field_name in ("niche", "style", "voice", "music"):
        new = getattr(template, field_name)
        old = getattr(project, field_name)
        if old != new:
            changes[field_name] = f"{old or '(none)'} → {new}"
            setattr(project, field_name, new)
    project.topic_queue.extend(topics)
    store.save(project)
    return {"ok": True, "template": template.id, "project": project.slug,
            "niche": project.niche, "style": project.style,
            "voice": project.voice, "music": project.music,
            "topics": topics, "queued": len(project.topic_queue),
            "changes": changes}
