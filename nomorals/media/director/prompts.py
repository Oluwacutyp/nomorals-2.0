"""Prompt assembly — design spec + scene content → generation prompt.

Deterministic: the same shot and design always assemble the same
prompt, so a prompt is auditable back to the design decisions that
produced it (the production plan stores both).
"""

from __future__ import annotations

from .breakdown import Shot
from .design import DesignSpec

__all__ = ["assemble_prompt"]


def assemble_prompt(shot: Shot, design: DesignSpec | None = None,
                    *, style_suffix: str = "") -> str:
    """Compose the videogen prompt for one shot.

    Layout: subject/action first (what the model must render), then
    the design fragments (how it is shot and lit), then mood. A
    dialogue shot carries its line as spoken content; a performance
    note rides along as direction.
    """
    design = design or shot.design
    parts: list[str] = []
    desc = (shot.description or "").strip()
    if shot.kind == "dialogue" and shot.dialogue:
        who = shot.speaker or "the character"
        parts.append(f'{who} says: "{shot.dialogue}"')
        if shot.performance_note:
            parts.append(f"delivered {shot.performance_note}")
    elif desc:
        parts.append(desc)
    if shot.characters and shot.kind in ("action", "establishing"):
        cast = ", ".join(shot.characters)
        if cast.lower() not in desc.lower():
            parts.append(f"with {cast}")
    if design is not None:
        parts.extend(design.prompt_fragments())
    if style_suffix:
        parts.append(style_suffix)
    return ", ".join(p for p in parts if p)
