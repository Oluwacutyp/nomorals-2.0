"""Profile-aware target pick for the renderer ladder.

The surface chooses, not the device: an explicit surface always
wins; otherwise a 3D scene goes live (it has no PNG rung) and
everything else goes to chat as a PNG. The profile kind is
accepted and recorded in the render record, but it never clamps a
renderer away — SOUL §7, no artificial limitations: all three
renderers are built everywhere, the profile runtime only picks
which one a surface gets.
"""

from __future__ import annotations

from typing import Any

from .spec import KINDS, SceneSpecError

__all__ = ["pick_target"]

TARGETS = ("live", "chat", "terminal")


def pick_target(spec: dict[str, Any], *,
                surface: str | None = None,
                profile_kind: str | None = None) -> str:
    """Pick the render target for a spec: ``live`` | ``chat`` | ``terminal``.

    ``surface`` overrides everything. Without it, a scene goes live
    and a diagram/overlay goes to chat (PNG). ``profile_kind`` is
    recorded by the caller; it never changes the pick.
    """
    if surface is not None:
        if surface not in TARGETS:
            raise SceneSpecError(
                f"scene spec: surface must be one of {TARGETS}, "
                f"got {surface!r}")
        return surface
    kind = str(spec.get("kind", ""))
    if kind not in KINDS:
        raise SceneSpecError(
            f"scene spec: 'kind' must be one of {KINDS}, got {kind!r}")
    if kind == "scene":
        return "live"
    return "chat"
