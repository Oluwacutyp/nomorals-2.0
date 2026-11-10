"""Atlas settings + feature flags (docs 19+21 M1).

M1 ships the foundation dark: ``enabled`` defaults OFF, so no
surface lights up until packs exist (M2) and the renderer/surfaces
land (M3) — the flag is the seam those passes flip, declared here
so configuration has one home from the first pass.  Env-seeded,
following the repo's ``NM_*`` settings idiom (``core/config.py``):
``NM_MAPS_ENABLED`` and ``NM_MAPS_PACKS_DIR``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

__all__ = ["MapSettings", "map_settings"]

_TRUE = ("1", "true", "yes", "on")


@dataclass(frozen=True)
class MapSettings:
    """Atlas posture: the master flag + where packs live."""

    enabled: bool = False
    packs_dir: str = ""


def map_settings(env: Mapping[str, str] | None = None) -> MapSettings:
    """Resolve settings from ``env`` (defaults to ``os.environ``)."""
    source = os.environ if env is None else env
    return MapSettings(
        enabled=source.get("NM_MAPS_ENABLED", "").strip().lower() in _TRUE,
        packs_dir=source.get("NM_MAPS_PACKS_DIR", "").strip())
