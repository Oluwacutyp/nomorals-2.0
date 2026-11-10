"""Atlas settings + feature flags (docs 19+21 M1).

M1 ships the foundation dark: ``enabled`` defaults OFF, so no
surface lights up until packs exist (M2) and the renderer/surfaces
land (M3) — the flag is the seam those passes flip, declared here
so configuration has one home from the first pass.  Env-seeded,
following the repo's ``NM_*`` settings idiom (``core/config.py``):
``NM_MAPS_ENABLED`` and ``NM_MAPS_PACKS_DIR``.

M4 adds the geocoder's federation posture (doc 21 S3: local
first, federation only after owner policy permits):
``NM_MAPS_GEOCODE_FEDERATION`` — OFF by default, and even ON it
only *permits* the fall-through; a provider still has to be wired
for anything to answer from outside.

M5 mirrors it for routing: ``NM_MAPS_ROUTING_FEDERATION`` — OFF
by default; when ON a local no-route may fall to a wired
RouteProvider.  No provider ships in M5 either; the setting and
the honest unconfigured result are the seam.

M6 adds the live-share posture: ``NM_MAPS_LIVE_SHARE`` — OFF by
default; the LiveLocationBroker's verbs and its inbound supplied-
update hook stay dark until the owner turns it on, on top of Atlas
itself being enabled and a live ConsentStore grant per share.
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
    #: Owner policy for the geocoder (M4): may a query with no
    #: local answer fall through to a wired provider?  Default
    #: False — the local index's answer is final (doc 21 S3).
    geocode_federation: bool = False
    #: Owner policy for the router (M5): may a route with no
    #: local answer fall through to a wired provider?  Default
    #: False — the local road graph's answer is final (doc 21 S3).
    routing_federation: bool = False
    #: Owner policy for live sharing (M6): may the broker's verbs
    #: and inbound hook run at all?  Default False — consent alone
    #: does not light the surface; this flag is the second gate.
    live_share_enabled: bool = False


def map_settings(env: Mapping[str, str] | None = None) -> MapSettings:
    """Resolve settings from ``env`` (defaults to ``os.environ``)."""
    source = os.environ if env is None else env
    return MapSettings(
        enabled=source.get("NM_MAPS_ENABLED", "").strip().lower() in _TRUE,
        packs_dir=source.get("NM_MAPS_PACKS_DIR", "").strip(),
        geocode_federation=source.get(
            "NM_MAPS_GEOCODE_FEDERATION", "").strip().lower() in _TRUE,
        routing_federation=source.get(
            "NM_MAPS_ROUTING_FEDERATION", "").strip().lower() in _TRUE,
        live_share_enabled=source.get(
            "NM_MAPS_LIVE_SHARE", "").strip().lower() in _TRUE)
