"""Location layer — MORE_TO_ADD gap 5.

Location PRIMITIVES, not location tracking. This package never
collects a position by itself: there is no GPS access, no polling,
no background thread, no watch. Every function takes a location
update SUPPLIED by the caller (e.g. the owner's phone reporting
"I'm here") and evaluates it against geofences / reminders.

PRIVACY-FIRST, enforced in code (``privacy.py``):

* Owner-only: location state belongs to the single owner household
  context; there is no multi-person location sharing here.
* Opt-in consent required: every evaluation function takes
  ``consent=True`` explicitly. ``consent=False`` (or missing) →
  ``ConsentError`` and NOTHING is evaluated, stored, or returned.
* Consent is per-person + per-scope, grantable and revocable
  (``ConsentStore``). Revoking consent makes later evaluations refuse.

UNIVERSAL — N PACKS, NIGERIA DEFAULT, NEVER A BOUNDARY: the default
place is Lagos, Nigeria (``primitives.DEFAULT_PLACE``). Any lat/lon
on Earth works through the same code; no coordinate is rejected for
not being in Nigeria.
"""

from __future__ import annotations

from . import geofence, privacy, primitives, reminders, triggers
from .geofence import Geofence, circle_fence, polygon_fence
from .primitives import (
    DEFAULT_PLACE,
    NIGERIA_LAGOS,
    LatLon,
    Place,
    default_place,
    haversine_m,
)
from .privacy import ConsentError, ConsentStore, require_consent
from .reminders import LocationReminder, check_reminders
from .triggers import FenceState, evaluate_update

__all__ = [
    "geofence", "privacy", "primitives", "reminders", "triggers",
    "Geofence", "circle_fence", "polygon_fence",
    "DEFAULT_PLACE", "NIGERIA_LAGOS", "LatLon", "Place",
    "default_place", "haversine_m",
    "ConsentError", "ConsentStore", "require_consent",
    "LocationReminder", "check_reminders",
    "FenceState", "evaluate_update",
]
