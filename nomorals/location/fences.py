"""Fence + reminder store and the live evaluation seam (audit gap R1-45).

The geofence / trigger / reminder organs (``geofence.py``,
``triggers.py``, ``reminders.py``) were pure and fully pinned but
never fed: nothing stored a fence, no live update was ever evaluated
against one. This module is the missing seam:

* :class:`FenceStore` — durable fences + fence states + reminders in
  one JSON file (``fences.json``) under the location root (the same
  root ``ConsentStore`` and ``LiveLocationBroker`` use). Atomic
  tmp+replace writes; reads never raise — a corrupt store degrades
  to empty with a named reason and the corrupt file is quarantined
  (the ``maps.marks`` shape); bad rows are skipped, never fatal.
* :func:`evaluate_fence_update` — the one evaluation call: load the
  store, run ``triggers.evaluate_update`` (consent-gated, stateful)
  and ``reminders.check_reminders`` (fire-once), persist the states
  and fired marks, return the events + fired reminders. Consent is
  re-checked at evaluation time, never cached: without a live
  ``"geofence"`` grant nothing is evaluated and nothing is stored;
  reminders need their own live ``"reminders"`` grant.

Nothing here tracks anyone: the caller supplies the update (the
owner's location pin in chat); the store holds only the owner's
fences and the last-known inside/outside per fence.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from .geofence import Geofence
from .primitives import LatLon
from .privacy import ConsentStore, default_location_root
from .reminders import LocationReminder, check_reminders
from .triggers import FenceState, evaluate_update

__all__ = [
    "SCOPE_GEOFENCE", "SCOPE_REMINDERS",
    "FenceStore", "evaluate_fence_update",
    "new_fence_id", "new_reminder_id",
]

#: ConsentStore scopes. A grant for one never implies the other.
SCOPE_GEOFENCE = "geofence"
SCOPE_REMINDERS = "reminders"


def new_fence_id() -> str:
    """A fresh fence id (``f-`` + 12 hex, the live-share shape)."""
    return "f-" + uuid.uuid4().hex[:12]


def new_reminder_id() -> str:
    """A fresh reminder id (``r-`` + 12 hex)."""
    return "r-" + uuid.uuid4().hex[:12]


class FenceStore:
    """Where the owner's fences, fence states and reminders live.

    One JSON file, ``fences.json``, under the location root:
    ``{"fences": {id: Geofence}, "states": {fence_id: FenceState},
    "reminders": {id: LocationReminder}}``. Reads never raise — a
    corrupt file is quarantined to ``fences.json.corrupt-<epoch>``
    and the store reads empty with :meth:`degraded_reason` naming
    why; bad rows are skipped. Writes are atomic (tmp + replace).
    """

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = (Path(root) if root is not None
                     else default_location_root())
        self.root.mkdir(parents=True, exist_ok=True)
        self._degraded = ""

    # -- posture ------------------------------------------------------

    @property
    def path(self) -> Path:
        """The store file."""
        return self.root / "fences.json"

    def degraded_reason(self) -> str:
        """Why the store is degraded ("" when healthy)."""
        return self._degraded

    # -- storage ------------------------------------------------------

    def _load_raw(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - reads fail open
            self._quarantine(exc)
            return {}
        if not isinstance(data, dict):
            self._degraded = "fence store is not a JSON object"
            return {}
        return data

    def _quarantine(self, exc: Exception) -> None:
        self._degraded = f"fence store unreadable: {exc}"
        try:
            dest = self.path.with_name(
                f"fences.json.corrupt-{int(time.time())}")
            self.path.replace(dest)
        except Exception:  # noqa: BLE001 - quarantine is best-effort
            pass

    def _save_raw(self, data: dict[str, Any]) -> bool:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                           encoding="utf-8")
            tmp.replace(self.path)
            return True
        except Exception:  # noqa: BLE001 - writes fail closed, named
            self._degraded = "fence store unwritable"
            return False

    # -- sections -----------------------------------------------------

    def load_all(self) -> tuple[dict[str, Geofence],
                                 dict[str, FenceState],
                                 dict[str, LocationReminder]]:
        """All three sections, parsed. Never raises; bad rows skip."""
        data = self._load_raw()
        fences: dict[str, Geofence] = {}
        raw_fences = data.get("fences")
        if isinstance(raw_fences, dict):
            for payload in raw_fences.values():
                if not isinstance(payload, dict):
                    continue  # a bad row never kills the store
                try:
                    fence = Geofence.from_dict(payload)
                except (KeyError, TypeError, ValueError):
                    continue
                fences[fence.id] = fence
        states: dict[str, FenceState] = {}
        raw_states = data.get("states")
        if isinstance(raw_states, dict):
            for payload in raw_states.values():
                if not isinstance(payload, dict):
                    continue
                try:
                    state = FenceState.from_dict(payload)
                except (KeyError, TypeError, ValueError):
                    continue
                states[state.fence_id] = state
        reminders: dict[str, LocationReminder] = {}
        raw_reminders = data.get("reminders")
        if isinstance(raw_reminders, dict):
            for payload in raw_reminders.values():
                if not isinstance(payload, dict):
                    continue
                try:
                    rem = LocationReminder.from_dict(payload)
                except (KeyError, TypeError, ValueError):
                    continue
                reminders[rem.id] = rem
        return fences, states, reminders

    def save_all(self, fences: dict[str, Geofence],
                 states: dict[str, FenceState],
                 reminders: dict[str, LocationReminder]) -> bool:
        """Persist all three sections atomically."""
        return self._save_raw({
            "fences": {fid: f.to_dict() for fid, f in fences.items()},
            "states": {fid: s.to_dict() for fid, s in states.items()},
            "reminders": {rid: r.to_dict()
                          for rid, r in reminders.items()},
        })

    # -- fences -------------------------------------------------------

    def fences(self) -> list[Geofence]:
        """All fences, by id."""
        fences, _, _ = self.load_all()
        return [fences[k] for k in sorted(fences)]

    def get_fence(self, fence_id: str) -> Geofence | None:
        """One fence by exact id, else None."""
        fences, _, _ = self.load_all()
        return fences.get(str(fence_id))

    def resolve_fence(self, ref: str) -> Geofence | None:
        """A fence by exact id, unique id prefix, or exact label.

        An ambiguous id prefix resolves to nothing (honest, never
        first-match); a label matching more than one fence is
        likewise unresolved — the caller names the id instead.
        """
        ref = str(ref or "").strip()
        if not ref:
            return None
        fences, _, _ = self.load_all()
        if ref in fences:
            return fences[ref]
        prefixed = [f for fid, f in fences.items()
                    if fid.startswith(ref)]
        if len(prefixed) == 1:
            return prefixed[0]
        if prefixed:
            return None
        named = [f for f in fences.values() if f.name == ref]
        return named[0] if len(named) == 1 else None

    def add_fence(self, fence: Geofence) -> Geofence:
        """Store a fence (overwrites the same id)."""
        fences, states, reminders = self.load_all()
        fences[fence.id] = fence
        self.save_all(fences, states, reminders)
        return fence

    def remove_fence(self, fence_id: str) -> bool:
        """Drop a fence + its state + its reminders. True if any."""
        fences, states, reminders = self.load_all()
        if str(fence_id) not in fences:
            return False
        del fences[str(fence_id)]
        states.pop(str(fence_id), None)
        reminders = {rid: r for rid, r in reminders.items()
                     if r.fence_id != str(fence_id)}
        self.save_all(fences, states, reminders)
        return True

    # -- reminders ----------------------------------------------------

    def reminders(self) -> list[LocationReminder]:
        """All reminders, by id."""
        _, _, reminders = self.load_all()
        return [reminders[k] for k in sorted(reminders)]

    def add_reminder(self, reminder: LocationReminder) -> LocationReminder:
        """Store a reminder (overwrites the same id)."""
        fences, states, reminders = self.load_all()
        reminders[reminder.id] = reminder
        self.save_all(fences, states, reminders)
        return reminder

    def remove_reminder(self, reminder_id: str) -> bool:
        """Drop a reminder. True if it existed."""
        fences, states, reminders = self.load_all()
        if str(reminder_id) not in reminders:
            return False
        del reminders[str(reminder_id)]
        self.save_all(fences, states, reminders)
        return True

    def resolve_reminder(self, ref: str) -> LocationReminder | None:
        """A reminder by exact id or unique id prefix."""
        ref = str(ref or "").strip()
        if not ref:
            return None
        _, _, reminders = self.load_all()
        if ref in reminders:
            return reminders[ref]
        prefixed = [r for rid, r in reminders.items()
                    if rid.startswith(ref)]
        return prefixed[0] if len(prefixed) == 1 else None

    def rearm_reminder(self, reminder_id: str) -> bool:
        """Make a fired reminder fireable again. True if it existed."""
        fences, states, reminders = self.load_all()
        rem = reminders.get(str(reminder_id))
        if rem is None:
            return False
        rem.rearm()
        self.save_all(fences, states, reminders)
        return True


def evaluate_fence_update(store: FenceStore, consent: ConsentStore, *,
                          person_id: str, loc: LatLon,
                          ts: float | None = None) -> dict[str, Any]:
    """Evaluate ONE supplied location update against the store.

    The live feed: load the fences + states + reminders, run
    ``triggers.evaluate_update`` (stateful enter/exit) and
    ``reminders.check_reminders`` (fire-once), persist the states
    and fired marks, return ``{"events", "fired", "evaluated",
    "reason"}``. Consent is re-checked here, at evaluation time:
    without a live ``"geofence"`` grant nothing is evaluated and
    nothing is stored (``evaluated`` False, reason ``"no-consent"``);
    reminders need their own live ``"reminders"`` grant or they
    simply never fire. With no fences at all there is nothing to
    evaluate (``evaluated`` True, reason ``"no-fences"``).
    """
    person = str(person_id or "")
    if not consent.has_consent(person, SCOPE_GEOFENCE):
        return {"events": [], "fired": [], "evaluated": False,
                "reason": "no-consent"}
    fences, states, reminders = store.load_all()
    if not fences:
        return {"events": [], "fired": [], "evaluated": True,
                "reason": "no-fences"}
    events = evaluate_update(
        person_id=person, loc=loc, fences=list(fences.values()),
        states=states, consent=True, ts=ts)
    fired: list[LocationReminder] = []
    if reminders and consent.has_consent(person, SCOPE_REMINDERS):
        fired = check_reminders(list(reminders.values()), events,
                                consent=True, ts=ts)
    store.save_all(fences, states, reminders)
    return {"events": events, "fired": fired, "evaluated": True,
            "reason": "ok"}
