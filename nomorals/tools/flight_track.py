"""Flight-track tool: live flight tracking for the brain.

Devon asks where a flight is right now the way the owner asks her —
by flight number (``BA249``), callsign (``BAW249``) or hex id
(``4b1814``) — and gets a real live answer from the open ADS-B
feeds, never a schedule. One tool, action-dispatched, over
:mod:`nomorals.travel.tracking`: the organ owns the feed, the
track journal, the map render and the watcher; this module is only
the surface.

``action=track`` follows one flight (position, altitude, speed,
heading, ETA, data age, source) and optionally draws the track
map (``map=true`` writes the PNG under ``<home>/travel/``).
``action=area`` lists live aircraft in a box — explicit bounds,
or ``near=<IATA>`` for a box around an airport. ``action=watch``
arms a watch that pings when the flight lands, diverts, runs late
or vanishes; ``action=unwatch`` drops every watch on that flight,
``action=watches`` lists the armed watches. ``action=status``
fetches the feed once and reports it (live or why not).

A feed that is down or empty is reported in words with the reason
— a position is never invented, never filled from the schedule.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.policy import Capability
from ..travel.tracking import (
    FLIGHT_TRACK_JOB,
    FLIGHT_TRACK_SPEC,
    OpenSkyFeed,
    TrackStore,
    aircraft_in_area,
    coords_for_iata,
    render_track_map,
    track_flight,
)
from ..travel.tracking import FlightWatcher as _Watcher

__all__ = ["register"]


def _home_of(context: Any) -> str | None:
    settings = (getattr(context, "settings", None)
                if context is not None else None)
    home = (getattr(settings, "home", None)
            if settings is not None else None)
    return str(home) if home else None


def _feed() -> OpenSkyFeed:
    return OpenSkyFeed()


def _watcher(context: Any) -> _Watcher:
    return _Watcher(home=_home_of(context))


def _track_map(out: dict[str, Any], home: str | None,
               destination: str | None) -> dict[str, Any]:
    """Draw the track map for a successful track, when a home exists.

    The organ already journalled the fix into the track store; the
    map is drawn from the journal plus the current position. The
    PNG lands under ``<home>/travel/``; the render hash is the
    determinism receipt. A missing home means no map — the track
    itself still answers.
    """
    if home is None:
        return {}
    try:
        store = TrackStore(home)
        points = store.points(out["callsign"])
        current = {"latitude": out.get("latitude"),
                   "longitude": out.get("longitude")}
        png, digest = render_track_map(
            points, destination=destination or None, current=current)
        path = (Path(home) / "travel"
                / f"track_{out['callsign']}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png)
        return {"map_path": str(path), "map_hash": digest}
    except Exception as exc:  # noqa: BLE001 - a failed map never kills the track
        return {"map_error": str(exc)}


def register(registry: Any) -> None:
    """Attach the ``flight_track`` tool."""
    context = registry.context

    @registry.register(
        "flight_track",
        description=(
            "Live flight tracking: where a flight is RIGHT NOW — "
            "position, altitude, speed, heading, ETA — from the open "
            "ADS-B feeds, never the schedule. action=track follows "
            "one flight (flight=BA249, or a callsign BAW249, or a "
            "6-hex id 4b1814; destination=<IATA> adds the leg; "
            "map=true also draws the track map PNG under "
            "<home>/travel/). action=area lists live aircraft in a "
            "box (lat_min/lon_min/lat_max/lon_max, or near=<IATA> "
            "for a box around an airport). action=watch arms a watch "
            "that pings when the flight lands, diverts, runs late or "
            "vanishes (flight, destination=<IATA>, "
            "expect_at=<epoch seconds>); action=unwatch drops every "
            "watch on a flight; action=watches lists armed watches. "
            "action=status fetches the feed once and reports it. A "
            "down or empty feed is reported in words with the reason "
            "— a position is never invented."
        ),
        capability=Capability.NET_OUT,
    )
    def flight_track_tool(action: str = "track",
                          flight: str = "",
                          destination: str = "",
                          map: bool = False,
                          lat_min: float | None = None,
                          lon_min: float | None = None,
                          lat_max: float | None = None,
                          lon_max: float | None = None,
                          near: str = "",
                          expect_at: float | None = None,
                          limit: int = 25) -> dict[str, Any]:
        verb = (action or "track").strip().lower()
        home = _home_of(context)
        if verb == "track":
            if not (flight or "").strip():
                raise ValueError("flight_track track needs a flight "
                                 "(e.g. BA249, BAW249, 4b1814)")
            dest = (destination or "").strip().upper() or None
            store = TrackStore(home) if home else None
            out = track_flight(_feed(), flight, destination=dest,
                               store=store)
            out["action"] = "track"
            if out.get("ok") and map:
                out.update(_track_map(out, home, dest))
            return out
        if verb == "area":
            box: tuple[float, float, float, float] | None = None
            if (near or "").strip():
                coords = coords_for_iata(near)
                if coords is None:
                    raise ValueError(f"no airport {near.strip()!r}")
                lat, lon = coords
                box = (lat - 1.0, lon - 1.0, lat + 1.0, lon + 1.0)
            elif None not in (lat_min, lon_min, lat_max, lon_max):
                box = (float(lat_min), float(lon_min),  # type: ignore[arg-type]
                       float(lat_max), float(lon_max))  # type: ignore[arg-type]
            if box is None:
                raise ValueError("flight_track area needs a box "
                                 "(lat_min/lon_min/lat_max/lon_max) "
                                 "or near=<IATA>")
            states, meta = _feed().fetch(box)
            if meta.get("error"):
                return {"ok": False, "action": "area", "box": list(box),
                        "reason": meta["error"], "meta": meta}
            found = aircraft_in_area(states, box)
            return {"ok": True, "action": "area", "box": list(box),
                    "aircraft": [s.to_dict() for s in found[:limit]],
                    "count": len(found), "meta": meta}
        if verb == "watch":
            if not (flight or "").strip():
                raise ValueError("flight_track watch needs a flight")
            watch = _watcher(context).watch(
                flight, destination=destination or "",
                expect_at=float(expect_at) if expect_at else 0.0)
            return {"ok": True, "action": "watch",
                    "watch": watch.to_dict()}
        if verb == "unwatch":
            if not (flight or "").strip():
                raise ValueError("flight_track unwatch needs a flight")
            watcher = _watcher(context)
            query = (flight or "").strip().upper()
            removed = 0
            for watch in watcher.list_watches(active_only=False):
                if watch.flight == query:
                    removed += int(watcher.unwatch(watch.id))
            return {"ok": True, "action": "unwatch", "flight": query,
                    "removed": removed}
        if verb == "watches":
            return {"ok": True, "action": "watches",
                    "watches": [w.to_dict()
                                for w in _watcher(context).list_watches()]}
        if verb == "status":
            _states, meta = _feed().fetch()
            return {"ok": not bool(meta.get("error")),
                    "action": "status",
                    "source": "OpenSky Network (open ADS-B)",
                    "job": FLIGHT_TRACK_JOB, "spec": FLIGHT_TRACK_SPEC,
                    "meta": meta}
        raise ValueError(f"unknown flight_track action {verb!r} "
                         "(track | area | watch | unwatch | watches | "
                         "status)")
