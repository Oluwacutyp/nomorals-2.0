"""``nm flight`` — live flight tracking from the console.

The console twin of ``/flight`` (FlightRadar24 build-new): the
same ``travel/tracking.py`` organ, the same words, the same state.
Live positions come from the open ADS-B feeds (OpenSky) — never
the schedule.

* ``nm flight BA249`` — follow one flight: live position,
  altitude, speed, heading, ETA, data age, source; draws the
  track map PNG under ``<home>/travel/`` and names the path
* ``nm flight area LOS`` — live aircraft over an airport (or
  ``nm flight area <lat_min> <lon_min> <lat_max> <lon_max>``)
* ``nm flight watch BA249 [destination] [expect-epoch]`` — arm a
  watch that pings on landing, diversion, delay or loss
* ``nm flight unwatch BA249`` — drop every watch on a flight
* ``nm flight watches`` — list the armed watches
* ``nm flight tick`` — run one watch tick now (the 15-minute job,
  on demand); a feed failure reports the reason and never raises
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

__all__ = ["_cmd_flight"]


def _home_path(context: Any) -> Path | None:
    settings = getattr(context, "settings", None)
    if settings is None:
        return None
    home = getattr(settings, "home_path", None) or getattr(settings, "home", None)
    return Path(home) if home else None


def _usage_error(message: str) -> int:
    print(message, file=sys.stderr)
    return 2


def _when(epoch: float) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(epoch))
    except Exception:  # noqa: BLE001 - a bad timestamp is data
        return str(epoch)


def _track_lines(out: dict[str, Any]) -> list[str]:
    lines = [f"✈️ {out['callsign'] or out['icao24']} "
             f"({out.get('flight') or ''})"]
    if out.get("latitude") is not None:
        lines.append(f"position: {out['latitude']:.4f}, "
                     f"{out['longitude']:.4f}")
    lines.append(f"altitude: {out.get('altitude_ft') or '?'} ft · "
                 f"speed: {out.get('speed_kts') or '?'} kt · "
                 f"heading: {out.get('heading') or '?'}°")
    if out.get("on_ground"):
        lines.append("on the ground")
    if out.get("destination"):
        lines.append(f"→ {out['destination']} "
                     f"{out.get('distance_km', '?')} km away, "
                     f"bearing {out.get('bearing', '?')}°"
                     + (f", ETA ~{out['eta_minutes']} min"
                        if out.get("eta_minutes") is not None else ""))
    lines.append(f"data {out.get('age_s', 0)}s old "
                 f"({out.get('freshness') or 'unknown'}) · "
                 f"source: {out.get('source') or 'OpenSky'}")
    if out.get("map_path"):
        lines.append(f"map: {out['map_path']}")
    return lines


def _draw_map(out: dict[str, Any], home: str | None,
              destination: str | None) -> str | None:
    """Draw the track map for a successful track, under <home>/travel/."""
    if home is None:
        return None
    from ...travel.tracking import TrackStore, render_track_map

    try:
        store = TrackStore(home)
        points = store.points(out["callsign"])
        current = {"latitude": out.get("latitude"),
                   "longitude": out.get("longitude")}
        png, _digest = render_track_map(
            points, destination=destination or None, current=current)
        path = (Path(home) / "travel"
                / f"track_{out['callsign']}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png)
        return str(path)
    except Exception:  # noqa: BLE001 - a failed map never kills the track
        return None


def _cmd_flight(args: argparse.Namespace, context: Any) -> int:
    """``nm flight <flight> | area | watch | unwatch | watches | tick``."""
    from ...travel.tracking import (
        FlightWatcher,
        OpenSkyFeed,
        TrackStore,
        aircraft_in_area,
        coords_for_iata,
        run_flight_track_tick_once,
        track_flight,
    )

    home = _home_path(context)
    home_s = str(home) if home else None
    first = (getattr(args, "flight_action", None) or "").strip()
    rest = list(getattr(args, "flight_rest", None) or [])
    as_json = bool(getattr(args, "json", False))
    # The first word is a verb (area/watch/unwatch/watches/tick) or
    # the flight itself (BA249) — the parser takes it free-form, so
    # a flight number is never refused as "not an action".
    verbs = ("area", "watch", "unwatch", "watches", "tick")
    action = first.lower() if first.lower() in verbs else "track"
    if action == "track" and first:
        rest = [first] + rest

    if action == "tick":
        from ...agents.notifier import Notifier

        out = run_flight_track_tick_once(
            context=context, notifier=Notifier(context))
        if as_json:
            print(json.dumps(out, indent=1, default=str))
            return 0
        if out.get("ok"):
            print(f"flight track tick: {out.get('checked', 0)} watched, "
                  f"{out.get('alerts', 0)} alerts")
        else:
            print(f"flight track tick failed: "
                  f"{out.get('error') or 'unknown error'}")
        return 0

    if action == "watches":
        watches = FlightWatcher(home=home_s).list_watches()
        if as_json:
            print(json.dumps([w.to_dict() for w in watches], indent=1,
                             default=str))
            return 0
        if not watches:
            print("no flights watched.")
            return 0
        print("watched flights:")
        for watch in watches:
            line = f"• {watch.flight}"
            if watch.destination:
                line += f" → {watch.destination}"
            if watch.expect_at:
                line += f" — expected {_when(watch.expect_at)}"
            if watch.alerted:
                line += f" (alerted: {', '.join(watch.alerted)})"
            print(line)
        return 0

    if action == "watch":
        if not rest:
            return _usage_error(
                "usage: nm flight watch <flight> [destination] "
                "[expect-epoch]")
        destination = ""
        expect_at = 0.0
        if len(rest) >= 2 and len(rest[1]) == 3:
            destination = rest[1].upper()
        for token in rest[1:]:
            try:
                expect_at = float(token)
            except (TypeError, ValueError):
                continue
        watch = FlightWatcher(home=home_s).watch(
            rest[0], destination=destination, expect_at=expect_at)
        if as_json:
            print(json.dumps(watch.to_dict(), indent=1, default=str))
            return 0
        line = f"watching {watch.flight}"
        if watch.destination:
            line += f" → {watch.destination}"
        if watch.expect_at:
            line += f" — expected {_when(watch.expect_at)}"
        print(line + ".")
        return 0

    if action == "unwatch":
        if not rest:
            return _usage_error("usage: nm flight unwatch <flight>")
        watcher = FlightWatcher(home=home_s)
        query = rest[0].upper()
        removed = 0
        for watch in watcher.list_watches(active_only=False):
            if watch.flight == query:
                removed += int(watcher.unwatch(watch.id))
        if as_json:
            print(json.dumps({"flight": query, "removed": removed},
                             indent=1, default=str))
            return 0
        if removed:
            print(f"no longer watching {query}.")
        else:
            print(f"no watch on {query}.")
        return 0

    if action == "area":
        if not rest:
            return _usage_error(
                "usage: nm flight area <IATA> | nm flight area "
                "<lat_min> <lon_min> <lat_max> <lon_max>")
        box: tuple[float, float, float, float] | None = None
        label = ""
        if len(rest) == 1 and len(rest[0]) == 3:
            coords = coords_for_iata(rest[0])
            if coords is None:
                print(f"no airport {rest[0].upper()!r}", file=sys.stderr)
                return 2
            lat, lon = coords
            box = (lat - 1.0, lon - 1.0, lat + 1.0, lon + 1.0)
            label = f"over {rest[0].upper()}"
        elif len(rest) == 4:
            try:
                box = (float(rest[0]), float(rest[1]),
                       float(rest[2]), float(rest[3]))
            except (TypeError, ValueError):
                return _usage_error(
                    "usage: nm flight area <IATA> | nm flight area "
                    "<lat_min> <lon_min> <lat_max> <lon_max>")
            label = "in that box"
        if box is None:
            return _usage_error(
                "usage: nm flight area <IATA> | nm flight area "
                "<lat_min> <lon_min> <lat_max> <lon_max>")
        states, meta = OpenSkyFeed().fetch(box)
        if as_json:
            found = [] if meta.get("error") else aircraft_in_area(
                states, box)
            print(json.dumps(
                {"box": list(box),
                 "aircraft": [s.to_dict() for s in found],
                 "count": len(found), "meta": meta},
                indent=1, default=str))
            return 0
        if meta.get("error"):
            print(f"live feed unavailable: {meta['error']} "
                  "(the schedule is not a substitute)")
            return 0
        found = aircraft_in_area(states, box)
        if not found:
            print(f"no live aircraft {label} right now "
                  f"({meta.get('source') or 'OpenSky'})")
            return 0
        print(f"live aircraft {label} ({len(found)}, "
              f"{meta.get('source') or 'OpenSky'}):")
        for state in found[:15]:
            print(f"• {state.callsign or state.icao24} — "
                  f"{state.altitude_ft if state.altitude_ft is not None else '?'} ft, "
                  f"{state.speed_kts if state.speed_kts is not None else '?'} kt, "
                  f"{state.true_track if state.true_track is not None else '?'}°")
        if len(found) > 15:
            print(f"… and {len(found) - 15} more")
        return 0

    if action == "track":
        query = rest[0] if rest else ""
        if not query:
            return _usage_error(
                "usage: nm flight <flight> | nm flight area <IATA> | "
                "nm flight watch <flight> [destination] [expect-epoch] "
                "| nm flight unwatch <flight> | nm flight watches | "
                "nm flight tick")
        out = track_flight(OpenSkyFeed(), query,
                           store=TrackStore(home_s) if home_s else None)
        if as_json:
            print(json.dumps(out, indent=1, default=str))
            return 0
        if not out.get("ok"):
            print(f"no live position for {query.upper()} right now — "
                  f"{out.get('reason') or 'unknown reason'}")
            return 0
        map_path = _draw_map(out, home_s, out.get("destination"))
        if map_path:
            out["map_path"] = map_path
        print("\n".join(_track_lines(out)))
        return 0

    return _usage_error(
        "usage: nm flight <flight> | nm flight area <IATA> | "
        "nm flight watch <flight> [destination] [expect-epoch] | "
        "nm flight unwatch <flight> | nm flight watches | "
        "nm flight tick")
