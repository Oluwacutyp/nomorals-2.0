"""Radio tool: the Radio.Garden rung for the brain.

"What's on in Accra?" over the radio organ — not a standalone
fetcher. The organ (:mod:`nomorals.music_sources.radio`) owns the
station index, the place queries, the directory growth, the health
checks and the bounded live samples; this module is only the
surface between the brain and the organ.

A live stream has no end, so ``action=play`` never promises the
whole stream: it captures a bounded live sample (45 s, 2 MB cap),
returns the sample file for the play spine to send, and always
carries the direct stream URL in the reply so the owner can open
the full live stream in their own player. A dead stream degrades
via source health — the play names the reason and tries the next
station in the same place — never a silent hang.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.policy import Capability
from ..music_sources.radio import (
    StationStore,
    capture_sample,
    check_all,
    check_station,
    default_store_path,
    format_radio_reply,
    globe_dots,
    grow_from_directory,
    search_stations,
    station_by_id,
    station_by_name,
)

__all__ = ["register"]


def _home_of(context: Any) -> str | None:
    settings = (getattr(context, "settings", None)
                if context is not None else None)
    home = (getattr(settings, "home", None)
            if settings is not None else None)
    return str(home) if home else None


def _store_of(context: Any) -> StationStore:
    home = _home_of(context)
    return (StationStore(default_store_path(home)) if home
            else StationStore())


def _find(store: StationStore, ref: str) -> Any | None:
    ref = (ref or "").strip()
    if not ref:
        return None
    station = station_by_id(store, ref)
    if station is not None:
        return station
    return station_by_name(store, ref)


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _play_one(store: StationStore, station: Any, home: str | None,
              seconds: float) -> dict[str, Any]:
    sample = capture_sample(
        station, seconds=seconds,
        dest_dir=(Path(home) / "radio_samples") if home else None)
    if not sample.ok:
        return {"ok": False, "station": station.to_dict(),
                "error": sample.error}
    return {"ok": True, "station": station.to_dict(),
            "sample": {"path": sample.path, "bytes": sample.n_bytes,
                       "seconds": sample.seconds,
                       "content_type": sample.content_type},
            "stream_url": station.stream_url,
            "reply": (f"🔴 LIVE {station.name} — {station.city}, "
                      f"{station.country}\n"
                      f"  {sample.seconds}s live sample, "
                      f"{sample.n_bytes} bytes — the full live stream is "
                      f"at {station.stream_url}")}


def register(registry: Any) -> None:
    """Attach the ``radio`` tool."""
    context = registry.context

    @registry.register(
        "radio",
        description=(
            "Live radio stations indexed by place, Radio.Garden-style: "
            "action=stations takes place=<city> and optional "
            "genre=<genre> | lat=<n> lon=<n> radius_km=<n> and lists "
            "the stations for that place. action=play takes id=<station "
            "id or name> or place=<city> with optional genre=<genre> "
            "and captures a bounded live sample (45 s, 2 MB cap) of "
            "the stream, returning the sample file plus the direct "
            "stream URL — a dead stream is named and the next station "
            "in the place is tried. action=status reports how many "
            "stations the index holds and their health. action=health "
            "takes id=<station> or all=true and probes the stream(s). "
            "action=grow pulls the most-voted stations from the open "
            "radio directory into the index. action=globe returns one "
            "dot per city with its broadcaster count. action=list "
            "lists the whole index."
        ),
        capability=Capability.NET_OUT,
        result_protocol="dict_ok",
    )
    def _radio_tool(**kwargs: Any) -> dict[str, Any]:
        action = str(kwargs.get("action", "stations") or "stations").lower()
        home = _home_of(context)
        store = _store_of(context)
        if action == "stations":
            lat = _float(kwargs.get("lat"))
            lon = _float(kwargs.get("lon"))
            radius = _float(kwargs.get("radius_km")) or 100.0
            place = str(kwargs.get("place", "") or "").strip()
            genre = str(kwargs.get("genre", "") or "").strip() or None
            try:
                limit = int(kwargs.get("limit", 20))
            except (TypeError, ValueError):
                limit = 20
            found = search_stations(
                store, place=place or None, genre=genre,
                lat=lat, lon=lon, radius_km=radius, limit=limit)
            return {"ok": True,
                    "stations": [s.to_dict() for s in found],
                    "count": len(found),
                    "reply": format_radio_reply(found, place=place,
                                                genre=genre or "")}
        if action == "list":
            rows = store.all()
            return {"ok": True,
                    "stations": [s.to_dict() for s in rows],
                    "count": len(rows)}
        if action == "play":
            try:
                seconds = float(kwargs.get("seconds", 45) or 45)
            except (TypeError, ValueError):
                seconds = 45.0
            ref = str(kwargs.get("id", "") or "").strip()
            if ref:
                station = _find(store, ref)
                if station is None:
                    return {"ok": False,
                            "error": f"no station matches {ref!r}"}
                return _play_one(store, station, home, seconds)
            place = str(kwargs.get("place", "") or "").strip()
            genre = str(kwargs.get("genre", "") or "").strip() or None
            if not place and not genre:
                return {"ok": False,
                        "error": "play needs id=<station> or "
                                 "place=<city>"}
            found = search_stations(store, place=place or None,
                                    genre=genre, limit=3)
            if not found:
                return {"ok": False,
                        "error": format_radio_reply([], place=place,
                                                    genre=genre or "")}
            errors: list[str] = []
            for station in found:
                result = _play_one(store, station, home, seconds)
                if result["ok"]:
                    return result
                errors.append(f"{station.name}: {result['error']}")
            return {"ok": False,
                    "error": "every station tried was down — "
                             + "; ".join(errors)}
        if action == "status":
            return {"ok": True, "count": store.count(),
                    "health": store.health_summary(),
                    "store": str(store.path)}
        if action == "health":
            ref = str(kwargs.get("id", "") or "").strip()
            if ref:
                station = _find(store, ref)
                if station is None:
                    return {"ok": False,
                            "error": f"no station matches {ref!r}"}
                result = check_station(store, station.id)
                if result is None:
                    return {"ok": False, "error": "station not found"}
                status, detail = result
                return {"ok": status == "up", "station": station.name,
                        "health": status, "detail": detail}
            return {"ok": True, **check_all(store)}
        if action == "grow":
            try:
                limit = int(kwargs.get("limit", 50))
            except (TypeError, ValueError):
                limit = 50
            return {"ok": True,
                    **grow_from_directory(store, limit=limit)}
        if action == "globe":
            return {"ok": True, "dots": globe_dots(store)}
        return {"ok": False,
                "error": "unknown action — stations | play | list | "
                         "status | health | grow | globe"}
