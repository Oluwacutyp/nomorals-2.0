"""The ``weather`` tool: agent-callable weather, alerts, USA situations,
timezone-aware knowledge feed, and tz conversion.

Actions: now | forecast | alerts | usa | feed | tz | convert | status.
Keyless: Open-Meteo + api.weather.gov — no API keys involved at all.
"""

from __future__ import annotations

from typing import Any

from ..agents.weather import (
    KnowledgeFeed,
    USASituations,
    Weather,
    convert_time,
    now_in_tz,
    owner_tz,
    tz_abbr,
    tz_note,
)


def _settings(registry: Any) -> Any:
    context = getattr(registry, "context", None)
    return getattr(context, "settings", None) if context else None


def _default_place(settings: Any) -> str:
    w = getattr(settings, "weather", None) if settings else None
    loc = getattr(w, "location", None) if w else None
    if loc:
        return str(loc)
    import os
    return os.environ.get("DEVON_WEATHER_PLACE", "New York")


def register(registry: Any) -> None:
    from ..core.policy import Capability

    @registry.register(
        "weather",
        description=(
            "Weather awareness + timezone-aware knowledge feed (keyless: "
            "Open-Meteo + US National Weather Service). Actions: "
            "now <place> (current conditions + alerts), "
            "forecast <place> [days] (daily outlook), "
            "alerts <place> (NWS severe-weather alerts, US), "
            "usa (national situations overview: alerts + 4-city snapshot), "
            "feed [kinds] (chat-ready briefs: weather-now, today-outlook, "
            "usa-situation, tz-note), "
            "tz [place] (local time + zone info), "
            "convert <time> <from_tz> <to_tz> (e.g. '2026-10-02 15:00' "
            "America/New_York Europe/London), "
            "status (sources + owner tz)."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "action": "str — now|forecast|alerts|usa|feed|tz|convert|status",
            "place": "str — city name (default: owner's weather location)",
            "days": "int — forecast days 1-7 (default 3)",
            "kinds": "str — comma list for feed (default all)",
            "time": "str — for convert: 'YYYY-MM-DD HH:MM', ISO, or unix ts",
            "from_tz": "str — IANA zone for convert",
            "to_tz": "str — IANA zone for convert",
        },
    )
    def weather(action: str = "now", place: str = "", days: int = 3,
                kinds: str = "", time: str = "", from_tz: str = "",
                to_tz: str = "", **_: Any) -> dict[str, Any]:
        action = (action or "now").strip().lower()
        settings = _settings(registry)
        tz_name = owner_tz(settings)
        w = Weather(default_place=_default_place(settings))

        if action == "now":
            return w.now(place or None)
        if action == "forecast":
            try:
                n = max(1, min(int(days or 3), 7))
            except (TypeError, ValueError):
                n = 3
            return w.forecast(place or None, days=n)
        if action == "alerts":
            from ..agents.weather import geocode
            p = geocode(place or _default_place(settings))
            if p is None:
                return {"ok": False, "error": "place not found",
                        "text": f"couldn't find {place!r}"}
            alerts = w.alerts_for(p)
            lines = [a.one_liner() for a in alerts[:10]]
            detail = ""
            if alerts and alerts[0].instruction:
                detail = f"\n📋 {alerts[0].instruction[:300]}"
            return {"ok": True, "place": p.label, "count": len(alerts),
                    "alerts": lines,
                    "text": (f"⚠️ {len(alerts)} active alert(s) for "
                             f"{p.label}:\n" + "\n".join(f"• {ln}" for ln in lines)
                             + detail) if lines
                    else f"✅ no active NWS alerts for {p.label}"}
        if action == "usa":
            return USASituations(w).overview()
        if action == "feed":
            feed = KnowledgeFeed(weather=w,
                                 default_place=_default_place(settings),
                                 tz_name=tz_name)
            want = [k.strip() for k in kinds.split(",") if k.strip()] or None
            entries = feed.pull(want)
            return {"ok": True, "tz": tz_name,
                    "entries": [{"kind": e.kind, "title": e.title,
                                 "body": e.body, "at": e.at,
                                 "tz_abbr": e.tz_abbr} for e in entries],
                    "text": "\n\n".join(e.chat_line() for e in entries)}
        if action == "tz":
            if place:
                from ..agents.weather import geocode
                p = geocode(place)
                zone = p.timezone if p else tz_name
                label = p.label if p else place
            else:
                zone, label = tz_name, "owner"
            now = now_in_tz(zone)
            return {"ok": True, "tz": zone, "abbr": tz_abbr(now),
                    "dst": bool(now.dst()),
                    "text": f"🕐 {tz_note(zone)}"
                            + (f" — {label}" if place else "")}
        if action == "convert":
            if not (time and from_tz and to_tz):
                return {"ok": False,
                        "error": "convert needs time, from_tz and to_tz"}
            return convert_time(time, from_tz, to_tz)
        if action == "status":
            return {"ok": True,
                    "sources": ["Open-Meteo forecast (keyless)",
                                "Open-Meteo geocoding (keyless)",
                                "api.weather.gov alerts (keyless, US)"],
                    "owner_tz": tz_name,
                    "default_place": _default_place(settings),
                    "feed_kinds": list(KnowledgeFeed.TTL)}
        raise ValueError(f"unknown weather action {action!r} "
                         "(want now|forecast|alerts|usa|feed|tz|convert|status)")
