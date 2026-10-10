"""Weather awareness, USA situations overview, and a timezone-aware
knowledge feed for social chat.

Keyless by design: Open-Meteo (geocoding + forecast, no key) and the US
National Weather Service alerts API (no key, US only).  Everything is
stdlib + urllib — no new dependencies.

Pieces:
- :class:`Weather` — current conditions, daily forecast, severe-weather
  alerts for any place name on earth.
- :class:`USASituations` — the national picture: active NWS alerts grouped
  by event and severity, a 4-city weather snapshot, optional USA-filtered
  news items folded in.
- Timezone utilities — owner-tz resolution, tz-aware formatting with
  abbreviations (EDT, …), conversion between zones, DST-safe via
  :mod:`zoneinfo`.
- :class:`KnowledgeFeed` — small, timestamped, TTL-cached chat-ready
  briefs (weather-now, today-outlook, usa-situation, tz-note) the
  social/partner layer pulls on demand.
- :class:`WeatherProvider` / :class:`USASituationsProvider` — morning
  briefing section providers (wired into ``BriefingComposer`` lazily, so
  there is no import cycle).
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
# (timezone resolution lives in nomorals.core.tz.safe_zoneinfo — it never
# raises, even when the tz database is absent, e.g. Termux without tzdata)

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

UA = "DevonWeather/1.0 (devon)"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
NWS_ALERTS_URL = "https://api.weather.gov/alerts/active"

#: Fallback weather sources (keyless). Open-Meteo is primary; these kick in
#: when it's unreachable. All return current + daily in a normalized shape.
WTTR_URL = "https://wttr.in"
METNO_URL = "https://api.met.no/weatherapi/locationforecast/2.0/complete"
METNO_UA = "DevonWeather/1.0 (github.com/Oluwacutyp/nomorals-2.0)"

#: WMO weather-code → (plain description, emoji)
WMO: dict[int, tuple[str, str]] = {
    0: ("Clear sky", "☀️"),
    1: ("Mainly clear", "🌤️"),
    2: ("Partly cloudy", "⛅"),
    3: ("Overcast", "☁️"),
    45: ("Fog", "🌫️"),
    48: ("Rime fog", "🌫️"),
    51: ("Light drizzle", "🌦️"),
    53: ("Drizzle", "🌦️"),
    55: ("Dense drizzle", "🌧️"),
    56: ("Freezing drizzle", "🌧️"),
    57: ("Dense freezing drizzle", "🌧️"),
    61: ("Light rain", "🌧️"),
    63: ("Rain", "🌧️"),
    65: ("Heavy rain", "⛈️"),
    66: ("Freezing rain", "🌧️"),
    67: ("Heavy freezing rain", "⛈️"),
    71: ("Light snow", "🌨️"),
    73: ("Snow", "❄️"),
    75: ("Heavy snow", "❄️"),
    77: ("Snow grains", "🌨️"),
    80: ("Light showers", "🌦️"),
    81: ("Showers", "🌧️"),
    82: ("Violent showers", "⛈️"),
    85: ("Light snow showers", "🌨️"),
    86: ("Snow showers", "❄️"),
    95: ("Thunderstorm", "⛈️"),
    96: ("Thunderstorm with hail", "⛈️"),
    99: ("Thunderstorm with heavy hail", "⛈️"),
}

SEVERITY_RANK = {"Extreme": 4, "Severe": 3, "Moderate": 2, "Minor": 1,
                 "Unknown": 0}

#: major US hubs for the national snapshot (coords hardcoded — no geocode
#: call needed)
US_CITIES: tuple[tuple[str, float, float, str], ...] = (
    ("New York", 40.71, -74.00, "America/New_York"),
    ("Chicago", 41.88, -87.63, "America/Chicago"),
    ("Houston", 29.76, -95.37, "America/Chicago"),
    ("Los Angeles", 34.05, -118.24, "America/Los_Angeles"),
)

USA_KEYWORDS = (
    "u.s.", "u.s", "usa", "united states", "white house", "congress",
    "senate", "supreme court", "pentagon", "federal reserve",
)


# ── http ─────────────────────────────────────────────────────────────────

def _get_json(url: str, params: dict[str, Any] | None = None,
              timeout: float = 20.0) -> dict[str, Any]:
    target = url
    if params:
        sep = "&" if urllib.parse.urlparse(url).query else "?"
        target = f"{target}{sep}{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(target, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


# ── geocoding ────────────────────────────────────────────────────────────

@dataclass
class Place:
    name: str
    latitude: float
    longitude: float
    timezone: str = "UTC"
    country: str = ""
    admin1: str = ""

    @property
    def label(self) -> str:
        bits = [self.name]
        if self.admin1:
            bits.append(self.admin1)
        if self.country:
            bits.append(self.country)
        return ", ".join(bits)


def parse_geocode(payload: dict[str, Any]) -> Place | None:
    results = payload.get("results") or []
    if not results:
        return None
    r = results[0]
    return Place(
        name=str(r.get("name", "?")),
        latitude=float(r.get("latitude", 0.0)),
        longitude=float(r.get("longitude", 0.0)),
        timezone=str(r.get("timezone") or "UTC"),
        country=str(r.get("country") or ""),
        admin1=str(r.get("admin1") or ""),
    )


def geocode(place: str, timeout: float = 20.0) -> Place | None:
    """City name → coordinates.  None when Open-Meteo knows nothing."""
    try:
        payload = _get_json(GEOCODE_URL, {
            "name": place, "count": 1, "language": "en", "format": "json",
        }, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — network is best-effort
        _log.debug("geocode failed for %r: %s", place, exc)
        return None
    return parse_geocode(payload)


# ── forecast ─────────────────────────────────────────────────────────────

@dataclass
class CurrentConditions:
    temp: float
    feels_like: float | None
    humidity: int | None
    weather_code: int
    wind_speed: float | None
    wind_direction: float | None
    observed_at: str = ""
    units: str = "imperial"

    @property
    def description(self) -> str:
        return WMO.get(self.weather_code, ("Unknown", "🌡️"))[0]

    @property
    def emoji(self) -> str:
        return WMO.get(self.weather_code, ("Unknown", "🌡️"))[1]

    @property
    def temp_unit(self) -> str:
        return "°F" if self.units == "imperial" else "°C"

    @property
    def wind_unit(self) -> str:
        return "mph" if self.units == "imperial" else "km/h"

    def one_liner(self, place_label: str) -> str:
        feels = (f", feels like {self.feels_like:.0f}{self.temp_unit}"
                 if self.feels_like is not None else "")
        hum = f", humidity {self.humidity}%" if self.humidity is not None else ""
        wind = (f", wind {self.wind_speed:.0f} {self.wind_unit}"
                if self.wind_speed is not None else "")
        return (f"{self.emoji} {place_label}: {self.description}, "
                f"{self.temp:.0f}{self.temp_unit}{feels}{hum}{wind}")


@dataclass
class DayForecast:
    date: str
    weather_code: int
    temp_max: float
    temp_min: float
    precip_prob: int | None
    units: str = "imperial"

    @property
    def description(self) -> str:
        return WMO.get(self.weather_code, ("Unknown", "🌡️"))[0]

    @property
    def emoji(self) -> str:
        return WMO.get(self.weather_code, ("Unknown", "🌡️"))[1]

    def one_liner(self) -> str:
        unit = "°F" if self.units == "imperial" else "°C"
        pp = (f", {self.precip_prob}% precip" if self.precip_prob else "")
        return (f"{self.emoji} {self.date}: {self.description}, "
                f"{self.temp_max:.0f}/{self.temp_min:.0f}{unit}{pp}")


def parse_current(payload: dict[str, Any],
                  units: str = "imperial") -> CurrentConditions | None:
    cur = payload.get("current") or {}
    if "temperature_2m" not in cur:
        return None
    return CurrentConditions(
        temp=float(cur["temperature_2m"]),
        feels_like=(float(cur["apparent_temperature"])
                    if cur.get("apparent_temperature") is not None else None),
        humidity=(int(cur["relative_humidity_2m"])
                  if cur.get("relative_humidity_2m") is not None else None),
        weather_code=int(cur.get("weather_code", -1)),
        wind_speed=(float(cur["wind_speed_10m"])
                    if cur.get("wind_speed_10m") is not None else None),
        wind_direction=(float(cur["wind_direction_10m"])
                        if cur.get("wind_direction_10m") is not None
                        else None),
        observed_at=str(cur.get("time", "")),
        units=units,
    )


def parse_daily(payload: dict[str, Any],
                units: str = "imperial") -> list[DayForecast]:
    daily = payload.get("daily") or {}
    dates = daily.get("time") or []
    codes = daily.get("weather_code") or []
    tmax = daily.get("temperature_2m_max") or []
    tmin = daily.get("temperature_2m_min") or []
    pp = daily.get("precipitation_probability_max") or []
    out = []
    for i, date in enumerate(dates):
        out.append(DayForecast(
            date=str(date),
            weather_code=int(codes[i]) if i < len(codes) else -1,
            temp_max=float(tmax[i]) if i < len(tmax) else 0.0,
            temp_min=float(tmin[i]) if i < len(tmin) else 0.0,
            precip_prob=(int(pp[i]) if i < len(pp) and pp[i] is not None
                         else None),
            units=units,
        ))
    return out


def fetch_forecast(lat: float, lon: float, days: int = 3,
                   units: str = "imperial",
                   timeout: float = 20.0) -> dict[str, Any] | None:
    """Current + daily forecast with automatic source fallback.

    Tries Open-Meteo → wttr.in → met.no. Returns None only when all
    sources fail. Each fallback normalizes to the same shape.
    """
    result = _fetch_openmeteo(lat, lon, days, units, timeout)
    if result is not None:
        return result
    _log.debug("open-meteo failed, trying wttr.in for %s,%s", lat, lon)
    result = _fetch_wttr(lat, lon, days, units, timeout)
    if result is not None:
        return result
    _log.debug("wttr.in failed, trying met.no for %s,%s", lat, lon)
    return _fetch_metno(lat, lon, days, units, timeout)


def _fetch_openmeteo(lat: float, lon: float, days: int,
                     units: str, timeout: float) -> dict[str, Any] | None:
    params = {
        "latitude": lat, "longitude": lon,
        "current": ("temperature_2m,relative_humidity_2m,apparent_temperature,"
                    "weather_code,wind_speed_10m,wind_direction_10m"),
        "daily": ("weather_code,temperature_2m_max,temperature_2m_min,"
                  "precipitation_probability_max"),
        "timezone": "auto",
        "forecast_days": max(1, min(days, 7)),
    }
    if units == "imperial":
        params["temperature_unit"] = "fahrenheit"
        params["wind_speed_unit"] = "mph"
    try:
        payload = _get_json(FORECAST_URL, params, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        _log.debug("forecast failed for %s,%s: %s", lat, lon, exc)
        return None
    return {
        "timezone": str(payload.get("timezone") or "UTC"),
        "current": parse_current(payload, units),
        "daily": parse_daily(payload, units),
        "source": "open-meteo",
    }


def _fetch_wttr(lat: float, lon: float, days: int,
                units: str, timeout: float) -> dict[str, Any] | None:
    """wttr.in fallback — keyless JSON API."""
    try:
        payload = _get_json(f"{WTTR_URL}/{lat},{lon}", {"format": "j1"},
                            timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        _log.debug("wttr.in failed for %s,%s: %s", lat, lon, exc)
        return None
    try:
        current_list = payload.get("current_condition") or []
        if not current_list:
            return None
        cc = current_list[0]
        # wttr.in gives both C and F; pick based on units
        temp_key = "temp_F" if units == "imperial" else "temp_C"
        feels_key = "FeelsLikeF" if units == "imperial" else "FeelsLikeC"
        weather_desc = (cc.get("weatherDesc") or [{}])[0].get("value", "")
        # Map wttr.in weather code to WMO-ish code for emoji lookup
        wttr_code = int(cc.get("weatherCode") or 0)
        wmo = _wttr_to_wmo(wttr_code)
        current = {
            "temperature": float(cc.get(temp_key) or 0),
            "feels_like": float(cc.get(feels_key) or 0),
            "humidity": int(cc.get("humidity") or 0),
            "weather_code": wmo,
            "description": weather_desc,
            "wind_speed": float(cc.get("windspeedMiles" if units == "imperial"
                                      else "windspeedKmph") or 0),
            "wind_direction": int(cc.get("winddirDegree") or 0),
        }
        daily = []
        for w in (payload.get("weather") or [])[:days]:
            tmax_key = "maxtempF" if units == "imperial" else "maxtempC"
            tmin_key = "mintempF" if units == "imperial" else "mintempC"
            hourly = w.get("hourly") or []
            # Average chance of rain across hourly slots
            chances = [int(h.get("chanceofrain") or 0) for h in hourly]
            avg_rain = sum(chances) // len(chances) if chances else 0
            daily.append({
                "date": str(w.get("date", "")),
                "temp_max": float(w.get(tmax_key) or 0),
                "temp_min": float(w.get(tmin_key) or 0),
                "precipitation_probability": avg_rain,
                "weather_code": _wttr_to_wmo(int((hourly[4] if len(hourly) > 4
                                                  else {}).get("weatherCode") or 0)),
            })
        tz_name = str((payload.get("nearest_area") or [{}])[0].get("timezone")
                      or "UTC")
        return {
            "timezone": tz_name,
            "current": current,
            "daily": daily,
            "source": "wttr.in",
        }
    except Exception as exc:  # noqa: BLE001
        _log.debug("wttr.in parse failed: %s", exc)
        return None


def _wttr_to_wmo(wttr_code: int) -> int:
    """Map wttr.in weather codes to WMO codes for emoji lookup."""
    mapping = {
        113: 0,    # Sunny → Clear sky
        116: 1,    # Partly cloudy → Mainly clear
        119: 3,    # Cloudy → Overcast
        122: 3,    # Overcast → Overcast
        143: 45,   # Mist → Fog
        176: 80,   # Patchy rain → Light showers
        179: 71,   # Patchy snow → Light snow
        182: 66,   # Patchy sleet → Freezing rain
        185: 56,   # Patchy freezing drizzle → Freezing drizzle
        200: 95,   # Thundery outbreaks → Thunderstorm
        227: 75,   # Blowing snow → Heavy snow
        230: 75,   # Blizzard → Heavy snow
        248: 45,   # Fog → Fog
        260: 48,   # Freezing fog → Rime fog
        263: 51,   # Patchy light drizzle → Light drizzle
        266: 53,   # Light drizzle → Drizzle
        281: 57,   # Freezing drizzle → Dense freezing drizzle
        284: 67,   # Heavy freezing drizzle → Heavy freezing rain
        293: 51,   # Patchy light rain → Light drizzle
        296: 61,   # Light rain → Light rain
        299: 81,   # Moderate rain at times → Showers
        302: 63,   # Moderate rain → Rain
        305: 82,   # Heavy rain at times → Violent showers
        308: 65,   # Heavy rain → Heavy rain
        311: 66,   # Light freezing rain → Freezing rain
        314: 67,   # Moderate/heavy freezing rain → Heavy freezing rain
        317: 71,   # Light sleet → Light snow
        320: 73,   # Moderate/heavy sleet → Snow
        323: 71,   # Patchy light snow → Light snow
        326: 73,   # Light snow → Snow
        329: 75,   # Patchy moderate snow → Heavy snow
        332: 75,   # Moderate snow → Heavy snow
        335: 75,   # Patchy heavy snow → Heavy snow
        338: 75,   # Heavy snow → Heavy snow
        350: 77,   # Ice pellets → Snow grains
        353: 80,   # Light rain shower → Light showers
        356: 81,   # Moderate/heavy rain shower → Showers
        359: 82,   # Torrential rain shower → Violent showers
        362: 66,   # Light sleet showers → Freezing rain
        365: 73,   # Moderate/heavy sleet showers → Snow
        368: 85,   # Light snow showers → Light snow showers
        371: 86,   # Moderate/heavy snow showers → Snow showers
        374: 77,   # Light showers of ice pellets → Snow grains
        377: 77,   # Moderate/heavy showers of ice pellets → Snow grains
        386: 95,   # Patchy light rain with thunder → Thunderstorm
        389: 95,   # Moderate/heavy rain with thunder → Thunderstorm
        392: 96,   # Patchy light snow with thunder → Thunderstorm with hail
        395: 99,   # Moderate/heavy snow with thunder → Thunderstorm heavy hail
    }
    return mapping.get(wttr_code, 3)  # default to overcast


def _fetch_metno(lat: float, lon: float, days: int,
                 units: str, timeout: float) -> dict[str, Any] | None:
    """Norwegian Met Institute fallback — keyless, government infrastructure."""
    try:
        req = urllib.request.Request(
            f"{METNO_URL}?lat={lat}&lon={lon}",
            headers={"User-Agent": METNO_UA},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception as exc:  # noqa: BLE001
        _log.debug("met.no failed for %s,%s: %s", lat, lon, exc)
        return None
    try:
        timeseries = (payload.get("properties") or {}).get("timeseries") or []
        if not timeseries:
            return None
        # Current = first entry
        first = timeseries[0]
        details = (first.get("data") or {}).get("instant", {}).get("details", {})
        temp_c = float(details.get("air_temperature") or 0)
        current = {
            "temperature": temp_c * 9/5 + 32 if units == "imperial" else temp_c,
            "feels_like": temp_c * 9/5 + 32 if units == "imperial" else temp_c,
            "humidity": int(details.get("relative_humidity") or 0),
            "weather_code": 3,  # met.no uses symbol codes, not WMO
            "description": str((first.get("data") or {}).get("next_1_hours", {})
                               .get("summary", {}).get("symbol_code", "")
                               .replace("_", " ")),
            "wind_speed": float(details.get("wind_speed") or 0),
            "wind_direction": int(details.get("wind_from_direction") or 0),
        }
        # Daily = group by date, take min/max
        by_date: dict[str, list[float]] = {}
        for entry in timeseries:
            ts = str(entry.get("time", ""))[:10]
            d = (entry.get("data") or {}).get("instant", {}).get("details", {})
            t = d.get("air_temperature")
            if t is not None:
                by_date.setdefault(ts, []).append(float(t))
        daily = []
        for date_str in sorted(by_date)[:days]:
            temps = by_date[date_str]
            tmax = max(temps)
            tmin = min(temps)
            if units == "imperial":
                tmax = tmax * 9/5 + 32
                tmin = tmin * 9/5 + 32
            daily.append({
                "date": date_str,
                "temp_max": tmax,
                "temp_min": tmin,
                "precipitation_probability": 0,  # met.no needs separate calc
                "weather_code": 3,
            })
        return {
            "timezone": "UTC",  # met.no doesn't provide tz in this endpoint
            "current": current,
            "daily": daily,
            "source": "met.no",
        }
    except Exception as exc:  # noqa: BLE001
        _log.debug("met.no parse failed: %s", exc)
        return None


# ── NWS alerts ───────────────────────────────────────────────────────────

@dataclass
class WeatherAlert:
    event: str
    severity: str
    headline: str
    areas: str
    onset: str = ""
    ends: str = ""
    description: str = ""
    instruction: str = ""
    sender: str = ""

    @property
    def rank(self) -> int:
        return SEVERITY_RANK.get(self.severity, 0)

    def one_liner(self) -> str:
        when = ""
        if self.onset or self.ends:
            when = f" ({self.onset or '?'} → {self.ends or '?'})"
        areas = f" — {self.areas}" if self.areas else ""
        return f"⚠️ {self.event} [{self.severity}]{areas}{when}"


def parse_nws_alerts(payload: dict[str, Any]) -> list[WeatherAlert]:
    out = []
    for feat in payload.get("features") or []:
        p = feat.get("properties") or {}
        # skip tests / cancels / administrative noise
        if str(p.get("messageType", "")).lower() not in ("alert", "update"):
            continue
        if str(p.get("event", "")).lower() in ("test message",):
            continue
        out.append(WeatherAlert(
            event=str(p.get("event") or "Weather alert"),
            severity=str(p.get("severity") or "Unknown"),
            headline=str(p.get("headline") or ""),
            areas=str(p.get("areaDesc") or ""),
            onset=str(p.get("onset") or ""),
            ends=str(p.get("ends") or ""),
            description=str(p.get("description") or "")[:600],
            instruction=str(p.get("instruction") or "")[:400],
            sender=str(p.get("senderName") or ""),
        ))
    out.sort(key=lambda a: (-a.rank, a.event))
    return out


def fetch_nws_alerts(lat: float | None = None, lon: float | None = None,
                     timeout: float = 20.0) -> list[WeatherAlert] | None:
    """Active NWS alerts — for a point, or nationwide when no coords."""
    params = {}
    if lat is not None and lon is not None:
        params["point"] = f"{lat},{lon}"
    try:
        payload = _get_json(NWS_ALERTS_URL, params or None, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        _log.debug("NWS alerts failed: %s", exc)
        return None
    return parse_nws_alerts(payload)


# ── Weather facade ───────────────────────────────────────────────────────

class Weather:
    """One-call weather for a place name.  All methods return plain dicts
    with a chat-ready ``text`` field; ``None`` data means the network
    failed, never an exception."""

    def __init__(self, default_place: str = "New York") -> None:
        self.default_place = default_place

    def _place(self, place: str | None) -> Place | None:
        return geocode(place or self.default_place)

    def now(self, place: str | None = None,
            units: str = "imperial") -> dict[str, Any]:
        p = self._place(place)
        if p is None:
            return {"ok": False, "error": "place not found",
                    "text": f"couldn't find {place or self.default_place!r}"}
        fc = fetch_forecast(p.latitude, p.longitude, days=1, units=units)
        if fc is None or fc["current"] is None:
            return {"ok": False, "error": "forecast unavailable",
                    "text": f"weather service unreachable for {p.label}"}
        cur: CurrentConditions = fc["current"]
        alerts = self.alerts_for(p)
        text = cur.one_liner(p.label)
        if alerts:
            top = alerts[0]
            text += f"\n{top.one_liner()}"
            if len(alerts) > 1:
                text += f" (+{len(alerts) - 1} more alerts)"
        return {"ok": True, "place": p.label,
                "timezone": fc["timezone"],
                "current": {"temp": cur.temp, "feels_like": cur.feels_like,
                            "humidity": cur.humidity,
                            "description": cur.description,
                            "emoji": cur.emoji,
                            "wind_speed": cur.wind_speed,
                            "observed_at": cur.observed_at},
                "alerts": [a.one_liner() for a in alerts[:5]],
                "text": text}

    def forecast(self, place: str | None = None, days: int = 3,
                 units: str = "imperial") -> dict[str, Any]:
        p = self._place(place)
        if p is None:
            return {"ok": False, "error": "place not found",
                    "text": f"couldn't find {place or self.default_place!r}"}
        fc = fetch_forecast(p.latitude, p.longitude, days=days, units=units)
        if fc is None:
            return {"ok": False, "error": "forecast unavailable",
                    "text": f"weather service unreachable for {p.label}"}
        lines = [d.one_liner() for d in fc["daily"]]
        return {"ok": True, "place": p.label, "timezone": fc["timezone"],
                "days": [{"date": d.date, "description": d.description,
                          "temp_max": d.temp_max, "temp_min": d.temp_min,
                          "precip_prob": d.precip_prob} for d in fc["daily"]],
                "text": f"📅 {p.label} — {days}-day outlook:\n"
                        + "\n".join(f"• {ln}" for ln in lines)}

    def alerts_for(self, place: Place) -> list[WeatherAlert]:
        """NWS alerts for a US place; [] for non-US or on failure."""
        if (place.country or "").lower() not in (
                "united states", "united states of america", "usa", ""):
            # geocoding sometimes omits country — still try the point query
            # for anything plausibly in the US lon/lat box.
            if not (-130 <= place.longitude <= -65
                    and 22 <= place.latitude <= 50):
                return []
        alerts = fetch_nws_alerts(place.latitude, place.longitude)
        return alerts or []


# ── USA situations ───────────────────────────────────────────────────────

class USASituations:
    """The national picture: severe-weather alerts grouped by event,
    a 4-city snapshot, optional USA-filtered news folded in."""

    def __init__(self, weather: Weather | None = None) -> None:
        self.weather = weather or Weather()

    def overview(self, news_items: list[dict[str, Any]] | None = None
                 ) -> dict[str, Any]:
        alerts = fetch_nws_alerts() or []
        # group by event: keep the worst severity, count areas
        grouped: dict[str, dict[str, Any]] = {}
        for a in alerts:
            g = grouped.setdefault(a.event, {
                "event": a.event, "worst": a.severity, "rank": a.rank,
                "count": 0, "sample_areas": "", "sample_headline": ""})
            g["count"] += 1
            if a.rank > SEVERITY_RANK.get(g["worst"], 0):
                g["worst"] = a.severity
                g["rank"] = a.rank
                g["sample_headline"] = a.headline
            if not g["sample_areas"] and a.areas:
                g["sample_areas"] = a.areas[:120]
        top = sorted(grouped.values(), key=lambda g: (-g["rank"], -g["count"]))
        cities = []
        for name, lat, lon, _tz in US_CITIES:
            fc = fetch_forecast(lat, lon, days=1)
            if fc and fc["current"]:
                cities.append({"city": name,
                               "line": fc["current"].one_liner(name)})
        lines = ["🇺🇸 USA right now:"]
        if top:
            lines.append(f"• {len(alerts)} active weather alerts nationwide")
            for g in top[:5]:
                lines.append(f"• ⚠️ {g['event']} [{g['worst']}] ×{g['count']}"
                             + (f" — {g['sample_areas']}"
                                if g["sample_areas"] else ""))
        else:
            lines.append("• no active NWS weather alerts nationwide")
        if cities:
            lines.append("• " + " · ".join(
                c["line"].split(": ", 1)[-1] if ": " in c["line"] else c["line"]
                for c in cities))
        usa_news = [n for n in (news_items or [])
                    if self._is_usa(n)] if news_items else []
        for n in usa_news[:3]:
            lines.append(f"• 📰 {n.get('title', '')}")
        text = "\n".join(lines)
        return {"ok": True, "alert_count": len(alerts),
                "top_events": top[:8], "cities": cities,
                "usa_news": [n.get("title", "") for n in usa_news[:5]],
                "text": text}

    @staticmethod
    def _is_usa(item: dict[str, Any]) -> bool:
        blob = f"{item.get('title', '')} {item.get('summary', '')}".lower()
        return any(k in blob for k in USA_KEYWORDS)


# ── timezone utilities ───────────────────────────────────────────────────

def owner_tz(settings: Any = None) -> str:
    """Owner's IANA timezone: settings → partner → TZ env → UTC."""
    for obj in (settings, getattr(settings, "partner", None)):
        if obj is None:
            continue
        for attr in ("timezone", "tz", "owner_timezone"):
            tz = getattr(obj, attr, None)
            if tz:
                return str(tz)
    import os
    return os.environ.get("TZ", "UTC")


def _zone(name: str):
    """tzinfo for ``name``; never raises — falls back to fixed UTC when the
    tz database is absent (e.g. Termux without tzdata)."""
    from ..core.tz import safe_zoneinfo

    return safe_zoneinfo(name)


def now_in_tz(tz_name: str) -> datetime:
    return datetime.now(_zone(tz_name))


def tz_abbr(moment: datetime) -> str:
    """'EDT', 'PST', … — falls back to the UTC offset."""
    abbr = moment.strftime("%Z")
    if abbr:
        return abbr
    off = moment.strftime("%z")
    return f"UTC{off[:3]}:{off[3:]}" if off else "UTC"


def format_with_tz(moment: datetime, tz_name: str,
                   fmt: str = "%Y-%m-%d %H:%M") -> str:
    """Render a moment in a zone, labeled: '2026-10-01 10:30 EDT'."""
    zoned = moment.astimezone(_zone(tz_name))
    return f"{zoned.strftime(fmt)} {tz_abbr(zoned)}"


def convert_time(when: Any, from_tz: str, to_tz: str) -> dict[str, Any]:
    """Convert a time between zones.  ``when``: 'YYYY-MM-DD HH:MM',
    ISO-8601, unix ts, or '@ts'.  Returns labeled strings both ends."""
    moment = _parse_when(when)
    if moment is None:
        return {"ok": False,
                "error": f"couldn't parse time {when!r} "
                         "(try 'YYYY-MM-DD HH:MM', ISO, or unix ts)"}
    src = _zone(from_tz)
    dst = _zone(to_tz)
    if moment.tzinfo is None:
        # Ambiguous local times (fall-back hour): fold=0 = first occurrence.
        moment = moment.replace(tzinfo=src)
    converted = moment.astimezone(dst)
    return {
        "ok": True,
        "from": format_with_tz(moment, from_tz),
        "to": format_with_tz(converted, to_tz),
        "from_tz": from_tz, "to_tz": to_tz,
        "text": f"🕐 {format_with_tz(moment, from_tz)} → "
                f"{format_with_tz(converted, to_tz)}",
    }


def _parse_when(when: Any) -> datetime | None:
    if isinstance(when, (int, float)):
        try:
            return datetime.fromtimestamp(float(when), timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    s = str(when or "").strip()
    if s.startswith("@"):
        return _parse_when(s[1:])
    if s.replace(".", "", 1).isdigit():
        return _parse_when(float(s))
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
                "%Y/%m/%d %H:%M", "%m/%d/%Y %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    try:
        iso = s.replace("Z", "+00:00")
        return datetime.fromisoformat(iso)
    except ValueError:
        return None


def tz_note(tz_name: str) -> str:
    """One-liner for the knowledge feed: local time + offset + DST state."""
    now = now_in_tz(tz_name)
    dst = " (DST)" if now.dst() else ""
    return (f"🕐 {now.strftime('%Y-%m-%d %H:%M')} {tz_abbr(now)}{dst} "
            f"— {tz_name}")


# ── knowledge feed ───────────────────────────────────────────────────────

@dataclass
class FeedEntry:
    kind: str
    title: str
    body: str
    at: str          #: ISO timestamp of generation
    tz: str          #: zone the entry is labeled in
    tz_abbr: str = ""

    def chat_line(self) -> str:
        stamp = f" [{self.tz_abbr}]" if self.tz_abbr else ""
        return f"{self.title}{stamp}\n{self.body}"


class KnowledgeFeed:
    """Chat-ready briefs with per-kind TTL caching.  The social/partner
    layer pulls on demand; a failed source yields a short note, never
    an exception."""

    TTL = {"weather-now": 1800, "today-outlook": 3600,
           "usa-situation": 3600, "tz-note": 86400}

    def __init__(self, weather: Weather | None = None,
                 default_place: str = "New York",
                 tz_name: str = "UTC") -> None:
        self.weather = weather or Weather(default_place)
        self.default_place = default_place
        self.tz_name = tz_name
        self._cache: dict[str, tuple[float, FeedEntry]] = {}

    def _stamp(self) -> tuple[str, str]:
        now = now_in_tz(self.tz_name)
        return now.isoformat(timespec="seconds"), tz_abbr(now)

    def _fresh(self, kind: str) -> FeedEntry | None:
        hit = self._cache.get(kind)
        if hit and time.time() < hit[0]:
            return hit[1]
        return None

    def _store(self, kind: str, entry: FeedEntry) -> FeedEntry:
        self._cache[kind] = (time.time() + self.TTL[kind], entry)
        return entry

    def pull(self, kinds: list[str] | None = None) -> list[FeedEntry]:
        kinds = kinds or list(self.TTL)
        out = []
        for kind in kinds:
            if kind not in self.TTL:
                continue
            hit = self._fresh(kind)
            if hit is not None:
                out.append(hit)
                continue
            try:
                out.append(self.refresh(kind))
            except Exception as exc:  # noqa: BLE001 — feed never raises
                _log.debug("knowledge feed %s failed: %s", kind, exc)
                at, abbr = self._stamp()
                out.append(FeedEntry(kind, kind.replace("-", " ").title(),
                                     "unavailable right now", at,
                                     self.tz_name, abbr))
        return out

    def refresh(self, kind: str) -> FeedEntry:
        at, abbr = self._stamp()
        if kind == "weather-now":
            res = self.weather.now(self.default_place)
            entry = FeedEntry(kind, "🌤️ Weather now",
                              res.get("text", "unavailable"),
                              at, self.tz_name, abbr)
        elif kind == "today-outlook":
            res = self.weather.forecast(self.default_place, days=1)
            entry = FeedEntry(kind, "📅 Today",
                              res.get("text", "unavailable"),
                              at, self.tz_name, abbr)
        elif kind == "usa-situation":
            res = USASituations(self.weather).overview()
            entry = FeedEntry(kind, "🇺🇸 USA right now",
                              res.get("text", "unavailable"),
                              at, self.tz_name, abbr)
        elif kind == "tz-note":
            entry = FeedEntry(kind, "🕐 Time", tz_note(self.tz_name),
                              at, self.tz_name, abbr)
        else:
            raise ValueError(f"unknown feed kind {kind!r}")
        return self._store(kind, entry)

    def invalidate(self, kind: str | None = None) -> None:
        if kind:
            self._cache.pop(kind, None)
        else:
            self._cache.clear()


# ── briefing providers ───────────────────────────────────────────────────

try:
    from .morning_briefing import _Provider, BriefingSection
except Exception:  # noqa: BLE001 — standalone use without the briefing
    class _Provider:  # type: ignore[no-redef]
        name = "base"
        title = "Base"
        priority = 999

    @dataclass
    class BriefingSection:  # type: ignore[no-redef]
        name: str
        title: str
        priority: int
        source: str = ""
        items: list[dict[str, Any]] = field(default_factory=list)
        lines: list[str] = field(default_factory=list)


def _weather_place(ctx: Any) -> str:
    settings = getattr(ctx, "settings", None)
    w = getattr(settings, "weather", None) if settings else None
    loc = getattr(w, "location", None) if w else None
    if loc:
        return str(loc)
    import os
    return os.environ.get("DEVON_WEATHER_PLACE", "New York")


class WeatherProvider(_Provider):
    """Owner's local weather: now + today's outlook + any alerts."""
    name = "weather"
    title = "Weather"
    priority = 25

    def collect(self, ctx: Any, since: float) -> Any | None:
        place = _weather_place(ctx)
        w = Weather(default_place=place)
        lines: list[str] = []
        try:
            now = w.now(place)
        except Exception:  # noqa: BLE001
            return None
        if not now.get("ok"):
            return None
        lines.append(f"• {now['text']}".replace("\n", " — "))
        try:
            fc = w.forecast(place, days=1)
            if fc.get("ok"):
                for ln in fc["text"].splitlines()[1:2]:
                    lines.append(f"• {ln.lstrip('• ')}")
        except Exception:  # noqa: BLE001
            pass
        return BriefingSection(
            name=self.name, title=self.title, priority=self.priority,
            source="open-meteo/nws",
            items=[{"id": "wx-now", "title": now["text"][:120],
                    "body": now["text"][:300]}],
            lines=lines[:4])


class USASituationsProvider(_Provider):
    """National picture: NWS alerts + 4-city snapshot + USA news."""
    name = "usa"
    title = "USA right now"
    priority = 55

    def collect(self, ctx: Any, since: float) -> Any | None:
        news_items: list[dict[str, Any]] = []
        try:
            from .news import NewsAgent
            agent = NewsAgent(ctx)
            for i in agent.recent(limit=40):
                news_items.append({"title": i.get("title", ""),
                                   "summary": i.get("summary", "")})
        except Exception:  # noqa: BLE001 — news optional
            pass
        try:
            res = USASituations().overview(
                news_items=news_items or None)
        except Exception:  # noqa: BLE001
            return None
        if not res.get("ok"):
            return None
        lines = [f"• {ln.lstrip('• ')}"
                 for ln in res["text"].splitlines()[1:8]]
        if not lines:
            return None
        return BriefingSection(
            name=self.name, title=self.title, priority=self.priority,
            source="nws/open-meteo",
            items=[{"id": f"usa-{n}", "title": t,
                    "body": t} for n, t in enumerate(
                        [g["event"] for g in res["top_events"][:5]])],
            lines=lines)
