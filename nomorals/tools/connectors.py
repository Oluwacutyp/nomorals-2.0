"""API connector framework: plug in external services cleanly.

A *connector* is a small, self-describing unit: name, parameter manifest,
and a ``call(params) -> dict`` that returns JSON-able data. Built-ins ship
keyless (open-meteo, ER-api, ipwho.is, Wikipedia REST, GitHub search with an
optional token) plus a generic ``http_api`` escape hatch for anything else.

Adding a service is one function:

    @connector("joke", description="...", params={"category": "str (optional)"})
    def _joke(params: dict, http: HttpClient) -> dict:
        return http.get("https://api.chucknorris.io/jokes/random").json()

The agent sees one tool — ``api_call`` — plus ``api_list`` for the manifest;
Devon gets a matching planning tool. New connectors appear in both places
automatically.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.errors import ToolError
from ..core.http import HttpClient
from ..core.logging_setup import get_logger
from ..core.policy import Capability

__all__ = ["Connector", "ConnectorRegistry", "connectors", "register"]

_log = get_logger(__name__)


# ── framework ────────────────────────────────────────────────────────────────


@dataclass
class Connector:
    """One external service, described well enough for a model to drive it."""

    name: str
    description: str
    fn: Callable[[dict[str, Any], HttpClient], dict[str, Any]]
    params: dict[str, str] = field(default_factory=dict)
    #: True when the connector needs the network (all built-ins do)
    needs_network: bool = True


class ConnectorRegistry:
    """The pluggable set. Module-level instance is the live one."""

    def __init__(self) -> None:
        self._connectors: dict[str, Connector] = {}

    def add(self, connector: Connector) -> None:
        self._connectors[connector.name] = connector

    def connector(self, name: str) -> Connector:
        try:
            return self._connectors[name]
        except KeyError:
            raise ToolError(
                f"unknown connector {name!r}; available: {', '.join(self.names())}"
            ) from None

    def register_connector(self, name: str, description: str = "", *,
                           params: dict[str, str] | None = None,
                           needs_network: bool = True) -> Callable:
        """Decorator: register a connector into THIS registry.

        The module-level ``@connector`` decorator registers into the shared
        ``connectors`` registry; use this one for isolated registries.
        """
        def deco(fn: Callable[[dict[str, Any], Any], dict[str, Any]]):
            self.add(Connector(
                name=name, description=description, fn=fn,
                params=params or {}, needs_network=needs_network,
            ))
            return fn
        return deco

    def names(self) -> list[str]:
        return sorted(self._connectors)

    def manifest(self) -> list[dict[str, Any]]:
        return [
            {"name": c.name, "description": c.description, "params": c.params}
            for c in (self._connectors[n] for n in self.names())
        ]

    def call(self, name: str, params: dict[str, Any], http: HttpClient) -> dict[str, Any]:
        connector = self.connector(name)
        unknown = set(params) - set(connector.params) - {"timeout"}
        if unknown:
            _log.debug("connector %s got unknown params %s", name, sorted(unknown))
        try:
            result = connector.fn(params, http)
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001 - a dead API is a result, not a crash
            raise ToolError(f"connector {name!r} failed: {exc}") from exc
        if not isinstance(result, dict):
            result = {"result": result}
        return result


connectors = ConnectorRegistry()


def connector(name: str, description: str, *, params: dict[str, str] | None = None,
              needs_network: bool = True) -> Callable:
    """Decorator: register a connector by function name override."""
    def deco(fn: Callable[[dict[str, Any], HttpClient], dict[str, Any]]):
        connectors.add(Connector(
            name=name, description=description, fn=fn,
            params=params or {}, needs_network=needs_network,
        ))
        return fn
    return deco


def _json_body(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except (ValueError, TypeError) as exc:
        raise ToolError(f"json body is not valid JSON: {exc}") from exc


def _http_error_check(response: Any) -> None:
    if not response.ok:
        snippet = (response.text or "")[:200]
        raise ToolError(f"API returned {response.status}: {snippet}")


# ── built-in connectors ──────────────────────────────────────────────────────


@connector(
    "weather",
    description=(
        "current weather for a city or lat/lon (open-meteo, no key): "
        "temperature, wind, humidity, condition"
    ),
    params={
        "city": "str (optional) — e.g. 'Lagos' (geocoded)",
        "lat": "float (optional) — latitude",
        "lon": "float (optional) — longitude",
    },
)
def _weather(params: dict[str, Any], http: HttpClient) -> dict[str, Any]:
    settings = http._nm_api_settings  # type: ignore[attr-defined]
    lat = params.get("lat") or getattr(settings, "weather_latitude", "")
    lon = params.get("lon") or getattr(settings, "weather_longitude", "")
    city = str(params.get("city") or "").strip()
    if (lat in ("", None)) or (lon in ("", None)):
        if not city:
            raise ToolError("weather needs a city or lat/lon")
        geo = http.get(f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1")
        _http_error_check(geo)
        results = (geo.json() or {}).get("results") or []
        if not results:
            raise ToolError(f"could not geocode {city!r}")
        top = results[0]
        lat, lon = top.get("latitude"), top.get("longitude")
        city = top.get("name", city)
    forecast = http.get(
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m"
        "&timezone=auto"
    )
    _http_error_check(forecast)
    data = forecast.json() or {}
    current = data.get("current") or {}
    return {
        "location": city if city else f"{lat},{lon}",
        "temperature_c": current.get("temperature_2m"),
        "feels_like_c": current.get("apparent_temperature"),
        "humidity_pct": current.get("relative_humidity_2m"),
        "wind_kmh": current.get("wind_speed_10m"),
        "condition": _wmo_condition(current.get("weather_code")),
        "time": current.get("time"),
    }


def _wmo_condition(code: Any) -> str:
    table = {
        0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
        45: "fog", 48: "depositing rime fog",
        51: "light drizzle", 53: "drizzle", 55: "dense drizzle",
        61: "light rain", 63: "rain", 65: "heavy rain",
        71: "light snow", 73: "snow", 75: "heavy snow",
        80: "rain showers", 81: "rain showers", 82: "violent rain showers",
        95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
    }
    return table.get(int(code) if code is not None else -1, f"code {code}")


@connector(
    "fx",
    description="current exchange rates (open.er-api, no key): latest rates vs a base currency",
    params={
        "base": "str (optional, default USD) — base currency code",
        "symbols": "str (optional) — comma list to filter, e.g. 'NGN,USD,EUR'",
    },
)
def _fx(params: dict[str, Any], http: HttpClient) -> dict[str, Any]:
    base = str(params.get("base") or "USD").strip().upper() or "USD"
    response = http.get(f"https://open.er-api.com/v6/latest/{base}")
    _http_error_check(response)
    data = response.json() or {}
    rates = data.get("rates") or {}
    wanted = {s.strip().upper() for s in str(params.get("symbols") or "").split(",") if s.strip()}
    if wanted:
        rates = {k: v for k, v in rates.items() if k in wanted} or dict(list(rates.items())[:20])
    else:
        rates = dict(list(rates.items())[:25])
    return {
        "base": base,
        "time": data.get("time_last_update_utc"),
        "rates": rates,
    }


@connector(
    "ip_info",
    description="geolocate an IP (ipwho.is, no key); empty = your own egress IP",
    params={"ip": "str (optional) — the IP to look up"},
)
def _ip_info(params: dict[str, Any], http: HttpClient) -> dict[str, Any]:
    ip = str(params.get("ip") or "").strip()
    response = http.get(f"https://ipwho.is/{ip}" if ip else "https://ipwho.is/")
    _http_error_check(response)
    data = response.json() or {}
    if data.get("success") is False:
        raise ToolError(f"ip lookup failed: {data.get('message', 'unknown error')}")
    return {
        "ip": data.get("ip"),
        "city": data.get("city"),
        "region": data.get("region"),
        "country": data.get("country"),
        "country_code": data.get("country_code"),
        "latitude": data.get("latitude"),
        "longitude": data.get("longitude"),
        "timezone": (data.get("timezone") or {}).get("id") if isinstance(data.get("timezone"), dict) else data.get("timezone"),
    }


@connector(
    "github",
    description=(
        "GitHub search: repos, code, or users (api.github.com; optional "
        "GITHUB token via NM_API_GITHUB_TOKEN raises the rate limit)"
    ),
    params={
        "query": "str — search text, e.g. 'language:python sandboxing'",
        "kind": "str (optional) — repo | code | user (default repo)",
    },
)
def _github(params: dict[str, Any], http: HttpClient) -> dict[str, Any]:
    query = str(params.get("query") or "").strip()
    if not query:
        raise ToolError("github needs a query")
    kind = str(params.get("kind") or "repo").strip().lower()
    endpoint = {"repo": "repositories", "code": "code", "user": "users"}.get(kind)
    if endpoint is None:
        raise ToolError("github kind must be repo, code, or user")
    headers = {}
    token = getattr(http._nm_api_settings, "github_token", "")  # type: ignore[attr-defined]
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["Accept"] = "application/vnd.github+json"
    response = http.get(f"https://api.github.com/search/{endpoint}?q={query}&per_page=8", headers=headers)
    _http_error_check(response)
    data = response.json() or {}
    items = data.get("items") or []
    if kind == "repo":
        results = [
            {
                "name": it.get("full_name"),
                "stars": it.get("stargazers_count"),
                "description": (it.get("description") or "")[:160],
                "url": it.get("html_url"),
                "language": it.get("language"),
            }
            for it in items
        ]
    elif kind == "user":
        results = [
            {"login": it.get("login"), "url": it.get("html_url"), "type": it.get("type")}
            for it in items
        ]
    else:
        results = [
            {
                "repo": (it.get("repository") or {}).get("full_name"),
                "path": it.get("path"),
                "url": it.get("html_url"),
            }
            for it in items
        ]
    return {"total": data.get("total_count"), "count": len(results), "results": results}


@connector(
    "wikipedia",
    description="Wikipedia article summary (REST API, no key)",
    params={"title": "str — article title, e.g. 'Lagos'"},
)
def _wikipedia(params: dict[str, Any], http: HttpClient) -> dict[str, Any]:
    title = str(params.get("title") or "").strip()
    if not title:
        raise ToolError("wikipedia needs a title")
    safe = re.sub(r"\s+", "_", title)
    response = http.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe}")
    _http_error_check(response)
    data = response.json() or {}
    return {
        "title": data.get("title"),
        "description": data.get("description"),
        "extract": (data.get("extract") or "")[:1500],
        "url": data.get("content_urls", {}).get("desktop", {}).get("page"),
    }


@connector(
    "http_api",
    description=(
        "generic JSON API call: GET/POST any URL with optional JSON body and "
        "bearer token (power tool — use for services that have no connector)"
    ),
    params={
        "method": "str (optional) — GET | POST (default GET)",
        "url": "str — full https URL",
        "body": "str (optional) — JSON body for POST",
        "token": "str (optional) — bearer token",
        "headers": "str (optional) — JSON object of extra headers",
    },
)
def _http_api(params: dict[str, Any], http: HttpClient) -> dict[str, Any]:
    url = str(params.get("url") or "").strip()
    if not url.startswith(("http://", "https://")):
        raise ToolError("http_api needs an http(s) url")
    method = str(params.get("method") or "GET").strip().upper()
    if method not in {"GET", "POST"}:
        raise ToolError("http_api method must be GET or POST")
    headers: dict[str, str] = {"Accept": "application/json"}
    raw_headers = str(params.get("headers") or "").strip()
    if raw_headers:
        parsed = _json_body(raw_headers) or {}
        if not isinstance(parsed, dict):
            raise ToolError("headers must be a JSON object")
        headers.update({str(k): str(v) for k, v in parsed.items()})
    token = str(params.get("token") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = _json_body(str(params.get("body") or "")) if method == "POST" else None
    if method == "POST":
        response = http.post_json(url, body or {}, headers=headers)
    else:
        response = http.get(url, headers=headers)
    _http_error_check(response)
    try:
        return {"status": response.status, "json": response.json()}
    except (ValueError, TypeError):
        return {"status": response.status, "text": (response.text or "")[:8000]}


# ── tool registration ────────────────────────────────────────────────────────


def register(registry: Any) -> None:
    """Attach the api_call / api_list tools to a registry."""
    context = registry.context
    settings = getattr(context, "settings", None) if context is not None else None
    api_settings = getattr(settings, "api", None) if settings else None
    timeout = float(getattr(api_settings, "request_timeout", 20.0)) if api_settings else 20.0
    client = HttpClient(timeout=timeout)
    # connectors read connector-specific settings (tokens, weather defaults)
    client._nm_api_settings = api_settings  # type: ignore[attr-defined]

    @registry.register(
        "api_call",
        description=(
            "call an external API through the connector framework: "
            + ", ".join(f"{c['name']}({', '.join(c['params']) or 'no params'})"
                        for c in connectors.manifest()[:6])
            + " — see api_list for all"
        ),
        capability=Capability.NET_OUT,
        parameters={
            "connector": "str — connector name (see api_list)",
            "params": "str (optional) — JSON object of the connector's parameters",
        },
    )
    def api_call(connector: str, *, params: str = "") -> dict[str, Any]:
        parsed = _json_body(params) or {}
        if not isinstance(parsed, dict):
            raise ToolError("params must be a JSON object")
        return connectors.call(str(connector or "").strip(), parsed, client)

    @registry.register(
        "api_list",
        description="list the available API connectors with their parameters",
        capability=Capability.NET_OUT,
    )
    def api_list() -> dict[str, Any]:
        return {"count": len(connectors.names()), "connectors": connectors.manifest()}
