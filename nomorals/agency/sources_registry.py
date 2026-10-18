"""Single-sourced registry of every verifiable source in Devon.

This module contains NO URL lists of its own. Every entry is collected
from the REAL module that owns the list:

* news — ``agents.news.FEED_CATEGORIES``
* search — ``tools.web._SEARCH_ENGINES`` + ``_engine_url``
* music — ``media.sources.SOURCE_CHAIN`` (plus the adapter-gated
  entries of ``MUSIC_SEARCH_SOURCES`` under ``music-adapter``)
* film — ``media.film_sources.FILM_CHAIN``
* weather — ``agents.weather`` endpoint constants
* wisdom — ``wisdom.ingestor`` endpoint constants
* price — probe URLs mirroring ``ta.feeds`` (that module builds its
  endpoint URLs inline; the probes here are the same public endpoints
  with a fixed test query, marked as probes)

If a list moves, this registry follows it. If a module fails to import
(e.g. an optional dependency is missing), ``collect()`` records the
error and keeps the other families — a broken import in one corner
must not blind the whole health system.

Probe queries are fixed, neutral, and real ("Burna Boy" for music,
"Avatar" for film, "Lagos Nigeria" for search): verification must hit
the source the way production does.
"""

from __future__ import annotations

import urllib.parse
from typing import Any

from .sourcehealth import SourceSpec

__all__ = [
    "collect", "spec_by_key",
    "PROBE_MUSIC_QUERY", "PROBE_FILM_QUERY", "PROBE_SEARCH_QUERY",
]

PROBE_MUSIC_QUERY = "Burna Boy"
PROBE_FILM_QUERY = "Avatar"
PROBE_SEARCH_QUERY = "Lagos Nigeria"


def _q(query: str) -> str:
    return urllib.parse.quote_plus(query)


def _collect_news(specs: list[SourceSpec], errors: list[str]) -> None:
    try:
        from ..agents.news import FEED_CATEGORIES
    except Exception as exc:  # noqa: BLE001
        errors.append(f"news: {exc}")
        return
    for cat, feeds in FEED_CATEGORIES.items():
        for name, url in feeds:
            specs.append(SourceSpec(
                key=f"news:{cat}:{name}", category="news", name=name,
                url=url, kind="rss"))


def _collect_search(specs: list[SourceSpec], errors: list[str]) -> None:
    try:
        from ..tools.web import _SEARCH_ENGINES, _engine_url
    except Exception as exc:  # noqa: BLE001
        errors.append(f"search: {exc}")
        return
    for engine in _SEARCH_ENGINES:
        specs.append(SourceSpec(
            key=f"search:{engine}", category="search", name=engine,
            url=_engine_url(engine, _q(PROBE_SEARCH_QUERY)),
            kind="html-links"))


def _music_kind(inst: Any) -> str:
    try:
        from ..media.sources import _JsonApiSource
    except Exception:  # noqa: BLE001
        _JsonApiSource = ()  # type: ignore[assignment]
    if isinstance(inst, _JsonApiSource):
        return "json"
    if getattr(inst, "name", "") == "archiveorg":
        return "html-archive"
    if getattr(inst, "name", "") == "bandcamp":
        return "html-bandcamp"
    return "html-posts"


def _collect_music(specs: list[SourceSpec], errors: list[str]) -> None:
    try:
        from ..media.sources import MUSIC_SEARCH_SOURCES, SOURCE_CHAIN
    except Exception as exc:  # noqa: BLE001
        errors.append(f"music: {exc}")
        return
    chain_names: set[str] = set()
    for chain_name, inst in SOURCE_CHAIN:
        chain_names.add(chain_name)
        # The registry uses the CHAIN strategy name (e.g. "netnaija-search"),
        # matching MUSIC_SEARCH_SOURCES — not the instance's short name.
        search_url = getattr(inst, "search_url", None)
        if callable(search_url):
            url = search_url(PROBE_MUSIC_QUERY)
        else:  # pragma: no cover — every chain member has search_url
            url = getattr(inst, "_SEARCH", getattr(inst, "_BASE", ""))
        specs.append(SourceSpec(
            key=f"music:{chain_name}", category="music", name=chain_name,
            url=url, kind=_music_kind(inst)))
    # Adapter-gated /play strategies: not plain-HTTP verifiable, so
    # health records them as external-adapter instead of pretending.
    adapter_notes = {
        "soundcloud-search": "SoundCloud API adapter (resolver path).",
        "youtube-search": "yt-dlp YouTube search (resolver path).",
        "youtube-music-search": "yt-dlp YouTube Music search (resolver path).",
        "spotify-search": "Spotify OAuth metadata only; streams protected.",
        "workspace-scan": "Owner's own local files, not a network source.",
    }
    for strategy in MUSIC_SEARCH_SOURCES:
        if strategy in chain_names:
            continue
        specs.append(SourceSpec(
            key=f"music-adapter:{strategy}", category="music-adapter",
            name=strategy, url="",
            kind="external-adapter",
            notes=adapter_notes.get(strategy, "Adapter-gated source.")))
    specs.append(SourceSpec(
        key="music-adapter:audiomack", category="music-adapter",
        name="audiomack", url="https://audiomack.com",
        kind="external-adapter",
        notes="URL-first-class: paste an audiomack.com link. "
              "No keyless text search endpoint exists."))


def _collect_film(specs: list[SourceSpec], errors: list[str]) -> None:
    try:
        from ..media.film_sources import FILM_CHAIN
    except Exception as exc:  # noqa: BLE001
        errors.append(f"film: {exc}")
        return
    for inst in FILM_CHAIN:
        search_url = getattr(inst, "search_url", None)
        url = search_url(PROBE_FILM_QUERY) if callable(search_url) else \
            getattr(inst, "_BASE", "")
        specs.append(SourceSpec(
            key=f"film:{inst.name}", category="film", name=inst.name,
            url=url, kind="html-posts"))


def _collect_weather(specs: list[SourceSpec], errors: list[str]) -> None:
    try:
        from ..agents.weather import (
            FORECAST_URL, GEOCODE_URL, METNO_URL, NWS_ALERTS_URL, WTTR_URL)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"weather: {exc}")
        return
    specs.extend([
        SourceSpec(key="weather:open-meteo-geocode", category="weather",
                   name="open-meteo-geocode",
                   url=f"{GEOCODE_URL}?name=Lagos&count=1", kind="json"),
        SourceSpec(key="weather:open-meteo-forecast", category="weather",
                   name="open-meteo-forecast",
                   url=(f"{FORECAST_URL}?latitude=6.45&longitude=3.39"
                        "&hourly=temperature_2m&forecast_days=1"),
                   kind="json-weather"),
        SourceSpec(key="weather:wttr", category="weather", name="wttr",
                   url=f"{WTTR_URL}/Lagos?format=j1", kind="json"),
        SourceSpec(key="weather:metno", category="weather", name="metno",
                   url=f"{METNO_URL}?lat=6.45&lon=3.39", kind="json"),
        SourceSpec(key="weather:nws-alerts", category="weather",
                   name="nws-alerts",
                   url=f"{NWS_ALERTS_URL}?point=40.7,-74.0", kind="json"),
    ])


def _collect_wisdom(specs: list[SourceSpec], errors: list[str]) -> None:
    try:
        from ..wisdom.ingestor import (
            _ARCHIVE_SEARCH, _GUTENDEX, _GUTENBERG_TEXT, _WIKISOURCE_API)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"wisdom: {exc}")
        return
    specs.extend([
        SourceSpec(key="wisdom:archiveorg", category="wisdom",
                   name="archiveorg",
                   url=(f"{_ARCHIVE_SEARCH}?q=title%3A%28meditations%29"
                        "&fl%5B%5D=identifier&rows=3&output=json"),
                   kind="html-archive"),
        SourceSpec(key="wisdom:gutendex", category="wisdom",
                   name="gutendex",
                   url=f"{_GUTENDEX}?search=meditations", kind="json"),
        SourceSpec(key="wisdom:wikisource", category="wisdom",
                   name="wikisource",
                   url=(f"{_WIKISOURCE_API}?action=query&list=search"
                        "&srsearch=Meditations&format=json"),
                   kind="json"),
        SourceSpec(key="wisdom:gutenberg", category="wisdom",
                   name="gutenberg",
                   url=_GUTENBERG_TEXT.format(id=2680),
                   kind="json-gutenberg"),
    ])


def _collect_price(specs: list[SourceSpec], errors: list[str]) -> None:
    # Probes mirroring ta.feeds' public endpoints (that module builds
    # its URLs inline per fetch function; these are the same endpoints
    # with a fixed test symbol). If ta.feeds cannot import, the probes
    # still describe the endpoints — but record the import failure.
    try:
        from ..ta import feeds as _feeds  # noqa: F401 — import check only
    except Exception as exc:  # noqa: BLE001
        errors.append(f"price: {exc}")
        return
    specs.extend([
        SourceSpec(key="price:binance", category="price", name="binance",
                   url=("https://api.binance.com/api/v3/klines"
                        "?symbol=BTCUSDT&interval=1h&limit=2"),
                   kind="json"),
        SourceSpec(key="price:coinbase", category="price", name="coinbase",
                   url=("https://api.exchange.coinbase.com/products/"
                        "BTC-USD/candles?granularity=3600"),
                   kind="json"),
        SourceSpec(key="price:kraken", category="price", name="kraken",
                   url=("https://api.kraken.com/0/public/OHLC"
                        "?pair=XBTUSD&interval=60"),
                   kind="json"),
        SourceSpec(key="price:bybit", category="price", name="bybit",
                   url=("https://api.bybit.com/v5/market/kline"
                        "?category=spot&symbol=BTCUSDT&interval=60&limit=2"),
                   kind="json"),
    ])


def collect() -> tuple[list[SourceSpec], list[str]]:
    """All verifiable sources + import errors (empty on a healthy tree)."""
    specs: list[SourceSpec] = []
    errors: list[str] = []
    _collect_news(specs, errors)
    _collect_search(specs, errors)
    _collect_music(specs, errors)
    _collect_film(specs, errors)
    _collect_weather(specs, errors)
    _collect_wisdom(specs, errors)
    _collect_price(specs, errors)
    return specs, errors


def spec_by_key(specs: list[SourceSpec], key: str) -> SourceSpec | None:
    for spec in specs:
        if spec.key == key:
            return spec
    return None
