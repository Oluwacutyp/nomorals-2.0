"""Sweep instruments — the existing search organs, adapted to one
honest protocol.

WS4 builds NO new fetchers (census §3.1): a sweep observes through
organs that already exist — ``VideoFinder.find`` (cross-web video
search), ``PlaybackEngine._youtube_search_many`` (YouTube candidates
via yt-dlp), and the keyless web search (``media.video.web_search_impl``,
the engines behind the source registry's web-search class).

The protocol every instrument honours:

* ``search(query, limit=...)`` returns a list of normalised result
  rows (possibly empty — an empty list means the instrument ANSWERED
  and found nothing);
* an instrument that cannot answer (missing dependency, transport
  failure, a backend that reports an error) raises
  :class:`InstrumentError`. Silence is never an answer: the engine
  records ``empty`` and ``failed`` as different facts, so wrapping an
  organ that swallows its own failures means disambiguating its
  empty results honestly (see the YouTube instrument).

Normalised row: ``{rank, title, url, source, channel, view_count,
duration_s}`` — ``channel``/``view_count``/``duration_s`` are None
when the organ does not expose them; a missing value is recorded as
missing, never as zero.

Exploration reads are unauthenticated public reads (census §7):
they ride direct connections and the free proxy tier's charter —
these instruments attach no cookies, no credentials, no vault
material. The named dependency is resolved (doc 17 P1): the
proxy catalogue's ``refresh_sources`` has a production caller —
every proxy refresh activates the catalogue first — so sweeps
run direct + whatever the pool holds, and the pool now grows
from the verified catalogue on every refresh.
"""

from __future__ import annotations

from typing import Any

from ...core.errors import ToolError
from ...core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "InstrumentError",
    "VideoFinderInstrument",
    "WebSearchInstrument",
    "YouTubeSearchInstrument",
    "default_instruments",
]


class InstrumentError(Exception):
    """An instrument could not answer. Distinct from an empty
    answer — the engine records the two as different facts."""


def _num(value: Any) -> float | int | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


class VideoFinderInstrument:
    """Cross-web video search via ``media.video.VideoFinder``."""

    name = "video_finder"

    def __init__(self, context: Any) -> None:
        self.context = context

    def search(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        from ..video import VideoFinder
        try:
            out = VideoFinder(self.context).find(query, max_results=limit)
        except ToolError as exc:
            raise InstrumentError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 — an organ failure is a failure
            raise InstrumentError(
                f"video_finder failed: {type(exc).__name__}: {exc}") from exc
        rows = []
        for i, entry in enumerate(out.get("results") or []):
            rows.append({
                "rank": i + 1,
                "title": str(entry.get("title") or ""),
                "url": str(entry.get("url") or ""),
                "source": str(entry.get("source") or ""),
                "channel": str(entry.get("author") or "") or None,
                "view_count": _num(entry.get("view_count")),
                "duration_s": _num(entry.get("duration")),
            })
        return rows


class YouTubeSearchInstrument:
    """YouTube candidates via the playback engine's yt-dlp search.

    ``_youtube_search_many`` never raises — it returns [] both for
    "no results" and for "yt-dlp missing / search failed". A sweep
    must not record a failure as an empty field, so an empty answer
    is disambiguated with the engine's own raising search
    (``_youtube_search_id``): its ToolError names which of the two
    happened ("no YouTube results for …" vs a dependency/transport
    failure), and only the first maps back to an empty list.
    """

    name = "youtube_search"

    def search(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        from ..playback import PlaybackEngine
        entries = PlaybackEngine._youtube_search_many(query, limit=limit)
        if not entries:
            try:
                PlaybackEngine._youtube_search_id(query)
            except ToolError as exc:
                message = str(exc)
                if message.startswith("no YouTube results"):
                    return []
                raise InstrumentError(message) from exc
            raise InstrumentError(
                "youtube search returned no entries but a single-result "
                "search succeeded — inconsistent instrument state")
        rows = []
        for i, entry in enumerate(entries):
            vid = str(entry.get("video_id") or "")
            rows.append({
                "rank": i + 1,
                "title": str(entry.get("title") or ""),
                "url": PlaybackEngine._youtube_watch_url(vid) if vid else "",
                "source": "youtube",
                "channel": str(entry.get("uploader") or "") or None,
                "view_count": _num(entry.get("view_count")),
                "duration_s": _num(entry.get("duration")),
            })
        return rows


class WebSearchInstrument:
    """Keyless web search — the source registry's web-search class
    engines (DuckDuckGo → Bing → Mojeek), via ``web_search_impl``.

    This is the cross-class corroboration leg: the open web seeing
    the same topic the video organs see. The impl returns the first
    engine that answers; when every engine fails it returns [] and
    raises nothing, so this instrument's empty is recorded as empty
    (its failure mode is indistinguishable from no-results at the
    organ's own seam — stated here rather than papered over).
    """

    name = "web_search"

    def __init__(self, context: Any = None) -> None:
        self.context = context

    def search(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        from ..video import web_search_impl
        tools = getattr(self.context, "tools", None) if self.context else None
        call = getattr(tools, "call", None) if tools else None
        raw: list[dict[str, Any]] = []
        if call is not None:
            try:
                out = call("web_search", query=query, max_results=limit)
            except Exception as exc:  # noqa: BLE001
                raise InstrumentError(
                    f"web_search tool failed: {type(exc).__name__}: {exc}"
                ) from exc
            if not getattr(out, "ok", False):
                raise InstrumentError(
                    f"web_search tool failed: {getattr(out, 'error', '')}")
            raw = list(out.unwrap().get("results", []))
        else:
            try:
                raw = web_search_impl(query, max_results=limit)
            except Exception as exc:  # noqa: BLE001
                raise InstrumentError(
                    f"web search failed: {type(exc).__name__}: {exc}"
                ) from exc
        rows = []
        for i, entry in enumerate(raw[:limit]):
            rows.append({
                "rank": i + 1,
                "title": str(entry.get("title") or ""),
                "url": str(entry.get("url") or ""),
                "source": str(entry.get("engine") or "web"),
                "channel": None,
                "view_count": None,
                "duration_s": None,
            })
        return rows


def default_instruments(context: Any) -> list[Any]:
    """The production sweep's instrument set, in census order:
    cross-web video search, YouTube candidates, web search."""
    return [
        VideoFinderInstrument(context),
        YouTubeSearchInstrument(),
        WebSearchInstrument(context),
    ]
