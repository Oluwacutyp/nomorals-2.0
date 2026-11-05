"""Spotify metadata — identify a Spotify link, never its audio.

Spotify audio is DRM-protected; there is no download route, and the
source registry already says so (``open.spotify.com`` is a
metadata-only, not-downloadable source).  What a Spotify link CAN do
is identify exactly what it points at — track, album, playlist,
episode — so the caller can fetch the full track through the
ordinary full-source ladder (YouTube et al.) and label it as a
match.  This module is that identification step, and nothing else:
the only audio URLs Spotify's pages carry are ~30-second previews,
which are preview-class under the preview law — they are never
read, never returned, never delivered.

Two keyless routes, both reading Spotify's own public surfaces:

1. **the embed player payload** — ``open.spotify.com/embed/<kind>/
   <id>`` serves a page whose ``__NEXT_DATA__`` JSON carries the
   entity: for albums/playlists a ``trackList`` (title, subtitle =
   artist, duration in ms per track); for a track/episode the
   entity itself.
2. **oEmbed** — ``open.spotify.com/oembed?url=…`` returns the
   entity's title and author name.  It cannot enumerate a
   collection, so it is the fallback for single entities and the
   name source when the embed payload is thin.

Both routes are plain GETs of public pages — no credentials, no
token dance.  Every function never raises; failures come back as
``None`` / empty fields so the caller can say honestly what it
could and couldn't identify.

Layering: media is L4.  No connector imports here (the linked
Spotify connector, when present, is a playback surface — device
playback — and stays that; identification here is keyless).
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger
from .failures import record_failure
from .sources import BROWSER_UA

_log = get_logger(__name__)

__all__ = [
    "SpotifyTrack",
    "SpotifyMeta",
    "parse_spotify_ref",
    "canonical_spotify_url",
    "fetch_spotify_meta",
    "SPOTIFY_KINDS",
]

SPOTIFY_KINDS = ("track", "album", "playlist", "episode", "show")

_URL_RE = re.compile(
    r"open\.spotify\.com/(?:intl-[a-zA-Z-]+/)?"
    r"(track|album|playlist|episode|show)/([A-Za-z0-9]+)")
_URI_RE = re.compile(
    r"^spotify:(track|album|playlist|episode|show):([A-Za-z0-9]+)$")
_HTTP_TIMEOUT = 20.0


@dataclass
class SpotifyTrack:
    """One identified track: metadata only, by construction."""

    title: str = ""
    artist: str = ""
    duration_s: float = 0.0

    @property
    def query(self) -> str:
        """The full-source ladder query for this track."""
        if self.title and self.artist:
            return f"{self.title} by {self.artist}"
        return self.title or self.artist


@dataclass
class SpotifyMeta:
    """What a Spotify link points at — identification, not audio."""

    kind: str = ""               # track | album | playlist | episode | show
    spotify_id: str = ""
    name: str = ""               # track title / album / playlist name
    artist: str = ""             # primary artist or collection owner
    duration_s: float = 0.0      # single-entity duration when known
    tracks: list[SpotifyTrack] = field(default_factory=list)
    source: str = ""             # "embed" | "oembed" | "embed+oembed"
    note: str = ""               # honest gap, e.g. couldn't enumerate


def parse_spotify_ref(text: str) -> tuple[str, str] | None:
    """``(kind, id)`` from a Spotify URL or URI inside *text*.

    Handles ``open.spotify.com`` URLs (locale-prefixed or not, with
    or without query strings) and bare ``spotify:`` URIs.  Returns
    None when no Spotify reference is present — note ``spoti.fi``
    short links need expansion first (the caller expands, then
    re-parses), since the id lives behind the redirect.
    """
    raw = str(text or "")
    m = _URI_RE.match(raw.strip())
    if m:
        return m.group(1), m.group(2)
    m = _URL_RE.search(raw)
    if m:
        return m.group(1), m.group(2)
    return None


def canonical_spotify_url(kind: str, spotify_id: str) -> str:
    return f"https://open.spotify.com/{kind}/{spotify_id}"



# ── HTTP (injectable for tests) ─────────────────────────────────────


def _http_get_text(url: str, timeout: float = _HTTP_TIMEOUT) -> str:
    """GET *url* → body text ("" on any failure).  Never raises."""
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": BROWSER_UA,
                          "Accept": "text/html,application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        try:
            charset = resp.headers.get_content_charset() or "utf-8"
        except Exception:  # noqa: BLE001
            charset = "utf-8"
        return raw.decode(charset, errors="replace")
    except Exception as exc:  # noqa: BLE001 — network is hostile
        _log.info("spotify_meta GET failed for %s: %s", url[:80], exc)
        return ""


def _http_get_json(url: str, timeout: float = _HTTP_TIMEOUT) -> dict | None:
    """GET *url* → parsed JSON dict or None.  Never raises."""
    text = _http_get_text(url, timeout=timeout)
    if not text:
        return None
    try:
        data = json.loads(text)
    except Exception as exc:  # noqa: BLE001
        record_failure("spotify_meta._http_get_json", exc)
        return None
    return data if isinstance(data, dict) else None


# ── embed payload parsing ───────────────────────────────────────────

_NEXT_DATA_RE = re.compile(
    r'<script[^>]+id="__NEXT_DATA__"[^>]*>(.*?)</script>',
    re.DOTALL | re.IGNORECASE)


def _find_entity(node: Any, depth: int = 0) -> dict | None:
    """Locate the embed entity dict in the __NEXT_DATA__ tree.

    The entity is the dict carrying a ``trackList`` (collections)
    or a track-shaped ``title``/``subtitle`` pair (singles).  The
    walk is structural, not path-pinned: Spotify reshuffles the
    pageProps nesting between deploys, but the entity's own shape
    is what the player renders from.
    """
    if depth > 12 or not isinstance(node, (dict, list)):
        return None
    if isinstance(node, dict):
        if isinstance(node.get("trackList"), list):
            return node
        if node.get("title") and ("subtitle" in node
                                  or "duration" in node):
            return node
        for value in node.values():
            found = _find_entity(value, depth + 1)
            if found is not None:
                return found
    else:
        for value in node:
            found = _find_entity(value, depth + 1)
            if found is not None:
                return found
    return None


def _track_from_embed_item(item: Any) -> SpotifyTrack | None:
    """One trackList entry → SpotifyTrack (None when shapeless).

    Embed track entries carry ``title``, ``subtitle`` (the artist
    line) and ``duration`` in milliseconds.  Their ``audioPreview``
    field — a ~30s preview URL — is deliberately never read: it is
    preview-class and this module returns metadata only.
    """
    if not isinstance(item, dict):
        return None
    title = str(item.get("title") or "").strip()
    if not title:
        return None
    artist = str(item.get("subtitle") or "").strip()
    duration_ms = item.get("duration")
    try:
        duration_s = float(duration_ms or 0) / 1000.0
    except (TypeError, ValueError):
        duration_s = 0.0
    return SpotifyTrack(title=title, artist=artist,
                        duration_s=duration_s)


def _parse_embed(html: str) -> dict | None:
    """The embed entity dict from an embed page's HTML, or None."""
    if not html:
        return None
    m = _NEXT_DATA_RE.search(html)
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
    except Exception as exc:  # noqa: BLE001
        record_failure("spotify_meta._parse_embed", exc)
        return None
    return _find_entity(data)



# ── the fetch ───────────────────────────────────────────────────────


def fetch_spotify_meta(
        kind: str, spotify_id: str, *,
        get_text: Callable[[str], str] | None = None,
        get_json: Callable[[str], dict | None] | None = None,
        ) -> SpotifyMeta | None:
    """Identify the Spotify entity ``kind``/``spotify_id``.

    Returns a :class:`SpotifyMeta`, or None when neither route could
    identify anything (dead link, region wall, page shape changed).
    A collection whose tracks could not be enumerated comes back
    with its name set, ``tracks`` empty and ``note`` saying so —
    partial truth, labeled.
    """
    if kind not in SPOTIFY_KINDS or not spotify_id:
        return None
    text_get = get_text if get_text is not None else _http_get_text
    json_get = get_json if get_json is not None else _http_get_json
    meta = SpotifyMeta(kind=kind, spotify_id=spotify_id)
    sources: list[str] = []

    # Route 1: the embed payload (enumerates collections).
    entity = None
    try:
        html = text_get(
            f"https://open.spotify.com/embed/{kind}/{spotify_id}")
        entity = _parse_embed(html)
    except Exception as exc:  # noqa: BLE001 — injected getters behave
        record_failure("spotify_meta.fetch.embed", exc)
    if entity is not None:
        sources.append("embed")
        track_list = entity.get("trackList")
        if isinstance(track_list, list):
            for item in track_list:
                track = _track_from_embed_item(item)
                if track is not None:
                    meta.tracks.append(track)
            meta.name = str(entity.get("name")
                            or entity.get("title") or "").strip()
            meta.artist = str(entity.get("subtitle") or "").strip()
            if kind in ("track", "episode") and meta.tracks:
                first = meta.tracks[0]
                meta.name = meta.name or first.title
                meta.artist = meta.artist or first.artist
                meta.duration_s = first.duration_s
        else:
            single = _track_from_embed_item(entity)
            if single is not None:
                meta.name = single.title
                meta.artist = single.artist
                meta.duration_s = single.duration_s
                if kind in ("track", "episode"):
                    meta.tracks = [single]
            meta.name = meta.name or str(
                entity.get("name") or "").strip()
            meta.artist = meta.artist or str(
                entity.get("subtitle") or "").strip()

    # Route 2: oEmbed (names singles authoritatively; fills gaps).
    try:
        oembed = json_get(
            "https://open.spotify.com/oembed?url="
            + urllib.parse.quote(
                canonical_spotify_url(kind, spotify_id), safe=""))
    except Exception as exc:  # noqa: BLE001
        record_failure("spotify_meta.fetch.oembed", exc)
        oembed = None
    if isinstance(oembed, dict) and oembed.get("title"):
        sources.append("oembed")
        if not meta.name:
            meta.name = str(oembed.get("title") or "").strip()
        if not meta.artist:
            meta.artist = str(oembed.get("author_name") or "").strip()

    if not meta.name and not meta.tracks:
        return None
    if kind in ("album", "playlist", "show") and not meta.tracks:
        meta.note = ("identified the collection, but couldn't list "
                     "its tracks from Spotify's public pages")
    if kind in ("track", "episode") and not meta.tracks and meta.name:
        meta.tracks = [SpotifyTrack(
            title=meta.name, artist=meta.artist,
            duration_s=meta.duration_s)]
    meta.source = "+".join(sources)
    return meta
