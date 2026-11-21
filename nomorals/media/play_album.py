"""``/play L <artist>`` — a random album from the artist, every track sent.

Metadata layer: iTunes Search/Lookup (keyless, official Apple endpoint)
for the album list + the album's track list, Deezer public API as the
fallback. Both are identification-only here — they name the album and
its tracks and never supply audio (their only audio field is a 30s
preview, which the resolver already refuses to deliver). The AUDIO for
each track comes from the normal music chain (``SourceResolver.resolve``
+ the chat fall-through), mp3.pm first, mdundo last — the same path a
single ``/play`` uses, so an album track lands exactly like a ``/play``
track.

Randomness: :func:`pick_random_album` draws from a runtime-seeded
``random.Random()`` — a fresh pick per invocation, never a fixed seed.

Never raises: every public function returns ``[]`` / ``None`` / an
error string on failure. The fetcher is injectable (``url -> str``) so
tests run offline against canned payloads; the default fetcher is plain
urllib with the sources' browser UA and a 15s timeout.
"""

from __future__ import annotations

import json
import random
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field

from ..core.logging_setup import get_logger
from .sources import BROWSER_UA

_log = get_logger(__name__)

_FETCH_TIMEOUT = 15.0
_ALBUM_SEARCH_LIMIT = 25
_TRACK_LOOKUP_LIMIT = 200


@dataclass
class AlbumTrack:
    """One track of an album, in album order."""

    number: int = 0
    title: str = ""


@dataclass
class Album:
    """One album (or single/EP) from an artist search."""

    title: str = ""
    artist: str = ""
    source: str = ""          # "itunes" | "deezer"
    source_id: str = ""
    kind: str = "album"       # "album" | "single" (singles + EPs)
    track_count: int = 0


@dataclass
class AlbumResult:
    """The outcome of :func:`find_album_for_artist` — never raises."""

    album: Album | None = None
    tracks: list[AlbumTrack] = field(default_factory=list)
    error: str = ""           # "" on success, plain words otherwise


def _default_fetch(url: str) -> str:
    """GET *url*, return text. Empty string on any failure."""
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": BROWSER_UA,
                          "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001 - network is hostile
        _log.info("play_album: GET failed for %s: %s", url[:80], exc)
        return ""


def _norm(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _artist_matches(candidate_artist: str, wanted: str) -> bool:
    """Does the result's artist name name the wanted artist?"""
    got, want = _norm(candidate_artist), _norm(wanted)
    if not got or not want:
        return False
    if len(want) < 3:
        return got == want
    return want in got or got in want


def _kind_for_title(title: str) -> str:
    """iTunes marks singles/EPs as collectionType Album with a name
    suffix — a " - Single" / " - EP" is not a full album."""
    low = (title or "").strip().lower()
    if low.endswith(" - single") or low.endswith(" - ep"):
        return "single"
    return "album"


def _dedupe(albums: list[Album]) -> list[Album]:
    seen: set[tuple[str, str]] = set()
    out: list[Album] = []
    for album in albums:
        key = (_norm(album.title), _norm(album.artist))
        if not key[0] or key in seen:
            continue
        seen.add(key)
        out.append(album)
    return out


def _search_albums_itunes(artist: str,
                          fetcher: Callable[[str], str]) -> list[Album]:
    url = ("https://itunes.apple.com/search?term="
           + urllib.parse.quote_plus(artist)
           + "&media=music&entity=album&attribute=artistTerm&limit="
           + str(_ALBUM_SEARCH_LIMIT))
    try:
        payload = json.loads(fetcher(url) or "{}")
    except Exception:  # noqa: BLE001 - fetcher/network is hostile
        return []
    out: list[Album] = []
    for row in (payload or {}).get("results") or []:
        if not isinstance(row, dict):
            continue
        if row.get("wrapperType") != "collection":
            continue
        title = str(row.get("collectionName") or "")
        name = str(row.get("artistName") or "")
        if not title or not _artist_matches(name, artist):
            continue
        try:
            count = int(row.get("trackCount") or 0)
        except (TypeError, ValueError):
            count = 0
        out.append(Album(title=title, artist=name, source="itunes",
                         source_id=str(row.get("collectionId") or ""),
                         kind=_kind_for_title(title), track_count=count))
    return _dedupe(out)


def _search_albums_deezer(artist: str,
                          fetcher: Callable[[str], str]) -> list[Album]:
    url = ("https://api.deezer.com/search/album?q="
           + urllib.parse.quote_plus(f'artist:"{artist}"')
           + "&limit=" + str(_ALBUM_SEARCH_LIMIT))
    try:
        payload = json.loads(fetcher(url) or "{}")
    except Exception:  # noqa: BLE001 - fetcher/network is hostile
        return []
    out: list[Album] = []
    for row in (payload or {}).get("data") or []:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title") or "")
        name = str((row.get("artist") or {}).get("name") or "")
        if not title or not _artist_matches(name, artist):
            continue
        try:
            count = int(row.get("nb_tracks") or 0)
        except (TypeError, ValueError):
            count = 0
        out.append(Album(title=title, artist=name, source="deezer",
                         source_id=str(row.get("id") or ""),
                         kind=_kind_for_title(title), track_count=count))
    return _dedupe(out)


def search_albums(artist: str, *,
                  fetcher: Callable[[str], str] | None = None,
                  limit: int = _ALBUM_SEARCH_LIMIT) -> list[Album]:
    """The artist's albums, full albums first in the list order the
    pick draws from — iTunes first, Deezer when iTunes finds nothing.
    Never raises; [] when nothing is found or every fetch fails."""
    artist = (artist or "").strip()
    if not artist:
        return []
    fetch = fetcher or _default_fetch
    albums = _search_albums_itunes(artist, fetch)
    if not albums:
        albums = _search_albums_deezer(artist, fetch)
    full = [a for a in albums if a.kind == "album"]
    singles = [a for a in albums if a.kind != "album"]
    return (full + singles)[:max(1, int(limit or _ALBUM_SEARCH_LIMIT))]


def _album_tracks_itunes(album: Album,
                         fetcher: Callable[[str], str]) -> list[AlbumTrack]:
    url = ("https://itunes.apple.com/lookup?id="
           + urllib.parse.quote_plus(album.source_id)
           + "&entity=song&limit=" + str(_TRACK_LOOKUP_LIMIT))
    try:
        payload = json.loads(fetcher(url) or "{}")
    except Exception:  # noqa: BLE001 - fetcher/network is hostile
        return []
    rows: list[AlbumTrack] = []
    for row in (payload or {}).get("results") or []:
        if not isinstance(row, dict):
            continue
        if row.get("wrapperType") != "track":
            continue
        title = str(row.get("trackName") or "")
        if not title:
            continue
        try:
            number = int(row.get("trackNumber") or 0)
        except (TypeError, ValueError):
            number = 0
        rows.append(AlbumTrack(number=number, title=title))
    rows.sort(key=lambda t: (t.number if t.number > 0 else 10 ** 6,
                             t.title))
    return rows


def _album_tracks_deezer(album: Album,
                         fetcher: Callable[[str], str]) -> list[AlbumTrack]:
    url = ("https://api.deezer.com/album/"
           + urllib.parse.quote_plus(album.source_id)
           + "/tracks?limit=" + str(_TRACK_LOOKUP_LIMIT))
    try:
        payload = json.loads(fetcher(url) or "{}")
    except Exception:  # noqa: BLE001 - fetcher/network is hostile
        return []
    rows: list[AlbumTrack] = []
    for row in (payload or {}).get("data") or []:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title") or "")
        if not title:
            continue
        try:
            number = int(row.get("track_position") or 0)
        except (TypeError, ValueError):
            number = 0
        rows.append(AlbumTrack(number=number, title=title))
    rows.sort(key=lambda t: (t.number if t.number > 0 else 10 ** 6,
                             t.title))
    return rows


def album_tracks(album: Album, *,
                 fetcher: Callable[[str], str] | None = None
                 ) -> list[AlbumTrack]:
    """The album's tracks in album order. Never raises; [] when the
    lookup fails or the album comes back with no track list."""
    if album is None or not album.source_id:
        return []
    fetch = fetcher or _default_fetch
    if album.source == "deezer":
        return _album_tracks_deezer(album, fetch)
    return _album_tracks_itunes(album, fetch)


def pick_random_album(albums: list[Album], *,
                      rng: random.Random | None = None) -> Album | None:
    """A random album from *albums* — full albums preferred, singles
    only when the artist has no full album in the list. The draw uses
    a runtime-seeded ``random.Random()`` unless the caller injects one
    (tests inject a seeded rng for determinism). Never a fixed seed in
    production: the same artist twice must be able to land on two
    different albums. None when the list is empty."""
    pool = [a for a in (albums or []) if a.kind == "album"] or list(albums or [])
    if not pool:
        return None
    return (rng or random.Random()).choice(pool)


def find_album_for_artist(artist: str, *,
                          fetcher: Callable[[str], str] | None = None,
                          rng: random.Random | None = None) -> AlbumResult:
    """Search → random pick → track list. Never raises; ``error`` is
    plain words naming what is missing."""
    artist = (artist or "").strip()
    if not artist:
        return AlbumResult(
            error="usage: /play L <artist> — pick a random album from "
                  "the artist and send every track in it.")
    albums = search_albums(artist, fetcher=fetcher)
    if not albums:
        return AlbumResult(
            error=f"no albums found for {artist!r} — nothing to pick "
                  "from. Check the spelling, or try another artist.")
    album = pick_random_album(albums, rng=rng)
    if album is None:
        return AlbumResult(
            error=f"no albums found for {artist!r} — nothing to pick "
                  "from. Check the spelling, or try another artist.")
    tracks = album_tracks(album, fetcher=fetcher)
    if not tracks:
        return AlbumResult(
            album=album,
            error=f"{album.title} by {album.artist} came back with no "
                  "track list — I won't guess the tracks. Try again "
                  "and I'll pick another album.")
    return AlbumResult(album=album, tracks=tracks)


def format_album_summary(album: Album, delivered: list[str],
                         missing: list[tuple[str, str]]) -> str:
    """The final line for the album: what it is, how many tracks
    landed, and — for every track that didn't — its name and why."""
    total = len(delivered) + len(missing)
    lines = [f"🎵 {album.title} — {album.artist}",
             f"{len(delivered)} of {total} track"
             f"{'s' if total != 1 else ''} sent, in album order."]
    if missing:
        lines.append(f"missing ({len(missing)}):")
        for title, why in missing:
            why = " ".join((why or "couldn't download it").split())[:120]
            lines.append(f"  - {title} — {why}")
    return "\n".join(lines)
