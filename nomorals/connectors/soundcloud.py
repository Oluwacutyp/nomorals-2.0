"""SoundCloud connector — search, resolve, stream, playlists.

Docs: SoundCloud's public ``api-v2`` (the same API the web player uses).

Auth: none required for public content (``AuthMethod.NONE``). The API
needs a ``client_id`` query parameter; this connector obtains one the
free way — extracted from SoundCloud's own public JavaScript bundles,
exactly like the web player does. Set ``SOUNDCLOUD_CLIENT_ID`` to pin a
known-good id and skip discovery. Discovery results are cached in
memory; a 401 response invalidates the cache and rediscovers once.

Capabilities:
* ``search_tracks`` / ``search_playlists`` / ``search_users`` — catalog
  search (public content, no account needed)
* ``resolve`` — turn any soundcloud.com URL (track, playlist, user,
  including ``on.soundcloud.com`` short links) into an API object
* ``stream_url`` — the playable stream for a track. Prefers progressive
  MP3 (plays in mpv/ffplay directly); falls back to HLS (mpv plays it)
  when only HLS transcodings exist. Fails fast when a track is not
  streamable at all (private, geo/preview-restricted, removed).
* ``playlist_tracks`` / ``user_tracks`` — expand a playlist or artist
  page into playable track summaries

Everything here works without a SoundCloud account. User-only actions
(likes, reposts, follows, private-track access) need SoundCloud OAuth
and are intentionally not stubbed — the public API surface is the
complete, working surface.
"""

from __future__ import annotations

import os
import re
import urllib.parse
from typing import Any

from ..core.logging_setup import get_logger
from .base import (
    AuthMethod,
    Connector,
    ConnectorError,
    ConnectorStatus,
    ConnectResult,
)
from .registry import register_connector

__all__ = ["SoundCloudConnector", "SoundCloudError"]

_log = get_logger(__name__)

API_BASE = "https://api-v2.soundcloud.com"
HOMEPAGE = "https://soundcloud.com"

SOUNDCLOUD_CLIENT_ID_ENV = "SOUNDCLOUD_CLIENT_ID"

#: Browser-ish UA: soundcloud.com serves the full page (with script
#: tags) to browsers; bare python user-agents can get a thin shell.
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

#: Script bundles on soundcloud.com look like
#: https://a-v2.sndcdn.com/assets/0-abc123-3.js
_SCRIPT_RE = re.compile(
    r'https://a-v2\.sndcdn\.com/assets/[0-9a-z\-]+\.js'
)
#: Inside the bundle: client_id:"<32 alnum chars>"
_CLIENT_ID_RE = re.compile(r'client_id\s*[:=]\s*"([a-zA-Z0-9]{32})"')

#: In-memory client_id cache (per process). Env var always wins.
_client_id_cache: str = ""

#: Normalized track kinds the generic ``search`` accepts.
_SEARCH_KINDS = ("track", "playlist", "user", "album")


class SoundCloudError(ConnectorError):
    """A SoundCloud API call failed."""

    def __init__(
        self, message: str, *, status_code: int = 0, reason: str = ""
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.reason = reason


@register_connector
class SoundCloudConnector(Connector):
    """Devon's SoundCloud adapter: search, resolve, stream, playlists."""

    id = "soundcloud"
    name = "SoundCloud"
    description = (
        "SoundCloud public api-v2: track/playlist/user search, URL "
        "resolution (incl. on.soundcloud.com short links), playable "
        "stream URLs (progressive MP3, HLS fallback), playlist and artist "
        "track listings. No account needed — keyless public access."
    )
    auth_methods = (AuthMethod.NONE,)

    # ── lifecycle ────────────────────────────────────────────────

    def connect(self, **kwargs: Any) -> ConnectResult:
        """Validate the keyless public API path.

        There is no account to link — "connected" means a working
        ``client_id`` was discovered (or pinned via
        ``SOUNDCLOUD_CLIENT_ID``) and a live search round-trips. An
        honest failure here means SoundCloud changed something, never a
        fake success.
        """
        try:
            client_id = self._client_id()
        except SoundCloudError as exc:
            return ConnectResult(
                ok=False,
                account="",
                message=str(exc),
            )
        try:
            self.search_tracks("test", limit=1, client_id=client_id)
        except SoundCloudError as exc:
            return ConnectResult(
                ok=False,
                account="",
                message=f"client_id {client_id[:6]}… failed a live search: "
                        f"{exc}",
            )
        return ConnectResult(
            ok=True,
            account="soundcloud (public api, no account)",
            message=(
                "SoundCloud public API is live — search, resolve, and "
                "streaming work with no account. Pin "
                f"{SOUNDCLOUD_CLIENT_ID_ENV}={client_id[:6]}… to skip "
                "discovery."
            ),
        )

    def disconnect(self) -> None:
        global _client_id_cache
        _client_id_cache = ""
        _log.info("soundcloud: client_id cache cleared")

    def status(self) -> ConnectorStatus:
        if os.environ.get(SOUNDCLOUD_CLIENT_ID_ENV, "").strip() \
                or _client_id_cache:
            return ConnectorStatus(
                connected=self.test_connection(),
                account="soundcloud (public api, no account)",
                detail="keyless public access; client_id cached",
            )
        return ConnectorStatus(
            connected=self.test_connection(),
            account="soundcloud (public api, no account)",
            detail="keyless public access; client_id auto-discovered",
        )

    def test_connection(self) -> bool:
        try:
            self.search_tracks("test", limit=1)
            return True
        except ConnectorError:
            return False

    def connect_instructions(self) -> str:
        return (
            "SoundCloud needs no account: `nm connectors connect --name "
            "soundcloud` discovers a public client_id from SoundCloud's "
            "own web player assets and verifies a live search. To skip "
            f"discovery, set {SOUNDCLOUD_CLIENT_ID_ENV} to a known "
            "client_id."
        )

    # ── client_id discovery ──────────────────────────────────────

    def _client_id(self) -> str:
        """A working client_id: env pin, cache, or fresh discovery."""
        pinned = os.environ.get(SOUNDCLOUD_CLIENT_ID_ENV, "").strip()
        if pinned:
            return pinned
        global _client_id_cache
        if _client_id_cache:
            return _client_id_cache
        _client_id_cache = self._discover_client_id()
        return _client_id_cache

    def _invalidate_client_id(self) -> None:
        """Drop the cached id (401s) so the next call rediscovers."""
        global _client_id_cache
        if not os.environ.get(SOUNDCLOUD_CLIENT_ID_ENV, "").strip():
            _client_id_cache = ""

    def _discover_client_id(self) -> str:
        """Scrape a client_id from soundcloud.com's JS bundles."""
        try:
            resp = self.http.get(HOMEPAGE, headers={"User-Agent": _UA})
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise SoundCloudError(
                f"could not reach soundcloud.com to discover a client_id: "
                f"{exc} — set {SOUNDCLOUD_CLIENT_ID_ENV} manually"
            ) from exc
        if not resp.ok:
            raise SoundCloudError(
                f"soundcloud.com returned {resp.status} during client_id "
                f"discovery — set {SOUNDCLOUD_CLIENT_ID_ENV} manually"
            )
        scripts = _SCRIPT_RE.findall(resp.text)
        if not scripts:
            raise SoundCloudError(
                "soundcloud.com's page layout changed (no script bundles "
                f"found) — set {SOUNDCLOUD_CLIENT_ID_ENV} manually"
            )
        # The client_id lives in an app chunk; later bundles are the
        # bigger app chunks, so scan from the end.
        for src in reversed(scripts[-4:]):
            try:
                js = self.http.get(src, headers={"User-Agent": _UA})
            except Exception:  # noqa: BLE001 - try the next bundle
                continue
            if not js.ok:
                continue
            match = _CLIENT_ID_RE.search(js.text)
            if match:
                found = match.group(1)
                _log.info("soundcloud: discovered client_id %s…",
                          found[:6])
                return found
        raise SoundCloudError(
            "no client_id found in soundcloud.com's JS bundles (layout "
            f"changed) — set {SOUNDCLOUD_CLIENT_ID_ENV} manually"
        )

    # ── HTTP plumbing ────────────────────────────────────────────

    def _api(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        _retried: bool = False,
    ) -> Any:
        """One api-v2 call; errors become SoundCloudError.

        A 401 means the cached client_id died: invalidate, rediscover,
        and retry once — then fail fast.
        """
        client_id = self._client_id()
        query = dict(params or {})
        query["client_id"] = client_id
        url = f"{API_BASE}{path}"
        try:
            resp = self.http.get(url, params=query,
                                 headers={"User-Agent": _UA})
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise SoundCloudError(
                f"soundcloud request failed: {exc}") from exc
        if resp.status == 401 and not _retried and not os.environ.get(
                SOUNDCLOUD_CLIENT_ID_ENV, "").strip():
            _log.info("soundcloud: client_id rejected (401), rediscovering")
            self._invalidate_client_id()
            return self._api(path, params, _retried=True)
        if resp.status == 401:
            raise SoundCloudError(
                "soundcloud rejected the client_id (401) — it was revoked; "
                f"set a fresh {SOUNDCLOUD_CLIENT_ID_ENV} or reconnect to "
                "rediscover",
                status_code=401,
            )
        if resp.status == 404:
            raise SoundCloudError(
                f"soundcloud {path}: not found (404) — the track/playlist "
                "is private, deleted, or the id/URL is wrong",
                status_code=404,
            )
        if resp.status == 429:
            raise SoundCloudError(
                "soundcloud rate limit hit (429) — wait a minute and retry",
                status_code=429,
            )
        if not resp.ok:
            raise SoundCloudError(
                f"soundcloud {path} failed ({resp.status}): "
                f"{resp.text[:200]}",
                status_code=resp.status,
            )
        try:
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - invalid JSON is an error
            raise SoundCloudError(
                f"soundcloud {path} returned invalid JSON") from exc

    # ── resolve ──────────────────────────────────────────────────

    def resolve(self, url: str) -> dict[str, Any]:
        """Resolve any soundcloud.com URL (track/playlist/user, incl.
        ``on.soundcloud.com`` short links) to its API object."""
        url = (url or "").strip()
        if not url:
            raise SoundCloudError("empty SoundCloud URL")
        if not re.match(r"https?://", url, re.IGNORECASE):
            raise SoundCloudError(
                f"{url!r} is not a URL — pass a soundcloud.com link or a "
                "numeric id to get_track/get_playlist"
            )
        data = self._api("/resolve", {"url": url})
        if not isinstance(data, dict):
            raise SoundCloudError(
                f"could not resolve {url!r} — SoundCloud returned "
                "something unexpected"
            )
        kind = str(data.get("kind", ""))
        if kind == "track":
            return {"kind": "track", "track": self.summarize_track(data)}
        if kind == "playlist":
            return {"kind": "playlist",
                    "playlist": self.summarize_playlist(data)}
        if kind == "user":
            return {"kind": "user", "user": self.summarize_user(data)}
        raise SoundCloudError(
            f"resolved {url!r} to unsupported kind {kind!r}"
        )

    # ── search ───────────────────────────────────────────────────

    def search(
        self,
        kind: str,
        query: str,
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Generic catalog search.

        ``kind``: track, playlist, user, album.
        """
        kind = (kind or "").strip().lower()
        if kind not in _SEARCH_KINDS:
            raise SoundCloudError(
                f"unknown search kind {kind!r}: use "
                + ", ".join(_SEARCH_KINDS)
            )
        if not (query or "").strip():
            raise SoundCloudError("empty search query")
        data = self._api(
            f"/search/{kind}s",
            {"q": query.strip(), "limit": max(1, min(limit, 50)),
             "linked_partitioning": 1},
        )
        collection = (data or {}).get("collection", []) \
            if isinstance(data, dict) else []
        if kind in ("track", "album"):
            return [self.summarize_track(t) for t in collection]
        if kind == "playlist":
            return [self.summarize_playlist(p) for p in collection]
        return [self.summarize_user(u) for u in collection]

    def search_tracks(
        self, query: str, *, limit: int = 10, client_id: str = ""
    ) -> list[dict[str, Any]]:
        """Search public tracks; → normalized track summaries."""
        if client_id:
            global _client_id_cache
            _client_id_cache = client_id
        return self.search("track", query, limit=limit)

    def search_playlists(
        self, query: str, *, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Search public playlists; → normalized playlist summaries."""
        return self.search("playlist", query, limit=limit)

    def search_users(
        self, query: str, *, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Search users/artists; → normalized user summaries."""
        return self.search("user", query, limit=limit)

    # ── tracks ───────────────────────────────────────────────────

    def get_track(self, track_id: int | str) -> dict[str, Any]:
        """One track by id (``GET /tracks/{id}``)."""
        tid = self._numeric_id(track_id, "track")
        data = self._api(f"/tracks/{tid}")
        if not isinstance(data, dict):
            raise SoundCloudError(
                f"soundcloud returned an unexpected track payload for "
                f"id {tid}"
            )
        return self.summarize_track(data)

    def stream_url(
        self, track: int | str | dict[str, Any]
    ) -> dict[str, Any]:
        """The playable stream URL for a track.

        ``track``: id, soundcloud.com URL, or a track summary/full dict.
        Prefers progressive MP3 (direct file, plays anywhere); falls back
        to HLS (mpv/ffplay handle it). Fails fast when the track has no
        playable transcoding — private, removed, or restricted tracks.
        """
        full = self._full_track(track)
        summary = self.summarize_track(full)
        transcodings = ((full.get("media") or {}).get("transcodings")
                        or [])
        progressive = [t for t in transcodings
                       if (t.get("format") or {}).get("protocol")
                       == "progressive"]
        hls = [t for t in transcodings
               if (t.get("format") or {}).get("protocol") == "hls"]
        chosen = (progressive or hls or [None])[0]
        if chosen is None:
            policy = str(full.get("policy", ""))
            raise SoundCloudError(
                f"\"{summary['title']}\" has no playable stream "
                f"(policy={policy or 'unknown'}): the track is private, "
                "removed, or SoundCloud restricts it to previews only",
            )
        stream_api_url = str(chosen.get("url", ""))
        if not stream_api_url:
            raise SoundCloudError(
                f"\"{summary['title']}\" has a transcoding entry but no "
                "stream URL — SoundCloud changed its API shape"
            )
        data = self._api_raw_url(stream_api_url)
        final = str(data.get("url", ""))
        if not final:
            raise SoundCloudError(
                f"SoundCloud gave no final stream URL for "
                f"\"{summary['title']}\""
            )
        fmt = chosen.get("format") or {}
        return {
            "url": final,
            "protocol": fmt.get("protocol", ""),
            "mime_type": fmt.get("mime_type", ""),
            "track": summary,
        }

    def _full_track(self, track: int | str | dict[str, Any]
                    ) -> dict[str, Any]:
        """A full track payload from an id, URL, or (summary) dict."""
        if isinstance(track, dict):
            tid = track.get("id")
            if track.get("media") and isinstance(tid, int):
                return track  # already full
            if isinstance(tid, int):
                return self._api(f"/tracks/{tid}")
            url = str(track.get("permalink_url", ""))
            if url:
                resolved = self.resolve(url)
                if resolved["kind"] == "track":
                    return self._api(
                        f"/tracks/{resolved['track']['id']}")
            raise SoundCloudError(
                "cannot identify the track — pass an id, a soundcloud.com "
                "URL, or a track dict with an id"
            )
        text = str(track).strip()
        if text.isdigit():
            return self._api(f"/tracks/{text}")
        if re.match(r"https?://", text, re.IGNORECASE):
            resolved = self.resolve(text)
            if resolved["kind"] != "track":
                raise SoundCloudError(
                    f"{text!r} resolved to a {resolved['kind']}, not a "
                    "track"
                )
            return self._api(f"/tracks/{resolved['track']['id']}")
        raise SoundCloudError(
            f"{text!r} is not a track id or soundcloud.com URL"
        )

    def _api_raw_url(self, url: str) -> dict[str, Any]:
        """GET a full api-v2 URL (transcoding endpoints)."""
        client_id = self._client_id()
        sep = "&" if "?" in url else "?"
        try:
            resp = self.http.get(f"{url}{sep}client_id={client_id}",
                                 headers={"User-Agent": _UA})
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise SoundCloudError(
                f"soundcloud stream lookup failed: {exc}") from exc
        if not resp.ok:
            raise SoundCloudError(
                f"soundcloud stream lookup failed ({resp.status}): "
                f"{resp.text[:200]}",
                status_code=resp.status,
            )
        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 - invalid JSON is an error
            raise SoundCloudError(
                "soundcloud stream lookup returned invalid JSON") from exc
        return data if isinstance(data, dict) else {}

    # ── playlists / users ────────────────────────────────────────

    def get_playlist(self, playlist_id: int | str) -> dict[str, Any]:
        """One playlist by id (``GET /playlists/{id}``)."""
        pid = self._numeric_id(playlist_id, "playlist")
        data = self._api(f"/playlists/{pid}")
        if not isinstance(data, dict):
            raise SoundCloudError(
                f"soundcloud returned an unexpected playlist payload for "
                f"id {pid}"
            )
        return self.summarize_playlist(data)

    def playlist_tracks(
        self, target: int | str
    ) -> list[dict[str, Any]]:
        """Every track in a playlist: id or soundcloud.com playlist URL.

        Playlist payloads embed full track objects, so each summary
        carries its transcodings — ready for :meth:`stream_url`.
        """
        if isinstance(target, int) or str(target).strip().isdigit():
            pid = self._numeric_id(target, "playlist")
            data = self._api(f"/playlists/{pid}")
        else:
            url = str(target).strip()
            data = self._api("/resolve", {"url": url})
            if not isinstance(data, dict) or \
                    data.get("kind") != "playlist":
                raise SoundCloudError(
                    f"{url!r} did not resolve to a playlist"
                )
        if not isinstance(data, dict):
            raise SoundCloudError("soundcloud returned an unexpected "
                                  "playlist payload")
        tracks = data.get("tracks") or []
        return [self.summarize_track(t, full=t)
                for t in tracks if isinstance(t, dict)]

    def user_tracks(
        self, target: int | str, *, limit: int = 50
    ) -> list[dict[str, Any]]:
        """An artist's public tracks: user id or soundcloud.com user URL."""
        uid: str
        if isinstance(target, int) or str(target).strip().isdigit():
            uid = str(target).strip()
        else:
            url = str(target).strip()
            resolved = self.resolve(url)
            if resolved["kind"] != "user":
                raise SoundCloudError(
                    f"{url!r} did not resolve to a user"
                )
            uid = str(resolved["user"]["id"])
        data = self._api(f"/users/{uid}/tracks",
                         {"limit": max(1, min(limit, 200)),
                          "linked_partitioning": 1})
        collection = (data or {}).get("collection", []) \
            if isinstance(data, dict) else []
        return [self.summarize_track(t) for t in collection]

    # ── normalization ────────────────────────────────────────────

    @staticmethod
    def summarize_track(raw: dict[str, Any],
                        full: dict[str, Any] | None = None) -> dict[str, Any]:
        """A stable track dict for the player and the CLI."""
        src = full or raw
        user = raw.get("user") or {}
        artwork = str(raw.get("artwork_url") or "")
        return {
            "id": raw.get("id", 0),
            "title": str(raw.get("title", "")),
            "artist": str(user.get("username", "")),
            "duration_ms": int(raw.get("duration") or 0),
            "genre": str(raw.get("genre") or ""),
            "artwork_url": artwork,
            "permalink_url": str(raw.get("permalink_url", "")),
            "playback_count": int(raw.get("playback_count") or 0),
            "likes_count": int(raw.get("likes_count") or 0),
            "policy": str(raw.get("policy", "")),
            "streamable": bool(raw.get("streamable", False)),
            "kind": "track",
            "_full": src if full is not None else None,
        }

    @staticmethod
    def summarize_playlist(raw: dict[str, Any]) -> dict[str, Any]:
        user = raw.get("user") or {}
        tracks = raw.get("tracks") or []
        return {
            "id": raw.get("id", 0),
            "title": str(raw.get("title", "")),
            "artist": str(user.get("username", "")),
            "track_count": int(raw.get("track_count") or len(tracks)),
            "duration_ms": int(raw.get("duration") or 0),
            "artwork_url": str(raw.get("artwork_url") or ""),
            "permalink_url": str(raw.get("permalink_url", "")),
            "kind": "playlist",
        }

    @staticmethod
    def summarize_user(raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": raw.get("id", 0),
            "username": str(raw.get("username", "")),
            "track_count": int(raw.get("track_count") or 0),
            "followers_count": int(raw.get("followers_count") or 0),
            "permalink_url": str(raw.get("permalink_url", "")),
            "kind": "user",
        }

    @staticmethod
    def _numeric_id(value: int | str, what: str) -> str:
        text = str(value).strip()
        if not text.isdigit():
            raise SoundCloudError(
                f"{text!r} is not a numeric {what} id — pass the id or "
                "use resolve() with a soundcloud.com URL"
            )
        return text


def _reset_client_id_cache() -> None:
    """Tests: drop the discovery cache."""
    global _client_id_cache
    _client_id_cache = ""
