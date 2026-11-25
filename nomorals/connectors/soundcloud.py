"""SoundCloud connector — search, resolve, stream, playlists.

Docs: SoundCloud's public ``api-v2`` (the same API the web player uses).

Auth: none required for public content (``AuthMethod.NONE``). The API
needs a ``client_id`` query parameter; this connector obtains one the
free way — extracted from SoundCloud's own public JavaScript bundles,
exactly like the web player does. Set ``SOUNDCLOUD_CLIENT_ID`` to pin a
known-good id and skip discovery. Discovery results are cached in
memory; a 401 OR 403 response invalidates the cache and rediscovers
once (yt-dlp's own SoundCloud extractor retries on both — a
rate-limited or revoked id can refuse either way). A discovered id
may also be handed in from outside (:meth:`set_client_id`) — the
downloader resolves it through
:mod:`nomorals.media.soundcloud_credentials` (env pin → vault-stored
id → live discovery) and persists fresh discoveries in the vault, so
a restart does not re-scrape the homepage on every boot.

Session: a signed-in owner's ``oauth_token`` cookie — the cookie
SoundCloud's own web player sets and the one yt-dlp's SoundCloud
extractor reads from the cookie jar (``SoundcloudBaseIE.
_real_initialize``, yt-dlp master) — can be handed in with
:meth:`set_session_cookies` and is then sent on api-v2 calls, so
login-gated tracks resolve when the owner has a session. The
connector never logs a cookie value, and never logs in itself.

Capabilities:
* ``search_tracks`` / ``search_playlists`` / ``search_users`` — catalog
  search (public content, no account needed)
* ``resolve`` — turn any soundcloud.com URL (track, playlist, user,
  including ``on.soundcloud.com`` short links) into an API object
* ``stream_url`` — the playable stream for a track. Prefers progressive
  MP3 (plays in mpv/ffplay directly); falls back to HLS (mpv plays it)
  when only HLS transcodings exist. Fails fast when a track is not
  streamable at all (private, geo/preview-restricted, removed) — with
  the block classified (``SoundCloudError.reason``: preview / drm /
  region / login / private), never a bare "protected". The track
  payload's ``track_authorization`` token is forwarded on the
  transcoding call, as api-v2 expects for authorization-gated tracks.
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

__all__ = [
    "SoundCloudConnector",
    "SoundCloudError",
    "BLOCK_PREVIEW",
    "BLOCK_DRM",
    "BLOCK_REGION",
    "BLOCK_LOGIN",
    "BLOCK_PRIVATE",
    "block_reason_text",
    "classify_block",
]

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

#: A client_id is 32 alphanumeric characters (the shape SoundCloud's
#: own bundles carry, and the shape :func:`_CLIENT_ID_RE` matches).
_CLIENT_ID_SHAPE = re.compile(r"^[a-zA-Z0-9]{32}$")

#: Statuses that mean the cached client_id died (revoked or
#: rate-limited): invalidate, rediscover once, then fail fast.
#: yt-dlp's own SoundCloud extractor retries on both 401 AND 403
#: (``SoundcloudBaseIE._call_api``, master) — a dead id can refuse
#: either way, so this connector does the same.
_STALE_CLIENT_ID_STATUSES = (401, 403)

#: Normalized track kinds the generic ``search`` accepts.
_SEARCH_KINDS = ("track", "playlist", "user", "album")

#: Structured block classes — WHY a track will not stream in full.
#: Callers (resolver, playback, the download ladder) carry the class,
#: not a guessed sentence, and the user surface renders it with
#: :func:`block_reason_text`.  Classes:
#: ``preview`` — SoundCloud serves only a ~30 s preview (policy
#: PREVIEW/SNIP, or the stream it handed back IS a preview);
#: ``drm`` — the stream is DRM-protected;
#: ``region`` — the rights holder blocks the track in this region
#: (policy BLOCK/DENY, or an explicit geo refusal);
#: ``login`` — the stream needs an authorized session (the track's
#: ``track_authorization`` or a signed-in account);
#: ``private`` — the track is private or removed.
BLOCK_PREVIEW = "preview"
BLOCK_DRM = "drm"
BLOCK_REGION = "region"
BLOCK_LOGIN = "login"
BLOCK_PRIVATE = "private"

_BLOCK_REASON_TEXT = {
    BLOCK_PREVIEW: "SoundCloud only serves a ~30-second preview of "
                   "this track",
    BLOCK_DRM: "this track's SoundCloud stream is DRM-protected",
    BLOCK_REGION: "SoundCloud blocks this track in this region",
    BLOCK_LOGIN: "SoundCloud wants a signed-in account for this track",
    BLOCK_PRIVATE: "this track is private or has been removed",
}


def block_reason_text(reason: str) -> str:
    """The human sentence for a structured block class ("" if none)."""
    return _BLOCK_REASON_TEXT.get(str(reason or ""), "")


def classify_block(*, policy: str = "", status: int = 0,
                   text: str = "") -> str:
    """Classify a SoundCloud block into a :data:`BLOCK_*` class.

    Evidence, in order: the api-v2 ``policy`` on the track payload,
    the HTTP status of the refused stream call, and markers in the
    API's own error text.  Returns "" when nothing identifies the
    block — an unclassified refusal is reported as-is, never dressed
    up as a class it didn't earn.
    """
    pol = str(policy or "").upper()
    low = str(text or "").lower()
    if "drm" in low or "protected content" in low:
        return BLOCK_DRM
    if pol in ("PREVIEW", "SNIP"):
        return BLOCK_PREVIEW
    if "preview" in low and ("only" in low or "30" in low):
        return BLOCK_PREVIEW
    if pol in ("BLOCK", "DENY"):
        return BLOCK_REGION
    if any(marker in low for marker in (
            "not available in your country", "geo", "region",
            "territory")):
        return BLOCK_REGION
    if status in (401, 403) or any(marker in low for marker in (
            "unauthorized", "authorization", "login", "log in",
            "sign in", "authenticated")):
        return BLOCK_LOGIN
    if status == 404 or "private" in low or "not found" in low:
        return BLOCK_PRIVATE
    return ""


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

    def __init__(self, vault: Any, http: Any = None, *,
                 page_fetcher: Any = None) -> None:
        """``page_fetcher`` (optional) is the web-expansion rung for
        discovery: a zero-argument callable returning the
        soundcloud.com homepage HTML (or "" when it cannot) — the
        downloader supplies one that opens the page in the existing
        persistent browser session, so a homepage fetch that the
        plain HTTP client cannot make (bot check on this box's IP)
        still yields the script tags discovery scans.  None means
        discovery is HTTP-only, as before."""
        super().__init__(vault, http=http)
        self._page_fetcher = page_fetcher
        #: The owner's session cookies for soundcloud.com, by name
        #: (``oauth_token`` — see :meth:`set_session_cookies`).  Values
        #: live here in memory only and are never logged.
        self._session_cookies: dict[str, str] = {}

    def set_page_fetcher(self, page_fetcher: Any) -> None:
        """Install (or clear, with None) the discovery page fetcher."""
        self._page_fetcher = page_fetcher

    def set_client_id(self, client_id: str) -> bool:
        """Install a client_id resolved elsewhere (the downloader's
        credential resolution: env pin → vault → live discovery).
        A malformed value is refused (False) and the cache is left
        untouched — a bad id must never overwrite a working one.
        The env pin still wins over an installed id at resolve time.
        """
        cid = (client_id or "").strip()
        if not _CLIENT_ID_SHAPE.match(cid):
            _log.warning("soundcloud: refusing malformed client_id "
                         "(not 32 alphanumeric characters)")
            return False
        global _client_id_cache
        _client_id_cache = cid
        return True

    def client_id(self) -> str:
        """A working client_id (public): env pin, cache, or fresh
        discovery — the same resolution :meth:`_client_id` performs."""
        return self._client_id()

    def set_session_cookies(
            self, cookies: Any) -> int:
        """Install the owner's soundcloud.com session cookies for
        api-v2 calls (``oauth_token`` is the session cookie
        SoundCloud's web player sets — yt-dlp's SoundCloud
        extractor reads exactly that name from the cookie jar).
        ``cookies`` is estate-shaped records (``{"name", "value",
        ...}``) or a name→value mapping.  Returns how many cookies
        were stored.  Values are never logged; only the count is."""
        stored: dict[str, str] = {}
        items: Any = []
        if isinstance(cookies, dict):
            items = list(cookies.items())
        elif cookies:
            items = list(cookies)
        for item in items:
            if isinstance(item, tuple) and len(item) == 2:
                name, value = str(item[0] or "").strip(), str(item[1] or "")
            elif isinstance(item, dict):
                name = str(item.get("name") or "").strip()
                value = str(item.get("value") or "")
            else:
                continue
            if name and value:
                stored[name] = value
        self._session_cookies = stored
        if stored:
            _log.info("soundcloud: session cookies installed (%d)",
                      len(stored))
        return len(stored)

    def _cookie_header(self) -> dict[str, str]:
        """``Cookie`` header for api-v2 calls, "" when no session is
        installed.  Built per call so a session installed after
        construction still rides the next request."""
        if not self._session_cookies:
            return {}
        return {"Cookie": "; ".join(
            f"{name}={value}"
            for name, value in self._session_cookies.items())}

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

    def _homepage_html(self) -> str:
        """The soundcloud.com homepage HTML, by the route that works.

        The plain HTTP fetch first; when it fails (exception, or a
        non-OK status — the bot check that kills discovery on some
        boxes), the injected page fetcher (the rendered browser
        session) is the second rung.  "" when neither route yields
        the page.  Never raises — discovery reports the failure
        with the manual-pin instruction instead."""
        headers = {"User-Agent": _UA, **self._cookie_header()}
        try:
            resp = self.http.get(HOMEPAGE, headers=headers)
        except Exception:  # noqa: BLE001 - the fetcher is the fallback
            resp = None
        if resp is not None and resp.ok:
            return str(resp.text or "")
        fetcher = self._page_fetcher
        if not callable(fetcher):
            return ""
        try:
            html = fetcher()
        except Exception as exc:  # noqa: BLE001 - a dead fetcher is ""
            _log.info("soundcloud: browser page fetch failed: %s", exc)
            return ""
        if isinstance(html, dict):
            html = html.get("html", "")
        return str(html or "")

    def _discover_client_id(self) -> str:
        """Scrape a client_id from soundcloud.com's JS bundles.

        The homepage arrives by the route that works (plain HTTP,
        else the browser session — see :meth:`_homepage_html`); the
        bundles themselves are fetched over HTTP with the session
        cookies riding along, so a signed-in session's homepage is
        less likely to be bot-checked in the first place.  Honest
        about the source: the id is SoundCloud's own public
        client_id, embedded in its public JS — discovered, never
        invented, and refreshed whenever the API refuses it."""
        html = self._homepage_html()
        if not html:
            raise SoundCloudError(
                "could not reach soundcloud.com to discover a client_id "
                "(both the direct fetch and the browser page fetch "
                "failed) — set "
                f"{SOUNDCLOUD_CLIENT_ID_ENV} manually"
            )
        scripts = _SCRIPT_RE.findall(html)
        if not scripts:
            raise SoundCloudError(
                "soundcloud.com's page layout changed (no script bundles "
                f"found) — set {SOUNDCLOUD_CLIENT_ID_ENV} manually"
            )
        # The client_id lives in an app chunk; later bundles are the
        # bigger app chunks, so scan from the end.
        headers = {"User-Agent": _UA, **self._cookie_header()}
        for src in reversed(scripts[-4:]):
            try:
                js = self.http.get(src, headers=headers)
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

        A 401 or a 403 means the cached client_id died (revoked or
        rate-limited — either status can carry it, as yt-dlp's own
        extractor treats them): invalidate, rediscover, and retry
        once — then fail fast.  A 403 that survives a fresh id is
        not a dead id — the refusal itself is reported, with the
        block class the API's own words earn (a per-track region or
        login refusal is not a credential failure).
        """
        client_id = self._client_id()
        query = dict(params or {})
        query["client_id"] = client_id
        url = f"{API_BASE}{path}"
        try:
            resp = self.http.get(
                url, params=query,
                headers={"User-Agent": _UA, **self._cookie_header()})
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise SoundCloudError(
                f"soundcloud request failed: {exc}") from exc
        if resp.status in _STALE_CLIENT_ID_STATUSES and not _retried \
                and not os.environ.get(
                    SOUNDCLOUD_CLIENT_ID_ENV, "").strip():
            _log.info("soundcloud: client_id rejected (%s), rediscovering",
                      resp.status)
            self._invalidate_client_id()
            return self._api(path, params, _retried=True)
        if resp.status == 401:
            raise SoundCloudError(
                "soundcloud rejected the client_id (401) — it was revoked; "
                f"set a fresh {SOUNDCLOUD_CLIENT_ID_ENV} or reconnect to "
                "rediscover",
                status_code=401,
            )
        if resp.status == 403:
            reason = classify_block(status=403, text=resp.text[:200])
            detail = ""
            if reason:
                detail = f" — {block_reason_text(reason)}"
            raise SoundCloudError(
                f"soundcloud refused this request (403){detail} — the "
                "client_id was already refreshed once, so this is the "
                "track's own refusal, not a stale credential",
                status_code=403,
                reason=reason,
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
        # Preview honesty (owner audit N3): a transcoding SoundCloud
        # itself marks ``snipped`` is a ~30 s preview, not the track
        # — and a SNIP/PREVIEW policy says the same about the track.
        # Full transcodings win wherever one exists; when only
        # previews exist the call fails as preview instead of
        # landing a 30 s clip as the track.  The old shape checked
        # only ``"/preview/"`` in the resolved URL — true for the
        # progressive preview, false for the HLS preview, so the
        # same track landed as success over HLS and raised over
        # progressive.  The payload's own evidence decides now, on
        # both protocols; the URL check below stays as backup.
        policy = str(full.get("policy", ""))
        full_stream = [t for t in transcodings
                       if not t.get("snipped")]
        pool = full_stream or transcodings
        progressive = [t for t in pool
                       if (t.get("format") or {}).get("protocol")
                       == "progressive"]
        hls = [t for t in pool
               if (t.get("format") or {}).get("protocol") == "hls"]
        chosen = (progressive or hls or [None])[0]
        if chosen is None:
            reason = classify_block(policy=policy) or BLOCK_PRIVATE
            raise SoundCloudError(
                f"\"{summary['title']}\" has no playable stream: "
                f"{block_reason_text(reason)} "
                f"(policy={policy or 'unknown'})",
                reason=reason,
            )
        if chosen.get("snipped") or policy.upper() in ("PREVIEW", "SNIP"):
            raise SoundCloudError(
                f"\"{summary['title']}\" is preview-only on SoundCloud "
                f"(~30s): {block_reason_text(BLOCK_PREVIEW)}; falling "
                "back to full-track sources",
                reason=BLOCK_PREVIEW,
            )
        stream_api_url = str(chosen.get("url", ""))
        if not stream_api_url:
            raise SoundCloudError(
                f"\"{summary['title']}\" has a transcoding entry but no "
                "stream URL — SoundCloud changed its API shape"
            )
        # Authorization-gated tracks carry a per-track token in the
        # payload; api-v2 expects it forwarded on the transcoding
        # call alongside client_id — without it those calls refuse
        # even with a valid client_id.
        data = self._api_raw_url(
            stream_api_url,
            track_authorization=str(
                full.get("track_authorization", "") or ""))
        final = str(data.get("url", ""))
        if not final:
            raise SoundCloudError(
                f"SoundCloud gave no final stream URL for "
                f"\"{summary['title']}\""
            )
        if "/preview/" in final:
            raise SoundCloudError(
                f"\"{summary['title']}\" is preview-only on SoundCloud "
                f"(~30s): {block_reason_text(BLOCK_PREVIEW)}; falling "
                "back to full-track sources",
                reason=BLOCK_PREVIEW,
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

    def _api_raw_url(self, url: str,
                     track_authorization: str = "") -> dict[str, Any]:
        """GET a full api-v2 URL (transcoding endpoints).

        ``track_authorization`` — the per-track token from the track
        payload — is forwarded when present: api-v2 expects it
        alongside ``client_id`` for authorization-gated tracks, and
        those stream calls refuse without it.
        """
        client_id = self._client_id()
        sep = "&" if "?" in url else "?"
        wire = f"{url}{sep}client_id={client_id}"
        if track_authorization:
            wire += (f"&track_authorization="
                     f"{urllib.parse.quote(track_authorization, safe='')}")
        try:
            resp = self.http.get(
                wire,
                headers={"User-Agent": _UA, **self._cookie_header()})
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise SoundCloudError(
                f"soundcloud stream lookup failed: {exc}") from exc
        if not resp.ok:
            reason = classify_block(status=resp.status,
                                    text=resp.text[:200])
            detail = ""
            if reason:
                detail = f" — {block_reason_text(reason)}"
            raise SoundCloudError(
                f"soundcloud stream lookup failed ({resp.status}): "
                f"{resp.text[:200]}{detail}",
                status_code=resp.status,
                reason=reason,
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
            #: WHY SoundCloud won't serve this track in full, when it
            #: won't — a BLOCK_* class from the payload's own evidence
            #: (policy / streamable), "" for a freely streamable one.
            #: Search/resolve consumers carry it to the user surface
            #: instead of collapsing every block to "protected".
            "block_reason": classify_block(
                policy=str(raw.get("policy", ""))) if (
                    not raw.get("streamable", False)
                    or str(raw.get("policy", "")).upper()
                    in ("PREVIEW", "SNIP", "BLOCK", "DENY")) else "",
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
