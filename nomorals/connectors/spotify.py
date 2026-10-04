"""Spotify connector — profile, playlists, and playback control.

Docs: https://developer.spotify.com/documentation/web-api

Auth: OAuth 2.0 authorization-code flow (``AuthMethod.OAUTH2``),
implemented inline against ``accounts.spotify.com`` (authorize) and
``https://accounts.spotify.com/api/token`` (exchange/refresh), mirroring
the shape of the shared ``_google_oauth`` mixin. The owner creates an app
in the Spotify developer dashboard (human step), grants access in their
browser, and pastes back the redirect URL. The refresh token is vaulted;
short-lived access tokens auto-refresh (Spotify rotates refresh tokens on
refresh, so the newest one is always stored).

Scopes requested:
* ``user-read-private`` — profile (account identity)
* ``playlist-read-private`` — list the owner's playlists
* ``playlist-modify-public`` / ``playlist-modify-private`` — create
  playlists, add tracks
* ``user-read-playback-state`` — devices, now-playing
* ``user-modify-playback-state`` — play / pause

Playback (``play``/``pause``) needs an *active* Spotify device — Spotify
only plays through an open app (phone, desktop, web player, speaker).
With no active device the connector fails fast with a clear message
instead of sending a command into the void. Creating playlists and adding
tracks are consequential: ``confirmed=True`` or a human checkpoint.
"""

from __future__ import annotations

import base64
import os
import time
import urllib.parse
from typing import Any

from ..core.logging_setup import get_logger
from .auth import prompt_secret
from ._confirm import confirm_or_checkpoint
from .base import (
    AuthMethod,
    Connector,
    ConnectorError,
    ConnectorStatus,
    ConnectResult,
)
from .checkpoints import (
    CheckpointKind,
    CheckpointState,
    HumanCheckpointPending,
)
from .registry import register_connector

__all__ = ["SpotifyConnector", "SpotifyError"]

_log = get_logger(__name__)

SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
API_BASE = "https://api.spotify.com/v1"

SPOTIFY_CLIENT_ID_ENV = "SPOTIFY_CLIENT_ID"
SPOTIFY_CLIENT_SECRET_ENV = "SPOTIFY_CLIENT_SECRET"

#: Suggested loopback redirect URI — the owner must add it in the app
#: dashboard ("Redirect URIs"), Spotify-side requirement.
DEFAULT_REDIRECT_URI = "http://127.0.0.1:8888/callback"

#: Refresh the access token this far ahead of expiry.
_REFRESH_LEEWAY = 60.0

_SCOPES = [
    "user-read-private",
    "playlist-read-private",
    "playlist-modify-public",
    "playlist-modify-private",
    "user-read-playback-state",
    "user-modify-playback-state",
]

_SEARCH_TYPES = ("track", "album", "artist", "playlist", "episode", "show")


class SpotifyError(ConnectorError):
    """A Spotify API call failed."""

    def __init__(
        self, message: str, *, status_code: int = 0, reason: str = ""
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.reason = reason


@register_connector
class SpotifyConnector(Connector):
    """Devon's Spotify adapter: profile, playlists, playback."""

    id = "spotify"
    name = "Spotify"
    description = (
        "Spotify Web API: profile, playlists (list/create/add tracks), "
        "devices, now-playing, search, play/pause. OAuth 2.0 with "
        "auto-refreshing tokens; playlist changes need explicit owner "
        "confirmation."
    )
    auth_methods = (AuthMethod.OAUTH2,)

    # ── lifecycle ────────────────────────────────────────────────

    def connect(
        self,
        *,
        client_id: str | None = None,
        client_secret: str | None = None,
        code: str | None = None,
        redirect_url: str | None = None,
        redirect_uri: str = "",
        scopes: list[str] | None = None,
        db: Any = None,
        context: Any = None,
    ) -> ConnectResult:
        """Complete the Spotify OAuth flow and vault the tokens.

        * ``redirect_url=...`` (or ``code=...``) — the URL Spotify sent the
          browser to after the grant (the code is parsed out of it), or
          the bare code. Exchanges it for tokens, validates via ``/v1/me``,
          stores the refresh token.
        * neither — prints the app-setup guide + authorization URL. With
          ``db`` the flow pauses at a human checkpoint the owner resolves
          with the redirect URL.
        """
        existing = self._load_credential()
        if existing is not None:
            raise ConnectorError(
                "spotify is already connected — one account per service. "
                "Disconnect first to switch accounts."
            )
        cid = self._client_id(client_id)
        redirect = (redirect_uri or "").strip() or DEFAULT_REDIRECT_URI
        wanted = list(scopes or _SCOPES)
        raw_code = (code or "").strip() or self._code_from_url(
            redirect_url or "")
        if raw_code:
            secret = self._client_secret(client_secret)
            tokens = self._exchange_code(cid, secret, raw_code, redirect)
            me = self._me(tokens.get("access_token", ""))
            user_id = str(me.get("id", ""))
            self._store_spotify_tokens(
                user_id or "spotify", cid, tokens,
                account=str(me.get("display_name", "") or user_id),
                email=str(me.get("email", "")),
                scopes=wanted,
            )
            return ConnectResult(
                ok=True,
                account=str(me.get("display_name", "") or user_id),
                scopes=wanted,
                message=(
                    f"connected to Spotify as "
                    f"{me.get('display_name', '') or user_id}. Tokens are "
                    "in the encrypted vault."
                ),
            )
        guide = self._connect_guide(cid, redirect, wanted)
        print(guide)
        if db is None:
            return ConnectResult(
                ok=False,
                account="",
                scopes=wanted,
                message=(
                    "create the Spotify app, grant access in your browser, "
                    "then call connect(..., redirect_url=<the URL Spotify "
                    "sent you to>)"
                ),
            )
        try:
            self.request_human(
                CheckpointKind.MANUAL_STEP,
                "Grant Spotify access",
                guide
                + "\n\nAfter Spotify redirects you, resolve this checkpoint "
                "with the full redirect URL, e.g. note "
                "'url=<paste the URL here>'.",
                db=db,
                context=context,
                resume_state={
                    "stage": "oauth_code",
                    "client_id": cid,
                    "redirect_uri": redirect,
                    "scopes": wanted,
                },
            )
        except HumanCheckpointPending as pending:
            return ConnectResult(
                ok=False,
                account="",
                scopes=wanted,
                message=(
                    "grant access in your browser, then resolve "
                    f"checkpoint {pending.checkpoint.id} with the redirect "
                    "URL"
                ),
            )
        raise ConnectorError(
            "access granted interactively but no redirect URL was "
            "captured — call connect(..., redirect_url=<URL>) with the "
            "URL Spotify sent your browser to"
        )

    def disconnect(self) -> None:
        self._clear_credential()

    def status(self) -> ConnectorStatus:
        cred = self._load_credential()
        if cred is None:
            return ConnectorStatus(
                connected=False,
                detail="not connected — run `nm connectors connect "
                       "--name spotify`",
            )
        try:
            me = self._me(self._access_token())
        except SpotifyError as exc:
            return ConnectorStatus(
                connected=False,
                account=str((cred.metadata or {}).get("account", "")),
                scopes=list((cred.metadata or {}).get("scopes", [])),
                last_checked=time.time(),
                detail=f"token rejected ({exc}): reconnect with a fresh grant",
            )
        return ConnectorStatus(
            connected=True,
            account=str(me.get("display_name", "") or me.get("id", "")),
            scopes=list((cred.metadata or {}).get("scopes", [])),
            last_checked=time.time(),
            detail="oauth tokens valid",
        )

    def test_connection(self) -> bool:
        if self._load_credential() is None:
            return False
        try:
            self._me(self._access_token())
            return True
        except ConnectorError:
            return False

    def connect_instructions(self) -> str:
        return self._connect_guide("", DEFAULT_REDIRECT_URI, _SCOPES)

    def resume_checkpoint(
        self,
        checkpoint: Any,
        *,
        db: Any,
        context: Any = None,
        client_secret: str | None = None,
    ) -> dict[str, Any]:
        """Continue after a human checkpoint resolved."""
        if checkpoint.state != CheckpointState.RESOLVED:
            raise ConnectorError(
                f"checkpoint {checkpoint.id} is {checkpoint.state.value}, "
                "not resolved — finish the human step first"
            )
        stage = (checkpoint.resume_state or {}).get("stage", "")
        if stage == "oauth_code":
            code = self._code_from_url(checkpoint.result_note or "")
            if not code:
                raise ConnectorError(
                    "the resolved checkpoint has no redirect URL — "
                    "resolve it again with note 'url=<redirect URL>'"
                )
            cid = str((checkpoint.resume_state or {}).get("client_id", ""))
            redirect = str(
                (checkpoint.resume_state or {}).get(
                    "redirect_uri", DEFAULT_REDIRECT_URI)
            )
            scopes = list(
                (checkpoint.resume_state or {}).get("scopes", _SCOPES)
            )
            secret = self._client_secret(client_secret)
            tokens = self._exchange_code(cid, secret, code, redirect)
            me = self._me(tokens["access_token"])
            user_id = str(me.get("id", ""))
            self._store_spotify_tokens(
                user_id or "spotify", cid, tokens,
                account=str(me.get("display_name", "") or user_id),
                email=str(me.get("email", "")),
                scopes=scopes,
            )
            return {"connected": True,
                    "account": str(me.get("display_name", "") or user_id)}
        if stage == "create_playlist":
            payload = dict((checkpoint.resume_state or {}).get("payload", {}))
            if not payload.get("name"):
                raise ConnectorError(
                    "the resolved checkpoint has no playlist payload — "
                    "it cannot create the playlist"
                )
            return self._create_playlist_now(payload)
        if stage == "add_tracks":
            payload = dict((checkpoint.resume_state or {}).get("payload", {}))
            if not payload.get("uris"):
                raise ConnectorError(
                    "the resolved checkpoint has no track payload — "
                    "it cannot add tracks"
                )
            return self._add_tracks_now(payload)
        raise ConnectorError(
            f"spotify cannot resume checkpoint stage {stage!r}"
        )

    # ── profile / library ────────────────────────────────────────

    def get_me(self) -> dict[str, Any]:
        """The connected profile (``GET /v1/me``)."""
        return self._me(self._access_token())

    def list_playlists(
        self, *, limit: int = 50, offset: int = 0
    ) -> dict[str, Any]:
        """The owner's playlists (``GET /v1/me/playlists``)."""
        data = self._api(
            "GET",
            "/v1/me/playlists",
            params={"limit": max(1, min(limit, 50)),
                    "offset": max(0, offset)},
        )
        return {
            "playlists": [
                self._summarize_playlist(p)
                for p in data.get("items", [])
            ],
            "total": data.get("total", 0),
        }

    @staticmethod
    def _summarize_playlist(raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": raw.get("id", ""),
            "name": raw.get("name", ""),
            "description": raw.get("description", ""),
            "public": raw.get("public"),
            "tracks_total": (raw.get("tracks") or {}).get("total", 0),
            "uri": raw.get("uri", ""),
            "url": ((raw.get("external_urls") or {}).get("spotify", "")),
        }

    def search(
        self,
        query: str,
        *,
        types: list[str] | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Search the catalog (``GET /v1/search``).

        ``types``: any of track, album, artist, playlist, episode, show.
        """
        if not (query or "").strip():
            raise ConnectorError("empty search query")
        wanted = [t for t in (types or ["track"]) if t in _SEARCH_TYPES]
        if not wanted:
            raise ConnectorError(
                f"no valid search types in {types}: use "
                + ", ".join(_SEARCH_TYPES)
            )
        return self._api(
            "GET",
            "/v1/search",
            params={
                "q": query,
                "type": ",".join(wanted),
                "limit": max(1, min(limit, 50)),
            },
        )

    # ── track lookup ─────────────────────────────────────────

    @staticmethod
    def normalize_uri(target: str) -> str:
        """Normalize anything Spotify-ish to a ``spotify:<type>:<id>`` URI.

        Accepts canonical URIs (``spotify:track:...`` — also album,
        playlist, episode, show, artist), ``open.spotify.com`` /
        ``play.spotify.com`` links, and fails fast on anything else.
        """
        text = (target or "").strip()
        if not text:
            raise SpotifyError("empty Spotify target")
        if text.startswith("spotify:"):
            parts = text.split(":")
            if len(parts) == 3 and parts[1] in (
                    "track", "album", "playlist", "episode", "show",
                    "artist") and parts[2]:
                return text
            raise SpotifyError(
                f"{text!r} is not a valid Spotify URI — want "
                "spotify:<track|album|playlist|episode|show|artist>:<id>"
            )
        parsed = urllib.parse.urlparse(text)
        host = (parsed.hostname or "").lower()
        if host in ("open.spotify.com", "play.spotify.com"):
            segs = [s for s in parsed.path.split("/") if s]
            # /track/<id>[/...] or /<locale>/track/<id>
            for i, seg in enumerate(segs):
                if seg in ("track", "album", "playlist", "episode",
                           "show", "artist") and i + 1 < len(segs):
                    return f"spotify:{seg}:{segs[i + 1]}"
        raise SpotifyError(
            f"{text!r} is not a Spotify URI or open.spotify.com link — "
            "pass spotify:track:<id>, an open.spotify.com URL, or use "
            "search() for a text query"
        )

    def get_track(self, track_id: str) -> dict[str, Any]:
        """One track's metadata (``GET /v1/tracks/{id}``).

        Accepts a bare id, a ``spotify:track:`` URI, or an
        open.spotify.com link.
        """
        uri = self.normalize_uri(track_id)
        if not uri.startswith("spotify:track:"):
            raise SpotifyError(
                f"{track_id!r} is not a track — get_track needs a track "
                "id/URI/link"
            )
        tid = uri.split(":")[2]
        data = self._api("GET", f"/v1/tracks/{tid}")
        return self._summarize_track(data)

    @staticmethod
    def _summarize_track(raw: dict[str, Any]) -> dict[str, Any]:
        artists = [a.get("name", "") for a in raw.get("artists", [])]
        return {
            "id": raw.get("id", ""),
            "uri": raw.get("uri", ""),
            "name": raw.get("name", ""),
            "artists": artists,
            "album": (raw.get("album") or {}).get("name", ""),
            "duration_ms": raw.get("duration_ms", 0),
            "url": (raw.get("external_urls") or {}).get("spotify", ""),
            "explicit": bool(raw.get("explicit", False)),
        }

    # ── playlists (confirmation-gated) ───────────────────────────

    def create_playlist(
        self,
        name: str,
        *,
        description: str = "",
        public: bool = False,
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Create a playlist on the owner's account.

        Creates a playlist under the owner's identity — consequential, so
        it needs ``confirmed=True`` (owner approved the exact name) or a
        human checkpoint via ``db``.
        """
        name = (name or "").strip()
        if not name:
            raise ConnectorError("a playlist name is required")
        payload = {
            "name": name,
            "description": description or "",
            "public": bool(public),
        }
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="create_playlist",
            title=f"Create Spotify playlist '{name}'",
            instructions="\n".join([
                "Devon wants to create this playlist on your Spotify.",
                f"Name: {name}",
                f"Visibility: {'public' if public else 'private'}",
                f"Description: {description or '(none)'}",
            ]),
            resume_state={"payload": payload},
        )
        return self._create_playlist_now(payload)

    def _create_playlist_now(self, payload: dict[str, Any]) -> dict[str, Any]:
        me = self._me(self._access_token())
        user_id = str(me.get("id", ""))
        if not user_id:
            raise SpotifyError("spotify did not return a user id")
        data = self._api(
            "POST",
            f"/v1/users/{user_id}/playlists",
            body={
                "name": payload["name"],
                "description": payload.get("description", ""),
                "public": bool(payload.get("public", False)),
            },
        )
        _log.info("spotify playlist created: %s (%s)",
                  data.get("name", "?"), data.get("id", "?"))
        return self._summarize_playlist(data)

    def add_tracks(
        self,
        playlist_id: str,
        uris: list[str],
        *,
        position: int | None = None,
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Add tracks to a playlist (``POST /v1/playlists/{id}/tracks``).

        ``uris`` are Spotify track URIs (``spotify:track:...``). Modifies
        the owner's playlist — needs ``confirmed=True`` or a checkpoint.
        """
        playlist_id = (playlist_id or "").strip()
        if not playlist_id:
            raise ConnectorError("a playlist id is required")
        clean = [u.strip() for u in (uris or []) if u.strip()]
        if not clean:
            raise ConnectorError("no track URIs given")
        bad = [u for u in clean
               if not u.startswith(("spotify:track:", "spotify:episode:"))]
        if bad:
            raise ConnectorError(
                f"invalid track URIs (need spotify:track:...): "
                f"{bad[:3]}{'...' if len(bad) > 3 else ''}"
            )
        payload = {"playlist_id": playlist_id, "uris": clean}
        if position is not None:
            if position < 0:
                raise ConnectorError("position cannot be negative")
            payload["position"] = position
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="add_tracks",
            title=f"Add {len(clean)} track(s) to playlist {playlist_id}",
            instructions="\n".join([
                "Devon wants to add tracks to your Spotify playlist.",
                f"Playlist: {playlist_id}",
                f"Tracks: {len(clean)}",
                *[f"  - {u}" for u in clean[:10]],
                *(["  ..."] if len(clean) > 10 else []),
            ]),
            resume_state={"payload": payload},
        )
        return self._add_tracks_now(payload)

    def _add_tracks_now(self, payload: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {"uris": payload["uris"]}
        if payload.get("position") is not None:
            body["position"] = payload["position"]
        data = self._api(
            "POST",
            f"/v1/playlists/{payload['playlist_id']}/tracks",
            body=body,
        )
        _log.info("spotify added %d track(s) to %s",
                  len(payload["uris"]), payload["playlist_id"])
        return {
            "playlist_id": payload["playlist_id"],
            "snapshot_id": data.get("snapshot_id", ""),
            "added": len(payload["uris"]),
        }

    # ── playback ─────────────────────────────────────────────────

    def list_devices(self) -> list[dict[str, Any]]:
        """Spotify Connect devices on the account (``/v1/me/player/devices``)."""
        data = self._api("GET", "/v1/me/player/devices")
        return [
            {
                "id": d.get("id", ""),
                "name": d.get("name", ""),
                "type": d.get("type", ""),
                "is_active": bool(d.get("is_active", False)),
                "is_restricted": bool(d.get("is_restricted", False)),
            }
            for d in data.get("devices", [])
        ]

    def now_playing(self) -> dict[str, Any]:
        """Currently playing track, or ``{"playing": False}`` (204)."""
        resp = self._raw(
            "GET", "/v1/me/player/currently-playing",
            params={"additional_types": "track,episode"},
        )
        if resp.status == 204:
            return {"playing": False}
        data = resp.json()
        item = (data.get("item") or {}) if isinstance(data, dict) else {}
        artists = [a.get("name", "") for a in item.get("artists", [])]
        return {
            "playing": bool(data.get("is_playing", False)),
            "track": item.get("name", ""),
            "artists": artists,
            "album": (item.get("album") or {}).get("name", ""),
            "progress_ms": data.get("progress_ms", 0),
            "device": ((data.get("device") or {}).get("name", "")),
            "uri": item.get("uri", ""),
        }

    def play(
        self,
        *,
        device_id: str = "",
        context_uri: str = "",
        uris: list[str] | None = None,
    ) -> dict[str, Any]:
        """Start/resume playback (``PUT /v1/me/player/play``).

        Needs an active device: Spotify only plays through an open app.
        Fails fast with a clear message when no device is active instead
        of silently doing nothing.
        """
        devices = self.list_devices()
        active = [d for d in devices if d["is_active"]]
        target = (device_id or "").strip()
        if not target and not active:
            names = ", ".join(d["name"] for d in devices) or "none found"
            raise SpotifyError(
                "no active Spotify device — open Spotify on a phone, "
                "computer, or the web player and start anything once, "
                f"then retry (devices seen: {names})"
            )
        body: dict[str, Any] = {}
        if context_uri:
            body["context_uri"] = context_uri
        if uris:
            body["uris"] = list(uris)
        self._raw(
            "PUT",
            "/v1/me/player/play",
            params={"device_id": target} if target else None,
            body=body,
        )
        return {"playing": True,
                "device_id": target or (active[0]["id"] if active else "")}

    def pause(self, *, device_id: str = "") -> dict[str, Any]:
        """Pause playback (``PUT /v1/me/player/pause``).

        Fails fast when nothing is playing instead of returning a soft
        no-op.
        """
        try:
            self._raw(
                "PUT",
                "/v1/me/player/pause",
                params={"device_id": device_id} if device_id else None,
                body={},
            )
        except SpotifyError as exc:
            if exc.status_code == 404:
                raise SpotifyError(
                    "nothing is playing — pause refused because there is "
                    "no active playback (open Spotify and play something "
                    "first)"
                ) from exc
            raise
        return {"playing": False}

    # ── OAuth plumbing ───────────────────────────────────────────

    def _client_id(self, client_id: str | None) -> str:
        cid = ((client_id or "").strip()
               or os.environ.get(SPOTIFY_CLIENT_ID_ENV, "").strip())
        if not cid:
            raise ConnectorError(
                "no Spotify client id — create an app at "
                "https://developer.spotify.com/dashboard and pass "
                f"client_id=... or set {SPOTIFY_CLIENT_ID_ENV}"
            )
        return cid

    def _client_secret(self, client_secret: str | None) -> str:
        return ((client_secret or "").strip() or prompt_secret(
            "Spotify client secret", env_var=SPOTIFY_CLIENT_SECRET_ENV
        ))

    def authorize_url(
        self, client_id: str, redirect_uri: str, scopes: list[str]
    ) -> str:
        """The URL the owner opens to grant access."""
        params = {
            "client_id": client_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": " ".join(scopes),
            "show_dialog": "false",
        }
        return f"{SPOTIFY_AUTH_URL}?{urllib.parse.urlencode(params)}"

    def _basic_auth(self, client_id: str, client_secret: str) -> str:
        raw = f"{client_id}:{client_secret}".encode()
        return "Basic " + base64.b64encode(raw).decode("ascii")

    def _token_request(
        self, client_id: str, client_secret: str,
        form: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            resp = self.http.post_form(
                SPOTIFY_TOKEN_URL,
                form,
                headers={
                    "Authorization": self._basic_auth(client_id,
                                                      client_secret),
                },
            )
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise SpotifyError(f"spotify token request failed: {exc}") from exc
        if not resp.ok:
            raise SpotifyError(
                f"spotify rejected the token request ({resp.status}): "
                f"{resp.text[:200]}",
                status_code=resp.status,
            )
        data = resp.json()
        if not isinstance(data, dict):
            raise SpotifyError(
                "spotify token request returned an unexpected response"
            )
        return data

    def _exchange_code(
        self,
        client_id: str,
        client_secret: str,
        code: str,
        redirect_uri: str,
    ) -> dict[str, Any]:
        if not code:
            raise ConnectorError(
                "empty authorization code — open the authorization URL, "
                "grant access, and hand back the redirect URL"
            )
        return self._token_request(
            client_id, client_secret,
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
            },
        )

    def _refresh(
        self, client_id: str, client_secret: str, refresh_token: str
    ) -> dict[str, Any]:
        return self._token_request(
            client_id, client_secret,
            {"grant_type": "refresh_token", "refresh_token": refresh_token},
        )

    def _store_spotify_tokens(
        self,
        username: str,
        client_id: str,
        tokens: dict[str, Any],
        *,
        account: str = "",
        email: str = "",
        scopes: list[str] | None = None,
    ) -> None:
        refresh = str(tokens.get("refresh_token", ""))
        access = str(tokens.get("access_token", ""))
        if not refresh:
            raise ConnectorError(
                "spotify did not return a refresh token on exchange"
            )
        if not access:
            raise ConnectorError(
                "spotify did not return an access token on exchange"
            )
        expires_in = float(tokens.get("expires_in", 3600) or 3600)
        self._store_credential(
            username,
            refresh,
            credential_type="oauth_token",
            scopes=scopes,
            metadata={
                "client_id": client_id,
                "account": account,
                "email": email,
                "access_token": access,
                "access_expires_at": time.time() + expires_in,
            },
        )
        _log.info("spotify: tokens stored for %s", username)

    def _access_token(self) -> str:
        """A fresh access token, refreshing within the leeway."""
        cred = self._require_credential()
        meta = cred.metadata or {}
        client_id = str(meta.get("client_id", ""))
        access = str(meta.get("access_token", ""))
        expires_at = float(meta.get("access_expires_at", 0) or 0)
        if access and client_id and expires_at - time.time() > _REFRESH_LEEWAY:
            return access
        return self._force_refresh_access()

    def _force_refresh_access(self) -> str:
        """Refresh the access token right now and store the new pair.

        Used by ``_access_token`` when the leeway check fails, and by
        ``_raw`` when Spotify rejected the leeway-checked token anyway
        (clock skew, an early-revoked grant, or a server-side token
        rotation the leeway check cannot see).
        """
        cred = self._require_credential()
        meta = cred.metadata or {}
        client_id = str(meta.get("client_id", ""))
        secret = self._client_secret(None)
        tokens = self._refresh(client_id, secret, cred.password)
        new_access = str(tokens.get("access_token", ""))
        if not new_access:
            raise ConnectorError(
                "spotify did not return an access token on refresh"
            )
        # Spotify rotates refresh tokens: always store the newest one.
        merged = dict(tokens)
        merged.setdefault("refresh_token", cred.password)
        self._store_spotify_tokens(
            cred.username,
            client_id,
            merged,
            account=str(meta.get("account", "")),
            email=str(meta.get("email", "")),
            scopes=list(meta.get("scopes", [])),
        )
        _log.info("spotify: access token refreshed")
        return new_access

    def _connect_guide(
        self, client_id: str, redirect_uri: str, scopes: list[str]
    ) -> str:
        url = self.authorize_url(client_id or "<your-client-id>",
                                 redirect_uri, scopes)
        return "\n".join([
            "Connect your Spotify account (only you can grant this):",
            "1. Create an app: https://developer.spotify.com/dashboard",
            "   -> Create app, then in Settings add this Redirect URI:",
            f"   {redirect_uri}",
            "   (Spotify requires the URI to be registered — it is never",
            "   contacted; you paste the redirect URL back by hand.)",
            "2. Open this URL in your browser and agree:",
            f"   {url}",
            "3. Spotify redirects your browser to a URL starting with",
            f"   {redirect_uri}?code=... — copy that whole URL.",
            "4. Hand it to Devon: connect(..., redirect_url=<paste it>).",
            "The refresh token lands in the encrypted vault; Devon never",
            "sees your Spotify password.",
        ])

    @staticmethod
    def _code_from_url(url: str) -> str:
        """Pull the authorization code out of a redirect URL or bare code."""
        text = (url or "").strip()
        if not text:
            return ""
        if "://" not in text and "?" not in text and "=" not in text:
            return text  # a bare code was pasted
        try:
            query = urllib.parse.urlparse(text).query
            params = urllib.parse.parse_qs(query)
            codes = params.get("code", [])
            if codes:
                return codes[0]
        except Exception:  # noqa: BLE001 - best effort only
            pass
        return ""

    # ── HTTP plumbing ────────────────────────────────────────────

    def _require_credential(self):
        cred = self._load_credential()
        if cred is None:
            raise ConnectorError(
                "spotify is not connected — run "
                "`nm connectors connect --name spotify` first"
            )
        return cred

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._access_token()}"}

    def _send(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> Any:
        """One HTTP exchange against the Spotify API."""
        try:
            if method == "GET":
                resp = self.http.get(url, headers=headers, params=params)
            elif method == "POST":
                resp = self.http.post_json(url, body or {}, headers=headers,
                                           params=params)
            elif method == "PUT":
                resp = self.http.put_json(url, body or {}, headers=headers,
                                          params=params)
            else:
                raise ConnectorError(f"unsupported method {method}")
        except ConnectorError:
            raise
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise SpotifyError(f"spotify request failed: {exc}") from exc
        return resp

    def _raw(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> Any:
        """One Spotify Web API call; errors become SpotifyError."""
        url = f"{API_BASE}{path}"
        resp = self._send(method, url, self._headers(), params=params,
                          body=body)
        if resp.status == 401:
            # The leeway-checked token was rejected anyway (clock skew,
            # an early-revoked grant, a server-side rotation).  Refresh
            # once and retry the exact same request transparently —
            # only when the retry also fails do we fail loudly.
            _log.info("spotify 401 on %s %s — forcing token refresh and "
                      "retrying once", method, path)
            try:
                headers = {"Authorization":
                           f"Bearer {self._force_refresh_access()}"}
            except (ConnectorError, SpotifyError):
                raise  # the refresh itself failed — surface the real cause
            resp = self._send(method, url, headers, params=params, body=body)
            if resp.status == 401:
                raise SpotifyError(
                    "spotify rejected the access token (401) — the grant was "
                    "revoked; reconnect with a fresh grant",
                    status_code=401,
                )
        if resp.status == 204:
            # 204 is success for play/pause (empty body). For
            # currently-playing it means "nothing playing" — the caller
            # decides via resp.status.
            return resp
        if resp.status == 403:
            raise SpotifyError(
                "spotify refused (403): the grant lacks this scope — "
                "reconnect granting the missing scope",
                status_code=403,
            )
        if resp.status == 429:
            retry = resp.headers.get("retry-after", "")
            raise SpotifyError(
                "spotify rate limit exceeded (429)"
                + (f" — retry after {retry}s" if retry else ""),
                status_code=429,
            )
        if resp.status == 404:
            raise SpotifyError(
                f"spotify {method} {path}: not found (404) — "
                f"{self._api_reason(resp) or 'check the id'}",
                status_code=404,
                reason=self._api_reason(resp),
            )
        if not resp.ok:
            raise SpotifyError(
                f"spotify {method} {path} failed ({resp.status}): "
                f"{resp.text[:200]}",
                status_code=resp.status,
                reason=self._api_reason(resp),
            )
        return resp

    def _api(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resp = self._raw(method, path, params=params, body=body)
        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 - invalid JSON is an error
            raise SpotifyError(
                f"spotify {method} {path} returned invalid JSON"
            ) from exc
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _api_reason(resp: Any) -> str:
        try:
            body = resp.json()
            return str((body.get("error") or {}).get("reason", "")
                       or (body.get("error") or {}).get("message", ""))
        except Exception:  # noqa: BLE001 - best effort only
            pass
        return ""

    def _me(self, access_token: str) -> dict[str, Any]:
        """The connected profile (``GET /v1/me``)."""
        try:
            resp = self.http.get(
                f"{API_BASE}/v1/me",
                headers={"Authorization": f"Bearer {access_token}"},
            )
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise SpotifyError(f"spotify profile lookup failed: {exc}") from exc
        if resp.status == 401:
            raise SpotifyError(
                "spotify rejected the access token (401) — the grant was "
                "revoked; reconnect with a fresh grant",
                status_code=401,
            )
        if not resp.ok:
            raise SpotifyError(
                f"spotify profile lookup failed ({resp.status}): "
                f"{resp.text[:200]}",
                status_code=resp.status,
            )
        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 - invalid JSON is an error
            raise SpotifyError(
                "spotify profile lookup returned invalid JSON"
            ) from exc
        return data if isinstance(data, dict) else {}
