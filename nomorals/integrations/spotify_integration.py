"""Spotify integration via Web API.

Supports:
- Search tracks, albums, artists, playlists
- Manage playlists (create, add, remove tracks)
- Playback control (play, pause, skip, volume)
- Get currently playing track
- Podcast management

Usage:
    spotify = SpotifyIntegration(account_manager, session_manager)
    
    # Search
    results = await spotify.search("Bohemian Rhapsody", type="track")
    
    # Get currently playing
    current = await spotify.get_currently_playing()
    
    # Create playlist
    playlist = await spotify.create_playlist("My Vibes", tracks=["track_id1", "track_id2"])
    
    # Control playback
    await spotify.play(track_uri="spotify:track:xxx")
    await spotify.pause()
    await spotify.skip_next()
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

from ..accounts.manager import AccountManager
from ..accounts.sessions import OAuthToken, SessionManager
from ..core.logging_setup import get_logger

__all__ = ["SpotifyIntegration", "Track", "Playlist", "Artist", "Album"]

_log = get_logger(__name__)

SPOTIFY_API = "https://api.spotify.com/v1"


@dataclass
class Track:
    track_id: str
    name: str
    artists: list[str] = field(default_factory=list)
    album: str = ""
    duration_ms: int = 0
    uri: str = ""
    preview_url: str = ""
    popularity: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id, "name": self.name,
            "artists": self.artists, "album": self.album,
            "duration_ms": self.duration_ms, "uri": self.uri,
        }


@dataclass
class Playlist:
    playlist_id: str
    name: str
    description: str = ""
    track_count: int = 0
    owner: str = ""
    uri: str = ""
    is_public: bool = True
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "playlist_id": self.playlist_id, "name": self.name,
            "description": self.description, "track_count": self.track_count,
        }


@dataclass
class Artist:
    artist_id: str
    name: str
    genres: list[str] = field(default_factory=list)
    popularity: int = 0
    followers: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return {"artist_id": self.artist_id, "name": self.name, "genres": self.genres}


@dataclass
class Album:
    album_id: str
    name: str
    artists: list[str] = field(default_factory=list)
    release_date: str = ""
    track_count: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return {"album_id": self.album_id, "name": self.name, "artists": self.artists}


class SpotifyIntegration:
    """Spotify Web API integration."""
    
    def __init__(self, account_manager: AccountManager, session_manager: SessionManager) -> None:
        self.account_manager = account_manager
        self.session_manager = session_manager
        _log.info("Spotify integration initialized")
    
    async def _api_request(
        self, method: str, endpoint: str, account: str,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make authenticated Spotify API request."""
        cred = self.account_manager.get_credential("spotify_oauth", account)
        token_data = json.loads(cred.password)
        access_token = token_data["access_token"]
        
        url = f"{SPOTIFY_API}/{endpoint}"
        body = json.dumps(data).encode() if data else None
        
        req = urllib.request.Request(
            url, data=body, method=method,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
        )
        
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                if response.status == 204:
                    return {}
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else ""
            _log.error(f"Spotify API error: {e.code} - {error_body}")
            raise
    
    # ── Search ───────────────────────────────────────────────────────────────
    
    async def search(
        self, query: str, account: str, *,
        search_type: str = "track", limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search Spotify."""
        params = urllib.parse.urlencode({
            "q": query, "type": search_type, "limit": limit,
        })
        
        result = await self._api_request("GET", f"search?{params}", account)
        
        if search_type == "track":
            items = result.get("tracks", {}).get("items", [])
            return [self._parse_track(item) for item in items]
        elif search_type == "artist":
            items = result.get("artists", {}).get("items", [])
            return [self._parse_artist(item) for item in items]
        elif search_type == "album":
            items = result.get("albums", {}).get("items", [])
            return [self._parse_album(item) for item in items]
        elif search_type == "playlist":
            items = result.get("playlists", {}).get("items", [])
            return [self._parse_playlist(item) for item in items]
        
        return []
    
    # ── Playback ─────────────────────────────────────────────────────────────
    
    async def get_currently_playing(self, account: str) -> Optional[dict[str, Any]]:
        """Get currently playing track."""
        result = await self._api_request("GET", "me/player/currently-playing", account)
        
        if not result or not result.get("is_playing"):
            return None
        
        item = result.get("item", {})
        return {
            "track": self._parse_track(item),
            "progress_ms": result.get("progress_ms", 0),
            "is_playing": result.get("is_playing", False),
            "device": result.get("device", {}).get("name", ""),
        }
    
    async def play(self, account: str, *, track_uri: str = "", context_uri: str = "") -> bool:
        """Start playback."""
        data: dict[str, Any] = {}
        if track_uri:
            data["uris"] = [track_uri]
        elif context_uri:
            data["context_uri"] = context_uri
        
        await self._api_request("PUT", "me/player/play", account, data=data or None)
        return True
    
    async def pause(self, account: str) -> bool:
        """Pause playback."""
        await self._api_request("PUT", "me/player/pause", account)
        return True
    
    async def skip_next(self, account: str) -> bool:
        """Skip to next track."""
        await self._api_request("POST", "me/player/next", account)
        return True
    
    async def skip_previous(self, account: str) -> bool:
        """Skip to previous track."""
        await self._api_request("POST", "me/player/previous", account)
        return True
    
    async def set_volume(self, account: str, volume_percent: int) -> bool:
        """Set playback volume (0-100)."""
        params = urllib.parse.urlencode({"volume_percent": max(0, min(100, volume_percent))})
        await self._api_request("PUT", f"me/player/volume?{params}", account)
        return True
    
    async def seek(self, account: str, position_ms: int) -> bool:
        """Seek to position."""
        params = urllib.parse.urlencode({"position_ms": position_ms})
        await self._api_request("PUT", f"me/player/seek?{params}", account)
        return True
    
    # ── Playlists ────────────────────────────────────────────────────────────
    
    async def get_playlists(self, account: str, *, limit: int = 50) -> list[Playlist]:
        """Get user's playlists."""
        params = urllib.parse.urlencode({"limit": limit})
        result = await self._api_request("GET", f"me/playlists?{params}", account)
        
        return [self._parse_playlist(item) for item in result.get("items", [])]
    
    async def create_playlist(
        self, name: str, account: str, *,
        description: str = "", tracks: list[str] | None = None,
        public: bool = True,
    ) -> Playlist:
        """Create a new playlist."""
        # Get user ID
        user = await self._api_request("GET", "me", account)
        user_id = user["id"]
        
        # Create playlist
        data = {"name": name, "description": description, "public": public}
        result = await self._api_request("POST", f"users/{user_id}/playlists", account, data=data)
        
        playlist = self._parse_playlist(result)
        
        # Add tracks if provided
        if tracks:
            await self.add_tracks_to_playlist(playlist.playlist_id, tracks, account)
        
        return playlist
    
    async def add_tracks_to_playlist(
        self, playlist_id: str, track_uris: list[str], account: str,
    ) -> bool:
        """Add tracks to a playlist."""
        data = {"uris": track_uris}
        await self._api_request("POST", f"playlists/{playlist_id}/tracks", account, data=data)
        return True
    
    async def remove_tracks_from_playlist(
        self, playlist_id: str, track_uris: list[str], account: str,
    ) -> bool:
        """Remove tracks from a playlist."""
        data = {"tracks": [{"uri": uri} for uri in track_uris]}
        await self._api_request("DELETE", f"playlists/{playlist_id}/tracks", account, data=data)
        return True
    
    async def get_playlist_tracks(
        self, playlist_id: str, account: str, *, limit: int = 100,
    ) -> list[Track]:
        """Get tracks in a playlist."""
        params = urllib.parse.urlencode({"limit": limit})
        result = await self._api_request("GET", f"playlists/{playlist_id}/tracks?{params}", account)
        
        return [
            self._parse_track(item["track"])
            for item in result.get("items", [])
            if item.get("track")
        ]
    
    # ── Library ──────────────────────────────────────────────────────────────
    
    async def get_saved_tracks(self, account: str, *, limit: int = 50) -> list[Track]:
        """Get user's saved/liked tracks."""
        params = urllib.parse.urlencode({"limit": limit})
        result = await self._api_request("GET", f"me/tracks?{params}", account)
        
        return [
            self._parse_track(item["track"])
            for item in result.get("items", [])
        ]
    
    async def save_track(self, track_id: str, account: str) -> bool:
        """Save a track to library."""
        await self._api_request("PUT", f"me/tracks?ids={track_id}", account)
        return True
    
    async def get_top_tracks(self, account: str, *, time_range: str = "medium_term", limit: int = 20) -> list[Track]:
        """Get user's top tracks."""
        params = urllib.parse.urlencode({"time_range": time_range, "limit": limit})
        result = await self._api_request("GET", f"me/top/tracks?{params}", account)
        
        return [self._parse_track(item) for item in result.get("items", [])]
    
    # ── Podcasts ─────────────────────────────────────────────────────────────
    
    async def get_podcasts(self, account: str) -> list[dict[str, Any]]:
        """Get user's saved podcasts/shows."""
        result = await self._api_request("GET", "me/shows?limit=50", account)
        
        return [
            {
                "show_id": item["show"]["id"],
                "name": item["show"]["name"],
                "publisher": item["show"].get("publisher", ""),
                "description": item["show"].get("description", ""),
            }
            for item in result.get("items", [])
        ]
    
    # ── Parsers ──────────────────────────────────────────────────────────────
    
    def _parse_track(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "track_id": item.get("id", ""),
            "name": item.get("name", ""),
            "artists": [a["name"] for a in item.get("artists", [])],
            "album": item.get("album", {}).get("name", ""),
            "duration_ms": item.get("duration_ms", 0),
            "uri": item.get("uri", ""),
            "preview_url": item.get("preview_url", ""),
            "popularity": item.get("popularity", 0),
        }
    
    def _parse_artist(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "artist_id": item.get("id", ""),
            "name": item.get("name", ""),
            "genres": item.get("genres", []),
            "popularity": item.get("popularity", 0),
            "followers": item.get("followers", {}).get("total", 0),
        }
    
    def _parse_album(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "album_id": item.get("id", ""),
            "name": item.get("name", ""),
            "artists": [a["name"] for a in item.get("artists", [])],
            "release_date": item.get("release_date", ""),
            "track_count": item.get("total_tracks", 0),
        }
    
    def _parse_playlist(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "playlist_id": item.get("id", ""),
            "name": item.get("name", ""),
            "description": item.get("description", ""),
            "track_count": item.get("tracks", {}).get("total", 0),
            "owner": item.get("owner", {}).get("display_name", ""),
            "uri": item.get("uri", ""),
            "is_public": item.get("public", True),
        }
