"""AudD connector — music recognition (Shazam-style audio fingerprinting).

Drives the AudD Music Recognition API (https://docs.audd.io):

* ``POST https://api.audd.io/`` (multipart: ``api_token``, ``file``,
  ``return=apple_music,spotify``) → ``{"status": "success",
  "result": {"artist", "title", "album", "release_date", "label",
  "timecode", "song_link", "apple_music": {...}, "spotify": {...}}}``
* ``result`` is ``null`` when nothing matches — not an error.

Auth: API token (``AUDD_API_TOKEN``) from https://dashboard.audd.io.
Free tier: 300 requests on signup, no card. Then $5 / 1,000 requests.

Why AudD over ACRCloud: simple token auth (ACRCloud needs HMAC
request signing), published pricing, free tier with no card, and rich
metadata (Spotify/Apple Music links) in the recognition response.
"""

from __future__ import annotations

import os
from pathlib import Path
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

__all__ = ["AudDConnector", "AudDError"]

_log = get_logger(__name__)

API_URL = "https://api.audd.io/"
KEY_ENV = "AUDD_API_TOKEN"
DASHBOARD_URL = "https://dashboard.audd.io"

#: AudD standard endpoint caps files at 10 MB.
MAX_FILE_BYTES = 10 * 1024 * 1024

#: Minimum audio worth sending — shorter clips rarely fingerprint.
MIN_FILE_BYTES = 4 * 1024


class AudDError(ConnectorError):
    """An AudD API call failed."""

    def __init__(
        self,
        message: str,
        *,
        error_code: int = 0,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code


@register_connector
class AudDConnector(Connector):
    """Devon's AudD music-recognition adapter."""

    id = "audd"
    name = "AudD"
    description = (
        "AudD music recognition: identify songs from audio files or "
        "voice notes. 300 free requests on signup (no card), then "
        "$5 / 1,000. Powers /sham."
    )
    auth_methods = (AuthMethod.API_KEY,)

    # ── lifecycle ────────────────────────────────────────────────

    def connect(
        self,
        *,
        api_key: str | None = None,
        context: Any = None,
    ) -> ConnectResult:
        """Validate an AudD API token and vault it."""
        existing = self._load_credential()
        if existing is not None:
            raise ConnectorError(
                "audd is already connected — one account per service. "
                "Disconnect first to switch keys."
            )
        key = (api_key or os.environ.get(KEY_ENV) or "").strip()
        if not key:
            raise ConnectorError(
                "no AudD API token. Get a free one (300 requests, no "
                f"card) at {DASHBOARD_URL}, then reconnect with the "
                f"token or set {KEY_ENV}."
            )
        # Validate with a cheap call: an empty recognize returns a
        # fingerprinting error (#300/#700), NOT an auth error (#900).
        # Auth errors raise; anything else means the token is fine.
        try:
            self._api_post(key, fields={"return": "apple_music,spotify"})
        except AudDError as exc:
            if exc.error_code in (900, 901):
                raise ConnectorError(
                    f"AudD rejected the token (error {exc.error_code}). "
                    "Check it at " + DASHBOARD_URL
                ) from exc
            # Any other error = token accepted, request malformed. Good.
        self._store_credential(
            "audd-user",
            key,
            credential_type="api_key",
            scopes=["music.recognize"],
            metadata={"free_tier": "300 requests"},
        )
        _log.info("audd connected")
        return ConnectResult(
            ok=True,
            account="audd-user",
            scopes=["music.recognize"],
            message=(
                "connected to AudD. 300 free recognitions, no card. "
                "The token is in the encrypted vault."
            ),
        )

    def disconnect(self) -> None:
        self._clear_credential()

    def status(self) -> ConnectorStatus:
        cred = self._load_credential()
        if cred is None:
            return ConnectorStatus(
                connected=False,
                detail=(
                    "audd is not connected — get a free token "
                    f"(300 requests, no card) at {DASHBOARD_URL}"
                ),
            )
        return ConnectorStatus(
            connected=True,
            account="audd-user",
            detail="audd connected",
        )

    def test_connection(self) -> bool:
        cred = self._load_credential()
        if cred is None:
            return False
        try:
            self._api_post(
                cred.password,
                fields={"return": "apple_music,spotify"},
            )
        except AudDError as exc:
            return exc.error_code not in (900, 901)
        except Exception:  # noqa: BLE001 - any failure is "not working"
            return False
        return True

    # ── recognition ──────────────────────────────────────────────

    def recognize(self, audio_path: str | Path) -> dict[str, Any]:
        """Identify the song in an audio file.

        Returns ``{"ok": True, "artist", "title", "album",
        "release_date", "song_link", "spotify": {...} | None,
        "apple_music": {...} | None}`` on a match, or
        ``{"ok": False, "reason": ...}`` on no-match / problems.
        Never raises.
        """
        try:
            return self._recognize(audio_path)
        except Exception as exc:  # noqa: BLE001 - never raises
            _log.debug("audd recognize failed", exc_info=True)
            return {"ok": False, "reason": f"recognition failed: {exc}"}

    def _recognize(self, audio_path: str | Path) -> dict[str, Any]:
        cred = self._load_credential()
        if cred is None:
            return {
                "ok": False,
                "reason": (
                    "music recognition isn't set up — it's one free API "
                    f"key away: grab a token at {DASHBOARD_URL} "
                    "(300 free requests, no card) and I'll wire it up."
                ),
                "needs_key": True,
            }
        path = self._prepare_audio(audio_path)
        if path is None:
            return {"ok": False, "reason": "could not read the audio file"}
        # Clean up the trimmed temp file afterwards (not the user's file).
        _tmp = path if str(path) != str(audio_path) else None
        try:
            size = path.stat().st_size
            if size > MAX_FILE_BYTES:
                return {
                    "ok": False,
                    "reason": (
                        f"audio is {size // (1024 * 1024)} MB — AudD's "
                        "standard endpoint caps at 10 MB. Send a shorter clip."
                    ),
                }
            if size < MIN_FILE_BYTES:
                return {
                    "ok": False,
                    "reason": "audio is too short to fingerprint — send a longer clip",
                }
            data = self._api_post(
                cred.password,
                fields={"return": "apple_music,spotify"},
                files=[("file", path, "")],
            )
        finally:
            if _tmp is not None:
                _tmp.unlink(missing_ok=True)
        result = data.get("result")
        if not result:
            return {
                "ok": False,
                "reason": (
                    "couldn't identify that audio — no match in AudD's "
                    "catalog. Try a clearer/longer clip."
                ),
            }
        return {
            "ok": True,
            "artist": str(result.get("artist") or "Unknown artist"),
            "title": str(result.get("title") or "Unknown title"),
            "album": str(result.get("album") or ""),
            "release_date": str(result.get("release_date") or ""),
            "song_link": str(result.get("song_link") or ""),
            "spotify": result.get("spotify") or None,
            "apple_music": result.get("apple_music") or None,
        }

    # ── audio prep ─────────────────────────────────────────────────

    def _prepare_audio(self, audio_path: str | Path) -> Path | None:
        """Return a fingerprint-ready audio file, or None.

        Fingerprinting only needs ~30 seconds. Long files are trimmed
        to a 30s window (starting at 10s to skip intros) via ffmpeg when
        available — faster upload, less data. Short files pass through
        untouched. Never raises.
        """
        try:
            path = Path(audio_path)
            if not path.is_file():
                return None
            import shutil
            import subprocess
            import tempfile

            if shutil.which("ffmpeg") is None:
                return path
            # Only trim when the file is big enough to be worth it.
            if path.stat().st_size < 1024 * 1024:
                return path
            tmp = Path(tempfile.mkstemp(suffix=".mp3")[1])
            proc = subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-ss", "10", "-t", "30",
                 "-i", str(path), "-ac", "1", "-ar", "44100",
                 "-b:a", "128k", str(tmp)],
                capture_output=True, timeout=60,
            )
            if proc.returncode == 0 and tmp.stat().st_size > 0:
                return tmp
            tmp.unlink(missing_ok=True)
            return path
        except Exception:  # noqa: BLE001 - best-effort
            _log.debug("audd audio prep failed", exc_info=True)
            try:
                return Path(audio_path) if Path(audio_path).is_file() else None
            except Exception:
                return None

    # ── HTTP ─────────────────────────────────────────────────────

    def _api_post(
        self,
        key: str,
        *,
        fields: dict[str, str] | None = None,
        files: list[tuple[str, Path, str]] | None = None,
    ) -> dict[str, Any]:
        """POST to AudD. Raises AudDError on API errors."""
        body_fields = {"api_token": key, **(fields or {})}
        try:
            resp = self.http.post_multipart(
                API_URL, fields=body_fields, files=files, timeout=30
            )
        except Exception as exc:
            raise AudDError(f"AudD request failed: {exc}") from exc
        try:
            data = resp.json()
        except Exception as exc:
            raise AudDError(
                f"AudD returned non-JSON (HTTP {resp.status_code})"
            ) from exc
        if not isinstance(data, dict):
            raise AudDError("AudD returned an unexpected response shape")
        if data.get("status") == "error":
            err = data.get("error") or {}
            code = int(err.get("error_code") or 0)
            msg = str(err.get("error_message") or "unknown AudD error")
            raise AudDError(f"AudD error {code}: {msg}", error_code=code)
        return data
