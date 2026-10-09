"""YouTube cookie support for yt-dlp.

YouTube aggressively bot-checks datacenter IPs. The standard fix is
passing the user's own browser cookies to yt-dlp via ``--cookies`` /
``cookiefile``. This module discovers a user-provided Netscape-format
cookies file and exposes helpers to wire it into every yt-dlp
invocation — python API and CLI alike.

The cookies file is USER-PROVIDED and never logged in full. If no file
exists, everything behaves exactly as before.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

__all__ = [
    "find_cookies_file",
    "ytdlp_cookie_opts",
    "ytdlp_cookie_args",
    "is_bot_detection_error",
    "cookies_status",
    "BOT_COOKIE_HELP",
    "COOKIES_ENV",
]

#: Env var override — path to a Netscape-format cookies file.
COOKIES_ENV = "YTDLP_COOKIES"

#: 2-line user-facing help when YouTube bot-blocks a download.
BOT_COOKIE_HELP = (
    "YouTube is blocking automated downloads. Export your YouTube cookies: "
    "open youtube.com in your browser, use a cookie-exporter extension to save "
    "as Netscape format to ~/.nomorals/youtube-cookies.txt, then retry."
)

#: Markers (case-insensitive) that identify YouTube's bot check.
_BOT_MARKERS = (
    "not a bot",
    "sign in to confirm",
    "confirm you're not a bot",
    "confirm you are not a bot",
    "--cookies-from-browser",
    "--cookies for the authentication",
)


def _candidate_paths() -> list[Path]:
    """Cookie file locations, in discovery order."""
    home = Path.home()
    cands = [
        home / ".nomorals" / "youtube-cookies.txt",
    ]
    prefix = os.environ.get("PREFIX", "").strip()
    if prefix:
        cands.append(Path(prefix) / "etc" / "nomorals" / "youtube-cookies.txt")
    env_path = os.environ.get(COOKIES_ENV, "").strip()
    if env_path:
        cands.append(Path(env_path).expanduser())
    cands.append(Path.cwd() / "youtube-cookies.txt")
    return cands


def find_cookies_file() -> str | None:
    """Path to a usable YouTube cookies file, or None. Never raises."""
    try:
        for cand in _candidate_paths():
            try:
                if cand.is_file() and cand.stat().st_size > 0:
                    return str(cand)
            except OSError:
                continue
        return None
    except Exception:  # noqa: BLE001 - discovery must never break callers
        return None


def ytdlp_cookie_opts() -> dict[str, Any]:
    """Extra kwargs for ``yt_dlp.YoutubeDL({...})``. Empty when no file."""
    path = find_cookies_file()
    return {"cookiefile": path} if path else {}


def ytdlp_cookie_args() -> list[str]:
    """Extra argv for the ``yt-dlp`` CLI. Empty when no file."""
    path = find_cookies_file()
    return ["--cookies", path] if path else []


def is_bot_detection_error(message: Any) -> bool:
    """True when *message* looks like YouTube's bot check. Never raises."""
    try:
        text = str(message or "").lower()
    except Exception:  # noqa: BLE001
        return False
    if not text:
        return False
    return any(marker in text for marker in _BOT_MARKERS)


def cookies_status() -> dict[str, Any]:
    """Where Devon looks for cookies + whether one was found. Never raises."""
    try:
        found = find_cookies_file()
        return {
            "found": found,
            "checked": [str(p) for p in _candidate_paths()],
            "env_var": COOKIES_ENV,
        }
    except Exception:  # noqa: BLE001
        return {"found": None, "checked": [], "env_var": COOKIES_ENV}
