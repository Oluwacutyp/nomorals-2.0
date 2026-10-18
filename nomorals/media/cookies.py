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
    "YTDLP_INSTALL_HELP",
    "COOKIES_ENV",
]

#: Env var override — path to a Netscape-format cookies file.
COOKIES_ENV = "YTDLP_COOKIES"

#: Install help for yt-dlp, grounded in the official yt-dlp docs
#: (https://github.com/yt-dlp/yt-dlp/wiki/Installation) — never
#: invented. YouTube also needs ffmpeg to merge its separate
#: audio+video formats (official FAQ: "Do I need any other programs?").
YTDLP_INSTALL_HELP = (
    "yt-dlp is not installed — install it (official docs: "
    "https://github.com/yt-dlp/yt-dlp/wiki/Installation): "
    "`python3 -m pip install -U \"yt-dlp[default]\"` "
    "(Windows: `py -m pip install -U \"yt-dlp[default]\"` or "
    "`winget install --source winget --id yt-dlp.yt-dlp`; "
    "Termux: `pkg install python python-pip`, then "
    "`pip install -U \"yt-dlp[default]\"`). "
    "YouTube also needs ffmpeg to merge audio+video formats."
)

#: 3-line user-facing help when YouTube bot-blocks a download.
#: Grounded in the official yt-dlp FAQ ("How do I pass cookies to
#: yt-dlp?", https://github.com/yt-dlp/yt-dlp/wiki/FAQ) — never
#: invented. The cookies file must be Mozilla/Netscape format (first
#: line `# Netscape HTTP Cookie File`), CRLF newlines on Windows, LF
#: on Unix; HTTP 400 with --cookies means bad newline format.
#: WARNING from the same FAQ: the old "Get cookies.txt" Chrome
#: extension (without "LOCALLY") was reported as malware and removed
#: from the Chrome Web Store — uninstall it if present.
BOT_COOKIE_HELP = (
    "YouTube is blocking automated requests (bot check). Pass your own "
    "browser cookies to yt-dlp: easiest is "
    "`yt-dlp --cookies-from-browser chrome`, or export once with "
    "`yt-dlp --cookies-from-browser chrome --cookies cookies.txt` and "
    "then use `--cookies cookies.txt`. The cookies file must be "
    "Mozilla/Netscape format (first line `# Netscape HTTP Cookie File`), "
    "CRLF newlines on Windows, LF on Unix. Or use the 'Get cookies.txt "
    "LOCALLY' (Chrome) / 'cookies.txt' (Firefox) extension — uninstall "
    "the old 'Get cookies.txt' extension, it was reported as malware. "
    "Devon looks for the file at ~/.nomorals/youtube-cookies.txt or "
    "$YTDLP_COOKIES. Details: "
    "https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp"
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
