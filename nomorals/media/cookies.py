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
import time
from pathlib import Path
from typing import Any

__all__ = [
    "find_cookies_file",
    "browser_session_dirs",
    "export_browser_cookies",
    "cookies_file_for",
    "ytdlp_cookie_opts",
    "ytdlp_cookie_args",
    "ytdlp_cookie_opts_for",
    "ytdlp_cookie_args_for",
    "is_bot_detection_error",
    "cookies_status",
    "BOT_COOKIE_HELP",
    "BOT_COOKIE_SHORT_HELP",
    "YTDLP_INSTALL_HELP",
    "COOKIES_ENV",
    "BROWSER_COOKIES_NAME",
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

#: 1-line user-facing help when YouTube bot-blocks a download in chat.
#: Phone-actionable: points at /cookies, never a yt-dlp command dump.
#: The full FAQ-grounded help stays in BOT_COOKIE_HELP (used by
#: /cookies itself, where the user asked for setup detail).
BOT_COOKIE_SHORT_HELP = (
    "YouTube is bot-blocking downloads — run /cookies to check/add "
    "your browser cookies."
)
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


# ── browser-profile cookies ─────────────────────────────────────────
#
# The browser tool persists every session's cookie jar to disk
# (``data/browser/sessions/<name>.json`` under the data home) — that IS
# Devon's own browser profile: a login she earned there (TikTok,
# Instagram, …) should serve downloads too, instead of the downloader
# asking the owner to "sign in once in the browser" and then never
# using the result.  These helpers export the persisted jars to the
# Netscape format yt-dlp reads.

#: Export filename, written inside the sessions dir it came from.
BROWSER_COOKIES_NAME = "browser-cookies.txt"


def _data_home() -> Path:
    """Data home root: ``$NM_HOME`` or ``~/.nomorals`` (config default)."""
    return Path(os.environ.get("NM_HOME") or "~/.nomorals").expanduser()


def browser_session_dirs(session_dir: str = "") -> list[Path]:
    """Where persisted browser session cookie files live.

    An explicit *session_dir* (resolved by a context-aware caller from
    its settings, mirroring the browser tool) comes first; the default
    data-home location is always a candidate.
    """
    out: list[Path] = []
    if session_dir:
        out.append(Path(session_dir).expanduser())
    default = _data_home() / "data" / "browser" / "sessions"
    if default not in out:
        out.append(default)
    return out


def _session_cookie_entries(session_file: Path) -> list[dict[str, Any]]:
    """Cookie state dicts from one persisted browser session file."""
    import json
    try:
        payload = json.loads(session_file.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a corrupt file means no cookies
        return []
    if not isinstance(payload, list):
        return []
    return [e for e in payload
            if isinstance(e, dict) and e.get("name") and e.get("domain")]


def _netscape_line(entry: dict[str, Any]) -> str | None:
    domain = str(entry.get("domain") or "").strip()
    name = str(entry.get("name") or "").strip()
    if not domain or not name:
        return None
    tailmatch = "TRUE" if (domain.startswith(".")
                           or entry.get("domain_initial_dot")) else "FALSE"
    path = str(entry.get("path") or "/")
    secure = "TRUE" if entry.get("secure") else "FALSE"
    try:
        expires = int(entry.get("expires") or 0)
    except (TypeError, ValueError):
        expires = 0
    if expires < 0:
        expires = 0
    value = str(entry.get("value") or "")
    return "\t".join([domain, tailmatch, path, secure, str(expires),
                      name, value])


def export_browser_cookies(session_dir: str = "") -> str | None:
    """Merge all persisted browser-session cookies into one Netscape
    cookies file for yt-dlp.  Returns its path, or None when the
    browser profile holds no cookies.

    The export is cached: it is rewritten only when a session file is
    newer than the export.  Never raises.
    """
    try:
        for directory in browser_session_dirs(session_dir):
            try:
                files = sorted(directory.glob("*.json"))
            except OSError:
                continue
            if not files:
                continue
            export = directory / BROWSER_COOKIES_NAME
            try:
                newest = max(f.stat().st_mtime for f in files)
                if export.is_file() and \
                        export.stat().st_mtime >= newest:
                    return str(export)
            except OSError:
                pass
            rows: dict[tuple[str, str, str], str] = {}
            for f in files:
                for entry in _session_cookie_entries(f):
                    line = _netscape_line(entry)
                    if line is None:
                        continue
                    key = (str(entry.get("domain") or ""),
                           str(entry.get("path") or "/"),
                           str(entry.get("name") or ""))
                    rows[key] = line  # later sessions win a name clash
            if not rows:
                continue
            text = ("# Netscape HTTP Cookie File\n"
                    "# Exported from Devon's own browser profile "
                    "(persisted browser sessions) for yt-dlp.\n"
                    + "\n".join(rows.values()) + "\n")
            try:
                export.write_text(text, encoding="utf-8")
            except OSError:
                return None
            return str(export)
        return None
    except Exception:  # noqa: BLE001 - cookie export must never break callers
        return None


def _file_has_cookie_for(path: str, host: str) -> bool:
    """True when a Netscape cookies file holds a cookie for *host*
    (exact domain or a parent domain of it)."""
    if not host:
        return False
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 7:
                    continue
                domain = parts[0].strip().lower().lstrip(".")
                if domain and (host == domain
                               or host.endswith("." + domain)):
                    return True
    except Exception:  # noqa: BLE001
        return False
    return False


def cookies_file_for(url: str, session_dir: str = "") -> str | None:
    """The best cookies file for downloading *url*.

    A file that actually holds a cookie for the URL's host wins: the
    browser-profile export when it covers the host, else the
    user-provided file, else the export (a jar with other domains'
    cookies is ignored by yt-dlp for this host, and may still help an
    extractor's API path).  None when no cookies exist anywhere.
    Never raises.
    """
    try:
        from urllib.parse import urlparse
        host = (urlparse(str(url or "")).hostname or "").lower()
        user_file = find_cookies_file()
        export = export_browser_cookies(session_dir)
        if export and host and _file_has_cookie_for(export, host):
            return export
        if user_file:
            return user_file
        return export
    except Exception:  # noqa: BLE001
        return None


def ytdlp_cookie_opts() -> dict[str, Any]:
    """Extra kwargs for ``yt_dlp.YoutubeDL({...})``. Empty when no file."""
    path = find_cookies_file()
    return {"cookiefile": path} if path else {}


def ytdlp_cookie_args() -> list[str]:
    """Extra argv for the ``yt-dlp`` CLI. Empty when no file."""
    path = find_cookies_file()
    return ["--cookies", path] if path else []


def ytdlp_cookie_opts_for(url: str, session_dir: str = "") -> dict[str, Any]:
    """Per-URL kwargs: the best cookies file for *url*'s host (the
    browser-profile export included).  Empty when no cookies exist."""
    path = cookies_file_for(url, session_dir)
    return {"cookiefile": path} if path else {}


def ytdlp_cookie_args_for(url: str, session_dir: str = "") -> list[str]:
    """Per-URL argv: companion to :func:`ytdlp_cookie_opts_for`."""
    path = cookies_file_for(url, session_dir)
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


def _count_valid_youtube_cookies(path: str) -> int:
    """Non-expired youtube.com entries in a Netscape cookies file.

    A file that EXISTS is not a file that WORKS: YouTube cookies
    expire, and an expired file fails the bot check exactly like no
    file at all — while /cookies would still report "found".  Session
    cookies (expiry 0) count as valid.  Never raises.
    """
    try:
        now = time.time()
        count = 0
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 7:
                    continue
                domain = parts[0].strip().lower().lstrip(".")
                if domain != "youtube.com" \
                        and not domain.endswith(".youtube.com"):
                    continue
                try:
                    expiry = int(parts[4])
                except (TypeError, ValueError):
                    continue
                if expiry == 0 or expiry > now:
                    count += 1
        return count
    except Exception:  # noqa: BLE001 - counting must never break callers
        return 0


def cookies_status() -> dict[str, Any]:
    """Where Devon looks for cookies + whether one was found. Never raises."""
    try:
        found = find_cookies_file()
        return {
            "found": found,
            "valid_youtube": (_count_valid_youtube_cookies(found)
                              if found else None),
            "checked": [str(p) for p in _candidate_paths()],
            "env_var": COOKIES_ENV,
        }
    except Exception:  # noqa: BLE001
        return {"found": None, "valid_youtube": None, "checked": [],
                "env_var": COOKIES_ENV}
