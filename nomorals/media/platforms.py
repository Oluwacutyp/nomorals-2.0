"""Dedicated platform downloaders — TikTok, Instagram, and the
URL→platform routing the universal ``/download`` / ``/aio`` surface
shares.

These are NOT a parallel download stack.  The heavy lifting stays in
:mod:`nomorals.media.downloader` (the yt-dlp → kit → proxy → browser
chain, with the source-registry descriptors speaking first).  This
module adds what the chain does not know:

* **which platform a URL belongs to** — one detection table
  (:func:`detect_platform`) covering each platform's shortlink hosts,
  so ``/aio``, ``/download`` and the named commands all route the
  same way;
* **TikTok's watermark truth** — yt-dlp's TikTok extractor marks the
  ``download_addr`` / ``download`` renditions ``format_note:
  'watermarked'`` (and deprioritizes them) while the ``play_addr`` /
  PlayAddr renditions carry no such mark (verified against yt-dlp's
  extractor source, 2026-11-05).  The TikTok route probes first,
  prefers a format spec that excludes the watermarked renditions,
  and labels the outcome honestly: a file is called watermark-free
  only when a clean rendition is what the probe said exists and the
  clean spec is what fetched it.  An independent second route (the
  public tikwm API, whose ``play`` / ``hdplay`` fields are the
  no-watermark renditions — its ``wmplay`` field is never requested)
  covers a yt-dlp failure;
* **Instagram's login wall** — public reels/posts flow through the
  same chain (the owner's exported cookies, when on file, ride the
  direct stage as they do for every download).  When every stage
  dies on a login/checkpoint wall the failure says so in plain
  words instead of a raw extractor dump.  Devon keeps no Instagram
  credentials for downloads and fabricates none: a wall is a wall,
  named.

Every fetch returns a :class:`PlatformResult` and never raises;
each route tried is recorded in ``attempts`` in the resolver's
tried-list style.  Downloads are verified on arrival — the file
exists, is non-trivial, and is where the route said it would be —
before anything is called a success.

Layering: media is L4.  No connector imports here.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .failures import record_failure
from .sources import BROWSER_UA

_log = get_logger(__name__)

__all__ = [
    "PlatformResult",
    "detect_platform",
    "is_media_file_url",
    "platform_label",
    "SUPPORTED_PLATFORMS",
    "fetch_tiktok",
    "fetch_instagram",
    "probe_info",
    "tiktok_format_facts",
    "verify_download",
    "TIKTOK_CLEAN_FORMAT_SPEC",
]


# ── platform detection ──────────────────────────────────────────────

#: Platform → hosts (bare + shortlink).  Subdomains of a listed host
#: count (``www.``, ``m.``, ``music.``…).  Shortlink hosts are listed
#: under the platform they MEAN: detection answers "whose content is
#: this", and the download chain expands the short link itself.
_PLATFORM_HOSTS: dict[str, tuple[str, ...]] = {
    "tiktok": ("tiktok.com", "vm.tiktok.com", "vt.tiktok.com"),
    "instagram": ("instagram.com", "instagr.am"),
    "spotify": ("open.spotify.com", "play.spotify.com", "spoti.fi"),
    "youtube": ("youtube.com", "youtu.be", "music.youtube.com"),
    "soundcloud": ("soundcloud.com", "on.soundcloud.com"),
    "twitter": ("twitter.com", "x.com", "t.co"),
    "facebook": ("facebook.com", "fb.watch", "fb.com"),
    "vimeo": ("vimeo.com",),
    "dailymotion": ("dailymotion.com", "dai.ly"),
    "twitch": ("twitch.tv",),
    "reddit": ("reddit.com", "redd.it"),
    "audiomack": ("audiomack.com",),
    "bandcamp": ("bandcamp.com",),
}

_PLATFORM_LABELS: dict[str, str] = {
    "tiktok": "TikTok",
    "instagram": "Instagram",
    "spotify": "Spotify",
    "youtube": "YouTube",
    "soundcloud": "SoundCloud",
    "twitter": "X/Twitter",
    "facebook": "Facebook",
    "vimeo": "Vimeo",
    "dailymotion": "Dailymotion",
    "twitch": "Twitch",
    "reddit": "Reddit",
    "audiomack": "Audiomack",
    "bandcamp": "Bandcamp",
}

#: Platforms with a route in this organ: the dedicated TikTok /
#: Instagram / Spotify shapes above, plus everything the universal
#: chain (yt-dlp) extracts natively.  ``/aio``'s honest refusal names
#: exactly this set.
SUPPORTED_PLATFORMS: tuple[str, ...] = tuple(_PLATFORM_HOSTS)

_MEDIA_FILE_EXTS = (
    ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v",
    ".mp3", ".m4a", ".ogg", ".opus", ".flac", ".wav", ".aac",
)


def _host_of(url: str) -> str:
    try:
        host = (urllib.parse.urlparse(str(url or "")).hostname or "")
        return host.lower()
    except Exception as exc:  # noqa: BLE001 — detection never raises
        record_failure("platforms._host_of", exc)
        return ""


def detect_platform(url: str) -> str:
    """The platform a URL belongs to, or ``""`` when unknown.

    Spotify URIs (``spotify:track:…``) count as Spotify.  A host
    matches when it equals a listed host or is a subdomain of one —
    so ``vm.tiktok.com`` and ``www.tiktok.com`` are both TikTok,
    while ``faketiktok.com`` is nothing.
    """
    text = str(url or "").strip()
    if not text:
        return ""
    if text.lower().startswith("spotify:"):
        return "spotify"
    host = _host_of(text)
    if not host:
        return ""
    for platform, hosts in _PLATFORM_HOSTS.items():
        for known in hosts:
            if host == known or host.endswith("." + known):
                return platform
    return ""


def platform_label(platform: str) -> str:
    """Human name for a detected platform key."""
    return _PLATFORM_LABELS.get(platform, platform or "that site")


def is_media_file_url(url: str) -> bool:
    """True when the URL path itself names a media file."""
    try:
        path = urllib.parse.urlparse(str(url or "")).path.lower()
    except Exception as exc:  # noqa: BLE001
        record_failure("platforms.is_media_file_url", exc)
        return False
    return path.endswith(_MEDIA_FILE_EXTS)


# ── result ──────────────────────────────────────────────────────────


@dataclass
class PlatformResult:
    """The outcome of one platform fetch — never raises."""

    ok: bool = False
    path: str = ""
    title: str = ""
    size_bytes: int = 0
    duration: float = 0.0
    #: Which route delivered: "yt-dlp" | "tikwm" | "browser" | …
    route: str = ""
    #: TikTok only: True = the fetched rendition is a watermarked
    #: one; False = a clean rendition was fetched; None = the route
    #: could not prove either way (never upgraded to a claim).
    watermarked: bool | None = None
    attempts: list[str] = field(default_factory=list)
    #: Plain-words failure summary when not ok.
    failure: str = ""
    #: "login" when a login wall is what stopped the fetch.
    wall: str = ""


def verify_download(path: str) -> tuple[bool, float]:
    """Arrival verification: the file is real and non-trivial.

    Returns ``(ok, duration_seconds)`` — duration is best-effort
    (0.0 when unprobeable; absence of a duration is never itself a
    failure, matching the preview law's verification rule).
    """
    try:
        p = Path(str(path or ""))
        if not p.is_file():
            return False, 0.0
        if p.stat().st_size < 1024:
            return False, 0.0
        duration = 0.0
        try:
            from .library import read_metadata
            duration = float(read_metadata(str(p)).get("duration") or 0)
        except Exception as exc:  # noqa: BLE001 — probe is best-effort
            record_failure("platforms.verify_download.probe", exc)
        return True, duration
    except Exception as exc:  # noqa: BLE001
        record_failure("platforms.verify_download", exc)
        return False, 0.0


# ── yt-dlp probe (raw info — the watermark facts live in formats) ───


def probe_info(url: str) -> dict | None:
    """Raw yt-dlp info dict for *url*, or None when unprobeable.

    Same module→CLI ladder and cookie discovery as
    :func:`nomorals.tools.media.probe`, but returns the RAW info:
    the summary shape drops ``format_note``, which is exactly where
    the TikTok watermark truth lives.  Never raises.
    """
    try:
        from .cookies import ytdlp_cookie_args_for, ytdlp_cookie_opts_for
    except Exception as exc:  # noqa: BLE001
        record_failure("platforms.probe_info.cookies", exc)
        ytdlp_cookie_args_for = None  # type: ignore[assignment]
        ytdlp_cookie_opts_for = None  # type: ignore[assignment]
    try:
        import yt_dlp  # type: ignore[import-not-found]

        opts: dict[str, Any] = {
            "quiet": True, "no_warnings": True, "skip_download": True}
        if ytdlp_cookie_opts_for is not None:
            try:
                opts.update(ytdlp_cookie_opts_for(url))
            except Exception as exc:  # noqa: BLE001
                record_failure("platforms.probe_info.cookie_opts", exc)
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
        return info if isinstance(info, dict) else None
    except Exception as exc:  # noqa: BLE001 — fall through to the CLI
        _log.info("yt-dlp module probe failed for %s: %s", url[:80], exc)
    exe = shutil.which("yt-dlp")
    if not exe:
        return None
    try:
        argv = [exe, "--dump-single-json", "--no-warnings"]
        if ytdlp_cookie_args_for is not None:
            try:
                argv += list(ytdlp_cookie_args_for(url))
            except Exception as exc:  # noqa: BLE001
                record_failure("platforms.probe_info.cookie_args", exc)
        argv.append(url)
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=90)
        if proc.returncode != 0:
            return None
        info = json.loads(proc.stdout)
        return info if isinstance(info, dict) else None
    except Exception as exc:  # noqa: BLE001
        record_failure("platforms.probe_info.cli", exc)
        return None


# ── TikTok ──────────────────────────────────────────────────────────

#: yt-dlp format ids / notes that mark a WATERMARKED TikTok
#: rendition.  Grounded in yt-dlp's extractor: the app path builds
#: ``format_id='download_addr'`` with ``format_note='Download video,
#: watermarked'`` when the aweme carries a watermark; the web path
#: builds ``format_id='download'`` with ``format_note='watermarked'``
#: unconditionally.  Play renditions (``play_addr``, ``play``, the
#: bit_rate gears) are never marked.
_WM_FORMAT_IDS = frozenset({"download", "download_addr"})

#: Format spec preferring every rendition EXCEPT the watermarked
#: download ones, best first.  When only watermarked renditions
#: exist this matches nothing and the chain falls through honestly
#: instead of quietly fetching the watermarked file under a
#: clean-sounding spec.
TIKTOK_CLEAN_FORMAT_SPEC = (
    "best[format_id!=download][format_id!=download_addr]"
    "/bestvideo[format_id!=download][format_id!=download_addr]+bestaudio"
    "/best[format_id!=download][format_id!=download_addr]"
)


def tiktok_format_facts(info: dict | None) -> tuple[bool, bool]:
    """``(clean_available, watermarked_available)`` from a probe.

    A format counts as watermarked when its id is a download
    rendition or its note says so; any other format with a URL is a
    clean rendition.  ``(False, False)`` when the probe carried no
    usable formats.
    """
    clean = False
    watermarked = False
    for fmt in (info or {}).get("formats") or []:
        if not isinstance(fmt, dict) or not fmt.get("url"):
            continue
        fid = str(fmt.get("format_id") or "")
        note = str(fmt.get("format_note") or "").lower()
        if fid in _WM_FORMAT_IDS or "watermark" in note:
            watermarked = True
        else:
            clean = True
    return clean, watermarked


def _winning_stage(report: Any) -> str:
    """The stage that landed the file, from a DownloadReport."""
    for stage in getattr(report, "stages", None) or []:
        if getattr(stage, "ok", False):
            return str(getattr(stage, "stage", "") or "")
    return ""


def _report_reason(report: Any) -> str:
    """One short line on why a DownloadReport failed."""
    cause = str(getattr(report, "likely_cause", "") or "").strip()
    if cause:
        return cause
    for stage in reversed(getattr(report, "stages", None) or []):
        err = str(getattr(stage, "error", "") or "").strip()
        if err:
            return err
    return "download failed"


_TIKWM_API = "https://www.tikwm.com/api/"


def _http_get_json(url: str, timeout: float = 25.0) -> dict | None:
    """GET *url* → parsed JSON dict, or None.  Never raises."""
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": BROWSER_UA,
                          "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        return data if isinstance(data, dict) else None
    except Exception as exc:  # noqa: BLE001
        _log.info("GET %s failed: %s", url[:80], exc)
        return None


def _http_fetch_bytes(url: str, dest: Path, timeout: float = 300.0) -> bool:
    """Stream *url* to *dest*.  True when bytes landed.  Never raises."""
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": BROWSER_UA,
                          "Referer": "https://www.tiktok.com/"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, \
                open(dest, "wb") as fh:
            shutil.copyfileobj(resp, fh, length=1024 * 256)
        return dest.is_file() and dest.stat().st_size > 0
    except Exception as exc:  # noqa: BLE001
        _log.info("fetch %s failed: %s", url[:80], exc)
        try:
            dest.unlink(missing_ok=True)
        except Exception as exc2:  # noqa: BLE001
            record_failure("platforms._http_fetch_bytes.unlink", exc2)
        return False



def fetch_tiktok(url: str, *, downloader: Any,
                 dest_dir: str | Path | None = None,
                 audio_only: bool = False,
                 probe_fn: Callable[[str], dict | None] | None = None,
                 api_get: Callable[[str], dict | None] | None = None,
                 fetch_bytes: Callable[[str, Path], bool] | None = None,
                 ) -> PlatformResult:
    """Fetch one TikTok video (or its audio) — route chain:

    1. **yt-dlp via the download chain** — probed first so the
       watermark label is evidence, not hope: when the probe shows a
       clean rendition, the chain runs under
       :data:`TIKTOK_CLEAN_FORMAT_SPEC`; when it shows ONLY
       watermarked renditions, the default spec runs and the result
       is labeled watermarked; when the probe itself fails, the
       chain runs and the watermark is reported unverified.
    2. **tikwm** — the independent API route; its ``hdplay``/``play``
       fields are the no-watermark renditions.

    ``audio_only`` skips route 2 (it serves video renditions) and
    asks the chain for audio only.  Never raises.
    """
    result = PlatformResult()
    probe = probe_fn if probe_fn is not None else probe_info
    get_json = api_get if api_get is not None else _http_get_json
    fetch = fetch_bytes if fetch_bytes is not None else _http_fetch_bytes

    # ── route 1: yt-dlp through the universal chain ─────────────
    clean: bool | None = None
    wm_only = False
    if audio_only:
        result.attempts.append("yt-dlp — audio requested")
    else:
        try:
            info = probe(url)
        except Exception as exc:  # noqa: BLE001 — a probe never kills
            record_failure("platforms.fetch_tiktok.probe", exc)
            info = None
        if info:
            has_clean, has_wm = tiktok_format_facts(info)
            clean = has_clean
            wm_only = has_wm and not has_clean
            result.title = str(info.get("title") or "")
        else:
            result.attempts.append("yt-dlp probe — no metadata back")
    try:
        if audio_only:
            report = downloader.download(url, audio_only=True,
                                         dest_dir=dest_dir)
        elif clean:
            report = downloader.download(
                url, dest_dir=dest_dir,
                format_spec=TIKTOK_CLEAN_FORMAT_SPEC)
        else:
            report = downloader.download(url, dest_dir=dest_dir)
    except Exception as exc:  # noqa: BLE001 — the chain never raises,
        # but an injected/foreign downloader might
        record_failure("platforms.fetch_tiktok.chain", exc)
        report = None
    if report is not None and getattr(report, "ok", False):
        ok_file, duration = verify_download(str(report.path))
        if ok_file:
            stage = _winning_stage(report)
            result.ok = True
            result.path = str(report.path)
            result.title = result.title or str(report.title or "")
            result.size_bytes = int(report.size_bytes or 0)
            result.duration = duration
            result.route = "yt-dlp" if stage in ("", "direct") \
                else f"yt-dlp chain ({stage})"
            if audio_only:
                result.watermarked = None
            elif stage in ("", "direct") and clean:
                result.watermarked = False
            elif stage in ("", "direct") and wm_only:
                result.watermarked = True
            else:
                result.watermarked = None
            result.attempts.append(f"{result.route} — ok")
            return result
        result.attempts.append(
            "yt-dlp — landed file failed verification (empty/not media)")
    else:
        reason = _report_reason(report) if report is not None \
            else "download chain error"
        result.attempts.append(f"yt-dlp — {reason}")

    if audio_only:
        result.failure = ("couldn't get that TikTok's audio — "
                          + "; ".join(result.attempts))
        return result

    # ── route 2: tikwm (independent API) ────────────────────────
    api_url = (_TIKWM_API + "?url="
               + urllib.parse.quote(url, safe="") + "&hd=1")
    payload = None
    try:
        payload = get_json(api_url)
    except Exception as exc:  # noqa: BLE001
        record_failure("platforms.fetch_tiktok.tikwm", exc)
    data = (payload or {}).get("data") if isinstance(payload, dict) \
        else None
    if not isinstance(data, dict) or not data:
        msg = ""
        if isinstance(payload, dict):
            msg = str(payload.get("msg") or "")
        result.attempts.append(
            f"tikwm — {msg or 'no answer from the API'}")
    else:
        # hdplay (HD clean) → play (clean).  wmplay is the
        # watermarked rendition and is never requested.
        media_url = str(data.get("hdplay") or data.get("play") or "")
        vid = re.sub(r"[^A-Za-z0-9_-]+", "",
                     str(data.get("id") or "video")) or "video"
        if not media_url:
            result.attempts.append("tikwm — no clean rendition offered")
        else:
            try:
                out_dir = Path(dest_dir) if dest_dir else Path(
                    downloader._dest_dir(None))
            except Exception as exc:  # noqa: BLE001
                record_failure("platforms.fetch_tiktok.destdir", exc)
                out_dir = Path(".")
            dest = out_dir / f"tiktok-{vid}.mp4"
            if fetch(media_url, dest):
                ok_file, duration = verify_download(str(dest))
                if ok_file:
                    result.ok = True
                    result.path = str(dest)
                    result.title = (result.title
                                    or str(data.get("title") or ""))
                    result.size_bytes = dest.stat().st_size
                    result.duration = duration or float(
                        data.get("duration") or 0)
                    result.route = "tikwm"
                    result.watermarked = False
                    result.attempts.append("tikwm — ok")
                    return result
            result.attempts.append(
                "tikwm — clean rendition didn't arrive intact")
    result.failure = ("couldn't fetch that TikTok — "
                      + "; ".join(result.attempts))
    return result



# ── Instagram ───────────────────────────────────────────────────────

#: Substrings (lowercase) that mark a login/checkpoint wall in
#: yt-dlp / chain failure text for Instagram.
_IG_LOGIN_MARKERS = (
    "login required",
    "log in",
    "sign in",
    "checkpoint",
    "cookies are required",
    "login_required",
    "please log in",
    "rate limit exceeded. please wait",
    "loginpage",
)


def _looks_like_login_wall(text: str) -> bool:
    low = (text or "").lower()
    return any(marker in low for marker in _IG_LOGIN_MARKERS)


def _report_full_text(report: Any) -> str:
    """All the failure text a DownloadReport carries, joined."""
    parts = [str(getattr(report, "likely_cause", "") or ""),
             str(getattr(report, "hint", "") or "")]
    for stage in getattr(report, "stages", None) or []:
        parts.append(str(getattr(stage, "error", "") or ""))
        parts.append(str(getattr(stage, "note", "") or ""))
    return " · ".join(p for p in parts if p)


def fetch_instagram(url: str, *, downloader: Any,
                    dest_dir: str | Path | None = None,
                    ) -> PlatformResult:
    """Fetch one public Instagram reel/post through the chain.

    Public content comes down via yt-dlp (the owner's exported
    cookies ride along when on file, as with every download).  A
    login/checkpoint wall is reported as a wall — plainly, with the
    routes that hit it — never as a raw extractor dump, and never
    "solved" with credentials Devon does not hold for this.
    """
    result = PlatformResult()
    try:
        report = downloader.download(url, dest_dir=dest_dir)
    except Exception as exc:  # noqa: BLE001
        record_failure("platforms.fetch_instagram.chain", exc)
        result.attempts.append(f"download chain — error: {exc}")
        result.failure = "couldn't fetch that Instagram post — chain error"
        return result
    if getattr(report, "ok", False):
        ok_file, duration = verify_download(str(report.path))
        if ok_file:
            stage = _winning_stage(report)
            result.ok = True
            result.path = str(report.path)
            result.title = str(report.title or "")
            result.size_bytes = int(report.size_bytes or 0)
            result.duration = duration
            result.route = "yt-dlp" if stage in ("", "direct") \
                else f"yt-dlp chain ({stage})"
            result.attempts.append(f"{result.route} — ok")
            return result
        result.attempts.append(
            "yt-dlp — landed file failed verification (empty/not media)")
        result.failure = ("Instagram sent something that wasn't a "
                          "usable media file")
        return result
    reason = _report_reason(report)
    tried = ", ".join(report.tried()) if hasattr(report, "tried") \
        else "the download chain"
    result.attempts.append(f"yt-dlp chain ({tried}) — {reason}")
    if _looks_like_login_wall(_report_full_text(report)):
        result.wall = "login"
        result.failure = (
            "Instagram wants a login for this one — it serves the "
            "post only to signed-in viewers, and I don't hold an "
            "Instagram login to download with. Public reels and "
            "posts usually come down fine; stories, private and "
            "age-gated posts are the ones that wall off.")
        return result
    result.failure = f"couldn't fetch that Instagram post — {reason}"
    return result


# ── file hygiene for callers ────────────────────────────────────────


def file_size(path: str) -> int:
    """Size of *path* in bytes (0 when unreadable).  Never raises."""
    try:
        return os.path.getsize(str(path or ""))
    except Exception as exc:  # noqa: BLE001
        record_failure("platforms.file_size", exc)
        return 0
