"""Media downloading.

Primary path is ``yt-dlp`` — the only realistic way to cover ~1800 sites, whose
extractors change weekly and are not worth reimplementing. Fallback path is a
direct HTTP download with resume, which covers direct file links and any site that
serves media without extraction.

If neither is available the tool says so clearly rather than failing obscurely,
because "yt-dlp is not installed" is a one-line fix and an agent should report it
as such.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from ..compat import available, load_optional, which
from ..core.errors import MediaError, ToolError
from ..core.logging_setup import get_logger
from ..core.policy import Capability

__all__ = ["download", "probe", "register"]

_log = get_logger(__name__)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]+")


def _clean(name: str, limit: int = 120) -> str:
    cleaned = _SAFE_NAME.sub("_", name).strip(" ._")
    return cleaned[:limit] or "media"


#: Error text shared by every backend when a job cancellation lands.
CANCELLED_MESSAGE = "download cancelled by request"


def _cancel_requested() -> bool:
    """True when the background job running this download was cancelled.

    Cooperative cancellation: the job registry cannot kill threads, so
    download backends poll this at every progress tick and stop
    themselves.  Outside a job context this is always False.
    """
    try:
        from ..core.jobs import current_job

        job = current_job()
        return bool(job is not None
                    and getattr(job, "cancel_requested", False))
    except Exception:  # noqa: BLE001 - cancellation is best-effort
        return False


def _snapshot_dir(target_dir: Path) -> dict[str, tuple[int, int]]:
    """name -> (size, mtime_ns) for every file in *target_dir*.

    Taken before a download starts so the result can be attributed to
    THIS download: the downloads directory is shared and accumulates
    every previous download, so "the largest file in the directory"
    is not evidence about what just landed.
    """
    snap: dict[str, tuple[int, int]] = {}
    try:
        entries = list(target_dir.iterdir())
    except OSError:
        return snap
    for entry in entries:
        try:
            if entry.is_file():
                st = entry.stat()
                snap[entry.name] = (st.st_size, st.st_mtime_ns)
        except OSError:
            continue
    return snap


def _fresh_files(target_dir: Path,
                 before: dict[str, tuple[int, int]]) -> list[Path]:
    """Non-empty files created or changed since the *before* snapshot."""
    out: list[Path] = []
    try:
        entries = list(target_dir.iterdir())
    except OSError:
        return out
    for entry in entries:
        try:
            if not entry.is_file():
                continue
            st = entry.stat()
        except OSError:
            continue
        if st.st_size <= 0:
            continue
        if before.get(entry.name) != (st.st_size, st.st_mtime_ns):
            out.append(entry)
    return out


def _looks_like_html(head: bytes) -> bool:
    """True when the first bytes are an HTML page, not media.

    The landing check for downloads: a soft-404 / interstitial page
    served with HTTP 200 must never be saved as ``.mp3``/``.mp4``.
    Real media containers (ID3/MP3 frames, ftyp, OggS, RIFF, fLaC)
    never begin with markup, so this cannot false-positive on audio
    or video bytes.
    """
    text = head[:1024].decode("utf-8", errors="replace")
    text = text.lstrip("\ufeff \t\r\n").lower()
    if text.startswith(("<!doctype html", "<html", "<head", "<title",
                        "<body", "<!--")):
        return True
    return "<html" in text[:512] and "<head" in text[:512]


def _looks_like_media(head: bytes) -> bool:
    """True when the first bytes carry a known media-container magic.

    Used to let a real media body win over a wrong ``text/html``
    Content-Type header (some CDNs mislabel everything).
    """
    if head.startswith((b"ID3", b"OggS", b"RIFF", b"fLaC", b"FLV",
                        b"\x1a\x45\xdf\xa3", b"\x30\x26\xb2\x75")):
        return True
    if len(head) >= 8 and head[4:8] == b"ftyp":  # mp4/m4a family
        return True
    # MP3 frame sync: 11 bits set at the start of a frame header.
    return len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0


def probe(url: str, *, timeout: float = 60.0) -> dict[str, Any]:
    """Fetch metadata without downloading."""
    from ..media.cookies import ytdlp_cookie_args, ytdlp_cookie_opts
    module = load_optional("yt_dlp")
    if module is not None:
        try:
            opts = {"quiet": True, "no_warnings": True, "skip_download": True}
            opts.update(ytdlp_cookie_opts())
            with module.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
            return _summarize(info)
        except Exception as exc:  # noqa: BLE001 - fall through to the CLI
            _log.debug("yt_dlp python API probe failed: %s", exc)
    if which("yt-dlp"):
        try:
            argv = (["yt-dlp", "--dump-single-json", "--no-download"]
                    + ytdlp_cookie_args() + [url])
            completed = subprocess.run(
                argv,
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, check=False,
            )
            if completed.returncode == 0:
                return _summarize(json.loads(completed.stdout))
        except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as exc:
            _log.debug("yt-dlp CLI probe failed: %s", exc)
    raise MediaError(f"could not probe {url}: install yt-dlp for site extraction")


def _summarize(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": info.get("id", ""),
        "title": info.get("title", ""),
        "uploader": info.get("uploader") or info.get("channel") or "",
        "duration": float(info.get("duration") or 0),
        "extractor": info.get("extractor_key") or info.get("extractor") or "",
        "webpage_url": info.get("webpage_url", ""),
        "formats": [
            {"format_id": f.get("format_id"), "ext": f.get("ext"),
             "resolution": f.get("resolution"), "filesize": f.get("filesize")}
            for f in (info.get("formats") or [])[-6:]
        ],
    }


def download(
    url: str,
    destination: str | Path,
    *,
    format_spec: str = "bestvideo*+bestaudio/best",
    audio_only: bool = False,
    progress: Callable[[str], None] | None = None,
    timeout: float = 1800.0,
) -> dict[str, Any]:
    """Download media, preferring yt-dlp and falling back to direct HTTP."""
    target_dir = Path(destination).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    if audio_only:
        format_spec = "bestaudio/best"

    module = load_optional("yt_dlp")
    if module is not None:
        return _download_python_api(module, url, target_dir, format_spec, audio_only, progress, started)
    if which("yt-dlp"):
        return _download_cli(url, target_dir, format_spec, audio_only, timeout, started)
    return _download_direct(url, target_dir, progress, started)


def _download_python_api(
    module: Any,
    url: str,
    target_dir: Path,
    format_spec: str,
    audio_only: bool,
    progress: Callable[[str], None] | None,
    started: float,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "outtmpl": str(target_dir / "%(title).120B [%(id)s].%(ext)s"),
        "format": format_spec,
        "quiet": True,
        "no_warnings": True,
        "noprogress": progress is None,
        "retries": 3,
        "concurrent_fragment_downloads": 4,
        "continuedl": True,
    }
    from ..media.cookies import ytdlp_cookie_opts
    _cookie_opts = ytdlp_cookie_opts()
    if _cookie_opts:
        options.update(_cookie_opts)
        _log.debug("yt-dlp: using cookies from %s",
                   _cookie_opts.get("cookiefile", "?"))
    if audio_only:
        options["postprocessors"] = [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}
        ]
    def hook(state: dict[str, Any]) -> None:
        if _cancel_requested():
            raise MediaError(CANCELLED_MESSAGE)
        if progress is not None and state.get("status") == "downloading":
            progress(f"{state.get('_percent_str', '?').strip()} of {state.get('_total_bytes_str', '?')}")

    # Always installed (not only when a progress callback was given):
    # the hook is also the cancellation checkpoint for this backend.
    options["progress_hooks"] = [hook]
    if _cancel_requested():
        raise MediaError(CANCELLED_MESSAGE)
    before = _snapshot_dir(target_dir)
    try:
        with module.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info) if info else ""
    except MediaError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise MediaError(f"download failed: {exc}") from exc

    # Attribute the result to THIS download.  Preference order: the
    # filepath yt-dlp reports for the finished item, the prepared
    # filename when it exists, then files that are new/changed since
    # the snapshot (preferring the one carrying this item's id).
    # NEVER a pre-existing untouched file: that is some earlier
    # download, and returning it reports the wrong content as success.
    path: Path | None = None
    if info:
        for key in ("filepath", "_filename"):
            reported = info.get(key)
            if reported and Path(str(reported)).is_file():
                path = Path(str(reported))
                break
    if path is None and filename and Path(filename).is_file():
        path = Path(filename)
    if path is None:
        fresh = _fresh_files(target_dir, before)
        video_id = str((info or {}).get("id") or "")
        if video_id:
            named = [p for p in fresh if f"[{video_id}]" in p.name]
            if named:
                fresh = named
        if fresh:
            path = max(fresh, key=lambda p: p.stat().st_size)
    if path is None:
        raise MediaError("download reported success but produced no file")
    return {
        "url": url, "path": str(path), "bytes": path.stat().st_size,
        "title": (info or {}).get("title", ""), "extractor": "yt_dlp",
        "seconds": round(time.perf_counter() - started, 2),
    }


def _download_cli(
    url: str,
    target_dir: Path,
    format_spec: str,
    audio_only: bool,
    timeout: float,
    started: float,
) -> dict[str, Any]:
    from ..media.cookies import ytdlp_cookie_args
    _cookie_args = ytdlp_cookie_args()
    if _cookie_args:
        _log.debug("yt-dlp: using cookies from %s", _cookie_args[-1])
    argv = [
        "yt-dlp", "-f", format_spec, "-o", str(target_dir / "%(title).120B [%(id)s].%(ext)s"),
        "--no-playlist", "--retries", "3", "--newline",
    ] + _cookie_args + [url]
    if audio_only:
        argv += ["-x", "--audio-format", "mp3"]
    if which("ffmpeg"):
        argv += ["--ffmpeg-location", which("ffmpeg") or "ffmpeg"]
    if _cancel_requested():
        raise MediaError(CANCELLED_MESSAGE)
    before = _snapshot_dir(target_dir)
    # Popen + communicate slices (not subprocess.run): the loop is the
    # cancellation checkpoint for this backend — a cancelled job
    # terminates the subprocess instead of letting it run to timeout.
    try:
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace")
    except OSError as exc:
        raise MediaError(f"yt-dlp failed to start: {exc}") from exc
    deadline = time.perf_counter() + timeout
    stdout_text, stderr_text = "", ""
    while True:
        try:
            stdout_text, stderr_text = proc.communicate(timeout=0.25)
            break
        except subprocess.TimeoutExpired:
            if _cancel_requested():
                proc.terminate()
                try:
                    proc.communicate(timeout=5)
                except Exception:  # noqa: BLE001 - already terminating
                    proc.kill()
                raise MediaError(CANCELLED_MESSAGE) from None
            if time.perf_counter() > deadline:
                proc.kill()
                try:
                    proc.communicate(timeout=5)
                except Exception:  # noqa: BLE001 - already killing
                    pass
                raise MediaError(
                    f"download timed out after {timeout}s") from None
    if proc.returncode != 0:
        raise MediaError(
            f"yt-dlp failed: {(stderr_text or '').strip()[-400:]}")
    # Attribute the result to THIS run: files yt-dlp named on stdout
    # (Destination / already-downloaded / merge target) inside the
    # target dir win; otherwise the largest file that is new/changed
    # since the snapshot.  A pre-existing untouched file is never the
    # result — it is an earlier download wearing this one's name.
    fresh = _fresh_files(target_dir, before)
    named: list[tuple[Path, bool]] = []  # (path, already_downloaded)
    for line in (stdout_text or "").splitlines():
        candidate = ""
        already = False
        if "Destination: " in line:
            candidate = line.split("Destination: ", 1)[1].strip()
        elif " has already been downloaded" in line:
            candidate = line.split(" has already been downloaded",
                                   1)[0].strip()
            candidate = candidate.split("] ", 1)[-1].strip()
            already = True
        elif "Merging formats into " in line:
            candidate = line.split("Merging formats into ", 1)[1] \
                .strip().strip('"')
        if candidate:
            p = Path(candidate)
            try:
                if p.is_file() and p.parent == target_dir:
                    named.append((p, already))
            except OSError:
                continue
    path: Path | None = None
    for p, already in reversed(named):
        # A path yt-dlp names is this run's file when it is fresh —
        # or when yt-dlp says it was already downloaded (this run's
        # item, fetched by an earlier run of the same request).
        if p in fresh or already:
            path = p
            break
    if path is None and fresh:
        path = max(fresh, key=lambda p: p.stat().st_size)
    if path is None:
        raise MediaError("yt-dlp produced no file")
    return {
        "url": url, "path": str(path), "bytes": path.stat().st_size,
        "extractor": "yt-dlp-cli", "seconds": round(time.perf_counter() - started, 2),
    }


def _download_direct(
    url: str,
    target_dir: Path,
    progress: Callable[[str], None] | None,
    started: float,
) -> dict[str, Any]:
    """Plain HTTP download with resume — no site extraction."""
    from ..core.http import HttpClient, url_filename

    name = _clean(url_filename(url))
    target = target_dir / name

    def hook(done: int, total: int) -> None:
        if _cancel_requested():
            raise MediaError(CANCELLED_MESSAGE)
        if progress is not None and total:
            progress(f"{done * 100 // total}% ({done}/{total} bytes)")

    if _cancel_requested():
        raise MediaError(CANCELLED_MESSAGE)
    response = HttpClient(timeout=600.0).request(
        "GET", url, stream_to=target, resume=True, progress=hook)
    if not target.exists() or target.stat().st_size == 0:
        raise MediaError("direct download produced an empty file")
    # Landing truth: an HTTP 200 HTML page (soft-404, interstitial,
    # parked domain) is not the media the URL promised.  HTTP error
    # statuses already raised inside the client; what remains is the
    # 200-with-an-error-page case.  The bytes decide — a real media
    # body served with a wrong Content-Type still passes.
    try:
        with target.open("rb") as fh:
            head = fh.read(1024)
    except OSError:
        head = b""
    ctype = str((response.headers or {}).get("content-type", "")).lower()
    if _looks_like_html(head) or (
            ctype.startswith(("text/html", "application/xhtml+xml"))
            and not _looks_like_media(head)):
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass
        raise MediaError(
            f"direct download from {url} returned an HTML page, "
            "not media (soft-404 or interstitial)")
    return {
        "url": url, "path": str(target), "bytes": target.stat().st_size,
        "extractor": "direct", "seconds": round(time.perf_counter() - started, 2),
    }


def register(registry: Any) -> None:
    """Attach the media tools to a registry."""
    context = registry.context

    @registry.register(
        "media_download",
        description="Download video or audio from a URL (yt-dlp when available, direct HTTP otherwise).",
        capability=Capability.NET_DOWNLOAD,
    )
    def media_download(
        url: str,
        *,
        audio_only: bool = False,
        format_spec: str = "bestvideo*+bestaudio/best",
        timeout: float = 1800.0,
        page_url: str = "",
    ) -> dict[str, Any]:
        """Download with the full fallback chain: direct (yt-dlp, retry +
        backoff) → proxy lab → browser-context fetch → honest failure.

        ``page_url`` is the source page for the browser stage (e.g. the
        YouTube watch page or a download-blog post) when it differs from
        ``url`` itself.
        """
        from ..media.downloader import MediaDownloader

        downloader = MediaDownloader(context)
        report = downloader.download(
            url, audio_only=audio_only, format_spec=format_spec,
            page_url=page_url, timeout=timeout)
        if not report.ok:
            raise MediaError(report.summary())
        result = {
            "url": url, "path": report.path, "bytes": report.size_bytes,
            "title": report.title, "extractor": report.extractor,
            "seconds": report.total_s,
            "stages": report.tried(),
        }
        db = getattr(context, "db", None) if context is not None else None
        if db is not None:
            from ..core.ids import new_id

            try:
                db.insert(
                    "media_items",
                    {
                        "id": new_id(), "url": url, "extractor": result.get("extractor", ""),
                        "kind": "audio" if audio_only else "video", "title": result.get("title", ""),
                        "path": result["path"], "size_bytes": result["bytes"],
                        "status": "done", "created_at": time.time(), "metadata": "{}",
                    },
                )
            except Exception as exc:  # noqa: BLE001 - bookkeeping must not fail the download
                _log.debug("could not record media item: %s", exc)
        return result

    @registry.register(
        "media_probe",
        description="Fetch media metadata (title, duration, formats) without downloading.",
        capability=Capability.NET_OUT,
    )
    def media_probe(url: str) -> dict[str, Any]:
        return probe(url)

    @registry.register(
        "media_capability",
        description="Report which media backends are available in this environment.",
        capability=Capability.FS_READ,
    )
    def media_capability() -> dict[str, Any]:
        return {
            "yt_dlp_python": available("yt_dlp"),
            "yt_dlp_cli": bool(which("yt-dlp")),
            "ffmpeg": bool(which("ffmpeg")),
            "ffprobe": bool(which("ffprobe")),
            "direct_http": True,
            "site_extraction": available("yt_dlp") or bool(which("yt-dlp")),
        }
