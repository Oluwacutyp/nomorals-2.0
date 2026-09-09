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


def probe(url: str, *, timeout: float = 60.0) -> dict[str, Any]:
    """Fetch metadata without downloading."""
    module = load_optional("yt_dlp")
    if module is not None:
        try:
            with module.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
                info = ydl.extract_info(url, download=False)
            return _summarize(info)
        except Exception as exc:  # noqa: BLE001 - fall through to the CLI
            _log.debug("yt_dlp python API probe failed: %s", exc)
    if which("yt-dlp"):
        try:
            completed = subprocess.run(
                ["yt-dlp", "--dump-single-json", "--no-download", url],
                capture_output=True, text=True, timeout=timeout, check=False,
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
    if audio_only:
        options["postprocessors"] = [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}
        ]
    if progress is not None:
        def hook(state: dict[str, Any]) -> None:
            if state.get("status") == "downloading":
                progress(f"{state.get('_percent_str', '?').strip()} of {state.get('_total_bytes_str', '?')}")

        options["progress_hooks"] = [hook]
    try:
        with module.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info) if info else ""
    except Exception as exc:  # noqa: BLE001
        raise MediaError(f"download failed: {exc}") from exc

    path = Path(filename) if filename else None
    if path is None or not path.exists():
        candidates = sorted(target_dir.iterdir(), key=lambda p: -p.stat().st_size)
        path = candidates[0] if candidates else None
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
    argv = [
        "yt-dlp", "-f", format_spec, "-o", str(target_dir / "%(title).120B [%(id)s].%(ext)s"),
        "--no-playlist", "--retries", "3", "--newline", url,
    ]
    if audio_only:
        argv += ["-x", "--audio-format", "mp3"]
    if which("ffmpeg"):
        argv += ["--ffmpeg-location", which("ffmpeg") or "ffmpeg"]
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"download timed out after {timeout}s") from exc
    if completed.returncode != 0:
        raise MediaError(f"yt-dlp failed: {completed.stderr.strip()[-400:]}")
    candidates = sorted(target_dir.iterdir(), key=lambda p: -p.stat().st_size)
    if not candidates:
        raise MediaError("yt-dlp produced no file")
    path = candidates[0]
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
        if progress is not None and total:
            progress(f"{done * 100 // total}% ({done}/{total} bytes)")

    HttpClient(timeout=600.0).download(url, target, resume=True, progress=hook)
    if not target.exists() or target.stat().st_size == 0:
        raise MediaError("direct download produced an empty file")
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
    ) -> dict[str, Any]:
        from .filesystem import safe_path

        target_dir = safe_path(context, "media")
        result = download(
            url, target_dir, format_spec=format_spec, audio_only=audio_only, timeout=timeout
        )
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
                        "status": "done", "created_at": time.time(), "metadata": {},
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
