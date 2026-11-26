"""Stock footage for the shorts factory — Pexels + Pixabay, free keys.

MoneyPrinterTurbo's stock rung, ported fresh: per-scene real footage
search over Pexels and Pixabay, downloaded through the house HTTP
client. Keys are free accounts (a login each, no payment) read from
the environment — ``NM_PEXELS_API_KEY`` / ``NM_PIXABAY_API_KEY``. No
key means the provider is honestly unavailable (``ok: False`` naming
the wall), never a fake clip; the pipeline falls back to generated
stills per scene.

Portrait clips only (the factory renders 9:16 vertical): the best
portrait mp4 file of the best result wins. The clip itself is the
artifact; the pipeline extracts a poster frame from it for the
Ken Burns scene — the edit engine's scene model is stills-only, so
clip-native video scenes are a named follow-up, not smuggled in here.

Do NOT vendor any stock provider's assets; clips are fetched live per
scene and the manifest records the source URL, so the run is honest
about where every frame came from.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ...core.http import HttpClient
from ...core.logging_setup import get_logger

_log = get_logger(__name__)

PEXELS_API = "https://api.pexels.com/videos/search"
PIXABAY_API = "https://pixabay.com/api/videos/"

PEXELS_ENV = "NM_PEXELS_API_KEY"
PIXABAY_ENV = "NM_PIXABAY_API_KEY"

# A per-scene stock clip bigger than this is a mistake, not footage.
MAX_CLIP_BYTES = 200_000_000

# Target vertical frame for the 9:16 render.
TARGET_W, TARGET_H = 1080, 1920


def stock_status() -> dict[str, Any]:
    """Which stock providers are configured (keys present)."""
    pexels = bool(os.environ.get(PEXELS_ENV))
    pixabay = bool(os.environ.get(PIXABAY_ENV))
    return {
        "pexels": pexels,
        "pixabay": pixabay,
        "configured": [n for n, ok in (("pexels", pexels),
                                       ("pixabay", pixabay)) if ok],
        "env": {PEXELS_ENV: pexels, PIXABAY_ENV: pixabay},
    }


# ── house-client adapter ─────────────────────────────────────────────

class _HouseClient:
    """Thin adapter over the house HttpClient.

    The stock functions take any object with ``get_json`` / ``download``
    (tests inject a fake); this is the real one.
    """

    def __init__(self) -> None:
        self._http = HttpClient(timeout=30.0)

    def get_json(self, url: str, *, headers: dict[str, str] | None = None,
                 params: dict[str, Any] | None = None) -> dict[str, Any]:
        resp = self._http.get(url, headers=headers or {}, params=params or {})
        if not resp.ok:
            raise RuntimeError(
                f"stock search failed: HTTP {resp.status} from {url}")
        data = resp.json()
        if not isinstance(data, dict):
            raise RuntimeError("stock search failed: non-object JSON")
        return data

    def download(self, url: str, dest: str | Path) -> Path:
        return self._http.download(url, dest)


# ── search + normalize ───────────────────────────────────────────────

def _pick_video_file(files: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Best portrait mp4 file: height > width, closest to 1080x1920."""
    best: dict[str, Any] | None = None
    best_score: float | None = None
    for f in files:
        link = str(f.get("link") or f.get("url") or "")
        if not link:
            continue
        if str(f.get("file_type") or "").lower() not in ("video/mp4", ""):
            continue
        w = int(f.get("width") or 0)
        h = int(f.get("height") or 0)
        if w <= 0 or h <= 0 or h <= w:
            continue  # portrait only
        score = abs(w - TARGET_W) + abs(h - TARGET_H)
        if best_score is None or score < best_score:
            best, best_score = f, score
    return best


def search_pexels(query: str, *, per_page: int = 5,
                  client: Any | None = None) -> list[dict[str, Any]]:
    """Search Pexels videos; returns normalized clip records."""
    key = os.environ.get(PEXELS_ENV, "")
    if not key:
        raise RuntimeError(
            f"Pexels not configured — set {PEXELS_ENV} (free Pexels "
            "account) to enable stock footage")
    client = client or _HouseClient()
    data = client.get_json(
        PEXELS_API,
        headers={"Authorization": key},
        params={"query": query, "per_page": per_page,
                "orientation": "portrait", "size": "medium"})
    clips: list[dict[str, Any]] = []
    for video in data.get("videos") or []:
        f = _pick_video_file(list(video.get("video_files") or []))
        if not f:
            continue
        clips.append({
            "provider": "pexels",
            "id": str(video.get("id", "")),
            "width": int(f.get("width") or 0),
            "height": int(f.get("height") or 0),
            "duration": float(video.get("duration") or 0),
            "file_url": str(f.get("link") or ""),
            "source_url": str(video.get("url") or ""),
            "preview_url": str(video.get("image") or ""),
        })
    return clips


def search_pixabay(query: str, *, per_page: int = 5,
                   client: Any | None = None) -> list[dict[str, Any]]:
    """Search Pixabay videos; returns normalized clip records."""
    key = os.environ.get(PIXABAY_ENV, "")
    if not key:
        raise RuntimeError(
            f"Pixabay not configured — set {PIXABAY_ENV} (free Pixabay "
            "account) to enable stock footage")
    client = client or _HouseClient()
    data = client.get_json(
        PIXABAY_API,
        params={"key": key, "q": query, "per_page": per_page,
                "orientation": "vertical", "video_type": "all"})
    clips: list[dict[str, Any]] = []
    for hit in data.get("hits") or []:
        variants = hit.get("videos") or {}
        files = []
        for name in ("large", "medium", "small", "tiny"):
            v = variants.get(name) or {}
            if v.get("url"):
                files.append({"url": v.get("url"),
                              "width": v.get("width") or 0,
                              "height": v.get("height") or 0,
                              "file_type": "video/mp4"})
        f = _pick_video_file(files)
        if not f:
            continue
        clips.append({
            "provider": "pixabay",
            "id": str(hit.get("id", "")),
            "width": int(f.get("width") or 0),
            "height": int(f.get("height") or 0),
            "duration": float(hit.get("duration") or 0),
            "file_url": str(f.get("url") or ""),
            "source_url": str(hit.get("pageURL") or ""),
            "preview_url": "",
        })
    return clips


# ── fetch + poster ───────────────────────────────────────────────────

def fetch_stock_clip(query: str, out_dir: str | Path, *,
                     prefer: str = "pexels", per_page: int = 5,
                     timeout: float = 120.0,
                     max_bytes: int = MAX_CLIP_BYTES,
                     client: Any | None = None) -> dict[str, Any]:
    """Download the best portrait stock clip for ``query``.

    Tries Pexels first, then Pixabay (or the reverse when
    ``prefer="pixabay"``); providers without a key are skipped, not
    faked. Returns ``{"ok": True, "path", "provider", "source_url",
    ...}`` or ``{"ok": False, "reason"}`` — never raises for an
    unconfigured or empty provider.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    order = ["pexels", "pixabay"] if prefer != "pixabay" else [
        "pixabay", "pexels"]
    searchers = {"pexels": search_pexels, "pixabay": search_pixabay}
    tried: list[str] = []
    for name in order:
        if name == "pexels" and not os.environ.get(PEXELS_ENV):
            tried.append("pexels: no key")
            continue
        if name == "pixabay" and not os.environ.get(PIXABAY_ENV):
            tried.append("pixabay: no key")
            continue
        try:
            clips = searchers[name](query, per_page=per_page, client=client)
        except Exception as exc:  # noqa: BLE001 - provider down = fallback
            _log.warning("stock: %s search failed: %s", name, exc)
            tried.append(f"{name}: {exc}")
            continue
        if not clips:
            tried.append(f"{name}: no portrait clips for {query!r}")
            continue
        clip = clips[0]
        dest = out / f"stock_{name}_{clip['id']}.mp4"
        try:
            http = client if client is not None and hasattr(
                client, "download") else _HouseClient()
            http.download(clip["file_url"], dest)
        except Exception as exc:  # noqa: BLE001 - download fail = fallback
            _log.warning("stock: %s download failed: %s", name, exc)
            tried.append(f"{name}: download failed: {exc}")
            continue
        if not dest.exists() or dest.stat().st_size == 0:
            tried.append(f"{name}: empty download")
            continue
        if dest.stat().st_size > max_bytes:
            dest.unlink(missing_ok=True)
            tried.append(f"{name}: clip over {max_bytes} bytes")
            continue
        return {"ok": True, "path": str(dest), "provider": name,
                "source_url": clip["source_url"],
                "clip_id": clip["id"], "width": clip["width"],
                "height": clip["height"], "duration": clip["duration"],
                "query": query}
    return {"ok": False,
            "reason": "no stock footage: " + "; ".join(tried),
            "tried": tried}


def extract_poster(clip_path: str | Path, out_path: str | Path,
                   *, at: float | None = None) -> dict[str, Any]:
    """Extract one frame from a stock clip (midpoint by default).

    Returns ``{"ok": True, "path"}`` or ``{"ok": False, "reason"}``.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return {"ok": False, "reason": "ffmpeg not on PATH"}
    clip = Path(clip_path)
    if not clip.exists():
        return {"ok": False, "reason": f"clip missing: {clip}"}
    if at is None:
        # Midpoint via ffprobe when available; 1.0s otherwise.
        probe = shutil.which("ffprobe")
        at = 1.0
        if probe:
            try:
                r = subprocess.run(
                    [probe, "-v", "error", "-show_entries",
                     "format=duration", "-of",
                     "default=noprint_wrappers=1:nokey=1", str(clip)],
                    capture_output=True, text=True, timeout=30)
                at = max(0.1, float(r.stdout.strip()) / 2)
            except Exception:  # noqa: BLE001 - midpoint guess stands
                at = 1.0
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-ss", f"{at:.2f}",
             "-i", str(clip), "-frames:v", "1", str(out)],
            check=True, timeout=60)
    except (subprocess.CalledProcessError,
            subprocess.TimeoutExpired) as exc:
        return {"ok": False, "reason": f"ffmpeg poster failed: {exc}"}
    if not out.exists():
        return {"ok": False, "reason": "poster frame not written"}
    return {"ok": True, "path": str(out)}
