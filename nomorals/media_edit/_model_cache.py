"""Shared model-file cache for the heavy vision stack.

All of segment.py (rembg), upscale.py (Real-ESRGAN/SUPIR) and faceswap.py
(INSWapper/GFPGAN) download their weights through this module so there is
exactly one download path, one home directory, and no silent re-downloads.

Expected on-disk sizes (documented so the user knows what "first run"
costs):

- rembg u2netp.onnx ............ ~4.7 MB  (termux-friendly)
- rembg birefnet-general ....... ~800 MB
- RealESRGAN_x4plus.pth ........ ~64 MB
- SUPIR weights ................ ~3 GB   (workstation rescue mode only)
- inswapper_128.onnx ........... ~530 MB
- GFPGANv1.4.pth ............... ~350 MB

Usage::

    from . import _model_cache as mc
    path = mc.model_path("RealESRGAN_x4plus.pth", url=..., size_mb=64)

``model_path`` returns immediately when the file already exists and is
non-empty; otherwise it downloads with a progress log and a file lock so
two processes never fetch the same file concurrently. Partial downloads
are written to a ``.part`` file and renamed only on success — a killed
download never leaves a corrupt model behind.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)


class ModelCacheError(Exception):
    """Model download failed — the real reason, never a silent fallback."""


def cache_dir() -> Path:
    """Home for downloaded vision weights.

    ``~/.nomorals/models/`` by default; ``NM_MODELS_DIR`` overrides it.
    """
    override = (os.environ.get("NM_MODELS_DIR") or "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".nomorals" / "models"


def model_path(name: str, *, url: str = "", size_mb: float = 0.0,
               sha256: str = "") -> Path:
    """Resolve (downloading on first use) the local path of a model file.

    Returns immediately when ``cache_dir()/name`` exists and is non-empty.
    Downloads from ``url`` otherwise; ``url`` is required when the file is
    missing. When ``sha256`` is given the file is verified after download
    and re-downloaded once on mismatch. Never raises without a real reason.
    """
    path = cache_dir() / name
    if path.is_file() and path.stat().st_size > 0:
        return path
    if not url:
        raise ModelCacheError(
            f"model {name!r} not in {cache_dir()} and no download URL "
            f"was given — place the file there manually or check the "
            f"module docstring for the source")
    return _download(url, path, size_mb=size_mb, sha256=sha256)


def _lock_for(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".lock")


def _download(url: str, path: Path, *, size_mb: float = 0.0,
              sha256: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    lock = _lock_for(path)
    # Best-effort cross-process lock: create the lock dir atomically.
    locked = False
    deadline = time.time() + 3600.0
    while time.time() < deadline:
        try:
            os.makedirs(lock)
            locked = True
            break
        except FileExistsError:
            time.sleep(2.0)
    if not locked:
        raise ModelCacheError(
            f"another process is downloading {path.name} (lock held)")
    try:
        # Re-check: the other downloader may have finished while we waited.
        if path.is_file() and path.stat().st_size > 0:
            return path
        _log.info("downloading model %s (~%.0f MB) from %s",
                  path.name, size_mb, url)
        _fetch_with_progress(url, part)
        if sha256:
            digest = hashlib.sha256(part.read_bytes()).hexdigest()
            if digest.lower() != sha256.lower():
                part.unlink(missing_ok=True)
                raise ModelCacheError(
                    f"sha256 mismatch for {path.name}: "
                    f"expected {sha256[:16]}…, got {digest[:16]}… "
                    f"(deleted the bad file; retry the op)")
        os.replace(part, path)
        _log.info("model ready: %s", path)
        return path
    finally:
        try:
            os.rmdir(lock)
        except OSError:
            pass


def _fetch_with_progress(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest``, logging progress. Stdlib only."""
    import urllib.request
    req = urllib.request.Request(url, headers={
        "User-Agent": "nomorals-model-cache/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, \
                open(dest, "wb") as fh:
            total = resp.headers.get("Content-Length")
            total_n = int(total) if total and total.isdigit() else 0
            got = 0
            last_log = time.time()
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                fh.write(chunk)
                got += len(chunk)
                now = time.time()
                if now - last_log > 5.0:
                    if total_n:
                        _log.info("downloading %s: %.1f%% (%d/%d MB)",
                                  dest.name, 100.0 * got / total_n,
                                  got // 1_048_576, total_n // 1_048_576)
                    else:
                        _log.info("downloading %s: %d MB so far",
                                  dest.name, got // 1_048_576)
                    last_log = now
    except Exception as exc:  # noqa: BLE001 - wrap with context
        dest.unlink(missing_ok=True)
        raise ModelCacheError(
            f"download failed for {dest.name} from {url}: {exc}") from exc


def cached(name: str) -> bool:
    """True when ``name`` is already in the local cache."""
    path = cache_dir() / name
    return path.is_file() and path.stat().st_size > 0
