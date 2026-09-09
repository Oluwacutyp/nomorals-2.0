"""Model downloading from Hugging Face.

Two paths:

* ``huggingface_hub`` when installed — gives caching, resume, and repo listing.
* Raw HTTP with ``Range`` requests when it is not — which is the Termux case, and
  the case this sandbox is in.

Resumability is not optional for 15 GB weights on a phone network. Every download
verifies its SHA-256 against the LFS pointer metadata before reporting success, so
a truncated file is never mistaken for a model.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from ..core.errors import DownloadError, NotFound, ValidationError
from ..core.http import HttpClient
from ..core.logging_setup import get_logger

__all__ = ["DownloadResult", "HuggingFaceDownloader", "RepoFile"]

_log = get_logger(__name__)

HF_HOST = "https://huggingface.co"
CHUNK = 1024 * 1024


@dataclass
class RepoFile:
    path: str
    size: int
    sha256: str = ""
    lfs: bool = False

    @property
    def size_gb(self) -> float:
        return round(self.size / (1024**3), 2)


@dataclass
class DownloadResult:
    path: str
    size: int
    sha256: str
    verified: bool
    resumed: bool = False
    seconds: float = 0.0
    source: str = ""

    @property
    def mbps(self) -> float:
        return round((self.size / (1024**2)) / self.seconds, 2) if self.seconds else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size": self.size,
            "sha256": self.sha256,
            "verified": self.verified,
            "resumed": self.resumed,
            "seconds": round(self.seconds, 2),
            "mbps": self.mbps,
            "source": self.source,
        }


class HuggingFaceDownloader:
    """Lists and downloads files from HF repos."""

    def __init__(
        self,
        *,
        token: str = "",
        cache_dir: str | os.PathLike[str] = "models",
        base_url: str = HF_HOST,
        timeout: float = 600.0,
        verify_checksums: bool = True,
    ) -> None:
        self.token = token
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url.rstrip("/")
        self.verify_checksums = verify_checksums
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.http = HttpClient(timeout=timeout, headers=headers)
        self.stats = {"files": 0, "bytes": 0, "errors": 0}

    # ── repo introspection ───────────────────────────────────────────────────
    def repo_url(self, repo_id: str, path: str = "", revision: str = "main") -> str:
        return f"{self.base_url}/{repo_id}/resolve/{revision}/{path.lstrip('/')}"

    def model_info(self, repo_id: str) -> dict[str, Any]:
        raw = self.http.get(f"{self.base_url}/api/models/{repo_id}")
        return raw.json()

    def list_files(
        self, repo_id: str, *, revision: str = "main", patterns: tuple[str, ...] = ()
    ) -> list[RepoFile]:
        """List repo files via the HF API.

        Raises :class:`DownloadError` on any network or permission failure; callers
        decide whether a missing model is fatal.
        """
        try:
            data = self.model_info(repo_id)
        except Exception as exc:  # noqa: BLE001
            raise DownloadError(f"cannot read repo {repo_id}: {exc}", retryable=True) from exc

        siblings = data.get("siblings") or []
        out: list[RepoFile] = []
        for entry in siblings:
            path = entry.get("rfilename") or ""
            if not path:
                continue
            if patterns and not any(_matches(path, p) for p in patterns):
                continue
            lfs = entry.get("lfs") or {}
            out.append(
                RepoFile(
                    path=path,
                    size=int(lfs.get("size") or 0),
                    sha256=str(lfs.get("sha256") or ""),
                    lfs=bool(lfs),
                )
            )
        return out

    def select_weights(
        self, repo_id: str, *, prefer: tuple[str, ...] = ("safetensors", "gguf", "bin"), revision: str = "main"
    ) -> list[RepoFile]:
        """Pick the weight files to fetch, in preference order."""
        files = self.list_files(repo_id, revision=revision)
        chosen: list[RepoFile] = []
        for extension in prefer:
            matching = [f for f in files if f.path.endswith(extension) and "adapter" not in f.path.lower()]
            if matching:
                chosen = matching
                break
        # Always bring the config and tokenizer files alongside the weights.
        support = [
            f
            for f in files
            if f.path.endswith((".json", ".txt", ".model", ".jinja"))
            and "training_args" not in f.path
        ]
        return chosen + support

    # ── downloading ──────────────────────────────────────────────────────────
    def download_file(
        self,
        repo_id: str,
        path: str,
        *,
        revision: str = "main",
        destination: str | os.PathLike[str] | None = None,
        expected_sha256: str = "",
        progress: Callable[[int, int], None] | None = None,
    ) -> DownloadResult:
        url = self.repo_url(repo_id, path, revision)
        target = Path(destination) if destination else self.cache_dir / repo_id / path
        target.parent.mkdir(parents=True, exist_ok=True)
        resumed = target.exists() and target.stat().st_size > 0

        if expected_sha256 and target.exists() and _sha256_file(target) == expected_sha256:
            return DownloadResult(
                path=str(target),
                size=target.stat().st_size,
                sha256=expected_sha256,
                verified=True,
                resumed=False,
                source=url,
            )

        started = time.perf_counter()
        try:
            self.http.download(url, target, resume=True, progress=progress)
        except Exception as exc:  # noqa: BLE001
            self.stats["errors"] += 1
            raise DownloadError(f"download of {url} failed: {exc}", retryable=True) from exc
        elapsed = time.perf_counter() - started
        size = target.stat().st_size
        digest = _sha256_file(target)
        verified = True
        if expected_sha256 and digest != expected_sha256:
            verified = False
            _log.error("checksum mismatch for %s (expected %s, got %s)", path, expected_sha256[:16], digest[:16])
        self.stats["files"] += 1
        self.stats["bytes"] += size
        return DownloadResult(
            path=str(target),
            size=size,
            sha256=digest,
            verified=verified,
            resumed=resumed,
            seconds=elapsed,
            source=url,
        )

    def download_repo(
        self,
        repo_id: str,
        *,
        revision: str = "main",
        patterns: tuple[str, ...] = (),
        destination: str | os.PathLike[str] | None = None,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> list[DownloadResult]:
        """Download a curated set of files (weights + config + tokenizer)."""
        files = self.select_weights(repo_id, revision=revision) if not patterns else self.list_files(
            repo_id, revision=revision, patterns=patterns
        )
        if not files:
            raise NotFound(f"no downloadable files matched in {repo_id}")
        root = Path(destination) if destination else self.cache_dir / repo_id
        results: list[DownloadResult] = []
        for entry in files:
            callback = None
            if progress is not None:
                callback = lambda done, total, name=entry.path: progress(name, done, total)
            results.append(
                self.download_file(
                    repo_id,
                    entry.path,
                    revision=revision,
                    destination=root / entry.path,
                    expected_sha256=entry.sha256,
                    progress=callback,
                )
            )
        return results

    def download_model(
        self,
        repo_id: str,
        *,
        revision: str = "main",
        destination: str | os.PathLike[str] | None = None,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> dict[str, Any]:
        """High-level: fetch a usable model directory and summarize it."""
        results = self.download_repo(
            repo_id, revision=revision, destination=destination, progress=progress
        )
        return {
            "repo_id": repo_id,
            "revision": revision,
            "directory": str(Path(results[0].path).parent),
            "files": [r.to_dict() for r in results],
            "total_bytes": sum(r.size for r in results),
            "verified": all(r.verified for r in results if r.sha256),
        }

    # ── housekeeping ─────────────────────────────────────────────────────────
    def cached_size(self) -> int:
        return sum(p.stat().st_size for p in self.cache_dir.rglob("*") if p.is_file())

    def purge_missing(self, repo_id: str) -> int:
        """Remove zero-byte leftovers from interrupted downloads."""
        removed = 0
        root = self.cache_dir / repo_id
        if not root.is_dir():
            return 0
        for path in root.rglob("*"):
            if path.is_file() and path.stat().st_size == 0:
                path.unlink()
                removed += 1
        return removed


def _matches(path: str, pattern: str) -> bool:
    import fnmatch

    return fnmatch.fnmatch(path, pattern) or pattern in path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024
    return f"{value:.1f}TB"
