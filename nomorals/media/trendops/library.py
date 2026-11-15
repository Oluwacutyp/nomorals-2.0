"""Format library — trend formats as durable JSON, with expiry.

One file per format under ``<root>/formats/<slug>.json`` (the same
atomic tmp+rename pattern the shorts :class:`JobStore` uses), so a
crash mid-write never leaves a torn format. A fresh library seeds the
Roadster format from :mod:`nomorals.media.trendops.formats`.

Expiry is enforced, never advisory: :meth:`FormatLibrary.list` hides
dead formats by default, :meth:`FormatLibrary.get` raises on an
expired slug naming the expiry, and :meth:`FormatLibrary.prune`
deletes them. A trend that peaked is gone — the library never applies
a dead trend to new footage.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ...core.logging_setup import get_logger
from ...media_edit.videos import MediaEditError
from .formats import (
    DEFAULT_EXPIRY_DAYS,
    ROADSTER_SEED,
    SEED_EXPIRY_DAYS,
    TrendFormat,
)

__all__ = ["FormatLibrary"]

_log = get_logger(__name__)


class FormatLibrary:
    """Durable store for :class:`TrendFormat` records."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else (
            Path.home() / "workspace" / "devon" / ".trendops")
        self.dir = self.root / "formats"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._seeded = False

    # -- paths ---------------------------------------------------------

    def _path(self, slug: str) -> Path:
        return self.dir / f"{slug}.json"

    def _seed(self) -> None:
        if self._seeded:
            return
        self._seeded = True
        if not any(self.dir.glob("*.json")):
            now = time.time()
            seed = dict(ROADSTER_SEED)
            seed["created_at"] = now
            seed["expires_at"] = now + SEED_EXPIRY_DAYS * 86400.0
            self.save(TrendFormat.from_dict(seed))
            _log.info("trendops: seeded the roadster format")

    # -- CRUD ----------------------------------------------------------

    def save(self, fmt: TrendFormat) -> TrendFormat:
        """Write a format (atomic tmp+rename). Returns the format."""
        if not isinstance(fmt, TrendFormat):
            raise MediaEditError(
                f"FormatLibrary.save needs a TrendFormat, got "
                f"{type(fmt).__name__}")
        path = self._path(fmt.slug)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(fmt.to_dict(), indent=2),
                       encoding="utf-8")
        tmp.replace(path)
        return fmt

    def get(self, slug: str, *, live_only: bool = True,
            now: float | None = None) -> TrendFormat:
        """Load a format. Raises on unknown slug, and on an expired one
        when ``live_only`` (naming the expiry — a dead trend never
        applies silently)."""
        self._seed()
        slug = str(slug or "").strip().lower()
        path = self._path(slug)
        if not path.exists():
            raise MediaEditError(
                f"unknown trend format {slug!r}; use: {self.list()}")
        try:
            fmt = TrendFormat.from_dict(
                json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, MediaEditError) as exc:
            raise MediaEditError(
                f"trend format {slug!r} is unreadable: {exc}") from exc
        if live_only and not fmt.is_live(now):
            raise MediaEditError(
                f"trend format {slug!r} expired "
                f"{time.strftime('%Y-%m-%d', time.localtime(fmt.expires_at))} "
                "— capture it fresh or pick a live format")
        return fmt

    def list(self, *, live_only: bool = True,
             now: float | None = None) -> list[str]:
        """Slugs on disk (live ones only by default, sorted)."""
        self._seed()
        now = float(now if now is not None else time.time())
        out: list[str] = []
        for path in sorted(self.dir.glob("*.json")):
            try:
                fmt = TrendFormat.from_dict(
                    json.loads(path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, MediaEditError):
                continue
            if live_only and not fmt.is_live(now):
                continue
            out.append(fmt.slug)
        return out

    def all(self, *, live_only: bool = True,
            now: float | None = None) -> list[TrendFormat]:
        """Formats on disk (live ones only by default)."""
        now = float(now if now is not None else time.time())
        out: list[TrendFormat] = []
        for slug in self.list(live_only=False, now=now):
            fmt = self.get(slug, live_only=False, now=now)
            if live_only and not fmt.is_live(now):
                continue
            out.append(fmt)
        return out

    def remove(self, slug: str) -> bool:
        """Delete a format. True when a file was removed."""
        self._seed()
        path = self._path(str(slug or "").strip().lower())
        if not path.exists():
            return False
        path.unlink()
        return True

    def bump_uses(self, slug: str) -> None:
        """Count one successful apply (best effort — a counter must
        never fail a render)."""
        try:
            fmt = self.get(slug, live_only=False)
        except MediaEditError:
            return
        fmt.uses = int(fmt.uses) + 1
        try:
            self.save(fmt)
        except OSError as exc:
            _log.debug("trendops: could not bump uses for %s: %s", slug, exc)

    def prune(self, *, now: float | None = None) -> list[str]:
        """Delete expired formats. Returns the pruned slugs."""
        now = float(now if now is not None else time.time())
        pruned: list[str] = []
        for slug in self.list(live_only=False, now=now):
            fmt = self.get(slug, live_only=False, now=now)
            if not fmt.is_live(now):
                self.remove(slug)
                pruned.append(slug)
        return pruned

    # -- capture helper -------------------------------------------------

    def new_format(self, slug: str, *, source: str = "",
                   niche: str = "", reference_url: str = "",
                   expires_days: float = DEFAULT_EXPIRY_DAYS,
                   **overrides: Any) -> TrendFormat:
        """Build + save a format from the roadster defaults, with the
        caller's overrides. ``expires_days`` from now (0 = never)."""
        now = time.time()
        data = dict(ROADSTER_SEED)
        data.update({
            "slug": slug, "source": source, "niche": niche,
            "reference_url": reference_url, "created_at": now,
            "expires_at": (now + float(expires_days) * 86400.0
                           if expires_days else 0.0),
            "uses": 0, "measured": {},
        })
        data.update(overrides)
        fmt = TrendFormat.from_dict(data)
        return self.save(fmt)
