"""Durable show store: shows + composed episodes as one JSON file.

Layout: ``<home>/data/shows.json`` (the /show surface points it at
Devon's home; in-memory when no path is given)::

    {"shows": [<Show.to_dict()>, ...],
     "episodes": [<Episode.to_dict() plus publish state>, ...]}

The persistence shape mirrors
:class:`nomorals.publicface.publish.PublicPageStore`: tolerant load
(a missing file starts empty, a bad row is skipped), atomic tmp+replace
save after every mutation. One difference, deliberate: a file that is
not JSON at all is QUARANTINED (renamed to
``shows.json.corrupt-<epoch>``) and the load RAISES — a torn or
hand-edited store must never silently start empty and lose every
show. The composer itself (``composer.py``) stays pure: it never
touches this file.

Stdlib only.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

__all__ = ["ShowStore", "default_shows_path"]


def default_shows_path(home: Path | str | None = None) -> Path:
    """The durable shows file under a home (``<home>/data/shows.json``)."""
    base = Path(home) if home is not None else Path.home()
    return base / "data" / "shows.json"


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (tmp file + ``os.replace``)."""
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class ShowStore:
    """Shows + episodes, durable JSON, one atomic file."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._lock = threading.RLock()
        self._shows: dict[str, dict[str, Any]] = {}
        self._episodes: dict[str, dict[str, Any]] = {}
        if self.path is not None:
            self._load()

    # ── load / save ────────────────────────────────────────────────
    def _load(self) -> None:
        assert self.path is not None
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            bad = self.path.with_name(
                f"{self.path.name}.corrupt-{int(time.time())}")
            try:
                os.replace(self.path, bad)
            except OSError:
                bad = self.path
            raise ValueError(
                f"shows store is corrupt — quarantined to {bad}: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise ValueError(
                f"shows store must hold an object, found "
                f"{type(raw).__name__}")
        for row in raw.get("shows", []) or []:
            if (isinstance(row, dict) and row.get("id")
                    and str(row.get("title") or "").strip()):
                self._shows[str(row["id"])] = dict(row)
        for row in raw.get("episodes", []) or []:
            if (isinstance(row, dict) and row.get("id")
                    and row.get("show_id")):
                self._episodes[str(row["id"])] = dict(row)

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            self.path,
            json.dumps({"shows": list(self._shows.values()),
                        "episodes": list(self._episodes.values())},
                       indent=2, sort_keys=True))

    # ── shows ──────────────────────────────────────────────────────
    def put_show(self, show: dict[str, Any]) -> dict[str, Any]:
        """Store a show dict (a ``Show.to_dict()``); upsert by id."""
        row = dict(show)
        if not str(row.get("id") or "").strip():
            raise ValueError("show needs an id")
        if not str(row.get("title") or "").strip():
            raise ValueError("show needs a title")
        with self._lock:
            self._shows[str(row["id"])] = row
            self._save()
            return dict(row)

    def list_shows(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._shows.values()]

    def get_show(self, show_id: str) -> dict[str, Any] | None:
        row = self._shows.get(str(show_id))
        return dict(row) if row is not None else None

    def delete_show(self, show_id: str) -> bool:
        """Delete a show and every episode composed for it."""
        with self._lock:
            if str(show_id) not in self._shows:
                return False
            del self._shows[str(show_id)]
            doomed = [eid for eid, ep in self._episodes.items()
                      if str(ep.get("show_id")) == str(show_id)]
            for eid in doomed:
                del self._episodes[eid]
            self._save()
            return True

    # ── episodes ───────────────────────────────────────────────────
    def put_episode(self, episode: dict[str, Any]) -> dict[str, Any]:
        """Store an episode dict (an ``Episode.to_dict()``); upsert."""
        row = dict(episode)
        if not str(row.get("id") or "").strip():
            raise ValueError("episode needs an id")
        if not str(row.get("show_id") or "").strip():
            raise ValueError("episode needs a show_id")
        with self._lock:
            self._episodes[str(row["id"])] = row
            self._save()
            return dict(row)

    def list_episodes(self, show_id: str) -> list[dict[str, Any]]:
        return [dict(ep) for ep in self._episodes.values()
                if str(ep.get("show_id")) == str(show_id)]

    def get_episode(self, episode_id: str) -> dict[str, Any] | None:
        row = self._episodes.get(str(episode_id))
        return dict(row) if row is not None else None

    def delete_episode(self, episode_id: str) -> bool:
        with self._lock:
            if str(episode_id) not in self._episodes:
                return False
            del self._episodes[str(episode_id)]
            self._save()
            return True

    def set_episode_published(self, episode_id: str, *,
                              published: bool, link: str | None = None
                              ) -> dict[str, Any] | None:
        """Record a publish attempt's outcome on the episode row."""
        with self._lock:
            row = self._episodes.get(str(episode_id))
            if row is None:
                return None
            row = dict(row)
            row["published"] = bool(published)
            row["publish_link"] = link or ""
            self._episodes[str(episode_id)] = row
            self._save()
            return dict(row)
