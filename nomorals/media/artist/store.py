"""Durable artist-IDE scene store.

Scenes live as one JSON document under ``<home>/artist/scenes.json``
(the triage/photo-editor house pattern): atomic tmp+replace writes,
a corrupt file is quarantined to ``scenes.json.corrupt.<ts>`` and the
store comes back empty instead of crashing the tool.
"""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .scene import ArtistError, Scene


def artist_path(settings: Any = None) -> Path:
    """The artist store path, honouring the runtime settings home."""
    if settings is not None:
        home = getattr(settings, "home_path", None)
        if home is not None:
            return Path(home) / "artist" / "scenes.json"
    return Path.home() / ".nomorals" / "artist" / "scenes.json"


def artist_dir(settings: Any = None) -> Path:
    """The artist home directory (renders + scenes live under it)."""
    return artist_path(settings).parent


class ArtistStore:
    """JSON scene store: id -> scene dict, loaded lazily, saved whole."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._data: dict[str, dict[str, Any]] | None = None

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._data is not None:
            return self._data
        if not self.path.exists():
            self._data = {}
            return self._data
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            quarantine = self.path.with_name(
                f"scenes.json.corrupt.{stamp}")
            with contextlib.suppress(OSError):
                self.path.replace(quarantine)
            self._data = {}
            return self._data
        if not isinstance(raw, dict):
            self._data = {}
            return self._data
        self._data = {str(k): v for k, v in raw.items()
                      if isinstance(v, dict)}
        return self._data

    def _save(self) -> None:
        data = self._load()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, tmp_name = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=".scenes-", suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as handle_file:
                json.dump(data, handle_file, indent=2, sort_keys=True)
                handle_file.write("\n")
            os.replace(tmp_name, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise

    def save(self, scene: Scene) -> None:
        self._load()[scene.id] = scene.to_dict()
        self._save()

    def get(self, scene_id: str) -> Scene | None:
        data = self._load().get(scene_id)
        if data is None:
            return None
        try:
            return Scene.from_dict(data)
        except ArtistError:
            return None

    def list(self) -> list[dict[str, Any]]:
        """Scene summaries, most recently updated first."""
        rows = []
        for data in self._load().values():
            try:
                scene = Scene.from_dict(data)
            except ArtistError:
                continue
            rows.append({
                "id": scene.id,
                "title": scene.title,
                "width": scene.width,
                "height": scene.height,
                "layers": len(scene.layers),
                "characters": len(scene.characters),
                "shots": len(scene.shots),
                "artifacts": sorted(scene.artifacts),
                "updated_at": scene.updated_at,
            })
        rows.sort(key=lambda row: row["updated_at"], reverse=True)
        return rows

    def delete(self, scene_id: str) -> bool:
        if scene_id not in self._load():
            return False
        del self._data[scene_id]  # type: ignore[union-attr]
        self._save()
        return True
